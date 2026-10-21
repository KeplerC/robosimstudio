# SPDX-License-Identifier: Apache-2.0
"""A collection writes the episode store a reader expects, resumes, and reports a policy that fails.

Two short episodes on two workers, an arm and a cube filmed by two small
cameras, read back through :class:`~robosimstudio.record.store.EpisodeStore`:
every frame of every camera is in its mp4, the actions are one per control
step, and a frame's body poses are kept for a re-render. The workers build
the scene from this module's own ``build`` and drive it with its ``policy``,
named by path, which is how a user's script reaches a spawned worker.

Rendered by ``tiled`` on the CPU so that nothing here needs an OpenGL context;
``tests/test_gl_renderer.py`` holds the rasteriser to what tiled draws.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

TIER = "cpu"


def build():
    import robosimstudio as rss

    scene = rss.Scene(cell="bench_075", physics=rss.MUJOCO_500HZ)
    scene.add(rss.box(name="cube", half=(0.025, 0.025, 0.025), at=(0.0, 0.1, 0.03)))
    scene.add(rss.robot("panda"))
    for name, eye in (("front", (0.6, -0.6, 1.2)), ("top", (0.0, -0.1, 1.6))):
        scene.add(rss.CameraSpec(name=name, width=96, height=64, fov_y=60.0, eye=eye, look_at=(0.0, 0.0, 0.75)))
    return scene


def policy(sim, episode):
    """Swing every joint a little, differently per episode; episode 3 fails on purpose."""
    if episode == 3:
        raise RuntimeError("this policy has no plan for episode 3")
    home = np.asarray(sim.idle_action("joint_position"), dtype=np.float32)
    offset = np.random.default_rng(episode).uniform(-0.2, 0.2, home.shape).astype(np.float32)

    class Policy:
        def act(self, sim, step):
            return home + offset * float(np.sin(0.2 * step))

        def success(self, sim):
            return episode % 2 == 0

    return Policy()


class TestACollection(unittest.TestCase):
    def _spec(self, out: str, **kwargs):
        from robosimstudio.record.collect import CollectSpec

        here = Path(__file__).resolve()
        fields = {
            "scene": f"{here}:build",
            "policy": f"{here}:policy",
            "out": out,
            "episodes": 2,
            "seconds": 1.0,
            "fps": 10,
            "device": "cpu",
            "renderer": "tiled",
            "workers": 2,
            "preset": "ultrafast",
            "threads": 1,
            **kwargs,
        }
        return CollectSpec(**fields)

    def test_the_store_holds_every_frame_and_step_and_a_rerun_skips_them(self) -> None:
        import imageio.v2 as imageio

        from robosimstudio.record.collect import collect
        from robosimstudio.record.store import EpisodeStore

        with tempfile.TemporaryDirectory() as out:
            summary = collect(self._spec(out), verbose=False)
            self.assertEqual((summary["ran"], summary["failed"]), (2, 0), summary["errors"])
            store = EpisodeStore(out)
            self.assertEqual(store.summary()["episodes"], 2)
            self.assertEqual(store.run_meta["renderer"], "tiled")
            for episode in (0, 1):
                data = store.episode(episode)
                self.assertEqual(data.success, episode == 0)
                self.assertEqual(data.length, 50)  # 1 s at 50 Hz
                # A pose a step, as the store has it, so a replay pairs it with joint_q row for row ...
                self.assertEqual(data.privileged["body_q"].shape[0], 50)
                self.assertEqual(len(list(data.replay_steps())), 50)
                # ... and which step each frame shows: the reset, then 10 Hz for 1 s.
                np.testing.assert_array_equal(data.privileged["frame_step"], [-1, *range(4, 50, 5)])
                self.assertEqual(data.privileged["body_q_reset"].shape, data.privileged["body_q"].shape[1:])
                self.assertEqual(sorted(data.videos()), ["front", "top"])
                for path in data.videos().values():
                    with imageio.get_reader(str(path)) as reader:
                        self.assertEqual(sum(1 for _ in reader), 11)
            first = store.episode(1).steps["action"]
            again = collect(self._spec(out), verbose=False)
            self.assertEqual((again["ran"], again["skipped"]), (0, 2))
            np.testing.assert_array_equal(store.episode(1).steps["action"], first)
            # A rerun that ran nothing keeps the run's totals rather than writing its own zeroes.
            self.assertEqual((store.run_meta["episodes"], store.run_meta["successes"]), (2, 1))
            self.assertEqual(store.run_meta["renderer"], "tiled")

    def test_a_policy_that_raises_is_reported_and_the_rest_are_written(self) -> None:
        from robosimstudio.record.collect import collect
        from robosimstudio.record.store import EpisodeStore

        with tempfile.TemporaryDirectory() as out:
            summary = collect(self._spec(out, episodes=2, first=2), verbose=False)
            self.assertEqual((summary["ran"], summary["failed"]), (1, 1))
            self.assertIn("no plan for episode 3", summary["errors"][0])
            self.assertEqual(EpisodeStore(out).episode_ids(), ["0002"])

    def test_a_dropped_failure_is_not_run_again_nor_read_as_an_episode(self) -> None:
        """Its files go and its meta stays as dropped.json: a resume skips it, and the store does not list it."""
        import json

        from robosimstudio.record.collect import collect
        from robosimstudio.record.store import EpisodeStore

        with tempfile.TemporaryDirectory() as out:
            spec = self._spec(out, episodes=2, first=0, keep_failures=False)  # episode 1 reports failure
            collect(spec, verbose=False)
            again = collect(spec, verbose=False)
            store = EpisodeStore(out)
            self.assertEqual((again["ran"], again["skipped"]), (0, 2))
            self.assertEqual(len(store.results()), 2)
            self.assertEqual(store.episode_ids(), ["0000"])
            self.assertEqual([e.meta["episode_id"] for e in store.episodes()], [0])
            dropped = Path(out) / "0001"
            self.assertEqual(sorted(p.name for p in dropped.iterdir()), ["dropped.json"])
            self.assertTrue(json.loads((dropped / "dropped.json").read_text())["dropped"])

    def test_a_worker_that_cannot_build_counts_every_episode_it_held(self) -> None:
        from robosimstudio.record.collect import collect

        with tempfile.TemporaryDirectory() as out:
            here = Path(__file__).resolve()
            summary = collect(self._spec(out, scene=f"{here}:no_such_scene", episodes=3), verbose=False)
            self.assertEqual((summary["ran"], summary["failed"]), (0, 3))
            self.assertIn("no_such_scene", summary["errors"][0])

    def test_frames_faster_than_the_control_rate_are_refused(self) -> None:
        from robosimstudio.record.collect import collect

        with tempfile.TemporaryDirectory() as out:
            summary = collect(self._spec(out, episodes=1, fps=60, workers=1), verbose=False)
            self.assertEqual(summary["failed"], 1)
            self.assertIn("above the scene's control rate", summary["errors"][0])


class TestWhichGpus(unittest.TestCase):
    def test_a_gpu_is_named_as_cuda_visible_devices_names_it(self) -> None:
        """A UUID or a MIG slice is a GPU too; dropping it pinned every worker to nothing."""
        from unittest import mock

        from robosimstudio.record.collect import _visible_gpus

        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "GPU-8f3a, 3,MIG-1c2d"}):
            self.assertEqual(_visible_gpus(), ("GPU-8f3a", "3", "MIG-1c2d"))
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}):
            self.assertEqual(_visible_gpus(), ())


if __name__ == "__main__":
    unittest.main()
