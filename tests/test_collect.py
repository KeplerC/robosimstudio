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
                self.assertEqual(data.privileged["body_q"].shape[0], 11)  # the reset, then 10 Hz for 1 s
                self.assertEqual(sorted(data.videos()), ["front", "top"])
                for path in data.videos().values():
                    with imageio.get_reader(str(path)) as reader:
                        self.assertEqual(sum(1 for _ in reader), 11)
            first = store.episode(1).steps["action"]
            again = collect(self._spec(out), verbose=False)
            self.assertEqual((again["ran"], again["skipped"]), (0, 2))
            np.testing.assert_array_equal(store.episode(1).steps["action"], first)

    def test_a_policy_that_raises_is_reported_and_the_rest_are_written(self) -> None:
        from robosimstudio.record.collect import collect
        from robosimstudio.record.store import EpisodeStore

        with tempfile.TemporaryDirectory() as out:
            summary = collect(self._spec(out, episodes=2, first=2, keep_failures=False), verbose=False)
            self.assertEqual((summary["ran"], summary["failed"]), (1, 1))
            self.assertIn("no plan for episode 3", summary["errors"][0])
            store = EpisodeStore(out)
            self.assertEqual(store.episode_ids(), ["0002"])


if __name__ == "__main__":
    unittest.main()
