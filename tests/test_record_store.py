# SPDX-License-Identifier: Apache-2.0
"""Reading a run back: its results, its episodes, and what an offline re-render replays.

``EpisodeStore`` is the one read API over a run directory, so everything
downstream -- replay, re-render, export, analysis -- reads what it says. The
results file is appended an episode at a time so a run killed halfway still has
its finished rows; an episode is found by its zero-padded number; and a results
row says where it came from whether a task or a bare scene ran it.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TIER = "fast"
"""No GPU: numpy arrays, and a run directory written to a temp dir."""


def _store(test: unittest.TestCase):
    import tempfile

    from robosimstudio.record.store import EpisodeStore

    return EpisodeStore(Path(test.enterContext(tempfile.TemporaryDirectory())) / "run")


def _write_episode(root: Path, name: str, *, steps: dict | None = None, privileged: dict | None = None, **meta):
    path = root / name
    path.mkdir(parents=True)
    (path / "meta.json").write_text(json.dumps(meta))
    if steps is not None:
        np.savez(path / "steps.npz", **steps)
    if privileged is not None:
        np.savez(path / "privileged.npz", **privileged)
    return path


class TestReplaySteps(unittest.TestCase):
    """What an offline re-render replays, and what it must not take a length from.

    The bug this guards: ``privileged.npz`` carries an episode's *constants*
    beside its per-step state -- ``<cloth>/bond_index``, ``<cloth>/bond_anchor``,
    ``<cloth>/bond_row``, ``particle_subset`` -- and the slicer behind
    :meth:`EpisodeData.replay_steps` truncates to its shortest array. A 340-step
    tape peel with 45 bonds replayed as 45 steps: no error and no warning, just
    the first eighth of the episode and then the end of the clip.
    """

    def _episode(self, **privileged):
        from pathlib import Path

        from robosimstudio.record.store import EpisodeData

        return EpisodeData(path=Path("."), meta={}, steps={}, privileged=privileged)

    def test_an_episode_constant_does_not_shorten_the_replay(self):
        """Twelve steps stay twelve steps next to four bonds and a subset."""
        episode = self._episode(
            body_q=np.zeros((12, 3, 7), dtype=np.float32),
            particle_q=np.zeros((12, 5, 3), dtype=np.float32),
            tape_bond_active=np.zeros((12, 4), dtype=np.uint8),
            particle_subset=np.arange(5, dtype=np.int32),
            **{
                "tape/bond_index": np.zeros(4, dtype=np.int32),
                "tape/bond_anchor": np.zeros((4, 3), dtype=np.float32),
                "tape/bond_row": np.zeros(4, dtype=np.int32),
            },
        )
        steps = list(episode.replay_steps())
        self.assertEqual(len(steps), 12)
        # And narrowed to what a state can be written from: a per-step channel
        # no replay reads is still a length nothing should be measured against.
        self.assertEqual(sorted(steps[0]), ["body_q", "particle_q"])

    def test_a_recording_cut_mid_write_still_truncates(self):
        """The shortest *replayable* channel is still the honest length.

        Truncation is the point when a run was interrupted: the last step's
        particles may never have been written, and rendering a body pose against
        a cloth from another step would be a frame that never existed.
        """
        episode = self._episode(
            body_q=np.zeros((12, 3, 7), dtype=np.float32),
            particle_q=np.zeros((7, 5, 3), dtype=np.float32),
        )
        self.assertEqual(len(list(episode.replay_steps())), 7)


class TestTheResultsFile(unittest.TestCase):
    def test_rows_read_back_in_the_order_written_each_stamped_with_the_schema(self) -> None:
        from robosimstudio.record.store import SCHEMA

        store = _store(self)
        self.assertEqual([], store.results(), "a run not yet written has no rows")
        for episode in range(3):
            store.append_result({"episode_id": episode, "success": episode != 1, "episode_length": 10 * episode})
        rows = store.results()
        self.assertEqual([0, 1, 2], [row["episode_id"] for row in rows])
        self.assertTrue(all(row["schema"] == SCHEMA for row in rows))

    def test_the_summary_is_over_the_rows(self) -> None:
        store = _store(self)
        self.assertEqual({"episodes": 0, "successes": 0, "success_rate": 0.0, "mean_length": 0.0}, store.summary())
        for success, length in ((True, 10), (False, 30), (True, 20), (True, 40)):
            store.append_result({"success": success, "episode_length": length})
        self.assertEqual({"episodes": 4, "successes": 3, "success_rate": 0.75, "mean_length": 25.0}, store.summary())

    def test_the_run_meta_reads_back_as_written(self) -> None:
        from dataclasses import asdict

        from robosimstudio.record.store import RunMeta

        store = _store(self)
        self.assertEqual({}, store.run_meta)
        meta = RunMeta(run_name="r", task="t", episodes=2, provenance={"commit": "abc"})
        store.write_run_meta(meta)
        self.assertEqual(asdict(meta), store.run_meta)


class TestTheEpisodes(unittest.TestCase):
    def test_only_directories_with_a_meta_are_episodes_and_they_come_sorted(self) -> None:
        store = _store(self)
        self.assertEqual([], store.episode_ids(), "no directory is no episodes, not a raise")
        for name in ("0002", "0000", "0001"):
            _write_episode(store.root, name, episode_id=int(name))
        (store.root / "cameras").mkdir()
        (store.root / "notes.txt").write_text("not an episode")
        self.assertEqual(["0000", "0001", "0002"], store.episode_ids())
        self.assertEqual([0, 1, 2], [episode.episode_id for episode in store.episodes()])

    def test_an_episode_is_found_by_its_number_or_its_name(self) -> None:
        store = _store(self)
        _write_episode(store.root, "0007", episode_id=7, success=True)
        self.assertEqual(store.episode(7).path, store.episode("0007").path)
        self.assertTrue(store.episode(7).success)
        with self.assertRaises(FileNotFoundError) as caught:
            store.episode(8)
        self.assertIn("0008", str(caught.exception))

    def test_the_arrays_load_and_a_missing_file_is_empty(self) -> None:
        store = _store(self)
        actions = np.arange(12, dtype=np.float32).reshape(4, 3)
        _write_episode(store.root, "0000", steps={"action": actions}, episode_id=0)
        _write_episode(store.root, "0001", episode_id=1)
        episode = store.episode(0)
        np.testing.assert_array_equal(actions, episode.steps["action"])
        self.assertEqual(4, episode.length)
        self.assertEqual({}, episode.privileged, "no privileged.npz is no ground truth, not a raise")
        bare = store.episode(1)
        self.assertEqual((0, {}), (bare.length, bare.steps))

    def test_a_meta_that_names_no_id_or_outcome_reads_as_none_and_a_failure(self) -> None:
        store = _store(self)
        _write_episode(store.root, "0009")
        self.assertEqual((-1, False), (store.episode(9).episode_id, store.episode(9).success))

    def test_videos_are_named_by_camera(self) -> None:
        store = _store(self)
        path = _write_episode(store.root, "0000", episode_id=0)
        self.assertEqual({}, store.episode(0).videos())
        (path / "cameras").mkdir()
        for name in ("wrist.mp4", "front.mp4", "front.json"):
            (path / "cameras" / name).write_bytes(b"")
        self.assertEqual(["front", "wrist"], list(store.episode(0).videos()))


class TestTheResultsRow(unittest.TestCase):
    EPISODE = SimpleNamespace(episode_id=3, seed=11, instruction="fold it", model_key="m")

    def _row(self, task=None, request=None, **given):
        from robosimstudio.record.store import episode_row

        kwargs = {
            "success": True,
            "reward": 1,
            "episode_length": 40,
            "close_kind": "",
            "termination_reason": "",
            "metrics": {},
            "wall_s": 2.0,
            "sim_s": 1.0,
            "recorded_steps": 40,
            "captured_steps": 0,
            "video_frames": 0,
            **given,
        }
        return episode_row(task, self.EPISODE, request=request, **kwargs)

    def test_a_bare_scene_is_written_down_from_its_request(self) -> None:
        request = SimpleNamespace(cell="bench_075", embodiment=SimpleNamespace(name="panda"))
        row = self._row(request=request)
        self.assertEqual(("", "bench_075", "panda"), (row["task"], row["scene"], row["embodiment"]))

    def test_a_task_is_written_down_from_itself(self) -> None:
        task = SimpleNamespace(full_name="suite/fold", scene="bench_090", embodiment="yam_bimanual")
        row = self._row(task=task, request=SimpleNamespace(cell="ignored"))
        self.assertEqual(("suite/fold", "bench_090", "yam_bimanual"), (row["task"], row["scene"], row["embodiment"]))

    def test_the_rate_and_the_defaults(self) -> None:
        row = self._row(task=SimpleNamespace(full_name="t", scene="s", embodiment="e"), extra={"note": 1})
        self.assertEqual(0.5, row["realtime_factor"])
        self.assertEqual("unknown", row["close_kind"], "an unsaid close is said to be unknown")
        self.assertEqual((1.0, 1), (row["reward"], row["note"]))
        self.assertIsInstance(row["reward"], float)
        self.assertEqual(0.0, self._row(task=None, request=SimpleNamespace(), wall_s=0.0)["realtime_factor"])


if __name__ == "__main__":
    unittest.main()
