# SPDX-License-Identifier: Apache-2.0
"""``EpisodeRecorder``: steps of ragged length are stacked, padded and written."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robosimstudio.record import recorder as recorder_module
from robosimstudio.record.recorder import EpisodeRecorder, _stack_steps
from robosimstudio.specs import CaptureSpec, RecordSpec

TIER = "fast"
"""No GPU: a recorder over stand-in envs, writing to a temp dir."""


class StackStepsTest(unittest.TestCase):
    def test_held_steps_pad_the_action(self) -> None:
        steps = {
            "action": [np.zeros(0, np.float32), np.ones(7, np.float32), np.ones(3, np.float32)],
            "reward": [np.float32(0.0)] * 3,
        }
        arrays = _stack_steps(steps)
        self.assertEqual(arrays["action"].shape, (3, 7))
        np.testing.assert_array_equal(arrays["action_len"], [0, 7, 3])
        self.assertTrue(np.isnan(arrays["action"][0]).all())
        self.assertTrue(np.isnan(arrays["action"][2, 3:]).all())
        np.testing.assert_array_equal(arrays["action"][1], np.ones(7))
        self.assertEqual(arrays["reward"].shape, (3,))
        self.assertNotIn("reward_len", arrays)

    def test_episode_with_held_steps_is_written(self) -> None:
        root = tempfile.mkdtemp()
        rec = EpisodeRecorder(
            root, RecordSpec(record_video=False, privileged=False), capture=CaptureSpec(enabled=False)
        )
        rec._choose_particles = lambda env: None
        rec._record_anchors = lambda env: None
        original = recorder_module.episode_row
        recorder_module.episode_row = lambda *a, **k: {"recorded_steps": k.get("recorded_steps")}
        try:
            env = SimpleNamespace(step_count=0, sim_time=0.0, episode=SimpleNamespace(episode_id=0, seed=0), task=None)
            rec.begin(SimpleNamespace(episode_id=0, seed=0), env)
            for action in (None, np.ones(7), np.ones(7)):
                env.step_count += 1
                env.sim_time += 0.01
                rec.step(env, action, {}, 0.0, {})
            rec.end(env, {})
        finally:
            recorder_module.episode_row = original
        self.assertTrue(rec.enabled, rec.failures)
        with np.load(Path(root) / "0000" / "steps.npz") as data:
            self.assertEqual(data["action"].shape, (3, 7))


if __name__ == "__main__":
    unittest.main()
