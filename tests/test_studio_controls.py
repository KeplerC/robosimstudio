# SPDX-License-Identifier: Apache-2.0
"""The page's recorders: a take's run directory and meta, and the video buffer.

A rebuilt Record panel keeps the run it was recording into, a run's meta keeps
the first take's start, and a scene with no bodies can still be filmed.
"""

from __future__ import annotations

import unittest
from collections import deque
from types import SimpleNamespace
from unittest import mock

import numpy as np

TIER = "fast"
"""No GPU: the recorders are handed stand-in envs and a temp directory."""


class ARebuiltRecordPanelKeepsTheRun(unittest.TestCase):
    def test_the_run_start_and_the_kept_count_come_off_the_directory(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from robosimstudio_studio.controls import PanelRecord

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run_meta.json").write_text(json.dumps({"started": "2026-01-01T00:00:00+00:00"}))
            (root / "episode_results.jsonl").write_text('{"episode_id": 0}\n{"episode_id": 1}\n')
            session = SimpleNamespace(env=None, server=None, run_dir=root)
            panel = PanelRecord(session, root=root)
            self.assertEqual(panel.run_started, "2026-01-01T00:00:00+00:00")
            self.assertEqual(panel.kept, 2)

    def test_a_fresh_directory_starts_from_nothing(self) -> None:
        import tempfile
        from pathlib import Path

        from robosimstudio_studio.controls import PanelRecord

        with tempfile.TemporaryDirectory() as tmp:
            session = SimpleNamespace(env=None, server=None, run_dir=Path(tmp))
            panel = PanelRecord(session)
            self.assertEqual((panel.run_started, panel.kept, panel.take), ("", 0, 0))


class PanelVideoCapture(unittest.TestCase):
    def test_a_scene_without_bodies_is_captured(self) -> None:
        from robosimstudio_studio.controls import PanelVideo

        panel = object.__new__(PanelVideo)
        particles = np.ones((4, 3), np.float32)
        panel.env = SimpleNamespace(
            step_count=2,
            state=SimpleNamespace(body_q=None, particle_q=SimpleNamespace(numpy=lambda: particles)),
            model=SimpleNamespace(particle_count=4),
        )
        panel.every = 2
        panel.buffer = deque(maxlen=30)
        panel.dropped = 0
        panel.capture()
        self.assertEqual(len(panel.buffer), 1)
        self.assertNotIn("body_q", panel.buffer[0])
        np.testing.assert_array_equal(panel.buffer[0]["particle_q"], particles)


class PanelRecordRunMeta(unittest.TestCase):
    def test_run_meta_keeps_the_first_take_start(self) -> None:
        from robosimstudio_studio import controls

        panel = object.__new__(controls.PanelRecord)
        panel.root = mock.MagicMock()
        panel.root.name = "demos"
        panel.spec = None
        panel.take = 0
        panel.recorder = None
        panel.started = ""
        panel.run_started = ""
        panel._state = SimpleNamespace(value="")
        panel.session = SimpleNamespace(status="")
        written = []
        panel.store = SimpleNamespace(results=lambda: [], write_run_meta=written.append)
        panel.env = SimpleNamespace(
            episode=SimpleNamespace(episode_id=0),
            request=SimpleNamespace(physics=SimpleNamespace(control_hz=50), full_name="", scene="s", embodiment=""),
            renderer_name="r",
            model=SimpleNamespace(device="cpu"),
        )
        stamps = iter(["t0", "t1", "t2", "t3"])
        fake_datetime = SimpleNamespace(now=lambda tz: SimpleNamespace(isoformat=lambda: next(stamps)))
        with (
            mock.patch.object(controls, "EpisodeRecorder", mock.MagicMock()),
            mock.patch.object(controls, "replace", lambda episode, **kw: episode),
            mock.patch.object(controls, "datetime", fake_datetime),
        ):
            panel.start()
            panel.recorder = None
            panel.start()
            self.assertEqual(panel.started, "t1")
            panel._write_run_meta()
        self.assertEqual(written[-1].started, "t0")


if __name__ == "__main__":
    unittest.main()
