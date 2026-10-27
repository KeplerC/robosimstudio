# SPDX-License-Identifier: Apache-2.0
"""The region boxes: unticking Names takes the labels down, unticking Tint repaints a green box."""

from __future__ import annotations

import types
import unittest
import unittest.mock

import numpy as np

from robosimstudio_studio.regions import HELD_COLOR, IDLE_COLOR, PanelRegions
from tests.fixtures.viser_stubs import _StubGui, _StubSession

TIER = "fast"
"""No GPU, no browser: a stub scene records the boxes and labels."""


def _regions_panel():
    region = types.SimpleNamespace()
    env = types.SimpleNamespace(
        handles=types.SimpleNamespace(regions={"bin": region}),
        goal=None,
        step_count=0,
        model=types.SimpleNamespace(body_count=0),
    )
    session = _StubSession(env, _StubGui())
    panel = PanelRegions(session)
    panel.bounds = lambda name, body_q: (np.zeros(3), np.ones(3))
    panel.build(session.server.gui)
    return panel, env


class TestRegionToggles(unittest.TestCase):
    def test_unticking_names_takes_the_labels_down(self) -> None:
        panel, _env = _regions_panel()
        panel.drain()
        label = panel._nodes["bin/label"]
        panel.server.gui.find("Names").fire(False)
        panel.drain()  # same step: a paused page
        self.assertTrue(label.removed, "a label left standing, frozen where the region was")
        self.assertNotIn("bin/label", panel._nodes)
        panel.server.gui.find("Names").fire(True)
        panel.drain()
        self.assertIn("bin/label", panel._nodes)

    def test_unticking_tint_repaints_a_green_box(self) -> None:
        panel, env = _regions_panel()
        panel._clause = {"bin": "in bin"}
        env.last_result = types.SimpleNamespace(metrics={"in bin": True})
        panel.drain()
        box = panel._nodes["bin/box"]
        self.assertEqual(box.color, HELD_COLOR)
        panel.server.gui.find("Tint when satisfied").fire(False)
        panel.drain()
        self.assertEqual(box.color, IDLE_COLOR)


if __name__ == "__main__":
    unittest.main()
