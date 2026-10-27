# SPDX-License-Identifier: Apache-2.0
"""The studio's panels, built over stand-ins: the cloth fold's Set list, Attach on an arm.

What a panel does with a click or a dropdown is bookkeeping between its
callbacks and its ``drain``; none of it needs a scene. The page built over a
real scene is ``test_studio_page``'s, in the medium tier.
"""

from __future__ import annotations

import types
import unittest
import unittest.mock

from robosimstudio_studio.panels import Panel, PanelCloth, PanelObject
from tests.fixtures.viser_stubs import _StubGui, _StubSession

TIER = "fast"
"""No GPU, no browser: panels built without their constructors over stub guis."""


class TestTheSetListFollowsTheClothDropdown(unittest.TestCase):
    def _cloth_panel(self) -> tuple[PanelCloth, _StubGui]:
        deformables = {
            "towel": types.SimpleNamespace(sets={"towel/corner": [0, 1]}),
            "shirt": types.SimpleNamespace(sets={"shirt/hem": [2, 3], "shirt/collar": [4]}),
        }
        env = types.SimpleNamespace(handles=types.SimpleNamespace(deformables=deformables))
        gui = _StubGui()
        session = _StubSession(env, gui)
        # Built without its constructor, which wants a GPU model for its draggers.
        panel = PanelCloth.__new__(PanelCloth)
        Panel.__init__(panel, session)
        panel.names = ["towel", "shirt"]
        panel.selected = "towel"
        panel.draggers = [object()]
        panel.chrome = True
        panel.grab_on_click = True
        panel.follow_arm = [""]
        panel._dropdown = None
        panel.build(gui)
        return panel, gui

    def test_changing_the_cloth_by_hand_moves_the_set_list(self) -> None:
        panel, gui = self._cloth_panel()
        self.assertEqual(panel.selected_set, "corner")
        gui.find("Cloth").fire("shirt")
        self.assertEqual(panel.selected, "shirt")
        self.assertEqual(tuple(gui.find("Set").options), ("collar", "hem"))
        self.assertIn(panel.selected_set, ("collar", "hem"), "the towel's strip is not on the shirt")


class _NoHandles(list):
    def drain(self) -> None:
        pass


class TestAttachAnArm(unittest.TestCase):
    def test_attach_takes_the_selected_arm(self) -> None:
        panel = PanelObject.__new__(PanelObject)
        panel.pending_body = None
        panel.attach_requested = True
        panel.release_requested = panel.release_all_requested = False
        panel.selected = "left"
        panel.chains = {}
        panel.arm_bodies = {"left": (7, 5, 3)}
        panel.env = types.SimpleNamespace(handles=types.SimpleNamespace(objects={}))
        panel.handles = _NoHandles()
        panel.draggers = []
        attached = []
        panel._attach = attached.append
        panel.drain()
        self.assertEqual(attached, [7])


if __name__ == "__main__":
    unittest.main()
