# SPDX-License-Identifier: Apache-2.0
"""The studio's panels, built over stand-ins: the arm's grip, the cloth fold's Set list, Attach on an arm.

What a panel does with a click or a dropdown is bookkeeping between its
callbacks and its ``drain``; none of it needs a scene. The page built over a
real scene is ``test_studio_page``'s, in the medium tier.
"""

from __future__ import annotations

import types
import unittest
import unittest.mock

from robosimstudio_studio.panels import Panel, PanelArm, PanelCloth, PanelObject
from tests.fixtures.studio_fakes import FakeEnv, FakeSession
from tests.fixtures.viser_stubs import _StubGui, _StubSession

TIER = "fast"
"""No GPU, no browser: panels built without their constructors over stub guis."""


def _arm_panel(name: str = "right") -> tuple[PanelArm, _StubGui]:
    """A real arm panel, built over a fake env whose IK and fingers record what the panel asks of them."""
    session = FakeSession(FakeEnv(arms=(name,)))
    panel = PanelArm(session, name)
    session.arms[name] = panel
    panel.build(session.server.gui)
    return panel, session.server.gui


class TestTheGrip(unittest.TestCase):
    """A grip is a button, beside the slider, in a folder that opens by default; ``drain`` applies it."""

    def test_one_click_shuts_the_hand_and_one_opens_it(self) -> None:
        """What the slider took a drag to say."""
        panel, gui = _arm_panel()
        for asked, expected in (("Close", 1.0), ("Open", 0.0)):
            gui.find("Grip").fire(asked)
            panel.drain()
            self.assertEqual(expected, panel.gripper_command)
            self.assertEqual(expected, panel._grip.value, "the slider did not follow the button")
            self.assertEqual(expected, panel.gripper.applied[-1], "the fingers were not told")

    def test_the_request_is_applied_on_the_main_thread_not_in_the_callback(self) -> None:
        """The panel contract: a callback assigns, ``drain`` acts."""
        panel, _gui = _arm_panel()
        panel.gripper_command = 0.0
        panel.grip_request = 1.0
        self.assertEqual(0.0, panel.gripper_command, "the request was acted on before drain")
        panel.drain()
        self.assertIsNone(panel.grip_request)
        self.assertEqual(1.0, panel.gripper_command)

    def test_the_grip_button_is_there_and_writes_only_a_request(self) -> None:
        panel, gui = _arm_panel()
        grip = gui.find("Grip")
        grip.fire("Close")
        self.assertEqual(1.0, panel.grip_request)
        self.assertEqual(0.0, panel.gripper_command, "the callback acted rather than asked")
        self.assertEqual([], panel.gripper.applied, "the fingers were written from viser's thread")
        grip.fire("Open")
        self.assertEqual(0.0, panel.grip_request)

    def test_the_arm_folder_opens_by_default(self) -> None:
        """Half of what a person came here to do should not start behind a fold."""
        _panel, gui = _arm_panel()
        folder = gui.find("Grip").parent
        self.assertEqual("Arm: right", folder.label)
        self.assertTrue(folder.expand_by_default)


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
