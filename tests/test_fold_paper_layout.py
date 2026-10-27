# SPDX-License-Identifier: Apache-2.0
"""What lets the paper fold be done by two hands, by contact, each on its own side, read off the scene the example declares.

``examples/03_materials/05_paper`` folds an A4 sheet in half with both of the robot's hands. A parallel
jaw cannot get a pad under a sheet lying flat (seven top-down pinches raised one 0 mm), so the sheet lies
on a box with its near edge 35 mm past the box's edge, where each hand slides its lower pad under it
from its own corner. The sheet lies across the middle so the two corners are one in each hand's half;
on the bench edge the old layout hung it over, only the right arm reached and the left hand crossed
into its space to help. These checks hold that layout and its realism settings in place without
building anything: the overhang the pads close on, the sheet on the box and across the middle, the arms
that collide with each other, a grip capped at 15 N or less and fed the sheet's push, a pad friction
under 1, and nothing pinned to a hand.

The example's own ``main()`` runs, with ``Scene.build`` replaced: what it declared is
:attr:`Scene.request`, and what it then does with the built sim and on its page is
recorded against a stand-in that remembers every name it reached for.
"""

from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np

from tests.fixtures.examples import _run, scripts_module

TIER = "fast"
"""Runs the example's ``main()`` with the build replaced: nothing is built."""


class _Recorder:
    """Whatever ``main()`` reaches for on the built sim or the studio, remembering every name it reached.

    Arrays read back to the host are empty, so the example's numpy over them runs.
    """

    def __init__(self, path: str, seen: list[str]) -> None:
        self._path, self._seen = path, seen

    def __getattr__(self, name: str) -> _Recorder:
        if name.startswith("__"):
            raise AttributeError(name)
        self._seen.append(f"{self._path}.{name}")
        return _Recorder(f"{self._path}.{name}", self._seen)

    def __call__(self, *args, **kwargs):
        return np.zeros(0) if self._path.endswith(".numpy") else _Recorder(f"{self._path}()", self._seen)

    def __getitem__(self, key) -> _Recorder:
        return _Recorder(f"{self._path}[{key!r}]", self._seen)

    def __iter__(self):
        return iter(())

    def __enter__(self) -> _Recorder:
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _declared():
    """``main()`` run against a build that builds nothing: ``(request, things, reached)``.

    ``things`` maps a declared thing's name to its spec; ``reached`` is every name
    ``main()`` and its page asked of the sim and the studio.
    """
    import robosimstudio as rss

    reached: list[str] = []
    scenes = []

    def build(scene, **kwargs):
        scenes.append(scene)
        return _Recorder("sim", reached)

    with mock.patch.object(rss.Scene, "build", build):
        _, kwargs = _run("03_materials/05_paper.py")
    kwargs["page"](_Recorder("studio", reached))
    (scene,) = scenes
    request = scene.request
    return request, {spec.name: spec for spec in request.entities}, reached


class TestTheFoldIsLaidOutForTwoHands(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.request, cls.things, cls.reached = _declared()
        sheet, box = cls.things["paper"], cls.things["box"]
        cls.size, cls.corner = sheet.size, sheet.pose
        half, at = box.shape.half_extents, box.xform[:3]
        cls.box = tuple(2.0 * h for h in half)
        cls.box_near = at[0] - half[0]
        cls.box_top = at[2] + half[2]

    def test_the_pads_close_on_the_overhang_clear_of_the_box(self) -> None:
        takes = scripts_module("recording.takes")

        overhang = self.box_near - self.corner[0]
        far_side = takes.FOLD_PINCH + 0.006  # the 12 mm pads' far side, in from the edge
        self.assertGreaterEqual(
            overhang - far_side, 0.005, f"the pads end {far_side * 1e3:.0f} mm in, the box at {overhang * 1e3:.0f}"
        )
        self.assertLessEqual(overhang, 0.05, f"the edge runs {overhang * 1e3:.0f} mm past the box")

    def test_the_sheet_lies_on_the_box_across_the_middle(self) -> None:
        size, box, corner = self.size, self.box, self.corner
        overhang = self.box_near - corner[0]
        self.assertAlmostEqual(corner[2], self.box_top + 0.005, places=6, msg="laid 5 mm over the box's top")
        self.assertAlmostEqual(corner[1], -size[1] / 2, places=6, msg="one corner in each hand's half")
        self.assertGreaterEqual(box[0], size[0] - overhang, "the rest of the sheet is on the box")
        self.assertGreater(box[1], size[1], "and the box is wider than it")

    def test_the_arms_collide_with_each_other(self) -> None:
        from robosimstudio.robots.registry import get_embodiment

        self.assertTrue(get_embodiment("yam_bimanual").cross_arm_collisions, "the YAM's own setting")
        self.assertTrue(self.request.embodiment.cross_arm_collisions, "and the example keeps it")

    def test_the_grip_is_capped_and_feels_the_sheet(self) -> None:
        hands = self.request.gripper_actuation
        # One motor drives a hand's two fingers through the asset's joint equality: its cap bounds the driven
        # finger's slide, not either finger's contact push, which test_fold_paper_robot holds to 15 N.
        self.assertLessEqual(hands.effort_limit, 15.0, "the motor's cap")
        # Stiff enough that a finger held 0.2 mm off shut already asks the whole cap: Close stalls at the cap.
        self.assertGreaterEqual(hands.target_ke * 0.0002, hands.effort_limit)
        coupling = dict(self.request.physics.coupling.params)
        self.assertTrue(coupling["slide_feedback"], "the sheet's push reaches the fingers' slides")
        self.assertTrue(coupling["contact_forces"])

    def test_no_pad_grips_paper_harder_than_friction_one(self) -> None:
        """VBD mixes paper against a pad as the geometric mean: read the pads' friction off the hand's asset."""
        import xml.etree.ElementTree as ET

        from robosimstudio.paths import resolve_path

        asset = ET.parse(resolve_path("embodiments/yam/yam_bimanual_arms.xml"))
        pads = [float(g.get("friction").split()[0]) for g in asset.iter("geom") if g.get("friction")]
        self.assertTrue(pads)
        self.assertLessEqual(math.sqrt(self.things["paper"].material.friction * max(pads)), 1.0)
        written = [name for name in self.reached if name.endswith("_mu") or "_mu." in name]
        self.assertEqual([], written, "and the example writes no friction of its own")

    def test_nothing_on_the_page_pins_the_sheet_to_a_hand(self) -> None:
        sheet = self.things["paper"]
        self.assertEqual((), sheet.particle_sets, "the fold is carried by contact; a particle set is an attachment's")
        self.assertEqual("", sheet.pinned)
        self.assertEqual({"box", "paper"}, set(self.things), "and nothing else is declared to hold it")
        for attachment in ("grab_set_requested", "follow_arm", "particle_sets", "pin", "attach"):
            reached = [name for name in self.reached if attachment in name.rsplit(".", 1)[-1]]
            self.assertEqual([], reached, f"the fold is carried by contact; {attachment!r} is an attachment's")


if __name__ == "__main__":
    unittest.main()
