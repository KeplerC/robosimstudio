# SPDX-License-Identifier: Apache-2.0
"""What lets the paper fold be done by two hands, by contact, each on its own side, read off the example's source.

``examples/03_materials/05_paper`` folds an A4 sheet in half with both of the robot's hands. A parallel
jaw cannot get a pad under a sheet lying flat (seven top-down pinches raised one 0 mm), so the sheet lies
on a box with its near edge 35 mm past the box's edge, where each hand slides its lower pad under it
from its own corner. The sheet lies across the middle so the two corners are one in each hand's half;
on the bench edge the old layout hung it over, only the right arm reached and the left hand crossed
into its space to help. These checks hold that layout in place without building it: the overhang, the
sheet on the box and across the middle, the arms that collide with each other, a capped grip, a pad
friction under 1, and nothing pinned to a hand.
"""

from __future__ import annotations

import ast
import math
import unittest

from tests.fixtures.examples import EXAMPLES

TIER = "fast"
"""Parses one file; builds nothing."""

FOLD = EXAMPLES / "03_materials" / "05_paper.py"


def _locals() -> dict:
    """The literal locals ``main()`` opens with: the sheet, the box, the overhang, the nubs' friction."""
    tree = ast.parse(FOLD.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    out = {}
    for node in main.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                continue
            if isinstance(target, ast.Name):
                out[target.id] = value
            elif isinstance(target, ast.Tuple):
                out.update({t.id: v for t, v in zip(target.elts, value, strict=True)})
    return out


class TestTheFoldIsLaidOutForTwoHands(unittest.TestCase):
    def setUp(self) -> None:
        self.found = _locals()
        self.source = FOLD.read_text()

    def test_the_near_edge_runs_past_the_box(self) -> None:
        # The pads (12 mm wide) close 6 mm in from the edge, and the finger blades reach 10 mm past
        # their far side: 28 mm of overhang keeps them off the box; at 50 the strip droops off the pad.
        overhang = self.found["overhang"]
        self.assertGreaterEqual(overhang, 0.028, f"the edge runs {overhang * 1e3:.0f} mm past the box")
        self.assertLessEqual(overhang, 0.05, f"the edge runs {overhang * 1e3:.0f} mm past the box")

    def test_the_sheet_lies_on_the_box_across_the_middle(self) -> None:
        size, box, corner, overhang = (self.found[k] for k in ("size", "box", "corner", "overhang"))
        self.assertAlmostEqual(corner[2], box[2] + 0.005, places=6, msg="laid 5 mm over the box's top")
        self.assertAlmostEqual(corner[1], -size[1] / 2, places=6, msg="one corner in each hand's half")
        self.assertGreaterEqual(box[0], size[0] - overhang, "the rest of the sheet is on the box")
        self.assertGreater(box[1], size[1], "and the box is wider than it")

    def test_the_arms_collide_with_each_other_and_the_grip_is_capped(self) -> None:
        self.assertIn("cross_arm_collisions=True", self.source)
        tree = ast.parse(self.source)
        caps = [
            kw.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "ActuationSpec"
            for kw in node.keywords
            if kw.arg == "effort_limit"
        ]
        self.assertEqual(1, len(caps), "the hands' actuation states an effort limit")
        self.assertLessEqual(caps[0] / 2, 15.0, "one driven joint and its mirrored twin: half the cap a pad")

    def test_no_pad_grips_paper_harder_than_friction_one(self) -> None:
        from robosimstudio_materials.paper import PAPER

        self.assertLessEqual(math.sqrt(PAPER.friction * self.found["nubs"]), 1.0)

    def test_nothing_on_the_page_pins_the_sheet_to_a_hand(self) -> None:
        for attachment in ("grab_set_requested", "follow_arm", "particle_sets", "pinned=", "attach"):
            self.assertNotIn(
                attachment, self.source, f"the fold is carried by contact; {attachment!r} is an attachment's"
            )


if __name__ == "__main__":
    unittest.main()
