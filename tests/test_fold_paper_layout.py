# SPDX-License-Identifier: Apache-2.0
"""What lets the paper fold be carried by contact, read off the example's source without building it.

``examples/13_paper/00_fold_paper`` used to carry the sheet's edge on a labelled
attachment, because a parallel jaw cannot get a pad under a sheet lying flat on a
bench: seven top-down pinches raised it 0 mm (``pinch_top.py``). The example now
lays the sheet's near edge over the bench's right edge, where the lower pad goes
under it and the pinch lifts the strip 59.6 of a 60 mm lift (``pinch_side.py``,
GPU 5; 59.8 mm on Newton 1.6, so the layout, not the version, is what does it).
These checks hold that layout and that promise in place: the overhang is long
enough for a pad and short enough not to droop, the far end lies on the bench for
the left hand to hold, and nothing on the page pins particles to a hand.
"""

from __future__ import annotations

import ast
import unittest

from tests.fixtures.examples import EXAMPLES

TIER = "fast"
"""Parses one file and reads one cell's numbers; builds nothing."""

FOLD = EXAMPLES / "13_paper" / "00_fold_paper.py"


def _locals() -> dict:
    """The literal locals ``main()`` opens with: the sheet's size and corner, the right arm's joints."""
    tree = ast.parse(FOLD.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    out = {}
    for node in main.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return out


class TestTheFoldIsLaidOutForAPinch(unittest.TestCase):
    def setUp(self) -> None:
        from robosimstudio.scene.builtin.benches import BENCH_0728

        self.found = _locals()
        self.half_y = float(BENCH_0728.half[1])

    def test_the_near_edge_hangs_over_the_bench(self) -> None:
        corner = self.found["corner"]
        overhang = -self.half_y - corner[1]
        # A 12 mm pad 13.5 mm in from the edge wants 25 mm; at 35 mm the strip droops 1.7 mm
        # (bending length 5.96 cm), and a longer one droops away from the lower pad.
        self.assertGreaterEqual(overhang, 0.025, f"the edge hangs {overhang * 1e3:.0f} mm over the bench")
        self.assertLessEqual(overhang, 0.05, f"the edge hangs {overhang * 1e3:.0f} mm over the bench")

    def test_the_far_end_lies_on_the_bench(self) -> None:
        corner, size = self.found["corner"], self.found["size"]
        self.assertLess(
            corner[1] + size[1], self.half_y - 0.05, "the far end is on the bench, where the left hand holds it"
        )

    def test_the_right_hand_starts_at_the_overhang(self) -> None:
        self.assertEqual(len(self.found["beside"]), 6, "a solved home for the right arm's six joints")

    def test_nothing_on_the_page_pins_the_sheet_to_a_hand(self) -> None:
        source = FOLD.read_text()
        for attachment in ("grab_set_requested", "follow_arm", "particle_sets", "Edge on the right hand"):
            self.assertNotIn(attachment, source, f"the fold is carried by contact; {attachment!r} is an attachment's")


if __name__ == "__main__":
    unittest.main()
