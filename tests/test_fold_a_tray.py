# SPDX-License-Identifier: Apache-2.0
"""The carton example's task, done by the right hand through the example's arm panel: fold a tray wall over a block.

``examples/03_materials/16_carton`` promises that a folding-carton wall pushed down round the block's edge stays
folded. This builds the example and walks the right hand's cursor through the page's own IK and setpoint law along
the path its take films (``scripts/recording/takes.py``'s :func:`tray_fold`): jaws shut, down onto the -y wall
30 mm out from its crease, on an arc about the crease from 8 to 92 degrees, a half-second hold against the
block's side, and away, back and up.

What it holds the run to: the wall turns past 75 degrees; two seconds after the hand has gone it still stands
past 60 and its crease keeps past 55; the walls the hand did not touch sag less than 15 degrees.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""One GPU: the example built once and about 900 control steps of a 343-particle sheet at 20 VBD iterations."""

PORT = 9852
SPEED = 0.06


class TestTheHandFoldsTheTray(unittest.TestCase):
    """Press the overhanging wall down round the block's edge: it stays folded."""

    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from recording.takes import tray_fold

        sim, kwargs = _run("03_materials/16_carton.py")
        cls.sim = sim
        cls.session = session = open_session(sim, PORT, kwargs.get("page"))
        hz = float(session.env.request.physics.control_hz)
        blank = sim["tray"].entity

        session.arms["right"].grip_request = 1.0
        tick(session, int(0.5 * hz))
        cls.settled = {n: math.degrees(blank.fold_angle(n)) for n in blank.layout["hinge"]}
        path = tray_fold(sim, session)
        go(session, "right", path["over"], speed=0.1, steps=150, tolerance=0.006)
        go(session, "right", path["on"], speed=0.02, steps=150, tolerance=0.006)
        peak = 0.0
        for point in path["arc"]:
            go(session, "right", point, speed=SPEED, steps=60, tolerance=0.006)
            peak = max(peak, math.degrees(blank.fold_angle("wall_yneg")))
        go(session, "right", path["hold"], speed=0.01, steps=30, tolerance=0.006)
        tick(session, int(0.5 * hz))
        cls.peak = max(peak, math.degrees(blank.fold_angle("wall_yneg")))
        go(session, "right", path["back"], speed=0.05, steps=60, tolerance=0.006)
        go(session, "right", path["up"], speed=0.08, steps=60, tolerance=0.006)
        tick(session, 2 * int(hz))
        cls.after = {n: math.degrees(blank.fold_angle(n)) for n in blank.layout["hinge"]}
        cls.kept = math.degrees(blank.kept_angle("wall_yneg"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_wall_turns_past_75_degrees(self) -> None:
        self.assertGreater(self.peak, 75.0)

    def test_the_fold_stays_once_the_hand_has_gone(self) -> None:
        self.assertGreater(self.after["wall_yneg"], 60.0)
        self.assertGreater(self.kept, 55.0)

    def test_the_other_walls_hold_themselves_out(self) -> None:
        for name in ("wall_ypos", "wall_xneg", "wall_xpos"):
            self.assertLess(self.after[name], 15.0, name)


if __name__ == "__main__":
    unittest.main()
