# SPDX-License-Identifier: Apache-2.0
"""The box example's task, done by the right hand through the example's arm panel: cut the tape, open a flap.

``examples/03_materials/15_box`` promises that a cut box's flaps spring up a little and that a flap a finger
swings over stays open. This builds the example, cuts the tape the way its **Cut tape** button does, and walks
the right hand's cursor through the page's own IK and setpoint law along the path its take films
(``scripts/recording/takes.py``'s :func:`box_hook`): jaws open, down into the 6 mm seam, the +y finger under the
-y major flap 56 mm out from the crease, then round the crease in 5 degree steps to 150 degrees, up and away.
Nothing holds the box but its 1 kg of contents and the bench.

What it holds the run to: the cut flaps spring to their crease's yield (14.3 degrees) and the minor flaps under
them stay down; the swung flap passes 120 degrees, and two seconds after the hand has gone keeps more than 110;
the other major flap does not follow it; the box moves less than 5 mm.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""One GPU: the example built once and about 1,700 control steps of the hand opening it."""

PORT = 9851
SPEED = 0.06
"""How fast the cursor walks the arc [m/s]."""


class TestTheHandOpensTheBox(unittest.TestCase):
    """Cut the tape, hook a flap from the seam, swing it over: it stays open and the box stays put."""

    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from recording.takes import box_hook

        sim, kwargs = _run("03_materials/15_box.py")
        cls.sim = sim
        cls.session = session = open_session(sim, PORT, kwargs.get("page"))
        hz = float(session.env.request.physics.control_hz)
        box = sim["box"].entity
        cls.names = [h["name"] for h in box.hinges]

        session.arms["right"].grip_request = 0.0  # open: shut, the 9 mm jaw is wider than the gap the flaps leave
        tick(session, int(hz))
        box.cut_tape()
        tick(session, int(hz))
        cls.sprung = np.degrees(box.angles())
        path = box_hook(sim, session)
        go(session, "right", path["over"], speed=0.15)
        go(session, "right", path["seam"], speed=SPEED / 2)
        go(session, "right", path["hook"], speed=SPEED / 2)
        peak = 0.0
        for point in path["arc"]:
            go(session, "right", point, steps=40, speed=SPEED)
            peak = max(peak, math.degrees(box.angle("major_neg")))
        cls.peak = peak
        go(session, "right", path["up"], steps=60, speed=0.1)
        go(session, "right", path["away"], steps=120, speed=0.1)
        tick(session, 2 * int(hz))
        cls.after = np.degrees(box.angles())
        cls.moved = float(np.linalg.norm(session.env.state.body_q.numpy()[box.body, :3] - path["middle"]))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_cut_flaps_spring_to_their_yield(self) -> None:
        sprung = dict(zip(self.names, self.sprung, strict=True))
        for name in ("major_neg", "major_pos"):
            self.assertGreater(sprung[name], 10.0, name)
            self.assertLess(sprung[name], 18.0, name)
        for name in ("minor_neg", "minor_pos"):
            self.assertLess(sprung[name], 5.0, f"{name}: the major flaps over it keep it down")

    def test_the_swung_flap_passes_upright_and_stays_open(self) -> None:
        after = dict(zip(self.names, self.after, strict=True))
        self.assertGreater(self.peak, 120.0)
        self.assertGreater(after["major_neg"], 110.0, "the crease kept the bend")
        self.assertLess(after["major_pos"], 25.0, "the other flap did not follow")

    def test_the_box_stays_put(self) -> None:
        self.assertLess(self.moved, 0.005)


if __name__ == "__main__":
    unittest.main()
