# SPDX-License-Identifier: Apache-2.0
"""The dough example's gesture, made the way its page makes it, leaves a dent the puck keeps.

``examples/03_materials/13_dough`` promises that a closed hand dragged down
into the puck and lifted away leaves it lower, and its page reads the puck's
height as a share of the height it settled at. This drives the example's own
arm panel through :mod:`tests.fixtures.grasp` -- the grip the Close button
writes, the cursor the gizmo writes, the session's own setpoint law -- and reads
the same number under the hand and after it lifts. Nothing is pinned and no
joint target is written: the dough goes down because the fingers push it.
"""

from __future__ import annotations

import time
import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""One GPU: the example built once and stepped through its page for about 6 s of scene time."""

PORT = 9650
PRESS = 0.015
"""How far under the puck's settled top the cursor goes [m], as ``tests/test_mpm_kinds`` presses."""
SPEED = 0.04
"""The cursor's walk into the dough and out of it [m/s]."""


class TestThePressByHand(unittest.TestCase):
    """Close, straight down from where the hand is to 15 mm under the top, hold, lift: the dent stays."""

    @classmethod
    def setUpClass(cls) -> None:
        from robosimstudio_materials.mpm.measures import heap_height

        cls.sim, kwargs = _run("03_materials/13_dough.py")
        cls.sim.run(20).sync()
        t = time.perf_counter()
        cls.sim.run(100).sync()
        cls.ms_per_step = 1e3 * (time.perf_counter() - t) / 100
        cls.session = open_session(cls.sim, PORT, kwargs.get("page"))
        lump = cls.sim["dough"]
        rest, floor = lump.entity.rest["particle_q"][lump.particles], float(lump.entity.floor_z)
        settled = heap_height(rest, floor)

        def height() -> float:
            """The page's readout: the puck's height over the height it settled at."""
            return heap_height(np.asarray(lump.particle_q, dtype=np.float64), floor) / settled

        session, arm = cls.session, cls.session.arms["right"]
        hz = float(cls.sim.request.physics.control_hz)
        tick(session, 30)
        cls.before = height()
        arm.grip_request = 1.0
        tick(session, round(0.5 * hz))
        hand = arm.ik.ee_pose(session.env.state)[0]
        top = floor + settled
        over = np.array([hand[0], hand[1], top + 0.04])
        cls.miss_over = go(session, "right", over)
        cls.miss_press = go(session, "right", np.array([hand[0], hand[1], top - PRESS]), speed=SPEED, steps=30)
        tick(session, round(0.5 * hz))
        cls.pressed = height()
        go(session, "right", over, speed=SPEED)
        tick(session, round(1.0 * hz))
        cls.after = height()
        cls.finite = bool(session.env.state_is_finite())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_scene_steps_captured_inside_its_budget(self) -> None:
        self.assertTrue(self.sim.pipeline.captured, self.sim.pipeline.capture_error)
        self.assertLess(self.ms_per_step, 40.0, f"{self.ms_per_step:.1f} ms per control step; the docstring says 23.4")

    def test_the_puck_stood_at_its_settled_height_before_the_hand(self) -> None:
        self.assertGreater(self.before, 0.97, f"the puck read {self.before:.3f} of its settled height untouched")
        self.assertLess(self.miss_over, 0.006, "the hand got over the puck")

    def test_the_hand_took_the_dough_down(self) -> None:
        self.assertTrue(self.finite)
        self.assertLessEqual(self.pressed, 0.90, f"the puck read {self.pressed:.3f} of its height under the hand")

    def test_most_of_the_dent_stays_after_the_hand_lifts(self) -> None:
        kept = (1.0 - self.after) / (1.0 - self.pressed)
        self.assertGreaterEqual(
            kept, 0.6, f"{self.pressed:.3f} under the hand, {self.after:.3f} after: {kept:.2f} kept"
        )
        self.assertTrue(self.sim.pipeline.captured, "the gesture kept the graph")


if __name__ == "__main__":
    unittest.main()
