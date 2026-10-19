# SPDX-License-Identifier: Apache-2.0
"""The between-posts example's hook, made by the robot's hand through the arm panel: close, pull, open, it springs back.

``tests/scenes/01_between_posts`` says the right hand can be driven down
onto the span's middle, closed on it, pulled back up and opened, and that the
cable springs back to where it hung. Its old clip did this with a click handle;
this drives the arm panel instead (:mod:`tests.fixtures.grasp`): the cursor
along the hand's own axis, as the gizmo's arrow drags it, and the grip **Close**
and **Open** ask for. Nothing latches the cable (``allow_grasp_latch=False``).

Measured on GPU 2 (RTX PRO 6000, Newton 1.6): the hand starts 80.0 mm back
along its axis from the middle and lands 2.9 mm from it; the jaws close at
16.3 mm on the 16 mm cable; a 30 mm pull lifts the middle 14.3 mm (and 13.8 mm
toward the arms, along the leaning axis) with the hand 9.9 mm short of its
cursor, the span being 22 mm of sag that does not stretch; 2 s after **Open**
the middle is within 0.1 mm of where it hung.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""Builds the example and steps it through a page for about 8 s of scene time."""

PORT = 9463
PULL = 0.03
"""How far the cursor is pulled back along the hand's axis once the jaws are shut [m]."""


class TestTheHandHooksTheSpan(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from robosimstudio.core.math import quat_to_mat

        cls.sim, kwargs = _run("tests/scenes/01_between_posts.py")
        sim = cls.sim
        cls.session = session = open_session(sim, PORT, kwargs.get("page"))
        cable, arm = sim["cable"], session.arms["right"]

        def middle() -> np.ndarray:
            p = sim.state.body_q.numpy()[cable.bodies, :3].astype(np.float64)
            return p[len(p) // 2 - 1 : len(p) // 2 + 1].mean(axis=0)

        tick(session, 90)
        cls.rest = middle()
        hand, quat = arm.ik.ee_pose(sim.state)
        axis = quat_to_mat(quat)[:, 2]
        ahead = cls.rest - hand
        cls.along = float(ahead @ axis)
        cls.off_axis = float(np.linalg.norm(ahead - cls.along * axis))
        cls.miss = go(session, "right", hand + cls.along * axis, speed=0.05)
        arm.grip_request = 1.0
        tick(session, 60)
        cls.closed = float(sim.gripper("right").width(sim.state))
        go(session, "right", hand + (cls.along - PULL) * axis, speed=0.05)
        tick(session, 60)
        cls.pulled = middle()
        cls.held = float(sim.gripper("right").width(sim.state))
        arm.grip_request = 0.0
        tick(session, 30)
        go(session, "right", hand + (cls.along - PULL - 0.05) * axis, speed=0.10)
        tick(session, 120)
        cls.back = middle()
        cls.finite = bool(sim.state_is_finite())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_hand_starts_pointing_at_the_middle(self) -> None:
        self.assertAlmostEqual(0.08, self.along, delta=0.01, msg="80 mm back along the hand's axis")
        self.assertLess(self.off_axis, 0.005, "and the axis through the middle")
        self.assertLess(self.miss, 0.008, "the hand got down to the cable")

    def test_the_jaws_close_on_the_cable_and_hold_it_through_the_pull(self) -> None:
        self.assertAlmostEqual(0.016, self.closed, delta=0.002, msg="the jaws on the 16 mm cable")
        self.assertAlmostEqual(0.016, self.held, delta=0.002, msg="still on it at the top of the pull")
        self.assertFalse(self.sim["cable"].entity.spec.allow_grasp_latch)

    def test_the_pull_lifts_the_middle(self) -> None:
        rise = float(self.pulled[2] - self.rest[2])
        self.assertGreater(rise, 0.008, f"the middle rose {rise * 1e3:.1f} mm")

    def test_opened_it_springs_back(self) -> None:
        self.assertTrue(self.finite)
        self.assertLess(float(np.linalg.norm(self.back - self.rest)), 0.002, "the middle is back where it hung")
        self.assertTrue(self.sim.pipeline.captured, self.sim.pipeline.capture_error)


if __name__ == "__main__":
    unittest.main()
