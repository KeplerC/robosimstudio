# SPDX-License-Identifier: Apache-2.0
"""A Sharpa Wave hand on a UR5e pinches a cube off the bench and holds it, by contact.

The gesture the studio asks for, through the code the gesture goes through
(:mod:`tests.fixtures.grasp`): the right arm's cursor goes to the cube, the
**Close** button asks for the pinch, the hand rises. Nothing is pinned; what
comes up came up between the thumb and the index finger's elastomers. The
left hand is asked to make a fist and to open again, through the posture path
the scalar command cannot express.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.fixtures.grasp import drag_to_grasp, note, open_session, tick

TIER = "medium"
"""One GPU: the bimanual robot built once, both hands driven through the page."""

BENCH = np.array([0.6, 0.0, 0.75])
"""``bench_075``'s surface frame in world [m]."""

CUBE = 0.0125
"""The cube's half-width [m]: 25 mm across, which a 124 mm pinch takes with room."""


class TestThePinchHoldsACube(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        scene = rss.Scene(cell="bench_075", name="sharpa_pinch", physics=rss.MUJOCO_500HZ)
        # Under the right hand's pinch point at home: on the middle line of the
        # bench, 0.2 m to the robot's right.
        cls.cube = scene.add(rss.box(name="cube", half=(CUBE, CUBE, CUBE), at=(0.0, -0.2, CUBE), mass=0.05))
        scene.add(rss.robot("ur5e_sharpa_bimanual"))
        cls.sim = sim = scene.build(control_mode="joint_position")
        sim.run(100).sync()
        cls.session = session = open_session(sim)
        tick(session, 50)
        cls.before = np.array(cls.cube.position, dtype=np.float64)
        # The tool centre 10 mm over the cube's centre: the pads are 12 mm thick,
        # so that puts their lower halves on the cube's sides, clear of the bench.
        cls.grasp = drag_to_grasp(
            session,
            "right",
            lambda: np.asarray(cls.cube.position),
            at=cls.before + np.array([0.0, 0.0, 0.01]),
            hover=0.06,
            lift=0.10,
            close=1.5,
            hold=2.0,
            speed=0.08,
        )
        note(
            f"sharpa pinch: rise {cls.grasp.rise * 1000:.1f} mm, drift {cls.grasp.drift * 1000:.1f} mm, reached {cls.grasp.reached * 1000:.1f} mm"
        )

    def test_the_scene_is_finite(self) -> None:
        self.assertTrue(self.grasp.finite)

    def test_the_cube_lay_on_the_bench_under_the_hand(self) -> None:
        np.testing.assert_allclose(self.before[2], BENCH[2] + CUBE, atol=0.004)

    def test_the_hand_got_where_it_was_dragged(self) -> None:
        self.assertLess(self.grasp.reached, 0.02)

    def test_the_cube_rose_with_the_hand_and_stayed_in_the_pinch(self) -> None:
        self.assertGreater(self.grasp.rise, 0.06)
        self.assertLess(self.grasp.drift, 0.02)

    def test_the_other_hand_makes_a_fist_and_opens_on_the_posture_path(self) -> None:
        sim, session = self.sim, self.session
        gripper = sim.gripper("left")
        fist = np.asarray(sim.scene.embodiment.hands["left"].postures["fist"])
        gripper.set_joint_targets(fist)
        tick(session, 100)
        q = sim.state.joint_q.numpy()[gripper.coords]
        self.assertLess(float(np.abs(q - fist).max()), 0.05)
        gripper.set_joint_targets(None)
        session.arms["left"].grip_request = 0.0
        tick(session, 100)
        q = sim.state.joint_q.numpy()[gripper.coords]
        # Back on the slider, open is the ready pinch, not the flat hand.
        self.assertLess(float(np.abs(q - gripper.open_q).max()), 0.05)


if __name__ == "__main__":
    unittest.main()
