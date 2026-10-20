# SPDX-License-Identifier: Apache-2.0
"""The pour example's gesture, made through its arm panel: the hand carries the cup and pours the honey into the bowl.

``examples/03_materials/12_liquid`` promises that the hand, dragged down over
the cup and closed, lifts it, carries it over the bowl and pours it when the
gizmo is turned. This drives the example's own page the way
:mod:`tests.fixtures.grasp` does -- the cursor the gizmo writes, the grip the
Close button writes, the session's own setpoint law -- and turns the cursor the
way the gizmo's rings do. Nothing is pinned and no handle is down: the cup comes
up because the pads hold it, and the honey leaves it because the wrist tips it.
"""

from __future__ import annotations

import math
import time
import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""One GPU: the example built once and stepped through its page for about 13 s of scene time."""

PORT = 9647
BENCH = np.array([0.6, 0.0, 0.75])
"""``bench_075``'s surface frame in world [m]; the example's numbers are in it."""
BOWL_LO = BENCH + np.array([0.05, -0.23, -0.005])
BOWL_HI = BENCH + np.array([0.19, -0.09, 0.04])
"""The bowl's inside as the example states it: 140 mm square about bench (0.12, -0.16), 40 mm walls."""

GRASP = 0.055
"""Tool centre over the cup's bottom at the close [m]: the pads from 48 mm up to the 78 mm rim."""
LIFT_TO = 0.15
"""Where the cursor lifts the tool centre to, over the bench [m]; the loaded arm hangs about 30 mm under it."""
ACROSS = 0.14
"""How far the cup is carried toward the middle of the bench [m]: the cup's lip ends over the bowl."""
YAW, TILT = 40.0, 140.0
"""The two turns of the gizmo [deg]: about the vertical, then about world x, tipping the cup toward the bowl."""
SPEED = 0.10
"""The cursor's walk for the moves [m/s], under the setpoint law's 0.31 m/s cruise."""
THROUGH = 12
"""How many of the 8,000 particles may be more than 20 mm under the bench top at any step watched.

The honey leaves the cup in one mass from a lip 79 mm over the bench, and the
leading edge lands at ~1.3 m/s, past the ~0.83 m/s at which a 120 Hz step still
finds the bench's collider band; the few that cross it fall through. Seven
pours of this gesture ended with 0, 1, 1, 2, 2, 7 and 8 under the bench
(``robot_pour`` and this test's own, GPU 1, RTX PRO 6000, Newton 1.6.0); a
lower pour into a 20 mm dish lost 3, and an 8 mm bowl floor with the
container margin under it lost 2 and 6, so neither is in the example. Twelve
is above what was seen and far below anything a missed bowl would show."""


def _turn(session, axis, degrees: float, seconds: float) -> None:
    """Turn the right arm's cursor ``degrees`` about the world ``axis``, through its centre, at a steady rate.

    What a drag of the gizmo's ring writes, a control step at a time, held as a
    drag until the last step.
    """
    from robosimstudio.core.math import quat_from_axis_angle, quat_multiply

    panel = session.arms["right"]
    pos0 = np.array(panel.cursor_pos, dtype=np.float64)
    q0 = np.array(panel.cursor_quat, dtype=np.float64)
    n = max(1, round(seconds * float(session.env.request.physics.control_hz)))
    for k in range(1, n + 1):
        turn = quat_from_axis_angle(np.asarray(axis, dtype=np.float64), math.radians(degrees) * k / n)
        panel.cursor_pos, panel.cursor_quat, panel.dragging = pos0, quat_multiply(turn, q0), k < n
        tick(session, 1)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    """The angle between two rotation matrices [deg]."""
    return math.degrees(math.acos(float(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0))))


def _pour(session, cup, honey) -> dict:
    """Down onto the cup, **Close**, up, across, turn about the vertical, tip, hold, and back upright."""
    from robosimstudio.core.math import quat_to_mat

    panel = session.arms["right"]
    hz = float(session.env.request.physics.control_hz)
    out: dict = {"through": 0, "slip_mm": 0.0, "slip_deg": 0.0}

    def hand() -> tuple[np.ndarray, np.ndarray]:
        pos, quat = panel.ik.ee_pose(session.env.state)
        return pos, quat_to_mat(quat)

    def in_hand() -> tuple[np.ndarray, np.ndarray]:
        pos, rot = hand()
        pose = np.asarray(cup.pose, dtype=np.float64)
        return rot.T @ (pose[:3] - pos), rot.T @ quat_to_mat(pose[3:7])

    def watch(steps: int, grip: tuple | None = None) -> None:
        for _ in range(steps):
            tick(session, 1)
            points = np.asarray(honey.particle_q, dtype=np.float64)
            out["through"] = max(out["through"], int((points[:, 2] < BENCH[2] - 0.02).sum()))
            if grip is not None:
                p, r = in_hand()
                out["slip_mm"] = max(out["slip_mm"], 1e3 * float(np.linalg.norm(p - grip[0])))
                out["slip_deg"] = max(out["slip_deg"], _angle(grip[1], r))

    standing = np.asarray(cup.position, dtype=np.float64)
    at = standing + np.array([0.0, 0.0, GRASP])
    out["miss_grasp"] = go(session, "right", at, speed=SPEED)
    panel.grip_request = 1.0
    watch(round(hz))
    out["width"] = float(session.env.gripper("right").width(session.env.state))
    grip = in_hand()
    out["miss_lift"] = go(session, "right", np.array([at[0], at[1], BENCH[2] + LIFT_TO]), speed=SPEED, steps=60)
    out["miss_carry"] = go(
        session, "right", np.array(panel.cursor_pos) + np.array([0.0, ACROSS, 0.0]), speed=SPEED, steps=60
    )
    watch(round(0.5 * hz), grip)
    out["carried"] = np.asarray(cup.position, dtype=np.float64)
    before = hand()[1]
    _turn(session, (0.0, 0.0, 1.0), YAW, 1.0)
    _turn(session, (1.0, 0.0, 0.0), -TILT, 2.0)
    watch(round(2.0 * hz), grip)
    out["turned_deg"] = _angle(before, hand()[1])
    out["poured"] = float(
        np.mean(np.all((np.asarray(honey.particle_q) >= BOWL_LO) & (np.asarray(honey.particle_q) <= BOWL_HI), axis=1))
    )
    _turn(session, (1.0, 0.0, 0.0), TILT, 2.0)
    watch(round(1.0 * hz), grip)
    out["up_z"] = float(quat_to_mat(np.asarray(cup.pose, dtype=np.float64)[3:7])[2, 2])
    out["width_end"] = float(session.env.gripper("right").width(session.env.state))
    out["finite"] = bool(session.env.state_is_finite())
    return out


class TestTheHandPoursTheHoney(unittest.TestCase):
    """The cup in the fingers, over the bowl, tipped 140 degrees: the honey ends in the bowl."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim, kwargs = _run("03_materials/12_liquid.py")
        cls.sim.run(20).sync()
        t = time.perf_counter()
        cls.sim.run(100).sync()
        cls.ms_per_step = 1e3 * (time.perf_counter() - t) / 100
        cls.session = open_session(cls.sim, PORT, kwargs.get("page"))
        tick(cls.session, 30)
        cls.honey = cls.sim["honey"]
        cls.result = _pour(cls.session, cls.sim["cup"], cls.honey)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_scene_steps_captured_inside_its_budget(self) -> None:
        self.assertTrue(self.sim.pipeline.captured, self.sim.pipeline.capture_error)
        self.assertLess(self.ms_per_step, 60.0, f"{self.ms_per_step:.1f} ms per control step; the docstring says 29.8")

    def test_the_fingers_closed_on_the_cup(self) -> None:
        self.assertLess(self.result["miss_grasp"], 0.006, "the hand did not get down around the cup")
        self.assertAlmostEqual(0.066, self.result["width"], delta=0.003, msg="the jaws are not on the cup's walls")

    def test_the_cup_was_carried_over_the_bowl(self) -> None:
        carried = self.result["carried"]
        self.assertGreater(float(carried[2]), BOWL_HI[2], "the cup's bottom is not above the bowl's walls")
        self.assertGreater(float(carried[1]), BOWL_LO[1], "the cup did not reach the bowl")

    def test_the_cup_stayed_in_the_fingers(self) -> None:
        r = self.result
        self.assertTrue(r["finite"])
        self.assertLess(r["slip_mm"], 10.0, f"the cup moved {r['slip_mm']:.1f} mm in the hand")
        self.assertLess(r["slip_deg"], 3.0, f"the cup turned {r['slip_deg']:.2f} deg in the hand")
        self.assertAlmostEqual(0.066, r["width_end"], delta=0.003, msg="the jaws are no longer on the cup")

    def test_the_wrist_tipped_the_cup(self) -> None:
        self.assertGreater(self.result["turned_deg"], 135.0, f"the hand turned {self.result['turned_deg']:.1f} deg")

    def test_the_honey_is_in_the_bowl(self) -> None:
        poured = self.result["poured"]
        self.assertGreaterEqual(poured, 0.9, f"{poured:.3f} of the honey is in the bowl")
        self.assertLessEqual(self.result["through"], THROUGH, "honey went through the bench")

    def test_the_cup_came_back_upright(self) -> None:
        self.assertGreater(self.result["up_z"], 0.98, "the cup was not turned back upright")
        self.assertTrue(self.sim.pipeline.captured, "the gesture kept the graph")


if __name__ == "__main__":
    unittest.main()
