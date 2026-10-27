# SPDX-License-Identifier: Apache-2.0
"""How far a part is into a bore, and how far a part is from the hands.

:func:`insertion` reports depth, lateral offset, the travel still to go and the
tilt together, because depth alone scores a plug that slid past the port face as
seated; its readings are checked against the closed form of a part on, off and
across the bore axis, and against the identity ``depth + mate_gap == bore
depth``. ``tcp_distance`` and ``hands_clear`` are the release tests: the work
arm's TCP resolved by role, through its body pose and offset, and the nearest
TCP of every arm -- both infinite, never zero, when there is no arm to ask.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from robosimstudio.assetlib.semantics.snapshot import Snapshot
from robosimstudio.core.math import quat_from_axis_angle
from robosimstudio.domains.rigid.measures.insertion import _rigid_insertion, insertion
from robosimstudio.domains.rigid.measures.jaws import _hands_clear, _tcp_distance
from robosimstudio.scene.handles import ArmHandle, SceneHandles
from tests.fixtures.snapshots import ctx

TIER = "fast"
"""Closed-form poses and a scripted snapshot: no model, no GPU."""

BORE = {"axis": [0.0, 0.0, -1.0], "mouth": [0.2, 0.1, 0.5], "depth": 0.013}
"""A 13 mm bore opening upward at (0.2, 0.1, 0.5): a part goes in by moving down."""

DOWN = quat_from_axis_angle((1.0, 0.0, 0.0), math.pi)
"""A part whose own +z points down the bore."""


def _pose(position, quat=DOWN) -> np.ndarray:
    return np.array([*position, *quat], dtype=np.float64)


class TestInsertion(unittest.TestCase):
    def test_a_part_on_the_axis_is_as_deep_as_it_has_travelled_and_dead_centre(self) -> None:
        got = insertion(_pose((0.2, 0.1, 0.5 - 0.009)), BORE)
        self.assertAlmostEqual(0.009, got["depth"], places=12)
        self.assertAlmostEqual(0.0, got["lat"], places=12)
        self.assertAlmostEqual(0.004, got["mate_gap"], places=12)
        self.assertAlmostEqual(0.0, got["tilt"], places=7)

    def test_depth_and_the_travel_left_always_sum_to_the_bore(self) -> None:
        rng = np.random.default_rng(3)
        for _ in range(20):
            quat = quat_from_axis_angle(rng.normal(size=3), float(rng.uniform(0.0, np.pi)))
            pose = _pose(rng.uniform(-0.1, 0.6, size=3), quat)
            got = insertion(pose, BORE, tip_offset=float(rng.uniform(0.0, 0.01)))
            self.assertAlmostEqual(BORE["depth"], got["depth"] + got["mate_gap"], places=12)

    def test_a_plug_that_slid_past_the_face_is_deep_and_far_off_the_axis(self) -> None:
        """The case depth alone scores as inserted: beyond the mouth plane, 40 mm to the side."""
        got = insertion(_pose((0.2 + 0.04, 0.1, 0.5 - 0.011)), BORE)
        self.assertAlmostEqual(0.011, got["depth"], places=12)
        self.assertAlmostEqual(0.04, got["lat"], places=12)

    def test_a_part_still_in_transport_reads_a_negative_depth(self) -> None:
        got = insertion(_pose((0.2, 0.1, 0.5 + 0.05)), BORE)
        self.assertAlmostEqual(-0.05, got["depth"], places=12)
        self.assertAlmostEqual(0.063, got["mate_gap"], places=12)

    def test_the_tip_is_read_through_the_parts_own_tilt(self) -> None:
        """A 6 mm tab tilted 10 degrees reaches 6 cos(10) mm down the axis and 6 sin(10) mm across it."""
        tilt = math.radians(10.0)
        quat = quat_from_axis_angle((1.0, 0.0, 0.0), math.pi - tilt)
        got = insertion(_pose((0.2, 0.1, 0.5), quat), BORE, tip_offset=0.006)
        self.assertAlmostEqual(0.006 * math.cos(tilt), got["depth"], places=12)
        self.assertAlmostEqual(0.006 * math.sin(tilt), got["lat"], places=12)
        self.assertAlmostEqual(tilt, got["tilt"], places=9)

    def test_tilt_is_unsigned_nose_up_reads_as_nose_down(self) -> None:
        up = insertion(_pose((0.2, 0.1, 0.49), (0.0, 0.0, 0.0, 1.0)), BORE)
        down = insertion(_pose((0.2, 0.1, 0.49)), BORE)
        self.assertAlmostEqual(0.0, up["tilt"], places=7)
        self.assertAlmostEqual(down["tilt"], up["tilt"], places=7)
        sideways = insertion(_pose((0.2, 0.1, 0.49), quat_from_axis_angle((0.0, 1.0, 0.0), math.pi / 2)), BORE)
        self.assertAlmostEqual(math.pi / 2, sideways["tilt"], places=9, msg="the most it can be")

    def test_an_axis_given_at_any_length_measures_the_same(self) -> None:
        pose = _pose((0.21, 0.09, 0.49), quat_from_axis_angle((0.2, 1.0, 0.1), 2.9))
        unit, long = insertion(pose, BORE), insertion(pose, {**BORE, "axis": [0.0, 0.0, -25.0]})
        for key in ("depth", "lat", "mate_gap", "tilt"):
            self.assertAlmostEqual(unit[key], long[key], places=12, msg=key)

    def test_the_handler_takes_the_tip_offset_from_the_bore(self) -> None:
        pose = _pose((0.2, 0.1, 0.5 - 0.002))
        self.assertEqual(insertion(pose, BORE, tip_offset=0.005), _rigid_insertion(pose, {**BORE, "tip_offset": 0.005}))
        self.assertEqual(insertion(pose, BORE), _rigid_insertion(pose, BORE))


QUARTER_ABOUT_X = quat_from_axis_angle((1.0, 0.0, 0.0), math.pi / 2)


def _hands(*arms: ArmHandle, work: str = "") -> Snapshot:
    """Bodies 0 and 1 are the two wrists: one at the origin turned about x, one a metre along +x."""
    body_q = np.array([[0.0, 0.0, 0.0, *QUARTER_ABOUT_X], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    handles = SceneHandles(arms={arm.name: arm for arm in arms}, work_arm=work)
    return Snapshot(ctx=ctx(handles=handles, body_q=body_q))


LEFT = ArmHandle(name="left", ee_body=0, ee_offset=(0.0, 0.0, 0.1))
RIGHT = ArmHandle(name="right", ee_body=1, ee_offset=(0.0, 0.0, 0.1))


class TestTheHandsDistance(unittest.TestCase):
    def test_with_no_arm_the_hands_are_infinitely_far_never_zero(self) -> None:
        empty = _hands()
        self.assertEqual(math.inf, _tcp_distance(empty, (0.0, 0.0, 0.0))["distance"])
        self.assertEqual(math.inf, _hands_clear(empty, (0.0, 0.0, 0.0))["distance"])

    def test_the_tcp_is_the_offset_carried_through_the_wrists_pose(self) -> None:
        """A quarter turn about x carries the 100 mm offset along +z onto -y."""
        self.assertAlmostEqual(0.0, _tcp_distance(_hands(LEFT), (0.0, -0.1, 0.0))["distance"], places=12)
        self.assertAlmostEqual(
            0.1 * math.sqrt(2.0), _tcp_distance(_hands(LEFT), (0.0, 0.0, 0.1))["distance"], places=12
        )

    def test_the_work_arm_is_the_one_asked_whatever_order_the_arms_are_listed_in(self) -> None:
        near_right = (1.0, 0.0, 0.1)
        for arms in ((LEFT, RIGHT), (RIGHT, LEFT)):
            with self.subTest(first=arms[0].name):
                self.assertAlmostEqual(
                    0.0, _tcp_distance(_hands(*arms, work="right"), near_right)["distance"], places=12
                )
        self.assertGreater(_tcp_distance(_hands(LEFT, RIGHT, work="left"), near_right)["distance"], 0.9)

    def test_with_no_work_arm_named_the_first_arm_is_asked(self) -> None:
        self.assertAlmostEqual(0.0, _tcp_distance(_hands(RIGHT, LEFT), (1.0, 0.0, 0.1))["distance"], places=12)

    def test_hands_clear_is_the_nearest_hand_and_never_further_than_the_work_hand(self) -> None:
        snapshot = _hands(LEFT, RIGHT, work="left")
        for point in ((0.9, 0.0, 0.1), (0.0, -0.05, 0.0), (0.5, 0.3, -0.2)):
            with self.subTest(point=point):
                nearest = min(np.linalg.norm(np.subtract(point, tcp)) for tcp in ((0.0, -0.1, 0.0), (1.0, 0.0, 0.1)))
                self.assertAlmostEqual(nearest, _hands_clear(snapshot, point)["distance"], places=12)
                self.assertLessEqual(
                    _hands_clear(snapshot, point)["distance"], _tcp_distance(snapshot, point)["distance"]
                )

    def test_an_arm_with_no_tool_body_in_this_state_is_not_a_hand(self) -> None:
        detached = ArmHandle(name="ghost", ee_body=7, ee_offset=(0.0, 0.0, 0.0))
        unset = ArmHandle(name="unset")
        self.assertEqual(math.inf, _tcp_distance(_hands(detached, work="ghost"), (0.0, 0.0, 0.0))["distance"])
        self.assertAlmostEqual(
            0.0, _hands_clear(_hands(detached, unset, RIGHT), (1.0, 0.0, 0.1))["distance"], places=12
        )


if __name__ == "__main__":
    unittest.main()
