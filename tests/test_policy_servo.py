# SPDX-License-Identifier: Apache-2.0
"""The serial servo's orientation law and roll contract.

The law (``solver_goal``, ``rotation_error``) is shared with the fleet stepper, and
whether arrival tests the roll follows the IK backend, read each time rather than
cached at construction."""

from __future__ import annotations

import types
import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.core.math import quat_from_axis_angle, quat_multiply, quat_to_mat
from robosimstudio_policy_agent import servo
from robosimstudio_policy_agent.fleet import stepper
from robosimstudio_policy_agent.servo import CartesianServo
from tests.fixtures.fleet import FakeSnapshot, step_service

TIER = "fast"
"""Quaternions and stub backends: no GPU, no scene."""


class SharedOrientationLawTest(unittest.TestCase):
    def test_stepper_uses_the_servo_helpers(self) -> None:
        """One law, shared: the stepper holds the servo's own two functions, and decides arrival by them.

        By identity, so a copy that drifted could not hide behind agreeing on a
        sample; and by what the stepper decides, so the law it holds is the one
        it applies: a roll-honouring move whose tool sits at ``-q`` of its goal
        ``q`` has arrived, as one at ``q`` has, because the two are one rotation.
        """
        self.assertIs(stepper.solver_goal, servo.solver_goal)
        self.assertIs(stepper.rotation_error, servo.rotation_error)
        rng = np.random.default_rng(11)
        for honours_roll in (False, True):
            for _ in range(20):
                a, b = rng.normal(size=4), rng.normal(size=4)
                a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
                np.testing.assert_allclose(
                    stepper.solver_goal(a, b, honours_roll), servo.solver_goal(a, b, honours_roll), atol=1e-12
                )
                self.assertAlmostEqual(
                    stepper.rotation_error(a, b, honours_roll), servo.rotation_error(a, b, honours_roll), places=12
                )
        goal = rng.normal(size=4)
        goal /= np.linalg.norm(goal)
        service = step_service()
        for measured in (goal, -goal):
            program = stepper._Program(
                "servo",
                goal=(np.array([0.4, 0.0, 0.3]), "arm", None),
                deadline=100,
                tolerance=0.01,
                requested_quat=goal,
                rot_tolerance=0.01,
                honours_roll=True,
            )
            snapshot = FakeSnapshot([[0.0, 0.0, 0.0]], body=[[0.4, 0.0, 0.3, *measured]])
            done, _error, result = service._finished(0, 0, program, snapshot)
            self.assertEqual((done, (result or {}).get("outcome")), (True, "reached"))

    def test_servo_methods_wrap_the_helpers(self) -> None:
        rng = np.random.default_rng(3)
        for honours_roll in (False, True):
            holder = SimpleNamespace(honours_roll=honours_roll)
            for _ in range(10):
                a = rng.normal(size=4)
                b = rng.normal(size=4)
                a /= np.linalg.norm(a)
                b /= np.linalg.norm(b)
                np.testing.assert_allclose(
                    servo.CartesianServo._solver_goal(holder, a, b), servo.solver_goal(a, b, honours_roll)
                )
                self.assertEqual(
                    servo.CartesianServo._rotation_error(holder, a, b), servo.rotation_error(a, b, honours_roll)
                )
        # Exactly opposed approach axes still give a unit quaternion.
        flipped = servo.solver_goal((0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0), False)
        self.assertAlmostEqual(float(np.linalg.norm(flipped)), 1.0)
        self.assertAlmostEqual(servo.rotation_error((0.0, 0.0, 0.0, 1.0), flipped, False), np.pi)


class ServoRollContract(unittest.TestCase):
    def test_follows_the_backend_after_a_fallback(self):
        ik = types.SimpleNamespace(honours_roll=True)
        servo = CartesianServo.__new__(CartesianServo)
        servo.ik = ik
        self.assertTrue(servo.honours_roll)
        ik.honours_roll = False  # what IKBackend._adopt does on a roll-dropping fallback
        self.assertFalse(servo.honours_roll)

    def test_a_direct_assignment_still_overrides(self):
        servo = CartesianServo.__new__(CartesianServo)
        servo.ik = types.SimpleNamespace(honours_roll=True)
        servo.honours_roll = False
        self.assertFalse(servo.honours_roll)


def _unit_quats(seed: int, count: int = 25) -> np.ndarray:
    rng = np.random.default_rng(seed)
    quats = rng.normal(size=(count, 4))
    return quats / np.linalg.norm(quats, axis=1, keepdims=True)


def _approach(quat) -> np.ndarray:
    return quat_to_mat(np.asarray(quat, dtype=np.float64))[:, 2]


class TestTheOrientationLawsProperties(unittest.TestCase):
    """``solver_goal`` and ``rotation_error`` checked against what they claim, not against themselves."""

    def test_a_roll_free_goal_points_the_tool_where_the_goal_does(self):
        for measured, goal in zip(_unit_quats(1), _unit_quats(2), strict=True):
            np.testing.assert_allclose(_approach(servo.solver_goal(measured, goal, False)), _approach(goal), atol=1e-9)

    def test_it_turns_the_measured_pose_by_no_more_than_the_axes_are_apart(self):
        """The shortest arc: the turn is the angle between the approach axes, so no roll is added."""
        for measured, goal in zip(_unit_quats(3), _unit_quats(4), strict=True):
            restated = servo.solver_goal(measured, goal, False)
            turned = servo.rotation_error(measured, restated, True)
            apart = float(np.arccos(np.clip(np.dot(_approach(measured), _approach(goal)), -1.0, 1.0)))
            self.assertAlmostEqual(turned, apart, places=6)

    def test_a_roll_honouring_goal_is_the_goal_and_an_aligned_one_is_the_measurement(self):
        measured, goal = _unit_quats(5, 2)
        np.testing.assert_array_equal(servo.solver_goal(measured, goal, True), goal)
        rolled = quat_multiply(measured, quat_from_axis_angle((0.0, 0.0, 1.0), 0.7))
        np.testing.assert_array_equal(servo.solver_goal(measured, rolled, False), measured)

    def test_q_and_minus_q_are_one_rotation(self):
        for quat in _unit_quats(6):
            for honours_roll in (False, True):
                with self.subTest(honours_roll=honours_roll):
                    self.assertAlmostEqual(servo.rotation_error(quat, -quat, honours_roll), 0.0, places=6)

    def test_a_roll_about_the_approach_axis_counts_only_when_the_roll_is_honoured(self):
        for quat in _unit_quats(7, 10):
            rolled = quat_multiply(quat, quat_from_axis_angle((0.0, 0.0, 1.0), 0.5))
            self.assertAlmostEqual(servo.rotation_error(quat, rolled, False), 0.0, places=6)
            self.assertAlmostEqual(servo.rotation_error(quat, rolled, True), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
