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

from robosimstudio_policy_agent import servo
from robosimstudio_policy_agent.fleet import stepper
from robosimstudio_policy_agent.servo import CartesianServo

TIER = "fast"
"""Quaternions and stub backends: no GPU, no scene."""


class SharedOrientationLawTest(unittest.TestCase):
    def test_stepper_uses_the_servo_helpers(self) -> None:
        self.assertIs(stepper.solver_goal, servo.solver_goal)
        self.assertIs(stepper.rotation_error, servo.rotation_error)
        self.assertFalse(hasattr(stepper, "_solver_goal"))
        self.assertFalse(hasattr(stepper.StepService, "_rotation_error"))

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


if __name__ == "__main__":
    unittest.main()
