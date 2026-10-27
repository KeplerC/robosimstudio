# SPDX-License-Identifier: Apache-2.0
"""The canonical quaternion and transform helpers in ``core.math``."""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio.core.math import (
    quat_from_axis_angle,
    quat_inverse,
    quat_multiply,
    quat_slerp,
    quat_to_axis_angle,
    quat_to_mat,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    transform_matrix,
    xform_from_matrix,
)

TIER = "fast"
"""No GPU: numpy only."""


class TestMath(unittest.TestCase):
    """The canonical quaternion helpers."""

    def test_wxyz_round_trip(self):
        """Converting to wxyz and back is the identity."""
        q = np.array([0.1, -0.2, 0.3, 0.9])
        q = q / np.linalg.norm(q)
        np.testing.assert_allclose(quat_wxyz_to_xyzw(quat_xyzw_to_wxyz(q)), q, atol=1e-12)

    def test_quat_multiply_matches_matrix_product(self):
        """Quaternion composition agrees with composing the rotation matrices."""
        a = quat_from_axis_angle((0, 0, 1), 0.7)
        b = quat_from_axis_angle((1, 0, 0), -0.4)
        np.testing.assert_allclose(quat_to_mat(quat_multiply(a, b)), quat_to_mat(a) @ quat_to_mat(b), atol=1e-12)

    def test_quat_inverse(self):
        """A rotation composed with its inverse is the identity."""
        q = quat_from_axis_angle((0.3, -0.5, 0.8), 2.1)
        np.testing.assert_allclose(quat_to_mat(quat_multiply(q, quat_inverse(q))), np.eye(3), atol=1e-12)

    def test_axis_angle_round_trip_takes_the_short_way(self):
        """Past half a turn the rotation vector comes back the near way round.

        A servo handed the long way round would take a joint through its limits
        to reach an orientation a few degrees away.
        """
        for angle in (0.1, 1.0, 3.0):
            back = quat_to_axis_angle(quat_from_axis_angle((0, 0, 1), angle))
            np.testing.assert_allclose(back, [0.0, 0.0, angle], atol=1e-9)
        self.assertAlmostEqual(quat_to_axis_angle(quat_from_axis_angle((0, 0, 1), 2 * np.pi - 0.3))[2], -0.3)

    def test_slerp_interpolates_the_short_arc(self):
        """Halfway between two rotations, the near way, whatever hemisphere they are in."""
        a = quat_from_axis_angle((0, 0, 1), 0.0)
        b = quat_from_axis_angle((0, 0, 1), 1.0)
        self.assertAlmostEqual(quat_to_axis_angle(quat_slerp(a, b, 0.5))[2], 0.5, places=9)
        # ``-b`` is the same rotation, so it must interpolate the same way.
        self.assertAlmostEqual(quat_to_axis_angle(quat_slerp(a, -b, 0.5))[2], 0.5, places=9)
        np.testing.assert_allclose(quat_slerp(a, b, 0.0), a, atol=1e-12)

    def test_xform_matrix_round_trip_at_180_degrees(self):
        """A 180-degree yaw survives the matrix round trip.

        The branch selection matters here: at 180 degrees ``w`` is 0, so a
        conversion that assumes ``w`` is the largest component divides by zero
        and produces a NaN pose. Half the camera rigs sit at this angle.
        """
        for angle in (0.0, np.pi / 2, np.pi, -np.pi + 1e-6):
            xform = np.append([1.0, -2.0, 3.0], quat_from_axis_angle((0, 0, 1), angle))
            back = xform_from_matrix(transform_matrix(xform))
            np.testing.assert_allclose(back[:3], xform[:3], atol=1e-12)
            # A quaternion and its negation are the same rotation.
            self.assertLess(min(np.linalg.norm(back[3:] - xform[3:]), np.linalg.norm(back[3:] + xform[3:])), 1e-9)


if __name__ == "__main__":
    unittest.main()
