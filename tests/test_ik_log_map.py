# SPDX-License-Identifier: Apache-2.0
"""The rotation log maps the IK solvers read their orientation errors with: ``|log R|`` is the angle, for every ``R``.

Two of them: :func:`robosimstudio.solvers.ik.fleet_dls._rotvecs`, batched, which
the full-pose polish measures its rotation residual with, and
:func:`robosimstudio.solvers.ik.pyroki._rotvec`, which PyRoKi's root-frame
measurement reads each joint's turn with. The property is the definition: the
vector's length is the rotation's angle and the exponential of the vector is
the rotation, on random rotations and on the ones a log map is easy to get
wrong at -- no turn, a tiny one, and a half turn, approached to within 1e-9 rad
and reached. A half turn read as nothing is a wrist upside down that the
polish calls closed.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "fast"
"""Pure numpy over 3x3 matrices: no GPU."""

HARD_ANGLES = (0.0, 1.0e-8, 1.0e-4, 0.5, np.pi / 2, 3.0, np.pi - 1.0e-3, np.pi - 1.0e-6, np.pi - 1.0e-7)
HARD_ANGLES += (np.pi - 1.0e-9, np.pi)
"""Where a log map divides by ``sin`` of the angle, and so where one goes wrong."""


def _exp(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues' formula: the rotation a rotation vector stands for."""
    angle = float(np.linalg.norm(rotvec))
    if angle < 1.0e-15:
        return np.eye(3)
    x, y, z = rotvec / angle
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _rotations(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``(angles [n], rotations [n, 3, 3])``: every hard angle about random axes, then random rotations."""
    rng = np.random.default_rng(seed)
    angles = np.concatenate([np.repeat(HARD_ANGLES, 4), rng.uniform(0.0, np.pi, 40)])
    axes = rng.normal(size=(len(angles), 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    return angles, np.stack([_exp(axis * angle) for axis, angle in zip(axes, angles, strict=True)])


class _LogMapLaw:
    """The two claims, over whichever log map a subclass reads with."""

    def log(self, rotations: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def test_the_length_is_the_angle(self) -> None:
        angles, rotations = _rotations(0)
        lengths = np.linalg.norm(self.log(rotations), axis=1)
        for angle, length in zip(angles, lengths, strict=True):
            self.assertAlmostEqual(angle, length, delta=1.0e-6, msg=f"a turn of pi - {np.pi - angle:.3g} rad")

    def test_the_exponential_of_the_log_is_the_rotation(self) -> None:
        angles, rotations = _rotations(1)
        for angle, rotation, rotvec in zip(angles, rotations, self.log(rotations), strict=True):
            np.testing.assert_allclose(_exp(rotvec), rotation, atol=1.0e-6, err_msg=f"a turn of {angle!r} rad")


class TestTheBatchedLogMap(_LogMapLaw, unittest.TestCase):
    """``fleet_dls._rotvecs``: what the full-pose polish measures the orientation it closes with."""

    def log(self, rotations: np.ndarray) -> np.ndarray:
        from robosimstudio.solvers.ik.fleet_dls import _rotvecs

        return _rotvecs(rotations)


class TestPyrokisLogMap(_LogMapLaw, unittest.TestCase):
    """``pyroki._rotvec``: how far each probe turned the tool, when PyRoKi measures the arm's root frame."""

    def log(self, rotations: np.ndarray) -> np.ndarray:
        from robosimstudio.solvers.ik.pyroki import _rotvec

        return np.stack([_rotvec(r) for r in rotations])


class TestAHalfTurnLeftOpenIsSeen(unittest.TestCase):
    """What the log map is for: the polish measures a wrist half a turn out as half a turn, not as closed."""

    def test_the_polish_error_of_a_half_turn_is_pi(self) -> None:
        from robosimstudio.solvers.ik.fleet_dls import _rotvecs

        half = np.diag([1.0, -1.0, -1.0])  # a half turn about x: the tool upside down
        for off in (0.0, 1.0e-7):
            with self.subTest(short_of_a_half_turn_by=off):
                turn = _exp(np.array([np.pi - off, 0.0, 0.0])) if off else half
                self.assertAlmostEqual(np.pi - off, float(np.linalg.norm(_rotvecs(turn[None])[0])), places=6)


if __name__ == "__main__":
    unittest.main()
