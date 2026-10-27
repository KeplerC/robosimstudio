# SPDX-License-Identifier: Apache-2.0
"""``IKSolverPyroki``, the default solver family, over a URDF of an arm whose forward kinematics is known exactly.

:mod:`tests.fixtures.arms`' arm, its URDF written to a scratch directory, and
its closed-form forward kinematics standing in for the simulator's, placed on a
base turned and lifted off the world's origin. What is held: the base frame,
when nothing states it, is measured from how the tool moves; the URDF tool link
is calibrated onto the tool centre, so the solver's own forward kinematics is
the arm's; a held-roll solve reaches the whole pose and a roll-free one the
approach axis, each error measured, never assumed; a world that did not ask is
its seed; and a URDF that does not describe the arm -- links ten percent long --
is refused by the drift it would put in every solve.

Two jaxls problems are compiled here, one per roll contract, at three worlds:
the module's few seconds.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from robosimstudio.core.math import mat_to_quat, quat_from_axis_angle, quat_to_mat
from robosimstudio.solvers.ik.solver import pose_errors
from tests.fixtures.arms import LIMIT, analytic_arm, analytic_fk, reachable, urdf_arm

TIER = "fast"
"""JAX on the CPU over a six-joint URDF, and forward kinematics in closed form: no GPU."""

BASE = np.eye(4)
BASE[:3, :3] = quat_to_mat(quat_from_axis_angle((0.0, 0.0, 1.0), 0.7))
BASE[:3, 3] = (0.3, -0.2, 0.75)
"""Where the arm stands: turned 0.7 rad about the vertical and lifted 0.75 m."""


def _targets(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three reachable poses of the placed arm and seeds a fifth of a radian off: ``(seeds, position, quat)``."""
    q, _, _ = reachable(seed, 3)
    position, quat = analytic_fk(q, BASE)
    return q + np.random.default_rng(seed).normal(0.0, 0.2, q.shape), position, quat


class TestThePyrokiSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from robosimstudio.solvers.ik.pyroki import IKSolverPyroki

        cls.scratch = tempfile.TemporaryDirectory()
        cls.urdf = urdf_arm(Path(cls.scratch.name))
        cls.solver = IKSolverPyroki(analytic_arm(base=BASE, urdf=str(cls.urdf)))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.scratch.cleanup()

    def test_the_root_frame_is_measured_from_how_the_tool_moves(self) -> None:
        np.testing.assert_allclose(self.solver._t_base, BASE, atol=1e-6)

    def test_its_own_forward_kinematics_is_the_arms(self) -> None:
        q, _, _ = reachable(20, 5, spread=2.0)
        got_p, got_q = self.solver.tcp_poses(q)
        want_p, want_q = analytic_fk(q, BASE)
        np.testing.assert_allclose(got_p, want_p, atol=1e-6)
        for got, want in zip(got_q, want_q, strict=True):
            np.testing.assert_allclose(quat_to_mat(got), quat_to_mat(want), atol=1e-6)

    def test_a_held_roll_solve_reaches_the_whole_pose_inside_the_limits(self) -> None:
        seeds, position, quat = _targets(21)
        q, err = self.solver.solve(seeds, position, quat, roll="held")
        self.assertLess(float(err[:, 0].max()), 1.0e-5)
        self.assertLess(float(err[:, 1].max()), 1.0e-4)
        pos, got = analytic_fk(q, BASE)
        np.testing.assert_allclose(err, pose_errors(pos, got, position, quat, roll="held"), atol=1e-6)
        self.assertTrue(np.all(np.abs(q) <= LIMIT))

    def test_a_free_roll_solve_reaches_the_position_and_the_approach_axis(self) -> None:
        seeds, position, quat = _targets(22)
        _, err = self.solver.solve(seeds, position, quat, roll="free")
        self.assertLess(float(err[:, 0].max()), 1.0e-5)
        self.assertLess(float(err[:, 1].max()), 1.0e-4)

    def test_a_world_that_did_not_ask_is_its_seed(self) -> None:
        seeds, position, quat = _targets(23)
        asking = np.array([True, False, True])
        q, _ = self.solver.solve(seeds, position, quat, roll="free", active=asking)
        np.testing.assert_array_equal(seeds[1], q[1])
        self.assertFalse(np.array_equal(seeds[0], q[0]))


class TestWhatItRefuses(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.directory = Path(self.scratch.name)

    def test_a_urdf_with_links_a_tenth_too_long_is_refused_by_its_drift(self) -> None:
        from robosimstudio.solvers.ik.pyroki import IKSolverPyroki

        long_arm = urdf_arm(self.directory, scale=1.1)
        with self.assertRaisesRegex(ValueError, r"drifted \d+\.\d+ mm"):
            IKSolverPyroki(analytic_arm(base=BASE, urdf=str(long_arm)))

    def test_an_arm_that_names_no_urdf_is_refused(self) -> None:
        from robosimstudio.solvers.ik.pyroki import IKSolverPyroki

        with self.assertRaisesRegex(ValueError, "ships a URDF"):
            IKSolverPyroki(analytic_arm(base=BASE))

    def test_joints_the_urdf_does_not_have_are_refused(self) -> None:
        from robosimstudio.solvers.ik.pyroki import IKSolverPyroki

        kinematics = analytic_arm(base=BASE, urdf=str(urdf_arm(self.directory)))
        with self.assertRaisesRegex(ValueError, "do not cover"):
            IKSolverPyroki(kinematics, joints=("joint0", "joint1", "joint2", "joint3", "joint4", "elbow"))

    def test_a_stated_base_is_taken_as_given(self) -> None:
        """With ``base_xform`` stated nothing is measured, and the calibration still holds the URDF to the arm."""
        from robosimstudio.solvers.ik.pyroki import IKSolverPyroki

        stated = np.concatenate([BASE[:3, 3], mat_to_quat(BASE[:3, :3])])
        solver = IKSolverPyroki(analytic_arm(base=BASE, urdf=str(urdf_arm(self.directory)), base_xform=stated))
        np.testing.assert_allclose(solver._t_base, BASE, atol=1e-12)
        q, _, _ = reachable(24, 2)
        np.testing.assert_allclose(solver.tcp_poses(q)[0], analytic_fk(q, BASE)[0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
