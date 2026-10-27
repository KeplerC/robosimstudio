# SPDX-License-Identifier: Apache-2.0
"""``ArmIK``'s own solves on a Newton model of a small arm: forward, inverse, and forward again.

The arm is :mod:`tests.fixtures.arms`' six-joint chain, built and finalized on
the CPU, with no URDF -- so every solve here is this tree's damped least
squares over Newton's forward kinematics, the path every arm without PyRoKi
takes. What is held is what the docstrings promise: the device FK, the host
chain walk and the closed form agree; a reachable target is reached (roll free
by :meth:`~ArmIK.solve_q`, the whole pose by :meth:`~ArmIK.solve_q_full`);
every answer is inside the joint limits; :meth:`~ArmIK.solve_best` keeps the
closest seed's answer; and :meth:`~ArmIK.plan_cartesian` cuts a line into its
waypoints, each on the line, or gives up past its tolerance.
"""

from __future__ import annotations

import functools
import unittest

import numpy as np

from robosimstudio.core.math import quat_from_axis_angle, quat_multiply, quat_to_mat
from tests.fixtures.arms import LIMIT, analytic_fk, cpu_arm, reachable

TIER = "fast"
"""A six-joint Newton model finalized on the CPU, and host-side solves over it: no GPU."""


def _near(q: np.ndarray, seed: int) -> np.ndarray:
    """Seeds a fifth of a radian off each answer: a solver is seeded from where the arm is, which is near.

    From the zero posture a third of these targets land in another basin, centimetres off,
    which is :meth:`ArmIK.solve_best`'s reason to exist and not a property of one solve.
    """
    return (q + np.random.default_rng(seed).normal(0.0, 0.2, q.shape)).astype(np.float32)


_arm = functools.cache(cpu_arm)
"""One model and its ``ArmIK`` for the module: the solves read it and change nothing on it."""


class _OnTheCpuArm(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.arm = _arm()
        cls.seed = np.zeros(cls.model.joint_coord_count, dtype=np.float32)


class TestTheForwardKinematics(_OnTheCpuArm):
    def test_the_device_path_the_host_walk_and_the_closed_form_agree(self) -> None:
        self.assertEqual("", self.arm.fk_host_reason, "the host chain walk validated against the device")
        q, position, quat = reachable(0, 8, spread=2.0)
        for row, want_p, want_q in zip(q, position, quat, strict=True):
            device_p, device_q = self.arm.fk_ee(row.astype(np.float32))
            with self.arm.host_fk():
                host_p, host_q = self.arm.fk_ee(row)
            for got_p, got_q in ((device_p, device_q), (host_p, host_q)):
                np.testing.assert_allclose(got_p, want_p, atol=1e-6)
                np.testing.assert_allclose(quat_to_mat(got_q), quat_to_mat(want_q), atol=1e-6)

    def test_the_home_pose_is_the_forward_kinematics_of_home(self) -> None:
        position, quat = self.arm.home_pose
        want_p, want_q = analytic_fk(np.zeros(6))
        np.testing.assert_allclose(position, want_p[0], atol=1e-6)
        np.testing.assert_allclose(quat_to_mat(quat), quat_to_mat(want_q[0]), atol=1e-6)


class TestSolveQ(_OnTheCpuArm):
    """Position and the tool's +z; the roll is the solver's to choose."""

    def test_a_reachable_target_is_reached_along_its_approach_axis(self) -> None:
        q, position, quat = reachable(1, 6)
        with self.arm.host_fk():
            for seed, target, orientation in zip(_near(q, 1), position, quat, strict=True):
                approach = quat_to_mat(orientation)[:, 2]
                solved = self.arm.solve_q(seed, target, approach)
                got_p, got_q = self.arm.fk_ee(solved)
                self.assertLess(float(np.linalg.norm(got_p - target)), 1.5e-3)
                self.assertLess(float(np.linalg.norm(np.cross(quat_to_mat(got_q)[:, 2], approach))), 3.0e-2)

    def test_every_answer_is_inside_the_joint_limits(self) -> None:
        """Out of reach too: each iteration clips, so the answer is a configuration the arm has."""
        targets = ((2.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.05, 0.0, 0.2), (-0.3, 0.3, 0.9))
        with self.arm.host_fk():
            for target in targets:
                with self.subTest(target=target):
                    solved = self.arm.solve_q(self.seed, target, (0.0, 0.0, -1.0), iterations=60)
                    self.assertTrue(np.all(np.abs(solved[:6]) <= LIMIT + 1e-6), solved)


class TestSolveQFull(_OnTheCpuArm):
    def test_forward_inverse_forward_round_trips_the_whole_pose(self) -> None:
        q, position, quat = reachable(2, 12)
        with self.arm.host_fk():
            for seed, target, orientation in zip(_near(q, 2), position, quat, strict=True):
                solved = self.arm.solve_q_full(seed, target, orientation)
                distance, angle = self.arm.pose_errors(solved, target, orientation)
                self.assertLess(distance, 1.0e-4)
                self.assertLess(angle, 1.0e-3)

    def test_a_quarter_turn_about_the_tool_axis_is_closed(self) -> None:
        """Same position and approach, rolled 90 degrees: what a roll-free solve leaves and the roll stage closes."""
        q = np.array([0.3, 0.4, 0.6, -0.2, 0.5, 0.2])
        (target,), (orientation,) = analytic_fk(q)
        rolled = quat_multiply(orientation, quat_from_axis_angle((0.0, 0.0, 1.0), np.pi / 2))
        with self.arm.host_fk():
            solved = self.arm.solve_q_full(q.astype(np.float32), target, rolled)
            distance, angle = self.arm.pose_errors(solved, target, rolled)
        self.assertLess(distance, 1.0e-4)
        self.assertLess(angle, 1.0e-3, "the roll was left open")


class TestPoseErrors(_OnTheCpuArm):
    def test_a_configuration_is_off_its_own_pose_by_nothing_and_off_a_moved_one_by_the_move(self) -> None:
        q = np.array([0.2, -0.3, 0.8, 0.1, 0.4, -0.6], dtype=np.float32)
        (position,), (quat,) = analytic_fk(q)
        distance, angle = self.arm.pose_errors(q, position, quat)
        self.assertLess(distance, 1e-6)
        self.assertLess(angle, 1e-3)
        moved = position + np.array([0.03, 0.04, 0.0])
        turned = quat_multiply(quat_from_axis_angle((1.0, 0.0, 0.0), 0.4), quat)
        distance, angle = self.arm.pose_errors(q, moved, turned)
        self.assertAlmostEqual(0.05, distance, places=6)
        self.assertAlmostEqual(0.4, angle, places=3)


class TestSolveBest(_OnTheCpuArm):
    def test_the_closest_seeds_answer_is_kept_with_its_own_error(self) -> None:
        (target,), (quat,) = analytic_fk(np.array([0.5, 0.6, 0.9, 0.0, 0.4, 0.0]))
        approach = quat_to_mat(quat)[:, 2]
        seeds = [np.full(6, -2.0, dtype=np.float32), np.array([0.5, 0.5, 0.8, 0.0, 0.4, 0.0], dtype=np.float32)]
        with self.arm.host_fk():
            alone = [self.arm.solve_q(s, target, approach, iterations=6) for s in seeds]
            errors = [float(np.linalg.norm(self.arm.fk_ee(q)[0] - target)) for q in alone]
            best, error = self.arm.solve_best(seeds, target, approach, iterations=6)
        self.assertNotAlmostEqual(errors[0], errors[1], places=4, msg="the two seeds must land apart to tell")
        self.assertEqual(min(errors), error)
        np.testing.assert_array_equal(alone[int(np.argmin(errors))], best)


class TestPlanCartesian(_OnTheCpuArm):
    def setUp(self) -> None:
        self.start = np.array([0.2, 0.1, 0.7, 0.0, 0.5, 0.0], dtype=np.float32)
        (self.first,), (quat,) = analytic_fk(self.start)
        self.approach = quat_to_mat(quat)[:, 2]

    def test_the_line_is_cut_into_its_waypoints_and_each_lies_on_it(self) -> None:
        end = self.first + np.array([0.08, -0.05, -0.06])
        with self.arm.host_fk():
            path = self.arm.plan_cartesian(self.start, self.first, end, self.approach, waypoints=5, tolerance=0.004)
            self.assertIsNotNone(path)
            self.assertEqual(5, len(path))
            for i, q in enumerate(path, start=1):
                want = self.first + (end - self.first) * i / 5
                self.assertLess(float(np.linalg.norm(self.arm.fk_ee(q)[0] - want)), 0.004, f"waypoint {i}")

    def test_a_line_past_its_tolerance_is_abandoned(self) -> None:
        with self.arm.host_fk():
            path = self.arm.plan_cartesian(
                self.start, self.first, (2.0, 0.0, 0.5), self.approach, waypoints=3, tolerance=0.01
            )
        self.assertIsNone(path)

    def test_no_tolerance_accepts_whatever_the_solver_reached(self) -> None:
        with self.arm.host_fk():
            path = self.arm.plan_cartesian(
                self.start, self.first, (2.0, 0.0, 0.5), self.approach, waypoints=3, tolerance=0.0, iterations=30
            )
        self.assertEqual(3, len(path))


class TestTheArmsKinematics(_OnTheCpuArm):
    """:class:`ArmKinematics.of`: what a solver family is handed about this arm."""

    def test_the_kinematics_read_the_limits_and_answer_with_the_arms_own_fk(self) -> None:
        from robosimstudio.solvers.ik.arm import ArmKinematics

        kinematics = ArmKinematics.of(self.arm, template=self.model.joint_q.numpy())
        self.assertEqual(6, kinematics.dof)
        np.testing.assert_allclose(kinematics.lower, -LIMIT)
        np.testing.assert_allclose(kinematics.upper, LIMIT)
        rows, position, quat = reachable(3, 3)
        got_p, got_q = kinematics.poses(rows)
        np.testing.assert_allclose(got_p, position, atol=1e-6)
        for got, want in zip(got_q, quat, strict=True):
            np.testing.assert_allclose(quat_to_mat(got), quat_to_mat(want), atol=1e-6)

    def test_without_a_template_there_is_no_whole_joint_vector(self) -> None:
        from robosimstudio.solvers.ik.arm import ArmKinematics

        kinematics = ArmKinematics.of(self.arm)
        with self.assertRaisesRegex(ValueError, "template"):
            kinematics.full(np.zeros(6))

    def test_a_batch_of_one_is_the_arms_own_solve(self) -> None:
        """``IKSolverDLS`` over one world on an arm runs ``ArmIK.solve_q_full`` itself, and measures its error by FK."""
        from robosimstudio.solvers.ik.arm import ArmKinematics
        from robosimstudio.solvers.ik.dls import IKSolverDLS

        template = self.model.joint_q.numpy()
        solver = IKSolverDLS(ArmKinematics.of(self.arm, template=template))
        _, position, quat = reachable(4, 1)
        with self.arm.host_fk():
            q, err = solver.solve(np.zeros((1, 6)), position, quat, roll="held")
            alone = self.arm.solve_q_full(template.astype(np.float64), position[0], quat[0])
            distance, angle = self.arm.pose_errors(alone, position[0], quat[0])
        np.testing.assert_array_equal(alone[:6].astype(np.float64), q[0])
        np.testing.assert_allclose([distance, angle], err[0], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
