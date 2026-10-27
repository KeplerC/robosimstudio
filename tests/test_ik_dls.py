# SPDX-License-Identifier: Apache-2.0
"""The arithmetic under the batched boundary solve, against the arithmetic it copies.

:mod:`robosimstudio.solvers.ik.fleet_dls` runs
:meth:`~robosimstudio.solvers.ik.arm_ik.ArmIK.solve_q`'s search for a whole wave
at once, and its whole claim is that it is the *same* search -- the descent is a
fixed-point iteration whose basin choice flips on float noise, so an operand
that differs in the fifth decimal is not a rounding difference, it is a
different answer.

This module exists because that went wrong once, in the way it will go wrong
again. The batched rotation matrix was written as the textbook formula with its
hard-coded factor of two; the scalar one this tree has always used divides by
the quaternion's own norm instead, because -- as its docstring says -- "a
quaternion that has drifted off the unit sphere (which read-back float32 poses
routinely have) still yields an orthogonal matrix rather than one that quietly
scales". These quaternions come straight off a float32 ``body_q`` read, so they
have drifted, and the two forms disagreed by 5e-05 rad in the first iteration.

Then the batched solve itself, over :mod:`tests.fixtures.arms`' closed-form
arm with no Newton at all: a wave equals its rows solved alone, a world that
did not ask comes back as it went in, a reachable target is reached and every
answer is inside the limits, the full-pose polish keeps the promises its
docstring lists, and a straight line names the waypoint it could not meet.

No GPU: numpy, and forward kinematics in closed form.
"""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio.core.math import mat_to_axis_angle, quat_from_axis_angle, quat_multiply, quat_to_mat
from robosimstudio.solvers.ik.fleet_dls import (
    POLISH_MAX_MOVE,
    POLISH_MAX_STEP,
    POLISH_POS_SLACK,
    POLISH_ROT_WEIGHT,
    _quat_to_mats,
    polish_pose_batched,
    pose_errors_batched,
    solve_q_batched,
    solve_q_full_batched,
)
from robosimstudio.solvers.ik.solver import approach_axes, plan_line, pose_errors
from tests.fixtures.arms import LIMIT, analytic_arm, reachable

TIER = "fast"
"""Pure numpy, the arm's forward kinematics in closed form. Gates every commit."""


def _drifted(rng, n: int, scale: float) -> np.ndarray:
    """Unit quaternions nudged off the sphere, the way a float32 read leaves them."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1)[:, None]
    return q * (1.0 + rng.normal(scale=scale, size=(n, 1)))


class TestTheBatchedRotationIsTheScalarOne(unittest.TestCase):
    """Row for row, including the normalisation that is the whole point."""

    def test_unit_quaternions_agree(self):
        rng = np.random.default_rng(3)
        quats = _drifted(rng, 64, 0.0)

        batched = _quat_to_mats(quats)
        one_at_a_time = np.stack([quat_to_mat(q) for q in quats])

        self.assertLess(float(np.abs(batched - one_at_a_time).max()), 1e-15)

    def test_drifted_quaternions_agree(self):
        """The case that actually occurs, and the one the bug was in.

        A float32 ``body_q`` read comes back off the unit sphere by ~1e-7; this
        exaggerates it to make the failure unmistakable rather than marginal.
        """
        rng = np.random.default_rng(4)
        for scale in (1e-7, 1e-4, 1e-2):
            with self.subTest(drift=scale):
                quats = _drifted(rng, 64, scale)
                batched = _quat_to_mats(quats)
                one_at_a_time = np.stack([quat_to_mat(q) for q in quats])
                self.assertLess(float(np.abs(batched - one_at_a_time).max()), 1e-15)

    def test_a_drifted_quaternion_still_gives_an_orthogonal_matrix(self):
        """What the normalisation buys, stated directly: an un-normalised form
        returns a matrix that quietly scales, and a scaled 'rotation' inside a
        Jacobian is a wrong gradient rather than a small one."""
        rng = np.random.default_rng(5)
        mats = _quat_to_mats(_drifted(rng, 32, 1e-2))

        for mat in mats:
            np.testing.assert_allclose(mat @ mat.T, np.eye(3), atol=1e-12)

    def test_a_zero_quaternion_is_the_identity(self):
        """The scalar form's guard, which a batched divide would turn into a nan."""
        out = _quat_to_mats(np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]))

        np.testing.assert_allclose(out[0], np.eye(3), atol=0)
        np.testing.assert_allclose(out[1], quat_to_mat([0.0, 0.0, 0.0, 1.0]), atol=0)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_a_quarter_turn_about_z_is_what_it_should_be(self):
        """One matrix checked by eye, so the whole family is anchored to
        something a reader can verify without running anything."""
        half = np.sqrt(0.5)
        mat = _quat_to_mats(np.array([[0.0, 0.0, half, half]]))[0]

        np.testing.assert_allclose(mat @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


class _Arm:
    """The closed-form arm as the ``fk.poses`` object the batched solves walk."""

    def __init__(self) -> None:
        self.kinematics = analytic_arm()
        self.lower, self.upper = self.kinematics.lower, self.kinematics.upper

    def poses(self, arm_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.kinematics.fk(arm_q)


def _wave(seed: int, worlds: int = 8):
    """``worlds`` reachable targets and seeds a fifth of a radian off each answer: ``(seeds, position, quat)``."""
    q, position, quat = reachable(seed, worlds)
    return q + np.random.default_rng(seed).normal(0.0, 0.2, q.shape), position, quat


def _weighted(fk: _Arm, q: np.ndarray, position: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The polish's own measure: ``(position error [W], weighted 6-D error [W])``."""
    pos, quat = fk.poses(q)
    turn = mat_to_axis_angle(rotation @ np.transpose(_quat_to_mats(quat), (0, 2, 1)))
    gap = position - pos
    return np.linalg.norm(gap, axis=1), np.linalg.norm(np.concatenate([gap, POLISH_ROT_WEIGHT * turn], axis=1), axis=1)


class TestTheBatchedSolveIsItsRows(unittest.TestCase):
    """A wave is ``W`` searches run as one, and each world's answer is the one it gets alone."""

    def setUp(self) -> None:
        self.fk = _Arm()
        self.seeds, self.position, self.quat = _wave(5)

    def test_a_wave_equals_each_world_solved_alone(self) -> None:
        axes = approach_axes(self.quat)
        kw = {"lower": self.fk.lower, "upper": self.fk.upper}
        wave = solve_q_batched(self.fk, self.seeds, self.position, axes, **kw)
        for w in range(len(wave)):
            alone = solve_q_batched(self.fk, self.seeds[w : w + 1], self.position[w : w + 1], axes[w : w + 1], **kw)
            np.testing.assert_array_equal(alone[0], wave[w], err_msg=f"world {w}")

    def test_the_full_pose_wave_equals_each_world_solved_alone(self) -> None:
        kw = {"lower": self.fk.lower, "upper": self.fk.upper}
        wave = solve_q_full_batched(self.fk, self.seeds, self.position, self.quat, **kw)
        for w in range(len(wave)):
            alone = solve_q_full_batched(
                self.fk, self.seeds[w : w + 1], self.position[w : w + 1], self.quat[w : w + 1], **kw
            )
            np.testing.assert_array_equal(alone[0], wave[w], err_msg=f"world {w}")

    def test_a_world_that_did_not_ask_comes_back_bit_identical(self) -> None:
        asking = np.array([True, False] * 4)
        kw = {"lower": self.fk.lower, "upper": self.fk.upper, "active": asking}
        rotation = _quat_to_mats(self.quat)
        for name, out in (
            ("solve_q", solve_q_batched(self.fk, self.seeds, self.position, approach_axes(self.quat), **kw)),
            ("solve_q_full", solve_q_full_batched(self.fk, self.seeds, self.position, self.quat, **kw)),
            ("polish", polish_pose_batched(self.fk, self.seeds, self.position, rotation, **kw)),
        ):
            with self.subTest(solve=name):
                np.testing.assert_array_equal(self.seeds[~asking], out[~asking])
                self.assertFalse(np.array_equal(self.seeds[asking], out[asking]), "the asking worlds moved")


class TestTheBatchedSolveReaches(unittest.TestCase):
    def setUp(self) -> None:
        self.fk = _Arm()

    def test_a_reachable_wave_is_reached_with_its_approach_axes(self) -> None:
        seeds, position, quat = _wave(6)
        q = solve_q_batched(self.fk, seeds, position, approach_axes(quat), lower=self.fk.lower, upper=self.fk.upper)
        pos, got = self.fk.poses(q)
        err = pose_errors(pos, got, position, quat, roll="free")
        self.assertLess(float(err[:, 0].max()), 1.5e-3, "the descent's own stopping ball")
        self.assertLess(float(err[:, 1].max()), 3.0e-2)

    def test_the_full_pose_wave_closes_position_and_rotation(self) -> None:
        seeds, position, quat = _wave(7)
        q = solve_q_full_batched(self.fk, seeds, position, quat, lower=self.fk.lower, upper=self.fk.upper)
        distance, angle = pose_errors_batched(self.fk, q, position, quat)
        self.assertLess(float(distance.max()), 1.0e-4)
        self.assertLess(float(angle.max()), 1.0e-3)

    def test_every_answer_is_inside_the_limits_reachable_or_not(self) -> None:
        position = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.05, 0.0, 0.2], [0.0, 0.3, 0.9]])
        quat = np.tile([0.0, 1.0, 0.0, 0.0], (4, 1))
        seeds = np.full((4, 6), 2.4)
        for name, q in (
            (
                "solve_q",
                solve_q_batched(
                    self.fk,
                    seeds,
                    position,
                    approach_axes(quat),
                    lower=self.fk.lower,
                    upper=self.fk.upper,
                    iterations=60,
                ),
            ),
            (
                "solve_q_full",
                solve_q_full_batched(
                    self.fk, seeds, position, quat, lower=self.fk.lower, upper=self.fk.upper, iterations=60
                ),
            ),
        ):
            with self.subTest(solve=name):
                self.assertTrue(np.all(np.abs(q) <= LIMIT), q)


class TestThePolishKeepsItsPromises(unittest.TestCase):
    """:func:`polish_pose_batched`, against the pose it was handed, as its docstring lists them."""

    def setUp(self) -> None:
        self.fk = _Arm()
        q, self.position, quat = reachable(8, 8)
        self.rotation = _quat_to_mats(quat)
        # Handed a pose a few degrees and millimetres off, as stage 2 hands it one.
        self.handed = q + np.random.default_rng(8).normal(0.0, 0.03, q.shape)
        self.polished = polish_pose_batched(
            self.fk, self.handed, self.position, self.rotation, lower=self.fk.lower, upper=self.fk.upper
        )

    def test_neither_the_position_nor_the_weighted_error_grows(self) -> None:
        before_p, before_w = _weighted(self.fk, self.handed, self.position, self.rotation)
        after_p, after_w = _weighted(self.fk, self.polished, self.position, self.rotation)
        np.testing.assert_array_less(after_p, np.maximum(before_p, POLISH_POS_SLACK) + 1e-12)
        np.testing.assert_array_less(after_w, before_w + 1e-12)
        self.assertLess(float(after_w.max()), 0.1 * float(before_w.min()), "and on a reachable pose it closes")

    def test_no_joint_moves_further_than_the_whole_polish_allows(self) -> None:
        """Handed a pose a third of a radian off, an unbounded polish swings a joint two radians to close it."""
        far = self.handed + np.random.default_rng(9).normal(0.0, 0.3, self.handed.shape)
        out = polish_pose_batched(self.fk, far, self.position, self.rotation, lower=self.fk.lower, upper=self.fk.upper)
        self.assertLessEqual(float(np.abs(out - far).max()), POLISH_MAX_MOVE + 1e-12)
        self.assertLessEqual(float(np.abs(self.polished - self.handed).max()), POLISH_MAX_MOVE + 1e-12)

    def test_one_step_moves_no_joint_more_than_its_cap(self) -> None:
        far = self.handed + 0.25
        once = polish_pose_batched(
            self.fk, far, self.position, self.rotation, lower=self.fk.lower, upper=self.fk.upper, steps=1
        )
        self.assertLessEqual(float(np.abs(once - far).max()), POLISH_MAX_STEP + 1e-12)

    def test_a_pose_handed_in_inside_the_limits_stays_inside_them(self) -> None:
        edge = np.clip(self.handed, -LIMIT, LIMIT)
        edge[:, 5] = LIMIT  # the wrist roll against its stop
        out = polish_pose_batched(self.fk, edge, self.position, self.rotation, lower=self.fk.lower, upper=self.fk.upper)
        self.assertTrue(np.all(np.abs(out) <= LIMIT))


class TestTheBatchedErrors(unittest.TestCase):
    def test_a_pose_is_off_itself_by_nothing_and_off_a_turned_one_by_the_turn(self) -> None:
        fk = _Arm()
        q, position, quat = reachable(9, 3)
        distance, angle = pose_errors_batched(fk, q, position, quat)
        np.testing.assert_allclose(distance, 0.0, atol=1e-12)
        np.testing.assert_allclose(angle, 0.0, atol=1e-6)
        turned = np.stack([quat_multiply(quat_from_axis_angle((0.0, 1.0, 0.0), 0.7), row) for row in quat])
        _, angle = pose_errors_batched(fk, q, position + np.array([0.0, 0.0, 0.02]), turned)
        np.testing.assert_allclose(angle, 0.7, atol=1e-6)

    def test_a_roll_about_the_approach_is_free_under_free_and_counted_under_held(self) -> None:
        _, position, quat = reachable(10, 3)
        rolled = np.stack([quat_multiply(row, quat_from_axis_angle((0.0, 0.0, 1.0), 0.5)) for row in quat])
        free = pose_errors(position, rolled, position, quat, roll="free")
        held = pose_errors(position, rolled, position, quat, roll="held")
        np.testing.assert_allclose(free, 0.0, atol=1e-6)
        np.testing.assert_allclose(held[:, 1], 0.5, atol=1e-6)


class TestTheDlsSolver(unittest.TestCase):
    """:class:`IKSolverDLS` over a batched forward kinematics: the protocol's contract."""

    def setUp(self) -> None:
        from robosimstudio.solvers.ik.dls import IKSolverDLS

        self.solver = IKSolverDLS(analytic_arm())
        self.seeds, self.position, self.quat = _wave(11, 4)

    def test_the_errors_it_reports_are_measured_by_forward_kinematics(self) -> None:
        for roll in ("free", "held"):
            with self.subTest(roll=roll):
                q, err = self.solver.solve(self.seeds, self.position, self.quat, roll=roll)
                pos, got = self.solver.kinematics.fk(q)
                want = pose_errors(pos, got, self.position, self.quat, roll=roll)
                np.testing.assert_allclose(err[:, 0], want[:, 0], atol=1e-12)
                np.testing.assert_allclose(err[:, 1], want[:, 1], atol=1e-6)  # arccos of a trace near 1
                self.assertLess(float(err[:, 0].max()), 1.5e-3)

    def test_an_inactive_world_is_its_seed(self) -> None:
        asking = np.array([False, True, True, False])
        q, _ = self.solver.solve(self.seeds, self.position, self.quat, roll="held", active=asking)
        np.testing.assert_array_equal(self.seeds[~asking], q[~asking])


class TestAStraightLine(unittest.TestCase):
    """:func:`plan_line`: each waypoint on the segment, or the first that missed, named."""

    def setUp(self) -> None:
        from robosimstudio.solvers.ik.dls import IKSolverDLS

        self.solver = IKSolverDLS(analytic_arm(), iterations=120)
        self.seed, start, self.quat = reachable(12, 2)
        self.start = start
        self.end = start + np.array([[0.06, -0.04, -0.05], [0.04, 0.03, -0.05]])

    def test_each_waypoint_lies_on_the_line_and_no_world_is_refused(self) -> None:
        path, refusals = plan_line(
            self.solver, self.seed, self.start, self.end, self.quat, waypoints=4, tolerance=0.004
        )
        self.assertEqual((2, 4, 6), path.shape)
        self.assertEqual([None, None], refusals)
        for w in range(2):
            pos, _ = self.solver.kinematics.fk(path[w])
            want = self.start[w] + np.outer(np.arange(1, 5) / 4, self.end[w] - self.start[w])
            self.assertLess(float(np.linalg.norm(pos - want, axis=1).max()), 0.004, f"world {w}")

    def test_an_unreachable_end_names_the_first_waypoint_that_missed_and_the_position(self) -> None:
        end = self.end.copy()
        end[1] = [2.0, 0.0, 0.3]
        path, (kept, refused) = plan_line(
            self.solver, self.seed, self.start, end, self.quat, waypoints=5, tolerance=0.01
        )
        self.assertIsNone(kept)
        self.assertEqual((1, 5, "the position"), (refused.world, refused.waypoints, refused.binding))
        self.assertGreater(refused.position_error, 0.01)
        self.assertGreaterEqual(refused.waypoint, 2, "the first fifth of the way is still in reach")
        self.assertTrue(np.all(path[1, refused.waypoint :] == path[1, refused.waypoint - 1]), "a refused world stops")
        text = refused.text(tolerance=0.01, rot_tolerance=np.radians(5.0))
        self.assertIn(f"at waypoint {refused.waypoint} of 5", text)
        self.assertIn("so the position is what could not be met", text)

    def test_a_held_rotation_past_its_tolerance_is_named_as_the_binding(self) -> None:
        _, (refused, _) = plan_line(
            self.solver, self.seed, self.start, self.end, self.quat, roll="held", waypoints=2, tolerance=1.0,
            rot_tolerance=1.0e-12,
        )  # fmt: skip
        self.assertEqual("the held rotation", refused.binding)
        self.assertEqual(1, refused.waypoint)

    def test_a_shorter_leg_repeats_its_last_row(self) -> None:
        path, refusals = plan_line(
            self.solver, self.seed, self.start, self.end, self.quat, waypoints=np.array([2, 4]), tolerance=0.004
        )
        self.assertEqual([None, None], refusals)
        np.testing.assert_array_equal(path[0, 2], path[0, 1])
        np.testing.assert_array_equal(path[0, 3], path[0, 1])


if __name__ == "__main__":
    unittest.main()
