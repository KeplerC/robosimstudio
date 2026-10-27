# SPDX-License-Identifier: Apache-2.0
"""One arm's IK for every world of a fleet at once: the batched forward kinematics and the device LM rig.

:mod:`tests.fixtures.arms`' arm replicated into three worlds on the CPU, as a
fleet's compiler replicates a scene. :class:`FleetArmFK` answers every world
with one ``eval_fk`` and has to answer each exactly as that world's arm does;
the batched DLS walks it to every world's target. :class:`RobotIKRig` with a
:class:`BatchedArmIK` runs Newton's Levenberg-Marquardt for every world and
writes the targets of the worlds it drives -- and nothing else -- into the
fleet's ``joint_target_q``. The refusals are the index-space mismatches its
docstrings name.
"""

from __future__ import annotations

import functools
import unittest

import numpy as np

from robosimstudio.core.math import quat_to_mat
from tests.fixtures.arms import TOOL, analytic_fk, fleet_arm, reachable

TIER = "fast"
"""A three-world copy of a six-joint arm on the CPU; the LM rig runs eager on Warp's CPU device: no GPU."""

WORLDS = 3

_fleet = functools.cache(lambda: fleet_arm(WORLDS))
"""One replicated model for the module; the tests read it and write only their own arrays."""


def _fk():
    from robosimstudio.solvers.ik.fleet_fk import FleetArmFK

    _, model, worlds, tool = _fleet()
    fk = FleetArmFK(model, worlds, arm_coords=range(6), ee_body=tool, ee_offset=TOOL)
    fk.seed_from(model.joint_q.numpy())
    return fk


class TestTheFleetForwardKinematics(unittest.TestCase):
    def test_each_world_is_answered_as_its_own_arm(self) -> None:
        q, position, quat = reachable(30, WORLDS, spread=2.0)
        got_p, got_q = _fk().poses(q)
        np.testing.assert_allclose(got_p, position, atol=1e-6)
        for got, want in zip(got_q, quat, strict=True):
            np.testing.assert_allclose(quat_to_mat(got), quat_to_mat(want), atol=1e-6)

    def test_a_world_not_asking_holds_where_it_was_seeded(self) -> None:
        fk = _fk()
        live = np.arange(WORLDS * 6, dtype=np.float32).reshape(WORLDS, 6) / 10.0
        fk.seed_from(live.reshape(-1))
        np.testing.assert_allclose(fk.live_arm_coords(), live, atol=1e-7)

    def test_a_worlds_pose_of_the_wrong_width_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "a world's pose is 6 coordinates, got 5"):
            _fk().seed_world(1, np.zeros(5))

    def test_the_batched_dls_over_it_reaches_every_world(self) -> None:
        from robosimstudio.solvers.ik.fleet_dls import pose_errors_batched, solve_q_full_batched

        q, position, quat = reachable(31, WORLDS)
        seeds = q + np.random.default_rng(31).normal(0.0, 0.2, q.shape)
        fk = _fk()
        lower, upper = np.full(6, -2.5), np.full(6, 2.5)
        solved = solve_q_full_batched(fk, seeds, position, quat, lower=lower, upper=upper)
        distance, angle = pose_errors_batched(fk, solved, position, quat)
        self.assertLess(float(distance.max()), 1.0e-4)
        self.assertLess(float(angle.max()), 1.0e-3)


class TestTheDeviceRig(unittest.TestCase):
    """Newton's LM over every world, scattered into the fleet's targets."""

    def _rig(self, *, iterations: int = 50):
        from robosimstudio.solvers.ik.batched import BatchedArmIK, RobotIKRig

        template, _, worlds, tool = _fleet()
        home = analytic_fk(np.zeros(6))
        arm = BatchedArmIK(
            template,
            num_worlds=WORLDS,
            arm_coords=range(6),
            arm_dofs=range(6),
            scene_coords=range(6),
            ee_body=tool,
            home_pose=(home[0][0], home[1][0]),
            ee_offset=TOOL,
            iterations=iterations,
        )
        rig = RobotIKRig(template, num_worlds=WORLDS, coords_per_world=worlds.coords_per_world, iterations=iterations)
        rig.add_arm("arm", arm)
        return rig, arm

    def test_the_driven_worlds_are_solved_and_only_they_are_written(self) -> None:
        import warp as wp

        rig, arm = self._rig()
        _, position, quat = reachable(32, WORLDS)
        arm.set_targets(position, quat)
        arm.set_enabled([1, 0, 1])
        rig.flush()
        joint_q = wp.zeros(WORLDS * 6, dtype=wp.float32, device="cpu")
        targets = wp.array(np.full(WORLDS * 6, 9.0, dtype=np.float32), dtype=wp.float32, device="cpu")
        rig.solve_into(joint_q, targets)
        written = targets.numpy().reshape(WORLDS, 6)
        np.testing.assert_array_equal(np.full(6, 9.0, dtype=np.float32), written[1], "a world not driven is left alone")
        solved = rig.solution()
        np.testing.assert_array_equal(solved[[0, 2]], written[[0, 2]])
        reached, _ = analytic_fk(solved)
        self.assertLess(float(np.linalg.norm(reached - position, axis=1)[[0, 2]].max()), 1.0e-4)
        self.assertTrue(rig.active)
        self.assertEqual([1, 0, 1], arm.enabled().tolist())

    def test_an_arm_whose_two_index_spaces_do_not_pair_is_refused(self) -> None:
        from robosimstudio.solvers.ik.batched import BatchedArmIK

        template, _, _, tool = _fleet()
        with self.assertRaisesRegex(ValueError, r"arm_coords \(6\) and scene_coords \(5\) must pair one to one"):
            BatchedArmIK(
                template,
                num_worlds=WORLDS,
                arm_coords=range(6),
                arm_dofs=range(6),
                scene_coords=range(5),
                ee_body=tool,
                home_pose=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0)),
            )

    def test_a_rig_refuses_a_map_of_the_wrong_length_and_an_arm_of_another_fleet(self) -> None:
        from robosimstudio.solvers.ik.batched import RobotIKRig

        template, _, worlds, _ = _fleet()
        with self.assertRaisesRegex(ValueError, "coord_map must have 6 entries, got 4"):
            RobotIKRig(template, num_worlds=WORLDS, coords_per_world=6, coord_map=[0, 1, 2, 3])
        _, arm = self._rig()
        other = RobotIKRig(template, num_worlds=WORLDS + 1, coords_per_world=worlds.coords_per_world)
        with self.assertRaisesRegex(ValueError, f"arm 'arm' solves {WORLDS} worlds, rig has {WORLDS + 1}"):
            other.add_arm("arm", arm)


if __name__ == "__main__":
    unittest.main()
