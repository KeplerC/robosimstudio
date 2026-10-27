# SPDX-License-Identifier: Apache-2.0
"""``control/batched``: every world's joint targets driven from device arrays, run on Warp's CPU device.

A fleet plans a motion per world once and lets the device walk it. What is held
is the walk's own contract: a segment lands on its end exactly and holds it, its
smoothstep is symmetric and monotone, a path visits its waypoints one per step
from the first and keeps its ends however it is resampled, a segment planned
late starts from the device's clock rather than finishing at once, and a world
nothing drives -- or a coordinate the driver does not own -- is left
bit-identical. The gripper bank ramps linearly and a re-grip continues from the
blend it interrupted.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "fast"
"""Warp kernels launched on its CPU device: no GPU."""

WORLDS, STRIDE, DT = 3, 4, 0.1
COORDS = (1, 3)
"""The two coordinates of each world's four that the driver owns."""


def _targets(fill: float = 7.0):
    import warp as wp

    return wp.array(np.full(WORLDS * STRIDE, fill, dtype=np.float32), dtype=wp.float32, device="cpu")


def _driver(**kwargs):
    from robosimstudio.solvers.control.batched import SegmentDriver

    return SegmentDriver(
        num_worlds=WORLDS, scene_coords=COORDS, coords_per_world=STRIDE, control_dt=DT, device="cpu", **kwargs
    )


def _owned(targets, world: int) -> np.ndarray:
    return targets.numpy().reshape(WORLDS, STRIDE)[world, list(COORDS)]


class TestASegment(unittest.TestCase):
    def setUp(self) -> None:
        self.driver = _driver()
        self.targets = _targets()
        self.q0, self.q1 = np.array([0.0, 1.0]), np.array([2.0, -3.0])

    def _walk(self, steps: int) -> list[np.ndarray]:
        out = []
        for _ in range(steps):
            self.driver.drive(self.targets)
            out.append(_owned(self.targets, 0).copy())
        return out

    def test_it_lands_on_its_end_exactly_and_holds_there(self) -> None:
        self.driver.plan(0, self.q0, self.q1, 4)
        self.driver.flush()
        walked = self._walk(7)
        for row in walked[3:]:
            np.testing.assert_array_equal(self.q1.astype(np.float32), row)

    def test_its_smoothstep_is_symmetric_and_monotone(self) -> None:
        self.driver.plan(0, self.q0, self.q1, 10)
        self.driver.flush()
        share = np.array([(row[0] - self.q0[0]) / (self.q1[0] - self.q0[0]) for row in self._walk(10)])
        s = np.concatenate([[0.0], share])  # s(k / 10), k = 0..10
        np.testing.assert_allclose(s + s[::-1], 1.0, atol=1e-5)
        self.assertTrue(np.all(np.diff(s) > 0.0), s)
        self.assertLess(s[1], 0.05, "it eases in: the first tenth of the time covers less than a tenth of the way")

    def test_it_is_finished_on_its_last_step_and_not_before(self) -> None:
        self.driver.plan(0, self.q0, self.q1, 5)
        self.driver.flush()
        self._walk(4)
        self.assertFalse(self.driver.finished()[0])
        self._walk(1)
        self.assertTrue(self.driver.finished()[0])
        np.testing.assert_array_equal(self.q1.astype(np.float32), self.driver.target(0))

    def test_one_planned_after_the_clock_ran_starts_from_the_clock(self) -> None:
        """Stamped off the device's clock: a segment planned a hundred steps in still takes its own ten."""
        self.driver.plan(0, self.q0, self.q0, 1)
        self.driver.flush()
        self._walk(100)
        self.driver.plan(0, self.q0, self.q1, 10)
        self.driver.flush()
        (first,) = self._walk(1)
        self.assertLess((first[0] - self.q0[0]) / (self.q1[0] - self.q0[0]), 0.05, "it jumped to the end")

    def test_a_world_not_driven_and_a_coordinate_not_owned_are_left_as_they_were(self) -> None:
        self.driver.plan(0, self.q0, self.q1, 3)
        self.driver.plan(2, self.q1, self.q0, 3)
        self.driver.stop(2)
        self.driver.flush()
        self._walk(5)
        grid = self.targets.numpy().reshape(WORLDS, STRIDE)
        np.testing.assert_array_equal(np.full(STRIDE, 7.0, dtype=np.float32), grid[1], "a world never planned")
        np.testing.assert_array_equal(np.full(STRIDE, 7.0, dtype=np.float32), grid[2], "a world stopped")
        np.testing.assert_array_equal(np.full(2, 7.0, dtype=np.float32), grid[0, [0, 2]], "coordinates not owned")


class TestAPath(unittest.TestCase):
    def test_it_visits_each_waypoint_in_turn_from_the_first_and_holds_the_last(self) -> None:
        driver, targets = _driver(), _targets()
        points = np.array([[0.0, 0.0], [1.0, -1.0], [3.0, 2.0]], dtype=np.float32)
        self.assertEqual(3, driver.plan_path(1, points))
        driver.flush()
        walked = []
        for _ in range(5):
            driver.drive(targets)
            walked.append(_owned(targets, 1).copy())
        np.testing.assert_array_equal(points, walked[:3])
        np.testing.assert_array_equal([points[-1]] * 2, walked[3:])

    def test_a_path_longer_than_its_room_keeps_its_ends(self) -> None:
        driver, targets = _driver(max_path=4), _targets()
        points = np.stack([np.linspace(0.0, 9.0, 10), np.linspace(5.0, -4.0, 10)], axis=1)
        steps = driver.plan_path(0, points, steps_per_waypoint=2)
        self.assertEqual((4 - 1) * 2 + 1, steps)
        driver.flush()
        driver.drive(targets)
        np.testing.assert_array_equal(points[0].astype(np.float32), _owned(targets, 0))
        for _ in range(steps):
            driver.drive(targets)
        np.testing.assert_array_equal(points[-1].astype(np.float32), _owned(targets, 0))

    def test_a_path_of_no_waypoints_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one waypoint"):
            _driver().plan_path(0, np.zeros((0, 2)))


class TestTheGripperBank(unittest.TestCase):
    def _bank(self):
        from robosimstudio.solvers.control.batched import GripperBank

        return GripperBank(
            num_worlds=WORLDS, coords=(0, 2), open_q=(0.04, -0.04), closed_q=(0.0, 0.0), coords_per_world=STRIDE,
            device="cpu",
        )  # fmt: skip

    def test_a_ramp_is_linear_and_a_regrip_continues_from_the_blend(self) -> None:
        bank, targets = self._bank(), _targets()
        bank.set(1, 1.0, ramp_steps=4)
        bank.flush()
        fingers = []
        for _ in range(2):
            bank.apply(targets)
            fingers.append(targets.numpy().reshape(WORLDS, STRIDE)[1, 0])
        np.testing.assert_allclose(fingers, [0.04 * 0.75, 0.04 * 0.5], atol=1e-7)
        bank.set(1, 0.0, ramp_steps=2)  # half closed, now opened again over two steps
        bank.flush()
        bank.apply(targets)
        self.assertAlmostEqual(0.04 * 0.75, float(targets.numpy().reshape(WORLDS, STRIDE)[1, 0]), places=6)
        self.assertEqual([0.0, 0.0, 0.0], bank.commands().tolist())

    def test_a_world_whose_hand_is_off_is_not_written(self) -> None:
        bank, targets = self._bank(), _targets()
        bank.set_all([1.0, 1.0, 1.0])
        bank.enables.set(2, False)
        bank.flush()
        bank.apply(targets)
        grid = targets.numpy().reshape(WORLDS, STRIDE)
        np.testing.assert_array_equal([0.0, 0.0], grid[0, [0, 2]])
        np.testing.assert_array_equal(np.full(STRIDE, 7.0, dtype=np.float32), grid[2])


class TestAbsoluteValues(unittest.TestCase):
    def test_only_the_enabled_worlds_owned_coordinates_are_written(self) -> None:
        import warp as wp

        from robosimstudio.solvers.control.batched import WorldEnables, write_coord_values

        targets = _targets()
        enables = WorldEnables(WORLDS, "cpu")
        enables.set_all([1, 0, 1])
        enables.flush()
        values = wp.array(np.arange(WORLDS * 2, dtype=np.float32).reshape(WORLDS, 2), dtype=wp.float32, device="cpu")
        coords = wp.array(np.asarray(COORDS, dtype=np.int32), dtype=wp.int32, device="cpu")
        write_coord_values(values, enables.array, coords, STRIDE, targets, num_worlds=WORLDS, width=2, device="cpu")
        grid = targets.numpy().reshape(WORLDS, STRIDE)
        np.testing.assert_array_equal([[7, 0, 7, 1], [7, 7, 7, 7], [7, 4, 7, 5]], grid)
        self.assertEqual([0, 2], enables.worlds().tolist())


if __name__ == "__main__":
    unittest.main()
