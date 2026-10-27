# SPDX-License-Identifier: Apache-2.0
"""The step's one read-back, and the finite check that stays on the device.

``launch_finite_check`` latches one sticky flag per world on the first
non-finite coordinate; ``Readback`` copies every observable array into host
buffers once a step and hands out one :class:`HostSnapshot` however many readers
ask. What matters is which world a NaN is charged to, that the flag outlives a
value the next solve squashed, and that a snapshot keeps the numbers of the step
it was taken at.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from robosimstudio.core.readback import HostSnapshot, Readback, launch_finite_check

TIER = "fast"
"""No GPU: Warp's CPU device runs the flag kernels, and the model is two bodies and two particles."""


def _state(**arrays):
    """A state of only the arrays given, as CPU Warp arrays; the rest ``None``, as a rigid-free scene has them."""
    kinds = {"joint_q": wp.float32, "joint_qd": wp.float32, "particle_q": wp.vec3, "body_q": wp.transform}
    return SimpleNamespace(
        **{
            name: None
            if name not in arrays
            else wp.array(np.asarray(arrays[name], np.float32), dtype=dtype, device="cpu")
            for name, dtype in kinds.items()
        }
    )


def _flags(worlds: int) -> wp.array:
    return wp.zeros(worlds, dtype=wp.int32, device="cpu")


def _model(worlds: int = 2):
    """One free body with a box and one particle per world: 7 coordinates, 6 dofs, 1 particle a world."""
    import newton

    builder = newton.ModelBuilder()
    for world in range(worlds):
        body = builder.add_body(xform=wp.transform((0.0, 0.0, 1.0 + world), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        builder.add_particle((0.0, 0.0, 2.0 + world), (0.0, 0.0, 0.0), 0.1)
    return builder.finalize(device="cpu")


class TestTheFiniteCheckChargesTheRightWorld(unittest.TestCase):
    def test_a_nan_in_world_one_flags_world_one_alone(self) -> None:
        joint_q = np.zeros(3 * 4)
        joint_q[1 * 4 + 2] = np.nan
        flags = _flags(3)
        launch_finite_check(_state(joint_q=joint_q), flags, coords_per_world=4)
        np.testing.assert_array_equal([0, 1, 0], flags.numpy())

    def test_each_array_is_strided_by_its_own_count(self) -> None:
        """A particle in world 2 and a body in world 0, with strides that differ from the coordinates'."""
        particles = np.zeros((3 * 5, 3))
        particles[2 * 5 + 4, 1] = np.inf
        bodies = np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (3 * 2, 1))
        bodies[1, 0] = np.nan
        flags = _flags(3)
        state = _state(joint_q=np.zeros(3 * 4), particle_q=particles, body_q=bodies)
        launch_finite_check(state, flags, coords_per_world=4, particles_per_world=5, bodies_per_world=2)
        np.testing.assert_array_equal([1, 0, 1], flags.numpy())

    def test_a_flag_outlives_the_value_that_raised_it_until_reset(self) -> None:
        """A NaN the next solve squashes has still lost its world the episode."""
        readback = Readback(_model(), SimpleNamespace(joint_target_q=None), num_worlds=2, coords_per_world=7)
        state = _state(joint_q=[np.nan] + [0.0] * 13)
        launch_finite_check(state, readback.flags, coords_per_world=7)
        state.joint_q.assign(np.zeros(14, np.float32))
        launch_finite_check(state, readback.flags, coords_per_world=7)
        np.testing.assert_array_equal([1, 0], readback.flags.numpy(), "sticky across a repaired value")
        readback.reset_flags()
        launch_finite_check(state, readback.flags, coords_per_world=7)
        np.testing.assert_array_equal([0, 0], readback.flags.numpy())

    def test_an_absurd_magnitude_counts_as_diverged(self) -> None:
        flags = _flags(2)
        launch_finite_check(_state(joint_q=[1.0e17, 0.0, -2.0e18, 0.0]), flags, coords_per_world=2)
        np.testing.assert_array_equal([0, 1], flags.numpy(), "1e17 is a number; past 1e18 is a solver that blew up")

    def test_no_stride_reads_the_whole_array_as_one_world(self) -> None:
        flags = _flags(3)
        launch_finite_check(_state(joint_q=[0.0, 0.0, 0.0, np.nan]), flags)
        np.testing.assert_array_equal([1, 0, 0], flags.numpy())

    def test_an_index_past_the_last_world_is_charged_to_the_last(self) -> None:
        """Never written out of bounds: a longer array than the strides say lands its excess in the last world."""
        flags = _flags(2)
        launch_finite_check(_state(joint_q=[0.0] * 6 + [np.nan]), flags, coords_per_world=3)
        np.testing.assert_array_equal([0, 1], flags.numpy())


class TestASnapshotAnswersFinite(unittest.TestCase):
    def _snapshot(self, finite) -> HostSnapshot:
        empty = np.zeros(0)
        return HostSnapshot(0, empty, empty, empty, empty, empty, empty, empty, np.asarray(finite, np.int32))

    def test_no_world_asked_is_every_world(self) -> None:
        self.assertTrue(self._snapshot([0, 0, 0]).is_finite())
        self.assertFalse(self._snapshot([0, 1, 0]).is_finite())

    def test_one_world_asked_is_that_world(self) -> None:
        snapshot = self._snapshot([0, 1, 0])
        self.assertEqual([True, False, True], [snapshot.is_finite(world) for world in range(3)])

    def test_no_flags_at_all_is_finite(self) -> None:
        self.assertTrue(self._snapshot([]).is_finite())
        self.assertTrue(self._snapshot([]).is_finite(0))


class TestOneBatchAStep(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _model()
        self.state, self.control = self.model.state(), self.model.control()
        self.readback = Readback(
            self.model, self.control, num_worlds=2, coords_per_world=7, dofs_per_world=6, particles_per_world=1
        )

    def test_the_snapshot_is_the_states_arrays_on_the_host_in_their_shapes(self) -> None:
        snapshot = self.readback.take(self.state, 5)
        self.assertEqual(5, snapshot.step)
        self.assertEqual(((2, 7), (2, 6), (14,), (12,), (14,), (2, 3)), tuple(
            getattr(snapshot, name).shape
            for name in ("body_q", "body_qd", "joint_q", "joint_qd", "joint_target_q", "particle_q")
        ))  # fmt: skip
        np.testing.assert_allclose(self.state.body_q.numpy(), snapshot.body_q, atol=1e-7)
        np.testing.assert_allclose(self.state.particle_q.numpy(), snapshot.particle_q, atol=1e-7)
        self.assertEqual(np.float64, snapshot.body_q.dtype)

    def test_two_readers_of_one_step_pay_one_wait(self) -> None:
        self.readback.enqueue(self.state, 1)
        first, second = self.readback.snapshot(), self.readback.snapshot()
        self.assertIs(first, second)
        self.assertEqual((1, 1), (self.readback.reads, self.readback.syncs))

    def test_a_snapshot_keeps_its_steps_numbers_after_the_next_batch(self) -> None:
        before = self.readback.take(self.state, 1)
        moved = self.state.body_q.numpy()
        moved[:, 2] += 0.5
        self.state.body_q.assign(moved)
        after = self.readback.take(self.state, 2)
        self.assertNotAlmostEqual(float(before.body_q[0, 2]), float(after.body_q[0, 2]))
        np.testing.assert_allclose(moved[:, 2] - 0.5, before.body_q[:, 2], atol=1e-6)

    def test_a_watched_array_rides_in_the_batch(self) -> None:
        source = wp.array(np.array([3.0, 4.0], np.float32), dtype=wp.float32, device="cpu")
        self.readback.watch("speeds", source)
        np.testing.assert_array_equal([3.0, 4.0], self.readback.take(self.state, 1).extra["speeds"])

    def test_a_diverged_world_reaches_the_snapshot_through_the_batch(self) -> None:
        joint_q = self.state.joint_q.numpy()
        joint_q[7 + 2] = np.nan
        self.state.joint_q.assign(joint_q)
        snapshot = self.readback.take(self.state, 1)
        self.assertEqual((True, False), (snapshot.is_finite(0), snapshot.is_finite(1)))


if __name__ == "__main__":
    unittest.main()
