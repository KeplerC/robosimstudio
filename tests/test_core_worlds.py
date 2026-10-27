# SPDX-License-Identifier: Apache-2.0
"""``WorldMap``'s other direction: one template index spread over every world, and rows stacked into one vector.

``tests.test_fleet`` holds the per-index arithmetic and the views. This holds
what a world-aware kernel is handed -- :meth:`WorldMap.spread` and its
per-kind wrappers -- and :meth:`WorldMap.stack_coords`, a wave reset's one
assign. The sentinel is the claim worth most: ``-1`` means "no such thing"
throughout this stack, and offsetting it would turn it into somebody's body.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.core.worlds import WorldMap

TIER = "fast"
"""No GPU: integer arithmetic on numpy arrays."""

MAP = WorldMap(
    num_worlds=3,
    bodies_per_world=4,
    shapes_per_world=9,
    joints_per_world=2,
    coords_per_world=5,
    dofs_per_world=6,
    particles_per_world=11,
)
"""Every stride different, so a wrapper reading its neighbour's is caught."""


class TestSpread(unittest.TestCase):
    def test_world_w_gets_the_index_shifted_by_w_strides(self) -> None:
        got = MAP.spread([0, 2, 3], per_world=4)
        np.testing.assert_array_equal([[0, 2, 3], [4, 6, 7], [8, 10, 11]], got)
        self.assertEqual(np.int32, got.dtype, "a Warp index array is int32")

    def test_the_no_such_thing_sentinel_passes_through_every_world(self) -> None:
        got = MAP.spread([-1, 1, -1], per_world=4)
        np.testing.assert_array_equal([-1, -1, -1], got[:, 0])
        np.testing.assert_array_equal([-1, -1, -1], got[:, 2])
        np.testing.assert_array_equal([1, 5, 9], got[:, 1])

    def test_one_index_is_a_column(self) -> None:
        np.testing.assert_array_equal([[3], [7], [11]], MAP.spread(3, per_world=4))
        self.assertEqual((3, 0), MAP.spread([], per_world=4).shape)

    def test_each_kind_spreads_by_its_own_stride(self) -> None:
        """``spread_<kind>(i)[w]`` is the same index the scalar ``<kind>(i, w)`` gives, for every kind and world."""
        kinds = {
            "bodies": MAP.body,
            "shapes": MAP.shape,
            "joints": MAP.joint,
            "coords": MAP.coord,
            "dofs": MAP.dof,
            "particles": MAP.particle,
        }
        for kind, scalar in kinds.items():
            spread = getattr(MAP, f"spread_{kind}")([1, 2])
            for world in range(MAP.num_worlds):
                with self.subTest(kind=kind, world=world):
                    self.assertEqual([scalar(1, world), scalar(2, world)], spread[world].tolist())


class TestStackCoords(unittest.TestCase):
    def test_each_worlds_row_reads_back_through_its_own_view(self) -> None:
        rows = [np.arange(5) + 10.0 * world for world in range(3)]
        flat = MAP.stack_coords(rows)
        self.assertEqual((15,), flat.shape)
        for world, row in enumerate(rows):
            np.testing.assert_array_equal(row, MAP.coord_view(flat, world))

    def test_a_row_of_the_wrong_width_is_refused_by_its_world(self) -> None:
        rows = [np.zeros(5), np.zeros(4), np.zeros(5)]
        with self.assertRaisesRegex(ValueError, r"world 1: expected 5 coordinates, got 4"):
            MAP.stack_coords(rows)

    def test_a_rows_shape_does_not_matter_only_its_count(self) -> None:
        """A ``[1, 5]`` row from a batched read is the same five coordinates as a flat one."""
        rows = [np.full((1, 5), world) for world in range(3)]
        np.testing.assert_array_equal(np.repeat([0.0, 1.0, 2.0], 5), MAP.stack_coords(rows))


class TestOf(unittest.TestCase):
    def _counts(self, worlds: int = 1, **overrides: int) -> SimpleNamespace:
        per = {
            "body_count": 4,
            "shape_count": 9,
            "joint_count": 2,
            "joint_coord_count": 5,
            "joint_dof_count": 6,
            "particle_count": 11,
        }
        return SimpleNamespace(**{name: n * worlds for name, n in per.items()}, world_count=worlds, **overrides)

    def test_the_strides_are_the_templates_counts(self) -> None:
        self.assertEqual(MAP, WorldMap.of(self._counts(), self._counts(3), 3))

    def test_a_particle_added_outside_the_worlds_is_refused_by_count(self) -> None:
        model = self._counts(3)
        model.particle_count += 1
        with self.assertRaisesRegex(ValueError, "particle_count: model has 34, 3 worlds need 33"):
            WorldMap.of(self._counts(), model, 3)


if __name__ == "__main__":
    unittest.main()
