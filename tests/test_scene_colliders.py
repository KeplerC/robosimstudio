# SPDX-License-Identifier: Apache-2.0
"""A collider as a stack of boxes along the object's longest axis, and the vertices it is cut from.

:func:`slab_decomposition` takes the bound per band instead of once, so a
tapered bottle is narrower where the bottle is, and it must give back the one
box a prism already was. The claims that matter are geometric: a prism is its
own AABB, a taper narrows band by band, the boxes tile the axis with no gap --
a low-poly barrel with vertices only at its ends once came back open along its
middle and a pair of jaws closed straight through it -- and every vertex lies
inside the stack. :func:`collider_vertices` places each mesh in its body's frame
through its scale and its shape transform.
"""

from __future__ import annotations

import itertools
import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.core.math import quat_from_axis_angle, quat_to_mat
from robosimstudio.scene.build.colliders import collider_vertices, slab_decomposition

TIER = "fast"
"""Numpy point clouds and a stand-in builder: nothing is finalized, so no GPU."""


def _rings(radii_at: list[tuple[float, float]], sides: int = 12) -> np.ndarray:
    """Polygon rings about z: one ``(z, radius)`` each, a low-poly solid of revolution."""
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    return np.concatenate(
        [np.c_[r * np.cos(angles), r * np.sin(angles), np.full(sides, z)] for z, r in radii_at], axis=0
    )


def _spans(boxes) -> list[tuple[float, float]]:
    return [(float(c[2] - h[2]), float(c[2] + h[2])) for c, h in boxes]


def _box_corners(half=(0.03, 0.02, 0.1), centre=(0.0, 0.0, 0.0)) -> np.ndarray:
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    return signs * np.asarray(half) + np.asarray(centre)


class TestSlabDecomposition(unittest.TestCase):
    def test_a_prism_comes_back_as_the_one_box_it_already_was(self) -> None:
        corners = _box_corners(half=(0.03, 0.02, 0.1), centre=(0.5, -0.2, 0.0))
        boxes, axis = slab_decomposition(corners)
        self.assertEqual(2, axis)
        self.assertEqual(1, len(boxes), "no regression for a can: same collider, same contact count")
        (centre, half) = boxes[0]
        np.testing.assert_allclose(centre, (0.5, -0.2, 0.0), atol=1e-12)
        np.testing.assert_allclose(half, (0.03, 0.02, 0.1), atol=1e-12)

    def test_a_taper_narrows_band_by_band_toward_its_narrow_end(self) -> None:
        cone = _rings([(z, 0.04 - 0.3 * z) for z in np.linspace(0.0, 0.1, 21)])
        boxes, axis = slab_decomposition(cone, slabs=4)
        self.assertEqual((2, 4), (axis, len(boxes)))
        widths = [float(h[0]) for _, h in boxes]
        self.assertEqual(widths, sorted(widths, reverse=True))
        self.assertGreater(widths[0] - widths[-1], 0.02, "a cone is not a column")

    def test_the_slabs_tile_the_axis_with_no_gap_even_where_no_vertex_sits(self) -> None:
        """A barrel with vertices only at its two end rings, of different radii, and nothing between."""
        barrel = _rings([(0.0, 0.010), (0.12, 0.006)])
        boxes, _ = slab_decomposition(barrel, slabs=8)
        spans = _spans(boxes)
        self.assertAlmostEqual(0.0, spans[0][0], places=12)
        self.assertAlmostEqual(0.12, spans[-1][1], places=12)
        for (_, top), (bottom, _) in itertools.pairwise(spans):
            self.assertAlmostEqual(top, bottom, places=12, msg="a gap the jaws could close through")

    def test_an_empty_band_takes_the_cross_section_below_it(self) -> None:
        barrel = _rings([(0.0, 0.010), (0.12, 0.006)])
        boxes, _ = slab_decomposition(barrel, slabs=8)
        self.assertEqual(2, len(boxes), "the empty middle merges into the wide end, the top band stays narrow")
        (_, below), (_, above) = boxes
        self.assertAlmostEqual(0.010, float(below[0]), places=12)
        self.assertAlmostEqual(0.006, float(above[0]), places=6)
        self.assertAlmostEqual(0.12 * 7 / 8, _spans(boxes)[0][1], places=12, msg="seven of eight bands are the base")

    def test_every_vertex_lies_inside_the_stack(self) -> None:
        bottle = _rings([(0.0, 0.03), (0.05, 0.03), (0.08, 0.02), (0.10, 0.01), (0.12, 0.01)])
        boxes, _ = slab_decomposition(bottle)
        inside = np.zeros(len(bottle), dtype=bool)
        for centre, half in boxes:
            inside |= np.all(np.abs(bottle - centre) <= half + 1e-12, axis=1)
        self.assertTrue(inside.all(), f"{int((~inside).sum())} vertices outside every box")

    def test_the_longest_extent_is_the_axis_that_is_cut(self) -> None:
        lying = _box_corners(half=(0.02, 0.15, 0.03))
        self.assertEqual(1, slab_decomposition(lying)[1])

    def test_merge_tolerance_decides_when_two_bands_are_one_box(self) -> None:
        """Two halves whose cross-sections differ by 1 mm: one box at the default 1.5 mm, two at 0.5 mm."""
        step = np.concatenate(
            [_box_corners((0.020, 0.02, 0.05), (0, 0, -0.05)), _box_corners((0.021, 0.02, 0.05), (0, 0, 0.05))]
        )
        self.assertEqual(1, len(slab_decomposition(step, slabs=2)[0]))
        self.assertEqual(2, len(slab_decomposition(step, slabs=2, merge_tol=0.0005)[0]))

    def test_a_band_that_bulges_on_one_side_only_is_a_box_of_its_own(self) -> None:
        """Both edges of a cross-section have to agree: a flange on +x alone is not a prism."""
        lower = _box_corners((0.02, 0.02, 0.05), (0, 0, -0.05))
        flange = _box_corners((0.02, 0.02, 0.05), (0, 0, 0.05)) + np.where(_box_corners()[:, :1] > 0, [0.01, 0, 0], 0)
        boxes, _ = slab_decomposition(np.concatenate([lower, flange]), slabs=2)
        self.assertEqual(2, len(boxes))
        self.assertAlmostEqual(0.005, float(boxes[1][0][0]), places=12, msg="the flange's box sits off-centre")

    def test_fewer_than_four_vertices_is_no_decomposition(self) -> None:
        self.assertEqual(([], 0), slab_decomposition(np.zeros((3, 3))))
        self.assertEqual(([], 0), slab_decomposition(np.zeros(12)), "a flat array is not a point cloud")


class TestColliderVertices(unittest.TestCase):
    def test_each_mesh_is_scaled_then_placed_by_its_shape_transform(self) -> None:
        quat = quat_from_axis_angle((0.0, 0.0, 1.0), np.pi / 2)
        mesh = SimpleNamespace(vertices=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.5]]))
        builder = SimpleNamespace(
            shape_source=[mesh, None, mesh],
            shape_scale=[(2.0, 3.0, 4.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
            shape_transform=[(0.1, 0.2, 0.3, *quat), (0.0,) * 6 + (1.0,), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
        )
        got = collider_vertices(builder, [0, 1, 2])
        scaled = mesh.vertices * (2.0, 3.0, 4.0)
        expected = np.vstack([(0.1, 0.2, 0.3) + scaled @ quat_to_mat(quat).T, mesh.vertices])
        np.testing.assert_allclose(got, expected, atol=1e-12)
        np.testing.assert_allclose(got[0], (0.1, 0.2 + 2.0, 0.3), atol=1e-12, err_msg="+x turned onto +y")

    def test_shapes_with_no_mesh_give_no_vertices(self) -> None:
        builder = SimpleNamespace(
            shape_source=[None, SimpleNamespace(vertices=np.zeros((0, 3)))],
            shape_scale=[(1.0, 1.0, 1.0)] * 2,
            shape_transform=[(0.0,) * 6 + (1.0,)] * 2,
        )
        self.assertEqual((0, 3), collider_vertices(builder, [0, 1]).shape)


if __name__ == "__main__":
    unittest.main()
