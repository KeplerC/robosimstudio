# SPDX-License-Identifier: Apache-2.0
"""The vendored cutter's polygon triangulation, as the cutting kind calls it to draw a cut face.

``triangulate_polygon_with_holes`` is upstream MIT code (``cutting/cagecut``),
reached through ``cutting/solver``; a cut face is a planar region bounded by an
outer loop and the holes a second cut left in it. Held here as invariants, not
line by line: the triangles cover exactly the region -- the outer loop's area
less its holes', plus an island's -- whatever way round each loop is given;
every triangle is counter-clockwise; and a loop with no area draws nothing.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "fast"
"""Planar geometry in numpy: no GPU."""


def _square(centre, half: float, *, clockwise: bool = False) -> np.ndarray:
    cx, cy = centre
    loop = np.array([[cx - half, cy - half], [cx + half, cy - half], [cx + half, cy + half], [cx - half, cy + half]])
    return loop[::-1] if clockwise else loop


def _areas(loops, triangles) -> np.ndarray:
    points = np.concatenate(loops)
    a, b, c = (points[triangles[:, i]] for i in range(3))
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


class TestTriangulatingACutFace(unittest.TestCase):
    def test_a_face_with_a_hole_is_covered_exactly_whichever_way_its_loops_run(self) -> None:
        from robosimstudio_materials.cutting.solver import triangulate_polygon_with_holes

        for outer_cw in (False, True):
            for hole_cw in (False, True):
                with self.subTest(outer_clockwise=outer_cw, hole_clockwise=hole_cw):
                    loops = [_square((0.0, 0.0), 1.0, clockwise=outer_cw), _square((0.2, -0.1), 0.3, clockwise=hole_cw)]
                    areas = _areas(loops, triangulate_polygon_with_holes(loops))
                    self.assertTrue(np.all(areas > 0.0), "every triangle counter-clockwise")
                    self.assertAlmostEqual(4.0 - 0.36, float(areas.sum()), places=12)

    def test_an_island_inside_a_hole_is_drawn_again(self) -> None:
        from robosimstudio_materials.cutting.solver import triangulate_polygon_with_holes

        loops = [_square((0.0, 0.0), 1.0), _square((0.0, 0.0), 0.6), _square((0.0, 0.0), 0.2)]
        self.assertAlmostEqual(4.0 - 1.44 + 0.16, float(_areas(loops, triangulate_polygon_with_holes(loops)).sum()))

    def test_two_holes_are_both_taken_out(self) -> None:
        from robosimstudio_materials.cutting.solver import triangulate_polygon_with_holes

        loops = [_square((0.0, 0.0), 1.0), _square((-0.5, 0.0), 0.2), _square((0.5, 0.3), 0.25)]
        self.assertAlmostEqual(4.0 - 0.16 - 0.25, float(_areas(loops, triangulate_polygon_with_holes(loops)).sum()))

    def test_a_loop_with_no_area_draws_nothing(self) -> None:
        from robosimstudio_materials.cutting.solver import triangulate_polygon_with_holes

        slit = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        self.assertEqual((0, 3), triangulate_polygon_with_holes([slit]).shape)
        self.assertEqual((0, 3), triangulate_polygon_with_holes([]).shape)


if __name__ == "__main__":
    unittest.main()
