# SPDX-License-Identifier: Apache-2.0
"""``core.shell``: giving a surface a thickness, and reading the surface back out of the slab."""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio.core.shell import grid_surface, mid_surface, prism_slab

TIER = "fast"
"""No GPU: numpy meshes."""


class TestPrismSlab(unittest.TestCase):
    """Giving a surface a thickness, which is the whole of a bath towel.

    The failures this guards against are all silent ones. A slab whose tets do
    not conform simulates a crack; a slab with an inverted tet is rejected at
    build with a message about volume that says nothing about thickness; a slab
    whose mass is lumped wrong weighs the wrong amount and drapes accordingly.
    None of them look like a thickness bug when they surface.
    """

    def _slab(self, thickness=0.005, cells=(6, 5)):
        vertices, faces = grid_surface((0.06, 0.05), cells)
        return vertices, faces, *prism_slab(vertices, faces, thickness)

    def test_volume_is_area_times_thickness(self):
        """A flat sheet's slab holds exactly the volume it claims to."""
        *_, slab, tets = self._slab()
        edges = slab[tets[:, 1:]] - slab[tets[:, [0]]]
        volume = float((np.linalg.det(edges) / 6.0).sum())
        self.assertAlmostEqual(volume, 0.06 * 0.05 * 0.005, places=12)

    def test_every_tet_is_positively_oriented(self):
        """``add_tetrahedron`` rejects a non-positive volume, so none may exist.

        The input surface's winding is not something an arbitrary mesh can be
        trusted about, so this holds for a mesh wound the wrong way too.
        """
        for flip in (False, True):
            vertices, faces = grid_surface((0.06, 0.05), (6, 5))
            if flip:
                faces = faces[:, ::-1]
            slab, tets = prism_slab(vertices, faces, 0.005)
            edges = slab[tets[:, 1:]] - slab[tets[:, [0]]]
            self.assertGreater(float((np.linalg.det(edges) / 6.0).min()), 0.0, f"flipped={flip}")

    def test_tets_conform(self):
        """Neighbouring prisms must agree about the wall they share.

        Split two adjacent prisms on opposite diagonals and their tets meet
        along a crack instead of a face. Nothing raises: VBD simulates the
        cracked mesh happily and the slab quietly comes apart under load. The
        signature is an interior face belonging to one tet instead of two.
        """
        _, faces, _, tets = self._slab()
        quads = np.sort(
            np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), axis=1
        )
        _, counts = np.unique(quads, axis=0, return_counts=True)
        self.assertEqual(int(counts.max()), 2, "a face shared by more than two tets is non-manifold")
        # The boundary is the two faces plus a rim of two triangles per edge.
        boundary_edges = 2 * (6 + 5)
        self.assertEqual(int((counts == 1).sum()), 2 * len(faces) + 2 * boundary_edges)

    def test_thickness_is_the_thickness(self):
        """The surface ends up in the middle, not on one side.

        Offsetting to one side instead would bury a sheet laid on a table half
        its own thickness into the table, and every pose stated for the cloth
        would be off by that much.
        """
        vertices, _, slab, _ = self._slab(thickness=0.004)
        n = len(vertices)
        np.testing.assert_allclose(slab[n:, 2] - slab[:n, 2], 0.004, atol=1e-15)
        np.testing.assert_allclose(mid_surface(slab, n), vertices, atol=1e-15)

    def test_rejects_a_thickness_that_does_not_fit(self):
        """A surface folded tighter than its own half-thickness has no slab."""
        with self.assertRaises(ValueError):
            prism_slab(*grid_surface((0.06, 0.05), (6, 5))[:2], 0.0)

    def test_mid_surface_is_unbiased_when_the_sheet_turns_over(self):
        """The midpoint survives a fold; either face alone does not.

        Score a fold against one face and the flip enters as a full thickness
        of error -- 5 mm on a bath towel, which is the size of the tolerance a
        fold is judged against.
        """
        vertices, faces = grid_surface((0.06, 0.05), (6, 5))
        slab, _ = prism_slab(vertices, faces, 0.005)
        n = len(vertices)
        # Turn the sheet upside down about z: the faces swap places.
        flipped = slab.copy()
        flipped[:, 2] *= -1.0
        np.testing.assert_allclose(mid_surface(flipped, n)[:, 2], 0.0, atol=1e-15)
        # The bottom face alone reads half a thickness on the wrong side.
        self.assertAlmostEqual(float(flipped[:n, 2].mean()), 0.0025, places=12)


if __name__ == "__main__":
    unittest.main()
