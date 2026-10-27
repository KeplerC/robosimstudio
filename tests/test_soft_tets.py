# SPDX-License-Identifier: Apache-2.0
"""``soft.read_tets``: a tetrahedral mesh off disk, or a refusal that names what is wrong with the file.

A soft body built from a tet mesh reads it here, and a mesh that is not a volume
-- keys missing, too few vertices, indices past the end, a surface where a volume
was meant -- has to fail at the read with the file and the reason, not later as
a body with no inside.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robosimstudio_materials.soft import read_tets

TIER = "fast"
"""Files in a scratch directory: no GPU."""

HAS_MESHIO = importlib.util.find_spec("meshio") is not None

VERTICES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
TETS = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)


class TestReadTets(unittest.TestCase):
    def setUp(self) -> None:
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.directory = Path(scratch.name)

    def _npz(self, **arrays) -> Path:
        path = self.directory / "body.npz"
        np.savez(path, **arrays)
        return path

    def test_an_npz_round_trips_as_float64_vertices_and_int64_tets(self) -> None:
        vertices, tets = read_tets(self._npz(vertices=VERTICES.astype(np.float32), tets=TETS))
        np.testing.assert_array_equal(VERTICES, vertices)
        np.testing.assert_array_equal(TETS, tets)
        self.assertEqual((np.float64, np.int64), (vertices.dtype, tets.dtype))

    def test_an_npz_missing_a_key_is_refused_naming_it(self) -> None:
        with self.assertRaisesRegex(ValueError, r"missing \['tets'\]"):
            read_tets(self._npz(vertices=VERTICES))

    def test_too_little_to_be_a_volume_is_refused(self) -> None:
        for what, arrays in (
            ("three vertices", {"vertices": VERTICES[:3], "tets": np.array([[0, 1, 2, 2]])}),
            ("no tets", {"vertices": VERTICES, "tets": np.zeros((0, 4), dtype=np.int64)}),
        ):
            with self.subTest(what), self.assertRaisesRegex(ValueError, "is not a volume"):
                read_tets(self._npz(**arrays))

    def test_an_index_past_the_vertices_is_refused_with_its_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "tet indices run 0..7 over 5 vertices"):
            read_tets(self._npz(vertices=VERTICES, tets=np.array([[0, 1, 2, 7]])))

    @unittest.skipIf(HAS_MESHIO, "meshio is installed here, so a .msh is read rather than refused")
    def test_another_format_without_meshio_names_the_package_to_add(self) -> None:
        path = self.directory / "body.msh"
        path.write_text("")
        with self.assertRaisesRegex(ImportError, "needs meshio"):
            read_tets(path)

    @unittest.skipUnless(HAS_MESHIO, "meshio is not installed here")
    def test_a_mesh_with_no_tetra_cells_is_refused_by_what_it_holds(self) -> None:
        import meshio

        path = self.directory / "surface.vtk"
        meshio.write_points_cells(path, VERTICES, [("triangle", np.array([[0, 1, 2]]))])
        with self.assertRaisesRegex(ValueError, r"no 'tetra' cells \(it holds \['triangle'\]\)"):
            read_tets(path)


if __name__ == "__main__":
    unittest.main()
