# SPDX-License-Identifier: Apache-2.0
"""How the studio draws a garment, and what it costs a frame.

The per-frame costs: no GLB kept for a deforming garment, a pushed static batch
skipped. And the arithmetic the drawn cloth is grown from, held to answers
that can be written down: a flat sheet's normals are its face normal (the
other way for the other winding), the weave's peak is its fabric's amplitude,
the pile stands between the shell's face and the tip of the nap and moves with
the cloth, a towel's woven bands are bald, and the crease a garment is banded
across is the one its asset declares.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

from robosimstudio.core.render.finish import GLOSS
from robosimstudio_studio import finish_gltf
from robosimstudio_studio.garments import (
    FUZZ_CAP,
    FUZZ_PER_M2,
    WEAVE_MAX,
    WEAVE_MIN,
    WEAVE_RATIO,
    GarmentVisual,
    ViewerStudio,
    _declared_crease_axis,
    _display_normals,
)

TIER = "fast"
"""No GPU, no browser: a viewer built without its constructor."""


class TestDeformingGlbIsNotKept(unittest.TestCase):
    def test_an_uncached_encode_leaves_the_cache_as_it_was(self) -> None:
        finish_gltf.clear_cache()
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        for frame in range(3):
            moved = points + 0.01 * frame
            blob = finish_gltf.glb_for(moved, faces, (0.5, 0.5, 0.5), GLOSS, cache=False)
            if blob is None:
                self.skipTest("no trimesh: nothing is encoded")
        self.assertEqual(len(finish_gltf._CACHE), 0)


class TestStaticPbrBatch(unittest.TestCase):
    def _viewer(self) -> ViewerStudio:
        viewer = ViewerStudio.__new__(ViewerStudio)
        viewer._qualify = lambda name: name
        viewer._static_skip_ok = True
        viewer._static_batches = {"/b"}
        viewer._scene_handles = {}
        viewer._pbr_handles = {"/b": {"/b/m0": object()}}
        viewer._finishes = {0: GLOSS}
        viewer._host_shape_color = None
        return viewer

    def test_a_pushed_pbr_batch_is_skipped(self) -> None:
        viewer = self._viewer()
        drawn = []
        viewer._log_pbr_instances = lambda *a: drawn.append(a) or True
        viewer.log_instances("/b", "/mesh", None, None, None, None)
        self.assertEqual(drawn, [])

    def test_colours_are_read_back_once_per_repack(self) -> None:
        viewer = self._viewer()
        reads = []

        class _Colors:
            def numpy(self):
                reads.append(1)
                return np.ones((2, 3), dtype=np.float32)

        viewer.model = types.SimpleNamespace(shape_color=_Colors())
        viewer._meshes = {"/mesh": object()}
        viewer._plane_meshes = set()
        viewer._live_shapes = frozenset()
        viewer._batch_shapes = lambda q: [0, 1]
        viewer._pbr_groups = {}
        viewer._glb = lambda data, color, finish: None  # stop after the colours
        viewer._retire_pbr_nodes = lambda *a: None
        xforms = np.zeros((2, 7), dtype=np.float32)
        xforms[:, 6] = 1.0
        for _ in range(3):
            viewer._log_pbr_instances("/b", "/mesh", xforms, None)
        self.assertEqual(len(reads), 1)
        viewer.mark_shape_colors_dirty()
        viewer._log_pbr_instances("/b", "/mesh", xforms, None)
        self.assertEqual(len(reads), 2)


def _grid(nx: int, ny: int, cell: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """A flat ``nx`` x ``ny``-cell sheet in the xy plane, laid out as ``grid_surface`` lays one, wound to face +z."""
    xs, ys = np.meshgrid(np.arange(nx + 1) * cell, np.arange(ny + 1) * cell)
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    faces = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b, c, d = a + 1, a + nx + 1, a + nx + 2
            faces += [(a, b, d), (a, d, c)]
    return verts, np.asarray(faces, dtype=np.int64)


def _garment(name: str = "towel", thickness: float = 0.005, nx: int = 4, ny: int = 4, **kw) -> GarmentVisual:
    verts, faces = _grid(nx, ny)
    return GarmentVisual(name=name, particles=np.arange(len(verts)), triangles=faces, thickness=thickness, **kw)


class TestTheDrawnNormals(unittest.TestCase):
    def test_a_flat_sheet_faces_the_way_it_is_wound(self) -> None:
        verts, faces = _grid(3, 2)
        np.testing.assert_allclose(_display_normals(verts, faces), np.tile([0.0, 0.0, 1.0], (len(verts), 1)))
        np.testing.assert_allclose(_display_normals(verts, faces[:, ::-1]), np.tile([0.0, 0.0, -1.0], (len(verts), 1)))

    def test_a_vertex_no_face_uses_points_up_rather_than_dividing_by_zero(self) -> None:
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 5.0, 5.0]])
        normals = _display_normals(verts, np.array([[0, 1, 2]]))
        np.testing.assert_array_equal(normals[3], [0.0, 0.0, 1.0])
        self.assertTrue(np.isfinite(normals).all())

    def test_a_corner_leans_toward_its_larger_face(self) -> None:
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.1]])
        faces = np.array([[0, 1, 2], [0, 3, 4]])  # a unit face up, and a hundredth of one facing -y
        expected = np.array([0.0, -0.01, 1.0]) / np.hypot(0.01, 1.0)
        np.testing.assert_allclose(_display_normals(verts, faces)[0], expected)

    def test_turning_the_cloth_turns_its_normals(self) -> None:
        verts, faces = _grid(3, 3)
        verts[:, 2] = 0.02 * np.sin(7.0 * verts[:, 0]) * np.cos(5.0 * verts[:, 1])  # a rumpled sheet
        c, s = np.cos(0.7), np.sin(0.7)
        turn = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        np.testing.assert_allclose(_display_normals(verts @ turn.T, faces), _display_normals(verts, faces) @ turn.T)


class TestTheWeave(unittest.TestCase):
    def test_a_film_has_none(self) -> None:
        np.testing.assert_array_equal(_garment(thickness=0.0).weave_offsets(), 0.0)

    def test_its_peak_is_its_fabrics_amplitude_clamped(self) -> None:
        for thickness, amplitude in ((0.005, WEAVE_RATIO * 0.005), (0.001, WEAVE_MIN), (0.02, WEAVE_MAX)):
            with self.subTest(thickness=thickness):
                weave = _garment(thickness=thickness).weave_offsets()
                self.assertAlmostEqual(amplitude, float(np.abs(weave).max()), places=12)

    def test_it_is_the_same_towel_every_reload_and_another_towel_is_another(self) -> None:
        towel = _garment("towel")
        self.assertIs(towel.weave_offsets(), towel.weave_offsets(), "computed once")
        np.testing.assert_array_equal(towel.weave_offsets(), _garment("towel").weave_offsets())
        self.assertFalse(np.allclose(towel.weave_offsets(), _garment("napkin").weave_offsets()))


class TestThePile(unittest.TestCase):
    def _fuzz(self, garment: GarmentVisual, shift=(0.0, 0.0, 0.0), light=(0.0, 0.0, 1.0)):
        verts, _ = _grid(4, 4)
        verts = verts + np.asarray(shift)
        normals = _display_normals(verts, garment.triangles)
        return garment.fuzz_arrays(verts, normals, np.asarray(light))

    def test_it_stands_between_the_shells_face_and_the_tip_of_the_nap(self) -> None:
        towel = _garment(thickness=0.004)
        points, colors, tufts = self._fuzz(towel)
        self.assertEqual(int(min(0.04 * 0.04 * FUZZ_PER_M2, FUZZ_CAP)), len(points))
        self.assertEqual((len(points),), tufts.shape)
        self.assertEqual(np.uint8, colors.dtype)
        height = np.abs(points[:, 2])
        self.assertGreaterEqual(float(height.min()), 0.002 - 1e-12, "no fibre inside the fabric")
        self.assertLessEqual(float(height.max()), 0.002 + towel.pile + 1e-12, "none above the nap")
        self.assertTrue((points[:, 2] > 0).any() and (points[:, 2] < 0).any(), "a membrane grows on both faces")

    def test_it_is_fixed_once_and_carried_by_the_cloth(self) -> None:
        towel = _garment()
        still, _, _ = self._fuzz(towel)
        moved, _, _ = self._fuzz(towel, shift=(0.3, -0.1, 0.2))
        np.testing.assert_allclose(moved - still, np.tile([0.3, -0.1, 0.2], (len(still), 1)), atol=1e-12)
        np.testing.assert_array_equal(still, self._fuzz(_garment())[0], "the same nap on every reload")

    def test_the_side_facing_the_key_is_the_brighter(self) -> None:
        towel = _garment()
        points, lit, _ = self._fuzz(towel, light=(0.0, 0.0, 1.0))
        _, unlit, _ = self._fuzz(towel, light=(0.0, 0.0, -1.0))
        up = points[:, 2] > 0
        self.assertTrue((lit[up].astype(int).sum(axis=1) >= unlit[up].astype(int).sum(axis=1)).all())
        self.assertGreater(int(lit[up].astype(int).sum()), int(unlit[up].astype(int).sum()))

    def test_a_towels_woven_bands_are_bald(self) -> None:
        towel = _garment(nx=4, ny=10, decor="towel", grid_cells=(4, 10))
        self.assertIsNotNone(towel.band_mid)
        verts, _ = _grid(4, 10)
        towel.fuzz_arrays(verts, _display_normals(verts, towel.triangles), np.array([0.0, 0.0, 1.0]))
        self.assertFalse(towel.band_mid[towel._fuzz["face"]].any(), "a tuft sprouted from a band")


class TestTheDeclaredCrease(unittest.TestCase):
    def _doc(self, name: str = "towel", asset: str = "test_towel"):
        return types.SimpleNamespace(objects=[types.SimpleNamespace(name=name, type=asset)])

    def _record(self, *panels):
        return types.SimpleNamespace(features_of_kind=lambda kind: list(panels) if kind == "panel" else [])

    def test_with_no_document_or_no_record_it_is_the_second_axis(self) -> None:
        from robosimstudio.assetlib.registry import ASSETS

        self.assertEqual(1, _declared_crease_axis(None, "towel"))
        self.assertEqual(1, _declared_crease_axis(self._doc(name="shirt"), "towel"), "a name the document lacks")
        with mock.patch.object(ASSETS, "get", side_effect=KeyError("test_towel")):
            self.assertEqual(1, _declared_crease_axis(self._doc(), "towel"), "an asset nobody registered")

    def test_a_declared_panel_says_which_axis_it_folds_along(self) -> None:
        from robosimstudio.assetlib.registry import ASSETS

        folded = self._record(types.SimpleNamespace(payload={"crease": {"axis": 0, "at": 0.15}}))
        with mock.patch.object(ASSETS, "get", return_value=folded):
            self.assertEqual(0, _declared_crease_axis(self._doc(), "towel"))
        with mock.patch.object(ASSETS, "get", return_value=self._record()):
            self.assertEqual(1, _declared_crease_axis(self._doc(), "towel"), "no panel, no crease")


if __name__ == "__main__":
    unittest.main()
