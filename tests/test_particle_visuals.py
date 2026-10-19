# SPDX-License-Identifier: Apache-2.0
"""The two rules a renderer reads a particle body's handle by, without a GPU.

:mod:`robosimstudio.core.render.particles` is what the studio page and the path
tracer share: which handles are points (:func:`draws_points`), and what a
handle's ``surface`` callable is allowed to return (:func:`surface_mesh`). Held
here on stub handles: the three-way split between a sheet, a centerline and a
particle body; every return shape the contract admits; and that the studio's
:class:`~robosimstudio_studio.garments.GarmentVisual` resolves a particle body
as ``is_points`` and keeps the handle's callable apart from its own
``surface`` method.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.core.render.particles import draws_points, surface_mesh

TIER = "fast"


def _handle(*, triangles=(), tube_radius: float = 0.0, point_radius: float = 0.0, surface=None):
    return SimpleNamespace(
        name="thing",
        particles=np.arange(8, dtype=np.int32),
        triangles=np.asarray(triangles, dtype=np.int32).reshape(-1, 3),
        tube_radius=tube_radius,
        point_radius=point_radius,
        surface=surface,
    )


class TestWhichHandlesArePoints(unittest.TestCase):
    def test_a_particle_body(self) -> None:
        self.assertTrue(draws_points(_handle(point_radius=0.0015)))

    def test_a_sheet_keeps_its_triangles(self) -> None:
        self.assertFalse(draws_points(_handle(triangles=[[0, 1, 2]], point_radius=0.0015)))

    def test_a_centerline_keeps_its_tube(self) -> None:
        self.assertFalse(draws_points(_handle(tube_radius=0.006, point_radius=0.0015)))

    def test_nothing_without_a_radius(self) -> None:
        self.assertFalse(draws_points(_handle()))
        self.assertFalse(draws_points(SimpleNamespace(particles=np.arange(3))), "a handle from before the field")


class TestWhatASurfaceMayReturn(unittest.TestCase):
    positions = np.zeros((4, 3), dtype=np.float32)
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

    def test_two_arrays_with_flat_triangles(self) -> None:
        v, t, n = surface_mesh(lambda p: (self.verts, [0, 1, 2]), self.positions)
        self.assertEqual((v.shape, v.dtype, t.shape, t.dtype, n), ((3, 3), np.float32, (1, 3), np.int32, None))

    def test_three_arrays_with_shaped_triangles(self) -> None:
        _, t, n = surface_mesh(lambda p: (self.verts, [[0, 1, 2]], np.ones((3, 3))), self.positions)
        self.assertEqual((n.shape, n.dtype), ((3, 3), np.float32))
        np.testing.assert_array_equal(t, [[0, 1, 2]])

    def test_arrays_with_a_numpy_method_are_read_through_it(self) -> None:
        class _Device:
            def __init__(self, a):
                self.a = a

            def numpy(self):
                return self.a

        v, t, n = surface_mesh(lambda p: (_Device(self.verts), _Device(np.array([0, 1, 2])), None), self.positions)
        self.assertEqual((len(v), len(t)), (3, 1))
        self.assertIsNone(n)

    def test_nothing_to_draw(self) -> None:
        for result in (None, (None, None), (None, None, None), (self.verts, np.zeros((0, 3))), (np.zeros((0, 3)), [])):
            v, t, n = surface_mesh(lambda p, r=result: r, self.positions)
            self.assertEqual((len(v), len(t), n), (0, 0, None), str(result))

    def test_mismatched_normals_are_dropped_not_drawn(self) -> None:
        _, _, n = surface_mesh(lambda p: (self.verts, [0, 1, 2], np.ones((2, 3))), self.positions)
        self.assertIsNone(n)

    def test_the_callable_sees_float32_positions(self) -> None:
        seen = {}

        def surface(p):
            seen["dtype"] = p.dtype
            return None

        surface_mesh(surface, np.zeros((2, 3), dtype=np.float64))
        self.assertEqual(seen["dtype"], np.float32)


class TestTheStudioResolvesAParticleBody(unittest.TestCase):
    def _visual(self, **kwargs):
        from robosimstudio_studio.garments import GarmentVisual

        return GarmentVisual(name="heap", particles=np.arange(8), triangles=np.zeros((0, 3)), **kwargs)

    def test_points_and_nothing_fluffy(self) -> None:
        visual = self._visual(point_radius=0.0015, thickness=0.005)
        self.assertTrue(visual.is_points)
        self.assertFalse(visual.is_tube)
        self.assertFalse(visual.has_body, "no shell was grown for a heap")
        self.assertIsNone(visual.band_drawn)

    def test_the_handles_callable_is_kept_apart_from_the_sheet_method(self) -> None:
        """``GarmentVisual.surface`` is a method; the handle's callable rides as ``surfacer``."""
        surfacer = lambda p: None  # noqa: E731
        visual = self._visual(point_radius=0.0015, surfacer=surfacer)
        self.assertIs(visual.surfacer, surfacer)
        self.assertTrue(callable(visual.surface) and visual.surface is not surfacer)

    def test_a_centerline_is_still_a_tube(self) -> None:
        visual = self._visual(tube_radius=0.006, point_radius=0.0015)
        self.assertTrue(visual.is_tube)
        self.assertFalse(visual.is_points)

    def test_garment_visuals_reads_both_fields(self) -> None:
        from robosimstudio_studio.garments import garment_visuals

        surfacer = lambda p: None  # noqa: E731
        handle = _handle(point_radius=0.002, surface=surfacer)
        scene = SimpleNamespace(
            handles=SimpleNamespace(deformables={"thing": handle}, deformable_finishes={}),
            by_name={"thing": SimpleNamespace(spec=SimpleNamespace(color=(0.9, 0.1, 0.1)))},
        )
        (visual,) = garment_visuals(scene)
        self.assertEqual((visual.is_points, visual.point_radius, visual.surfacer), (True, 0.002, surfacer))
        self.assertEqual(visual.color, (0.9, 0.1, 0.1))


class TestOneSurfacerForEveryParticleKind(unittest.TestCase):
    """The MPM kinds' surface is a :class:`ParticleSurfacer` at the solver's resolution, with the one margin rule."""

    def test_the_default_margin_is_twice_the_kernel_the_padding_and_a_tile(self) -> None:
        from robosimstudio.core.render.particles import ParticleSurfacer

        surfacer = ParticleSurfacer(0.001)
        self.assertEqual((0.001, 0.006), (surfacer.voxel_size, surfacer.options["kernel_radius"]))
        self.assertAlmostEqual(0.022, surfacer.margin)

    def test_an_mpm_body_surfaces_through_it_at_the_solvers_voxel(self) -> None:
        """0.45 of a 4 mm voxel, a kernel of three of those, and sixteen of them as the margin, as before."""
        import robosimstudio as rss
        from robosimstudio.core.render.particles import ParticleSurfacer
        from robosimstudio_materials.mpm.entity import EntityParticleBody

        entity = EntityParticleBody(rss.dough(name="d", render="surface"))
        entity.solver = SimpleNamespace(
            config=SimpleNamespace(voxel_size=0.004), model=SimpleNamespace(device="cpu", world_count=1)
        )
        entity._install_surface()
        surfacer = entity.surface
        self.assertIsInstance(surfacer, ParticleSurfacer)
        self.assertAlmostEqual(0.00125, surfacer.radius)
        self.assertAlmostEqual(0.0018, surfacer.voxel_size)
        self.assertAlmostEqual(0.0054, surfacer.options["kernel_radius"])
        self.assertAlmostEqual(16 * 0.0018, surfacer.margin)


class TestASurfaceNodeKeepsItsFaces(unittest.TestCase):
    """The page's surface node, over a scene that records what viser would have been sent."""

    def _viewer(self):
        from robosimstudio_studio.garments import ViewerStudio

        added = []

        def add_mesh_simple(name, *, vertices, faces, **kwargs):
            handle = SimpleNamespace(vertices=vertices, faces=faces, visible=kwargs.get("visible", True))
            added.append(handle)
            return handle

        viewer = ViewerStudio.__new__(ViewerStudio)
        viewer._qualify = lambda name: name
        viewer._garment_nodes, viewer._garment_hidden = {}, {}
        viewer._server = SimpleNamespace(scene=SimpleNamespace(add_mesh_simple=add_mesh_simple))
        return viewer, added

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)

    def test_new_faces_over_as_many_vertices_replace_the_node(self) -> None:
        """Two triangles over four vertices either way, split along the other diagonal: the counts match, the faces do not."""
        viewer, added = self._viewer()
        viewer._draw_surface_node("/s", self.verts, np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32), (9, 9, 9), False)
        flipped = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int32)
        viewer._draw_surface_node("/s", self.verts + 0.1, flipped, (9, 9, 9), False)
        self.assertEqual(2, len(added), "the regrown faces were not sent")
        np.testing.assert_array_equal(flipped, viewer._garment_nodes["/s"].faces)
        np.testing.assert_allclose(self.verts + 0.1, viewer._garment_nodes["/s"].vertices)

    def test_the_same_faces_move_the_vertices_in_place(self) -> None:
        viewer, added = self._viewer()
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        viewer._draw_surface_node("/s", self.verts, faces, (9, 9, 9), False)
        viewer._draw_surface_node("/s", self.verts + 0.1, faces.copy(), (9, 9, 9), False)
        self.assertEqual(1, len(added), "an unchanged mesh was re-added rather than moved")
        np.testing.assert_allclose(self.verts + 0.1, added[0].vertices)


if __name__ == "__main__":
    unittest.main()
