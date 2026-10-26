# SPDX-License-Identifier: Apache-2.0
"""What a renderer without a particle pass of its own is told a scene's deformables look like.

``gl`` and ``blender`` draw meshes; :func:`deformable_drawables` turns the
scene's deformable handles into meshes of fixed topology and spheres, the way
the path tracer reads the same handles. Held here on handles and a model made
of plain arrays, so nothing is built.
"""

from __future__ import annotations

import types
import unittest

import numpy as np

TIER = "fast"


def _model(count: int, radius: float = 0.004, tri_indices=None):
    return types.SimpleNamespace(
        particle_count=count,
        particle_radius=np.full(count, radius),
        particle_q=np.zeros((count, 3)),
        tri_indices=np.zeros((0, 3), dtype=np.int32) if tri_indices is None else np.asarray(tri_indices),
    )


def _handle(particles, triangles=(), tube_radius=0.0, point_radius=0.0, surface=None):
    return types.SimpleNamespace(
        particles=np.asarray(particles, dtype=np.int32),
        triangles=np.asarray(triangles, dtype=np.int32).reshape(-1, 3),
        tube_radius=tube_radius,
        point_radius=point_radius,
        surface=surface,
    )


class TestDrawables(unittest.TestCase):
    def test_a_sheet_is_a_mesh_of_the_particles_its_triangles_use(self) -> None:
        """A soft body's inner nodes are no vertex of its surface."""
        from robosimstudio.core.render.deformables import deformable_drawables

        # particles 10..14; the triangles use local 0, 1, 2 and 4 -- local 3 (particle 13) is inside
        handle = _handle([10, 11, 12, 13, 14], [[0, 1, 2], [0, 2, 4]])
        drawn = deformable_drawables(_model(20), {"towel": handle}, {"towel": (0.8, 0.2, 0.2)})
        (mesh,) = drawn.meshes
        self.assertEqual(mesh.particles.tolist(), [10, 11, 12, 14])
        self.assertEqual(mesh.triangles.tolist(), [[0, 1, 2], [0, 2, 3]])
        self.assertEqual(mesh.color, (0.8, 0.2, 0.2))
        # particle 13 is the towel's and drawn by nobody; every particle no handle names is a sphere
        (rest,) = drawn.points
        self.assertEqual(rest.name, "")
        self.assertEqual(rest.particles.tolist(), [*range(10), *range(15, 20)])

    def test_a_rope_is_a_tube_round_its_nodes(self) -> None:
        from robosimstudio.core.render.deformables import deformable_drawables
        from robosimstudio.core.render.visual import TUBE_SIDES

        drawn = deformable_drawables(_model(6), {"cord": _handle(range(6), tube_radius=0.003)})
        (tube,) = drawn.meshes
        self.assertEqual(tube.vertex_count, 6 * TUBE_SIDES + 2)
        q = np.zeros((6, 3))
        q[:, 0] = np.linspace(0.0, 0.5, 6)
        verts, _ = tube.vertices(q)
        self.assertEqual(verts.shape, (tube.vertex_count, 3))
        self.assertLess(int(tube.triangles.max()), tube.vertex_count)
        rings = verts[: 6 * TUBE_SIDES]
        np.testing.assert_allclose(np.linalg.norm(rings[:, 1:], axis=1), 0.003, atol=1e-9)
        self.assertFalse(drawn.points, "every particle is the rope's")

    def test_a_heap_is_its_grains_at_the_handles_radius(self) -> None:
        from robosimstudio.core.render.deformables import deformable_drawables

        drawn = deformable_drawables(_model(4, radius=0.01), {"rice": _handle(range(4), point_radius=0.002)})
        (points,) = drawn.points
        self.assertEqual(points.name, "rice")
        np.testing.assert_allclose(points.radii, 0.002)
        self.assertFalse(drawn.meshes)

    def test_a_surfaced_body_is_its_surface_for_a_renderer_that_can_regrow_one(self) -> None:
        from robosimstudio.core.render.deformables import deformable_drawables

        def surface(positions):
            return positions[:3], np.array([0, 1, 2])

        pool = _handle(range(4), point_radius=0.002, surface=surface)
        self.assertEqual(len(deformable_drawables(_model(4), {"water": pool}).points), 1, "gl draws its grains")
        drawn = deformable_drawables(_model(4), {"water": pool}, surfaces=True)
        self.assertFalse(drawn.points)
        (grown,) = drawn.surfaces
        q = np.arange(12, dtype=np.float64).reshape(4, 3)
        verts, tris = grown.mesh(q)
        self.assertEqual((verts.dtype, tris.dtype), (np.float32, np.int32))
        np.testing.assert_allclose(verts, q[:3])
        self.assertEqual(tris.tolist(), [[0, 1, 2]])

    def test_a_handle_into_another_models_particles_draws_nothing_of_it(self) -> None:
        from robosimstudio.core.render.deformables import deformable_drawables

        drawn = deformable_drawables(_model(3), {"stale": _handle([1, 2, 7], [[0, 1, 2]])})
        self.assertFalse(drawn.meshes)
        (rest,) = drawn.points
        self.assertEqual(rest.particles.tolist(), [0, 1, 2], "the particles are still drawn, as nobody's")

    def test_without_handles_the_models_triangles_are_one_mesh(self) -> None:
        from robosimstudio.core.render.deformables import deformable_drawables

        drawn = deformable_drawables(_model(6, tri_indices=[[1, 2, 3], [2, 3, 4]]))
        (mesh,) = drawn.meshes
        self.assertEqual(mesh.particles.tolist(), [1, 2, 3, 4])
        self.assertEqual(mesh.triangles.tolist(), [[0, 1, 2], [1, 2, 3]])
        (rest,) = drawn.points
        self.assertEqual(rest.particles.tolist(), [0, 5])
        np.testing.assert_allclose(rest.radii, 0.004)

    def test_a_handles_particles_are_not_drawn_twice_by_the_models_triangles(self) -> None:
        from robosimstudio.core.render.deformables import deformable_drawables

        model = _model(8, tri_indices=[[0, 1, 2], [4, 5, 6]])
        drawn = deformable_drawables(model, {"towel": _handle([0, 1, 2], [[0, 1, 2]])})
        self.assertEqual([m.name for m in drawn.meshes], ["towel", ""])
        self.assertEqual(drawn.meshes[1].particles.tolist(), [4, 5, 6], "only what no handle names")
        (rest,) = drawn.points
        self.assertEqual(rest.particles.tolist(), [3, 7])

    def test_a_model_without_particles_has_nothing_to_draw(self) -> None:
        from robosimstudio.core.render.deformables import deformable_drawables

        self.assertFalse(deformable_drawables(_model(0), {}))

    def test_a_flat_sheets_normals_point_one_way(self) -> None:
        from robosimstudio.core.render.deformables import vertex_normals

        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
        normals = vertex_normals(points, np.array([[0, 1, 2], [0, 2, 3]]))
        np.testing.assert_allclose(normals, np.tile([0.0, 0.0, 1.0], (4, 1)))


if __name__ == "__main__":
    unittest.main()
