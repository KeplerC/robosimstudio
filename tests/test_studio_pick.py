# SPDX-License-Identifier: Apache-2.0
"""What a click in the 3D view hits: a ray against boxes, triangles, shapes on bodies, and particles.

The picker answers *what did I click* host-side, in numpy, against geometry
gathered once: a slab test to cull, Möller-Trumbore over the survivors, double
sided, and for a cloud of particles the one nearest the ray within a tolerance.
Every answer here is one a ruler gives -- a ray down the x axis enters a box
whose near face is at x = 1 at a distance of 1 -- so the shapes are hand-made
:class:`~robosimstudio.core.render.visual.ShapeMesh` records laid into the
picker's cache, and the state is host arrays.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.core.render.visual import ShapeMesh
from robosimstudio_studio.pick import ScenePicker, _box_entry, _hits_box, _hits_triangles
from tests.fixtures.arrays import host

TIER = "fast"
"""No GPU: hand-made meshes and host arrays stand in for a model and its state."""

X = np.array([1.0, 0.0, 0.0])
ORIGIN = np.zeros(3)

# A unit box's twelve triangles, on its eight corners.
_CORNERS = np.array([[i, j, k] for i in (-1.0, 1.0) for j in (-1.0, 1.0) for k in (-1.0, 1.0)])
_FACES = np.array(
    [
        [0, 1, 3], [0, 3, 2],  # x = -1
        [4, 6, 7], [4, 7, 5],  # x = +1
        [0, 4, 5], [0, 5, 1],  # y = -1
        [2, 3, 7], [2, 7, 6],  # y = +1
        [0, 2, 6], [0, 6, 4],  # z = -1
        [1, 5, 7], [1, 7, 3],  # z = +1
    ],
    dtype=np.int64,
)  # fmt: skip


def _box(index: int, half: float = 0.1, *, at=(0.0, 0.0, 0.0), body: int = -1) -> ShapeMesh:
    """A cube of half-size ``half`` placed at ``at`` in its body's frame (or the world's, for body -1)."""
    local = np.eye(4)
    local[:3, 3] = at
    vertices = _CORNERS * half
    return ShapeMesh(index, body, local, vertices, _FACES, vertices.min(axis=0), vertices.max(axis=0))


def _picker(*shapes: ShapeMesh, **kwargs) -> ScenePicker:
    """A picker whose geometry is these shapes, laid straight into its cache."""
    picker = ScenePicker(SimpleNamespace(particle_count=0), **kwargs)
    picker._shapes = list(shapes)
    return picker


class TestABox(unittest.TestCase):
    LO, HI = np.full(3, -1.0), np.full(3, 1.0)

    def test_a_ray_from_outside_enters_at_the_near_face(self) -> None:
        self.assertAlmostEqual(2.0, _box_entry(np.array([-3.0, 0.0, 0.0]), X, self.LO, self.HI))

    def test_a_ray_from_inside_enters_where_it_starts(self) -> None:
        self.assertEqual(0.0, _box_entry(ORIGIN, X, self.LO, self.HI))

    def test_a_box_behind_the_ray_is_missed(self) -> None:
        self.assertIsNone(_box_entry(np.array([3.0, 0.0, 0.0]), X, self.LO, self.HI))
        self.assertFalse(_hits_box(np.array([3.0, 0.0, 0.0]), X, self.LO, self.HI))

    def test_a_ray_parallel_to_a_face_hits_only_between_its_slabs(self) -> None:
        self.assertTrue(_hits_box(np.array([-3.0, 0.5, 0.0]), X, self.LO, self.HI))
        self.assertFalse(_hits_box(np.array([-3.0, 1.5, 0.0]), X, self.LO, self.HI))


class TestTriangles(unittest.TestCase):
    TRI = np.array([[1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 0.0, 1.0]])
    FACE = np.array([[0, 1, 2]])

    def test_either_face_of_a_triangle_is_hit(self) -> None:
        self.assertAlmostEqual(1.0, _hits_triangles(ORIGIN, X, self.TRI, self.FACE))
        self.assertAlmostEqual(1.0, _hits_triangles(ORIGIN, X, self.TRI, self.FACE[:, ::-1]), msg="wound backwards")
        self.assertAlmostEqual(1.0, _hits_triangles(np.array([2.0, 0.0, 0.0]), -X, self.TRI, self.FACE))

    def test_a_triangle_behind_the_origin_is_not_hit(self) -> None:
        self.assertIsNone(_hits_triangles(np.array([2.0, 0.0, 0.0]), X, self.TRI, self.FACE))

    def test_a_ray_in_the_triangles_plane_is_not_hit(self) -> None:
        self.assertIsNone(_hits_triangles(np.array([1.0, -3.0, 0.0]), np.array([0.0, 1.0, 0.0]), self.TRI, self.FACE))

    def test_a_ray_past_the_triangles_edge_is_not_hit(self) -> None:
        self.assertIsNone(_hits_triangles(np.array([0.0, 2.0, 0.0]), X, self.TRI, self.FACE))

    def test_of_several_the_nearest_is_the_one_entered(self) -> None:
        tris = np.concatenate([self.TRI + np.array([2.0, 0.0, 0.0]), self.TRI])  # at x = 3, then at x = 1
        self.assertAlmostEqual(1.0, _hits_triangles(ORIGIN, X, tris, np.array([[0, 1, 2], [3, 4, 5]])))


class TestShapeAt(unittest.TestCase):
    STATE = SimpleNamespace(body_q=None)

    def test_of_two_shapes_on_the_ray_the_nearer_is_the_hit(self) -> None:
        picker = _picker(_box(4, at=(3.0, 0.0, 0.0)), _box(2, at=(1.0, 0.0, 0.0)))
        hit = picker.shape_at(self.STATE, ORIGIN, X)
        self.assertEqual((2, -1), (hit.shape, hit.body))
        self.assertAlmostEqual(0.9, hit.distance)
        np.testing.assert_allclose(hit.point, [0.9, 0.0, 0.0], atol=1e-12)

    def test_the_direction_need_not_be_a_unit_vector(self) -> None:
        hit = _picker(_box(2, at=(1.0, 0.0, 0.0))).shape_at(self.STATE, ORIGIN, 5.0 * X)
        self.assertAlmostEqual(0.9, hit.distance, msg="a distance in metres, not in ray lengths")

    def test_a_zero_direction_hits_nothing(self) -> None:
        self.assertIsNone(_picker(_box(2, at=(1.0, 0.0, 0.0))).shape_at(self.STATE, ORIGIN, np.zeros(3)))

    def test_a_shape_rides_its_body(self) -> None:
        turn = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))  # a quarter turn about z
        body_q = host([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, *turn]])
        picker = _picker(_box(5, at=(0.0, -1.0, 0.0), body=1))  # one metre down the body's -y: +x in the world
        hit = picker.shape_at(SimpleNamespace(body_q=body_q), ORIGIN, X)
        self.assertEqual((5, 1), (hit.shape, hit.body))
        self.assertAlmostEqual(0.9, hit.distance)
        self.assertIsNone(picker.shape_at(SimpleNamespace(body_q=body_q), ORIGIN, -np.array([0.0, 1.0, 0.0])))

    def test_with_no_bodies_in_the_state_only_world_shapes_are_pickable(self) -> None:
        picker = _picker(_box(5, at=(1.0, 0.0, 0.0), body=0), _box(6, at=(3.0, 0.0, 0.0)))
        self.assertEqual(6, picker.shape_at(SimpleNamespace(body_q=None), ORIGIN, X).shape)

    def test_a_shape_with_too_many_triangles_is_picked_on_its_box(self) -> None:
        wall = _box(3, at=(1.0, 0.0, 0.0))
        thin = ShapeMesh(3, -1, wall.local, wall.vertices, wall.faces[:0], wall.lo, wall.hi)  # a box with no faces
        self.assertIsNone(_picker(thin).shape_at(self.STATE, ORIGIN, X), "no triangle, no hit")
        hit = _picker(thin, max_triangles=-1).shape_at(self.STATE, ORIGIN, X)
        self.assertAlmostEqual(0.9, hit.distance, msg="past the cap the box alone answers")

    def test_invalidate_drops_the_gathered_geometry(self) -> None:
        picker = _picker(_box(2, at=(1.0, 0.0, 0.0)))
        picker.invalidate()
        self.assertIsNone(picker._shapes)


def _cloud(*points) -> tuple[ScenePicker, SimpleNamespace]:
    picker = ScenePicker(SimpleNamespace(particle_count=len(points)), particle_tolerance=0.05)
    return picker, SimpleNamespace(particle_q=host(np.array(points, dtype=np.float32).reshape(-1, 3)))


class TestParticleAt(unittest.TestCase):
    def test_the_particle_nearest_the_ray_is_the_hit_where_it_is(self) -> None:
        picker, state = _cloud([1.0, 0.04, 0.0], [2.0, 0.01, 0.0], [3.0, 0.3, 0.0])
        hit = picker.particle_at(state, ORIGIN, X)
        self.assertEqual(1, hit.index)
        np.testing.assert_allclose(hit.point, [2.0, 0.01, 0.0], atol=1e-6)
        self.assertAlmostEqual(2.0, hit.distance, places=6)
        self.assertAlmostEqual(0.01, hit.miss, places=6)

    def test_a_ray_that_passes_wide_of_every_particle_misses(self) -> None:
        picker, state = _cloud([1.0, 0.06, 0.0])
        self.assertIsNone(picker.particle_at(state, ORIGIN, X))

    def test_a_particle_behind_the_camera_is_not_clickable(self) -> None:
        picker, state = _cloud([-1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertIsNone(picker.particle_at(state, ORIGIN, X))

    def test_candidates_restrict_it_to_one_cloths_particles(self) -> None:
        picker, state = _cloud([1.0, 0.0, 0.0], [2.0, 0.02, 0.0])
        self.assertEqual(1, picker.particle_at(state, ORIGIN, X, candidates=[1]).index)
        self.assertIsNone(picker.particle_at(state, ORIGIN, X, candidates=[]))

    def test_a_model_with_no_particles_or_a_zero_ray_has_no_hit(self) -> None:
        picker, state = _cloud([1.0, 0.0, 0.0])
        self.assertIsNone(picker.particle_at(state, ORIGIN, np.zeros(3)))
        picker.model = SimpleNamespace(particle_count=0)
        self.assertIsNone(picker.particle_at(state, ORIGIN, X))


if __name__ == "__main__":
    unittest.main()
