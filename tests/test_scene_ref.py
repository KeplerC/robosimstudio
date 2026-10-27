# SPDX-License-Identifier: Apache-2.0
"""``Ref``: what ``scene.add`` hands back, and the same thing in another world.

A ``Ref`` reads the indices its entity owns -- ``owned_bodies()`` before a
``.bodies`` attribute, because an entity that overrides the method is correcting
the attribute (the fingertip hulls a rod's bracketed range would otherwise hand
a caller) -- and reads them against the live state, never a copy. In a
replicated scene ``world(n)`` is the same thing ``n`` strides along, the stride
being :class:`~robosimstudio.core.worlds.WorldMap`'s, and ``world(0)`` is the
base ref itself.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.core.worlds import WorldMap
from robosimstudio.scene.handles import DeformableHandle, EntityHandle, SceneHandles
from robosimstudio.scene.ref import Ref, _WorldRef, owned_indices
from tests.fixtures.arrays import host

TIER = "fast"
"""Stand-in entities over host arrays: no model, no scene, no GPU."""


class _Box:
    """A rigid thing whose method corrects its attribute, as a rod's bracketed range does."""

    bodies = (0,)
    shapes = (0, 1)

    def __init__(self) -> None:
        self.spec = SimpleNamespace(name="box")
        self.asked = []

    def owned_bodies(self):
        return [2]

    def owned_shapes(self):
        return [4, 5]

    def tunables(self, pipeline):
        self.asked.append(pipeline)
        return iter(("mass", "mu"))


class _Cloth:
    """A deformable: particles and no body."""

    particles = (1, 3)

    def __init__(self) -> None:
        self.spec = SimpleNamespace(name="towel")


def _scene(*, worlds: WorldMap | None = None) -> SimpleNamespace:
    """Two things over twelve bodies' and eight particles' worth of live state."""
    body_q = np.zeros((12, 7))
    body_q[:, 0] = np.arange(12)  # x is the row, so a pose names the row it came from
    body_q[:, 6] = 1.0
    body_qd = np.zeros((12, 6))
    body_qd[:, 5] = np.arange(12)
    particle_q = np.stack([np.arange(8.0), np.zeros(8), np.zeros(8)], axis=1)
    handles = SceneHandles(
        objects={"box": EntityHandle(name="box", body=2, shapes=np.array([9]))},
        deformables={"towel": DeformableHandle(name="towel")},
        statics={"post": EntityHandle(name="post", static=True, shapes=np.array([7, 8]))},
    )
    return SimpleNamespace(
        by_name={"box": _Box(), "towel": _Cloth()},
        handles=handles,
        pipeline=SimpleNamespace(
            state_0=SimpleNamespace(body_q=host(body_q), body_qd=host(body_qd), particle_q=host(particle_q))
        ),
        worlds=worlds,
    )


def _worlds(n: int = 3) -> WorldMap:
    return WorldMap(
        num_worlds=n,
        bodies_per_world=4,
        shapes_per_world=10,
        joints_per_world=4,
        coords_per_world=28,
        dofs_per_world=24,
        particles_per_world=2,
    )


class TestOwnedIndices(unittest.TestCase):
    def test_owned_bodies_beats_the_bodies_attribute(self) -> None:
        """The fingertip-hull bug: reading the attribute handed a caller shapes belonging to a hand."""
        np.testing.assert_array_equal(owned_indices(_Box(), "owned_bodies", "bodies"), [2])
        np.testing.assert_array_equal(owned_indices(_Box(), "owned_shapes", "shapes"), [4, 5])

    def test_without_the_method_the_attribute_is_read_as_flat_int32(self) -> None:
        got = owned_indices(SimpleNamespace(particles=((1, 3), (5, 7))), "owned_particles", "particles")
        self.assertEqual(got.dtype, np.int32)
        np.testing.assert_array_equal(got, [1, 3, 5, 7])

    def test_nothing_built_owns_nothing(self) -> None:
        for entity in (None, SimpleNamespace()):
            with self.subTest(entity=entity):
                got = owned_indices(entity, "owned_bodies", "bodies")
                self.assertEqual((got.size, got.dtype), (0, np.int32))


class TestARefReadsTheLiveState(unittest.TestCase):
    def test_the_pose_is_the_first_owned_bodys_row_read_live(self) -> None:
        scene = _scene()
        ref = Ref(scene, "box")
        self.assertEqual(ref.pose[0], 2.0, "body 2, which owned_bodies() names, not the attribute's body 0")
        scene.pipeline.state_0.body_q.data[2, 0] = 9.5
        self.assertEqual(ref.position[0], 9.5, "nothing is cached: the next read is the state that exists now")
        self.assertEqual(ref.velocity[5], 2.0)

    def test_a_cloth_has_no_pose_and_says_where_its_particles_are(self) -> None:
        ref = Ref(_scene(), "towel")
        with self.assertRaisesRegex(ValueError, r"'towel' owns no body.*\.particle_q"):
            _ = ref.pose
        with self.assertRaisesRegex(ValueError, "no velocity"):
            _ = ref.velocity

    def test_particle_q_reads_only_the_owned_particles(self) -> None:
        scene = _scene()
        got = Ref(scene, "towel").particle_q
        self.assertEqual(got.dtype, np.float32)
        np.testing.assert_array_equal(got[:, 0], [1.0, 3.0])
        self.assertEqual(Ref(scene, "box").particle_q.shape, (0, 3), "a rigid thing owns no particles")

    def test_shapes_fall_back_to_the_handle_only_when_the_entity_owns_none(self) -> None:
        scene = _scene()
        np.testing.assert_array_equal(Ref(scene, "box").shapes, [4, 5], "the entity's own beat the handle's")
        np.testing.assert_array_equal(Ref(scene, "post").shapes, [7, 8], "the arena's static: the handle's")

    def test_a_thing_the_scene_import_declared_has_no_entity_and_no_spec(self) -> None:
        ref = Ref(_scene(), "post")
        self.assertIsNone(ref.entity)
        self.assertIsNone(ref.spec)
        self.assertTrue(ref.handle.static)

    def test_tunables_are_asked_of_the_entity_with_the_live_pipeline(self) -> None:
        scene = _scene()
        self.assertEqual(Ref(scene, "box").tunables(), ("mass", "mu"))
        self.assertEqual(scene.by_name["box"].asked, [scene.pipeline])
        self.assertEqual(Ref(scene, "towel").tunables(), ())


class TestAnotherWorld(unittest.TestCase):
    def test_world_zero_and_an_unreplicated_scene_are_the_base_ref(self) -> None:
        ref = Ref(_scene(worlds=_worlds()), "box")
        self.assertIs(ref.world(0), ref)
        lone = Ref(_scene(), "box")
        self.assertIs(lone.world(2), lone, "a caller written for one world reads correctly in both")

    def test_world_n_shifts_every_index_by_its_stride(self) -> None:
        scene = _scene(worlds=_worlds())
        for n in (1, 2):
            with self.subTest(world=n):
                box, towel = Ref(scene, "box").world(n), Ref(scene, "towel").world(n)
                np.testing.assert_array_equal(box.bodies, [2 + 4 * n])
                np.testing.assert_array_equal(box.shapes, [4 + 10 * n, 5 + 10 * n])
                np.testing.assert_array_equal(towel.particles, [1 + 2 * n, 3 + 2 * n])

    def test_world_n_reads_its_own_row(self) -> None:
        ref = Ref(_scene(worlds=_worlds()), "box").world(2)
        self.assertEqual(ref.pose[0], 10.0)
        np.testing.assert_array_equal(ref.particle_q.shape, (0, 3))

    def test_an_empty_index_set_stays_empty_in_every_world(self) -> None:
        towel = Ref(_scene(worlds=_worlds()), "towel").world(2)
        self.assertEqual(towel.bodies.size, 0)
        with self.assertRaises(ValueError):
            _ = towel.pose

    def test_a_world_ref_hops_from_the_template_not_from_where_it_is(self) -> None:
        scene = _scene(worlds=_worlds())
        two = Ref(scene, "box").world(2)
        np.testing.assert_array_equal(two.world(1).bodies, [2 + 4], "world 1, not world 3")
        back = two.world(0)
        self.assertIs(type(back), Ref, "world 0 is the plain ref again")
        np.testing.assert_array_equal(back.bodies, [2])
        self.assertIsInstance(two, _WorldRef)


if __name__ == "__main__":
    unittest.main()
