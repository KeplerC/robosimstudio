# SPDX-License-Identifier: Apache-2.0
"""Three compiler steps that need no scene: the replication refusal, region anchoring, and the finish table.

**Replication** is refused by capability and by name: an entity whose substep
kernels carry world-0 indices, unless it says it is world aware, and an entity
served by a foreign engine that builds one scene from the whole model, unless
its bridge lays the worlds out itself.

**Regions** anchor where their frame is: a body's forward-kinematics pose, a
declared static's own transform (a region on a welded fixture used to anchor at
the world origin, where its containment could never fire -- the weave seats),
the arena's sidecar for anything else, and the work surface when no frame is
named at all.

**Finishes** cover every shape in the model: what the entities painted, then a
finish read off each remaining shape's colour and display role.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.core.render.finish import BRUSHED_METAL, DEFAULT, MATTE, SATIN, Finish
from robosimstudio.core.render.palette import METAL
from robosimstudio.scene.compiler import _collect_finishes, _replication_refusal, resolve_regions
from robosimstudio.scene.entity import Entity
from robosimstudio.scene.handles import DeformableHandle, EntityHandle, RegionHandle, SceneHandles
from robosimstudio.specs import ArrangementSpec, Binding, RegionSpec, SimRequirements, backend
from tests.fixtures.arrays import host
from tests.fixtures.models import tiny_model

TIER = "fast"
"""Bare entities, stand-in requests, and a two-body model finalized on the CPU: nothing is compiled."""


class _Kernels(Entity):
    """Contributes substep kernels, written for one world."""

    def contribute_hooks(self, hooks) -> None:
        del hooks


class _PerWorldKernels(_Kernels):
    world_aware = True


class _Soft(Entity):
    domain = "soft"


def _entity(cls: type = Entity, name: str = "thing", **spec) -> Entity:
    return cls(SimpleNamespace(name=name, **spec))


def _refusal(*entities, physics=None) -> str:
    return _replication_refusal(SimpleNamespace(entities=list(entities)), SimpleNamespace(physics=physics))


class TestWhatMayBeReplicated(unittest.TestCase):
    def test_entities_without_substep_kernels_may_be(self) -> None:
        self.assertEqual("", _refusal(_entity(), _entity(name="cube")))

    def test_kernels_written_for_one_world_are_refused_by_the_entitys_name(self) -> None:
        refusal = _refusal(_entity(), _entity(_Kernels, "latch"))
        self.assertIn("substep physics written for one world: latch", refusal)

    def test_every_contributor_is_named_in_order(self) -> None:
        refusal = _refusal(_entity(_Kernels, "zip"), _entity(_Kernels, "bond"), _entity(name="cube"))
        self.assertTrue(refusal.endswith(": bond, zip"), refusal)

    def test_an_entity_that_claims_world_awareness_is_let_through(self) -> None:
        self.assertEqual("", _refusal(_entity(_PerWorldKernels, "sponge")))

    def test_an_instance_may_say_its_class_kernels_are_not_its_own(self) -> None:
        """A robot is that case: only a vacuum cup's wrench is a kernel, so a robot with jaws replicates."""
        jaws = _entity(_Kernels, "robot")
        jaws.adds_substep_kernels = False
        self.assertEqual("", _refusal(jaws))

    def test_an_entity_on_a_foreign_engine_is_refused_naming_the_engine(self) -> None:
        physics = ArrangementSpec(backends=(("soft", backend("genesis")), ("*", backend("mujoco"))))
        refusal = _refusal(_entity(), _entity(_Soft, "sponge"), physics=physics)
        self.assertIn("served by a foreign engine", refusal)
        self.assertTrue(refusal.endswith(": sponge (genesis)"), refusal)

    def test_an_engine_whose_bridge_lays_out_the_worlds_is_let_through(self) -> None:
        physics = ArrangementSpec(backends=(("soft", backend("superdex")),))
        self.assertEqual("", _refusal(_entity(_Soft, "sponge"), physics=physics))

    def test_an_entitys_own_backend_beats_the_arrangements(self) -> None:
        physics = ArrangementSpec(backends=(("soft", backend("superdex")),))
        stated = _entity(_Soft, "sponge", sim=SimRequirements(domain="soft", backend=backend("genesis")))
        self.assertIn("sponge (genesis)", _refusal(stated, physics=physics))
        newton = _entity(_Soft, "towel", sim=SimRequirements(domain="soft", backend=backend("vbd")))
        self.assertEqual("", _refusal(newton, physics=ArrangementSpec(backends=(("soft", backend("genesis")),))))

    def test_a_domain_no_backend_serves_or_one_nothing_registered_is_not_guessed_at(self) -> None:
        self.assertEqual("", _refusal(_entity(_Soft), physics=ArrangementSpec(backends=(("cloth", backend("vbd")),))))
        unknown = ArrangementSpec(backends=(("soft", backend("no_such_engine")),))
        self.assertEqual("", _refusal(_entity(_Soft), physics=unknown))


def _request(*regions, statics=(), scene: str = "bench_075") -> SimpleNamespace:
    return SimpleNamespace(
        regions=list(regions),
        statics=[SimpleNamespace(name=n) for n in statics],
        binding_for=lambda: Binding(scene=scene),
    )


def _region(name: str, frame: str) -> RegionSpec:
    return RegionSpec(name=name, frame=frame, xy_ranges=(-0.1, -0.05, 0.1, 0.05), z_range=(0.0, 0.2))


class TestWhereARegionIsAnchored(unittest.TestCase):
    def setUp(self) -> None:
        self.model = tiny_model((("table", (0.0, 0.0, 0.0)), ("tote", (0.4, -0.2, 0.8))))
        self.handles = SceneHandles(
            objects={"tote": EntityHandle(name="tote", body=1)},
            statics={
                "station": EntityHandle(name="station", static=True, xform=(0.3, 0.1, 0.76, 0.0, 0.0, 0.0, 1.0)),
                "rack": EntityHandle(name="rack", static=True),
            },
            meta={"rack_pos": [1.0, 2.0, 3.0], "container_frames": {"bin2": [0.5, 0.6, 0.7]}},
        )

    def _resolve(self, *regions, **request):
        return resolve_regions(self.handles, _request(*regions, **request), self.model)

    def test_a_frame_with_a_body_anchors_at_its_pose(self) -> None:
        (handle,) = self._resolve(_region("in_tote", "tote")).values()
        self.assertEqual(handle.frame_body, 1)
        np.testing.assert_allclose(handle.origin, (0.4, -0.2, 0.8), atol=1e-6)

    def test_no_frame_is_the_work_surface_every_at_is_stated_in(self) -> None:
        (handle,) = self._resolve(_region("square", "")).values()
        np.testing.assert_allclose(handle.origin, (0.6, 0.0, 0.75), err_msg="bench_075's work surface")
        (bare,) = self._resolve(_region("square", ""), scene="").values()
        np.testing.assert_allclose(bare.origin, (0.0, 0.0, 0.0), err_msg="no cell: the world frame")

    def test_a_declared_static_anchors_at_its_own_transform_not_the_world_origin(self) -> None:
        """The weave seats: a region on a welded fixture has no body to read, and used to anchor at 0."""
        (handle,) = self._resolve(_region("seat", "station"), statics=("station",)).values()
        self.assertEqual(handle.frame_body, -1)
        np.testing.assert_allclose(handle.origin, (0.3, 0.1, 0.76))

    def test_an_arena_static_is_read_off_the_sidecar(self) -> None:
        """Not declared by the request, so its handle's transform is not this episode's to trust."""
        regions = self._resolve(_region("on_rack", "rack"), _region("in_bin", "bin2"), statics=("station",))
        np.testing.assert_allclose(regions["on_rack"].origin, (1.0, 2.0, 3.0))
        np.testing.assert_allclose(regions["in_bin"].origin, (0.5, 0.6, 0.7))

    def test_a_frame_nothing_knows_stays_at_the_origin(self) -> None:
        (handle,) = self._resolve(_region("lost", "nowhere")).values()
        np.testing.assert_allclose(handle.origin, (0.0, 0.0, 0.0))

    def test_the_table_is_fresh_and_carries_the_regions_extent(self) -> None:
        self.handles.regions = {"old": RegionHandle(name="old", frame="")}
        regions = self._resolve(_region("in_tote", "tote"), _region("square", ""))
        self.assertEqual(list(regions), ["in_tote", "square"])
        self.assertEqual(list(self.handles.regions), ["old"], "a caller's handles are not disturbed")
        self.assertEqual(
            (regions["square"].xy_ranges, regions["square"].z_range), ((-0.1, -0.05, 0.1, 0.05), (0.0, 0.2))
        )


class _Painted(Entity):
    """Painted its own shapes; ``recolors`` says whether it fades them while the scene runs."""

    def __init__(self, name: str, shapes: tuple[int, int], *, recolors: bool = False, **spec) -> None:
        super().__init__(SimpleNamespace(name=name, **spec))
        self.range.shapes = shapes
        self._recolors = recolors

    @property
    def recolors_live(self) -> bool:
        return self._recolors


class TestEveryShapeGetsAFinish(unittest.TestCase):
    ODD = (0.123, 0.456, 0.789)
    """A colour the palette does not key a finish to, so the role decides."""

    def _collect(self, entities, colors, *, robot=(), deformables=()):
        handles = SceneHandles(
            robot_shapes=np.asarray(robot, dtype=np.int32),
            deformables={name: DeformableHandle(name=name) for name in deformables},
        )
        model = SimpleNamespace(shape_count=len(colors), shape_color=host(np.asarray(colors, dtype=np.float32)))
        _collect_finishes(handles, model, entities)
        return handles

    def test_an_entitys_own_finish_is_kept_and_every_other_shape_is_filled(self) -> None:
        glossy = Finish(roughness=0.05)
        cup = _Painted("cup", (0, 1))
        cup.finishes = {0: glossy}
        handles = self._collect([cup], [self.ODD] * 4)
        self.assertEqual(sorted(handles.finishes), [0, 1, 2, 3])
        self.assertIs(handles.finishes[0], glossy)

    def test_an_unlisted_shape_takes_its_roles_finish(self) -> None:
        handles = self._collect([], [self.ODD] * 3, robot=(1,))
        self.assertEqual([handles.finishes[s] for s in range(3)], [DEFAULT, SATIN, DEFAULT])

    def test_a_palette_colour_says_its_finish_whatever_the_role(self) -> None:
        handles = self._collect([], [METAL, METAL], robot=(1,))
        self.assertEqual([handles.finishes[0], handles.finishes[1]], [BRUSHED_METAL, BRUSHED_METAL])

    def test_a_deformable_takes_its_stated_finish_before_its_colour(self) -> None:
        stated = Finish(roughness=0.2)
        towel = _Painted("towel", (0, 0), finish=stated, color=METAL)
        rope = _Painted("rope", (0, 0), finish=None, color=self.ODD)
        cube = _Painted("cube", (0, 1), finish=Finish(roughness=0.4))
        handles = self._collect([towel, rope, cube], [self.ODD], deformables=("towel", "rope"))
        self.assertEqual(handles.deformable_finishes, {"towel": stated, "rope": MATTE})

    def test_the_live_set_is_the_shapes_of_whatever_recolours_while_running(self) -> None:
        stain = _Painted("stain", (4, 7), recolors=True)
        seam = _Painted("seam", (1, 3), recolors=True)
        handles = self._collect([stain, _Painted("bench", (0, 1)), seam], [self.ODD] * 8)
        self.assertEqual(handles.live_shapes.tolist(), [1, 2, 4, 5, 6])
        self.assertEqual(handles.live_shapes.dtype, np.int32)


if __name__ == "__main__":
    unittest.main()
