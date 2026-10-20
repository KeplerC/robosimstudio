# SPDX-License-Identifier: Apache-2.0
"""``robosimstudio_materials/bag.py``: the surface a bag is built as, and what its measures read.

No GPU and no engine. The bag's physics -- the layers resting a contact radius apart, a hand
opening its mouth -- is driven through the examples' pages by
``tests/test_open_a_bag.py``. Here:

* **The surface.** Two layers share the nodes of the three sewn edges and nothing else, every
  edge borders at most two triangles, no triangle appears twice (a duplicated corner triangle
  sent SuperDex's solve to NaN on its first step), every normal points out of the bag, the
  layers lie two contact radii apart at rest, and the top layer's lip is turned up by the angle
  asked for.
* **The measures**, on positions written by hand: the gap between the layers is signed and
  ignores layers held apart, and the mouth's span and area are its rim's.
* **The declarations**: every parameter the bag hands SuperDex is one its builder reads, every
  field a shipped fabric states reaches the engine, and ``BAG_60HZ`` puts the bag on SuperDex
  beside the soft kind, on one island.
"""

from __future__ import annotations

import dataclasses
import math
import unittest

import numpy as np

TIER = "fast"
"""Numpy and the spec classes only: nothing is built or stepped."""


def _bag(**kwargs):
    from robosimstudio_materials.bag import EntityBag, bag

    entity = EntityBag(bag(**kwargs))
    vertices, faces = entity.surface()
    return entity, vertices, faces


class TestTheSurface(unittest.TestCase):
    """One surface, two layers, three sewn edges, one lip."""

    def test_the_layers_share_the_sewn_edges_and_nothing_else(self) -> None:
        entity, vertices, _ = _bag()
        top, bottom = entity.top, entity.bottom
        nx, ny = 24, 20
        self.assertEqual((nx + 1 + 3, ny + 1), top.shape, "the top layer runs on three 10 mm rows as the lip")
        self.assertEqual((nx + 1, ny + 1), bottom.shape)
        sewn = np.zeros((nx + 1, ny + 1), dtype=bool)
        sewn[0, :] = sewn[:, 0] = sewn[:, -1] = True
        np.testing.assert_array_equal(top[: nx + 1][sewn], bottom[sewn])
        self.assertFalse(set(top[: nx + 1][~sewn].ravel()) & set(bottom[~sewn].ravel()), "only the seams are shared")
        self.assertEqual(len(vertices), len(set(top.ravel()) | set(bottom.ravel())), "every vertex is on a layer")

    def test_the_mesh_is_a_surface_without_repeats(self) -> None:
        _, _, faces = _bag()
        self.assertEqual(len(faces), len(np.unique(np.sort(faces, axis=1), axis=0)), "a triangle is repeated")
        edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        _, count = np.unique(edges, axis=0, return_counts=True)
        self.assertEqual(2, count.max(), "an edge borders more than two triangles")

    def test_every_normal_points_out_of_the_bag(self) -> None:
        entity, vertices, faces = _bag()
        tri = vertices[faces]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nx = 24
        top_inner = set(entity.top[1:nx, 1:-1].ravel())
        bottom_inner = set(entity.bottom[1:nx, 1:-1].ravel())
        on_top = np.array([all(v in top_inner for v in f) for f in faces])
        on_bottom = np.array([all(v in bottom_inner for v in f) for f in faces])
        self.assertTrue(on_top.any() and on_bottom.any())
        self.assertTrue(np.all(normal[on_top, 2] > 0.0), "the top layer faces up")
        self.assertTrue(np.all(normal[on_bottom, 2] < 0.0), "the bottom layer faces down")

    def test_the_layers_rest_two_radii_apart(self) -> None:
        entity, vertices, _ = _bag()
        radius = entity.spec.material.particle_radius
        gap = vertices[entity.top[1:24, 1:-1], 2] - vertices[entity.bottom[1:24, 1:-1], 2]
        np.testing.assert_allclose(gap, 2.0 * radius)

    def test_the_lip_runs_on_from_the_top_layer_turned_up(self) -> None:
        for angle in (1.3, 0.5):
            with self.subTest(lip_angle=angle):
                entity, vertices, _ = _bag(lip_angle=angle)
                self.assertEqual(28, entity.top.shape[0])
                step = vertices[entity.top[25, 10]] - vertices[entity.top[24, 10]]
                self.assertAlmostEqual(angle, math.atan2(step[2], step[0]), places=6)
                self.assertAlmostEqual(0.01, float(np.linalg.norm(step)), places=6)

    def test_the_footprint_is_centred_on_its_pose(self) -> None:
        entity, vertices, _ = _bag()
        body = vertices[entity.bottom.ravel()]
        np.testing.assert_allclose(body[:, :2].min(axis=0), (-0.12, -0.10), atol=1e-12)
        np.testing.assert_allclose(body[:, :2].max(axis=0), (0.12, 0.10), atol=1e-12)


class TestTheMeasures(unittest.TestCase):
    """The gap, the mouth and the fold, on positions written by hand."""

    def test_the_gap_is_signed_and_ignores_layers_held_apart(self) -> None:
        from robosimstudio_materials.bag import layer_gap

        entity, vertices, _ = _bag()
        # Two radii, a little less beside the seams, where the bottom layer tilts up to meet the top.
        rest = layer_gap(vertices, entity)
        self.assertTrue(0.0075 < rest <= 0.008, rest)
        crossed = vertices.copy()
        crossed[entity.top[12, 10], 2] = vertices[entity.bottom[12, 10], 2] - 0.002
        self.assertAlmostEqual(-0.002, layer_gap(crossed, entity), places=9)
        held_open = crossed.copy()
        held_open[entity.top[12, 10], 2] += 0.5
        self.assertAlmostEqual(rest, layer_gap(held_open, entity), places=9, msg="a pair held apart has no say")

    def test_the_mouth_reads_its_rim(self) -> None:
        from robosimstudio_materials.bag import mouth_opening

        entity, vertices, _ = _bag()
        shut = mouth_opening(vertices, entity)
        self.assertAlmostEqual(0.008, shut["height"], places=9)
        # A band two radii tall between the inner nodes, tapering to the sewn corners over a cell.
        self.assertAlmostEqual(0.008 * 0.19, shut["area"], places=9)
        opened = vertices.copy()
        # The top rim lifted into a half-ellipse 60 mm high over the 200 mm mouth.
        rim = entity.top[24]
        y = vertices[rim, 1]
        opened[rim, 2] = vertices[entity.bottom[24], 2] + 0.06 * np.sqrt(np.clip(1.0 - (y / 0.1) ** 2, 0.0, 1.0))
        out = mouth_opening(opened, entity)
        self.assertAlmostEqual(0.06, out["height"], places=6)
        self.assertAlmostEqual(math.pi * 0.1 * 0.06 / 2.0, out["area"], delta=0.0003)


class TestTheDeclarations(unittest.TestCase):
    """What the bag hands the engine, and where the arrangement puts it."""

    def test_every_parameter_reaches_the_builder_and_every_stated_field_the_engine(self) -> None:
        import robosimstudio_materials.bag as bag_module
        from robosimstudio.specs.physics import actor_consumes
        from robosimstudio_materials.cloth import ClothMaterialSpec

        for fabric in ("CANVAS",):
            with self.subTest(fabric=fabric):
                material = getattr(bag_module, fabric)
                entity, _, _ = _bag(material=material)
                actor = entity.host_actor()
                self.assertEqual("bag", actor.kind)
                self.assertLessEqual(set(actor.params), set(actor_consumes("superdex", "bag")))
                default = ClothMaterialSpec()
                # Under the name each crosses as: the canvas's ``shell_young`` is the shell's ``young``.
                stated = {
                    ClothMaterialSpec.actor_names.get(f.name, f.name)
                    for f in dataclasses.fields(material)
                    if getattr(material, f.name) != getattr(default, f.name)
                }
                self.assertLessEqual(
                    stated - {"shell"}, set(actor.params), "a stated field would never reach the engine"
                )

    def test_self_contact_is_the_specs_and_a_slab_is_refused(self) -> None:
        from robosimstudio_materials.bag import CANVAS, bag
        from robosimstudio_materials.cloth import TERRY_TOWELLING

        for on in (True, False):
            entity, _, _ = _bag(self_contact=on)
            self.assertIs(on, entity.sim_requirements().self_contact)
            self.assertIs(on, entity.host_actor().params["self_contact"])
        with self.assertRaises(ValueError):
            bag(material=TERRY_TOWELLING)
        self.assertEqual(3, bag(lip=0.026).lip_cells)
        self.assertEqual(0.0, CANVAS.thickness - 1.0e-3)

    def test_a_declared_sim_states_the_engines_self_contact_as_it_states_the_scenes(self) -> None:
        """The record's requirements are what the scene is composed from, so the shell collides as they say."""
        from robosimstudio.specs.physics import SimRequirements

        for on in (True, False):
            with self.subTest(declared=on):
                entity, _, _ = _bag(self_contact=not on, sim=SimRequirements(domain="bag", self_contact=on))
                self.assertIs(on, entity.sim_requirements().self_contact)
                self.assertIs(on, entity.host_actor().params["self_contact"])

    def test_the_record_states_the_requirements_the_bag_runs(self) -> None:
        """``Bag.from_spec`` reads the spec as the entity does: a declared ``sim`` first, else its self-contact."""
        from robosimstudio.specs.physics import SimRequirements
        from robosimstudio_materials.bag import Bag, EntityBag, bag

        for on in (True, False):
            for declared in (None, SimRequirements(domain="bag", self_contact=on)):
                with self.subTest(self_contact=on, declared=declared is not None):
                    spec = bag(
                        name="pouch", cells=(4, 4), self_contact=on if declared is None else not on, sim=declared
                    )
                    record = Bag.from_spec(spec, name="pouch")
                    runs = EntityBag(spec).sim_requirements()
                    self.assertIs(runs.self_contact, record.sim.self_contact)
                    self.assertIs(on, record.sim.self_contact)
                    if declared is not None:
                        self.assertEqual(declared, record.sim)

    def test_a_bag_one_cell_across_is_refused(self) -> None:
        """One cell across, every node up to the mouth is sewn: the two layers would be one surface twice."""
        from robosimstudio_materials.bag import bag

        for cells in ((4, 1), (0, 4)):
            with self.subTest(cells=cells), self.assertRaisesRegex(ValueError, r"bag 'pouch': cells needs"):
                bag(name="pouch", cells=cells)
        _, _, faces = _bag(cells=(1, 2))
        self.assertEqual(len(faces), len(np.unique(np.sort(faces, axis=1), axis=0)), "a triangle is repeated")

    def test_a_bag_leaves_newtons_soft_contact_to_the_sheet_newton_simulates(self) -> None:
        """A VBD towel and a bag on SuperDex: the bag, applied after the towel, writes nothing over it."""
        from types import SimpleNamespace

        from robosimstudio_materials.cloth import TERRY_TOWELLING, ClothSpec, EntityCloth

        model = SimpleNamespace(soft_contact_ke=0.0, soft_contact_kd=0.0, soft_contact_mu=0.0)
        EntityCloth(ClothSpec(name="towel", material=TERRY_TOWELLING)).apply_model_materials(model)
        entity, _, _ = _bag()
        entity.apply_model_materials(model)
        towel = (TERRY_TOWELLING.contact_ke, TERRY_TOWELLING.contact_kd, TERRY_TOWELLING.friction)
        self.assertEqual(towel, (model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu))

    def test_the_settle_runs_at_the_substeps_it_states_or_the_runs_own(self) -> None:
        """As a soft body settles: ``settle_substeps`` 0 steps plainly, more steps inside ``substeps_at``."""
        import contextlib

        class Pipeline:
            def __init__(self) -> None:
                self.steps, self.at, self.sim_time, self.step_count = [], [], 1.0, 7

            @contextlib.contextmanager
            def substeps_at(self, substeps: int):
                self.at.append(substeps)
                yield

            def step(self) -> None:
                self.steps.append(len(self.at))

        for substeps, at in ((0, []), (4, [4])):
            with self.subTest(settle_substeps=substeps):
                pipeline = Pipeline()
                entity, _, _ = _bag(settle_frames=3, settle_substeps=substeps)
                entity.settle(pipeline)
                self.assertEqual(at, pipeline.at)
                self.assertEqual(3, len(pipeline.steps))
                self.assertEqual((0.0, 0), (pipeline.sim_time, pipeline.step_count))

    def test_only_superdex_builds_a_bag_and_it_builds_the_layers_inactive(self) -> None:
        """On GENERIC the bag's kind falls to VBD, which would leave the inactive layers frozen: refused."""
        import newton

        import robosimstudio as rss
        from robosimstudio.scene.registry import resolve_entity_class
        from robosimstudio_materials.bag import EntityBagSuperdex, bag

        spec = bag(name="pouch", cells=(4, 4))
        with self.assertRaisesRegex(NotImplementedError, r"pouch.*backend\('superdex'\)"):
            resolve_entity_class(spec, rss.GENERIC)(spec).build(newton.ModelBuilder(), None)
        cls = resolve_entity_class(spec, rss.BAG_60HZ)
        self.assertIs(EntityBagSuperdex, cls)
        builder = newton.ModelBuilder()
        cls(spec).build(builder, None)
        active = [int(f) & int(newton.ParticleFlags.ACTIVE) for f in builder.particle_flags]
        self.assertTrue(builder.particle_count > 0 and not any(active), "SuperDex's bag is left to SuperDex")

    def test_the_arrangement_puts_the_bag_on_superdex_beside_the_soft_kind(self) -> None:
        import robosimstudio as rss

        table = dict(rss.BAG_60HZ.backends)
        self.assertEqual("superdex", table["bag"].solver)
        self.assertEqual(table["soft"], table["bag"], "one backend, so one island")
        self.assertEqual("mujoco", table["articulated"].solver)
        self.assertIs(rss.bag, __import__("robosimstudio_materials.bag", fromlist=["bag"]).bag)

    def test_the_arrangement_keeps_every_other_route_generic_has(self) -> None:
        """Rebuilt from scratch, it sent a heap, a pool and a lump beside the bag to VBD rather than the MPM."""
        import robosimstudio as rss
        from robosimstudio.specs import GENERIC

        for domain, spec in GENERIC.backends:
            if domain != "soft":
                with self.subTest(domain=domain):
                    self.assertEqual(spec, rss.BAG_60HZ.backend_for(domain))
        self.assertEqual((rss.BAG_60HZ.control_hz, rss.BAG_60HZ.substeps), (GENERIC.control_hz, GENERIC.substeps))

    def test_the_kind_and_its_atom_are_registered(self) -> None:
        from robosimstudio.assetlib.semantics.atoms import ATOMS
        from robosimstudio.specs.physics import get_domain

        domain = get_domain("bag")
        self.assertEqual(("bag", "cloth"), tuple(domain.reads))
        self.assertIn("mouth_opening", ATOMS)


class TestTheEngineActor(unittest.TestCase):
    """The bag's SuperDex actor, against a stand-in engine: no wheel, no scene."""

    def test_its_nodes_read_back_through_its_root_as_a_placement_writes_them(self) -> None:
        """One rule for both ways: a root turned a quarter about z turns what the bag reads."""
        from types import SimpleNamespace
        from unittest import mock

        from robosimstudio.engines.superdex.bridge import SuperdexBridge
        from robosimstudio_materials.bag import build_bag_actor

        class Arr:
            def __init__(self, a) -> None:
                self.a = np.asarray(a)

            def numpy(self):
                return self.a

        entity, vertices, faces = _bag(cells=(4, 4))
        local = np.array([[0.01, 0.0, 0.0]] * len(vertices))
        actor = mock.Mock()
        actor.get_node_positions_local.return_value = local.ravel()
        half = math.sqrt(0.5)
        actor.get_root_transform.return_value = SimpleNamespace(
            rotation=np.array([0.0, 0.0, half, half]), translation=[0.6, 0.0, 0.75]
        )
        physics = mock.MagicMock()
        physics.uses_double_precision.return_value = False
        physics.experimental.create_shell_actor.return_value = actor
        # The bridge's own shell builder, which the bag's is, over a model that holds only the bag.
        bridge = object.__new__(SuperdexBridge)
        bridge.physics, bridge.recipe, bridge.scenes, bridge.num_worlds = physics, {}, [None], 1
        bridge.model = SimpleNamespace(particle_q=Arr(vertices), tri_indices=Arr(faces), tri_count=len(faces))
        record = {"name": "bag", "particle_start": 0, "particle_count": len(vertices)}
        entry = build_bag_actor(bridge, {**record, "params": entity.host_actor().params})
        np.testing.assert_allclose(entry["read"](), np.tile([0.6, 0.01, 0.75], (len(vertices), 1)), atol=1e-12)

    def test_a_record_stating_the_old_names_keeps_its_modulus(self) -> None:
        """A bag record from before it stated the shell's ``young``/``poisson`` says ``shell_young``/``shell_poisson``,
        and a repro bundle replays it verbatim: read as the shell's, not dropped for the canvas's 1e6 and 0.3. A shell
        name stated beside one wins."""
        from types import SimpleNamespace
        from unittest import mock

        from robosimstudio.engines.superdex.bridge import SuperdexBridge
        from robosimstudio_materials.bag import build_bag_actor

        class Arr:
            def __init__(self, a) -> None:
                self.a = np.asarray(a)

            def numpy(self):
                return self.a

        _, vertices, faces = _bag(cells=(4, 4))
        physics = mock.MagicMock()
        physics.uses_double_precision.return_value = False
        bridge = object.__new__(SuperdexBridge)
        bridge.physics, bridge.recipe, bridge.scenes, bridge.num_worlds = physics, {}, [None], 1
        bridge.model = SimpleNamespace(particle_q=Arr(vertices), tri_indices=Arr(faces), tri_count=len(faces))
        record = {"name": "bag", "particle_start": 0, "particle_count": len(vertices)}
        moduli = physics.experimental.shell_material_params_from3d_isotropic
        for params, want in (
            ({"shell_young": 2.0e5, "shell_poisson": 0.35}, (2.0e5, 0.35)),
            ({"shell_young": 2.0e5, "young": 3.0e5}, (3.0e5, 0.3)),
            ({}, (1.0e6, 0.3)),
        ):
            with self.subTest(**params):
                build_bag_actor(bridge, {**record, "params": params})
                self.assertEqual(want, tuple(moduli.call_args.args[:2]))


if __name__ == "__main__":
    unittest.main()
