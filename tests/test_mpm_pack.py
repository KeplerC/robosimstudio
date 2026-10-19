# SPDX-License-Identifier: Apache-2.0
"""The three particle kinds as registered knowledge: domains, presets, flat names, specs, on the CPU.

``granular``, ``liquid`` and ``dough`` are three modules of ``robosimstudio_materials``
over one shared package, ``robosimstudio_materials/mpm``. What the registries and
the library know of them before any scene is built is checked here, with no
GPU: the domains and their fallback chains, the one ``mpm`` route ``GENERIC``
gives them, the presets' completeness, the flat names, the specs' validation,
the asset records a task document places, and the sibling law (the kinds import
nothing beside the library and nothing of each other).
``tests/test_mpm_kinds`` builds them on a GPU.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

TIER = "fast"
"""No GPU: registries, dataclasses and a lattice sampled on the host."""

REPO = Path(__file__).resolve().parents[1]
MATERIALS = REPO / "robosimstudio_materials"
KINDS = ("granular", "liquid", "dough")


class TestTheDomains(unittest.TestCase):
    def test_each_kind_registers_its_domain_with_its_chain(self) -> None:
        from robosimstudio.specs.physics import get_domain

        self.assertEqual(("granular", "cloth"), get_domain("granular").reads)
        self.assertEqual(("liquid", "granular", "cloth"), get_domain("liquid").reads)
        self.assertEqual(("dough", "granular", "cloth"), get_domain("dough").reads)
        for kind in KINDS:
            info = get_domain(kind)
            self.assertEqual(("particles",), info.state, kind)
            self.assertEqual("deformables", info.role, kind)
            self.assertIn("hands", info.contacts, kind)
            self.assertIn("objects", info.contacts, kind)

    def test_the_layouts_fall_back_to_granular_then_cloth(self) -> None:
        from robosimstudio.specs.registries import layout_chain, reads_as

        self.assertEqual(("liquid", "granular", "cloth"), layout_chain("liquid"))
        self.assertEqual(("dough", "granular", "cloth"), layout_chain("dough"))
        self.assertTrue(reads_as("liquid", "granular"))
        self.assertTrue(reads_as("dough", "cloth"))
        self.assertFalse(reads_as("granular", "liquid"))

    def test_generic_routes_all_three_to_one_mpm_spec(self) -> None:
        from robosimstudio.specs.physics import GENERIC

        specs = {GENERIC.backend_for(kind) for kind in KINDS}
        self.assertEqual(1, len(specs))
        self.assertEqual("mpm", next(iter(specs)).solver)

    def test_the_entities_are_registered_under_their_names(self) -> None:
        from robosimstudio.scene.registry import get_entity
        from robosimstudio_materials.mpm import EntityParticleBody

        for name, domain in (("heap", "granular"), ("liquid", "liquid"), ("dough", "dough")):
            cls = get_entity(name)
            self.assertTrue(issubclass(cls, EntityParticleBody), name)
            self.assertEqual(domain, cls.domain, name)


class TestTheArrangements(unittest.TestCase):
    def test_the_tables_put_rigid_props_on_the_arms_solver(self) -> None:
        from robosimstudio_materials.mpm import DOUGH_60HZ, GRANULAR_60HZ, LIQUID_120HZ

        for table in (GRANULAR_60HZ, LIQUID_120HZ, DOUGH_60HZ):
            self.assertEqual(table.backend_for("articulated"), table.backend_for("rigid"))
            self.assertEqual("mujoco", table.backend_for("rigid").solver)
            self.assertEqual(1, len({table.backend_for(kind) for kind in KINDS}), "one island for the three")
            self.assertEqual(60.0, table.control_hz)
            self.assertEqual(16, table.substeps)

    def test_the_projection_the_iterations_and_the_rate_are_the_differences_between_the_tables(self) -> None:
        from robosimstudio_materials.mpm import DOUGH_60HZ, GRANULAR_60HZ, LIQUID_120HZ
        from robosimstudio_materials.mpm.physics import DOUGH_ITERATIONS, PROXY_MASS_SCALE

        self.assertTrue(GRANULAR_60HZ.backend_for("granular").get("project_outside"))
        self.assertFalse(LIQUID_120HZ.backend_for("liquid").get("project_outside"))
        self.assertTrue(DOUGH_60HZ.backend_for("dough").get("project_outside"))
        self.assertEqual(120.0, LIQUID_120HZ.backend_for("liquid").get("step_hz"), "a pour tunnels at 60 Hz")
        for table in (GRANULAR_60HZ, DOUGH_60HZ):
            self.assertIsNone(
                table.backend_for("granular").get("step_hz"), "the heap and the lump keep the registered 60"
            )
        self.assertEqual(DOUGH_ITERATIONS, DOUGH_60HZ.backend_for("dough").get("max_iterations"))
        self.assertIsNone(GRANULAR_60HZ.backend_for("granular").get("max_iterations"), "grains keep the registered 50")
        self.assertEqual(3, len({t.backend_for("granular") for t in (GRANULAR_60HZ, LIQUID_120HZ, DOUGH_60HZ)}))
        for table in (GRANULAR_60HZ, LIQUID_120HZ, DOUGH_60HZ):
            self.assertEqual("proxy", table.coupling.method)
            self.assertEqual(PROXY_MASS_SCALE, table.coupling.get("mass_scale"))

    def test_an_arrangement_that_states_a_mass_scale_keeps_it(self) -> None:
        from robosimstudio.specs.physics import GENERIC, backend, coupler
        from robosimstudio_materials.mpm import with_particles_on

        stated = with_particles_on(GENERIC.with_(coupling=coupler("proxy", mass_scale=20.0)), backend("mpm"))
        self.assertEqual(20.0, stated.coupling.get("mass_scale"))

    def test_the_rates_fit_the_composer(self) -> None:
        from robosimstudio.scene.physics.compose import _check_in_place_rates
        from robosimstudio.scene.physics.islands import Island
        from robosimstudio.specs.physics import get_backend
        from robosimstudio_materials.mpm import DOUGH_60HZ, GRANULAR_60HZ, LIQUID_120HZ

        for table in (GRANULAR_60HZ, LIQUID_120HZ, DOUGH_60HZ):
            for hz in (60.0, 120.0, 240.0):
                spec = table.backend_for("granular").with_(step_hz=hz)
                island = Island(name="mpm", backend=spec, solver_class=get_backend("mpm").cls)
                _check_in_place_rates(table, [island])


class TestThePresets(unittest.TestCase):
    def test_every_preset_carries_every_mpm_key_and_the_granular_kind(self) -> None:
        from robosimstudio.assetlib.materials import MATERIAL_KINDS, get_material, kind_of
        from robosimstudio_materials.mpm import MPM_KEYS, ParticleMaterialSpec, presets

        self.assertIs(MATERIAL_KINDS.get("granular"), ParticleMaterialSpec)
        names = presets()
        self.assertEqual(
            {"rice", "sand", "beans", "flour", "water", "honey", "ketchup", "dough", "plasticine", "clay", "butter"},
            set(names),
        )
        for name, record in names.items():
            with self.subTest(preset=name):
                self.assertIs(get_material(name), record)
                self.assertEqual("granular", kind_of(record.spec))
                attributes = record.spec.custom_attributes()
                self.assertEqual([f"mpm:{k}" for k in MPM_KEYS], list(attributes))
                self.assertTrue(all(np.isfinite(v) for v in attributes.values()))
                self.assertTrue(record.notes, "a preset says where its numbers come from")

    def test_the_mpm_keys_are_newtons_in_registration_order(self) -> None:
        import newton

        from robosimstudio_materials.mpm import MPM_KEYS

        builder = newton.ModelBuilder()
        newton.solvers.SolverImplicitMPM.register_custom_attributes(builder)
        registered = [
            attr.name
            for attr in builder.custom_attributes.values()
            if attr.namespace == "mpm" and attr.assignment == newton.Model.AttributeAssignment.MODEL
        ]
        self.assertEqual(list(MPM_KEYS), registered)

    def test_the_four_shapes_of_material(self) -> None:
        from robosimstudio_materials.mpm import DOUGH, HONEY, RICE, WATER

        self.assertEqual(0.0, RICE.spec.tensile_yield_ratio, "dry grains fly apart")
        self.assertEqual(0.0, WATER.spec.friction)
        self.assertEqual(1.0, WATER.spec.tensile_yield_ratio)
        self.assertGreater(HONEY.spec.viscosity, 0.0)
        self.assertLess(DOUGH.spec.young_modulus, 1.0e8, "a soft solid that can be pressed")
        self.assertGreater(DOUGH.spec.yield_stress, 0.0)
        self.assertIn("83%", DOUGH.notes, "the measured retention is written down")

    def test_the_stress_scale_halves_the_stresses_for_newton_1_7(self) -> None:
        """Newton 1.7's changelog halves these four for a scene tuned on 1.6; nothing else moves."""
        from robosimstudio_materials.mpm import DOUGH, HONEY, NEWTON_MPM_STRESS_SCALE

        self.assertEqual(0.5, NEWTON_MPM_STRESS_SCALE)
        for preset in (DOUGH, HONEY):
            written = preset.spec.custom_attributes()
            for key in ("young_modulus", "yield_pressure", "yield_stress", "viscosity"):
                self.assertEqual(0.5 * getattr(preset.spec, key), written[f"mpm:{key}"], key)
            for key in ("poisson_ratio", "damping", "friction", "tensile_yield_ratio", "hardening"):
                self.assertEqual(getattr(preset.spec, key), written[f"mpm:{key}"], key)


class TestTheFlatNames(unittest.TestCase):
    def test_the_constructors_resolve_through_the_pack_table(self) -> None:
        import robosimstudio as rss

        for name, module in (
            ("heap", "robosimstudio_materials.granular"),
            ("liquid", "robosimstudio_materials.liquid"),
            ("dough", "robosimstudio_materials.dough"),
            ("ParticleBodySpec", "robosimstudio_materials.granular"),
            ("ParticleMaterialSpec", "robosimstudio_materials.granular"),
            ("GRANULAR_60HZ", "robosimstudio_materials.granular"),
            ("LIQUID_120HZ", "robosimstudio_materials.liquid"),
            ("DOUGH_60HZ", "robosimstudio_materials.dough"),
        ):
            with self.subTest(name=name):
                self.assertEqual(module, rss._PACK_EXPORTS[name])
                self.assertNotIn(name, rss._EXPORTS)
                getattr(rss, name)

    def test_each_constructor_builds_its_kinds_spec(self) -> None:
        import robosimstudio as rss

        heap = rss.heap(name="rice", size=(0.12, 0.12, 0.03))
        self.assertEqual("heap", heap.entity)
        self.assertEqual(27648, heap.particle_count)
        self.assertAlmostEqual(0.648, heap.mass, places=3)
        self.assertEqual(0.68, heap.material.friction)
        pool = rss.liquid(name="h", material="honey", containers=("cup",))
        self.assertEqual("liquid", pool.entity)
        self.assertEqual(50.0, pool.material.viscosity)
        self.assertEqual(("cup",), pool.containers)
        lump = rss.dough(name="d", material="clay")
        self.assertEqual("dough", lump.entity)
        self.assertEqual(1.0e4, lump.material.yield_stress)


class TestTheSpecs(unittest.TestCase):
    def test_a_material_is_taken_by_name_record_or_spec(self) -> None:
        import robosimstudio as rss
        from robosimstudio_materials.mpm import RICE, ParticleMaterialSpec, resolve_material

        self.assertIs(resolve_material("rice"), RICE.spec)
        self.assertIs(resolve_material(RICE), RICE.spec)
        own = ParticleMaterialSpec(friction=0.1)
        self.assertIs(resolve_material(own), own)
        with self.assertRaises(TypeError):
            resolve_material("terry_towelling")
        with self.assertRaises(KeyError):
            rss.heap(material="granite")

    def test_validation_refuses_what_cannot_be_built(self) -> None:
        import robosimstudio as rss
        from robosimstudio_materials.mpm import ParticleMaterialSpec

        with self.assertRaises(ValueError):
            rss.heap(shape="sphere")
        with self.assertRaises(ValueError):
            rss.heap(shape="mesh")
        with self.assertRaises(ValueError):
            rss.heap(size=(0.1, 0.0, 0.1))
        with self.assertRaises(ValueError):
            rss.heap(particle_spacing=0.0)
        with self.assertRaises(ValueError):
            rss.heap(render="voxels")
        with self.assertRaises(ValueError):
            rss.heap(jitter=0.7)
        with self.assertRaises(ValueError):
            ParticleMaterialSpec(density=0.0)
        with self.assertRaises(ValueError):
            ParticleMaterialSpec(tensile_yield_ratio=1.5)
        with self.assertRaises(ValueError):
            ParticleMaterialSpec(poisson_ratio=0.5)

    def test_the_lattice_fills_the_stated_region(self) -> None:
        import robosimstudio as rss
        from robosimstudio_materials.mpm import sample_positions

        box = sample_positions(rss.heap(size=(0.04, 0.04, 0.02), particle_spacing=0.01, jitter=0.0))
        self.assertEqual((32, 3), box.shape)
        np.testing.assert_allclose(box.min(axis=0), (-0.015, -0.015, 0.005))
        np.testing.assert_allclose(box.max(axis=0), (0.015, 0.015, 0.015))
        puck = sample_positions(
            rss.dough(shape="cylinder", size=(0.04, 0.04, 0.02), particle_spacing=0.005, jitter=0.0)
        )
        self.assertLess(len(puck), 8 * 8 * 4)
        self.assertTrue((np.hypot(puck[:, 0], puck[:, 1]) <= 0.02 + 1e-9).all())
        jittered = sample_positions(rss.heap(size=(0.04, 0.04, 0.02), particle_spacing=0.01))
        again = sample_positions(rss.heap(size=(0.04, 0.04, 0.02), particle_spacing=0.01))
        np.testing.assert_array_equal(jittered, again, "the jitter is seeded from the name")
        self.assertFalse(np.array_equal(jittered, box))

    def test_a_body_weighs_its_material_over_the_region_it_fills(self) -> None:
        """The wipe's puddle: 6 mm deep at 2.5 mm is two layers of 3 mm, so a particle stands for 18.75 mm^3, not 15.6.

        Its mass and its radius both follow the cell: Newton's implicit MPM
        reads a particle's volume as ``8 r^3`` and its density as the mass over
        that, so the two together keep the solver's density the material's.
        """
        import newton

        import robosimstudio as rss
        from robosimstudio_materials.mpm.entity import EntityParticleBody

        spec = rss.liquid(name="ketchup", size=(0.10, 0.10, 0.006), material="ketchup")
        density = float(spec.material.density)
        self.assertEqual(3200, spec.particle_count)
        self.assertAlmostEqual(0.0025 * 0.0025 * 0.003, spec.particle_volume, places=15)
        self.assertAlmostEqual(density * 0.10 * 0.10 * 0.006, spec.mass, places=9)
        builder = newton.ModelBuilder()
        entity = EntityParticleBody(spec)
        entity.build(builder, None)
        mass = np.asarray(builder.particle_mass, dtype=np.float64)
        radius = np.asarray(builder.particle_radius, dtype=np.float64)
        self.assertEqual(3200, len(mass))
        self.assertAlmostEqual(density * 0.10 * 0.10 * 0.006, float(mass.sum()), places=9)
        np.testing.assert_allclose(8.0 * radius**3, spec.particle_volume, rtol=1e-9)
        np.testing.assert_allclose(mass / (8.0 * radius**3), density, rtol=1e-9)
        self.assertAlmostEqual(float(mass[0]), entity.particle_mass, places=12)
        self.assertAlmostEqual(float(radius[0]), entity.particle_radius, places=12)

    def test_the_drawn_radius_is_the_models(self) -> None:
        """The puddle's points and its surface take the 1.33 mm the model holds, not half the 2.5 mm spacing."""
        from types import SimpleNamespace

        import newton

        import robosimstudio as rss
        from robosimstudio_materials.mpm.entity import EntityParticleBody

        spec = rss.liquid(name="ketchup", size=(0.10, 0.10, 0.006), material="ketchup", render="surface")
        builder = newton.ModelBuilder()
        body = EntityParticleBody(spec)
        body.build(builder, None)
        model = builder.finalize(device="cpu")
        radius = float(model.particle_radius.numpy()[0])
        self.assertAlmostEqual(0.5 * (0.0025 * 0.0025 * 0.003) ** (1.0 / 3.0), radius, places=9)
        self.assertAlmostEqual(radius, body.handle(model).point_radius, places=9)
        body.solver = SimpleNamespace(
            config=SimpleNamespace(voxel_size=0.004), model=SimpleNamespace(device="cpu", world_count=1)
        )
        body._install_surface()
        self.assertAlmostEqual(radius, body.surface.radius, places=9)

    def test_a_mesh_is_not_counted_before_it_is_filled(self) -> None:
        """``size`` is ignored for a mesh, so a count, a mass or a summary off it would be the default box's."""
        import robosimstudio as rss
        from robosimstudio.specs.asset import AssetSpec
        from robosimstudio_materials.mpm import Heap

        spec = rss.heap(name="filled", shape="mesh", asset=AssetSpec(path="meshes/never_loaded.obj"))
        self.assertIsNone(spec.particle_count)
        self.assertIsNone(spec.mass)
        summary = Heap.from_spec(spec, name="filled_heap").summary
        self.assertEqual("a granular body of mesh shape", summary)
        self.assertEqual(
            "a granular body of box shape, 27648 particles", Heap.from_spec(rss.heap(), name="box_heap").summary
        )

    def test_the_material_fields_are_described_with_their_tiers(self) -> None:
        from robosimstudio.specs.params import REBUILD, RECAPTURE, iter_params, undescribed
        from robosimstudio_materials.mpm import MPM_KEYS, ParticleBodySpec, ParticleMaterialSpec

        self.assertEqual(("color",), tuple(undescribed(ParticleMaterialSpec)))
        tiers = {f.name: ui.tier for f, ui in iter_params(ParticleMaterialSpec)}
        self.assertEqual(REBUILD, tiers["density"])
        for key in MPM_KEYS:
            self.assertEqual(RECAPTURE, tiers[key], key)
        body = {f.name: ui.tier for f, ui in iter_params(ParticleBodySpec)}
        self.assertEqual(REBUILD, body["particle_spacing"])


class TestTheContainerMargins(unittest.TestCase):
    def test_a_container_is_matched_by_its_whole_name(self) -> None:
        """``cup`` takes its body's shapes and its world-fixed collision parts, never ``cup_holder``'s or ``cup_2``'s."""
        import newton

        import robosimstudio as rss
        from robosimstudio_materials.mpm.entity import DEFAULT_CONTAINER_MARGIN, EntityParticleBody

        builder = newton.ModelBuilder()
        cup, holder = builder.add_body(label="cup"), builder.add_body(label="cup_holder")
        shapes = {
            label: builder.add_shape_box(body=body, hx=0.01, hy=0.01, hz=0.01, label=label)
            for body, label in (
                (cup, "cup"),
                (cup, "cup_floor"),
                (-1, "cup_collision_00"),
                (holder, "cup_holder"),
                (-1, "cup_2"),
                (-1, "cup_2_collision_00"),
            )
        }
        pool = EntityParticleBody(rss.liquid(name="h", material="honey", containers=("cup",)))
        pool._apply_container_margins(builder)
        given = {label for label, s in shapes.items() if builder.shape_margin[s] == DEFAULT_CONTAINER_MARGIN}
        self.assertEqual({"cup", "cup_floor", "cup_collision_00"}, given)
        self.assertEqual(3, sum(builder.shape_margin[s] == 0.0 for s in shapes.values()))

    def test_a_static_containers_convex_hulls_are_its_own(self) -> None:
        """Newton labels the hulls past a decomposition's first ``<label>_convex_<i>``; each gets the margin.

        ``approximate_meshes(method="coacd")`` over a 100 mm U of three slabs
        welded to the world gave ``bowl`` and ``bowl_convex_1`` to
        ``bowl_convex_26``, all on body -1 (Newton 1.6.0's builder). Matched by
        the whole name alone, only the first hull had a margin and a liquid
        leaked through the other 26.
        """
        import newton

        import robosimstudio as rss
        from robosimstudio_materials.mpm.entity import DEFAULT_CONTAINER_MARGIN, EntityParticleBody

        builder = newton.ModelBuilder()
        labels = (
            "bowl",
            "bowl_convex_1",
            "bowl_convex_26",
            "bowl_collision",
            "bowl_collision_01_convex_2",
            "bowl_2",
            "bowl_2_convex_1",
            "bowl_holder_convex_1",
        )
        shapes = {label: builder.add_shape_box(body=-1, hx=0.01, hy=0.01, hz=0.01, label=label) for label in labels}
        pool = EntityParticleBody(rss.liquid(name="h", material="honey", containers=("bowl",)))
        pool._apply_container_margins(builder)
        given = {label for label, s in shapes.items() if builder.shape_margin[s] == DEFAULT_CONTAINER_MARGIN}
        self.assertEqual(set(labels[:5]), given)


class TestTheCupMesh(unittest.TestCase):
    def test_the_mesh_is_written_beside_its_name_and_renamed_onto_it(self) -> None:
        """Two processes building one cup share the cache file; neither may read it half written."""
        import os
        import pathlib
        import tempfile
        from unittest import mock

        import trimesh

        from robosimstudio_materials.liquid import cup

        export = trimesh.Trimesh.export
        writes = []

        def recorded(mesh, file_obj=None, *args, **kwargs):
            path = pathlib.Path(file_obj)
            writes.append((path, sorted(p.name for p in path.parent.iterdir())))
            return export(mesh, file_obj, *args, **kwargs)

        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {"ROBOSIMSTUDIO_CACHE": root}):
            with mock.patch.object(trimesh.Trimesh, "export", recorded):
                spec = cup(inner=(0.031, 0.032, 0.033), wall=0.005)
            final = pathlib.Path(spec.asset.path)
            ((written, present),) = writes
            self.assertNotEqual(final, written, "the mesh was exported straight onto the name others read")
            self.assertEqual(final.parent, written.parent, "a rename is atomic only within one directory")
            self.assertNotIn(final.name, present, "the final name existed before the mesh was whole")
            self.assertEqual([final.name], sorted(p.name for p in final.parent.iterdir()), "no temporary file is left")
            self.assertGreater(len(trimesh.load(final, force="mesh").faces), 0)


class TestTheRecords(unittest.TestCase):
    def test_a_record_carries_the_specs_domain_and_builds_the_spec(self) -> None:
        import robosimstudio as rss
        from robosimstudio_materials.mpm import Dough, Heap, Liquid

        spec = rss.heap(name="rice")
        record = Heap.from_spec(spec, name="rice_heap")
        self.assertEqual("granular", record.sim.domain)
        self.assertIs(spec, record.build())
        self.assertEqual("liquid", Liquid.from_spec(rss.liquid(name="h"), name="pool").sim.domain)
        self.assertEqual("dough", Dough.from_spec(rss.dough(name="d"), name="lump").sim.domain)
        with self.assertRaises(TypeError):
            Liquid.from_spec(spec, name="not_a_liquid")
        with self.assertRaises(TypeError):
            Heap.from_spec(rss.box(name="cube"), name="not_a_heap")

    def test_the_atoms_are_catalogued_on_the_granular_layout(self) -> None:
        from robosimstudio.assetlib.semantics.atoms import ATOMS
        from robosimstudio.assetlib.semantics.handlers import dispatch

        for name in ("fill_fraction", "mass_in_region", "spill_fraction", "heap_height", "spread"):
            with self.subTest(atom=name):
                self.assertIn(name, ATOMS)
                spec = ATOMS.get(name)
                self.assertEqual(("granular",), spec.args[0].substrates)
                self.assertEqual([], spec.problems())
        for layout in ("granular", "liquid", "dough"):
            self.assertEqual("fill_fraction", dispatch("fill_fraction", ("", "region"), (layout, "")).atom)
            self.assertEqual("containment", dispatch("containment", ("", "region"), (layout, "")).atom)

    def test_the_measures_on_arrays(self) -> None:
        from robosimstudio_materials.mpm.measures import fill_fraction, heap_height, spill_fraction, spread

        pts = np.array([[0.0, 0.0, 0.01], [0.05, 0.0, 0.01], [0.5, 0.0, 0.002], [0.5, 0.0, 0.2]])
        lo, hi = np.array([-0.1, -0.1, 0.0]), np.array([0.1, 0.1, 0.05])
        self.assertEqual(0.5, fill_fraction(pts, lo, hi))
        self.assertEqual(
            0.25, spill_fraction(pts, lo, hi, floor=0.0, band=0.005), "the one lying outside, not the one in the air"
        )
        self.assertEqual(0.5, spill_fraction(pts, lo, hi), "without a band, both that are outside, at any height")
        with self.assertRaises(TypeError):
            spill_fraction(pts, lo, hi, band=0.005)
        self.assertAlmostEqual(0.2 - 0.001, heap_height(pts, 0.001, quantile=1.0))
        self.assertAlmostEqual(4.0, spread(np.array([[0, 0, 0], [2, 2, 0]]), np.array([[0, 0, 0], [1, 1, 0]])))


class TestTheSiblingLaw(unittest.TestCase):
    """The kinds import the shared package and the library, and nothing else of the materials."""

    @staticmethod
    def _imports(path: Path) -> list[str]:
        found: list[str] = []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                found.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                found.append("." * node.level + (node.module or ""))
        return found

    def test_no_particle_kind_imports_another_kind(self) -> None:
        others = ("cloth", "soft", "rod", "paper")
        files = [MATERIALS / f"{kind}.py" for kind in KINDS] + sorted((MATERIALS / "mpm").glob("*.py"))
        for path in files:
            with self.subTest(file=path.name):
                for module in self._imports(path):
                    for other in (*others, *KINDS):
                        self.assertFalse(
                            module in (f".{other}", f"robosimstudio_materials.{other}")
                            or module.startswith((f".{other}.", f"robosimstudio_materials.{other}.")),
                            f"{path.name} imports {module}",
                        )
                    self.assertFalse(
                        module.startswith(("robosimstudio_studio", "robosimstudio_bench", "robosimstudio_policy")),
                        f"{path.name} imports {module}",
                    )

    def test_the_shared_package_is_not_a_kind(self) -> None:
        from robosimstudio.scene.registry import ENTITIES
        from robosimstudio.specs.physics import list_domains

        self.assertNotIn("mpm", list_domains())
        self.assertNotIn("mpm", ENTITIES.names())


if __name__ == "__main__":
    unittest.main()
