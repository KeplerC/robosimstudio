# SPDX-License-Identifier: Apache-2.0
"""The ``water`` kind without its engine: the spec, the measure and the routing.

What a pour does is ``tests/test_pour_water.py``'s, in the virtual environment that
holds the ``[genesis]`` extra. What needs no engine is here and gates every commit:
the lattice a block of water is built from, what ``fill_level`` reads off a set of
points, and that the arrangement sends ``water`` to Genesis at the precision and
substeps the kind measured.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "fast"
"""No GPU, no engine, no scene."""


class TestTheBlockOfWater(unittest.TestCase):
    def test_the_default_block_is_twelve_particles_a_side(self) -> None:
        from robosimstudio_materials.water import WaterSpec

        spec = WaterSpec()
        self.assertEqual((12, 12, 12), spec.counts)
        self.assertEqual(1728, spec.particle_count)
        self.assertAlmostEqual(0.110592, spec.mass, places=9, msg="1,728 particles of 4 mm at 1000 kg/m^3")

    def test_the_lattice_stands_on_its_bottom_face_centre(self) -> None:
        from robosimstudio_materials.water import WaterSpec, lattice

        points = lattice(WaterSpec())
        self.assertEqual((1728, 3), points.shape)
        self.assertAlmostEqual(0.002, float(points[:, 2].min()))
        np.testing.assert_allclose(points[:, :2].mean(axis=0), [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(points[:, :2].max(axis=0), [0.022, 0.022], atol=1e-12)

    def test_bad_values_are_refused(self) -> None:
        from robosimstudio_materials.water import WaterMaterialSpec, WaterSpec

        with self.assertRaises(ValueError):
            WaterSpec(render="mesh")
        with self.assertRaises(ValueError):
            WaterSpec(size=(0.05, 0.0, 0.05))
        with self.assertRaises(ValueError):
            WaterMaterialSpec(viscosity=-1.0e-4)

    def test_a_preset_named_to_the_spec_is_its_material(self) -> None:
        """``WaterSpec(material="oil")`` weighs oil and records as oil, as ``rss.water(material="oil")`` does."""
        from robosimstudio_materials.water import OIL, Water, WaterSpec

        spec = WaterSpec(material="oil")
        self.assertIs(OIL, spec.material)
        self.assertAlmostEqual(1728 * 0.004**3 * 900.0, spec.mass, places=9)
        self.assertIn("100 g", Water.from_spec(spec, name="oil").summary)
        with self.assertRaisesRegex(TypeError, "honey"):
            WaterSpec(material="honey")

    def test_the_engine_reads_what_the_spec_states(self) -> None:
        from robosimstudio.engines.genesis.sph import SPH_CONSUMES
        from robosimstudio_materials.water import OIL, EntityWater, water

        actor = EntityWater(water(material="oil")).host_actor()
        self.assertEqual("sph", actor.kind)
        self.assertLessEqual(set(actor.params), set(SPH_CONSUMES))
        self.assertEqual(OIL.viscosity, actor.params["viscosity"])


class TestFillLevel(unittest.TestCase):
    """A lattice that fills its box reads the height its count predicts; points outside lower only the share."""

    LO, HI, H = np.array([0.0, 0.0, 0.0]), np.array([0.06, 0.06, 0.055]), 0.004

    def _column(self, layers: int) -> np.ndarray:
        axes = [(np.arange(n) + 0.5) * self.H for n in (15, 15, layers)]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    def test_a_full_lattice_reads_its_own_height(self) -> None:
        from robosimstudio_materials.water import fill_level

        out = fill_level(self._column(8), self.LO, self.HI, self.H)
        self.assertEqual(1.0, out["fraction"])
        self.assertAlmostEqual(0.032, out["expected"])
        self.assertAlmostEqual(0.032, out["height"])

    def test_water_outside_lowers_the_share_not_the_height(self) -> None:
        from robosimstudio_materials.water import fill_level

        inside = self._column(8)
        spilled = inside[:600] + np.array([0.2, 0.0, 0.0])
        out = fill_level(np.concatenate([inside, spilled]), self.LO, self.HI, self.H)
        self.assertAlmostEqual(1800 / 2400, out["fraction"])
        self.assertAlmostEqual(0.032, out["height"])

    def test_an_empty_container_reads_zero(self) -> None:
        from robosimstudio_materials.water import fill_level

        out = fill_level(self._column(8) + np.array([0.5, 0.0, 0.0]), self.LO, self.HI, self.H)
        self.assertEqual({"fraction": 0.0, "height": 0.0, "expected": 0.0}, out)


class TestTheArrangement(unittest.TestCase):
    def test_water_goes_to_genesis_and_the_cup_to_the_arms_island(self) -> None:
        import robosimstudio as rss

        table = rss.WATER_60HZ
        self.assertEqual("genesis", table.backend_for("water").solver)
        self.assertEqual(20, table.backend_for("water").substeps)
        self.assertEqual("32", dict(table.backend_for("water").params)["precision"])
        self.assertEqual("mujoco", table.backend_for("rigid").solver)
        self.assertEqual("mujoco", table.backend_for("articulated").solver)

    def test_only_genesis_builds_water_and_any_other_engine_refuses_it_by_name(self) -> None:
        """On GENERIC the kind falls to VBD, whose settle died in an illegal memory access: refused at build."""
        import newton

        import robosimstudio as rss
        from robosimstudio.scene.registry import resolve_entity_class
        from robosimstudio_materials.water import EntityWaterGenesis

        spec = rss.water(name="spill", size=(0.008, 0.008, 0.008))
        with self.assertRaisesRegex(NotImplementedError, r"spill.*backend\('genesis'\).*rss\.WATER_60HZ"):
            resolve_entity_class(spec, rss.GENERIC)(spec).build(newton.ModelBuilder(), None)
        cls = resolve_entity_class(spec, rss.WATER_60HZ)
        self.assertIs(EntityWaterGenesis, cls)
        builder = newton.ModelBuilder()
        cls(spec).build(builder, None)
        active = [int(f) & int(newton.ParticleFlags.ACTIVE) for f in builder.particle_flags]
        self.assertTrue(builder.particle_count == 8 and not any(active), "Genesis's water is left to Genesis")

    def test_each_preset_takes_the_substeps_it_states(self) -> None:
        """Water holds at the arrangement's 20 a control step; oil's cohesion asks the 35 its docstring states."""
        import math

        import robosimstudio as rss
        from robosimstudio.engines.genesis.sph import longest_substep
        from robosimstudio_materials.water import EntityWater, water

        step = 1.0 / rss.WATER_60HZ.control_hz
        for material, need in (("water", 20), ("oil", 35)):
            longest, _ = longest_substep(EntityWater(water(material=material)).host_actor().params)
            self.assertEqual(need, math.ceil(step / longest - 1.0e-9), material)

    def test_the_flat_names_resolve(self) -> None:
        import robosimstudio as rss
        from robosimstudio_materials import water

        self.assertIs(water.water, rss.water)
        self.assertIs(water.WaterSpec, rss.WaterSpec)


if __name__ == "__main__":
    unittest.main()
