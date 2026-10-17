# SPDX-License-Identifier: Apache-2.0
"""The ``mpm`` backend as registered knowledge: what the composer reads of it before anything is built.

Newton's implicit MPM is a backend alias like ``vbd`` and ``mujoco``, and the
composer learns everything it needs from the registration -- that it owns
particles, that it detects its own contacts, that it is stepped in place, what
its parameters are and what changing each costs -- so these claims are checked
against the registry, on the CPU, with no solver constructed. The one object
that is constructed is :class:`~robosimstudio.engines.newton.mpm.SolverMPM`
asked for a parameter it does not have, which it refuses before it touches a
model. ``tests/test_mpm_island`` builds the island on a GPU.
"""

from __future__ import annotations

import unittest

TIER = "fast"
"""No GPU: registries and one constructor that raises before it needs one."""


class TestTheRegistration(unittest.TestCase):
    def test_the_alias_resolves_to_the_adapter_with_its_facts(self) -> None:
        from robosimstudio.specs.physics import get_backend

        info = get_backend("mpm")
        self.assertEqual("robosimstudio.engines.newton.mpm.SolverMPM", info.cls)
        self.assertEqual(("particles",), info.supports, "the MPM owns particles and nothing else")
        self.assertTrue(info.native_contacts, "its colliders are its view's shapes; it reads no contact buffer")
        self.assertTrue(info.steps_in_place, "the coupler steps it in place, Newton's contract for it")
        self.assertFalse(info.host_engine)
        self.assertFalse(info.maximal_coordinates)

    def test_steps_in_place_is_a_backend_fact_the_others_do_not_claim(self) -> None:
        from robosimstudio.specs.physics import get_backend

        for alias in ("mujoco", "vbd", "xpbd", "featherstone", "kinematic"):
            self.assertFalse(get_backend(alias).steps_in_place, alias)

    def test_steps_in_place_takes_part_in_equality(self) -> None:
        """Identical re-registration is accepted and a differing one refused, so the field must count."""
        from dataclasses import replace

        from robosimstudio.specs.physics import get_backend

        info = get_backend("mpm")
        self.assertEqual(info, replace(info))
        self.assertNotEqual(info, replace(info, steps_in_place=False))

    def test_the_schema_names_every_parameter_the_probes_set(self) -> None:
        from robosimstudio.specs.params import REBUILD, RECAPTURE
        from robosimstudio.specs.physics import get_backend

        schema = get_backend("mpm").schema
        expected = {
            "voxel_size",
            "grid_type",
            "grid_padding",
            "max_active_cell_count",
            "max_lower_node_count",
            "max_upper_node_count",
            "max_iterations",
            "tolerance",
            "strain_basis",
            "velocity_basis",
            "collider_basis",
            "collider_velocity_mode",
            "transfer_scheme",
            "warmstart_mode",
            "critical_fraction",
            "air_drag",
            "separate_worlds",
            "project_outside",
            "project_gap",
            "step_hz",
        }
        self.assertLessEqual(expected, set(schema))
        # The two plain instance attributes the solver reads every step are the
        # live-ish ones; everything else is consumed at construction.
        self.assertEqual(RECAPTURE, schema["max_iterations"].ui.tier)
        self.assertEqual("attr:max_iterations", schema["max_iterations"].apply)
        self.assertEqual(RECAPTURE, schema["tolerance"].ui.tier)
        self.assertEqual("attr:tolerance", schema["tolerance"].apply)
        self.assertEqual(REBUILD, schema["voxel_size"].ui.tier)
        self.assertIn("pic", schema["collider_basis"].ui.choices)
        self.assertIn("CANNOT be graph-captured", schema["collider_basis"].ui.doc, "the pic trap is written down")

    def test_the_defaults_are_the_probes_choices(self) -> None:
        from robosimstudio.specs.physics import get_backend

        defaults = dict(get_backend("mpm").defaults)
        self.assertEqual(0.004, defaults["voxel_size"])
        self.assertEqual("sparse", defaults["grid_type"])
        self.assertEqual(1 << 15, defaults["max_active_cell_count"])
        self.assertEqual(0, defaults["grid_padding"])
        self.assertEqual(4096, defaults["max_lower_node_count"])
        self.assertEqual(256, defaults["max_upper_node_count"])
        self.assertEqual(50, defaults["max_iterations"])
        self.assertEqual(1.0e-4, defaults["tolerance"])
        self.assertEqual("P0", defaults["strain_basis"])
        self.assertEqual("S2", defaults["collider_basis"])
        self.assertEqual("forward", defaults["collider_velocity_mode"])
        self.assertTrue(defaults["project_outside"])
        self.assertEqual(60.0, defaults["step_hz"])

    def test_the_max_iterations_and_tolerance_are_plain_attributes_of_the_solver(self) -> None:
        """``attr:`` application sets them with ``setattr``; the class must read them as instance attributes."""
        import inspect

        from newton.solvers import SolverImplicitMPM

        source = inspect.getsource(SolverImplicitMPM.__init__)
        self.assertIn("self.max_iterations = config.max_iterations", source)
        self.assertIn("self.tolerance = float(config.tolerance)", source)


class TestTheAdapterRefusesWhatItDoesNotKnow(unittest.TestCase):
    def test_an_unknown_parameter_is_a_type_error_naming_it(self) -> None:
        from robosimstudio.engines.newton.mpm import SolverMPM

        with self.assertRaises(TypeError) as caught:
            SolverMPM(model=None, voxel_sise=0.004)
        self.assertIn("voxel_sise", str(caught.exception))
        self.assertIn("step_hz", str(caught.exception), "the message lists what is accepted")

    def test_the_composer_wraps_it_as_a_composition_error(self) -> None:
        from robosimstudio.scene.physics import CompositionError
        from robosimstudio.scene.physics.backends import build_solver
        from robosimstudio.specs import backend

        with self.assertRaises(CompositionError) as caught:
            build_solver(backend("mpm", voxel_sise=0.004), None)
        self.assertIn("voxel_sise", str(caught.exception))


class TestGenericRoutesTheParticleMaterials(unittest.TestCase):
    def test_the_three_domains_share_one_mpm_backend(self) -> None:
        from robosimstudio.specs.physics import GENERIC

        specs = [GENERIC.backend_for(name) for name in ("granular", "liquid", "dough")]
        self.assertTrue(all(spec.solver == "mpm" for spec in specs))
        self.assertEqual(1, len(set(specs)), "equal specs are one island; the proxy coupler takes at most two")

    def test_a_cloth_still_falls_through_to_vbd(self) -> None:
        from robosimstudio.specs.physics import GENERIC

        self.assertEqual("vbd", GENERIC.backend_for("cloth").solver)


class TestTheInPlaceRateMustFitAControlStep(unittest.TestCase):
    """The host-side stride is baked into a captured graph, so the composer refuses a rate that does not fit."""

    def _island(self, **params):
        from robosimstudio.scene.physics.islands import Island
        from robosimstudio.specs import backend
        from robosimstudio.specs.physics import get_backend

        return Island(name="mpm", backend=backend("mpm", **params), solver_class=get_backend("mpm").cls)

    def test_the_rates_that_divide_the_substep_rate_pass(self) -> None:
        from robosimstudio.scene.physics.compose import _check_in_place_rates
        from robosimstudio.specs.physics import GENERIC

        for hz in (60.0, 120.0, 240.0, 480.0, 960.0, 1920.0):
            _check_in_place_rates(GENERIC, [self._island(step_hz=hz)])
        _check_in_place_rates(GENERIC, [self._island()])  # the registered default

    def test_a_rate_that_leaves_a_fraction_is_refused_naming_the_ones_that_fit(self) -> None:
        from robosimstudio.scene.physics import CompositionError
        from robosimstudio.scene.physics.compose import _check_in_place_rates
        from robosimstudio.specs.physics import GENERIC

        with self.assertRaises(CompositionError) as caught:
            _check_in_place_rates(GENERIC, [self._island(step_hz=320.0)])
        message = str(caught.exception)
        self.assertIn("320", message)
        self.assertIn("'120'", message)
        self.assertIn("'960'", message)


if __name__ == "__main__":
    unittest.main()
