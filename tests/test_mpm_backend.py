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
        self.assertEqual(0.03, defaults["project_gap"])
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


class TestTheHarvestCoversTheStepTaken(unittest.TestCase):
    """The proxy coupler harvests once per coupled step at its ``dt``; the entry calls the solver ``substeps`` times in it.

    Newton's own step and harvest are stubbed: the step records an impulse of
    one newton times its own length, as a load at rest would put on a
    collider, and the harvest divides the last impulse by the ``dt`` it is
    given. A wrench of exactly 1 N is the harvest dividing by the length of
    the step that made the impulse.
    """

    def _wrenches(
        self, *, substeps: int, step_hz: float, coupled_steps: int, reset_after: int | None = None
    ) -> list[float]:
        from types import SimpleNamespace
        from unittest import mock

        import numpy as np
        import warp as wp
        from newton.solvers import SolverImplicitMPM

        from robosimstudio.engines.newton.mpm import SolverMPM

        impulse = {"n_s": 0.0}

        def init(self, model, config, **kwargs) -> None:
            self.model = model

        def step(self, state_in, state_out, control, contacts, dt) -> None:
            impulse["n_s"] = 1.0 * dt

        def harvest(self, body_local_to_proxy_global, out_body_f, *, dt, **kwargs) -> None:
            out_body_f.assign(np.full((1, 6), impulse["n_s"] / dt, dtype=np.float32))

        def reset(self, state, world_mask=None, flags=None) -> None:
            impulse["n_s"] = 0.0

        dt = 1.0 / 960.0  # GENERIC's substep: sixteen at 60 Hz
        with (
            mock.patch.object(SolverImplicitMPM, "__init__", init),
            mock.patch.object(SolverImplicitMPM, "step", step),
            mock.patch.object(SolverImplicitMPM, "coupling_harvest_proxy_wrenches", harvest),
            mock.patch.object(SolverImplicitMPM, "reset", reset),
        ):
            solver = SolverMPM(SimpleNamespace(device=SimpleNamespace(is_capturing=False)), step_hz=step_hz)
            state = SimpleNamespace()
            out = wp.zeros(1, dtype=wp.spatial_vector, device="cpu")
            read = []
            for i in range(coupled_steps):
                if i == reset_after:
                    solver.reset(state)
                for _ in range(substeps):
                    solver.step(state, state, None, None, dt / substeps)
                solver.coupling_harvest_proxy_wrenches(None, out, dt=dt, state=state, contacts=None)
                read.append(float(out.numpy()[0][2]))
        return read

    def test_one_substep_harvests_on_the_step_and_holds_between(self) -> None:
        read = self._wrenches(substeps=1, step_hz=60.0, coupled_steps=32)
        self.assertEqual([0.0] * 15 + [1.0] * 17, [round(v, 6) for v in read])

    def test_two_substeps_divide_by_the_step_not_twice_it(self) -> None:
        """``backend("mpm", substeps=2)`` read half of every collider wrench, a cup of honey half its weight."""
        read = self._wrenches(substeps=2, step_hz=120.0, coupled_steps=16)
        self.assertEqual([0.0] * 7 + [1.0] * 9, [round(v, 6) for v in read])

    def test_a_step_on_a_call_other_than_the_last_is_harvested(self) -> None:
        """Three substeps and a stride of two: the first coupled step's MPM step falls on its second call."""
        read = self._wrenches(substeps=3, step_hz=1440.0, coupled_steps=4)
        self.assertEqual([1.0] * 4, [round(v, 6) for v in read])

    def test_a_reset_holds_nothing_of_the_episode_before(self) -> None:
        """After a reset the calls before the first MPM step read zero, not the last episode's last contact."""
        read = self._wrenches(substeps=1, step_hz=60.0, coupled_steps=48, reset_after=32)
        self.assertEqual([0.0] * 15 + [1.0] * 17 + [0.0] * 15 + [1.0], [round(v, 6) for v in read])


if __name__ == "__main__":
    unittest.main()
