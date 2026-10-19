# SPDX-License-Identifier: Apache-2.0
"""The Genesis backend, with Genesis not installed.

Everything here runs with the ``[genesis]`` extra absent, on purpose --- it is
the half of the story ``test_genesis.py`` cannot tell, and the half that matters
most for a checkout that will never build a Genesis island. The claim being
checked is the extension seam's: registering a backend imports nothing, an alias
resolves from a cold process, and the engine is asked for only when a scene
actually stages one --- not at import, and never as a silent fallback.

The arithmetic that decides whether the coupling is *correct* is also here,
because none of it needs an engine: the mass reconciliation, the quaternion
convention, and which mirrors may carry a reaction at all.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import subprocess
import sys
import unittest

import numpy as np

TIER = "fast"
"""No GPU, no engine, no scene."""


def _geom(**kw):
    from robosimstudio.engines.geometry import ShapeGeometry

    base = {
        "shape": 0,
        "body": 3,
        "kind": "box",
        "scale": np.ones(3),
        "local": np.array([0.0, 0, 0, 0, 0, 0, 1.0]),
        "mu": 0.5,
        "two_way": False,
    }
    return ShapeGeometry(**{**base, **kw})


class TestRegistrationImportsNothing(unittest.TestCase):
    def test_the_alias_resolves_and_is_a_host_engine(self):
        from robosimstudio.specs.physics import get_backend

        info = get_backend("genesis")
        self.assertTrue(info.host_engine)
        self.assertEqual(info.supports, ("particles",))
        self.assertEqual(info.cls, "robosimstudio.engines.genesis.solver.SolverGenesis")

    def test_it_resolves_from_a_cold_process(self):
        """``GENERIC`` may name an alias nothing has imported; ``get_backend``
        scans the engine bundles once and retries before giving up."""
        code = "from robosimstudio.specs.physics import get_backend\nassert get_backend('genesis').host_engine\n"
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_importing_the_bundle_does_not_import_the_engine(self):
        """The registration is a dotted-path string, so it costs no engine."""
        code = (
            "import sys, robosimstudio.engines.genesis as bundle\n"
            # Was `'genesis' not in sys.modules or sys.modules['genesis'] is not
            # None`, which is true whether or not the engine was imported. The
            # generic form of this law is TestNoBundleImportsItsEngine in
            # tests/test_engine_roots.py, over every bundle and every SDK.
            "assert 'genesis' not in sys.modules, 'the bundle imported the engine at module scope'\n"
            "assert bundle.BACKEND.host_engine\n"
            "import robosimstudio.engines.genesis.bridge  # the adapter too\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    @unittest.skipIf(importlib.util.find_spec("genesis") is not None, "genesis is installed here")
    def test_a_missing_wheel_is_refused_by_name(self):
        from robosimstudio.engines.genesis.runtime import ensure_initialized

        with self.assertRaises(ImportError) as caught:
            ensure_initialized()
        message = str(caught.exception)
        self.assertIn("[genesis]", message, "the refusal names the extra to install")
        self.assertIn("genesis-world", message)


class TestAPrecisionConflictNamesWhoseHarvestItHurts(unittest.TestCase):
    """``gs.init`` is first-caller-wins, so a later island asking for another precision is told what it got."""

    def _later(self, first: str, precision: str, coupler: str) -> str:
        from unittest import mock

        from robosimstudio.engines.genesis import runtime

        with (
            mock.patch.object(runtime, "require_genesis", return_value=object()),
            mock.patch.dict(runtime._ENGINE, {"precision": first}),
            self.assertLogs(runtime.log, "DEBUG") as logs,
        ):
            runtime.ensure_initialized(precision, coupler=coupler)
        return "\n".join(logs.output)

    def test_an_ipc_island_after_a_liquid_is_told_its_harvest_reads_zero(self):
        self.assertIn("IPC two-way harvest reads ZERO", self._later("32", "64", "ipc"))

    def test_a_liquid_asking_for_64_after_32_is_not_blamed_on_the_ipc_harvest(self):
        said = self._later("32", "64", "legacy")
        self.assertNotIn("IPC", said)
        self.assertIn("liquid island, which asks for 64, steps its SPH at float32", said)

    def test_a_liquid_after_an_ipc_island_is_told_it_runs_slow(self):
        self.assertIn("8x slower", self._later("64", "32", "legacy"))


class TestTheCouplingArithmetic(unittest.TestCase):
    """The numbers that decide whether the two-way coupling is right."""

    def test_cloth_density_divides_out_the_measured_factor(self):
        """Genesis's FEM.Cloth masses twice what rho*thickness*area predicts.

        Measured across a doubled density, a doubled thickness and a halved mesh
        spacing, constant to three figures. Unreconciled, the wrench fed back to
        the arm is twice what it should be -- and it reads as "the cloth feels
        heavy" rather than as an error, which is why this is a test and not a
        comment.
        """
        from robosimstudio.engines.genesis.bridge import CLOTH_MASS_FACTOR, cloth_density

        self.assertEqual(CLOTH_MASS_FACTOR, 2.0)
        areal, thickness = 0.4, 0.002
        rho = cloth_density(areal, thickness)
        simulated_areal = rho * thickness * CLOTH_MASS_FACTOR
        self.assertAlmostEqual(simulated_areal, areal, places=9, msg="the sheet weighs what Newton says")

    def test_cloth_density_survives_a_zero_thickness(self):
        from robosimstudio.engines.genesis.bridge import cloth_density

        self.assertGreater(cloth_density(0.4, 0.0), 0.0)

    def test_the_quaternion_convention_flips_at_this_boundary(self):
        """Newton is xyzw and Genesis is wxyz. A leaked swap is a plausible
        wrong rotation rather than an error, so it is checked here."""
        from robosimstudio.engines.genesis.bridge import wxyz

        np.testing.assert_allclose(wxyz([0.0, 0.0, 0.0, 1.0]), [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(wxyz([0.1, 0.2, 0.3, 0.4]), [0.4, 0.1, 0.2, 0.3])

    def test_only_a_free_two_way_mirror_can_carry_a_reaction(self):
        """A fixed link never deviates from its aim, so its reaction is
        identically zero -- measured, with the cloth visibly resting on it. A
        two-way mirror must therefore be free-based and gravity-compensated."""
        from robosimstudio.engines.genesis.bridge import mirror_role

        coup, fixed, gravcomp = mirror_role(_geom(two_way=True, body=3))
        self.assertEqual(coup, "two_way_soft_constraint")
        self.assertFalse(fixed, "a fixed mirror reads zero force")
        self.assertEqual(gravcomp, 1.0, "or it sags between prescriptions and reads a phantom load")

    def test_everything_else_is_a_fixed_kinematic_collider(self):
        from robosimstudio.engines.genesis.bridge import mirror_role

        for geom in (_geom(two_way=False, body=3), _geom(two_way=True, body=-1)):
            coup, fixed, gravcomp = mirror_role(geom)
            self.assertEqual(coup, "ipc_only")
            self.assertTrue(fixed)
            self.assertEqual(gravcomp, 0.0, "Genesis refuses gravity_compensation with ipc_only")

    def test_the_tet_surface_is_the_faces_used_once(self):
        """A tet body reaches Genesis as a surface plus its own lattice; the
        surface has to be the boundary, not every face."""
        from robosimstudio.engines.genesis.bridge import _surface_of

        one = np.array([[0, 1, 2, 3]])
        self.assertEqual(len(_surface_of(one)), 4, "a single tet is all boundary")
        two = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
        self.assertEqual(len(_surface_of(two)), 6, "two tets share one face, so 8 - 2")


class _Bridge:
    """What :func:`write_obj` reads off a bridge: its recipe, and a lifetime to hang the files on."""

    def __init__(self) -> None:
        self.recipe = {"island": "genesis"}


def _mesh(n: int):
    return np.full((n, 3), float(n)), np.zeros((n, 3), dtype=int)


def _vertices_in(path: str) -> int:
    with open(path) as handle:
        return sum(line.startswith("v ") for line in handle)


class TestTheMirrorFilesAreTheBridgesOwn(unittest.TestCase):
    def test_two_bridges_of_one_island_write_apart(self):
        """Two bridges of one island name -- two shards, two live pages -- each read back their own mesh."""
        from robosimstudio.engines.genesis.bridge import write_obj

        geom = _geom(kind="mesh", shape=3)
        one, two = _Bridge(), _Bridge()
        path_one, path_two = write_obj(one, geom, _mesh(4)), write_obj(two, geom, _mesh(5))
        self.assertNotEqual(path_one, path_two)
        self.assertEqual((_vertices_in(path_one), _vertices_in(path_two)), (4, 5))
        self.assertEqual(os.path.basename(path_one), "mirror-3.obj", "Genesis names the link after the file")

    def test_a_bridges_mirrors_go_with_it(self):
        from robosimstudio.engines.genesis.bridge import write_obj

        bridge = _Bridge()
        first = write_obj(bridge, _geom(kind="mesh", shape=3), _mesh(4))
        second = write_obj(bridge, _geom(kind="mesh", shape=4), _mesh(4))
        root = os.path.dirname(first)
        self.assertEqual(os.path.dirname(second), root, "one directory per bridge, not per mirror")
        del bridge
        gc.collect()
        self.assertFalse(os.path.exists(root))


class TestTheLiquidCouplerArithmetic(unittest.TestCase):
    """The legacy coupler's island: which coupler an island gets, and the numbers its liquid and mirrors are given."""

    def test_an_island_picks_its_coupler_from_its_actors(self):
        from robosimstudio.engines.genesis.bridge import coupler_for

        self.assertEqual("legacy", coupler_for({"actors": [{"kind": "sph", "name": "water"}]}))
        self.assertEqual("ipc", coupler_for({"actors": [{"kind": "shell", "name": "towel"}]}))
        self.assertEqual("ipc", coupler_for({"actors": []}))

    def test_a_liquid_beside_a_cloth_is_refused_by_name(self):
        """A Genesis scene holds one coupler, and an SPH body under the IPC coupler does not compile."""
        from robosimstudio.engines.genesis.bridge import coupler_for

        recipe = {"island": "host", "actors": [{"kind": "sph", "name": "water"}, {"kind": "tet", "name": "sponge"}]}
        with self.assertRaises(ValueError) as caught:
            coupler_for(recipe)
        for word in ("'sph'", "'tet'", "one coupler", "backend('genesis'"):
            self.assertIn(word, str(caught.exception))

    def test_the_liquid_weighs_the_volume_it_fills(self):
        """Genesis masses a particle at 0.8 of ``size**3 * rho``; density and stiffness are divided by it together,
        so the pressure's acceleration (stiffness over rest density) is the material's own."""
        from robosimstudio.engines.genesis.sph import GENESIS_MASS_VOLUME, liquid_args

        args = liquid_args({"density": 1000.0, "stiffness": 1.0e4, "viscosity": 5.0e-4, "surface_tension": 0.01})
        self.assertAlmostEqual(1000.0, args["rho"] * GENESIS_MASS_VOLUME)
        self.assertAlmostEqual(1.0e4, args["stiffness"] * GENESIS_MASS_VOLUME)
        self.assertAlmostEqual(args["stiffness"] / args["rho"], 10.0)
        self.assertEqual((5.0e-4, 0.01), (args["mu"], args["gamma"]))

    def test_a_mirror_turns_at_a_rate_in_its_own_frame(self):
        """Genesis integrates a free joint's angular dofs in the body's frame. A body yawed 90 degrees about z and
        turned about world x has its rate on its own -y axis."""
        from robosimstudio.core.math import quat_from_axis_angle, quat_multiply
        from robosimstudio.engines.genesis.sph import _angular_velocity

        start = quat_from_axis_angle((0.0, 0.0, 1.0), np.pi / 2)
        end = quat_multiply(quat_from_axis_angle((1.0, 0.0, 0.0), 0.02), start)
        np.testing.assert_allclose(_angular_velocity(start, end, 0.01), [0.0, -2.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(_angular_velocity(start, start, 0.01), np.zeros(3))


class TestTheSolverIsAHostSolver(unittest.TestCase):
    def test_it_inherits_the_shared_contract(self):
        from robosimstudio.engines.genesis.solver import SolverGenesis
        from robosimstudio.engines.host import SolverHost

        self.assertTrue(issubclass(SolverGenesis, SolverHost))
        self.assertTrue(SolverGenesis.host_engine)
        for name in (
            "step",
            "notify_model_changed",
            "update_contacts",
            "reset",
            "attach_pipeline",
            "capture_rest",
            "host_tick",
        ):
            self.assertTrue(hasattr(SolverGenesis, name), f"the composer calls {name}")

    def test_open_is_the_only_thing_it_writes(self):
        """The point of SolverHost: a second engine costs one method."""
        from robosimstudio.engines.genesis.solver import SolverGenesis

        own = {n for n, v in vars(SolverGenesis).items() if callable(v)}
        self.assertEqual(own, {"open"}, f"SolverGenesis defines more than open(): {sorted(own)}")


if __name__ == "__main__":
    unittest.main()
