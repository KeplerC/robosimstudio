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
import math
import os
import subprocess
import sys
import unittest

import numpy as np

TIER = "fast"
"""No GPU, no engine, no scene."""


class _Arr:
    def __init__(self, a) -> None:
        self._a = np.asarray(a)

    def numpy(self) -> np.ndarray:
        return self._a


def _model(particles, *, tris=(), shapes=(), mass=0.0):
    """A finalized model as both bridges read it before ``gs.init``: its particles and their masses, triangles, shapes.

    ``shapes`` are world-fixed ``(GeoType, half extents, transform)``; a mirror record ``{"shape": i, "body": -1}``
    names the i-th. ``mass`` [kg] is every particle's.
    """
    from types import SimpleNamespace

    shapes = list(shapes)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    return SimpleNamespace(
        particle_q=_Arr(np.asarray(particles, dtype=np.float32)),
        particle_mass=_Arr(np.full(len(particles), mass, dtype=np.float32)),
        tri_count=len(tris),
        tri_indices=_Arr(tris.reshape(-1)),
        shape_type=_Arr([int(kind) for kind, _, _ in shapes]),
        shape_scale=_Arr(np.asarray([half for _, half, _ in shapes], dtype=np.float32).reshape(-1, 3)),
        shape_transform=_Arr(np.asarray([pose for _, _, pose in shapes], dtype=np.float32).reshape(-1, 7)),
        shape_body=_Arr(np.full(len(shapes), -1)),
        shape_material_mu=_Arr(np.full(len(shapes), 0.5)),
    )


def _block(at: float) -> np.ndarray:
    """Eight particles of a 4 mm liquid, a cube 4 mm a side whose bottom stands ``at`` [m]."""
    return np.array([[x, y, at + z] for x in (0.0, 0.004) for y in (0.0, 0.004) for z in (0.0, 0.004)])


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
    """``gs.init`` is first-caller-wins, so a later island asking for another precision is told what it got, or is
    refused where that would read its harvest as zero."""

    def _later(self, first: str, precision: str, coupler: str, coupler_first: str = "legacy") -> str:
        from unittest import mock

        from robosimstudio.engines.genesis import runtime

        with (
            mock.patch.object(runtime, "require_genesis", return_value=object()),
            mock.patch.dict(runtime._ENGINE, {"precision": first, "coupler": coupler_first}),
            self.assertLogs(runtime.log, "DEBUG") as logs,
        ):
            runtime.ensure_initialized(precision, coupler=coupler)
        return "\n".join(logs.output)

    def test_an_ipc_island_after_a_liquid_is_refused(self):
        """Its harvest would read zero at the liquid's float32: refused, not warned, since nothing reads the warning."""
        with self.assertRaises(RuntimeError) as caught:
            self._later("32", "64", "ipc")
        for word in ("precision=32", "liquid island", "reads ZERO", "process of its own"):
            self.assertIn(word, str(caught.exception))

    def test_an_ipc_island_after_an_ipc_island_at_32_is_not_blamed_on_a_liquid(self):
        with self.assertRaises(RuntimeError) as caught:
            self._later("32", "64", "ipc", coupler_first="ipc")
        self.assertIn("an IPC island that asked for 32", str(caught.exception))
        self.assertNotIn("liquid", str(caught.exception))

    def test_a_liquid_asking_for_64_after_32_is_not_blamed_on_the_ipc_harvest(self):
        said = self._later("32", "64", "legacy")
        self.assertNotIn("IPC", said)
        self.assertIn("liquid island, which asks for 64, steps its SPH at float32", said)

    def test_a_liquid_after_an_ipc_island_is_told_it_runs_slow(self):
        self.assertIn("8x slower", self._later("64", "32", "legacy"))

    def _first(self, precision: str, coupler: str) -> None:
        from unittest import mock

        from robosimstudio.engines.genesis import runtime

        with (
            mock.patch.object(runtime, "require_genesis", return_value=mock.MagicMock()),
            mock.patch.dict(runtime._ENGINE, {"precision": None}),
        ):
            runtime.ensure_initialized(precision, coupler=coupler)

    def test_a_liquid_that_inits_at_64_is_told_it_runs_slow(self):
        """``backend("genesis")`` without ``precision="32"`` inits the process at float64 for a liquid island."""
        from robosimstudio.engines.genesis import runtime

        with self.assertLogs(runtime.log, "WARNING") as logs:
            self._first("64", "legacy")
        said = "\n".join(logs.output)
        self.assertIn("8x slower", said)
        self.assertIn("rss.WATER_60HZ", said)

    def test_a_liquid_that_inits_at_32_is_not_warned(self):
        from robosimstudio.engines.genesis import runtime

        with self.assertNoLogs(runtime.log, "WARNING"):
            self._first("32", "legacy")


class TestALiquidIslandStatesItsSubsteps(unittest.TestCase):
    """A liquid island that states no substeps takes one SPH substep a control step, and its water falls through the
    cup and the bench: it is refused before the engine is asked for, by the arrangement that states them."""

    def _open(
        self, substeps: int, hz: float = 60.0, *, actors: list | None = None, at: float = 0.1, ground: bool = False
    ) -> None:
        """Each actor a block of eight particles standing ``at``, over a world-fixed ground plane at 0 if ``ground``."""
        from unittest import mock

        from newton import GeoType

        from robosimstudio.engines.genesis import sph
        from robosimstudio.engines.genesis.solver import SolverGenesis

        actors = actors or [{"kind": "sph", "name": "water"}]
        records = [{**actor, "particle_start": 8 * i, "particle_count": 8} for i, actor in enumerate(actors)]
        plane = [(GeoType.PLANE, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))] if ground else []
        model = _model(np.concatenate([_block(at)] * len(actors)), shapes=plane)
        mirrors = [{"shape": 0, "body": -1}] if ground else []
        recipe = {"island": "water", "control_hz": hz, "actors": records, "mirrors": mirrors}
        with mock.patch.object(sph, "ensure_initialized", side_effect=LookupError("asked for the engine")):
            SolverGenesis(model, recipe=recipe, substeps=substeps)

    def test_one_substep_is_refused_by_name(self):
        with self.assertRaises(ValueError) as caught:
            self._open(1)
        for word in ("'water'", "substeps=20", "rss.WATER_60HZ"):
            self.assertIn(word, str(caught.exception))

    def test_the_measured_substep_is_the_floor_at_any_rate(self):
        with self.assertRaises(ValueError):
            self._open(19)
        for substeps, hz in ((20, 60.0), (10, 120.0), (40, 30.0)):
            with self.subTest(substeps=substeps, hz=hz), self.assertRaisesRegex(LookupError, "asked for the engine"):
                self._open(substeps, hz)

    def test_each_liquid_is_refused_by_what_its_own_material_holds(self):
        """The oil preset's cohesion asks 35 substeps where the water beside it asks 20: the island steps both."""
        water = {"kind": "sph", "name": "water", "params": {"density": 1000.0}}
        oil = {"kind": "sph", "name": "oil", "params": {"density": 900.0, "viscosity": 1.0e-3, "surface_tension": 0.03}}
        with self.assertRaises(ValueError) as caught:
            self._open(20, actors=[water, oil])
        for word in ("'oil'", "surface tension, gamma 0.03", "0.48 ms", "substeps=35"):
            self.assertIn(word, str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            self._open(1, actors=[water, oil])
        self.assertIn("substeps=35", str(caught.exception), "the most either liquid needs, not the first's 20")
        self.assertIn("'oil'", str(caught.exception))
        with self.assertRaisesRegex(LookupError, "asked for the engine"):
            self._open(35, actors=[water, oil])

    def test_liquids_at_two_spacings_are_refused_before_the_engine_is_asked_for(self):
        """Refused after ``gs.init``, the island would still have set the precision every later island gets."""
        fine = {"kind": "sph", "name": "fine", "params": {"spacing": 0.003}}
        with self.assertRaisesRegex(ValueError, "particle spacings"):
            self._open(40, actors=[{"kind": "sph", "name": "water"}, fine])

    def test_a_liquid_stating_no_spacing_has_the_defaults(self):
        """The spacing is read off the one merge of the actor's params over ``SPH_DEFAULTS`` the substeps are, not a
        number of its own beside it: with the default at 3 mm, a liquid stating none and one stating 3 mm agree."""
        from unittest import mock

        from robosimstudio.engines.genesis import sph

        fine = {"kind": "sph", "name": "fine", "params": {"spacing": 0.003}}
        with (
            mock.patch.dict(sph.SPH_DEFAULTS, {"spacing": 0.003}),
            self.assertRaisesRegex(LookupError, "asked for the engine"),
        ):
            self._open(40, actors=[{"kind": "sph", "name": "water"}, fine])

    def test_an_empty_box_is_refused_before_the_engine_is_asked_for(self):
        """Water started 0.6 m under the ground leaves the SPH box empty: its floor is the ground, its top 0.5 m over
        the water. Genesis refused it after ``gs.init`` and named neither, and a cloth island built next in the
        process was refused for the float32 it left (``g1_box.py``)."""
        with self.assertRaises(ValueError) as caught:
            self._open(20, at=-0.6, ground=True)
        for word in ("'water'", "is empty", "sph_reach=0.5", "the ground at 0 m", "Start the liquid over the ground"):
            self.assertIn(word, str(caught.exception))
        with self.assertRaisesRegex(LookupError, "asked for the engine"):
            self._open(20, at=0.1, ground=True)


class TestAnIpcIslandIsRefusedBeforeTheEngineIsAskedFor(unittest.TestCase):
    """``gs.init`` runs once a process: an IPC island refused after it still set float64 for every later island, and a
    water island built next stepped at 64 where it asked for 32, warned 8x slower (``g1_order.py``). Each refusal the
    model and the recipe can make comes first, the mass check among them: it reads what a sheet is handed, which
    Genesis's built entity does not say (``g1_mass.py``)."""

    BENCH = (0.3, 0.65, 0.375), (0.6, 0.0, 0.375, 0.0, 0.0, 0.0, 1.0)
    """bench_075's top as a world-fixed box: half extents, and its pose, its top face at 0.75 m."""

    WEIGHS = 0.003
    """[kg]: the 100 mm sheet at the 0.3 kg/m^2 a shell record stating none is handed."""

    def _open(self, kind: str = "shell", *, at: float = 0.76, weighs: float = WEIGHS, **params) -> None:
        """A 100 mm sheet of four particles weighing ``weighs`` standing ``at`` over the bench's middle, its mirror the
        bench, its record stating ``params``."""
        from unittest import mock

        from newton import GeoType

        from robosimstudio.engines.genesis import bridge
        from robosimstudio.engines.genesis.solver import SolverGenesis

        sheet = [[0.55 + x, y, at] for y in (-0.05, 0.05) for x in (0.0, 0.1)]
        model = _model(sheet, tris=[[0, 1, 2], [1, 3, 2]], shapes=[(GeoType.BOX, *self.BENCH)], mass=weighs / 4)
        actor = {"kind": kind, "name": "towel", "particle_start": 0, "particle_count": 4, "params": params}
        recipe = {"island": "cloth", "actors": [actor], "mirrors": [{"shape": 0, "body": -1}]}
        with mock.patch.object(bridge, "ensure_initialized", side_effect=LookupError("asked for the engine")):
            SolverGenesis(model, recipe=recipe, substeps=2)

    def test_a_kind_it_cannot_build_is_refused_by_name(self):
        with self.assertRaises(ValueError) as caught:
            self._open("rod")
        for word in ("'rod'", "'towel'", "'shell'", "register_actor_builder('genesis'"):
            self.assertIn(word, str(caught.exception))

    def test_a_sheet_started_inside_the_bench_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self._open(at=0.74)
        for word in ("'towel'", "inside mirror shape 0", "intersection-free"):
            self.assertIn(word, str(caught.exception))
        with self.assertRaisesRegex(LookupError, "asked for the engine"):
            self._open(at=0.76)

    def test_a_sheet_genesis_would_weigh_twice_is_refused(self):
        """A record stating 0.6 kg/m^2 over a sheet Newton built at 0.3: 6.0 g on Genesis to Newton's 3.0. Read off
        the entity, the check found no mass on genesis-world 1.3.3 and the sheet reached the engine with a warning."""
        with self.assertRaises(ValueError) as caught:
            self._open(areal_density=0.6)
        for word in ("'towel'", "6.0 g", "3.0 g", "ratio 2.000", "areal_density"):
            self.assertIn(word, str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            self._open(weighs=2.0 * self.WEIGHS)  # and a sheet stating none, which is handed 0.3
        self.assertIn("ratio 0.500", str(caught.exception))

    def test_a_sheet_inside_the_band_reaches_the_engine(self):
        """The band is 10 % either way, for a factor rather than a figure: a sheet built as stated weighs the same."""
        for weighs, params in ((self.WEIGHS, {}), (self.WEIGHS, {"areal_density": 0.3, "thickness": 0.004})):
            with self.subTest(**params), self.assertRaisesRegex(LookupError, "asked for the engine"):
                self._open(weighs=weighs, **params)
        for weighs in (0.91 * self.WEIGHS, 1.09 * self.WEIGHS):
            with self.subTest(weighs=weighs), self.assertRaisesRegex(LookupError, "asked for the engine"):
                self._open(weighs=weighs)
        for weighs in (0.89 * self.WEIGHS, 1.12 * self.WEIGHS):
            with self.subTest(weighs=weighs), self.assertRaisesRegex(ValueError, "would weigh"):
                self._open(weighs=weighs)


class TestOnlyTheParticlesNewtonWeighsAreWeighed(unittest.TestCase):
    """A held cloth's held particles and a pinned paper's pinned ones are massless on Newton, its kinematic particle,
    and Genesis is handed the whole sheet. Weighed whole against Newton's sum, a 200 mm towel held by one edge row of
    its 121 particles read 1.100 and a 120 x 130 mm board pinned by two of its 14 rows 1.167, and both were refused
    blaming a correct record (``r1_probe.py``). Built here by their own entities on a CPU model, no scene."""

    def _open(self, model, record: dict) -> None:
        from unittest import mock

        from robosimstudio.engines.genesis import bridge
        from robosimstudio.engines.genesis.solver import SolverGenesis

        recipe = {"island": "cloth", "actors": [record], "mirrors": []}
        with mock.patch.object(bridge, "ensure_initialized", side_effect=LookupError("asked for the engine")):
            SolverGenesis(model, recipe=recipe, substeps=2)

    def _built(self, spec, *, holder: bool = False):
        """``spec`` built by its entity, bound and its materials applied as a compile applies them, and its record."""
        import newton

        import robosimstudio as rss
        from robosimstudio.scene.registry import resolve_entity_class

        entity = resolve_entity_class(spec, rss.GENERIC)(spec)
        builder = newton.ModelBuilder()
        if holder:
            builder.add_body(label="holder")
        entity.build_into(builder, None)
        model = builder.finalize(device="cpu")
        entity.bind(model)
        entity.apply_model_materials(model)
        particles = entity.owned_particles()
        record = entity.host_actor().record(
            name=spec.name, particle_start=int(min(particles)), particle_count=len(particles)
        )
        return model, record

    def test_a_held_cloth_and_a_pinned_paper_reach_the_engine(self):
        """Their records state what Newton built them at; a record stating twice or half is still refused."""
        import robosimstudio as rss

        held = rss.sheet(
            name="towel",
            size=(0.2, 0.2),
            cells=(10, 10),
            held_by="holder",
            particle_sets=(("held", lambda rest: rest[:, 1] > rest[:, 1].max() - 0.012),),
        )
        paper = rss.paper(name="flap", size=(0.12, 0.13), cells=(12, 13), material=rss.papers.CARDBOARD)
        pinned = rss.replace(
            paper, particle_sets=(("bound", lambda rest: rest[:, 1] <= rest[:, 1].min() + 0.018),), pinned="bound"
        )
        for spec, holder, massless, factor in ((held, True, 11, 2.0), (pinned, False, 26, 0.5)):
            with self.subTest(spec.name):
                model, record = self._built(spec, holder=holder)
                count = record["particle_count"]
                self.assertEqual("grid", record["lumping"], "add_cloth_grid built it, and its record says so")
                self.assertEqual(massless, int((model.particle_mass.numpy() == 0.0).sum()), "massless, as built")
                with self.assertRaisesRegex(LookupError, "asked for the engine"):
                    self._open(model, record)
                params = {**record["params"], "areal_density": factor * record["params"]["areal_density"]}
                with self.assertRaises(ValueError) as caught:
                    self._open(model, {**record, "params": params})
                for word in (f"ratio {factor:.3f}", f"over the {count - massless} of its {count} particles"):
                    self.assertIn(word, str(caught.exception))

    def test_a_mesh_sheets_massless_corner_is_weighed_by_the_area_newton_gives_it(self):
        """``add_cloth_mesh`` masses a particle by a third of its triangles' area, ``add_cloth_grid`` every one alike.
        A strip whose last cell is 17 times the others', its far edge massless: 8 of its 10 particles, and 57.5 % of
        its area by Newton's lumping. Counted alike, Genesis's share read 1.391 of Newton's mass, refused."""
        import newton
        import warp as wp

        xs, ys = (0.0, 0.01, 0.02, 0.03, 0.2), (0.0, 0.1)
        vertices = [wp.vec3(x, y, 0.0) for y in ys for x in xs]
        quads = [(i, i + 1, i + 6, i + 5) for i in range(4)]
        faces = [v for a, b, c, d in quads for v in (a, b, c, a, c, d)]
        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0), vertices=vertices, indices=faces,
            density=0.3,
        )  # fmt: skip
        model = builder.finalize(device="cpu")
        mass = model.particle_mass.numpy()
        mass[[4, 9]] = 0.0  # the far edge, as a held or pinned set is left
        model.particle_mass.assign(mass)
        record = {"kind": "shell", "name": "strip", "particle_start": 0, "particle_count": 10, "lumping": "mesh"}
        for areal, raises in ((0.3, LookupError), (0.6, ValueError)):
            with self.subTest(areal_density=areal), self.assertRaises(raises):
                self._open(model, {**record, "params": {"areal_density": areal}})

    RIM = 40
    """Of an 11 x 11 sheet's 121 particles, the ones on its edge: what a hand round it or a pin along it leaves massless."""

    @staticmethod
    def _rim(rest) -> np.ndarray:
        """The particles on a square sheet's edge."""
        x, y = rest[:, 0], rest[:, 1]
        edge = 1.0e-6
        return (x < x.min() + edge) | (x > x.max() - edge) | (y < y.min() + edge) | (y > y.max() - edge)

    @staticmethod
    def _regular(n: int = 11, side: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
        """An ``n`` x ``n`` regular mesh ``side`` [m] square: its vertices, and two triangles to a cell."""
        ticks = np.linspace(0.0, side, n)
        vertices = np.array([[x, y, 0.0] for y in ticks for x in ticks])
        cells = [(a, a + 1, a + n + 1, a + n) for a in (j * n + i for j in range(n - 1) for i in range(n - 1))]
        return vertices, np.array([t for a, b, c, d in cells for t in ((a, b, c), (a, c, d))], dtype=np.int64)

    def test_a_regular_mesh_held_round_its_rim_is_weighed_by_the_area_newton_gives_it(self):
        """An 11 x 11 ``add_cloth_mesh`` sheet at 0.3 kg/m^2, its rim massless: its 81 weighed particles weigh alike, as
        a grid's do, and hold 81 of its 100 cells. Told apart by whether they weigh alike, it was counted as a grid,
        81/121 of its area, and a record stating 0.3 read 0.826, refused. Its record says how it was massed: read
        that way it reaches the engine, read as a grid it is refused at the same 0.826, and saying nothing it is not
        weighed at all."""
        import newton
        import warp as wp

        vertices, faces = self._regular()
        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0),
            vertices=[wp.vec3(*v) for v in vertices], indices=faces.reshape(-1).tolist(), density=0.3,
        )  # fmt: skip
        model = builder.finalize(device="cpu")
        mass = model.particle_mass.numpy()
        mass[self._rim(vertices)] = 0.0
        model.particle_mass.assign(mass)
        weighed = mass[mass > 0.0]
        self.assertEqual((121 - self.RIM, True), (len(weighed), bool(np.allclose(weighed, weighed[0]))), "alike")
        record = {"kind": "shell", "name": "sheet", "particle_start": 0, "particle_count": 121}
        mesh = {**record, "lumping": "mesh"}
        with self.assertRaisesRegex(LookupError, "asked for the engine"):
            self._open(model, {**mesh, "params": {"areal_density": 0.3}})
        with self.assertRaisesRegex(ValueError, "ratio 2.000"):
            self._open(model, {**mesh, "params": {"areal_density": 0.6}})
        with self.assertRaisesRegex(ValueError, "ratio 0.826"):
            self._open(model, {**record, "lumping": "grid", "params": {"areal_density": 0.3}})
        with self.assertLogs("robosimstudio.engines.genesis.bridge", "DEBUG") as said:
            with self.assertRaisesRegex(LookupError, "asked for the engine"):
                self._open(model, {**record, "params": {"areal_density": 0.6}})
        self.assertIn("not weighed", "\n".join(said.output))

    def test_a_mesh_sheet_held_round_its_rim_says_how_it_was_massed(self):
        """``rss.sheet(mesh="mesh")`` over a regular mesh, held round its rim: built by its entity, its record says
        ``add_cloth_mesh`` massed it, so a record of its built density reaches the engine and one of twice is refused."""
        import tempfile
        from pathlib import Path

        import trimesh

        import robosimstudio as rss

        vertices, faces = self._regular()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sheet.obj"
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)
            spec = rss.sheet(
                name="towel",
                mesh="mesh",
                asset=rss.AssetSpec(path=str(path)),
                held_by="holder",
                particle_sets=(("held", self._rim),),
            )
            model, record = self._built(spec, holder=True)
        self.assertEqual("mesh", record["lumping"])
        self.assertEqual(self.RIM, int((model.particle_mass.numpy() == 0.0).sum()), "massless, as built")
        with self.assertRaisesRegex(LookupError, "asked for the engine"):
            self._open(model, record)
        params = {**record["params"], "areal_density": 2.0 * record["params"]["areal_density"]}
        with self.assertRaisesRegex(ValueError, "ratio 2.000"):
            self._open(model, {**record, "params": params})

    def test_weighed_particles_holding_no_area_are_refused_by_name(self):
        """Newton weighs only two particles of the range, on no triangle of it, at two masses: the area they hold is
        0, and the refusal divided by it, a ZeroDivisionError for the ValueError meant."""
        corners = [[0.0, 0, 0.1], [0.1, 0, 0.1], [0.0, 0.1, 0.1], [0.1, 0.1, 0.1]]
        model = _model([*corners, [0.3, 0, 0.1], [0.4, 0, 0.1]], tris=[[0, 1, 2], [1, 3, 2]])
        model.particle_mass = _Arr(np.array([0.0, 0.0, 0.0, 0.0, 1.0e-3, 2.0e-3], dtype=np.float32))
        record = {"kind": "shell", "name": "strip", "particle_start": 0, "particle_count": 6, "lumping": "mesh"}
        with self.assertRaises(ValueError) as caught:
            self._open(model, {**record, "params": {"areal_density": 0.3}})
        for word in ("'strip'", "over the 2 of its 6 particles", "hold none of its triangles' area"):
            self.assertIn(word, str(caught.exception))


class TestTheLongestSubstepIsTheMaterials(unittest.TestCase):
    """Genesis's viscosity limit, and the pressure wave's and the cohesion's, scaled from the water the probes held."""

    def _need(self, **params) -> tuple[int, str]:
        """Substeps a 60 Hz control step, and the limit that sets them."""
        from robosimstudio.engines.genesis.sph import longest_substep

        longest, limit = longest_substep(params)
        return math.ceil(1.0 / 60.0 / longest - 1.0e-9), limit

    def test_the_water_measured_holds_at_the_measured_substep(self):
        from robosimstudio.engines.genesis.sph import LONGEST_SUBSTEP, SPH_DEFAULTS, longest_substep

        self.assertEqual((LONGEST_SUBSTEP, "stiffness"), longest_substep({}))
        self.assertEqual((LONGEST_SUBSTEP, "stiffness"), longest_substep(SPH_DEFAULTS))

    def test_the_viscosity_limit_is_genesiss_own(self):
        """``0.01 * (2 * spacing)**2 / mu``: 0.64 ms at 1e-3 and 4 mm, which a 20-substep step (0.83 ms) breaks."""
        from robosimstudio.engines.genesis.sph import longest_substep

        longest, limit = longest_substep({"viscosity": 1.0e-3})
        self.assertEqual("viscosity", limit)
        self.assertAlmostEqual(0.64e-3, longest, places=12)
        self.assertEqual((27, "viscosity"), self._need(viscosity=1.0e-3))

    def test_a_faster_pressure_wave_takes_more(self):
        """5e4 Pa fell through the cup at 30 and held at 40; a lighter liquid at the same stiffness is stiffer."""
        self.assertEqual((45, "stiffness"), self._need(stiffness=5.0e4))
        self.assertEqual((22, "stiffness"), self._need(density=900.0))

    def test_a_more_cohesive_liquid_takes_more(self):
        """At rest in the cup: 0.03 lost 353 at 27 and held at 35, 0.02 held at 27, 0.015 held at 25."""
        for gamma, need in ((0.03, 35), (0.02, 29), (0.015, 25)):
            with self.subTest(gamma=gamma):
                self.assertEqual((need, "surface_tension"), self._need(surface_tension=gamma))

    def test_nothing_is_given_longer_substeps_than_were_measured(self):
        """5e3 Pa lost 618 at 16: softer, coarser or less cohesive water still takes 20."""
        for params in ({"stiffness": 5.0e3}, {"spacing": 0.008}, {"surface_tension": 0.0}, {"viscosity": 0.0}):
            with self.subTest(**params):
                self.assertEqual(20, self._need(**params)[0])

    def test_finer_particles_take_more(self):
        """At 2 mm the cohesion binds before Genesis's viscosity limit (53) and the pressure wave (40)."""
        self.assertEqual((57, "surface_tension"), self._need(spacing=0.002))


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

    def test_only_a_world_fixed_plane_facing_up_is_the_floor(self):
        """The SPH box's floor is the ground's height: a tilted plane, one facing down or one a body carries would lift
        it over the water."""
        from robosimstudio.core.math import quat_from_axis_angle
        from robosimstudio.engines.genesis.sph import _ground

        tilted, down = quat_from_axis_angle((1.0, 0.0, 0.0), 0.3), quat_from_axis_angle((1.0, 0.0, 0.0), np.pi)
        shapes = [
            _geom(kind="plane", body=-1, local=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])),
            _geom(kind="plane", body=-1, local=np.array([0.0, 0.0, 0.4, *tilted])),
            _geom(kind="plane", body=-1, local=np.array([0.0, 0.0, 0.5, *down])),
            _geom(kind="plane", body=3, local=np.array([0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0])),
            _geom(kind="box", body=-1, local=np.array([0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0])),
        ]
        self.assertEqual([0.0], _ground(shapes))

    def test_a_mirror_turns_at_a_rate_in_its_own_frame(self):
        """Genesis integrates a free joint's angular dofs in the body's frame. A body yawed 90 degrees about z and
        turned about world x has its rate on its own -y axis."""
        from robosimstudio.core.math import quat_from_axis_angle, quat_multiply
        from robosimstudio.engines.genesis.sph import _angular_velocity

        start = quat_from_axis_angle((0.0, 0.0, 1.0), np.pi / 2)
        end = quat_multiply(quat_from_axis_angle((1.0, 0.0, 0.0), 0.02), start)
        np.testing.assert_allclose(_angular_velocity(start, end, 0.01), [0.0, -2.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(_angular_velocity(start, start, 0.01), np.zeros(3))

    def test_water_at_the_boxs_side_is_said_and_on_its_floor_is_not(self):
        """Genesis holds a particle at its box's face. The floor is the ground, or where spilled water is meant to rest;
        a side or the top is a wall nothing in the scene draws."""
        from robosimstudio.engines.genesis import sph

        bridge = object.__new__(sph.SphBridge)
        bridge.recipe, bridge.spacing, bridge._at_wall = {"island": "water", "sph_reach": 0.5}, 0.004, False
        bridge._box = (np.array([-0.5, -0.5, 0.002]), np.array([0.5, 0.5, 0.6]))
        with self.assertNoLogs(sph.log, "WARNING"):
            bridge._check_walls("water", np.array([[0.0, 0.0, 0.002], [0.3, -0.4, 0.002]]))
        self.assertFalse(bridge._at_wall)
        with self.assertLogs(sph.log, "WARNING") as logs:
            bridge._check_walls("water", np.array([[0.0, 0.0, 0.002], [0.3, -0.5, 0.002]]))
        self.assertTrue(bridge._at_wall, "said once: the tick stops looking")
        for word in ("'water'", "1 of 2", "sph_reach=0.5"):
            self.assertIn(word, logs.output[0])

    def test_a_reset_forgets_the_last_reaction(self):
        """``last_force`` is the last control step's reaction; after a reset there has been none."""
        from unittest import mock

        from robosimstudio.engines.genesis.sph import SphBridge

        bridge = object.__new__(SphBridge)
        bridge.actors, bridge.rig, bridge.feedback_enabled = [], mock.Mock(), False
        bridge.last_force = {3: np.array([0.0, 0.0, -1.1])}
        bridge.restore_rest()
        self.assertEqual({}, bridge.last_force)


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

    def test_a_bridge_states_no_default_the_solver_does_not(self):
        """The solver passes every option; a default of a bridge's own that differs from its own is one that only a
        bridge built by hand ever sees."""
        import inspect

        from robosimstudio.engines.genesis.bridge import GenesisBridge
        from robosimstudio.engines.genesis.solver import SolverGenesis
        from robosimstudio.engines.genesis.sph import SphBridge

        solver = inspect.signature(SolverGenesis.open).parameters
        for bridge in (GenesisBridge, SphBridge):
            for name, param in inspect.signature(bridge.__init__).parameters.items():
                if param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is not inspect.Parameter.empty:
                    with self.subTest(bridge=bridge.__name__, option=name):
                        self.assertEqual(solver[name].default, param.default)

    def test_open_is_the_only_thing_it_writes(self):
        """The point of SolverHost: a second engine costs one method."""
        from robosimstudio.engines.genesis.solver import SolverGenesis

        own = {n for n, v in vars(SolverGenesis).items() if callable(v)}
        self.assertEqual(own, {"open"}, f"SolverGenesis defines more than open(): {sorted(own)}")


if __name__ == "__main__":
    unittest.main()
