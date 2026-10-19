# SPDX-License-Identifier: Apache-2.0
"""Where a scene runs, decided in one place: the device, the physics that follows it, the renderer.

What :mod:`robosimstudio.machine` promises, with the machine patched -- the rule
has to come out right on a GPU box and on a laptop without either being in the
room -- and what the CPU adaptation of an arrangement does, read off the
arrangement and the backends' registered facts with nothing built. The scenes
themselves, built and stepped on a CPU, are :mod:`tests.test_cpu_scenes`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import io
import os
import types
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

TIER = "fast"
"""No GPU. Gates every commit."""


def _device(alias: str, *, name: str = "NVIDIA L40S", arch: int = 89):
    """A Warp device as the rule reads one."""
    cuda = alias.startswith("cuda")
    return types.SimpleNamespace(alias=alias, is_cuda=cuda, is_cpu=not cuda, name=name if cuda else "x86_64", arch=arch)


@contextlib.contextmanager
def _machine(*, gpu: bool, env: dict[str, str] | None = None):
    """Warp as a machine with or without a GPU, and the two variables as given.

    ``set_device`` is patched always: the rule calls it for ``ROBOSIMSTUDIO_DEVICE``,
    and the real one would move every later test in this process onto a CPU.
    """
    devices = {"cpu": _device("cpu")}
    if gpu:
        devices.update({"cuda": _device("cuda:0"), "cuda:0": _device("cuda:0"), "cuda:1": _device("cuda:1")})

    def preferred():
        return devices["cuda:0"] if gpu else devices["cpu"]

    def get_device(name: Any = None):
        if name is None:
            return devices["cuda:0"] if gpu else devices["cpu"]
        if not isinstance(name, str):
            return name
        if name not in devices:
            raise RuntimeError(f"Invalid device identifier: {name}")
        return devices[name]

    variables = {"ROBOSIMSTUDIO_DEVICE": "", "ROBOSIMSTUDIO_RENDERER": "", **(env or {})}
    with (
        patch.dict(os.environ, variables),
        patch("warp.get_device", side_effect=get_device),
        patch("warp.get_preferred_device", side_effect=preferred),
        patch("warp.is_cuda_available", return_value=gpu),
        patch("warp.set_device") as set_device,
        patch("robosimstudio.solvers.render.rtx_available", return_value=False),
    ):
        yield set_device


class TestTheDeviceRule(unittest.TestCase):
    """A name in code, else ``ROBOSIMSTUDIO_DEVICE``, else CUDA where there is one and the CPU where not."""

    def test_the_machine_decides_when_nobody_does(self) -> None:
        from robosimstudio.machine import platform

        with _machine(gpu=True):
            got = platform()
        self.assertEqual((got.device, got.device_why), ("cuda:0", "NVIDIA L40S (sm_89)"))
        with _machine(gpu=False):
            got = platform()
        self.assertEqual((got.device, got.device_why), ("cpu", "no CUDA device"))
        self.assertTrue(got.is_cpu)
        self.assertFalse(got.cuda_graphs)

    def test_auto_and_nothing_name_nothing(self) -> None:
        from robosimstudio.machine import platform

        for device in ("auto", "", None, " auto "):
            for env in ("", "auto"):
                with _machine(gpu=True, env={"ROBOSIMSTUDIO_DEVICE": env}):
                    self.assertEqual(platform(device).device, "cuda:0", (device, env))

    def test_the_variable_wins_over_the_machine_and_a_build_moves_warp_with_it(self) -> None:
        """Asking changes nothing; building under the variable makes its device Warp's default."""
        from robosimstudio import machine

        with _machine(gpu=True, env={"ROBOSIMSTUDIO_DEVICE": "cpu"}) as set_device:
            got = machine.platform()
            set_device.assert_not_called()
            machine.apply(got)
        self.assertEqual((got.device, got.device_why), ("cpu", "ROBOSIMSTUDIO_DEVICE=cpu"))
        set_device.assert_called_once_with("cpu")
        machine._SET_FOR_THE_VARIABLE.clear()

    def test_a_default_set_for_the_variable_does_not_outlive_it(self) -> None:
        """Built once under ROBOSIMSTUDIO_DEVICE=cpu, then the variable unset: the next auto build is the GPU's."""
        from robosimstudio import machine

        with _machine(gpu=True, env={"ROBOSIMSTUDIO_DEVICE": "cpu"}) as set_device:
            machine.apply(machine.platform())
        set_device.assert_called_once_with("cpu")
        with _machine(gpu=True) as set_device:
            with patch("warp.get_device", side_effect=lambda name=None: _device(name or "cpu")):
                got = machine.platform()
                self.assertEqual((got.device, got.device_why), ("cuda:0", "NVIDIA L40S (sm_89)"))
                machine.apply(got)
        self.assertEqual(set_device.call_args.args[0].alias, "cuda:0")
        self.assertEqual(machine._SET_FOR_THE_VARIABLE, [])

    def test_a_device_scope_opened_in_code_wins_over_the_variable(self) -> None:
        """``with wp.ScopedDevice("cuda:1"): scene.build()`` builds on cuda:1, as ``device=None`` always did."""
        from robosimstudio.machine import platform

        with _machine(gpu=True, env={"ROBOSIMSTUDIO_DEVICE": "cpu"}):
            with patch("warp.get_device", side_effect=lambda name=None: _device(name or "cuda:1")):
                got = platform()
        self.assertEqual((got.device, got.device_why), ("cuda:1", "Warp's current device"))

    def test_a_name_in_code_wins_over_the_variable_and_leaves_warp_alone(self) -> None:
        from robosimstudio.machine import platform

        with _machine(gpu=True, env={"ROBOSIMSTUDIO_DEVICE": "cpu"}) as set_device:
            got = platform("cuda:1")
        self.assertEqual((got.device, got.device_why), ("cuda:1", "asked for"))
        set_device.assert_not_called()

    def test_a_device_this_machine_lacks_is_refused_by_name(self) -> None:
        from robosimstudio.machine import platform

        with _machine(gpu=False, env={"ROBOSIMSTUDIO_DEVICE": "cuda"}), self.assertRaises(ValueError) as caught:
            platform()
        self.assertIn("ROBOSIMSTUDIO_DEVICE=cuda", str(caught.exception))
        self.assertIn("no CUDA device here", str(caught.exception))
        with _machine(gpu=True), self.assertRaisesRegex(ValueError, "device='gpu'"):
            platform("gpu")


class TestTheRendererOnACpu(unittest.TestCase):
    """The renderer runs where the physics does, so a CPU gets ``tiled``, and ``rtx`` named there fails by name."""

    def choose(self, requested: str = "", *, env: str = ""):
        from robosimstudio.solvers.render import choose_renderer

        with _machine(gpu=False, env={"ROBOSIMSTUDIO_RENDERER": env}):
            return choose_renderer(requested, device="cpu")

    def test_a_cpu_is_tiled_because_it_is_a_cpu(self) -> None:
        self.assertEqual(self.choose(), ("tiled", "cpu is not a CUDA device"))

    def test_rtx_named_on_a_cpu_is_refused_rather_than_tried(self) -> None:
        with self.assertRaisesRegex(ValueError, "runs on cuda only, and this scene is on cpu"):
            self.choose("rtx")
        with self.assertRaisesRegex(ValueError, "set by ROBOSIMSTUDIO_RENDERER"):
            self.choose(env="rtx")

    def test_tiled_and_a_dotted_path_are_names_a_cpu_takes(self) -> None:
        self.assertEqual(self.choose("tiled"), ("tiled", "asked for"))
        self.assertEqual(self.choose("my.renderers.Mine")[0], "my.renderers.Mine")

    def test_the_backends_say_where_they_run(self) -> None:
        from robosimstudio.solvers.render import backend_class

        self.assertEqual(backend_class("rtx").devices, ("cuda",))
        self.assertIn("cpu", backend_class("tiled").devices)

    def test_a_bad_renderer_variable_waits_for_a_frame(self) -> None:
        """A scene with no cameras still builds; the platform carries the reason, and asking raises."""
        from robosimstudio.machine import platform

        with _machine(gpu=False, env={"ROBOSIMSTUDIO_RENDERER": "rtx"}):
            got = platform()
            self.assertEqual(got.renderer, "")
            self.assertIn("ROBOSIMSTUDIO_RENDERER", got.renderer_why)
            self.assertIn("renderer unavailable", got.readout())
            with self.assertRaises(ValueError):
                got.renderer_for("")


class TestTheReadout(unittest.TestCase):
    """One line says everything, and a build prints it once."""

    def test_the_line_names_the_device_what_changed_the_graphs_and_the_renderer(self) -> None:
        from robosimstudio.machine import Platform

        plat = Platform(
            device="cpu", device_why="no CUDA device", renderer="tiled", renderer_why="cpu is not a CUDA device"
        )
        line = plat.with_notes(("GENERIC → CPU: ...",)).readout()
        self.assertEqual(
            line,
            "cpu (no CUDA device) · GENERIC → CPU: ... · CUDA graphs off · renderer tiled (cpu is not a CUDA device)",
        )
        self.assertEqual(str(plat), plat.readout())

    def test_graphs_read_what_the_build_will_do(self) -> None:
        from robosimstudio.machine import Platform

        gpu = Platform(device="cuda:0", device_why="x", renderer="tiled", renderer_why="y")
        self.assertIn("CUDA graphs on", gpu.readout())
        self.assertIn("CUDA graphs off", dataclasses.replace(gpu, graphs=False).readout())

    def test_an_unused_path_tracer_on_a_cpu_is_pointed_out(self) -> None:
        from robosimstudio.machine import Platform

        plat = Platform(device="cpu", device_why="x", renderer="tiled", renderer_why="y", rtx_installed=True)
        self.assertIn("uv sync --no-group rtx", plat.readout())
        on_gpu = dataclasses.replace(plat, device="cuda:0")
        self.assertNotIn("--no-group rtx", on_gpu.readout())

    def test_announce_says_a_line_once_unless_verbose(self) -> None:
        from robosimstudio import machine

        plat = machine.Platform(device="cpu", device_why="announce test", renderer="tiled", renderer_why="z")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            machine.announce(plat)
            machine.announce(plat)
            machine.announce(plat, verbose=True)
            machine.announce(plat.with_notes(("changed",)))
        self.assertEqual(out.getvalue().count("[rss] "), 3, out.getvalue())

    def test_platform_of_prefers_what_the_thing_already_knows(self) -> None:
        from robosimstudio.machine import Platform, platform_of

        plat = Platform(device="cpu", device_why="kept", renderer="tiled", renderer_why="kept")
        self.assertIs(platform_of(types.SimpleNamespace(platform=plat)), plat)


def _mujoco_of_my_own():
    import newton

    class MuJoCoOfMyOwn(newton.solvers.SolverMuJoCo):
        """A task's own MuJoCo, named by dotted path: it inherits MuJoCo's CPU facts."""

    MuJoCoOfMyOwn.__module__ = __name__
    return MuJoCoOfMyOwn


def __getattr__(name: str):
    # Built on first use, so importing this module imports no solver.
    if name == "MuJoCoOfMyOwn":
        cls = _mujoco_of_my_own()
        globals()[name] = cls
        return cls
    raise AttributeError(name)


class TestTheArrangementOnACpu(unittest.TestCase):
    """GENERIC becomes CPU whole; anything else keeps its numbers and loses only what a CPU cannot run."""

    def adapt(self, arrangement, *, is_cpu: bool = True):
        from robosimstudio.scene.physics.devices import arrangement_for

        return arrangement_for(arrangement, is_cpu=is_cpu)

    def test_generic_becomes_cpu_and_says_so(self) -> None:
        from robosimstudio.specs.physics import CPU, GENERIC

        got, notes = self.adapt(GENERIC)
        self.assertIs(got, CPU)
        self.assertEqual(len(notes), 1)
        self.assertIn("GENERIC → CPU", notes[0])

    def test_nothing_changes_on_a_gpu(self) -> None:
        from robosimstudio.specs.physics import GENERIC

        self.assertEqual(self.adapt(GENERIC, is_cpu=False), (GENERIC, ()))

    def test_an_arrangement_a_cpu_can_run_is_left_alone(self) -> None:
        from robosimstudio.specs.physics import CPU, MUJOCO_500HZ, backend

        for arrangement in (MUJOCO_500HZ, CPU, MUJOCO_500HZ.with_backend("soft", backend("superdex", substeps=2))):
            got, notes = self.adapt(arrangement)
            self.assertIs(got, arrangement)
            self.assertEqual(notes, ())

    def test_vbd_is_replaced_by_domain_and_free_bodies_join_the_arm(self) -> None:
        """The sponge's arrangement: GENERIC with its own SuperDex, VBD for everything else."""
        from robosimstudio.specs.physics import CPU, GENERIC, backend

        sponge = backend("superdex", substeps=2, feedback_dv_max=20.0)
        arrangement = GENERIC.with_backend("soft", sponge)
        got, notes = self.adapt(arrangement)
        self.assertEqual(got.backend_for("articulated"), backend("mujoco"))
        # The arm's own spec, so the cube and the arm are one MuJoCo island and not two.
        self.assertEqual(got.backend_for("rigid"), got.backend_for("articulated"))
        self.assertEqual(got.backend_for("soft"), sponge)
        # Cloth and rope fell to "*" = VBD; they land on CPU's own islands, not the sponge's.
        self.assertEqual(got.backend_for("cloth"), CPU.backend_for("cloth"))
        self.assertEqual(got.backend_for("rod"), CPU.backend_for("rod"))
        self.assertEqual(got.backend_for("rod").get("feedback"), "off")
        self.assertEqual((got.control_hz, got.substeps, got.coupling), (60.0, 16, arrangement.coupling))
        self.assertEqual(len(notes), 1)
        self.assertIn("vbd has no CPU path", notes[0])
        self.assertIn("everything else → mujoco", notes[0])

    def test_genesis_on_a_cpu_becomes_superdex(self) -> None:
        from robosimstudio.specs.physics import MUJOCO_500HZ, backend

        got, notes = self.adapt(MUJOCO_500HZ.with_backend("cloth", backend("genesis", substeps=2)))
        self.assertEqual(got.backend_for("cloth").solver, "superdex")
        self.assertIn("genesis has no CPU path: cloth → superdex", notes[0])

    def test_a_dotted_path_states_no_devices_and_is_kept(self) -> None:
        from robosimstudio.specs.physics import MUJOCO_500HZ, backend

        mine = MUJOCO_500HZ.with_backend("cloth", backend("newton.solvers.SolverVBD"))
        self.assertEqual(self.adapt(mine), (mine, ()))


class TestEachBackendOnACpu(unittest.TestCase):
    """What an island's spec becomes when the model is on a CPU: its registered CPU kwargs, or a refusal."""

    def on(self, spec, device: str = "cpu"):
        from robosimstudio.scene.physics.devices import on_device

        return on_device(spec, device, owner="thing", domain="cloth")

    def test_mujoco_is_its_c_library(self) -> None:
        from robosimstudio.specs.physics import backend

        got = self.on(backend("mujoco", nconmax=512))
        self.assertEqual(got.kwargs(), {"nconmax": 512, "use_mujoco_cpu": True, "use_mujoco_contacts": True})

    def test_past_its_world_limit_mujoco_is_left_as_stated(self) -> None:
        """compile_scene refuses that build; composed directly, MuJoCo-Warp on Warp's CPU device is what steps it."""
        from robosimstudio.scene.physics.devices import on_device
        from robosimstudio.specs.physics import backend

        self.assertEqual(on_device(backend("mujoco"), "cpu", owner="arm", domain="*", num_worlds=3), backend("mujoco"))

    def test_a_stated_kwarg_the_cpu_overrides_is_refused(self) -> None:
        from robosimstudio.scene.physics import CompositionError
        from robosimstudio.specs.physics import backend

        for stated in ({"use_mujoco_cpu": False}, {"use_mujoco_contacts": False}):
            with self.assertRaisesRegex(CompositionError, "on a CPU 'mujoco' runs with"):
                self.on(backend("mujoco", **stated))

    def test_an_entity_naming_a_backend_with_no_cpu_path_is_refused_naming_the_way_out(self) -> None:
        """Its entity class was chosen for that engine, so nothing can stand in for it."""
        from robosimstudio.scene.physics import CompositionError
        from robosimstudio.scene.physics.devices import on_device
        from robosimstudio.specs.physics import backend

        for solver in ("vbd", "genesis"):
            with self.assertRaises(CompositionError) as caught:
                on_device(backend(solver), "cpu", owner="thing", domain="cloth", declared=True)
            self.assertIn("rss.CPU", str(caught.exception))
            self.assertIn("rss.backend('superdex', substeps=2)", str(caught.exception))

    def test_an_arrangements_backend_composed_directly_on_a_cpu_runs(self) -> None:
        """compile_scene replaces it before the build; a caller composing by hand chose it, and VBD runs, slowly."""
        from robosimstudio.specs.physics import backend

        self.assertEqual(self.on(backend("vbd")), backend("vbd"))

    def test_mujoco_by_dotted_path_is_its_c_library_too(self) -> None:
        """Only the device facts follow the class: a dotted path is still handed exactly what it states."""
        from robosimstudio.scene.physics.devices import device_facts
        from robosimstudio.specs.physics import backend

        spec = backend("newton.solvers.SolverMuJoCo", nconmax=64)
        self.assertEqual(device_facts(spec).cpu_max_worlds, 1)
        self.assertEqual(self.on(spec).kwargs(), {"nconmax": 64, "use_mujoco_cpu": True, "use_mujoco_contacts": True})
        # By class, not by spelling: the module it is defined in, and a subclass of it.
        defined = backend("newton._src.solvers.mujoco.solver_mujoco.SolverMuJoCo")
        self.assertEqual(device_facts(defined).cpu_max_worlds, 1)
        subclass = backend("tests.test_platform.MuJoCoOfMyOwn")
        self.assertTrue(self.on(subclass).get("use_mujoco_cpu"))
        self.assertEqual(device_facts(backend("tests.test_platform.TestTheReadout")).cpu_params, ())

    def test_superdex_xpbd_and_unknown_paths_pass_untouched(self) -> None:
        from robosimstudio.specs.physics import backend

        for spec in (backend("superdex", substeps=2), backend("xpbd"), backend("my.solvers.Mine")):
            self.assertEqual(self.on(spec), spec)

    def test_on_a_gpu_nothing_changes(self) -> None:
        from robosimstudio.specs.physics import backend

        for spec in (backend("mujoco"), backend("vbd"), backend("genesis"), backend("superdex", substeps=2)):
            self.assertEqual(self.on(spec, "cuda:0"), spec)

    def test_a_cpu_only_solver_in_a_gpu_scene_must_be_a_host_engine(self) -> None:
        """A host engine steps on the CPU beside any device; a solver on Newton's arrays steps where they are."""
        from robosimstudio.scene.physics import CompositionError
        from robosimstudio.specs.physics import backend, register_backend

        register_backend("test_cpu_only_device", cls="my.solvers.OnTheArrays", devices=("cpu",))
        register_backend("test_cpu_only_host", cls="my.solvers.OnTheHost", devices=("cpu",), host_engine=True)
        with self.assertRaisesRegex(CompositionError, "host_engine=True"):
            self.on(backend("test_cpu_only_device"), "cuda:0")
        self.assertEqual(self.on(backend("test_cpu_only_host"), "cuda:0"), backend("test_cpu_only_host"))
        self.assertEqual(self.on(backend("test_cpu_only_device"), "cpu"), backend("test_cpu_only_device"))

    def test_a_gpu_scene_names_what_it_steps_on_the_cpu(self) -> None:
        from robosimstudio.machine import Platform
        from robosimstudio.scene.compiler import _on_the_host

        program = types.SimpleNamespace(
            islands=[
                types.SimpleNamespace(
                    is_host_engine=False, entities=("embodiment",), backend=types.SimpleNamespace(solver="mujoco")
                ),
                types.SimpleNamespace(
                    is_host_engine=True, entities=("towel", "sponge"), backend=types.SimpleNamespace(solver="superdex")
                ),
            ]
        )
        gpu = Platform(device="cuda:0", device_why="x", renderer="tiled", renderer_why="y")
        self.assertEqual(
            _on_the_host(program, gpu), ("towel, sponge on superdex, stepped on the CPU each control step",)
        )
        self.assertEqual(_on_the_host(program, dataclasses.replace(gpu, device="cpu")), ())


class TestTheCompilerRefusesWhatACpuCannotDo(unittest.TestCase):
    """Refused before anything is built, by name, rather than inside a solver's importer."""

    @dataclasses.dataclass(frozen=True)
    class _Request:
        physics: Any

    def adapt(self, physics, *, num_worlds: int = 1, device: str = "cpu"):
        from robosimstudio.machine import Platform
        from robosimstudio.scene.compiler import _for_platform

        plat = Platform(device=device, device_why="test", renderer="tiled", renderer_why="test")
        return _for_platform(self._Request(physics), plat, num_worlds)

    def test_generic_is_swapped_on_the_request(self) -> None:
        from robosimstudio.specs.physics import CPU, GENERIC

        request, notes = self.adapt(GENERIC)
        self.assertIs(request.physics, CPU)
        self.assertIn("GENERIC → CPU", notes[0])
        request, notes = self.adapt(GENERIC, device="cuda:0")
        self.assertIs(request.physics, GENERIC)
        self.assertEqual(notes, ())

    def test_more_worlds_than_mujocos_c_library_steps_is_refused(self) -> None:
        from robosimstudio.scene.compiler import CannotReplicate
        from robosimstudio.specs.physics import CPU

        with self.assertRaisesRegex(CannotReplicate, "build num_worlds=1 on a CPU"):
            self.adapt(CPU, num_worlds=2)
        self.adapt(CPU, num_worlds=2, device="cuda:0")

    def test_signed_distance_fields_are_refused(self) -> None:
        from robosimstudio.scene.physics import CompositionError
        from robosimstudio.specs.physics import CPU, CollisionSpec

        with self.assertRaisesRegex(CompositionError, "force_sdf"):
            self.adapt(dataclasses.replace(CPU, collision=CollisionSpec(force_sdf=True)))

    @unittest.skipUnless(importlib.util.find_spec("superdex") is None, "SuperDex is installed here")
    def test_without_superdex_the_note_names_the_extra(self) -> None:
        from robosimstudio.specs.physics import GENERIC

        _, notes = self.adapt(GENERIC)
        self.assertIn("robosimstudio[cpu]", notes[-1])


_OFF_ITS_ENGINE = (
    # kind, one of it, an arrangement that puts it on a backend that steps it on no device
    ("water", lambda rss: rss.water(name="it"), "GENERIC"),
    ("water", lambda rss: rss.water(name="it"), "CPU"),
    ("granular", lambda rss: rss.heap(name="it", size=(0.04, 0.04, 0.02)), "WATER_60HZ"),
    ("liquid", lambda rss: rss.liquid(name="it", size=(0.03, 0.03, 0.03)), "MUJOCO_500HZ"),
    ("dough", lambda rss: rss.dough(name="it", size=(0.04, 0.04, 0.02)), "WATER_60HZ"),
    ("paper", lambda rss: rss.paper(name="it", size=(0.1, 0.1), cells=(5, 5)), "MUJOCO_500HZ"),
    ("tape", lambda rss: rss.tape(name="it", on="box"), "CPU"),
    ("brush", lambda rss: rss.brush(name="it", rows=4, columns=4), "MUJOCO_500HZ"),
)
"""Each GPU-only kind once, on an arrangement without its engine: water on GENERIC's VBD falls there too."""


class TestAKindOffItsEngineIsToldItsRoute(unittest.TestCase):
    """Refused with the arrangement that steps it and what that needs, in one sentence on either device.

    Water under GENERIC falls to its ``"*"``, VBD, and a GPU refuses that too
    (the water's own build names Genesis), so a CPU's refusal that said to build
    on a GPU sent the reader to a second refusal. A heap or a liquid on
    WATER_60HZ's VBD died in a Warp copy error on a GPU, and a brush on
    MUJOCO_500HZ with a ``KeyError`` inside MuJoCo's converter.
    """

    @dataclasses.dataclass(frozen=True)
    class _Request:
        physics: Any
        specs: tuple = ()

        def build_order(self) -> tuple:
            return self.specs

    def refusal(self, arrangement, thing, device: str) -> str:
        from robosimstudio.machine import Platform
        from robosimstudio.scene.compiler import _for_platform
        from robosimstudio.scene.physics import CompositionError

        plat = Platform(device=device, device_why="test", renderer="tiled", renderer_why="test")
        with self.assertRaises(CompositionError) as caught:
            _for_platform(self._Request(arrangement, (thing,)), plat, 1)
        return str(caught.exception)

    def test_the_same_route_on_a_cpu_and_on_a_gpu(self) -> None:
        import robosimstudio as rss
        from robosimstudio.specs.physics import get_domain

        for kind, make, off in _OFF_ITS_ENGINE:
            with self.subTest(kind=kind, arrangement=off):
                info, arrangement = get_domain(kind), getattr(rss, off)
                gpu = self.refusal(arrangement, make(rss), "cuda:0")
                on = arrangement.backend_for(kind).solver
                self.assertTrue(
                    gpu.startswith(
                        f"'it' ({kind}) is on {on!r}, which steps no {kind} on any device: route the {kind!r} kind "
                        f"to backend({info.engine!r}), as rss.{info.staged_by} does (it needs a GPU"
                    ),
                    gpu,
                )
                self.assertEqual(self.refusal(arrangement, make(rss), "cpu"), f"this scene is on a CPU. {gpu}")

    def test_water_names_the_extra_as_its_own_build_does(self) -> None:
        import newton

        import robosimstudio as rss
        from robosimstudio.scene.registry import resolve_entity_class

        route = self.refusal(rss.GENERIC, rss.water(name="it"), "cuda:0").split(": ", 1)[1].removesuffix(".")
        self.assertEqual(
            route,
            "route the 'water' kind to backend('genesis'), as rss.WATER_60HZ does (it needs a GPU and the "
            "[genesis] extra)",
        )
        spec = rss.water(name="it")
        with self.assertRaises(NotImplementedError) as caught:
            resolve_entity_class(spec, rss.GENERIC)(spec).build(newton.ModelBuilder(), None)
        self.assertIn(route, str(caught.exception), "the entity's build refusal says what the compiler's does")

    def test_on_its_engine_a_gpu_builds_it_and_a_cpu_is_told_to_build_on_one(self) -> None:
        import robosimstudio as rss
        from robosimstudio.machine import Platform
        from robosimstudio.scene.compiler import _for_platform
        from robosimstudio.specs.physics import get_domain

        gpu = Platform(device="cuda:0", device_why="test", renderer="tiled", renderer_why="test")
        for kind, make, _ in _OFF_ITS_ENGINE:
            with self.subTest(kind=kind):
                info = get_domain(kind)
                staged = getattr(rss, info.staged_by)
                self.assertEqual(staged.backend_for(kind).solver, info.engine, "the route it names steps it")
                request = self._Request(staged, (make(rss),))
                self.assertEqual(_for_platform(request, gpu, 1), (request, ()))
                extra = " with the [genesis] extra" if kind == "water" else ""
                self.assertTrue(
                    self.refusal(staged, make(rss), "cpu").endswith(f"Build it on a GPU{extra}."), (kind, extra)
                )

    def test_a_dotted_path_an_alias_of_the_engine_and_a_kind_with_no_engine_pass(self) -> None:
        """Left to the solver and the composer: an explicit class, the engine by another name, a cloth on MuJoCo."""
        from robosimstudio.scene.physics.devices import misrouted
        from robosimstudio.specs.physics import MUJOCO_500HZ, backend, register_backend

        register_backend("test_platform_vbd_alias", cls="newton.solvers.SolverVBD", devices=("cuda",))
        for spec in (backend("newton.solvers.SolverXPBD"), backend("test_platform_vbd_alias")):
            self.assertEqual(misrouted(MUJOCO_500HZ.with_backend("paper", spec), {"paper": ["p"]}), (), spec)
        self.assertEqual(misrouted(MUJOCO_500HZ, {"cloth": ["towel"]}), ())


class TestTheRegisteredFacts(unittest.TestCase):
    """What each backend says about devices is data, compared like the rest of its registration."""

    def test_the_shipped_backends(self) -> None:
        from robosimstudio.specs.physics import get_backend

        mujoco = get_backend("mujoco")
        self.assertEqual(mujoco.devices, ("cuda", "cpu"))
        self.assertEqual(dict(mujoco.cpu_params), {"use_mujoco_cpu": True, "use_mujoco_contacts": True})
        self.assertEqual(mujoco.cpu_max_worlds, 1)
        self.assertEqual(get_backend("vbd").devices, ("cuda",))
        self.assertEqual(get_backend("genesis").devices, ("cuda",))
        self.assertIn("cpu", get_backend("superdex").devices)

    def test_a_reregistration_differing_only_in_devices_is_refused(self) -> None:
        from robosimstudio.specs.physics import register_backend

        register_backend("test_platform_facts", cls="my.solvers.Facts", devices=("cuda",))
        register_backend("test_platform_facts", cls="my.solvers.Facts", devices=("cuda",))
        with self.assertRaises(ValueError):
            register_backend("test_platform_facts", cls="my.solvers.Facts", devices=("cuda", "cpu"))
        with self.assertRaisesRegex(ValueError, "devices are 'cuda' and 'cpu'"):
            register_backend("test_platform_facts_2", cls="my.solvers.Facts", devices=("tpu",))

    def test_cpu_is_a_flat_name_and_a_bench_name(self) -> None:
        import robosimstudio as rss
        from robosimstudio.specs.physics import CPU
        from robosimstudio_bench.taskbuild.arrangements import get_arrangement

        self.assertIs(rss.CPU, CPU)
        self.assertIs(get_arrangement("cpu@v1"), CPU)
        self.assertTrue(callable(rss.platform))
        self.assertEqual(
            {domain for domain, spec in CPU.backends if spec.solver == "superdex"}, {"soft", "cloth", "bag", "rod"}
        )
        self.assertEqual(CPU.backend_for("rod").get("feedback"), "off", "a two-way cable slips out of a hand")

    def test_the_mujoco_solver_is_found_by_class(self) -> None:
        import newton

        from robosimstudio.scene.physics.compose import PhysicsProgram

        mujoco = MagicMock(spec=newton.solvers.SolverMuJoCo)
        program = types.SimpleNamespace(
            islands=[types.SimpleNamespace(name="superdex"), types.SimpleNamespace(name="arm")],
            solver_of_island=lambda name: mujoco if name == "arm" else object(),
        )
        self.assertIs(PhysicsProgram.mujoco_solver.fget(program), mujoco)


class TestMuJoCosCData(unittest.TestCase):
    """The C data a CPU build steps, read in MuJoCo-Warp's convention, on a real MuJoCo model and no Warp."""

    XML = """
    <mujoco><option timestep="0.002"/>
      <worldbody>
        <geom name="floor" type="plane" size="1 1 0.1"/>
        <body name="box" pos="0 0 0.1"><freejoint/><geom type="box" size="0.05 0.05 0.05" mass="2"/></body>
      </worldbody>
    </mujoco>"""

    def solver(self):
        import mujoco

        model = mujoco.MjModel.from_xml_string(self.XML)
        data = mujoco.MjData(model)
        for _ in range(500):
            mujoco.mj_step(model, data)
        return types.SimpleNamespace(use_mujoco_cpu=True, mj_model=model, mj_data=data)

    def test_a_resting_box_is_held_up_by_its_weight(self) -> None:
        """The force on each pair's second geom, in world axes: the floor holds the 2 kg box up with 19.6 N."""
        from robosimstudio.core import mjdata

        geoms, forces, dist = mjdata.contacts(self.solver())
        self.assertGreater(len(geoms), 0)
        self.assertTrue(np.all(geoms[:, 1] == 1), geoms)  # the box is every pair's second geom
        np.testing.assert_allclose(forces.sum(axis=0), [0.0, 0.0, 2.0 * 9.81], atol=0.05)
        self.assertTrue(np.all(dist < 1e-3))

    def test_options_and_gravity_compensation_land_on_the_stepped_model(self) -> None:
        from robosimstudio.core import mjdata

        solver = self.solver()
        solver.mjw_model = None  # nothing to mirror into
        self.assertTrue(mjdata.set_option(solver, "iterations", 7))
        self.assertEqual(solver.mj_model.opt.iterations, 7)
        self.assertFalse(mjdata.set_option(solver, "not_an_option", 1))
        gravcomp = types.SimpleNamespace(assign=MagicMock())
        solver.mjw_model = types.SimpleNamespace(body_gravcomp=gravcomp)
        table = np.array([[0.0, 1.0]], dtype=np.float32)
        mjdata.write_gravcomp(solver, table)
        gravcomp.assign.assert_called_once_with(table)
        self.assertEqual(mjdata.read_gravcomp(solver, 0, 1), 1.0)
        with self.assertRaises(IndexError):
            mjdata.read_gravcomp(solver, 1, 1)


class TestWhatANotifyLeavesOutOfTheCModel(unittest.TestCase):
    """Newton regenerates a shape's friction into MuJoCo-Warp's model and stops there; the C model needs it too."""

    def solver(self, *, cpu: bool = True):
        import mujoco

        model = mujoco.MjModel.from_xml_string(TestMuJoCosCData.XML)
        friction = np.tile(np.array([[0.05, 0.005, 0.0001]]), (model.ngeom, 1))[None]  # [world, geom, 3]
        warp_side = types.SimpleNamespace(geom_friction=types.SimpleNamespace(numpy=lambda: friction))
        return types.SimpleNamespace(use_mujoco_cpu=cpu, mj_model=model, mjw_model=warp_side)

    def test_a_shape_notify_reaches_the_c_model(self) -> None:
        import newton

        from robosimstudio.core.mjdata import mirror_notify

        solver = self.solver()
        self.assertEqual(mirror_notify(solver, int(newton.ModelFlags.SHAPE_PROPERTIES)), 1)
        np.testing.assert_allclose(solver.mj_model.geom_friction[:, 0], 0.05)

    def test_other_flags_and_a_gpu_copy_nothing(self) -> None:
        import newton

        from robosimstudio.core.mjdata import mirror_notify

        self.assertEqual(mirror_notify(self.solver(), int(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)), 0)
        self.assertEqual(mirror_notify(self.solver(cpu=False), int(newton.ModelFlags.SHAPE_PROPERTIES)), 0)


class TestTheRestFollowsTheDevice(unittest.TestCase):
    def test_a_joint_notify_carries_the_mocap_poses_into_the_c_data(self) -> None:
        import newton

        from robosimstudio.core import mjdata

        solver = types.SimpleNamespace(use_mujoco_cpu=True)
        with patch.object(mjdata, "sync_mocap") as sync, patch.object(mjdata, "mirror_notify", return_value=0):
            mjdata.after_notify(solver, int(newton.ModelFlags.JOINT_PROPERTIES))
            mjdata.after_notify(solver, int(newton.ModelFlags.SHAPE_PROPERTIES))
        sync.assert_called_once_with(solver)

    def test_a_plan_streams_at_the_rate_the_scene_was_built_at(self) -> None:
        """On a CPU a GENERIC task steps at CPU's 50 Hz while its document says 60."""
        from robosimstudio_policy_agent.planners.curobo import MotionBackendCuRobo

        task = types.SimpleNamespace(physics=types.SimpleNamespace(control_hz=60.0))
        built = types.SimpleNamespace(
            env=types.SimpleNamespace(task=task, pipeline=types.SimpleNamespace(control_hz=50.0))
        )
        self.assertEqual(MotionBackendCuRobo._stream_hz(built), 50.0)
        self.assertEqual(
            MotionBackendCuRobo._stream_hz(types.SimpleNamespace(env=types.SimpleNamespace(task=task))), 60.0
        )

    def test_the_device_clock_times_a_cpu_by_the_wall(self) -> None:
        from robosimstudio.core.profile import DeviceClock

        clock = DeviceClock(device="cpu")
        with clock.measure():
            pass
        self.assertEqual((clock.samples, clock.dropped), (1, 0))

    def test_runs_and_the_bench_script_default_to_auto(self) -> None:
        from robosimstudio.specs import JobSpec

        self.assertEqual(JobSpec(name="t", task="t").device, "auto")
        spec = importlib.util.spec_from_file_location("bench_run_platform", "scripts/bench_run.py")
        run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run)
        self.assertEqual(run.create_parser().parse_args(["--task", "t"]).device, "auto")

    def test_the_runner_knows_the_cpu_tier(self) -> None:
        spec = importlib.util.spec_from_file_location("testrun_platform", "scripts/testrun.py")
        run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run)
        self.assertIn("cpu", run.TIERS)
        # Both no-GPU tiers hide the GPUs, so a GPU machine runs what CI's runners do.
        self.assertEqual(set(run.NO_GPU_TIERS), {"fast", "cpu"})
        self.assertEqual(run.NO_GPU["CUDA_VISIBLE_DEVICES"], "")


if __name__ == "__main__":
    unittest.main()
