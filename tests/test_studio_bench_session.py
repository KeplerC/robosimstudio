# SPDX-License-Identifier: Apache-2.0
"""Loading a task onto the page: the pairing it resolves to, the fleet it asks for, and what a refusal keeps.

A declared binding is found however its hand is spelled, an untried pairing
reaches the fleet as a spec that declares it, a fleet base that does not
declare the pairing narrows the page instead of raising ``KeyError``, and a
load that was refused keeps the world count it had.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from robosimstudio.robots.registry import get_embodiment, hands_of
from robosimstudio.scene.compiler import CannotReplicate
from robosimstudio.specs import Binding, TaskRequirements, TaskSpec, replace
from robosimstudio_studio import bench_session
from robosimstudio_studio.session import Refused

TIER = "fast"
"""No GPU: the fleet and the compiler are patched out at the point they would build."""


def _quiet():
    return contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())


class LoadTask(unittest.TestCase):
    def _session(self):
        return SimpleNamespace(
            config=SimpleNamespace(
                obs_renderer=None, obs_render_every=1, device=None, max_steps=0, control_mode="joint"
            ),
            num_worlds=1,
            task_panel=None,
            status="",
            _narrowed="",
        )

    def test_a_refused_load_keeps_the_world_count(self) -> None:
        from robosimstudio_studio import bench_session

        session = self._session()
        out, err = _quiet()
        with mock.patch.object(bench_session, "unbuildable", return_value="cannot build"), out, err:
            self.assertFalse(bench_session.load_task(session, "fam/a", 0, num_worlds=64))
        self.assertEqual(session.num_worlds, 1)

    def test_a_failed_fleet_keeps_the_world_count(self) -> None:
        from robosimstudio_studio import bench_session

        session = self._session()
        with (
            mock.patch.object(bench_session, "unbuildable", return_value=""),
            mock.patch.object(bench_session, "load_fleet", return_value=False),
        ):
            self.assertFalse(bench_session.load_task(session, "fam/a", 0, num_worlds=64))
        self.assertEqual(session.num_worlds, 1)

    def test_a_failed_compile_after_narrowing_keeps_the_fleet_width(self) -> None:
        from robosimstudio.scene.compiler import CannotReplicate
        from robosimstudio_studio import bench_session

        session = self._session()
        session.num_worlds = 8
        out, err = _quiet()
        with (
            mock.patch.object(bench_session, "unbuildable", return_value=""),
            mock.patch.object(bench_session, "load_fleet", side_effect=CannotReplicate("no")),
            mock.patch.object(bench_session, "Env", side_effect=RuntimeError("no compile")),
            out,
            err,
        ):
            self.assertFalse(bench_session.load_task(session, "fam/a", 0, num_worlds=16))
        self.assertEqual(session.num_worlds, 8)
        self.assertEqual(session._narrowed, "")


ROOT = Path(__file__).resolve().parents[1]


def _task(**kw):
    native = Binding(scene="bench_0728", embodiment="panda", graphs=("pick",))
    return TaskSpec(name="t", scene="bench_0728", embodiment="panda", bindings=(native,), **kw)


class TestResolvePairing(unittest.TestCase):
    def test_the_own_hand_is_the_declared_key(self):
        task = _task()
        (own,) = hands_of(get_embodiment("panda"))
        resolved, *key = bench_session.resolve_pairing(task, "", "", own)
        self.assertIs(resolved, task)
        self.assertEqual(tuple(key), ("bench_0728", "panda", ""))
        self.assertIs(task.binding_for(*key), task.bindings[0])

    def test_no_overrides_is_the_task_as_it_stands(self):
        task = _task()
        self.assertEqual(bench_session.resolve_pairing(task), (task, "", "", ""))

    def test_an_admitted_pairing_is_a_spec_that_declares_it(self):
        task = _task()
        with mock.patch.object(bench_session, "admits", return_value=(True, "")):
            resolved, *key = bench_session.resolve_pairing(task, "", "panda_robotiq", "")
        self.assertEqual(tuple(key), ("", "", ""))
        self.assertIn(("bench_0728", "panda_robotiq", resolved.gripper), resolved.binding_keys)
        self.assertEqual(len(resolved.bindings), 2)

    def test_a_refused_pairing_raises_refused(self):
        task = _task(requires=TaskRequirements(monolithic=True))
        with self.assertRaises(Refused):
            bench_session.resolve_pairing(task, "", "panda_robotiq", "")

    def test_the_fleet_gets_the_resolved_spec(self):
        task = _task()
        session = SimpleNamespace(
            config=SimpleNamespace(
                fleet_solution="",
                fleet_solution_kwargs={},
                show_worlds=4,
                device=None,
                fleet_loop=False,
            ),
            num_worlds=2,
            viewer=None,
            task_panel=None,
        )
        seen = {}

        def fleet(task, **kw):
            seen.update(task=task, **kw)
            raise RuntimeError("stop here")

        with (
            mock.patch("robosimstudio_studio.fleet_session.FleetSession", fleet),
            mock.patch.object(bench_session, "admits", return_value=(True, "")),
            mock.patch("traceback.print_exc"),
        ):
            ok = bench_session.load_fleet(session, task, 0, "t", 0.0, "", "panda_robotiq", "")
        self.assertFalse(ok)
        self.assertIsInstance(seen["task"], TaskSpec)
        self.assertEqual((seen["scene"], seen["embodiment"], seen["gripper"]), ("", "", ""))
        self.assertEqual(seen["task"].embodiment, "panda_robotiq")


class TestRunScriptGraphClaims(unittest.TestCase):
    def test_the_own_hand_finds_the_declared_graphs(self):
        spec = importlib.util.spec_from_file_location("bench_run_followup", ROOT / "scripts" / "bench_run.py")
        run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run)
        task = _task()
        (own,) = hands_of(get_embodiment("panda"))
        job = SimpleNamespace(
            solution_kwargs={"graph": "pick"},
            allow_undeclared_graph=False,
            task="t",
            scene="",
            embodiment="",
            gripper=own,
        )
        with (
            mock.patch("robosimstudio_bench.taskbuild.registry.get_task", return_value=task),
            mock.patch.object(TaskSpec, "graph_path", return_value="pick"),
        ):
            run.check_graph_claims([job])  # raises SystemExit if the bare binding were found


def _base():
    native = Binding(scene="bench_0728", embodiment="panda", graphs=("pick",))
    return TaskSpec(name="base", scene="bench_0728", embodiment="panda", bindings=(native,))


def _session():
    return SimpleNamespace(
        config=SimpleNamespace(
            fleet_solution="",
            fleet_solution_kwargs={},
            show_worlds=4,
            device=None,
            fleet_loop=False,
        ),
        num_worlds=2,
        viewer=None,
        task_panel=None,
    )


class TestFleetBasePairing(unittest.TestCase):
    def test_an_undeclared_pairing_on_a_fleet_base_cannot_replicate(self):
        base = _base()
        instance = replace(base, name="instance", fleet_base="base")
        fleet = mock.Mock(side_effect=AssertionError("FleetSession must not be reached"))
        with (
            mock.patch("robosimstudio_studio.fleet_session.FleetSession", fleet),
            mock.patch("robosimstudio_bench.taskbuild.registry.get_task", return_value=base),
            mock.patch.object(bench_session, "admits", return_value=(True, "")),
            self.assertRaises(CannotReplicate) as caught,
        ):
            bench_session.load_fleet(_session(), instance, 0, "instance", 0.0, "", "panda_robotiq", "")
        self.assertIn("fleet base 'base'", str(caught.exception))
        fleet.assert_not_called()

    def test_a_pairing_the_base_declares_reaches_the_fleet(self):
        base = _base()
        instance = replace(base, name="instance", fleet_base="base")
        fleet = mock.Mock(side_effect=RuntimeError("stop here"))
        with (
            mock.patch("robosimstudio_studio.fleet_session.FleetSession", fleet),
            mock.patch("robosimstudio_bench.taskbuild.registry.get_task", return_value=base),
            mock.patch("traceback.print_exc"),
        ):
            ok = bench_session.load_fleet(_session(), instance, 0, "instance", 0.0)
        self.assertFalse(ok)
        fleet.assert_called_once()

    def test_a_task_without_a_base_is_not_checked(self):
        build = mock.Mock(side_effect=AssertionError("a task with no base was built as one"))
        with mock.patch("robosimstudio_bench.run.fleet.fleet_build_task", build):
            self.assertIsNone(bench_session.base_declares(_base().with_pairing("", "panda_robotiq", "")))
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
