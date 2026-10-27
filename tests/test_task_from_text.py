# SPDX-License-Identifier: Apache-2.0
"""A task written as a string, the examples that teach it, and a release with graphs only.

``robosimstudio_bench.task`` lowers a document through the same function a folder
of documents goes through; the first class checks what it registers and what it
refuses, in this process and with no scene built, and gives the registries back
as it found them. The release check runs in a subprocess with the scripted
routine module refused, which is the shape of a release that leaves it out: a
module an earlier test imported cannot be un-imported in this one. The last class
holds the task and policy examples to the docstring and length rules the other
examples keep.
"""

from __future__ import annotations

import ast
import contextlib
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from tests.fixtures import taskdocs
from tests.fixtures.registries import restored

TIER = "fast"

ROOT = Path(__file__).resolve().parents[1]

PREAMBLE = "from tests.fixtures.taskdocs import register_cube\nregister_cube()\n"
"""What a subprocess runs first: the cube the document places, registered for the life of the process."""

DOCUMENT = taskdocs.DOCUMENT.replace("(:goal", '(:artifacts (:graph pick "pick_place"))\n  (:goal')
"""The fixture's cube-in-a-square document, with a graph to resolve."""


def _run(script: str, *, refuse: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    """``script`` in a fresh interpreter that scans no task folder, with ``refuse`` unimportable."""
    blocker = textwrap.dedent(
        f"""
        import importlib.abc, sys
        class _Refuse(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if any(name == r or name.startswith(r + ".") for r in {refuse!r}):
                    raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
                return None
        sys.meta_path.insert(0, _Refuse())
        """
    )
    env = {
        **os.environ,
        "ROBOSIMSTUDIO_TASK_ROOTS": "",
        "ROBOSIMSTUDIO_TASK_ROOTS_ONLY": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    return subprocess.run(
        [sys.executable, "-c", blocker + PREAMBLE + textwrap.dedent(script)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


@contextlib.contextmanager
def _stated():
    """This process with ``text_probe_cube`` registered, scanning no task folder, and given back as it was.

    The asset, the suites a string states and the record of which ones it
    stated are all put back on the way out, so a task stated here is gone for
    whatever this process runs next.
    """
    from robosimstudio.assetlib.registry import ASSETS
    from robosimstudio.roots import task_roots
    from robosimstudio_bench.taskbuild import text
    from robosimstudio_bench.taskbuild.registry import SUITES

    with (
        mock.patch.dict(os.environ, {"ROBOSIMSTUDIO_TASK_ROOTS": "", "ROBOSIMSTUDIO_TASK_ROOTS_ONLY": "1"}),
        restored(ASSETS, SUITES),
        mock.patch.object(text, "_FROM_TEXT", set()),
    ):
        if task_roots():
            raise AssertionError(f"a string task's test would scan {task_roots()}")
        taskdocs.register_cube()
        yield


class TestATaskFromAString(unittest.TestCase):
    def test_it_registers_and_its_graph_resolves_beside_the_caller(self) -> None:
        import robosimstudio_bench as bench
        from robosimstudio_bench.taskbuild.artifacts import graph_dir

        with _stated():
            task = bench.task(DOCUMENT, home="/somewhere/else")
            self.assertEqual(task.full_name, "text_probe/text_probe")
            self.assertIs(bench.get_task("text_probe"), task)
            self.assertEqual(graph_dir(task.suite, task.graph_path("pick")), "/somewhere/else/pick_place")
            again = bench.task(DOCUMENT, home="/somewhere/else")
            self.assertIs(bench.get_task("text_probe"), again, "stating it again replaces it")

    def test_without_home_a_script_s_graphs_resolve_beside_it(self) -> None:
        from robosimstudio_bench.taskbuild.artifacts import graph_dir

        with tempfile.TemporaryDirectory(prefix="home-") as tmp, _stated():
            script = Path(tmp) / "probe.py"
            script.write_text("import robosimstudio_bench as bench\ntask = bench.task(DOCUMENT)\n")
            # Run as that file runs, so the caller ``bench.task`` reads off the stack is the script.
            namespace = {"DOCUMENT": DOCUMENT}
            exec(compile(script.read_text(), str(script), "exec"), namespace)
            task = namespace["task"]
            self.assertEqual(graph_dir(task.suite, task.graph_path("pick")), str(Path(tmp).resolve() / "pick_place"))

    def test_a_measure_nothing_implements_is_refused_when_stated(self) -> None:
        import robosimstudio_bench as bench

        bad = DOCUMENT.replace("(containment cube square)", "(sparkle cube)")
        with _stated(), self.assertRaisesRegex(Exception, "sparkle"):
            bench.task(bad)


class TestTheSurface(unittest.TestCase):
    def test_the_lazy_names_are_listed(self) -> None:
        """What the API page is built from: dir() names every flat name, imported or not."""
        import robosimstudio_bench
        import robosimstudio_policy_agent

        self.assertLessEqual({"task", "Env", "rollout", "register_sampler"}, set(dir(robosimstudio_bench)))
        self.assertLessEqual(
            {"SolutionGraph", "SolutionGraphBatched", "check_graph"}, set(dir(robosimstudio_policy_agent))
        )


class TestAReleaseWithGraphsOnly(unittest.TestCase):
    """With ``robosimstudio_bench.solutions.routines`` refused, every door says so in one sentence."""

    ROUTINES = ("robosimstudio_bench.solutions.routines",)

    def test_each_door_refuses_a_routine_and_offers_graphs(self) -> None:
        routine = DOCUMENT.replace('(:graph pick "pick_place")', '(:graph pick "pick_place") (:routine "x.y:Z")')
        done = _run(
            f"""
            import importlib.util
            import robosimstudio_bench as bench
            from robosimstudio_bench.solutions.declared import ROUTINES_ABSENT, declared_solutions, routines_installed
            assert not routines_installed()
            try:
                from robosimstudio_bench.solutions import SolutionRoutine
            except ImportError as exc:
                assert str(exc) == ROUTINES_ABSENT, exc
            else:
                raise AssertionError("SolutionRoutine imported")
            try:
                bench.task({routine!r})
            except Exception as exc:
                assert ROUTINES_ABSENT in str(exc), exc
            else:
                raise AssertionError("a document naming a routine was lowered")
            task = bench.task({DOCUMENT!r})
            assert declared_solutions(task) == ("pick",), declared_solutions(task)
            spec = importlib.util.spec_from_file_location("bench_run", "scripts/bench_run.py")
            run = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run)
            assert "routine" not in run.SOLUTIONS and "scripted" not in run.SOLUTION_ALIASES, run.SOLUTIONS
            import contextlib, io, sys
            sys.argv = ["bench_run.py", "--task", "text_probe", "--solve", "routine"]
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = run.main()
            assert code == 2 and ROUTINES_ABSENT in err.getvalue(), (code, err.getvalue())
            print("ok")
            """,
            refuse=self.ROUTINES,
        )
        self.assertEqual(0, done.returncode, done.stderr[-3000:])
        self.assertIn("ok", done.stdout)


class TestTheTaskAndPolicyExamples(unittest.TestCase):
    """The rules ``tests/test_examples.py`` holds the scene examples to, over these two folders."""

    FOLDERS = ("06_tasks", "07_policies")
    HEADLESS = ("07_policies/03_many_worlds.py",)
    """A batched run of sixteen worlds has no single scene to open."""

    def scripts(self) -> list[str]:
        found = [f"{f}/{p.name}" for f in self.FOLDERS for p in sorted((ROOT / "examples" / f).glob("[0-9]*.py"))]
        self.assertGreaterEqual(len(found), 7)
        return found

    def test_each_is_short_and_runs_alone(self) -> None:
        for name in self.scripts():
            with self.subTest(example=name):
                source = (ROOT / "examples" / name).read_text()
                self.assertLessEqual(len(source.splitlines()), 70)
                self.assertNotIn("argparse", source)
                tree = ast.parse(source)
                main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
                self.assertEqual([], main.args.args)
                if name not in self.HEADLESS:
                    self.assertEqual("rss.view", ast.unparse(main.body[-1].value.func))
                self.assertEqual("__name__ == '__main__'", ast.unparse(tree.body[-1].test))

    def test_each_docstring_is_the_page_it_becomes(self) -> None:
        for name in self.scripts():
            with self.subTest(example=name):
                doc = ast.get_docstring(ast.parse((ROOT / "examples" / name).read_text())) or ""
                blocks = [block.strip() for block in doc.split("\n\n")]
                self.assertTrue(2 <= len(blocks[0].split()) <= 5 and not blocks[0].endswith("."), blocks[0])
                self.assertNotIn("\n", blocks[1])
                self.assertTrue(blocks[1].endswith("."), blocks[1])
                self.assertIn(len(blocks), (2, 3), "at most one paragraph after the sentence")

    def test_no_example_ships_a_task_document_file(self) -> None:
        for folder in self.FOLDERS:
            found = [p for p in (ROOT / "examples" / folder).rglob("*") if p.suffix in (".task", ".suite")]
            self.assertEqual([], found, "a task is a string in its script; no task document ships")


if __name__ == "__main__":
    unittest.main()
