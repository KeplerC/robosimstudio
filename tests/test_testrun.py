# SPDX-License-Identifier: Apache-2.0
"""The tier runner: what it selects, how it runs a tier in parallel, and what it refuses."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import pathlib
import tempfile
import textwrap
import time
import unittest
from unittest import mock

TIER = "fast"
"""No GPU: the runner over a throwaway tree of tiny test modules."""

_SPEC = importlib.util.spec_from_file_location(
    "testrun_under_test", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "testrun.py"
)
run = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run)


@contextlib.contextmanager
def tree(modules: dict[str, str]):
    """A repo root with ``tests/<name>.py`` for each entry, and the runner pointed at it."""
    with tempfile.TemporaryDirectory() as root:
        tests = pathlib.Path(root) / "tests"
        tests.mkdir()
        (tests / "__init__.py").write_text("")
        for name, body in modules.items():
            (tests / f"{name}.py").write_text(textwrap.dedent(body))
        with mock.patch.object(run, "ROOT", pathlib.Path(root)), mock.patch.object(run, "TESTS", tests):
            yield pathlib.Path(root)


PASSING = """
    import unittest
    TIER = "fast"
    class TestOne(unittest.TestCase):
        def test_a(self): pass
        def test_b(self): self.skipTest("not here")
    class TestTwo(unittest.TestCase):
        def test_c(self): pass
"""
FAILING = """
    import unittest
    TIER = "fast"
    class TestBroken(unittest.TestCase):
        def test_fails(self): self.assertEqual(1, 2)
        def test_errors(self): raise RuntimeError("boom")
"""
CARD = """
    import os, unittest
    TIER = "medium"
    class TestCard(unittest.TestCase):
        def test_card(self): print("card=" + os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
"""


class TestReadingAnOutcome(unittest.TestCase):
    def test_a_failed_run_counts_its_failures_errors_and_skips(self) -> None:
        got = run.read_outcome("m", 1, 0.5, "...\nRan 5 tests in 0.01s\n\nFAILED (failures=1, errors=2, skipped=1)\n")
        self.assertEqual((got.ran, got.failures, got.errors, got.skipped, got.ok), (5, 1, 2, 1, False))

    def test_a_clean_run_counts_its_skips(self) -> None:
        got = run.read_outcome("m", 0, 0.5, "Ran 3 tests in 0.01s\n\nOK (skipped=2)\n")
        self.assertEqual((got.ran, got.failures, got.skipped, got.ok), (3, 0, 2, True))

    def test_a_run_that_died_before_reporting_ran_nothing_and_failed(self) -> None:
        got = run.read_outcome("m", 1, 0.1, "ImportError: No module named 'x'\n")
        self.assertEqual((got.ran, got.ok), (0, False))

    def test_only_the_last_summary_counts(self) -> None:
        # A test that prints a unittest summary of its own (a subprocess's) must not be read as the run's.
        text = "Ran 9 tests in 1s\nFAILED (failures=9)\nRan 2 tests in 0.1s\n\nOK\n"
        got = run.read_outcome("m", 0, 0.1, text)
        self.assertEqual((got.ran, got.failures), (2, 0))


class TestSelection(unittest.TestCase):
    def test_classes_are_read_without_importing_the_module(self) -> None:
        with tree({"test_x": textwrap.dedent(PASSING) + "raise SystemExit('imported')\n"}):
            self.assertEqual(run.classes("tests.test_x"), ["tests.test_x.TestOne", "tests.test_x.TestTwo"])

    def test_a_class_on_a_local_base_is_listed_and_a_test_less_helper_is_not(self) -> None:
        body = """
            import unittest
            from somewhere import ImportedCase
            TIER = "fast"
            class _Instant(unittest.TestCase):
                def helper(self): pass
            class TestOnTheHelper(_Instant):
                def test_x(self): pass
            class _Mixin:
                def test_shared(self): pass
            class TestMixedIn(_Mixin, unittest.TestCase):
                pass
            class TestOnAnImportedBase(ImportedCase):
                pass
            class NotATest:
                def test_looks_like_one(self): pass
            class _ProbeEntity(ImportedEntity):
                def step(self): pass
        """
        with tree({"test_x": body}):
            self.assertEqual(
                run.classes("tests.test_x"),
                ["tests.test_x.TestOnTheHelper", "tests.test_x.TestMixedIn", "tests.test_x.TestOnAnImportedBase"],
            )

    def test_a_module_with_no_test_case_runs_whole(self) -> None:
        with tree({"test_x": 'TIER = "fast"\n'}):
            self.assertEqual(run.classes("tests.test_x"), ["tests.test_x"])

    def test_list_with_a_tier_prints_only_that_tier(self) -> None:
        with tree({"test_a": PASSING, "test_card": CARD}):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(run.main(["--list", "--tier", "fast"]), 0)
        self.assertIn("tests.test_a", out.getvalue())
        self.assertNotIn("medium", out.getvalue())
        self.assertNotIn("tests.test_card", out.getvalue())

    def test_a_module_without_a_tier_stops_everything(self) -> None:
        with tree({"test_a": PASSING, "test_untiered": "import unittest\n"}):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(run.main(["--tier", "fast"]), 1)
        self.assertIn("test_untiered", err.getvalue())


class TestNothingToRun(unittest.TestCase):
    def test_a_run_that_found_no_tests_is_not_a_failure(self) -> None:
        got = run.read_outcome("m", run.NO_TESTS_RAN, 0.1, "Ran 0 tests in 0.000s\n\nNO TESTS RAN\n")
        self.assertTrue(got.ok)

    def test_a_helper_base_run_alone_passes(self) -> None:
        helper = """
            import unittest
            TIER = "fast"
            class _Helper(unittest.TestCase):
                def helper(self): pass
        """
        with tree({"test_h": helper}):
            outcomes = run.run_parallel(["tests.test_h._Helper"], 1, None, [], stream=io.StringIO())
        self.assertTrue(outcomes[0].ok, outcomes[0].output)


class TestRefusals(unittest.TestCase):
    def _main(self, *argv: str) -> tuple[int, str]:
        err = io.StringIO()
        with tree({"test_a": PASSING, "test_card": CARD}), contextlib.redirect_stderr(err):
            return run.main(list(argv)), err.getvalue()

    def test_a_gpu_tier_in_parallel_needs_a_card_per_worker(self) -> None:
        code, err = self._main("--tier", "medium", "-j", "4")
        self.assertEqual(code, 2)
        self.assertIn("--gpus", err)

    def test_a_card_named_twice_is_refused(self) -> None:
        code, err = self._main("--tier", "medium", "--gpus", "0,0")
        self.assertEqual(code, 2)
        self.assertIn("twice", err)

    def test_cards_are_refused_for_a_tier_that_hides_them(self) -> None:
        code, err = self._main("--tier", "fast", "--gpus", "0,1")
        self.assertEqual(code, 2)
        self.assertIn("hidden", err)


class TestRunningInParallel(unittest.TestCase):
    def test_every_module_runs_once_and_the_failures_are_named(self) -> None:
        with tree({"test_a": PASSING, "test_b": FAILING}):
            log = io.StringIO()
            outcomes = run.run_parallel(["tests.test_a", "tests.test_b"], 2, None, [], stream=log)
            out = io.StringIO()
            code = run.report(outcomes, 1.0, 5, stream=out)
        self.assertEqual([o.name for o in outcomes], ["tests.test_a", "tests.test_b"])
        self.assertEqual(code, 1)
        self.assertIn("Ran 5 tests in 2 processes", out.getvalue())
        self.assertIn("1 failures, 1 errors, 1 skipped", out.getvalue())
        self.assertIn("FAILED: tests.test_b", out.getvalue())
        self.assertIn("boom", out.getvalue(), "a failing module's own output is printed")

    def test_a_clean_run_exits_zero(self) -> None:
        with tree({"test_a": PASSING}):
            outcomes = run.run_parallel(["tests.test_a"], 1, None, [], stream=io.StringIO())
        self.assertEqual(run.report(outcomes, 1.0, 0, stream=io.StringIO()), 0)

    def test_each_worker_keeps_its_own_card(self) -> None:
        names = [f"tests.test_card_{i}" for i in range(6)]
        with tree({name.split(".")[1]: CARD for name in names}):
            outcomes = run.run_parallel(names, 0, None, [], gpus=["7", "3"], stream=io.StringIO())
        cards = {o.output.split("card=")[1].split()[0] for o in outcomes}
        self.assertTrue(cards <= {"7", "3"}, cards)
        self.assertEqual(len(outcomes), 6)

    def test_a_worker_forgets_the_shells_device_for_its_own_card(self) -> None:
        device = CARD.replace(
            'os.environ.get("CUDA_VISIBLE_DEVICES", "unset")', 'repr(os.environ.get("ROBOSIMSTUDIO_DEVICE"))'
        )
        with tree({"test_dev": device}), mock.patch.dict(os.environ, {"ROBOSIMSTUDIO_DEVICE": "cuda:3"}):
            outcomes = run.run_parallel(["tests.test_dev"], 0, None, [], gpus=["1"], stream=io.StringIO())
        self.assertIn("card=''", outcomes[0].output)

    def test_a_run_past_its_timeout_is_killed_and_shows_what_it_printed(self) -> None:
        hang = """
            import time, unittest
            TIER = "fast"
            class TestHang(unittest.TestCase):
                def test_hang(self):
                    print("got this far", flush=True)
                    time.sleep(60)
        """
        with tree({"test_hang": hang}):
            start = time.monotonic()
            outcomes = run.run_parallel(["tests.test_hang"], 1, None, [], stream=io.StringIO(), timeout=3.0)
        self.assertLess(time.monotonic() - start, 30.0)
        self.assertEqual(outcomes[0].code, run.TIMED_OUT)
        self.assertFalse(outcomes[0].ok)
        self.assertIn("got this far", outcomes[0].output)
        self.assertIn("killed after 3 s", outcomes[0].output)

    def test_classes_run_as_their_own_processes(self) -> None:
        with tree({"test_a": PASSING}):
            names = run.classes("tests.test_a")
            outcomes = run.run_parallel(names, 2, None, [], stream=io.StringIO())
        self.assertEqual(sorted(o.ran for o in outcomes), [1, 2])


class TestTheTiersThatHideTheGpu(unittest.TestCase):
    def test_a_no_gpu_tier_runs_with_every_card_hidden(self) -> None:
        with tree({"test_card": CARD.replace('"medium"', '"fast"')}):
            outcomes = run.run_parallel(["tests.test_card"], 1, {**os.environ, **run.NO_GPU}, [], stream=io.StringIO())
        self.assertIn("card=\n", outcomes[0].output + "\n")


if __name__ == "__main__":
    unittest.main()
