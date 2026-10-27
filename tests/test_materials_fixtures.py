# SPDX-License-Identifier: Apache-2.0
"""The materials area's own fixtures, held to what their docstrings promise.

:func:`tests.fixtures.packs.once` keeps a failure as well as an answer, so a
module whose fresh processes timed out does not start them all again for every
test; :func:`~tests.fixtures.packs.in_parallel` runs a bounded few at a time and
answers in order; and :func:`tests.fixtures.examples.scripts_modules` leaves
neither ``scripts/`` on ``sys.path`` nor a module it loaded from there in
``sys.modules`` -- the shadowing a tier run in one process would otherwise carry
into every module after it.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest

from tests.fixtures.examples import SCRIPTS, scripts_module, scripts_modules
from tests.fixtures.packs import in_parallel, once

TIER = "fast"
"""Threads, a counter and one import from ``scripts/``: no GPU."""


class TestOnce(unittest.TestCase):
    def test_a_failure_is_kept_and_raised_again_without_running_again(self) -> None:
        calls = []

        @once
        def fresh():
            calls.append(1)
            raise TimeoutError("the interpreters did not answer")

        causes = []
        for _ in range(3):
            with self.assertRaisesRegex(RuntimeError, "did not answer") as raised:
                fresh()
            causes.append(raised.exception.__cause__)
        self.assertEqual(1, len(calls))
        self.assertIsInstance(causes[0], TimeoutError)
        self.assertTrue(all(cause is causes[0] for cause in causes), "every ask chains the one kept failure")
        self.assertIsNot(causes[0], None)

    def test_an_answer_is_kept(self) -> None:
        calls = []

        @once
        def fresh():
            calls.append(1)
            return {"scan": len(calls)}

        self.assertIs(fresh(), fresh())
        self.assertEqual(1, len(calls))


class TestInParallel(unittest.TestCase):
    def test_a_bounded_few_run_at_once_and_the_answers_come_back_in_order(self) -> None:
        lock, running, most = threading.Lock(), [0], [0]

        def work(item: int) -> int:
            with lock:
                running[0] += 1
                most[0] = max(most[0], running[0])
            time.sleep(0.02)
            with lock:
                running[0] -= 1
            return item * item

        self.assertEqual([k * k for k in range(18)], in_parallel(work, range(18)))
        self.assertLessEqual(most[0], min(18, max(2, (os.cpu_count() or 1) // 4), 8))
        self.assertGreater(most[0], 1, "and more than one at a time")
        self.assertEqual([], in_parallel(work, []))


class TestScriptsModules(unittest.TestCase):
    def test_nothing_from_scripts_is_left_on_the_path_or_among_the_modules(self) -> None:
        before = set(sys.modules)
        takes = scripts_module("recording.takes")
        self.assertGreater(takes.FOLD_PINCH, 0.0, "the module object stays good")
        self.assertNotIn(str(SCRIPTS), sys.path)
        left = [name for name in set(sys.modules) - before if name.split(".")[0] in ("recording", "calibration")]
        self.assertEqual([], left)

    def test_inside_the_block_a_module_may_import_its_siblings(self) -> None:
        was = "recording.beats" in sys.modules
        with scripts_modules("recording.lockstep") as (lockstep,):
            self.assertIn("recording", sys.modules)
            from recording import beats  # what lockstep's own lazy imports do

            self.assertIs(sys.modules["recording.beats"], beats)
            self.assertTrue(hasattr(lockstep, "Film"))
        if not was:
            self.assertNotIn("recording.beats", sys.modules, "and the block takes back what it loaded")


if __name__ == "__main__":
    unittest.main()
