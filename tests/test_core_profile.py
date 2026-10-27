# SPDX-License-Identifier: Apache-2.0
"""The profiler: what a phase cost, what its children left unattributed, and which phase paid for a copy.

``core.profile`` is what a slow loop is diagnosed with, so a wrong number here
sends somebody to optimise the wrong phase. The claims are arithmetic, and they
are checked against a clock the test moves: a parent's unattributed remainder is
its total less its direct children's, a device-to-host copy is charged to the
phase open when it happened (the enclosing one again once a nested phase
closes), the order statistics are of the retained samples, and the sync tracer
puts ``wp.array.numpy`` back exactly as it found it.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from robosimstudio.core import profile
from robosimstudio.core.profile import NULL, PhaseStats, Profiler, SyncTracer, parse_modes
from tests.fixtures.waits import fake_clock

TIER = "fast"
"""Arithmetic on a clock the test moves, and Warp arrays on the CPU: no GPU."""


def _timed(profiler: Profiler, clock, *spans: tuple[str, float]) -> None:
    """Enter and leave each ``(name, seconds)`` in turn, the clock moving ``seconds`` inside it."""
    for name, seconds in spans:
        with profiler.phase(name):
            clock.advance(seconds)


class TestAPhaseIsSummarisedTwoWays(unittest.TestCase):
    def test_calls_total_and_peak_accumulate(self) -> None:
        stats = PhaseStats()
        for seconds in (0.2, 0.5, 0.1):
            stats.record(seconds)
        self.assertEqual(3, stats.calls)
        self.assertAlmostEqual(0.8, stats.total, places=12)
        self.assertEqual(0.5, stats.peak)
        self.assertAlmostEqual(0.8 / 3, stats.as_dict()["mean_s"], places=12)

    def test_the_first_sample_is_the_average_and_later_ones_are_blended(self) -> None:
        stats = PhaseStats()
        stats.record(1.0)
        self.assertEqual(1.0, stats.ema)
        stats.record(2.0)
        self.assertAlmostEqual((1.0 - profile._SMOOTHING) * 1.0 + profile._SMOOTHING * 2.0, stats.ema, places=12)

    def test_the_percentiles_are_of_the_retained_samples(self) -> None:
        stats = PhaseStats()
        self.assertEqual((0.0, 0.0), stats.quantiles(), "nothing recorded is zero, not a raise")
        for seconds in range(1, 101):
            stats.record(float(seconds))
        p50, p95 = stats.quantiles()
        self.assertAlmostEqual(50.5, p50)
        self.assertAlmostEqual(np.percentile(np.arange(1.0, 101.0), 95), p95)

    def test_only_the_newest_samples_are_retained(self) -> None:
        stats = PhaseStats()
        for _ in range(profile._RING):
            stats.record(100.0)
        for _ in range(profile._RING):
            stats.record(1.0)
        self.assertEqual((1.0, 1.0), stats.quantiles(), "the old samples left the ring")
        self.assertEqual(100.0, stats.peak, "the peak is all-time, not the ring's")

    def test_reset_forgets_everything(self) -> None:
        stats = PhaseStats()
        stats.record(3.0)
        stats.reads, stats.read_bytes = 2, 64
        stats.reset()
        self.assertEqual({"calls": 0, "total_s": 0.0, "mean_s": 0.0, "p50_s": 0.0, "p95_s": 0.0, "peak_s": 0.0,
                          "reads": 0, "read_bytes": 0}, stats.as_dict())  # fmt: skip


class TestTheUnattributedRemainder(unittest.TestCase):
    def test_a_parent_less_its_direct_children(self) -> None:
        profiler = Profiler()
        with fake_clock("robosimstudio.core.profile") as clock:
            with profiler.phase("step"):
                _timed(profiler, clock, ("step.physics", 0.6), ("step.render", 0.3))
                clock.advance(0.1)
        self.assertAlmostEqual(1.0, profiler.phases["step"].total, places=12)
        self.assertAlmostEqual(0.1, profiler._unattributed()["step"], places=12)

    def test_a_grandchild_is_its_own_parents_and_not_the_root_s(self) -> None:
        profiler = Profiler()
        with fake_clock("robosimstudio.core.profile") as clock:
            with profiler.phase("step"):
                with profiler.phase("step.physics"):
                    _timed(profiler, clock, ("step.physics.solve", 0.4))
                    clock.advance(0.2)
        left = profiler._unattributed()
        self.assertAlmostEqual(0.0, left["step"], places=12, msg="physics is step's only child")
        self.assertAlmostEqual(0.2, left["step.physics"], places=12)

    def test_a_phase_with_no_children_has_no_remainder(self) -> None:
        profiler = Profiler()
        with fake_clock("robosimstudio.core.profile") as clock:
            _timed(profiler, clock, ("step", 0.5), ("stepper", 0.5))
        self.assertEqual({}, profiler._unattributed(), "a shared prefix is not a child")

    def test_the_report_prints_the_remainder_under_its_parent(self) -> None:
        profiler = Profiler()
        with fake_clock("robosimstudio.core.profile") as clock:
            with profiler.phase("step"):
                _timed(profiler, clock, ("step.physics", 0.6))
                clock.advance(0.4)
            profiler.mark_step()
            profiler.phase("never")  # named and never entered
            lines = profiler.report().splitlines()
        rows = [line.split()[0] for line in lines[3:] if line.strip()]
        self.assertEqual(["step", "(unattributed)", "physics"], rows, "the remainder sits with its parent, a level in")


class TestACopyIsChargedToThePhaseThatMadeIt(unittest.TestCase):
    def test_the_open_phase_pays_and_the_enclosing_one_again_after_a_nested_one(self) -> None:
        profiler = Profiler()
        with profiler.phase("outer"):
            profiler.attribute_read(8, 0.0)
            with profiler.phase("outer.inner"):
                profiler.attribute_read(16, 0.0)
            profiler.attribute_read(32, 0.0)
        profiler.attribute_read(4, 0.0)
        self.assertEqual((2, 40), (profiler.phases["outer"].reads, profiler.phases["outer"].read_bytes))
        self.assertEqual((1, 16), (profiler.phases["outer.inner"].reads, profiler.phases["outer.inner"].read_bytes))
        self.assertEqual(1, profiler.phases["(outside any phase)"].reads)

    def test_the_tracer_counts_a_copy_and_puts_numpy_back(self) -> None:
        from warp._src.types import array as warp_array

        original = warp_array.numpy
        profiler = Profiler({"syncs"})
        values = wp.zeros(10, dtype=wp.float32, device="cpu")
        profiler.start()
        try:
            profiler.start()  # a second install must not wrap the wrapper
            with profiler.phase("read"):
                values.numpy()
        finally:
            profiler.stop()
            profiler.stop()
        self.assertIs(original, warp_array.numpy, "wp.array.numpy is the one it found")
        self.assertTrue(profiler.syncs.available)
        self.assertEqual((1, 40), (profiler.syncs.count, profiler.syncs.nbytes), "one copy of 10 float32")
        self.assertEqual((1, 40), (profiler.phases["read"].reads, profiler.phases["read"].read_bytes))
        values.numpy()
        self.assertEqual(1, profiler.syncs.count, "nothing is counted once it is removed")

    def test_a_tracer_never_installed_removes_nothing(self) -> None:
        from warp._src.types import array as warp_array

        original = warp_array.numpy
        SyncTracer(Profiler()).remove()
        self.assertIs(original, warp_array.numpy)


class TestTheModesAndTheNullProfiler(unittest.TestCase):
    def test_an_empty_spec_is_the_null_profiler(self) -> None:
        self.assertIs(NULL, parse_modes(""))
        self.assertIs(NULL, parse_modes(" , "))

    def test_any_mode_implies_wall_timing(self) -> None:
        self.assertEqual(frozenset({"syncs", "wall"}), parse_modes("syncs").modes)

    def test_an_unknown_mode_is_refused_with_the_known_ones(self) -> None:
        with self.assertRaises(ValueError) as caught:
            parse_modes("wall,gpu")
        self.assertIn("['gpu']", str(caught.exception))
        self.assertIn("'syncs'", str(caught.exception))

    def test_the_null_profiler_records_nothing(self) -> None:
        with NULL.phase("anything"):
            NULL.attribute_read(8, 0.0)
        NULL.mark_step()
        NULL.note(k=1)
        self.assertEqual(({}, 0, {}), (NULL.phases, NULL.steps, NULL.notes))


class TestTheSidecar(unittest.TestCase):
    def test_what_is_written_reads_back_as_the_numbers(self) -> None:
        profiler = Profiler()
        with fake_clock("robosimstudio.core.profile") as clock:
            _timed(profiler, clock, ("step", 0.25), ("step", 0.75))
            profiler.mark_step()
            profiler.note(scene="bench")
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "profile.json"
                profiler.write(str(path))
                data = json.loads(path.read_text())
        self.assertEqual(1, data["steps"])
        self.assertEqual({"scene": "bench"}, data["notes"])
        self.assertEqual(2, data["phases"]["step"]["calls"])
        self.assertAlmostEqual(0.5, data["phases"]["step"]["mean_s"], places=12)
        self.assertIsNone(data["device"])

    def test_reset_discards_the_warm_up(self) -> None:
        profiler = Profiler()
        with fake_clock("robosimstudio.core.profile") as clock:
            _timed(profiler, clock, ("step", 2.0))
            profiler.mark_step()
            profiler.reset()
            self.assertEqual(0.0, profiler.wall)
        self.assertEqual((0, 0), (profiler.steps, profiler.phases["step"].calls))


if __name__ == "__main__":
    unittest.main()
