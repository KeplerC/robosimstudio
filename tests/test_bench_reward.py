# SPDX-License-Identifier: Apache-2.0
"""The reward clause: masked to the episode's active set, labelled apart from the goal,
its gate refused at zero, and a non-finite value read as nothing."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest import mock

from robosimstudio_bench.goals import evaluate
from robosimstudio_bench.goals.combinators import Leaf
from robosimstudio_bench.goals.context import diverged_result, finite_reward
from robosimstudio_bench.goals.reward import Gated, Progress, _clip, build_reward
from robosimstudio_bench.taskdoc.parse import parse_problem
from robosimstudio_bench.taskdoc.reader import parse_sexp
from robosimstudio_bench.taskdoc.typecheck import _check_reward

TIER = "fast"
"""Leaves and scripted contexts: no scene, no GPU."""


TWO = """
(define (problem two)
  (:objects a b - block)
  (:regions (square (:ranges ((0.05 -0.20 0.15 -0.10)))))
  (:goal (and (>= (containment a square) 1.0) (>= (containment b square) 1.0)))
  (:reward (progress (containment REWARDED square) (:gate 1.0) (:gated-by (<= (speed GATED) 0.02)))))
"""


def _fake_bind(condition, doc, catalog, task=""):
    """A leaf that remembers which object it was built about."""
    del doc, catalog, task
    name = condition.args[0].base
    ok = condition.head != "speed"
    return Leaf(fn=lambda ctx: (ok, 0.5, {"fraction": 0.5}), about=(condition.head, (name,)))


def _evaluator(rewarded, gated, active):
    doc = parse_problem(parse_sexp(TWO.replace("REWARDED", rewarded).replace("GATED", gated)))
    with mock.patch.object(evaluate, "bind", _fake_bind):
        return evaluate.make_evaluator(doc, active=active)


class TestRewardIsMaskedToTheEpisode(unittest.TestCase):
    def test_a_measure_about_a_parked_object_is_refused(self):
        with self.assertRaisesRegex(ValueError, "outside this episode's active set"):
            _evaluator("b", "b", active={"a"})

    def test_a_gate_about_a_parked_object_is_dropped(self):
        scored = _evaluator("a", "b", active={"a"})
        self.assertIsInstance(scored.reward, Progress)
        self.assertEqual(scored.reward.of.about, ("containment", ("a",)))

    def test_an_active_gate_is_kept(self):
        scored = _evaluator("a", "a", active={"a"})
        self.assertIsInstance(scored.reward, Gated)
        self.assertEqual(scored.reward.by.about, ("speed", ("a",)))

    def test_unmasked_without_an_active_set(self):
        scored = _evaluator("b", "b", active=None)
        self.assertIsInstance(scored.reward, Gated)
        self.assertEqual(scored.reward.of.of.about, ("containment", ("b",)))


class TestRewardNodesHaveTheirOwnLabels(unittest.TestCase):
    def test_reward_verdicts_do_not_collide_with_the_goal(self):
        scored = _evaluator("a", "a", active=None)
        self.assertEqual(scored.reward.of.of.label, "reward")
        self.assertEqual(scored.reward.by.label, "reward.gated_by")
        metrics = scored(None).metrics
        self.assertIs(metrics["reward"], True)
        self.assertIs(metrics["reward.gated_by"], False)
        self.assertIs(metrics["goal"], True)


class TestRewardGate(unittest.TestCase):
    """A gate <= 0 is refused at build time, where it used to score 1.0 every step."""

    @staticmethod
    def _section(gate):
        return {"progress": [["depth", "plug"], [":gate", gate]]}

    def test_build_refuses_zero_and_negative_gates(self):
        for gate in (0, 0.0, -0.5):
            with self.subTest(gate=gate), self.assertRaisesRegex(ValueError, "must be > 0"):
                build_reward(self._section(gate), lambda form: Leaf(fn=lambda ctx: (True, 1.0, {})))

    def test_a_positive_gate_builds(self):
        clause = build_reward(self._section(0.02), lambda form: Leaf(fn=lambda ctx: (True, 1.0, {})))
        self.assertIsInstance(clause, Progress)

    def test_the_checker_reports_it(self):
        doc = SimpleNamespace(reward={"progress": [[":gate", 0]]})
        problems = _check_reward(doc, catalog=None)
        self.assertTrue(any("must be a number > 0" in str(p) for p in problems), problems)


class TestNonFinite(unittest.TestCase):
    def test_clip_reads_nan_as_zero(self):
        self.assertEqual(_clip(float("nan")), 0.0)
        self.assertEqual(_clip(0.4), 0.4)

    def test_finite_reward(self):
        self.assertEqual(finite_reward(float("nan")), 0.0)
        self.assertEqual(finite_reward(math.inf), 0.0)
        self.assertEqual(finite_reward(0.25), 0.25)

    def test_diverged_result(self):
        result = diverged_result()
        self.assertEqual((result.success, result.reward, result.metrics), (False, 0.0, {"diverged": True}))


if __name__ == "__main__":
    unittest.main()
