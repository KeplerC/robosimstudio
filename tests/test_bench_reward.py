# SPDX-License-Identifier: Apache-2.0
"""The reward clause: each clause's arithmetic, the closed clause set, masked to the
episode's active set, labelled apart from the goal, its gate refused at zero, and a
non-finite value read as nothing."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest import mock

from robosimstudio.scene.state import WorldState
from robosimstudio_bench.goals import evaluate
from robosimstudio_bench.goals.combinators import Leaf
from robosimstudio_bench.goals.context import EvalResult, diverged_result, finite_reward
from robosimstudio_bench.goals.reward import Fraction, Gated, Progress, Scored, _clip, build_reward
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


_CTX = WorldState(handles=None, state=None, sim_time=0.0)
_VERDICT = EvalResult(success=False, reward=0.0, metrics={})


def _measuring(value: float, *, ok: bool = True):
    """A node maker whose every node reports ``depth = value`` and holds when ``ok``."""
    return lambda form: Leaf(fn=lambda ctx: (ok, 1.0, {"depth": value}))


def _score(word: str, value: float, gate: float = 0.02) -> float:
    return build_reward({word: [["depth", "plug"], [":gate", gate]]}, _measuring(value)).score(_VERDICT, _CTX)[0]


class TestTheRewardArithmetic(unittest.TestCase):
    """Each clause's number, from its definition."""

    def test_progress_is_the_measure_over_its_gate_clipped_to_one(self):
        self.assertEqual([_score("progress", v) for v in (0.0, 0.005, 0.01, 0.03)], [0.0, 0.25, 0.5, 1.0])

    def test_shrinking_is_one_less_the_share_clipped_at_zero(self):
        self.assertEqual([_score("shrinking", v) for v in (0.0, 0.005, 0.01, 0.03)], [1.0, 0.75, 0.5, 0.0])

    def test_the_two_directions_sum_to_one_inside_the_gate(self):
        for value in (0.0, 0.003, 0.0125, 0.02):
            with self.subTest(value=value):
                self.assertAlmostEqual(_score("progress", value) + _score("shrinking", value), 1.0)

    def test_a_false_mask_pays_nothing_and_a_true_one_pays_the_progress(self):
        section = {"progress": [["depth", "plug"], [":gate", 0.02], [":gated-by", ["intact", "film"]]]}
        for ok, paid in ((False, 0.0), (True, 0.5)):
            with self.subTest(mask=ok):
                reward, metrics = build_reward(section, _measuring(0.01, ok=ok)).score(_VERDICT, _CTX)
                self.assertEqual(reward, paid)
                self.assertIs(metrics["reward_gate"], ok)

    def test_a_mask_about_nothing_in_the_episode_drops_the_gate(self):
        section = {"progress": [["depth", "plug"], [":gate", 0.02], [":gated-by", ["intact", "film"]]]}
        clause = build_reward(section, _measuring(0.01), make_mask=lambda form: None)
        self.assertIsInstance(clause, Progress)

    def test_fraction_is_the_goals_own_reward_clipped(self):
        for section in ({}, {"fraction": []}):
            clause = build_reward(section, _measuring(0.0))
            self.assertIsInstance(clause, Fraction)
            for given, paid in ((0.25, 0.25), (1.5, 1.0), (-0.5, 0.0), (float("nan"), 0.0)):
                with self.subTest(section=section, given=given):
                    self.assertEqual(clause.score(EvalResult(success=False, reward=given), _CTX)[0], paid)

    def test_a_metric_the_measure_did_not_report_raises_naming_what_it_did(self):
        clause = Progress(of=Leaf(fn=lambda ctx: (True, 1.0, {"lat": 0.1})), key="depth", gate=0.02)
        with self.assertRaisesRegex(KeyError, r"'depth', but the measure reported \['lat'\]"):
            clause.score(_VERDICT, _CTX)

    def test_a_gate_built_by_hand_at_zero_is_refused_when_scored(self):
        clause = Progress(of=Leaf(fn=lambda ctx: (True, 1.0, {"depth": 0.1})), key="depth", gate=0.0)
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            clause.score(_VERDICT, _CTX)


class TestTheClauseSetIsClosed(unittest.TestCase):
    def test_an_unknown_clause_is_refused(self):
        with self.assertRaisesRegex(ValueError, "the clause set is closed"):
            build_reward({"bonus": [["depth", "plug"]]}, _measuring(0.0))

    def test_progress_needs_a_gate_and_one_measure(self):
        cases = (
            ([["depth", "plug"]], "no \\(:gate V\\)"),
            ([[":gate", 0.1]], "names no measure"),
            ([["depth", "plug"], ["lat", "plug"], [":gate", 0.1]], "was given a second"),
        )
        for body, refusal in cases:
            with self.subTest(body=body), self.assertRaisesRegex(ValueError, refusal):
                build_reward({"progress": body}, _measuring(0.0))


class TestScoredEvaluatesEachNodeOncePerStep(unittest.TestCase):
    def test_the_reward_measure_runs_once_and_observations_gate_nothing(self):
        calls: list[int] = []

        def measure(ctx):
            calls.append(1)
            return True, 1.0, {"depth": 0.01}

        scored = Scored(
            goal=Leaf(fn=lambda ctx: (False, 0.0, {})),
            reward=Progress(of=Leaf(fn=measure), key="depth", gate=0.02),
            observe=(Leaf(fn=lambda ctx: (True, 1.0, {"seen": 3.0})),),
        )
        result = scored(_CTX)
        self.assertEqual(len(calls), 1)
        self.assertEqual((result.success, result.reward), (False, 0.5))
        self.assertEqual((result.metrics["seen"], result.metrics["depth"]), (3.0, 0.01))

    def test_check_start_has_no_answer_without_a_document(self):
        self.assertIsNone(Scored(goal=Leaf(fn=lambda ctx: (True, 1.0, {}))).check_start(_CTX))


if __name__ == "__main__":
    unittest.main()
