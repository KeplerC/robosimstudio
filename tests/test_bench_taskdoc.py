# SPDX-License-Identifier: Apache-2.0
"""Task documents: the reader, bindings applied to the reward too, and a suite's name
kept as written."""

from __future__ import annotations

import unittest

from robosimstudio_bench.goals.combinators import Leaf
from robosimstudio_bench.goals.gauntlet import _reward_measure
from robosimstudio_bench.goals.reward import Gated, build_reward, reading_key
from robosimstudio_bench.taskdoc import VAB_INDUSTRIAL_V1, apply_bindings, merge_suite, parse_problem, parse_suite
from robosimstudio_bench.taskdoc.parse import parse_condition
from robosimstudio_bench.taskdoc.reader import parse_sexp

TIER = "fast"
"""Parsing only: no scene, no GPU."""


def _suite(text):
    return parse_suite(parse_sexp(text))


def _problem(text):
    return parse_problem(parse_sexp(text))


class TestApplyBindingsBindsTheReward(unittest.TestCase):
    def test_measure_and_mask_are_bound(self):
        doc = _problem(
            "(define (problem p) (:objects hammer bin) (:goal (In hammer bin))"
            " (:reward (progress (In hammer bin) (:gate 1.0) (:gated-by (In hammer bin)))))"
        )
        bound = apply_bindings(doc, VAB_INDUSTRIAL_V1)
        measure, gate, mask = bound.reward["progress"]
        self.assertEqual(measure, [">=", ["containment", "hammer", "bin"], 1.0])
        self.assertEqual(gate, [":gate", 1.0])
        self.assertEqual(mask, [":gated-by", [">=", ["containment", "hammer", "bin"], 1.0]])
        parsed = parse_condition(mask[1])
        self.assertEqual((parsed.head, parsed.op, parsed.value), ("containment", ">=", 1.0))
        self.assertEqual(parsed, bound.goal)
        # The original is untouched.
        self.assertEqual(doc.reward["progress"][0], ["In", "hammer", "bin"])

    def test_the_bound_reward_reads_the_atom_not_the_comparison(self):
        doc = _problem(
            "(define (problem p) (:objects hammer bin) (:goal (In hammer bin))"
            " (:reward (progress (In hammer bin) (:gate 1.0))))"
        )
        bound = apply_bindings(doc, VAB_INDUSTRIAL_V1)

        def make_node(form):
            return Leaf(fn=lambda ctx: (True, 1.0, {}))

        clause = build_reward(bound.reward, make_node)
        self.assertEqual(clause.key, reading_key("containment"))
        self.assertNotEqual(clause.key, ">=")
        self.assertEqual(_reward_measure(bound.reward), ("containment", ("hammer", "bin")))

    def test_the_gated_bound_reward_reads_the_atom(self):
        doc = _problem(
            "(define (problem p) (:objects hammer bin) (:goal (In hammer bin))"
            " (:reward (progress (In hammer bin) (:gate 1.0) (:gated-by (In hammer bin)))))"
        )
        bound = apply_bindings(doc, VAB_INDUSTRIAL_V1)
        clause = build_reward(bound.reward, lambda form: Leaf(fn=lambda ctx: (True, 1.0, {})))
        self.assertIsInstance(clause, Gated)
        self.assertEqual(clause.of.key, reading_key("containment"))

    def test_bound_inside_connectives_and_modals(self):
        doc = _problem(
            "(define (problem p) (:objects hammer bin)"
            " (:reward (progress (and (hold-for 2 (In hammer bin)) (lifted hammer)) (:gate 1.0))))"
        )
        bound = apply_bindings(doc, VAB_INDUSTRIAL_V1)
        self.assertEqual(
            bound.reward["progress"][0],
            ["and", ["hold-for", 2, [">=", ["containment", "hammer", "bin"], 1.0]], ["lifted", "hammer"]],
        )


class TestSuiteName(unittest.TestCase):
    def test_kept_as_written(self):
        suite = _suite("(define (suite MySuite))")
        self.assertEqual(suite.name, "MySuite")
        doc = merge_suite(_problem("(define (problem v1) (:suite MySuite))"), suite)
        self.assertEqual(doc.suite, "MySuite")


class TestParameterisedReference(unittest.TestCase):
    """``cable.point(0.63)`` is one reference token, in a document as in code."""

    def test_reader_keeps_it_whole(self):
        (form,) = parse_sexp("(>= (winding-turns cable.point(0.63) peg0.axis) 0.5)")
        self.assertEqual(form, [">=", ["winding-turns", "cable.point(0.63)", "peg0.axis"], 0.5])

    def test_it_parses_to_a_reference_with_its_argument(self):
        (form,) = parse_sexp("(>= (winding-turns cable.point(0.63) peg0.axis) 0.5)")
        condition = parse_condition(form)
        self.assertEqual(condition.args[0].base, "cable")
        self.assertEqual(condition.args[0].arg, 0.63)
        self.assertIsNone(condition.args[1].arg)

    def test_forms_are_untouched(self):
        """A paren after whitespace, or after a number, is still a form."""
        (form,) = parse_sexp("(a x.y (0.5) 0.5(1) b(2) (c d))")
        self.assertEqual(form, ["a", "x.y", [0.5], 0.5, [1], "b", [2], ["c", "d"]])


if __name__ == "__main__":
    unittest.main()
