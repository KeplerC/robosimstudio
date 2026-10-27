# SPDX-License-Identifier: Apache-2.0
"""Lowering a parsed goal to a tree: each form to its combinator, labelled by path, and every
form the closed combinator set does not cover refused -- a comparison around a verdict among them."""

import unittest

from robosimstudio_bench.goals.combinators import After, AllOf, AnyOf, HoldFor, Integral, Leaf, Not, Once, Peak, WorstOf
from robosimstudio_bench.goals.goal import build_goal
from robosimstudio_bench.goals.reward import reading_key
from robosimstudio_bench.taskdoc.ast import Condition
from robosimstudio_bench.taskdoc.parse import parse_condition
from robosimstudio_bench.taskdoc.reader import parse_sexp

TIER = "fast"
"""Parsed conditions and stub leaves: no scene, no GPU."""


class TestComparisonAroundAVerdict(unittest.TestCase):
    def test_refused_for_once_hold_for_after(self):
        for text in (
            "(>= (once (fold-score towel)) 0.8)",
            "(>= (hold-for 2 (fold-score towel)) 0.8)",
            "(>= (after (grasped towel) (fold-score towel)) 0.8)",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_condition(parse_sexp(text)[0])

    def test_peak_still_takes_one(self):
        condition = parse_condition(parse_sexp("(<= (peak (tension film)) 12.0)")[0])
        self.assertEqual((condition.modal, condition.op, condition.value), ("peak", "<=", 12.0))

    def test_build_goal_refuses_one_that_got_through(self):
        inner = Condition(head="fold-score")
        condition = Condition(head="once", modal="once", children=(inner,), op=">=", value=0.8)
        with self.assertRaises(ValueError):
            build_goal(condition, lambda *_: None)


def _build(text: str):
    """``text`` parsed and lowered, every leaf a :class:`Leaf` remembering the clause it measures."""
    return build_goal(
        parse_condition(parse_sexp(text)[0]), lambda c: Leaf(fn=lambda ctx: (True, 1.0, {}), about=(c.head, ()))
    )


class TestEachFormLowersToItsCombinator(unittest.TestCase):
    """``build_goal``: one combinator per form, each node labelled with the path of the clause it came from."""

    def test_the_connectives(self):
        for word, node in (("and", AllOf), ("or", AnyOf), ("any", AnyOf), ("worst_of", WorstOf)):
            with self.subTest(word=word):
                built = _build(f"({word} (speed a) (speed b))")
                self.assertIsInstance(built, node)
                self.assertEqual([child.label for child in built.of], ["goal.0", "goal.1"])

    def test_not_negates_its_one_clause(self):
        built = _build("(not (speed a))")
        self.assertIsInstance(built, Not)
        self.assertEqual(built.of.label, "goal.0")

    def test_once_latches_its_clause(self):
        self.assertIsInstance(_build("(once (grasped a))"), Once)

    def test_hold_for_takes_its_seconds_as_a_number(self):
        built = _build("(hold-for 2 (speed a))")
        self.assertIsInstance(built, HoldFor)
        self.assertEqual(built.seconds, 2.0)
        self.assertIsInstance(built.seconds, float)

    def test_after_labels_its_event_apart_from_its_clause(self):
        built = _build("(after (grasped a) (>= (speed a) 1))")
        self.assertIsInstance(built, After)
        self.assertEqual((built.event.label, built.of.label), ("goal.event", "goal.0"))
        self.assertEqual(built.event.about, ("grasped", ()))

    def test_peak_and_integral_threshold_what_the_atom_reports(self):
        for word, node in (("peak", Peak), ("integral", Integral)):
            with self.subTest(word=word):
                built = _build(f"(<= ({word} (contact-force a b)) 12.0)")
                self.assertIsInstance(built, node)
                self.assertEqual((built.key, built.op, built.value), (reading_key("contact-force"), "<=", 12.0))
                self.assertNotEqual(built.key, "contact-force")

    def test_a_leaf_is_labelled_with_its_path_unless_it_already_has_a_label(self):
        condition = parse_condition(parse_sexp("(and (speed a) (speed b))")[0])
        named = Leaf(fn=lambda ctx: (True, 1.0, {}), label="mine")
        built = build_goal(
            condition, lambda c: named if c.args[0].base == "b" else Leaf(fn=lambda ctx: (True, 1.0, {}))
        )
        self.assertEqual([child.label for child in built.of], ["goal.0", "mine"])


class TestTheFormsTheCombinatorSetDoesNotCover(unittest.TestCase):
    """Refused rather than approximated, each with the reason a reader needs."""

    def test_a_bare_peak_or_integral_is_refused(self):
        for word in ("peak", "integral"):
            with self.subTest(word=word), self.assertRaisesRegex(ValueError, "must be compared"):
                _build(f"({word} (contact-force a b))")

    def test_not_over_two_clauses_is_refused(self):
        with self.assertRaisesRegex(ValueError, "negates exactly one clause, got 2"):
            _build("(not (speed a) (speed b))")

    def test_an_empty_connective_is_refused(self):
        for word in ("and", "or"):
            with self.subTest(word=word), self.assertRaisesRegex(ValueError, "has no clauses"):
                _build(f"({word})")

    def test_a_hold_for_without_a_number_is_refused(self):
        with self.assertRaisesRegex(ValueError, "needs a duration in seconds"):
            _build("(hold-for soon (speed a))")

    def test_an_after_without_an_event_clause_is_refused(self):
        condition = Condition(
            head="after", modal="after", modal_arg="later", children=(parse_condition(parse_sexp("(speed a)")[0]),)
        )
        with self.assertRaisesRegex(ValueError, "needs an event clause"):
            build_goal(condition, lambda c: Leaf(fn=lambda ctx: (True, 1.0, {})))

    def test_sub_clauses_under_a_measure_are_refused(self):
        condition = Condition(head="speed", children=(parse_condition(parse_sexp("(speed a)")[0]),))
        with self.assertRaisesRegex(ValueError, "not a combinator"):
            build_goal(condition, lambda c: Leaf(fn=lambda ctx: (True, 1.0, {})))


if __name__ == "__main__":
    unittest.main()
