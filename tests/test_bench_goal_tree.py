# SPDX-License-Identifier: Apache-2.0
"""Lowering a parsed goal to a tree: a comparison around a verdict is refused."""

import unittest

from robosimstudio_bench.goals.goal import build_goal
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


if __name__ == "__main__":
    unittest.main()
