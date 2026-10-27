# SPDX-License-Identifier: Apache-2.0
"""The document checker: one set of gate words shared with the reward, and a selector
refused on an atom that reports nothing."""

import unittest
from types import SimpleNamespace
from unittest import mock

from robosimstudio_bench.goals import reward
from robosimstudio_bench.goals.combinators import Leaf
from robosimstudio_bench.taskdoc import typecheck
from robosimstudio_bench.taskdoc.parse import parse_condition
from robosimstudio_bench.taskdoc.reader import parse_sexp

TIER = "fast"
"""Parsed documents and a stub catalog: no scene, no GPU."""


class TestOneSetOfGateWords(unittest.TestCase):
    def test_reward_reads_the_checkers_spellings(self):
        """Every gate and mask spelling the checker reads is one the reward builds, and refuses at zero."""

        def node(form):
            return Leaf(fn=lambda ctx: (True, 1.0, {}))

        for gate in typecheck._GATE_WORDS:
            for mask in typecheck._MASK_WORDS:
                with self.subTest(gate=gate, mask=mask):
                    section = {"progress": [["depth", "plug"], [f":{gate}", 0.5], [f":{mask}", ["intact", "film"]]]}
                    clause = reward.build_reward(section, node)
                    self.assertIsInstance(clause, reward.Gated)
                    self.assertEqual(clause.of.gate, 0.5)
                    zero = {"progress": [["depth", "plug"], [f":{gate}", 0]]}
                    with self.assertRaisesRegex(ValueError, "must be > 0"):
                        reward.build_reward(zero, node)
                    problems = typecheck._check_reward(SimpleNamespace(reward={"progress": [[f":{gate}", 0]]}), None)
                    self.assertTrue(any("must be a number > 0" in str(problem) for problem in problems), problems)

    def test_the_checker_takes_a_gate_prefixed_atom_as_the_measure(self):
        # `gate_depth` is not a gate word; the checker used to skip it by prefix.
        seen = []
        with mock.patch.object(typecheck, "_visit_reward_form", side_effect=lambda form, *a: seen.append(form) or []):
            typecheck._check_reward(SimpleNamespace(reward={"progress": [["gate_depth", "plug"]]}), catalog=None)
        self.assertEqual(seen, [["gate_depth", "plug"]])


class TestASelectorOnAnAtomThatReportsNothing(unittest.TestCase):
    def _problems(self, atom):
        condition = parse_condition(parse_sexp("(widget:lat)")[0])
        catalog = SimpleNamespace(atom=lambda name: atom)
        return typecheck._check_condition(condition, None, catalog, "goal", direction="check")

    def _atom(self, **extra):
        return SimpleNamespace(supports=lambda d: True, directions=("check",), args=(), **extra)

    def test_is_refused_as_bind_refuses_it(self):
        problems = self._problems(self._atom(returns=()))
        self.assertTrue(any(p.kind == typecheck.UNRESOLVED and "'lat'" in p.detail for p in problems), problems)

    def test_a_catalog_object_without_returns_is_not_checked(self):
        self.assertEqual(self._problems(self._atom()), [])


if __name__ == "__main__":
    unittest.main()
