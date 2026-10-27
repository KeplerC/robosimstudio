# SPDX-License-Identifier: Apache-2.0
"""The document checker: each kind of problem typed and located, a catalog gap told
apart from a mistake in the file, one set of gate words shared with the reward,
and a selector refused on an atom that reports nothing."""

import unittest
from types import SimpleNamespace
from unittest import mock

from robosimstudio_bench.goals import reward
from robosimstudio_bench.goals.combinators import Leaf
from robosimstudio_bench.taskdoc import typecheck
from robosimstudio_bench.taskdoc.parse import parse_condition
from robosimstudio_bench.taskdoc.reader import parse_sexp
from tests.fixtures import taskdocs

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


CATALOG = taskdocs.FakeCatalog(
    {
        "box_a": taskdocs.Asset(),
        "tape_a": taskdocs.Asset(states=("adhered", "bond-fraction")),
        "ring_a": taskdocs.Asset(features={"hook": "hook", "eye": "loop"}),
        "rod_a": taskdocs.Asset(),
    }
)


def _problems(
    goal: str,
    *,
    objects: str = "cube - box_a tape - tape_a rod - rod_a ring - ring_a",
    regions: str = "(square (:ranges ((0 0 1 1))))",
    extra: str = "",
):
    """``check`` over a document with ``goal``, the four objects above and a square, against :data:`CATALOG`."""
    text = (
        f"(define (problem p) (:objects {objects}) (:regions {regions}) {extra}"
        + (f" (:goal {goal})" if goal else "")
        + ")"
    )
    return [(p.kind, p.where) for p in typecheck.check(taskdocs.doc(text), CATALOG)]


class TestTheCheckersVerdicts(unittest.TestCase):
    """``check``: each problem typed, located, and every one of them reported rather than the first."""

    def test_a_document_that_names_what_exists_typechecks(self):
        self.assertEqual(_problems("(>= (containment cube square) 1.0)"), [])

    def test_an_instance_the_task_never_declares_is_malformed(self):
        self.assertEqual(_problems("(>= (containment ghost square) 1.0)"), [(typecheck.MALFORMED, "goal")])

    def test_an_asset_the_catalog_lacks_is_the_worlds_gap_not_the_files(self):
        (problem,) = typecheck.check(
            taskdocs.doc("(define (problem p) (:objects hook - hook_a) (:goal (<= (speed hook) 0.1)))"), CATALOG
        )
        self.assertEqual(problem.kind, typecheck.CATALOG_GAP)
        self.assertTrue(problem.is_gap)
        self.assertIn("'hook_a'", problem.detail)

    def test_a_feature_the_asset_does_not_have_is_a_gap_listing_what_it_has(self):
        (problem,) = typecheck.check(
            taskdocs.doc(
                "(define (problem p) (:objects rod - rod_a ring - ring_a) (:goal (>= (threaded rod ring.lip) 1)))"
            ),
            CATALOG,
        )
        self.assertEqual(problem.kind, typecheck.CATALOG_GAP)
        self.assertIn("['eye', 'hook']", problem.detail)

    def test_a_feature_of_the_wrong_kind_is_refused_and_the_right_kind_is_not(self):
        self.assertEqual(_problems("(>= (threaded rod ring.hook) 0.5)"), [(typecheck.MALFORMED, "goal")])
        self.assertEqual(_problems("(>= (threaded rod ring.eye) 0.5)"), [])

    def test_a_word_nothing_catalogues_is_unresolved(self):
        self.assertEqual(_problems("(>= (sparkle cube) 1.0)"), [(typecheck.UNRESOLVED, "goal")])

    def test_an_assets_own_states_are_words_a_goal_may_use(self):
        self.assertEqual(_problems("(and (adhered tape) (<= (bond-fraction tape) 0.05))"), [])
        # The same word about an object whose asset does not define it is not one.
        self.assertEqual(_problems("(adhered cube)"), [(typecheck.UNRESOLVED, "goal")])

    def test_too_few_operands_is_an_arity_problem(self):
        self.assertEqual(_problems("(>= (containment cube) 1.0)"), [(typecheck.ARITY, "goal")])

    def test_an_observe_only_atom_is_refused_as_a_goal_and_accepted_as_an_observation(self):
        self.assertEqual(_problems("(watch cube)"), [(typecheck.DIRECTION, "goal")])
        self.assertEqual(_problems("(<= (speed cube) 0.1)", extra="(:observe (watch cube))"), [])

    def test_an_init_clause_is_checked_as_a_verdict(self):
        self.assertEqual(
            _problems("(<= (speed cube) 0.1)", extra="(:init (watch cube))"), [(typecheck.DIRECTION, "init[0]")]
        )

    def test_a_hold_for_without_seconds_is_refused(self):
        self.assertEqual(_problems("(hold-for soon (<= (speed cube) 0.1))"), [(typecheck.MALFORMED, "goal")])
        self.assertEqual(_problems("(hold-for 2 (<= (speed cube) 0.1))"), [])

    def test_a_document_without_a_goal_is_refused(self):
        self.assertEqual(_problems(""), [(typecheck.MALFORMED, "goal")])

    def test_a_region_hung_off_nothing_is_refused(self):
        problems = _problems("(<= (speed cube) 0.1)", regions="(lip (:target bin) (:ranges ((0 0 1 1))))")
        self.assertEqual(problems, [(typecheck.MALFORMED, "regions")])

    def test_every_problem_is_reported_not_just_the_first(self):
        problems = _problems("(and (>= (sparkle cube) 1.0) (>= (containment ghost square) 1.0) (watch cube))")
        self.assertEqual(
            problems,
            [
                (typecheck.UNRESOLVED, "goal.and[0]"),
                (typecheck.MALFORMED, "goal.and[1]"),
                (typecheck.DIRECTION, "goal.and[2]"),
            ],
        )

    def test_both_catalogs_satisfy_the_checkers_protocol(self):
        from robosimstudio_bench.goals.catalog import AssetCatalog

        self.assertIsInstance(CATALOG, typecheck.Catalog)
        self.assertIsInstance(AssetCatalog(), typecheck.Catalog)


if __name__ == "__main__":
    unittest.main()
