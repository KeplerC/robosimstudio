# SPDX-License-Identifier: Apache-2.0
"""The gauntlet: its nine stages run in order and stop at the first failure, a
document read, bound and started the way the loader and the env do, the goal
linted for what it cannot measure, the do-nothing floor, and the reward measure
found past the gate words."""

import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from robosimstudio_bench.goals.combinators import AllOf, AnyOf, Leaf
from robosimstudio_bench.taskdoc.ast import Condition, Instance, Ref, TaskDoc
from tests.fixtures import taskdocs

TIER = "fast"
"""Documents in a scratch directory and stub envs: no scene, no GPU."""


# The package re-exports `run` as `gauntlet`, which shadows the module.
gauntlet = importlib.import_module("robosimstudio_bench.goals.gauntlet")


class TestGauntletStartReadsTheActiveSet(unittest.TestCase):
    def test_a_parked_objects_init_clause_is_not_checked(self):
        doc = TaskDoc(
            name="t",
            objects=(Instance("a", "block"), Instance("b", "block")),
            init=(Condition("on-table", (Ref("a"),)), Condition("on-table", (Ref("b"),))),
            goal=Condition("lifted", (Ref("a"),)),
        )

        def fake_bind(c, doc, asset_of, task=""):
            ok = c.head == "on-table" and c.args[0].base == "a"
            return Leaf(fn=lambda ctx: (ok, float(ok), {}))

        env = SimpleNamespace(
            reset=lambda: None,
            make_context=object,
            episode=SimpleNamespace(active=frozenset({"a"})),
        )
        with (
            mock.patch.object(gauntlet, "bind", fake_bind),
            mock.patch.object(gauntlet, "Snapshot", lambda ctx: SimpleNamespace(ctx=ctx)),
        ):
            stage = gauntlet._start(doc, env, "t")
        self.assertEqual(stage.status, "pass", stage.findings)


class TestGauntletReadsTheSuiteAndProfile(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        (root / "demo.suite").write_text(
            "(define (suite demo)\n  (:staging (:profile fake@v1))\n  (:fixtures bin - box))\n"
        )
        self.task = root / "one.task"
        self.task.write_text(
            "(define (problem one)\n  (:suite demo)\n  (:objects tool - hammer)\n  (:goal (In tool bin)))\n"
        )

    def tearDown(self):
        self.dir.cleanup()

    def test_read_merges_the_sibling_suite(self):
        stage, doc = gauntlet._read(self.task)
        self.assertEqual(stage.status, "pass", stage.findings)
        self.assertIn("bin", {f.name for f in doc.objects})
        self.assertEqual(str(dict(doc.staging).get("profile")), "fake@v1")

    def test_bind_falls_back_to_the_documents_profile(self):
        _, doc = gauntlet._read(self.task)
        with mock.patch(
            "robosimstudio_bench.taskbuild.staging.get_profile",
            side_effect=lambda name: SimpleNamespace(bindings=None) if name == "fake@v1" else None,
        ) as get:
            stage, _ = gauntlet._bind(doc, "")
        get.assert_called_once_with("fake@v1")
        self.assertEqual(stage.data.get("profile"), "fake@v1")


class TestVacuousThreshold(unittest.TestCase):
    @staticmethod
    def _flags(op, value):
        return [f.kind for f in gauntlet._vacuous([Condition("distance", (Ref("a"),), op=op, value=value)])]

    def test_strictly_positive_is_not_vacuous(self):
        self.assertEqual(self._flags(">", 0.0), [])

    def test_the_vacuous_ones_are_still_flagged(self):
        self.assertEqual(self._flags(">=", 0.0), ["vacuous-threshold"])
        self.assertEqual(self._flags(">", -0.1), ["vacuous-threshold"])


class TestOneSetOfGateWords(unittest.TestCase):
    def test_the_gauntlet_finds_the_measure_past_the_gate(self):
        section = {"progress": [[":gate-at", 0.1], [":gated_by", ["intact", "film"]], ["depth", "plug"]]}
        self.assertEqual(gauntlet._reward_measure(section), ("depth", ("plug",)))

    def test_the_gauntlet_leaves_are_the_goals(self):
        a, b = Leaf(fn=lambda ctx: (True, 1.0, {})), Leaf(fn=lambda ctx: (True, 1.0, {}))
        self.assertEqual(gauntlet._leaves(AllOf(of=(a, b))), (a, b))
        self.assertEqual(gauntlet._leaves(None), ())


def _text(goal: str) -> str:
    """:data:`~tests.fixtures.taskdocs.DOCUMENT` with another goal."""
    return taskdocs.DOCUMENT.replace("(>= (containment cube square) 1.0)", goal)


class TestTheGauntletRunsItsStagesInOrder(unittest.TestCase):
    """``run``: every stage reported, in order, and everything after the first failure skipped with a reason."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.enterContext(taskdocs.cube_asset())

    def _run(self, text: str, **kwargs) -> gauntlet.Report:
        return gauntlet.run(taskdocs.write(self.tmp.name, text), **kwargs)

    def _statuses(self, report: gauntlet.Report) -> list[tuple[str, str]]:
        return [(stage.name, stage.status) for stage in report.stages]

    def test_a_clean_document_passes_every_static_stage_and_says_why_the_scene_ones_did_not_run(self):
        report = self._run(taskdocs.DOCUMENT)
        self.assertEqual([name for name, _ in self._statuses(report)], list(gauntlet.STAGES))
        for stage in report.stages:
            expected = "skip" if stage.name in gauntlet.NEEDS_SCENE else "pass"
            self.assertEqual(stage.status, expected, stage.name)
        self.assertIn("no scene was offered", report.stage("floor").reason)
        self.assertTrue(report.ok)
        self.assertEqual(report.failed, "")

    def test_a_file_that_does_not_parse_fails_read_and_skips_the_other_eight(self):
        report = gauntlet.run(Path(self.tmp.name) / "missing.task")
        self.assertEqual(self._statuses(report)[0], ("read", "fail"))
        self.assertEqual({status for _, status in self._statuses(report)[1:]}, {"skip"})
        self.assertEqual({stage.reason for stage in report.stages[1:]}, {"the document did not parse"})
        self.assertEqual(report.failed, "read")
        self.assertFalse(report.ok)

    def test_a_profile_that_does_not_exist_fails_bind(self):
        report = self._run(taskdocs.DOCUMENT, profile="no_such_profile@v9")
        self.assertEqual(report.failed, "bind")
        self.assertEqual([f.kind for f in report.stage("bind").findings], ["unresolved"])
        self.assertEqual(report.stage("typecheck").reason, "the predicate binding did not resolve")

    def test_an_asset_the_catalog_lacks_is_a_catalog_gap(self):
        report = self._run(taskdocs.DOCUMENT, catalog=taskdocs.FakeCatalog())
        self.assertEqual(report.failed, "typecheck")
        self.assertEqual(report.stage("typecheck").data, {"problems": 1, "gaps": 1})
        self.assertEqual(report.stage("evaluate").status, "skip")

    def test_a_bare_peak_typechecks_and_still_has_no_evaluator(self):
        report = self._run(_text("(peak (containment cube square))"))
        self.assertEqual(report.stage("typecheck").status, "pass")
        self.assertEqual(report.failed, "evaluate")
        (finding,) = report.stage("evaluate").findings
        self.assertEqual(finding.kind, "malformed")
        self.assertIn("must be compared", finding.detail)

    def test_a_cloth_on_a_rigid_arrangement_fails_the_sim_stage(self):
        cloth = SimpleNamespace(domain="cloth", backend=None, problems=lambda: [])
        catalog = taskdocs.FakeCatalog({taskdocs.CUBE: taskdocs.Asset(sim=cloth)})
        report = self._run(taskdocs.DOCUMENT, catalog=catalog)
        self.assertEqual(report.failed, "sim")
        (finding,) = report.stage("sim").findings
        self.assertIn("cube is a cloth", finding.detail)
        self.assertIn("'particles'", finding.detail)

    def test_a_clause_written_twice_fails_the_lint(self):
        clause = "(>= (containment cube square) 1.0)"
        report = self._run(_text(f"(and {clause} {clause})"))
        self.assertEqual(report.failed, "lint")
        self.assertIn("duplicate-clause", [f.kind for f in report.stage("lint").errors])
        self.assertEqual(report.stage("compile").reason, "the goal did not pass its lint")

    def test_a_scene_that_does_not_build_fails_compile_and_skips_the_rest(self):
        def build():
            raise RuntimeError("no device")

        report = self._run(taskdocs.DOCUMENT, scene=build)
        self.assertEqual(report.failed, "compile")
        self.assertIn("RuntimeError: no device", report.stage("compile").findings[0].detail)
        self.assertEqual([report.stage(n).status for n in ("start", "floor")], ["skip", "skip"])

    def test_the_text_names_each_stage_its_status_and_its_findings(self):
        clause = "(>= (containment cube square) 1.0)"
        text = self._run(_text(f"(and {clause} {clause})"), task="probe/dup").text()
        lines = text.splitlines()
        self.assertEqual(lines[0], "probe/dup")
        self.assertTrue(any(line.startswith("  FAIL lint") for line in lines), text)
        self.assertTrue(any(line.startswith("  skip floor") and "lint" in line for line in lines), text)
        self.assertTrue(any("[duplicate-clause]" in line for line in lines), text)


class TestAReportIsOkUnlessAStageFailed(unittest.TestCase):
    def test_skips_and_notes_are_not_failures(self):
        note = gauntlet.Finding("flat-conjunction", "wide", severity="note")
        report = gauntlet.Report(
            "t", [gauntlet.StageResult("read", "pass", (note,)), gauntlet.StageResult("bind", "skip")]
        )
        self.assertTrue(report.ok)
        self.assertEqual(report.stage("read").notes, (note,))
        self.assertEqual(report.stage("read").errors, ())

    def test_the_first_failure_is_the_one_named(self):
        report = gauntlet.Report(
            "t",
            [
                gauntlet.StageResult("read", "pass"),
                gauntlet.StageResult("bind", "fail"),
                gauntlet.StageResult("lint", "fail"),
            ],
        )
        self.assertFalse(report.ok)
        self.assertEqual(report.failed, "bind")

    def test_a_note_says_so_and_an_error_does_not(self):
        self.assertEqual(str(gauntlet.Finding("arity", "two", "goal")), "[arity] goal: two")
        self.assertEqual(str(gauntlet.Finding("wide", "eight", severity="note")), "[note wide] eight")


def _scored(goal_text: str = "(>= (containment cube square) 1.0)"):
    """A parsed document around ``goal_text`` and a stand-in for its built evaluator with one leaf."""
    parsed = taskdocs.doc(_text(goal_text))
    return parsed, SimpleNamespace(goal=Leaf(fn=lambda ctx: (True, 1.0, {})))


class TestTheLint(unittest.TestCase):
    """``lint_goal``: two errors, four notes, and each one only where it belongs."""

    def _kinds(self, text: str, *, catalog=None, scored=None) -> dict[str, str]:
        catalog = catalog or taskdocs.FakeCatalog({taskdocs.CUBE: taskdocs.Asset()})
        findings = gauntlet.lint_goal(taskdocs.doc(text), scored or _scored()[1], catalog)
        return {f.kind: f.severity for f in findings}

    def test_a_clean_goal_has_nothing_to_say(self):
        self.assertEqual(self._kinds(taskdocs.DOCUMENT), {})

    def test_a_document_without_a_goal_is_that_one_finding(self):
        parsed = taskdocs.doc(taskdocs.DOCUMENT.replace("(:goal (>= (containment cube square) 1.0))", ""))
        self.assertEqual([f.kind for f in gauntlet.lint_goal(parsed, None, taskdocs.FakeCatalog())], ["no-goal"])

    def test_a_goal_whose_tree_has_no_leaf_measures_nothing(self):
        self.assertEqual(self._kinds(taskdocs.DOCUMENT, scored=SimpleNamespace(goal=None)), {"no-measure": "error"})

    def test_the_same_clause_spelt_with_a_dash_and_an_underscore_is_a_duplicate(self):
        goal = "(and (<= (contact-force cube square) 2.0) (<= (contact_force cube square) 2.0))"
        self.assertEqual(self._kinds(_text(goal)), {"duplicate-clause": "error"})

    def test_one_measure_at_two_thresholds_is_not_a_duplicate(self):
        goal = "(and (>= (containment cube square) 0.5) (>= (containment cube square) 1.0))"
        self.assertEqual(self._kinds(_text(goal)), {})

    def test_a_ceiling_nothing_reaches_is_vacuous(self):
        self.assertEqual(self._kinds(_text("(<= (speed cube) 1e6)")), {"vacuous-threshold": "error"})
        self.assertEqual(self._kinds(_text("(<= (speed cube) 1e5)")), {})

    def test_eight_clauses_under_one_and_is_a_note_with_each_ones_share(self):
        goal = "(and " + " ".join(f"(>= (containment cube square) 0.{i + 1})" for i in range(8)) + ")"
        parsed = taskdocs.doc(_text(goal))
        (finding,) = gauntlet.lint_goal(parsed, _scored()[1], taskdocs.FakeCatalog({taskdocs.CUBE: taskdocs.Asset()}))
        self.assertEqual((finding.kind, finding.severity), ("flat-conjunction", "note"))
        self.assertIn("0.125", finding.detail)
        seven = taskdocs.doc(_text(goal.replace(" (>= (containment cube square) 0.8)", "")))
        self.assertEqual(
            gauntlet.lint_goal(seven, _scored()[1], taskdocs.FakeCatalog({taskdocs.CUBE: taskdocs.Asset()})), []
        )

    def test_a_reward_on_a_measure_the_goal_never_reads_is_a_note(self):
        outside = taskdocs.DOCUMENT.replace("(:goal", "(:reward (progress (speed cube) (:gate 0.1)))\n  (:goal")
        self.assertEqual(self._kinds(outside), {"reward-outside-goal": "note"})
        inside = taskdocs.DOCUMENT.replace(
            "(:goal", "(:reward (progress (containment cube square) (:gate 1.0)))\n  (:goal"
        )
        self.assertEqual(self._kinds(inside), {})

    def test_an_object_no_clause_names_is_a_note(self):
        extra = taskdocs.DOCUMENT.replace(
            f"(:objects cube - {taskdocs.CUBE})", f"(:objects cube spare - {taskdocs.CUBE})"
        )
        self.assertEqual(self._kinds(extra), {"declared-but-unmeasured": "note"})

    def test_a_type_the_catalog_never_heard_of_is_a_note(self):
        self.assertEqual(self._kinds(taskdocs.DOCUMENT, catalog=taskdocs.FakeCatalog()), {"undeclared-type": "note"})

    def test_the_shape_counts_leaves_depth_and_the_widest_conjunction(self):
        goal = "(and (>= (containment cube square) 1.0) (or (<= (speed cube) 0.1) (<= (speed cube) 0.2)))"
        parsed = taskdocs.doc(_text(goal))
        a, b, c = (Leaf(fn=lambda ctx: (True, 1.0, {})) for _ in range(3))
        scored = SimpleNamespace(goal=AllOf(of=(a, AnyOf(of=(b, c)))))
        self.assertEqual(gauntlet.goal_shape(parsed, scored), {"leaves": 3, "depth": 3, "widest": 2})


class TestTheDoNothingFloor(unittest.TestCase):
    """``do_nothing_floor``: what the idle action is paid, read as the most it was ever paid."""

    def test_the_floor_is_the_largest_reward_not_the_last(self):
        env = taskdocs.FakeEnv([(0.0, False, False, {}), (0.4, False, False, {}), (0.1, False, False, {})])
        stage = gauntlet.do_nothing_floor(env, steps=3)
        self.assertEqual(stage.data, {"steps": 3, "floor": 0.4, "final": 0.1})
        self.assertEqual(env.actions, ["idle"] * 3)
        self.assertEqual(stage.status, "pass")
        self.assertEqual([(f.kind, f.severity) for f in stage.findings], [("do-nothing-reward", "note")])

    def test_a_goal_the_idle_action_meets_fails_and_names_the_step(self):
        env = taskdocs.FakeEnv([(0.0, False, False, {}), (1.0, False, False, {"success": True})])
        stage = gauntlet.do_nothing_floor(env, steps=4)
        self.assertEqual(stage.status, "fail")
        (finding,) = stage.findings
        self.assertEqual(finding.kind, "solved-by-nothing")
        self.assertIn("at step 2", finding.detail)

    def test_a_task_that_pays_nothing_for_nothing_has_nothing_to_say(self):
        stage = gauntlet.do_nothing_floor(taskdocs.FakeEnv([(0.0, False, False, {})]), steps=5)
        self.assertEqual((stage.status, stage.findings, stage.data["floor"]), ("pass", (), 0.0))

    def test_it_stops_where_the_episode_ends(self):
        env = taskdocs.FakeEnv([(0.0, False, False, {}), (0.0, True, False, {}), (0.9, False, False, {})])
        self.assertEqual(gauntlet.do_nothing_floor(env, steps=10).data["steps"], 2)

    def test_it_resets_only_when_asked(self):
        env = taskdocs.FakeEnv([(0.0, False, False, {})])
        gauntlet.do_nothing_floor(env, steps=1)
        gauntlet.do_nothing_floor(env, steps=1, reset=False)
        self.assertEqual(env.resets, 1)

    def test_an_env_that_raises_fails_the_stage_rather_than_the_caller(self):
        stage = gauntlet.do_nothing_floor(taskdocs.FakeEnv([(0.0, False, False, {})], raise_at=0), steps=3)
        self.assertEqual(stage.status, "fail")
        self.assertIn("RuntimeError", stage.findings[0].detail)


if __name__ == "__main__":
    unittest.main()
