# SPDX-License-Identifier: Apache-2.0
"""The goal panel's text: a goal as a tree of clauses, each its verdict, its measure and its reading.

A clause line is a tick or a cross, the measure, and the one number that moves;
``hold-for`` says the seconds it has banked, ``worst_of`` marks its weakest
child, operands come back only where two lines would otherwise read alike, and
a tree deeper than :data:`~robosimstudio_studio.goal_panel.MAX_DEPTH` is cut
off. A task with no tree falls back to its metrics, flat. The header behind the
fold says the instruction, the step against its limit, and the reward. All of
it is string building over an outline and a metrics dict, so the outline here is
hand-made :class:`~robosimstudio_bench.goals.goal.Clause` records and the
metrics are what the test writes.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from robosimstudio_bench.goals.goal import Clause
from robosimstudio_studio.goal_panel import CROSS, MAX_DEPTH, TICK, PanelGoal, _ambiguous, _refs
from tests.fixtures.studio_fakes import FakeEnv, FakeSession

TIER = "fast"
"""No GPU and no task: a fake env carries the metrics, and the outline is hand-made."""


def _env(metrics=None, **kw) -> FakeEnv:
    """An env as the goal panel reads it: a last result, a step count, a task and its instruction."""
    env = FakeEnv()
    env.doc = kw.get("doc")
    env.goal = kw.get("goal")
    env.last_result = SimpleNamespace(
        success=kw.get("success", False), reward=kw.get("reward", 0.0), metrics=dict(metrics or {})
    )
    env.max_steps = kw.get("max_steps", 0)
    env.step_count = kw.get("step", 0)
    env.episode = SimpleNamespace(instruction=kw.get("instruction", "fold the towel in half"))
    physics = SimpleNamespace(backends=(("*", SimpleNamespace(solver="mujoco")),), control_hz=50.0, substeps=4)
    env.request = SimpleNamespace(physics=physics)
    env.task = SimpleNamespace(
        max_steps=kw.get("task_max", 0),
        scene="bench_0728",
        embodiment="panda",
        gripper="",
        control_mode="joint_position",
        physics=physics,
    )
    env.handles.support_arm = ""
    env.evaluations = 0

    def evaluate() -> None:
        env.evaluations += 1

    env.evaluate = evaluate
    return env


def _panel(outline: Clause | None = None, metrics=None, **kw) -> PanelGoal:
    """A goal panel whose document's goal outlines to ``outline`` (none: a hand-written evaluator)."""
    env = _env(metrics, doc=SimpleNamespace(goal=object()) if outline is not None else None, **kw)
    with mock.patch("robosimstudio_bench.goals.goal.outline", return_value=outline):
        return PanelGoal(FakeSession(env))


def _leaf(label: str, head: str, *operands: str) -> Clause:
    return Clause(label=label, head=head, operands=operands)


def _and(label: str, *children: Clause, head: str = "and", modal_arg=None) -> Clause:
    return Clause(label=label, head=head, modal_arg=modal_arg, children=children)


TOWEL = _and("goal", _leaf("goal.0", "fold-extent", "towel", "towel.half_fold"), _leaf("goal.1", "fold-score", "towel"))


class TestWithoutATree(unittest.TestCase):
    def test_its_metrics_are_listed_flat(self) -> None:
        text = _panel(metrics={"coverage": 0.5, "held": True})._tree_text()
        self.assertEqual("- `coverage` 0.50\n- `held` True", text)

    def test_only_the_first_eight_are_listed(self) -> None:
        text = _panel(metrics={f"m{i}": float(i) for i in range(12)})._tree_text()
        self.assertEqual(8, len(text.splitlines()))

    def test_with_no_metrics_either_it_says_a_hand_written_evaluator_scores_it(self) -> None:
        self.assertIn("hand-written evaluator", _panel()._tree_text())


class TestTheTree(unittest.TestCase):
    def test_a_line_is_its_verdict_and_its_measure_indented_by_depth(self) -> None:
        text = _panel(TOWEL, {"goal": False, "goal.0": True, "goal.1": False})._tree_text()
        self.assertEqual(f"- {CROSS} and\n  - {TICK} fold-extent\n  - {CROSS} fold-score", text)

    def test_before_the_first_step_nothing_is_marked(self) -> None:
        self.assertEqual("- and\n  - fold-extent\n  - fold-score", _panel(TOWEL)._tree_text())

    def test_a_leaf_reads_its_number_and_a_predicate_does_not_say_its_verdict_twice(self) -> None:
        goal = _and("goal", _leaf("goal.0", "fold-extent", "towel"), _leaf("goal.1", "settled", "towel"))
        metrics = {"goal.0": False, "goal.0.fold_extent": 0.834, "goal.1": True}
        panel = _panel(goal, metrics, goal=object())
        found = [SimpleNamespace(label=c.label, about=(c.head, c.operands)) for c in goal.children]
        with mock.patch("robosimstudio_bench.goals.goal.leaves", return_value=found):
            lines = panel._tree_text().splitlines()
        self.assertEqual(f"  - {CROSS} fold-extent 0.83", lines[1])
        self.assertEqual(f"  - {TICK} settled", lines[2])

    def test_hold_for_says_the_seconds_it_has_banked(self) -> None:
        goal = _and("goal", _leaf("goal.0", "settled"), head="hold-for", modal_arg=5.0)
        self.assertIn("hold-for 1.2 s", _panel(goal, {"goal": False, "goal.held_s": 1.23})._tree_text())
        # under a `once`, which passes its child's readings through, the key is bare
        self.assertIn("hold-for 2.0 s", _panel(goal, {"held_s": 2.0})._tree_text())
        self.assertIn("hold-for 1.2 s", _panel(goal, {"goal.held_s": 1.23, "held_s": 2.0})._tree_text())

    def test_another_modal_says_its_argument(self) -> None:
        goal = _and("goal", _leaf("goal.0", "settled"), head="within", modal_arg=3.0)
        self.assertEqual("- within 3 s", _panel(goal)._tree_text().splitlines()[0])

    def test_worst_of_marks_its_weakest_member_and_no_other(self) -> None:
        creases = [_leaf(f"goal.{i}", "fold-score", "shirt", f"shirt.crease{i}") for i in range(3)]
        lines = _panel(_and("goal", *creases, head="worst_of"), {"goal.worst": "goal.1"})._tree_text().splitlines()
        self.assertEqual([False, True, False], [line.endswith(" ← worst") for line in lines[1:]])

    def test_a_tree_past_the_depth_bound_is_cut_off(self) -> None:
        node = _leaf("goal" + ".0" * (MAX_DEPTH + 3), "settled")
        for depth in range(MAX_DEPTH + 2, -1, -1):
            node = _and("goal" + ".0" * depth, node)
        lines = _panel(node)._tree_text().splitlines()
        self.assertEqual(MAX_DEPTH + 2, len(lines))
        self.assertEqual("  " * (MAX_DEPTH + 1) + "- *(deeper clauses elided)*", lines[-1])


class TestWhichLinesNameTheirOperands(unittest.TestCase):
    def test_a_measure_used_once_is_named_alone(self) -> None:
        self.assertEqual(frozenset(), _ambiguous(TOWEL))
        self.assertNotIn("towel", _panel(TOWEL)._tree_text())

    def test_a_measure_used_twice_names_what_each_is_over(self) -> None:
        tools = _and(
            "goal",
            _leaf("goal.0", "containment", "hammer_0", "bin2"),
            _leaf("goal.1", "containment", "drill_0", "bin2"),
        )
        self.assertEqual({"goal.0", "goal.1"}, _ambiguous(tools))
        text = _panel(tools)._tree_text()
        self.assertIn("containment hammer_0/bin2", text)
        self.assertIn("containment drill_0/bin2", text)

    def test_an_operand_another_reaches_through_is_said_once(self) -> None:
        self.assertEqual(["shirt.left_side"], _refs(("shirt", "shirt.left_side")))
        self.assertEqual(["hammer_0", "bin2_hammer_region"], _refs(("hammer_0", "bin2_hammer_region")))


class TestTheHeader(unittest.TestCase):
    def test_it_is_the_instruction_the_step_against_its_limit_and_the_reward(self) -> None:
        text = _panel(step=12, task_max=400, reward=0.25)._header_text()
        self.assertEqual("*fold the towel in half*\n\nstep 12/400 · reward 0.25", text)

    def test_the_envs_own_limit_wins_over_the_tasks(self) -> None:
        self.assertIn("step 12/90 ", _panel(step=12, max_steps=90, task_max=400)._header_text())

    def test_no_limit_counts_the_step_alone_and_success_is_said(self) -> None:
        self.assertTrue(_panel(step=7, success=True)._header_text().endswith(f"step 7 · {TICK} success"))


class TestThePage(unittest.TestCase):
    def test_the_clauses_are_open_and_the_rest_is_behind_one_fold(self) -> None:
        panel = _panel(TOWEL, {"goal.0": True})
        gui = panel.session.server.gui
        panel.build(gui)
        goal, detail = gui.folders
        self.assertEqual(("Goal", "Detail"), (goal.label, detail.label))
        self.assertIs(goal, detail.parent, "unrolled once, not twice")
        self.assertFalse(detail.expand_by_default)
        tree, *rest = gui.markdown
        self.assertIs(goal, tree.parent)
        self.assertTrue(all(block.parent is detail for block in rest))
        self.assertIn("physics `mujoco` at 50 Hz, 4 substeps", rest[-1].content)

    def test_drain_scores_the_step_once_and_refresh_never(self) -> None:
        panel = _panel(TOWEL)
        panel.build(panel.session.server.gui)
        panel.drain()
        panel.refresh()
        panel.refresh()
        self.assertEqual(1, panel.env.evaluations)

    def test_refresh_shows_the_verdicts_the_last_step_wrote(self) -> None:
        panel = _panel(TOWEL)
        panel.build(panel.session.server.gui)
        panel.env.last_result.metrics.update({"goal": True, "goal.0": True, "goal.1": True})
        panel.refresh()
        self.assertEqual(3, panel._tree.content.count(TICK))


if __name__ == "__main__":
    unittest.main()
