# SPDX-License-Identifier: Apache-2.0
"""The batched graph policy's per-wave state: a stale world thread writes nothing, and
an unknown graph name raises as the serial path does."""

from __future__ import annotations

import threading
import types
import unittest
from unittest import mock

from robosimstudio_policy_agent.fleet import solution as fleet_solution

TIER = "fast"
"""A fake ``gap`` module and bare policies: no GPU, no scene."""


def _policy(state):
    policy = fleet_solution.SolutionGraphBatched.__new__(fleet_solution.SolutionGraphBatched)
    policy.worlds = [state]
    policy.parks = types.SimpleNamespace(adopt=lambda world: None)
    policy.perception = ""
    policy.graph_path = "g"
    policy.skills = []
    return policy


def _run(policy, *, current: bool, outcome):
    state = policy.worlds[0]

    def execute(graph, **kwargs):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    fake_gap = types.ModuleType("gap")
    fake_gap.execute = execute
    with mock.patch.dict("sys.modules", {"gap": fake_gap}):
        thread = threading.Thread(target=policy._run_graph, args=(0, ""))
        # The next wave has started its own thread, or this one is still the world's.
        state.thread = thread if current else threading.Thread(target=lambda: None)
        thread.start()
        thread.join(5.0)


class TestAStaleWorldThreadWritesNothing(unittest.TestCase):
    def test_a_stale_error_is_dropped(self):
        state = fleet_solution._World()
        _run(_policy(state), current=False, outcome=RuntimeError("late"))
        self.assertIsNone(state.error)
        self.assertEqual(state.trace, {})

    def test_a_stale_result_is_dropped(self):
        state = fleet_solution._World()
        _run(_policy(state), current=False, outcome={"ok": True})
        self.assertIsNone(state.result)

    def test_the_current_thread_still_records(self):
        state = fleet_solution._World()
        _run(_policy(state), current=True, outcome=RuntimeError("now"))
        self.assertIsInstance(state.error, RuntimeError)
        self.assertIn("traceback", state.trace)
        state = fleet_solution._World()
        _run(_policy(state), current=True, outcome={"ok": True})
        self.assertEqual(state.result, {"ok": True})


class TestAnUnknownGraphNameRaises(unittest.TestCase):
    def test_typo_is_refused(self):
        def graph_path(name):
            raise KeyError(name)

        task = types.SimpleNamespace(graph_path=graph_path, graphs=[("gt", "graphs/weave")], full_name="suite/task")
        fleet = types.SimpleNamespace(task=task)
        with self.assertRaises(ValueError) as caught:
            fleet_solution.SolutionGraphBatched._resolve_graph(fleet, "/no/such/graph-weave", registries=[])
        self.assertIn("'gt'", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
