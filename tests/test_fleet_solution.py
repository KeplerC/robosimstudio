# SPDX-License-Identifier: Apache-2.0
"""The batched graph policy's per-wave bookkeeping: who the barrier waits for, a stalled
world retired alone, a finished graph settled then closed, what each world's trace and row
say, a stale world thread writing nothing, and an unknown graph name raising as the serial
path does."""

from __future__ import annotations

import threading
import types
import unittest
from unittest import mock

import numpy as np

from robosimstudio_policy_agent.fleet import solution as fleet_solution
from robosimstudio_policy_agent.fleet.barrier import StallTimeout

TIER = "fast"
"""A fake ``gap`` module, bare policies and recording collaborators: no GPU, no scene."""


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


class _Recorder:
    """Has the methods ``names``, each recording its calls; anything else is not there."""

    def __init__(self, *names: str) -> None:
        self.calls: list[tuple] = []
        self.names = names

    def __getattr__(self, name):
        if name not in self.__dict__.get("names", ()):
            raise AttributeError(name)
        return lambda *args, **kwargs: self.calls.append((name, *args))


class _Batched(unittest.TestCase):
    """A bare batched policy over ``num_worlds`` worlds, its collaborators recorders."""

    def batched(self, num_worlds: int = 2, *, settle_steps: int = 3, eval_every: int = 1):
        policy = fleet_solution.SolutionGraphBatched.__new__(fleet_solution.SolutionGraphBatched)
        policy.num_worlds = num_worlds
        policy.worlds = [fleet_solution._World() for _ in range(num_worlds)]
        policy.slots = [types.SimpleNamespace(success=False, reward=0.0, info={}) for _ in range(num_worlds)]
        policy._live = np.ones(num_worlds, dtype=bool)
        policy._stalled = set()
        policy._requests = {}
        policy.settle_steps = settle_steps
        policy.fleet = types.SimpleNamespace(eval_every=eval_every)
        policy.stepper = _Recorder("retire", "settle")
        policy.stepper.step = 42
        policy.service = _Recorder("pump")
        policy.rendezvous = _Recorder("drop_world")
        policy.barrier = types.SimpleNamespace(watch=types.SimpleNamespace(timeout=120.0))
        return policy

    def running(self) -> threading.Thread:
        """A thread that stays alive until the test ends."""
        release = threading.Event()
        thread = threading.Thread(target=release.wait, args=(10.0,), daemon=True)
        thread.start()
        self.addCleanup(thread.join, 10.0)
        self.addCleanup(release.set)
        return thread

    @staticmethod
    def finished() -> threading.Thread:
        thread = threading.Thread(target=lambda: None)
        thread.start()
        thread.join()
        return thread


class TestWhoTheBarrierWaitsFor(_Batched):
    """``_expect_parked``: only a live world whose graph is still running owes a park."""

    def test_a_running_graph_owes_a_park_and_a_finished_one_does_not(self):
        policy = self.batched(3)
        policy.worlds[0].thread = self.running()
        policy.worlds[1].thread = self.finished()
        np.testing.assert_array_equal(policy._expect_parked(), [True, False, False])

    def test_a_world_out_of_its_episode_or_given_up_on_owes_nothing(self):
        policy = self.batched(2)
        policy.worlds[0].thread = self.running()
        policy.worlds[1].thread = self.running()
        policy._live[0] = False
        policy._stalled.add(1)
        np.testing.assert_array_equal(policy._expect_parked(), [False, False])


class TestAStalledWorldIsRetiredAlone(_Batched):
    def test_its_motions_and_asks_end_and_its_episode_is_asked_to_close(self):
        policy = self.batched(2)
        policy.worlds[0].thread = self.running()
        policy.worlds[1].thread = self.running()
        policy._retire_stalled([1])
        self.assertEqual(policy._stalled, {1})
        np.testing.assert_array_equal(policy._expect_parked(), [True, False])
        ((_, world, reason),) = (call for call in policy.stepper.calls if call[0] == "retire")
        self.assertEqual(world, 1)
        self.assertIn("held the fleet for 120 s without parking", reason)
        ((_, dropped, error),) = policy.rendezvous.calls
        self.assertEqual(dropped, 1)
        self.assertIsInstance(error, StallTimeout)
        self.assertEqual(list(policy.termination_requests()), [1])
        self.assertTrue(policy.termination_requests()[1].startswith("stalled: "))

    def test_its_row_says_so_and_its_neighbours_do_not(self):
        policy = self.batched(2)
        policy._stalled.add(1)
        self.assertIs(policy.row_extras(1)["fleet_stalled"], True)
        self.assertNotIn("fleet_stalled", policy.row_extras(0))


class TestAFinishedGraphIsSettledThenClosed(_Batched):
    """``on_step``: a finished graph's world steps out its settle, counted in control steps, then asks to close."""

    def test_nothing_is_read_off_a_cadence_tick(self):
        policy = self.batched(1)
        policy.on_step(None)
        self.assertEqual([call[0] for call in policy.service.calls], ["pump"])
        self.assertEqual(policy.stepper.calls, [])

    def test_the_countdown_steps_by_the_readback_interval_then_requests_the_close(self):
        policy = self.batched(1, settle_steps=5, eval_every=2)
        policy.worlds[0].thread = self.finished()
        snapshot = types.SimpleNamespace(step=0)
        left = []
        for _ in range(3):
            self.assertEqual(policy.termination_requests(), {})
            policy.on_step(snapshot)
            left.append(policy.worlds[0].settle_left)
        self.assertEqual(left, [3, 1, -1])
        self.assertEqual(policy.termination_requests(), {0: "graph finished"})
        self.assertEqual(policy.worlds[0].trace["steps"], 42)

    def test_a_graph_that_raised_closes_saying_why(self):
        policy = self.batched(1, settle_steps=0)
        policy.worlds[0].thread = self.finished()
        policy.worlds[0].error = RuntimeError("no plan")
        policy.on_step(types.SimpleNamespace(step=0))
        self.assertEqual(policy.termination_requests(), {0: "graph failed: no plan"})
        self.assertEqual(policy.worlds[0].trace["error"], "RuntimeError: no plan")

    def test_a_running_graph_is_not_counted_down(self):
        policy = self.batched(1)
        policy.worlds[0].thread = self.running()
        policy.on_step(types.SimpleNamespace(step=0))
        self.assertEqual(policy.worlds[0].settle_left, -1)


class TestWhatAWorldsTraceSays(_Batched):
    def test_a_node_cut_off_by_a_success_is_a_cancellation_not_an_error(self):
        policy = self.batched(1)
        policy.worlds[0].result = types.SimpleNamespace(error=TimeoutError("waiting on a grip"))
        policy.slots[0].success = True
        policy._record_result(0)
        trace = policy.worlds[0].trace
        self.assertEqual(trace["graph_result"]["cancelled_at_episode_end"], "TimeoutError: waiting on a grip")
        self.assertNotIn("graph_error", trace)
        self.assertNotIn("graph_error", policy.row_extras(0))

    def test_the_same_failure_without_the_success_is_the_graphs_error(self):
        policy = self.batched(1)
        policy.worlds[0].result = types.SimpleNamespace(error=TimeoutError("x" * 500))
        policy._record_result(0)
        self.assertTrue(policy.worlds[0].trace["graph_error"].startswith("TimeoutError: xxx"))
        self.assertEqual(len(policy.row_extras(0)["graph_error"]), 400)

    def test_a_worlds_status_follows_its_thread(self):
        policy = self.batched(4)
        policy.worlds[1].thread = self.running()
        policy.worlds[2].thread = self.finished()
        policy.worlds[3].thread = self.finished()
        policy.worlds[3].error = RuntimeError("x")
        self.assertEqual([policy.status(w) for w in range(4)], ["idle", "running", "finished", "failed"])

    def test_a_graph_reads_the_evaluators_verdict_off_its_slot(self):
        policy = self.batched(2)
        policy.slots[0].info = {"success": True, "metrics": {"depth": 1.0}}
        policy.slots[1] = types.SimpleNamespace(success=False, reward=0.25)
        infos = policy._infos(None)
        self.assertEqual(infos[0], {"success": True, "metrics": {"depth": 1.0}, "reward": 0.0})
        self.assertEqual(infos[1], {"success": False, "reward": 0.25})


if __name__ == "__main__":
    unittest.main()
