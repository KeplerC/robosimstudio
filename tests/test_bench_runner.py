# SPDX-License-Identifier: Apache-2.0
"""The runners: one serial rollout and the row it writes, documents refused where only one
model is built, and a wave with no step budget refused rather than closing nothing."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from tests.fixtures import taskdocs

TIER = "fast"
"""A scripted env and refusals made before a scene is built: no GPU."""


class TestDocumentsNeedAWave(unittest.TestCase):
    def test_run_pooled_refuses_documents_on_the_serial_worker(self):
        from robosimstudio.specs import JobSpec
        from robosimstudio_bench.run import pool

        job = JobSpec(name="j", task="s/a", episodes=(0, 2), documents=("s/a", "s/b"))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            pool.run_pooled(job, root=tmp, workers=1, verbose=False)

    def test_runner_eval_refuses_documents(self):
        from robosimstudio.specs import JobSpec
        from robosimstudio_bench.run.runner import RunnerEval

        job = JobSpec(name="j", task="s/a", episodes=(0, 2), documents=("s/a", "s/b"))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            RunnerEval(job, root=tmp, verbose=False)


class TestAWaveWithANonPositiveBudget(unittest.TestCase):
    def test_is_refused_rather_than_closing_nothing(self):
        from robosimstudio.specs import JobSpec
        from robosimstudio_bench.run.fleet import RunnerFleet

        with tempfile.TemporaryDirectory() as tmp:
            runner = RunnerFleet(JobSpec(name="j", task="t", num_worlds=2, max_steps=-5), verbose=False, run_dir=tmp)
            fleet, policy = mock.MagicMock(), mock.MagicMock()
            episode = SimpleNamespace(episode_id=0, seed=0)
            with mock.patch.object(RunnerFleet, "_evaluator", return_value=None):
                with self.assertRaises(ValueError):
                    runner._run_wave(fleet, policy, 0, [(SimpleNamespace(max_steps=50), episode)])
            policy.act.assert_not_called()


class _Policy:
    """Steps with its call count as the action and reports one extra column."""

    control_mode = ""

    def __init__(self) -> None:
        self.resets = 0
        self.seen: list[float] = []

    def reset(self, obs, info) -> None:
        self.resets += 1

    def get_action(self, obs):
        return len(self.seen)

    def on_step(self, obs, reward, info) -> None:
        self.seen.append(reward)

    def row_extras(self) -> dict:
        return {"graph_nodes": 7}


class _Recorder:
    def __init__(self) -> None:
        self.steps = 0

    def begin(self, episode, env) -> None:
        pass

    def step(self, *args, **kwargs) -> None:
        self.steps += 1

    def end(self, env, info, extra=None) -> dict:
        return {"recorded": True, **(extra or {})}


class TestOneRollout(unittest.TestCase):
    """``rollout``: one episode to its end, and the row a serial run writes for it."""

    def test_an_exhausted_budget_is_truncated(self):
        from robosimstudio_bench.run.runner import rollout

        env, policy = taskdocs.FakeEnv([(0.1, False, False, {})]), _Policy()
        result = rollout(env, policy, max_steps=3)
        self.assertEqual(env.actions, [0, 1, 2])
        self.assertEqual((result.row["close_kind"], result.row["episode_length"]), ("truncated", 3))
        self.assertEqual(policy.seen, [0.1, 0.1, 0.1])

    def test_the_envs_own_close_is_kept_when_it_ends_the_episode(self):
        from robosimstudio_bench.run.runner import rollout

        steps = [(0.0, False, False, {}), (1.0, True, False, {"success": True, "close_kind": "success"})]
        row = rollout(taskdocs.FakeEnv(steps), _Policy(), max_steps=10).row
        self.assertEqual(
            (row["close_kind"], row["success"], row["reward"], row["episode_length"]), ("success", True, 1.0, 2)
        )

    def test_the_row_carries_why_the_policy_asked_out_and_its_extras(self):
        from robosimstudio_bench.run.runner import rollout

        info = {"close_kind": "policy_request", "termination_reason": "graph finished", "metrics": {"depth": 0.2}}
        row = rollout(taskdocs.FakeEnv([(0.0, True, False, info)]), _Policy(), episode_id=4).row
        self.assertEqual(row["termination_reason"], "graph finished")
        self.assertEqual(row["metrics"], {"depth": 0.2})
        self.assertEqual(row["graph_nodes"], 7)
        self.assertEqual((row["task"], row["episode_id"], row["seed"]), ("suite/fake", 4, 104))

    def test_the_budget_is_the_envs_unless_the_caller_overrides_it(self):
        from robosimstudio_bench.run.runner import rollout

        self.assertEqual(
            rollout(taskdocs.FakeEnv([(0.0, False, False, {})], max_steps=2), _Policy()).row["episode_length"], 2
        )
        env = taskdocs.FakeEnv([(0.0, False, False, {})], max_steps=2)
        self.assertEqual(rollout(env, _Policy(), max_steps=5).row["episode_length"], 5)

    def test_the_recorders_observer_is_removed_even_when_a_step_raises(self):
        from robosimstudio_bench.run.runner import rollout

        env = taskdocs.FakeEnv([(0.0, False, False, {})], raise_at=1)
        with self.assertRaisesRegex(RuntimeError, "blew up"):
            rollout(env, _Policy(), recorder=_Recorder(), max_steps=5)
        self.assertEqual(env.observers, [])

    def test_a_recorded_episode_returns_the_recorders_row_with_the_extras(self):
        from robosimstudio_bench.run.runner import rollout

        env = taskdocs.FakeEnv([(0.0, True, False, {})])
        row = rollout(env, _Policy(), recorder=_Recorder()).row
        self.assertEqual(row, {"recorded": True, "graph_nodes": 7})
        self.assertEqual(env.observers, [])


if __name__ == "__main__":
    unittest.main()
