# SPDX-License-Identifier: Apache-2.0
"""The runners' guards: documents refused where only one model is built, and a wave
with no step budget refused rather than closing nothing."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

TIER = "fast"
"""Refused before a scene is built: no GPU."""


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


if __name__ == "__main__":
    unittest.main()
