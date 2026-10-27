# SPDX-License-Identifier: Apache-2.0
"""A job's seed reaching its episodes, and which episodes a run still owes."""

from __future__ import annotations

import unittest

from robosimstudio.specs import EpisodeSpec, JobSpec, TaskSpec
from robosimstudio_bench.run.jobs import job_task
from robosimstudio_bench.run.summary import RunSummary
from robosimstudio_bench.taskbuild.episodes import make_episode, pick_instruction, with_base_seed

TIER = "fast"
"""Specs and rows: no scene, no GPU."""


def _task(seed: int = 5) -> TaskSpec:
    return TaskSpec(name="t", episodes=EpisodeSpec(count=3, seed=seed))


class TestJobSeed(unittest.TestCase):
    """``JobSpec.seed`` overrides the base seed, and the row's seed is the effective one."""

    def test_unset_keeps_the_task_itself(self):
        task = _task()
        self.assertIs(with_base_seed(task, -1), task)
        self.assertIs(job_task(JobSpec(name="j", task="x"), task), task)

    def test_override_reaches_the_episode_seed(self):
        task = job_task(JobSpec(name="j", task="x", seed=100), _task())
        self.assertEqual(task.episodes.seed, 100)
        self.assertEqual(make_episode(task, 2).seed, 102)

    def test_a_sampler_that_enumerates_reads_the_seed(self):
        """``pick_instruction`` cycles by the seed, so instance documents differ."""
        sample = pick_instruction("a", "b", "c")
        self.assertEqual(sample(_task(seed=0), 1, None)["instruction"], "b")
        self.assertEqual(sample(_task(seed=1), 1, None)["instruction"], "c")


class TestUnrun(unittest.TestCase):
    """A ``documents`` job runs each id once per document; ``unrun`` matches both."""

    def test_documents_are_matched_by_document_and_id(self):
        job = JobSpec(name="j", task="s/a", documents=("s/a", "s/b"), episodes=(0, 2))
        rows = [{"task": "s/a", "episode_id": 0}, {"task": "s/a", "episode_id": 1}, {"task": "s/b", "episode_id": 0}]
        self.assertEqual(RunSummary(job=job, root=None, rows=rows).unrun, (("s/b", 1),))

    def test_a_plain_job_still_reports_ids(self):
        job = JobSpec(name="j", task="s/a", episodes=(0, 3))
        self.assertEqual(RunSummary(job=job, root=None, rows=[{"episode_id": 1}]).unrun, (0, 2))


if __name__ == "__main__":
    unittest.main()
