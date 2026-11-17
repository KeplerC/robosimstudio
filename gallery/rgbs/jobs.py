# SPDX-License-Identifier: Apache-2.0
"""The RGBS suite's gallery runs: every variant with a policy, through ``scripts/bench_run.py``.

    uv run python scripts/bench_run.py --jobs gallery/rgbs/jobs.py
    RGBS_VARIANTS=seam,thread RGBS_RECORD=1 uv run python scripts/bench_run.py --jobs gallery/rgbs/jobs.py

A job file rather than ``--task`` and ``--solve`` because a scripted routine has
no name ``--solve`` can reach. Main ships no routine runner, so the suite carries
its own (``tasks/rgbs/routine.py:SolutionRoutine``), and the task loader imports
a task directory under a package named for the root's path. This file adds the
root, works that name out, and points each routine variant's job at it. The
graph variants run the policy agent's ``SolutionGraph`` on the document's ``gt``.

Environment:
    RGBS_VARIANTS: comma-separated variants (default: every one below).
    RGBS_RECORD: ``1`` keeps each episode's ground-truth state, for a path-traced re-render
        (the gallery's clips are that re-render of the scored episode, from the variant's hero camera).
    RGBS_EPISODES: a half-open range ``A:B`` (default ``0:1``).
    RGBS_TAG: appended to each run directory's name.
"""

from __future__ import annotations

import os
from pathlib import Path

from robosimstudio.roots import add_task_root
from robosimstudio.specs import JobSpec, RecordSpec
from robosimstudio_bench.taskbuild.discovery import package_for_root

ROOT = add_task_root(Path(__file__).resolve().parent / "tasks")
ROUTINE = f"{package_for_root(ROOT)}.rgbs.routine:SolutionRoutine"
GRAPH = "robosimstudio_policy_agent.graph_solution:SolutionGraph"

VARIANTS: dict[str, tuple[str, tuple[tuple[str, object], ...], dict[str, object]]] = {
    # variant: (solution, its kwargs, any other JobSpec field)
    "handle": (ROUTINE, (), {}),
    "panel": (ROUTINE, (), {}),
    "panel_gentle": (ROUTINE, (), {}),
    "seam": (ROUTINE, (), {}),
    "thread": (ROUTINE, (), {}),
    "strip": (GRAPH, (("graph", "gt"),), {"clock": "lockstep"}),
    "wipe": (ROUTINE, (), {}),
}


def _jobs() -> list[JobSpec]:
    wanted = [v.strip() for v in os.environ.get("RGBS_VARIANTS", "").split(",") if v.strip()] or list(VARIANTS)
    record = os.environ.get("RGBS_RECORD", "") == "1"
    start, _, stop = os.environ.get("RGBS_EPISODES", "0:1").partition(":")
    tag = os.environ.get("RGBS_TAG", "")
    jobs = []
    for variant in wanted:
        solution, kwargs, fields = VARIANTS[variant]
        jobs.append(
            JobSpec(
                name=f"rgbs_{variant}{'_' + tag if tag else ''}",
                task=f"rgbs/{variant}",
                solution=solution,
                solution_kwargs=kwargs,
                episodes=(int(start), int(stop)),
                record=RecordSpec(enabled=record, record_video=False),
                **fields,
            )
        )
    return jobs


JOBS = _jobs()
