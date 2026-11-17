# SPDX-License-Identifier: Apache-2.0
"""The policy that runs this suite's scripted routines through ``scripts/bench_run.py``.

Main ships no routine runner (``robosimstudio_bench.solutions.routines`` is
left out of the release, and a document naming ``(:routine ...)`` is refused at
lowering), so a variant whose solution is a routine names it in its
``(:policy-params (routine "MODULE:CLASS"))`` instead, and this class runs it.
``gallery/rgbs/jobs.py`` points ``bench_run`` at it::

    RGBS_VARIANTS=seam uv run python scripts/bench_run.py --jobs gallery/rgbs/jobs.py

A task directory is imported under a package named for its root's path, so the
job file works that dotted name out rather than a user typing it. This module
imports nothing of the suite at module level all the same: the routine class
comes from the copy the task loader executed
(:func:`~robosimstudio_bench.taskbuild.discovery.task_module`), so the suite's
registrations run once however this module was reached.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosimstudio_bench.solutions.base import SolutionBase, solution_param

__all__ = ["SolutionRoutine", "routine_class"]


def routine_class(env) -> type:
    """The routine class the environment's document names in ``(:policy-params (routine ...))``."""
    from robosimstudio_bench.taskbuild.discovery import task_module

    reference = solution_param(env, "routine")
    if not reference:
        raise ValueError(f"task {env.task.full_name!r} names no routine in its (:policy-params ...)")
    module, _, name = str(reference).partition(":")
    return getattr(task_module(env.task.suite, module), name)


class SolutionRoutine(SolutionBase):
    """Drives the environment with the routine its document names, one action per control step.

    Args:
        env: The environment.
        **kwargs: Forwarded to the routine's constructor.
    """

    def __init__(self, env, **kwargs: Any) -> None:
        super().__init__(env, **kwargs)
        self.routine = routine_class(env)(env, **kwargs)

    def reset(self, obs: dict[str, Any], info: dict[str, Any]) -> None:
        del obs, info
        if hasattr(self.routine, "reset"):
            self.routine.reset()

    def get_action(self, obs: dict[str, Any]) -> np.ndarray:
        del obs
        return self.routine.act()

    @property
    def control_mode(self) -> str:
        return str(getattr(self.routine, "MODE", ""))

    @property
    def name(self) -> str:
        return f"scripted:{type(self.routine).__qualname__}"

    def row_extras(self) -> dict[str, Any]:
        return dict(getattr(self.routine, "row_extras", dict)())
