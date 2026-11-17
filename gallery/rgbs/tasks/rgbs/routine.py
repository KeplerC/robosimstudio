# SPDX-License-Identifier: Apache-2.0
"""The policy that runs this suite's scripted routines through ``scripts/bench_run.py``.

Main ships no routine runner (``robosimstudio_bench.solutions.routines`` is
left out of the release, and a document naming ``(:routine ...)`` is refused at
lowering), so a variant whose solution is a routine names it in its
``(:policy-params (routine "MODULE:CLASS"))`` instead, and this class runs it::

    ROBOSIMSTUDIO_TASK_ROOTS=$PWD/gallery/rgbs/tasks uv run python scripts/bench_run.py \\
        --task rgbs/seam --solve gallery.rgbs.tasks.rgbs.routine:SolutionRoutine

``bench_run`` imports this module by its dotted path from the checkout, which is
a second import of this file beside the task loader's. So it imports nothing of
the suite at module level: the routine class comes from the copy the task
loader already executed (:func:`~robosimstudio_bench.taskbuild.discovery.task_module`),
and the suite's registrations run once.
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
