# SPDX-License-Identifier: Apache-2.0
"""The policy agent's backends, built for a test without a scene.

Five test modules each grew an ``IKBackend.__new__`` with seventeen private
attributes set by hand, and the copies had drifted: only one set
``rot_tolerance``, two left ``_arm_name`` and ``_seed`` as the real methods and
the rest replaced them. :func:`ik_backend` goes through the real constructor
instead, over the smallest env that answers what the constructor and a solve
read -- one arm, its coordinates, a joint vector -- so a field the constructor
grows reaches every test at once.

:class:`InlineBus` is the bus every one of those modules wrote as ``_Bus``: it
runs a marshalled call on the caller's thread, and counts it.
:class:`StubSolver` is the solver they wrote as ``_Solver`` and ``_StubSolver``:
it lands every solve a fixed miss away, whatever it is asked.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from robosimstudio_policy_agent.ik import IKBackend
from tests.fixtures.arrays import host

__all__ = ["InlineBus", "StubSolver", "ik_backend", "pose"]


class InlineBus:
    """``call(fn, *args, **kwargs)`` runs ``fn`` here and now; ``calls`` counts them."""

    def __init__(self) -> None:
        self.calls = 0

    def call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return fn(*args, **kwargs)


class StubSolver:
    """A one-joint IK solver that misses every solve by ``(position_error, rotation_error)``.

    It returns its seed unchanged as the answer, so a line it plans is the seed
    repeated.
    """

    capabilities = frozenset()
    kinematics = SimpleNamespace(dof=1, arm_coords=(0,))

    def __init__(self, position_error: float = 0.0, rotation_error: float = 0.0) -> None:
        self.errors = (float(position_error), float(rotation_error))

    def solve(self, seed, position, quat_xyzw, *, roll="free", active=None, iterations=0):
        del position, quat_xyzw, roll, active, iterations
        seed = np.asarray(seed, dtype=np.float64).reshape(-1, 1)
        return seed, np.tile(self.errors, (seed.shape[0], 1))


def ik_backend(
    solver: Any = None,
    *,
    planner: Any = None,
    roll: str = "free",
    tolerance: float = 0.03,
    waypoints: int = 40,
    iterations: int = 80,
    prefer_planned: bool = True,
    bus: Any = None,
) -> IKBackend:
    """A real :class:`IKBackend` over a one-arm stub env named ``"arm"``, solving with ``solver``.

    The env answers exactly what the constructor and a solve read: the arm's
    name (``scene.embodiment.iks``), its coordinates (``env.arm(name)``, the
    solver's own kinematics) and a zero joint vector to seed from. Without a
    ``solver`` the backend has none built, which is what a test of the
    planner's routing wants: any solve would try to build one from the scene.
    """
    kinematics = getattr(solver, "kinematics", None) or SimpleNamespace(dof=1, arm_coords=(0,))
    env = SimpleNamespace(
        scene=SimpleNamespace(embodiment=SimpleNamespace(iks={"arm": kinematics})),
        handles=None,
        state=SimpleNamespace(joint_q=host(np.zeros(len(kinematics.arm_coords)))),
        arm=lambda name: kinematics,
    )
    backend = IKBackend(
        env,
        bus if bus is not None else InlineBus(),
        roll=roll,
        planner=planner,
        tolerance=tolerance,
        waypoints=waypoints,
        iterations=iterations,
        solver=solver,
    )
    backend.prefer_planned = prefer_planned
    return backend


def pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> dict:
    """An ``Se3Pose`` at ``(x, y, z)`` with the identity rotation."""
    return {"position": {"x": x, "y": y, "z": z}, "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}
