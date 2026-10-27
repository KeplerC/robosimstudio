# SPDX-License-Identifier: Apache-2.0
"""The fleet side of a batched policy, without a fleet.

:class:`FakeFleet` is the part of a ``FleetEnv`` the stepper touches -- one arm
called ``"arm"``, its joint-segment driver and gripper bank (each recording what
it was told), a control vector of commanded targets, and the IK rig, absent
unless a test hands one in -- and :class:`FakeSnapshot` is one readback of it.
:func:`step_service` builds the real :class:`StepService` over the real
:class:`IntentTable`, so a test submits a motion the way a graph thread does and
reads its fate off the handle the thread would wait on.

:class:`GivesUpOnceStarted` is for the one race every "a late answer" test has:
the asker must give up *while* its request is being served. With a real
``Event.wait(0.1)`` that holds only when the serving thread gets there inside
the 100 ms, which a loaded host does not promise, and a test that lost the race
failed on ``started.is_set()`` rather than on anything it was about. Handed to
the request as its ``done`` event, the asker's timed wait times out exactly
once the request is running, and never before.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import numpy as np

from robosimstudio.core.worlds import WorldMap
from robosimstudio_policy_agent.fleet.intents import IntentTable
from robosimstudio_policy_agent.fleet.stepper import StepService
from tests.fixtures.arrays import host

__all__ = ["FakeBank", "FakeFleet", "FakeSegment", "FakeSnapshot", "GivesUpOnceStarted", "step_service"]

ARM_COORDS = (0, 1)
"""The arm's joint coordinates in one world's vector; the hand's finger is coordinate 2."""
FINGER = 2


class FakeSegment:
    """A joint-segment driver: records each plan and each stop."""

    def __init__(self) -> None:
        self.plans: list[tuple[int, np.ndarray, np.ndarray, int]] = []
        self.paths: list[tuple[int, int, int]] = []
        self.stopped: list[int] = []

    def plan(self, world: int, start, target, duration: int) -> None:
        self.plans.append((world, np.array(start), np.array(target), int(duration)))

    def plan_path(self, world: int, waypoints, steps_per_waypoint: int) -> int:
        self.paths.append((world, len(waypoints), int(steps_per_waypoint)))
        return len(waypoints) * int(steps_per_waypoint)

    def stop(self, world: int) -> None:
        self.stopped.append(world)


class FakeBank:
    """A gripper bank over one finger coordinate: records each command."""

    def __init__(self) -> None:
        self.coords = host(np.array([FINGER]))
        self.commands: list[tuple[int, float, int]] = []

    def set(self, world: int, command: float, ramp_steps: int = 0) -> None:
        self.commands.append((world, float(command), int(ramp_steps)))


class FakeFleet:
    """``num_worlds`` copies of a one-arm robot, as far as the stepper reads one.

    ``targets`` is each world's commanded joint vector (three coordinates:
    the arm's two, then the finger); ``segments`` and ``gripper`` say whether the
    arm has a joint-segment driver and a hand.
    """

    def __init__(self, num_worlds: int = 2, *, targets=None, segments: bool = True, gripper: bool = True, ik=None):
        width = len(ARM_COORDS) + 1
        self.worlds = WorldMap(
            num_worlds=num_worlds,
            bodies_per_world=1,
            shapes_per_world=0,
            joints_per_world=width,
            coords_per_world=width,
            dofs_per_world=width,
            particles_per_world=0,
        )
        flat = np.zeros(num_worlds * width, dtype=np.float32) if targets is None else np.asarray(targets, np.float32)
        arm = SimpleNamespace(ee_body=0, arm_coords=list(ARM_COORDS))
        self.scene = SimpleNamespace(embodiment=SimpleNamespace(arms={"arm": arm}, iks={}))
        self.pipeline = SimpleNamespace(control=SimpleNamespace(joint_target_q=host(flat.reshape(-1))))
        self.segments = {"arm": FakeSegment()} if segments else {}
        self.grippers = {"arm": FakeBank()} if gripper else {}
        self.ik = ik
        self.ik_plan = SimpleNamespace(tier="host", reason="the model is too hostile for a batched solve")


class FakeSnapshot:
    """One readback: each world's joint coordinates, commanded targets and body rows."""

    def __init__(self, coords, *, targets=None, body=None, step: int = 0) -> None:
        self._coords = np.atleast_2d(np.asarray(coords, dtype=np.float64))
        self._targets = None if targets is None else np.atleast_2d(np.asarray(targets, dtype=np.float64))
        self._body = None if body is None else np.asarray(body, dtype=np.float64)
        self.step = step

    def coords(self, world: int) -> np.ndarray:
        return self._coords[min(world, len(self._coords) - 1)]

    def targets(self, world: int) -> np.ndarray | None:
        return None if self._targets is None else self._targets[min(world, len(self._targets) - 1)]

    def body(self, world: int) -> np.ndarray:
        return self._body


def step_service(fleet: Any = None, *, cadence: int = 1, arm_names=("arm",)) -> StepService:
    """The real :class:`StepService` over the real :class:`IntentTable`, for ``fleet`` (a :class:`FakeFleet`).

    The marshalled-call queue is a recorder of the worlds it was told to drop;
    the snapshot table and step waiter are absent, so a test that reaches
    :meth:`StepService.settle` brings its own.
    """
    fleet = fleet or FakeFleet()
    intents = IntentTable(num_worlds=fleet.worlds.num_worlds, num_arms=len(arm_names))
    service = SimpleNamespace(dropped=[], pump=lambda: 0)
    service.drop_world = service.dropped.append
    return StepService(
        fleet, intents=intents, snapshots=None, service=service, waiter=None, arm_names=list(arm_names), cadence=cadence
    )


class GivesUpOnceStarted(threading.Event):
    """A ``done`` event whose *timed* wait returns ``False`` only once ``started`` is set.

    An untimed ``wait()`` is the real one: that is the asker waiting for an
    answer it knows is coming, which is not the race this removes.
    """

    def __init__(self, started: threading.Event, *, patience: float = 5.0) -> None:
        super().__init__()
        self.started = started
        self.patience = patience

    def wait(self, timeout: float | None = None) -> bool:
        if timeout is None:
            return super().wait()
        if not self.started.wait(self.patience):
            raise AssertionError(f"the request was never served in {self.patience} s")
        return super().wait(0)
