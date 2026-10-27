# SPDX-License-Identifier: Apache-2.0
"""The IK backend: which member answers each verb, how a line is gated, and why it was refused.

Two members: the solver family behind the protocol, which answers every
``solve_ik`` and the solved line; and an optional collision-aware planner,
which ``plan_linear`` prefers and ``plan_to_pose`` needs. Three rules the
runs were lost to before they were written down:

* ``honours_roll`` is the **solver's**, never the planner's. A servo that
  tested arrival on the planner's full-pose promise while the solver aligned
  an axis waited on a 4.3-degree roll error against a 3-degree gate, forever.
* each member gets the keywords its own signature declares and no others.
  Forwarding the union killed ``plan_linear`` with ``TypeError`` on the first
  descent of every hanging episode under the planner.
* a planner that finds no path falls back to the solved line -- unless the
  request carried world geometry or a contact allowance the line cannot
  honour, when a trajectory reported under a constraint nobody applied is
  worse than a refusal.

``planned: true`` is a gate, not an arrival, and the gate is adjustable.
``IKBackend.tolerance`` defaults to 30 mm. A hammer's handle is 36 mm across, so
a plan can pass the default gate and put the fingers beside the object rather
than around it -- measured on the hammer grid at 21-24 mm on the episodes that
failed, against 1.7-2.0 mm on the ones that worked. A line reports the arrival
of its own last waypoint, never an earlier line's or a fresh solve's.

A refused line says which of three things went wrong, because they want
opposite fixes: an out-of-reach target has to move, a reachable target at the
end of an impossible straight line wants ``motion.plan_joint``, and a wrist
pinned by ``orientation="lock"`` wants the lock dropped. One study iteration was
spent guessing between them, at a cost of twelve episodes.
"""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio_policy_agent.ik import IKBackend
from robosimstudio_policy_agent.motion_tools import MotionTools
from tests.fixtures.policy import StubSolver, ik_backend

TIER = "fast"
"""Real backends over a one-arm stub env and stubbed solvers and planners: no GPU, no scene."""


def _backend(planner=None, *, honours_roll=False, line=None, prefer_planned=True) -> IKBackend:
    """A backend with no solver, whose own solved line records its keywords: the routing under test needs none."""
    backend = ik_backend(planner=planner, roll="held" if honours_roll else "free", prefer_planned=prefer_planned)
    backend.line_calls = []

    def own(
        start, end, *, arm_id=0, tcp_offset=None, seed_joints=None, num_waypoints=0, hold_rotation=False, tolerance=None
    ):
        backend.line_calls.append(
            {"arm_id": arm_id, "num_waypoints": num_waypoints, "hold_rotation": hold_rotation, "tolerance": tolerance}
        )
        return line

    backend._own_plan_linear = own
    return backend


def _missing(error: float) -> IKBackend:
    """A backend whose solver lands every solve ``error`` metres short, whatever it is asked."""
    return ik_backend(StubSolver(error), waypoints=4, iterations=4)


def _line_backend(solver) -> IKBackend:
    return ik_backend(solver, waypoints=4, iterations=4)


def _solver(position_error, rotation_error) -> IKBackend:
    """A backend over a solver that always lands a fixed miss."""
    return ik_backend(StubSolver(position_error, rotation_error), waypoints=10, iterations=4)


def _ik_with(planner) -> IKBackend:
    ik = ik_backend(planner=planner)
    ik.last_position_error = 0.0123  # an earlier solved line's arrival
    return ik


def _pose(x: float) -> dict:
    return {"position": {"x": x, "y": 0.0, "z": 0.0}, "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}


class _Planner:
    """Stands in for a cuRobo planner: world-aware, full-pose, its own refusal text."""

    honours_roll = True
    trajectory_needs_joint_reverse = False
    supports_world_aware_plan = True

    def __init__(self, answer=None):
        self.answer = answer
        self.calls: list[dict] = []
        self.last_refusal = "planner: no path"
        self.planned_to_pose = 0

    def plan_linear(self, start, end, *, arm_id=0, axis="", allow_start_contact=False):
        self.calls.append({"arm_id": arm_id, "axis": axis, "allow_start_contact": allow_start_contact})
        return self.answer

    def plan_to_pose(self, pose, **kwargs):
        self.planned_to_pose += 1
        return {"waypoints": [], "via": "planner"}

    def robot_collision_spheres(self, arm_id=-1):
        return [{"radius": 0.01}]

    def build_world_tsdf(self, *args, **kwargs):
        return {"voxels": 0}


class TestTheRollIsTheSolvers(unittest.TestCase):
    def test_the_planner_cannot_make_the_backend_honour_roll(self):
        self.assertFalse(_backend(_Planner()).honours_roll)

    def test_a_held_roll_is_reported_whether_or_not_a_planner_is_installed(self):
        self.assertTrue(_backend(_Planner(), honours_roll=True).honours_roll)
        self.assertTrue(_backend(None, honours_roll=True).honours_roll)


class TestEachMemberGetsItsOwnKeywords(unittest.TestCase):
    def test_the_planner_gets_only_what_it_declares(self):
        planner = _Planner({"waypoints": [1]})
        backend = _backend(planner, line={"waypoints": [2]})
        out = backend.plan_linear({}, {}, arm_id=1, tolerance=0.01, hold_rotation=True, axis="z")
        self.assertEqual(out, {"waypoints": [1]})
        self.assertEqual(planner.calls, [{"arm_id": 1, "axis": "z", "allow_start_contact": False}])
        self.assertEqual(backend.line_calls, [])

    def test_the_fallback_gets_the_lines_keywords(self):
        planner = _Planner(None)
        backend = _backend(planner, line={"waypoints": [2]})
        out = backend.plan_linear({}, {}, arm_id=1, tolerance=0.01, hold_rotation=True, axis="z")
        self.assertEqual(out, {"waypoints": [2]})
        self.assertEqual(
            backend.line_calls, [{"arm_id": 1, "num_waypoints": 0, "hold_rotation": True, "tolerance": 0.01}]
        )

    def test_no_planner_means_the_line_alone(self):
        backend = _backend(None, line={"waypoints": [2]})
        self.assertEqual(backend.plan_linear({}, {}, axis="z", tolerance=0.02), {"waypoints": [2]})
        self.assertEqual(
            backend.line_calls, [{"arm_id": 0, "num_waypoints": 0, "hold_rotation": False, "tolerance": 0.02}]
        )


class TestTheFallbackIsRefusedWhenItCannotHonourTheRequest(unittest.TestCase):
    def test_world_geometry_is_never_discarded(self):
        backend = _backend(_Planner(None), line={"waypoints": [2]})
        self.assertIsNone(backend.plan_linear({}, {}, world_config={"cuboid": {}}))
        self.assertIn("world/attachment", backend.last_refusal)
        self.assertEqual(backend.line_calls, [])

    def test_a_contact_allowance_is_refused_rather_than_dropped(self):
        backend = _backend(_Planner(None), line={"waypoints": [2]})
        self.assertIsNone(backend.plan_linear({}, {}, allow_start_contact=True))
        self.assertIn("allow_start_contact", backend.last_refusal)
        self.assertEqual(backend.line_calls, [])

    def test_the_lines_own_refusal_is_not_shadowed(self):
        backend = _backend(_Planner(None), line=None)

        def own(start, end, **kwargs):
            backend.last_refusal = "line: waypoint 3 missed by 41 mm"
            return None

        backend._own_plan_linear = own
        self.assertIsNone(backend.plan_linear({}, {}, tolerance=0.01))
        self.assertEqual(backend.last_refusal, "line: waypoint 3 missed by 41 mm")

    def test_the_planners_refusal_is_reported_when_nothing_else_answered(self):
        backend = _backend(_Planner(None), line=None)
        self.assertIsNone(backend.plan_linear({}, {}))
        self.assertEqual(backend.last_refusal, "planner: no path")


class TestThePlannersVerbsAreThePlanners(unittest.TestCase):
    def test_plan_to_pose_is_none_without_a_planner_and_never_falls_back(self):
        self.assertIsNone(_backend(None).plan_to_pose({}))
        planner = _Planner()
        self.assertEqual(_backend(planner).plan_to_pose({})["via"], "planner")
        self.assertEqual(planner.planned_to_pose, 1)

    def test_world_awareness_follows_the_planner(self):
        self.assertTrue(_backend(_Planner()).supports_world_aware_plan)
        self.assertFalse(_backend(None).supports_world_aware_plan)
        self.assertFalse(_backend(_Planner(), prefer_planned=False).supports_world_aware_plan)

    def test_prefer_planned_false_routes_everything_to_the_line(self):
        planner = _Planner({"waypoints": [1]})
        backend = _backend(planner, line={"waypoints": [2]}, prefer_planned=False)
        self.assertEqual(backend.plan_linear({}, {}), {"waypoints": [2]})
        self.assertEqual(planner.calls, [])
        self.assertIsNone(backend.plan_to_pose({}))

    def test_the_world_builder_and_the_batch_solve_are_absent_without_a_planner(self):
        backend = _backend(None)
        with self.assertRaises(AttributeError):
            _ = backend.build_world_tsdf
        with self.assertRaises(AttributeError):
            _ = backend.solve_ik_batch
        self.assertEqual(_backend(_Planner()).build_world_tsdf(), {"voxels": 0})

    def test_a_planners_own_attribute_is_reachable_and_an_unknown_one_is_not(self):
        self.assertEqual(_backend(_Planner()).planned_to_pose, 0)
        with self.assertRaises(AttributeError):
            _ = _backend(None).no_such_thing


POSE = {"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}


class TestTheGateIsAdjustable(unittest.TestCase):
    def test_a_20mm_miss_passes_the_default_gate(self):
        self.assertIsNotNone(_missing(0.020).plan_linear(POSE, POSE))

    def test_and_is_refused_when_the_caller_asks_for_5mm(self):
        self.assertIsNone(_missing(0.020).plan_linear(POSE, POSE, tolerance=0.005))

    def test_a_plan_that_passes_still_reports_how_far_it_landed(self):
        b = _missing(0.020)
        b.plan_linear(POSE, POSE)
        self.assertAlmostEqual(b.last_position_error, 0.020, places=6)


class _Kinematics:
    dof = 1
    arm_coords = (0,)


class _Converging:
    """Each solve lands closer than the one before, as an unconverged one does."""

    capabilities = frozenset()
    kinematics = _Kinematics

    def __init__(self):
        self.calls = 0

    def solve(self, seed, position, quat_xyzw, *, roll="free", active=None, iterations=0):
        self.calls += 1
        seed = np.asarray(seed, dtype=np.float64).reshape(-1, 1)
        return seed, np.tile([0.02 / self.calls, 0.0], (seed.shape[0], 1))


class _ByPlace:
    """Misses by ``mid`` between the ends, ``end_error`` at x=1, and 0.5 m whenever roll is held."""

    capabilities = frozenset()
    kinematics = _Kinematics

    def __init__(self, mid, end_error):
        self.mid, self.end_error = mid, end_error

    def solve(self, seed, position, quat_xyzw, *, roll="free", active=None, iterations=0):
        seed = np.asarray(seed, dtype=np.float64).reshape(-1, 1)
        x = float(np.asarray(position).reshape(-1, 3)[0, 0])
        error = 0.5 if roll == "held" else (self.end_error if x >= 1.0 else self.mid)
        return seed, np.tile([error, 0.0], (seed.shape[0], 1))


class TestALinesArrival(unittest.TestCase):
    def test_is_the_last_waypoints_error_not_a_fresh_solves(self):
        b = _line_backend(_Converging())
        self.assertIsNotNone(b.plan_linear(_pose(0.0), _pose(1.0), num_waypoints=4))
        self.assertAlmostEqual(b.last_position_error, 0.02 / 4, places=9)


class TestWhyNoLine(unittest.TestCase):
    def test_a_roll_free_end_is_judged_roll_free_at_the_callers_tolerance(self):
        b = _line_backend(_ByPlace(mid=0.0, end_error=0.02))
        self.assertIsNone(b.plan_linear(_pose(0.0), _pose(1.0), tolerance=0.01))
        self.assertIn("out of reach", b.last_refusal)
        self.assertIn("20 mm against a 10 mm tolerance", b.last_refusal)

    def test_a_reachable_end_behind_an_impossible_line_says_so(self):
        b = _line_backend(_ByPlace(mid=0.2, end_error=0.04))
        self.assertIsNone(b.plan_linear(_pose(0.0), _pose(1.0), tolerance=0.05))
        self.assertIn("reachable on its own", b.last_refusal)


class _Backend:
    """Just enough of a solver to answer a plan_linear call."""

    honours_roll = False

    def __init__(self, plan=None, refusal=""):
        self._plan = plan
        self.last_refusal = refusal

    def plan_linear(self, start, end, **kwargs):
        return self._plan

    def current_pose(self, arm_id=0):
        return {"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}


class _Mute(_Backend):
    """A backend that refuses without explaining — curobo, pyroki, a stub."""

    def __init__(self):
        super().__init__(plan=None)
        del self.last_refusal


_END = {"position": {"x": 0.4, "y": 0.0, "z": 0.3}, "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}


class TestTheRefusalCarriesTheBackendsReason(unittest.TestCase):
    def test_the_solvers_own_account_reaches_the_caller(self):
        detail = (
            "orientation='lock' holds the end pose's whole rotation along "
            "the leg; at waypoint 3 of 40 the solver was 2 mm and 41 deg off"
        )
        tools = MotionTools(_Backend(refusal=detail))
        result = tools.plan_linear(end=_END)
        self.assertFalse(result["planned"])
        self.assertEqual(result["reason"], detail)

    def test_a_backend_that_does_not_explain_keeps_the_generic_reason(self):
        tools = MotionTools(_Mute())
        result = tools.plan_linear(end=_END)
        self.assertFalse(result["planned"])
        self.assertEqual(result["reason"], "no linear plan holds this move's constraint")

    def test_a_plan_that_succeeded_carries_no_reason(self):
        plan = {"waypoints": [{"positions": [0.0]}]}
        # Stale text from an earlier refusal must not be read as this call's.
        tools = MotionTools(_Backend(plan=plan, refusal="something from before"))
        result = tools.plan_linear(end=_END)
        self.assertTrue(result["planned"])
        self.assertEqual(result["reason"], "")


_POSE = {"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}


class TestTheHeldRotationRefusalNamesTheBindingConstraint(unittest.TestCase):
    """`orientation="lock"` is the cause a caller is least able to guess.

    Six of twelve episodes in one study iteration died here, and the agent
    spent the next iteration inferring from node names alone that the wrist
    was the problem.
    """

    def _refuse(self, position_error, rotation_error):
        backend = _solver(position_error, rotation_error)
        result = backend.plan_linear(_POSE, _POSE, hold_rotation=True)
        self.assertIsNone(result)
        return backend.last_refusal

    def test_a_rotation_that_cannot_be_held_says_so(self):
        reason = self._refuse(position_error=0.001, rotation_error=0.7)
        self.assertIn("orientation='lock'", reason)
        self.assertIn("the held rotation", reason)
        self.assertIn("40 deg", reason)  # 0.7 rad

    def test_a_position_miss_is_not_blamed_on_the_rotation(self):
        reason = self._refuse(position_error=0.25, rotation_error=0.0)
        self.assertIn("the position", reason)
        self.assertNotIn("the held rotation is what", reason)

    def test_both_missing_is_reported_as_both(self):
        reason = self._refuse(position_error=0.25, rotation_error=0.7)
        self.assertIn("both position and the held rotation", reason)

    def test_it_names_which_waypoint_of_how_many(self):
        reason = self._refuse(position_error=0.25, rotation_error=0.0)
        self.assertIn("waypoint 1 of 10", reason)


class _AnsweringPlanner:
    honours_roll = True
    last_refusal = ""

    def plan_linear(self, start, end, **kwargs):
        return {"waypoints": [{"positions": [0.0]}]}


class TestPlannedLineArrival(unittest.TestCase):
    def test_a_planner_answer_does_not_carry_the_last_lines_error(self):
        result = MotionTools(_ik_with(_AnsweringPlanner())).plan_linear(_pose(0.1), start=_pose(0.0))
        self.assertTrue(result["planned"])
        self.assertIsNone(result["position_error_m"])


class TestDescribePlanner(unittest.TestCase):
    def test_honours_roll_is_the_planners(self):
        self.assertTrue(MotionTools(_ik_with(_AnsweringPlanner())).describe_planner()["honours_roll"])

    def test_without_a_planner_it_is_the_solvers(self):
        self.assertFalse(MotionTools(_ik_with(None)).describe_planner()["honours_roll"])


if __name__ == "__main__":
    unittest.main()
