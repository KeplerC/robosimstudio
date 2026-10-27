# SPDX-License-Identifier: Apache-2.0
"""The fleet stepper's ``StepService``: how a motion is installed, what it is given, when it
is over, and the motions a wave cannot perform refused on the node that asked for them."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from robosimstudio.core.bus import BusTimeout
from robosimstudio_policy_agent.fleet import stepper
from robosimstudio_policy_agent.fleet.intents import GripperMove, JointTrajectory, MoveToJoints, ServoToPose
from tests.fixtures.fleet import FakeFleet, FakeSnapshot, step_service

TIER = "fast"
"""A fake fleet over the real intent table, and scripted snapshots: no GPU, no scene."""


class ReachVerdictCadenceTest(unittest.TestCase):
    def test_judged_once_per_stall_window(self) -> None:
        service = object.__new__(stepper.StepService)
        service.arm_names = ["arm"]
        service.cadence = 1
        service.step = 0
        service.fleet = SimpleNamespace(
            scene=SimpleNamespace(embodiment=SimpleNamespace(arms={"arm": SimpleNamespace(ee_body=0)}))
        )
        service._offsets = lambda name: ()
        service._release_arm = lambda world, name: None
        judged: list[int] = []
        service._beyond_reach = lambda world, name, program, target, snapshot: judged.append(program.stalled) or False
        program = stepper._Program(
            "servo",
            goal=(np.array([1.0, 0.0, 0.0]), "arm", None),
            deadline=1000,
            tolerance=0.001,
            stall_steps=12,
        )
        snapshot = SimpleNamespace(body=lambda world: None)
        with mock.patch.object(stepper, "tcp_pose", lambda *a: (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))):
            for tick in range(60):
                service.step = tick
                done, _error, _result = service._finished(0, 0, program, snapshot)
                self.assertFalse(done)
        # Stalled from the second readback: windows close at 12, 24, 36, 48.
        self.assertEqual(judged, [12, 24, 36, 48])


_POSE = {"position": np.array([0.4, 0.0, 0.3]), "quat_xyzw": np.array([0.0, 0.0, 0.0, 1.0])}


def _install(service, *intents, world: int = 0, live=(True, True)):
    """Submit ``intents`` for ``world`` as a graph thread would, install them, and return the handles."""
    handles = [service.intents.submit(world, intent.arm, intent) for intent in intents]
    service.install(np.array(live))
    return handles


def _refusal(handle) -> str:
    """What a node waiting on ``handle`` is woken with: the refusal it raises."""
    try:
        handle.wait(timeout=1.0)
    except stepper.MotionNotInstallable as exc:
        return str(exc)
    raise AssertionError("the motion was not refused")


class TestAMotionThisFleetCannotPerformIsRefused(unittest.TestCase):
    """``install``: a motion that cannot run raises on the waiting node, rather than completing as if it ran."""

    def test_an_arm_the_fleet_does_not_have(self):
        service = step_service()
        handle = service.intents.submit(0, 0, MoveToJoints(arm=0, target=np.zeros(2)))
        service.arm_names = []
        service.install(np.array([True, True]))
        self.assertIn("arm 0 is not one of this fleet's arms []", _refusal(handle))

    def test_a_joint_move_or_trajectory_without_a_segment_driver(self):
        for intent in (MoveToJoints(arm=0, target=np.zeros(2)), JointTrajectory(arm=0, waypoints=np.zeros((3, 2)))):
            with self.subTest(intent=type(intent).__name__):
                (handle,) = _install(step_service(FakeFleet(segments=False)), intent)
                self.assertIn("has no joint-segment driver", _refusal(handle))

    def test_a_trajectory_with_no_waypoints(self):
        (handle,) = _install(step_service(), JointTrajectory(arm=0, waypoints=np.zeros((0, 2))))
        self.assertIn("carries no waypoints", _refusal(handle))

    def test_a_hand_command_without_a_gripper_bank(self):
        (handle,) = _install(step_service(FakeFleet(gripper=False)), GripperMove(arm=0, command=1.0))
        self.assertIn("has no gripper bank", _refusal(handle))

    def test_a_cartesian_move_on_a_wave_with_no_ik_rig_says_what_to_do_instead(self):
        (handle,) = _install(step_service(), ServoToPose(arm=0, **_POSE))
        refusal = _refusal(handle)
        self.assertIn("no batched IK rig", refusal)
        self.assertIn("ik tier 'host': the model is too hostile", refusal)
        self.assertIn("motion.plan_linear", refusal)

    def test_a_cartesian_move_for_an_arm_the_rig_does_not_solve(self):
        rig = SimpleNamespace(arms={"other": object()})
        (handle,) = _install(step_service(FakeFleet(ik=rig)), ServoToPose(arm=0, **_POSE))
        self.assertIn("is not in this wave's IK rig ['other']", _refusal(handle))

    def test_a_world_whose_episode_ended_times_out_instead(self):
        service = step_service()
        (live,) = _install(service, MoveToJoints(arm=0, target=np.zeros(2)), world=1, live=(True, False))
        with self.assertRaisesRegex(BusTimeout, "world 1's episode has ended"):
            live.wait(timeout=1.0)
        service.retire(0)
        self.assertEqual(service.service.dropped, [0])
        (retired,) = _install(service, MoveToJoints(arm=0, target=np.zeros(2)), world=0)
        with self.assertRaises(BusTimeout):
            retired.wait(timeout=1.0)


class TestWhatAnInstalledMotionIsGiven(unittest.TestCase):
    def test_a_joint_move_starts_from_where_the_arm_is_commanded(self):
        fleet = FakeFleet(targets=[0.0, 0.0, 0.0, 0.3, -0.2, 0.9])
        service = step_service(fleet)
        service.step = 10
        (handle,) = _install(service, MoveToJoints(arm=0, target=np.array([1.0, 1.0]), duration_steps=20), world=1)
        self.assertFalse(handle.done)
        ((world, start, target, duration),) = fleet.segments["arm"].plans
        self.assertEqual((world, duration), (1, 20))
        np.testing.assert_allclose(start, [0.3, -0.2])
        np.testing.assert_allclose(target, [1.0, 1.0])
        (_, _, _, program, _) = service.intents.active()[0]
        self.assertEqual((program.settle_until, program.deadline), (30, 1010))

    def test_a_fire_and_forget_joint_move_is_given_its_duration_and_one_step(self):
        service = step_service()
        _install(service, MoveToJoints(arm=0, target=np.zeros(2), duration_steps=20, max_steps=0))
        (_, _, _, program, _) = service.intents.active()[0]
        self.assertEqual(program.deadline, 21)

    def test_a_trajectory_walks_every_waypoint(self):
        fleet = FakeFleet()
        service = step_service(fleet)
        waypoints = np.linspace([0.0, 0.0], [1.0, 1.0], 5)
        _install(service, JointTrajectory(arm=0, waypoints=waypoints, steps_per_waypoint=3, settle_max_steps=7))
        self.assertEqual(fleet.segments["arm"].paths, [(0, 5, 3)])
        (_, _, _, program, _) = service.intents.active()[0]
        np.testing.assert_allclose(program.goal, [1.0, 1.0])
        self.assertEqual((program.settle_until, program.deadline), (15, 22))

    def test_a_hand_is_given_its_ramp_and_settle_or_forty_steps_whichever_is_longer(self):
        for ramp, settle, deadline in ((50, 10, 60), (2, 3, 40)):
            with self.subTest(ramp=ramp, settle=settle):
                fleet = FakeFleet()
                service = step_service(fleet)
                _install(service, GripperMove(arm=0, command=1.0, ramp_steps=ramp, settle_steps=settle))
                (_, _, _, program, _) = service.intents.active()[0]
                self.assertEqual((program.deadline, program.settle_until), (deadline, ramp + settle))
                self.assertEqual(fleet.grippers["arm"].commands, [(0, 1.0, ramp)])

    def test_an_arm_motion_releases_the_arms_drivers_and_a_hand_command_does_not(self):
        fleet = FakeFleet()
        service = step_service(fleet)
        _install(service, GripperMove(arm=0, command=1.0))
        self.assertEqual(fleet.segments["arm"].stopped, [])
        _install(service, MoveToJoints(arm=0, target=np.zeros(2)))
        self.assertEqual(fleet.segments["arm"].stopped, [0])


def _program(kind: str, **kwargs):
    return stepper._Program(kind, **{"deadline": 100, "tolerance": 0.01, **kwargs})


class TestWhenAMotionIsOver(unittest.TestCase):
    """``_finished``: arrival, the budget, and what the waiting node is handed."""

    def setUp(self):
        self.fleet = FakeFleet()
        self.service = step_service(self.fleet)

    def test_a_joint_move_is_not_judged_before_its_segment_lands(self):
        program = _program("joints", goal=np.array([0.5, 0.5]), settle_until=20)
        self.service.step = 19
        self.assertEqual(self.service._finished(0, 0, program, FakeSnapshot([0.5, 0.5, 0.0])), (False, None, None))

    def test_a_joint_move_arrives_inside_its_tolerance_and_lets_go_of_the_arm(self):
        program = _program("joints", goal=np.array([0.5, 0.5]), settle_until=20)
        self.service.step = 20
        self.assertEqual(self.service._finished(0, 0, program, FakeSnapshot([0.5, 0.505, 0.0])), (True, None, True))
        self.assertEqual(self.fleet.segments["arm"].stopped, [0])
        self.assertEqual(self.service._finished(0, 0, program, FakeSnapshot([0.5, 0.52, 0.0])), (False, None, None))

    def test_a_joint_move_out_of_budget_ends_unconverged_rather_than_raising(self):
        program = _program("joints", goal=np.array([0.5, 0.5]), deadline=50)
        self.service.step = 50
        done, error, converged = self.service._finished(0, 0, program, FakeSnapshot([0.0, 0.0, 0.0]))
        self.assertEqual((done, error), (True, None))
        self.assertIs(converged, False)

    def test_a_hand_ends_early_on_reaching_its_command_and_on_its_budget_when_it_holds_something(self):
        program = _program("gripper", goal=(1.0, "arm"), deadline=40, settle_until=10, tolerance=0.002)
        self.service.step = 5
        self.assertFalse(self.service._finished(0, 0, program, FakeSnapshot([0, 0, 0.8], targets=[0, 0, 0.8]))[0])
        self.service.step = 12
        self.assertTrue(self.service._finished(0, 0, program, FakeSnapshot([0, 0, 0.799], targets=[0, 0, 0.8]))[0])
        self.assertFalse(self.service._finished(0, 0, program, FakeSnapshot([0, 0, 0.5], targets=[0, 0, 0.8]))[0])
        self.service.step = 40
        self.assertEqual(
            self.service._finished(0, 0, program, FakeSnapshot([0, 0, 0.5], targets=[0, 0, 0.8])), (True, None, None)
        )

    def test_a_servo_completes_with_the_serial_servos_outcome(self):
        body = np.array([[0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]])
        reached = _program("servo", goal=(np.array([0.4, 0.0, 0.305]), "arm", None))
        done, error, result = self.service._finished(0, 0, reached, FakeSnapshot([0, 0, 0], body=body))
        self.assertEqual((done, error, result["outcome"], result["steps"]), (True, None, "reached", 1))
        self.assertAlmostEqual(result["remaining_distance"], 0.005)
        late = _program("servo", goal=(np.array([0.6, 0.0, 0.3]), "arm", None), deadline=0)
        self.assertEqual(
            self.service._finished(0, 0, late, FakeSnapshot([0, 0, 0], body=body))[2]["outcome"], "timeout"
        )

    def test_a_stall_stopping_servo_reports_contact_after_its_window(self):
        body = np.array([[0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]])
        program = _program("servo", goal=(np.array([0.4, 0.0, 0.32]), "arm", None), stop_on_stall=True, stall_steps=3)
        outcomes = [self.service._finished(0, 0, program, FakeSnapshot([0, 0, 0], body=body)) for _ in range(4)]
        self.assertEqual([done for done, _, _ in outcomes], [False, False, False, True])
        self.assertEqual(outcomes[-1][2]["outcome"], "contact")


if __name__ == "__main__":
    unittest.main()
