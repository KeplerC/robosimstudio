# SPDX-License-Identifier: Apache-2.0
"""The action spaces a policy writes into, and the set an environment switches between.

Over stub arms and hand-built states: which modes :func:`make_action_spaces`
builds and which errors it surfaces, the one gripper convention the two
Cartesian spaces share, how a command naming a mode re-seeds it, and the
``"tool"`` convention of ``ee_delta``. Then the round trips each space promises:
``target_action`` reads back what ``apply`` wrote, a space's ``hold_action``
applied holds, and the gripper number maps out of its range and back.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.core.env import EnvBase
from robosimstudio.core.math import quat_from_axis_angle, quat_multiply
from robosimstudio.solvers.control.action import (
    ActionEEDelta,
    ActionJointPosition,
    ActionOsc,
    make_action_spaces,
)
from tests.fixtures.arrays import host

TIER = "fast"
"""No GPU. Gates every commit."""


class _StubArm:
    """The slice of :class:`ArmIK` the action spaces and the OSC law read."""

    def __init__(self, width=7):
        self.model = SimpleNamespace(body_com=SimpleNamespace(numpy=lambda: np.zeros((1, 3))))
        self.ee_body = 0
        self.ee_offset = np.zeros(3)
        self.ee_offset_rot = np.array([0.0, 0.0, 0.0, 1.0])
        self.home_pose = (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
        self.arm_coords = list(range(width))


def _stub_state(position, quat=(0.0, 0.0, 0.0, 1.0), twist=(0.0,) * 6):
    """A ``State``-shaped stand-in holding one body at a pose."""
    body_q = np.array([[*position, *quat]], dtype=np.float64)
    body_qd = np.array([list(twist)], dtype=np.float64)
    return SimpleNamespace(
        body_q=SimpleNamespace(numpy=lambda: body_q),
        body_qd=SimpleNamespace(numpy=lambda: body_qd),
    )


class TestMakeActionSpacesSkipsOnlyUnsupportedModes(unittest.TestCase):
    def test_a_bad_option_raises_instead_of_dropping_the_mode(self):
        arms = [(_StubArm(), None)]
        with self.assertRaisesRegex(ValueError, "convention"):
            make_action_spaces(arms, control_hz=20.0, options={"ee_delta": {"convention": "algin"}})
        with self.assertRaisesRegex(ValueError, "gripper_range"):
            make_action_spaces(arms, control_hz=20.0, options={"osc_pose": {"gripper_range": (1.0, 0.0)}})

    def test_arms_of_unequal_width_still_skip_the_joint_mode(self):
        spaces = make_action_spaces([(_StubArm(7), None), (_StubArm(6), None)], control_hz=20.0)
        self.assertNotIn("joint_position", spaces)
        self.assertIn("ee_delta", spaces)
        self.assertIn("osc_pose", spaces)

    def test_a_shared_argument_reaches_only_the_modes_that_take_it(self):
        spaces = make_action_spaces([(_StubArm(), None)], control_hz=20.0, gripper_range=(-1.0, 1.0))
        self.assertIn("joint_position", spaces)
        self.assertEqual(spaces["ee_delta"].gripper_range, (-1.0, 1.0))
        self.assertEqual(spaces["osc_pose"].gripper_range, (-1.0, 1.0))

    def test_a_shared_argument_no_mode_takes_raises(self):
        with self.assertRaisesRegex(TypeError, "gripper_rnage"):
            make_action_spaces([(_StubArm(), None)], control_hz=20.0, gripper_rnage=(-1.0, 1.0))

    def test_unsupported_mode_is_on_the_facade(self):
        from robosimstudio.solvers import UnsupportedMode
        from robosimstudio.solvers.control import action

        self.assertIs(UnsupportedMode, action.UnsupportedMode)

    def test_no_control_rate_skips_osc(self):
        spaces = make_action_spaces([(_StubArm(), None)], control_hz=0.0)
        self.assertNotIn("osc_pose", spaces)
        self.assertIn("joint_position", spaces)


class TestGripperRangeIsShared(unittest.TestCase):
    def test_one_convention_for_both_cartesian_spaces(self):
        self.assertIs(ActionOsc.grip_command, ActionEEDelta.grip_command)
        self.assertIs(ActionOsc.grip_action, ActionEEDelta.grip_action)

    def test_both_validate_and_map_the_range(self):
        arms = [(_StubArm(), None)]
        for space in (
            ActionEEDelta(arms, gripper_range=(-1.0, 1.0)),
            ActionOsc(arms, control_hz=20.0, gripper_range=(-1.0, 1.0)),
        ):
            self.assertEqual(space.gripper_range, (-1.0, 1.0))
            self.assertAlmostEqual(space.grip_command(0.0), 0.5)
            self.assertAlmostEqual(space.grip_action(1.0), 1.0)
        with self.assertRaises(ValueError):
            ActionEEDelta(arms, gripper_range=(1.0, 1.0))


class _RecordingSpace:
    """An action space that records what was asked of it, and nothing else."""

    per_arm = 3

    def __init__(self, dim: int = 3) -> None:
        self.action_dim = dim
        self.resets = 0
        self.applied: list[list[float]] = []

    def reset(self, state) -> None:
        del state
        self.resets += 1

    def apply(self, action, control, state) -> None:
        del control, state
        self.applied.append(list(action))

    def hold_action(self, state) -> np.ndarray:
        del state
        return np.zeros(self.action_dim, dtype=np.float32)


class TestControllerSet(unittest.TestCase):
    """An arm has several ways to be commanded, and a command names one.

    The behaviour under test is the one the reference runtime's own controller
    swap documents: a controller carries a *setpoint*, so handing the arm to one
    that still holds a stale setpoint drags the arm back there on the first step.
    Selecting a mode therefore re-seeds it.
    """

    def _env(self, **kwargs):
        pipeline = SimpleNamespace(
            model=SimpleNamespace(particle_count=0, body_count=1),
            state=_stub_state(np.zeros(3)),
            control=None,
            sim_time=0.0,
            step_count=0,
            step=lambda: None,
        )
        return EnvBase(pipeline, **kwargs)

    def test_a_single_space_still_works(self):
        """The old one-space construction is the one-entry case of the new one."""
        space = _RecordingSpace()
        env = self._env(action_space=space)
        self.assertIs(env.action_space, space)
        self.assertEqual(env.action_dim, 3)

    def test_the_declared_mode_starts_active(self):
        spaces = {"joint_position": _RecordingSpace(), "osc_pose": _RecordingSpace(7)}
        env = self._env(action_space=spaces, control_mode="osc_pose")
        self.assertEqual(env.control_mode, "osc_pose")
        self.assertEqual(env.action_dim, 7)

    def test_an_unsupported_declared_mode_raises_at_construction(self):
        with self.assertRaises(ValueError):
            self._env(action_space={"joint_position": _RecordingSpace()}, control_mode="osc_pose")

    def test_naming_a_mode_switches_and_reseeds_it(self):
        """The load-bearing behaviour: a handover re-seeds the incoming setpoint."""
        joint, osc = _RecordingSpace(), _RecordingSpace(7)
        env = self._env(action_space={"joint_position": joint, "osc_pose": osc}, control_mode="joint_position")
        before = osc.resets
        env.apply_action([0.0] * 7, mode="osc_pose")
        self.assertEqual(env.control_mode, "osc_pose")
        self.assertEqual(osc.resets, before + 1)
        self.assertEqual(len(osc.applied), 1)
        self.assertEqual(joint.applied, [])

    def test_renaming_the_active_mode_does_not_reseed(self):
        """So a caller may tag every command without paying for it."""
        osc = _RecordingSpace(7)
        env = self._env(action_space={"osc_pose": osc}, control_mode="osc_pose")
        seeded = osc.resets
        for _ in range(3):
            env.apply_action([0.0] * 7, mode="osc_pose")
        self.assertEqual(osc.resets, seeded)
        self.assertEqual(len(osc.applied), 3)

    def test_an_unknown_mode_names_what_there_is(self):
        env = self._env(action_space={"joint_position": _RecordingSpace()})
        with self.assertRaises(KeyError) as caught:
            env.use_control("osc_pose")
        self.assertIn("joint_position", str(caught.exception))

    def test_an_omitted_mode_keeps_the_active_one(self):
        joint, osc = _RecordingSpace(), _RecordingSpace(7)
        env = self._env(action_space={"joint_position": joint, "osc_pose": osc}, control_mode="joint_position")
        env.apply_action([0.0, 0.0, 0.0])
        self.assertEqual(env.control_mode, "joint_position")
        self.assertEqual(len(joint.applied), 1)

    def test_idle_action_may_be_asked_of_a_mode_without_switching(self):
        """Asking what a hold looks like elsewhere is not asking to hand over."""
        joint, osc = _RecordingSpace(), _RecordingSpace(7)
        env = self._env(action_space={"joint_position": joint, "osc_pose": osc}, control_mode="joint_position")
        env.idle_action("osc_pose")
        self.assertEqual(env.control_mode, "joint_position")


class _DeltaArm:
    """The three things ``ActionEEDelta`` asks of an ``ArmIK``, recorded."""

    def __init__(self, pos, quat) -> None:
        self.home_pose = (np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64))
        self.target = None

    def ee_pose(self, state):
        return self.home_pose

    def set_target(self, pos, quat) -> None:
        self.target = (np.asarray(pos), np.asarray(quat))

    def solve_into(self, joint_q, joint_target_q) -> None:
        pass


class TestEEDeltaToolConvention(unittest.TestCase):
    """``"tool"`` is ManiSkill's ``ee`` frame: ``prev_pose * delta``, translating along the pre-step axes."""

    def test_a_combined_move_translates_along_the_pre_step_orientation(self):
        from robosimstudio.solvers import ActionEEDelta

        arm = _DeltaArm((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        space = ActionEEDelta([(arm, None)], convention="tool")
        # 10 cm along the tool's x while turning a quarter about its z.
        space.apply(
            [0.1, 0.0, 0.0, 0.0, 0.0, np.pi / 2, 0.0],
            SimpleNamespace(joint_target_q=None),
            SimpleNamespace(joint_q=None),
        )
        pos, quat = arm.target
        np.testing.assert_allclose(pos, [0.1, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(np.abs(quat), np.abs(quat_from_axis_angle([0, 0, 1], np.pi / 2)), atol=1e-9)


class _JointArm:
    """``set_joint_targets`` as ``ArmIK`` writes it: by coordinate, or by DOF where the two differ."""

    def __init__(self, coords, dofs) -> None:
        self.arm_coords, self.arm_dofs = list(coords), list(dofs)

    def set_joint_targets(self, values, joint_target_q, *, by_dof: bool = False) -> None:
        joint_target_q.numpy()[self.arm_dofs if by_dof else self.arm_coords] = values


class _Hand:
    """A gripper as the action spaces see one: a command, which end shuts it, and a write."""

    def __init__(self, closed_at_high: bool = True) -> None:
        self.command = 0.0
        self.action_closed_at_high = closed_at_high
        self.writes = 0

    def apply(self, joint_target_q) -> None:
        del joint_target_q
        self.writes += 1


def _two_arms():
    """Two three-joint arms whose coordinates and DOFs differ, as a model with free joints has them."""
    return [(_JointArm((0, 1, 2), (6, 7, 8)), _Hand()), (_JointArm((3, 4, 5), (9, 10, 11)), _Hand())]


class TestTheJointSpaceRoundTrips(unittest.TestCase):
    def test_target_action_reads_back_what_apply_wrote(self) -> None:
        """Whichever indexing ``by_dof`` names, the read is the write's: a teleop recording replays as it was driven."""
        action = np.array([0.1, -0.2, 0.3, 0.75, 0.4, 0.5, -0.6, 0.25], dtype=np.float32)
        for by_dof in (False, True):
            with self.subTest(by_dof=by_dof):
                space = ActionJointPosition(_two_arms(), by_dof=by_dof)
                control = SimpleNamespace(joint_target_q=host(np.zeros(12, dtype=np.float32)))
                space.apply(action, control, None)
                np.testing.assert_array_equal(action, space.target_action(control))
                untouched = control.joint_target_q.numpy()[list(range(6, 12)) if not by_dof else list(range(6))]
                np.testing.assert_array_equal(np.zeros(6), untouched, "only the indexing named was written")

    def test_applying_the_hold_action_targets_where_the_arms_are(self) -> None:
        space = ActionJointPosition(_two_arms())
        space.arms[1][1].command = 0.4
        live = np.array([0.3, -0.1, 0.2, 0.5, 0.6, -0.7], dtype=np.float32)
        state = SimpleNamespace(joint_q=host(np.concatenate([live, np.zeros(6, dtype=np.float32)])))
        control = SimpleNamespace(joint_target_q=host(np.full(12, 9.0, dtype=np.float32)))
        space.apply(space.hold_action(state), control, state)
        np.testing.assert_array_equal(live, control.joint_target_q.numpy()[:6])
        self.assertEqual(
            [0.0, 0.4], [round(hand.command, 6) for _, hand in space.arms], "and the hands stay as they were"
        )

    def test_an_action_of_the_wrong_width_is_refused_by_its_count(self) -> None:
        space = ActionJointPosition(_two_arms())
        with self.assertRaisesRegex(ValueError, "action has 7 entries, expected 8"):
            space.apply(np.zeros(7), SimpleNamespace(joint_target_q=host(np.zeros(12))), None)

    def test_a_space_with_no_arm_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one arm"):
            ActionJointPosition([])


class TestTheGripperNumber(unittest.TestCase):
    """``grip_command`` maps a policy's number onto 0 open .. 1 closed, and ``grip_action`` maps it back."""

    def test_grip_action_inverts_grip_command_for_either_hand_and_range(self) -> None:
        for gripper_range in ((0.0, 1.0), (-1.0, 1.0), (0.2, 0.9)):
            for closed_at_high in (True, False):
                with self.subTest(gripper_range=gripper_range, closed_at_high=closed_at_high):
                    space = ActionEEDelta([(_DeltaArm((0, 0, 0), (0, 0, 0, 1)), None)], gripper_range=gripper_range)
                    hand = _Hand(closed_at_high)
                    for value in np.linspace(*gripper_range, 7):
                        command = space.grip_command(value, hand)
                        self.assertAlmostEqual(value, space.grip_action(command, hand), places=6)
                    self.assertEqual(1.0 if closed_at_high else 0.0, space.grip_command(gripper_range[1], hand))

    def test_a_number_outside_the_range_is_clipped_to_its_end(self) -> None:
        space = ActionOsc([(_StubArm(), None)], control_hz=20.0, gripper_range=(-1.0, 1.0))
        self.assertEqual(0.0, space.grip_command(-3.0))
        self.assertEqual(1.0, space.grip_command(5.0))


class TestTheDeltaSpaceHolds(unittest.TestCase):
    def _space(self, **kwargs):
        arm = _DeltaArm((0.3, 0.1, 0.4), quat_from_axis_angle((1.0, 0.0, 0.0), 0.3))
        return ActionEEDelta([(arm, _Hand())], **kwargs), arm

    def test_its_hold_action_leaves_the_setpoint_and_the_hand_where_they_stand(self) -> None:
        for gripper_delta in (False, True):
            with self.subTest(gripper_delta=gripper_delta):
                space, arm = self._space(gripper_delta=gripper_delta, gripper_range=(-1.0, 1.0))
                hand = space.arms[0][1]
                hand.command = 0.3
                before = space.setpoints[0]
                state = SimpleNamespace(joint_q=None, body_q=None)
                space.apply(space.hold_action(state), SimpleNamespace(joint_target_q=None), state)
                np.testing.assert_allclose(before[0], arm.target[0], atol=1e-12)
                np.testing.assert_allclose(before[1], arm.target[1], atol=1e-12)
                self.assertAlmostEqual(0.3, hand.command, places=6)

    def test_align_moves_and_turns_in_the_frame_bodys_axes(self) -> None:
        """A frame body turned a quarter about z: its +x is the world's +y, and a turn about its z is about the world's."""
        space, arm = self._space(frame_bodies=(0,))
        frame = quat_from_axis_angle((0.0, 0.0, 1.0), np.pi / 2)
        state = SimpleNamespace(joint_q=None, body_q=host(np.array([[0.0, 0.0, 0.0, *frame]])))
        start_pos, start_quat = space.setpoints[0]
        space.apply([0.1, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0], SimpleNamespace(joint_target_q=None), state)
        np.testing.assert_allclose(arm.target[0], start_pos + np.array([0.0, 0.1, 0.0]), atol=1e-12)
        want = quat_multiply(quat_from_axis_angle((0.0, 0.0, 1.0), 0.2), start_quat)
        np.testing.assert_allclose(np.abs(arm.target[1]), np.abs(want), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
