# SPDX-License-Identifier: Apache-2.0
"""The action spaces a policy writes into, and the set an environment switches between.

Over stub arms and hand-built states: which modes :func:`make_action_spaces`
builds and which errors it surfaces, the one gripper convention the two
Cartesian spaces share, how a command naming a mode re-seeds it, and the
``"tool"`` convention of ``ee_delta``.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.core.env import EnvBase
from robosimstudio.core.math import quat_from_axis_angle
from robosimstudio.solvers.control.action import ActionEEDelta, ActionOsc, make_action_spaces

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


if __name__ == "__main__":
    unittest.main()
