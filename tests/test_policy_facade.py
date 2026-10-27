# SPDX-License-Identifier: Apache-2.0
"""``GapEnvFacade``: the frame it builds passes the privilege contract, a step observes
and reads back once, publishes this step's render, and a reset forgets the hold."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

TIER = "fast"
"""A stub env with one arm and one camera: no GPU, no scene."""


class _Bus:
    def __init__(self) -> None:
        self.calls = 0
        self.snapshot = None

    def call(self, fn, *a, **kw):
        self.calls += 1
        return fn(*a, **kw)

    def publish(self, snapshot) -> None:
        self.snapshot = snapshot


def _facade():
    """A real GapEnvFacade over a stub env with one arm and one camera, counting readbacks."""
    from robosimstudio_policy_agent.facade import GapEnvFacade

    counts = {"observe": 0, "body_q": 0, "embodiment": 0, "step": 0}
    joints = np.zeros(7)

    def observe():
        counts["observe"] += 1
        return {
            "proprio": {"arm/joint_q": joints.copy(), "arm/gripper": 0.0, "arm/ee_pose": np.r_[0, 0, 0, 0, 0, 0, 1.0]},
            "cameras": {"overhead": {"rgb": np.zeros((2, 2, 3))}},
            "instruction": "go",
        }

    def body_q_numpy():
        counts["body_q"] += 1
        return np.zeros((1, 7), dtype=np.float32)

    def emb_observe(state):
        counts["embodiment"] += 1
        return {"arm/joint_q": joints.copy()}

    def step(action, mode=""):
        counts["step"] += 1
        joints[:] = action[:7]
        return observe(), 0.0, False, False, {}

    space = SimpleNamespace(per_arm=8)
    spec = SimpleNamespace(name="overhead", intrinsics=np.eye(3), pose_on_body=lambda body_q: np.eye(4))
    env = SimpleNamespace(
        max_steps=10,
        obs_render_enabled=True,
        renderer=None,
        cameras=[spec],
        frames={},
        step_count=0,
        sim_time=0.0,
        state=SimpleNamespace(body_q=SimpleNamespace(numpy=body_q_numpy)),
        observe=observe,
        step=step,
        use_control=lambda mode: space,
        idle_action=lambda mode: np.zeros(8, dtype=np.float32),
        controllers={"joint_position": space},
        scene=SimpleNamespace(embodiment=SimpleNamespace(grippers={}, observe=emb_observe)),
        reset=lambda episode_id: ({}, {}),
    )

    class OneArm(GapEnvFacade):
        @property
        def arm_names(self):
            return ["arm"]

        def _slot(self, name):
            return 0

        def _arm(self, arm_id):
            return "arm"

    facade = OneArm(env, _Bus())
    return facade, counts


class TestBridgeContractAcceptsTheFacadeFrame(unittest.TestCase):
    def test_full_observation_passes(self) -> None:
        from robosimstudio_policy_agent.contract import assert_non_privileged

        facade, _ = _facade()
        frame = facade._observation()
        self.assertIn("robot_joint_pos_0", frame)
        assert_non_privileged(frame, cameras=["overhead"])

    def test_format_runtime_obs_passes(self) -> None:
        from robosimstudio_policy_agent.contract import assert_non_privileged
        from robosimstudio_policy_agent.fleet.obs_format import format_runtime_obs

        snapshot = SimpleNamespace(
            obs={"proprio": {"a/joint_q": np.zeros(7), "a/ee_pose": np.r_[0, 0, 0, 0, 0, 0, 1.0]}},
            frames={"wrist": {"rgb": np.zeros((2, 2, 3)), "depth": np.zeros((2, 2))}},
            body_q=None,
        )
        frame = format_runtime_obs(snapshot, arm_count=1)
        assert_non_privileged(frame, cameras=["wrist"])

    def test_still_refuses_ground_truth(self) -> None:
        from robosimstudio_policy_agent.contract import assert_non_privileged

        with self.assertRaises(ValueError):
            assert_non_privileged({"robot_joint_pos_0": [0.0], "object_poses": {}})
        # An undeclared camera name is not a pass, and a camera may not be named for ground truth.
        with self.assertRaises(ValueError):
            assert_non_privileged({"overhead": {"images": {}}})
        with self.assertRaises(ValueError):
            assert_non_privileged({"objects": {"images": {}}}, cameras=["objects"])
        with self.assertRaises(ValueError):
            assert_non_privileged({"overhead": {"body_q": []}}, cameras=["overhead"])

    def test_refusal_names_what_is_allowed(self) -> None:
        from robosimstudio_policy_agent.contract import ALLOWED_PATTERNS, __all__, assert_non_privileged

        self.assertIn("ALLOWED_PATTERNS", __all__)
        with self.assertRaises(ValueError) as caught:
            assert_non_privileged({"overhead": {"images": {}}})
        message = str(caught.exception)
        self.assertIn("cameras=", message)
        for pattern in ALLOWED_PATTERNS:
            self.assertIn(pattern.pattern, message)
        # Ground truth gets no camera hint: it is refused for what it is.
        with self.assertRaises(ValueError) as caught:
            assert_non_privileged({"object_poses": {}})
        self.assertNotIn("cameras=", str(caught.exception))


class TestFacadeStepReadsOnce(unittest.TestCase):
    def test_blocking_move_counts(self) -> None:
        facade, counts = _facade()
        facade.move_to_joints_blocking(np.full(7, 0.5), tolerance=-1.0, max_steps=10)
        self.assertEqual(counts["step"], 10)
        self.assertEqual(counts["observe"], 10)
        self.assertEqual(counts["body_q"], 10)
        self.assertEqual(counts["embodiment"], 0)
        self.assertEqual(facade.bus.calls, 10)

    def test_blocking_move_still_converges(self) -> None:
        facade, _ = _facade()
        self.assertTrue(facade.move_to_joints_blocking(np.full(7, 0.5), tolerance=1e-3, max_steps=5))
        snap = facade.bus.snapshot
        np.testing.assert_allclose(snap.obs["robot_joint_pos_0"][:7], 0.5)


class TestStepPublishesTheFreshFrame(unittest.TestCase):
    def test_camera_block_is_this_steps_render(self) -> None:
        """``env.step`` observes, then renders in ``evaluate`` by rebinding ``env.frames``."""
        facade, _ = _facade()
        env = facade.env
        stale = {"overhead": {"rgb": np.zeros((2, 2, 3))}}
        fresh = {"overhead": {"rgb": np.ones((2, 2, 3))}}
        env.frames = stale
        inner_step = env.step

        def step(action, mode=""):
            obs, *rest = inner_step(action, mode)
            obs["cameras"] = env.frames  # the dict observe handed back
            env.frames = fresh  # evaluate's render
            return (obs, *rest)

        env.step = step
        facade.move_to_joints_blocking(np.full(7, 0.5), tolerance=-1.0, max_steps=1)
        snap = facade.bus.snapshot
        np.testing.assert_array_equal(snap.obs["overhead"]["images"]["rgb"], fresh["overhead"]["rgb"])
        self.assertIs(snap.frames["overhead"], fresh["overhead"])


class TestResetForgetsTheHold(unittest.TestCase):
    def test_hold_after_reset_is_idle(self) -> None:
        facade, _ = _facade()
        facade.apply_joint_action(np.full(7, 1.5), arm_id=0)
        self.assertEqual(float(facade.hold_action()[0]), 1.5)
        facade._gripper_fraction[0] = 0.0
        facade.reset(1)
        self.assertEqual(facade._held, {})
        self.assertEqual(facade._gripper_fraction, {})
        np.testing.assert_array_equal(facade.hold_action()[:7], np.zeros(7))


if __name__ == "__main__":
    unittest.main()
