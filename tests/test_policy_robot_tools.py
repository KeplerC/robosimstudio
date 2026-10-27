# SPDX-License-Identifier: Apache-2.0
"""``robot.*`` tools: the home posture, forward kinematics on the bus, fingertip reach
per env, and arm names and roles refused rather than guessed."""

from __future__ import annotations

import types
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from robosimstudio_policy_agent import robot_tools
from robosimstudio_policy_agent.grasp_tools import GraspTools
from robosimstudio_policy_agent.refusal import ToolRefusal
from robosimstudio_policy_agent.robot_tools import RobotTools

TIER = "fast"
"""Stub envs and a counting bus: no GPU, no scene."""


HOME = (np.array([0.4, 0.0, 0.35]), np.array([1.0, 0.0, 0.0, 0.0]))


class _Arm:
    """An arm whose live TCP is low (just grasped) and whose home is high."""

    home_pose = HOME

    def ee_pose(self, state):
        raise AssertionError("the live pose is not the home pose")


class _Bus:
    def __init__(self) -> None:
        self.inside = False
        self.calls = 0

    def call(self, fn, *args, **kwargs):
        self.calls += 1
        self.inside = True
        try:
            return fn(*args, **kwargs)
        finally:
            self.inside = False


def _tools(bus=None) -> RobotTools:
    hand = SimpleNamespace(
        tip_reach=0.02,
        housing_rise=0.05,
        pad_thickness=0.01,
        tcp_offset=(0.0, 0.0, 0.1),
        tcp_offset_rot=(0.0, 0.0, 0.0, 1.0),
    )
    handle = SimpleNamespace(arm_coords=[0, 1], arm_dofs=[0, 1], home_q=[0.0, 0.0])
    model = SimpleNamespace(
        joint_limit_lower=SimpleNamespace(numpy=lambda: np.array([-1.0, -1.0])),
        joint_limit_upper=SimpleNamespace(numpy=lambda: np.array([1.0, 1.0])),
    )
    env = SimpleNamespace(
        handles=SimpleNamespace(arms={"arm": handle}, surface_z=0.0),
        scene=SimpleNamespace(embodiment=SimpleNamespace(hands={"arm": hand})),
        model=model,
        state=SimpleNamespace(joint_q=SimpleNamespace(numpy=lambda: np.array([0.1, 0.2]))),
        arm=lambda name="": _Arm(),
    )
    tools = RobotTools(env, bus or _Bus())
    tools._arm = lambda arm_id, role: "arm"
    return tools


class HomePoseTest(unittest.TestCase):
    def setUp(self) -> None:
        for name in ("arm_id_of", "arm_role", "arm_order"):
            patcher = mock.patch.object(robot_tools, name, return_value=0 if name == "arm_id_of" else ())
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_describe_workspace_transport_is_the_home_height(self) -> None:
        out = _tools().describe_workspace()
        self.assertAlmostEqual(out["transport_z"], 0.35)
        self.assertEqual(out["transport_z_source"], "home_tcp")

    def test_describe_arm_reports_the_home_pose(self) -> None:
        out = _tools().describe_arm()
        self.assertAlmostEqual(out["home_pose"]["position"]["z"], 0.35)


class ForwardKinematicsOnBusTest(unittest.TestCase):
    def test_fk_runs_inside_bus_call(self) -> None:
        bus = _Bus()
        tools = _tools(bus)

        class _FkArm(_Arm):
            def fk_ee(self, q):
                assert bus.inside, "fk_ee ran off the bus"
                return np.array([float(q[0]), 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])

        tools.env.arm = lambda name="": _FkArm()
        out = tools.forward_kinematics([[0.5, 0.0]])
        self.assertEqual(bus.calls, 1)
        self.assertAlmostEqual(out["poses"][0]["position"]["x"], 0.5)
        self.assertEqual(out["joint_config"], [0.1, 0.2])


def _env(size: float):
    """A stub env whose one finger collider has a corner *size* past the TCP along +z."""
    arm = types.SimpleNamespace(
        finger_groups=(((1, np.array([0])),),),
        ee_body=0,
        ee_offset=np.zeros(3),
        ee_offset_rot=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    builder = types.SimpleNamespace(shape_flags=[0xFFFF], size=size)
    return types.SimpleNamespace(
        handles=types.SimpleNamespace(arms={"arm0": arm}),
        scene=types.SimpleNamespace(builder=builder),
    )


class FingertipReachCache(unittest.TestCase):
    def test_each_env_reads_its_own_corners(self):
        body_q = np.tile(np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64), (2, 1))
        try:
            import robosimstudio.solvers.grasp.obb as obb
        except Exception as error:  # pragma: no cover - environment without the solver stack
            self.skipTest(f"grasp solvers unavailable: {error}")
        with mock.patch.object(obb, "_shape_corners_local", lambda builder, s: [(0.0, 0.0, builder.size)]):
            a, b = _env(0.01), _env(0.05)
            # Force the collision the id-keyed cache could not survive.
            with mock.patch.object(robot_tools, "id", create=True, side_effect=lambda _obj: 1):
                reach_a = robot_tools.fingertip_reach(a, "arm0", body_q=body_q)
                reach_b = robot_tools.fingertip_reach(b, "arm0", body_q=body_q)
        self.assertAlmostEqual(abs(reach_a), 0.01)
        self.assertAlmostEqual(abs(reach_b), 0.05)


def _handles(**arms):
    return SimpleNamespace(
        objects={"cube": SimpleNamespace(body=3)},
        statics={"table": SimpleNamespace(body=-1)},
        arms={"left": None, "right": None},
        work_arm=arms.get("work", "left"),
        support_arm=arms.get("support", "right"),
        entity=lambda name: {"cube": SimpleNamespace(body=3), "table": SimpleNamespace(body=-1)}[name],
    )


class TestGraspNamesAndRoles(unittest.TestCase):
    def _tools(self, cls):
        tools = cls.__new__(cls)
        tools.env = SimpleNamespace(handles=_handles())
        return tools

    def test_an_unknown_object_is_a_refusal_naming_the_choices(self):
        tools = self._tools(GraspTools)
        tools.env.handles.entity = lambda name: (_ for _ in ()).throw(KeyError(name))
        with self.assertRaises(ToolRefusal) as caught:
            tools.grasp_candidates("cueb", role="work")
        self.assertIn("cube", str(caught.exception))

    def test_a_static_reaches_the_bodiless_refusal(self):
        with self.assertRaisesRegex(ToolRefusal, "bodiless static"):
            self._tools(GraspTools).grasp_candidates("table", role="work")

    def test_a_mistyped_role_is_refused_not_mapped_to_the_support_arm(self):
        for cls in (GraspTools, RobotTools):
            tools = self._tools(cls)
            self.assertEqual(tools._arm(0, "support"), "right")
            for role in ("Work", "left", "worker"):
                with self.assertRaises(ValueError):
                    tools._arm(0, role)


if __name__ == "__main__":
    unittest.main()
