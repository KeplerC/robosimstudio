# SPDX-License-Identifier: Apache-2.0
"""``sim.*`` tools: the collision world keeps the statics, ``check_grasp`` refuses an
unknown arm, and the grasp heading is square to the long axis."""

from __future__ import annotations

import math
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from robosimstudio_policy_agent import sim_tools
from robosimstudio_policy_agent.refusal import ToolRefusal
from robosimstudio_policy_agent.sim_tools import SimTools, _grasp_heading_deg

TIER = "fast"
"""Stub envs and pure functions: no GPU, no scene."""


class CollisionWorldExclusion(unittest.TestCase):
    def test_bodiless_handle_does_not_drop_the_statics(self):
        def shape(index, body):
            return types.SimpleNamespace(
                index=index,
                body=body,
                local=np.eye(4),
                vertices=np.zeros((3, 3)),
                faces=np.array([[0, 1, 2]]),
            )

        shapes = [shape(0, -1), shape(1, -1), shape(2, 0)]
        tools = sim_tools.SimTools.__new__(sim_tools.SimTools)
        tools.env = types.SimpleNamespace(
            scene=types.SimpleNamespace(embodiment=None, index_model=types.SimpleNamespace(shape_label=None)),
            handles=types.SimpleNamespace(
                objects={
                    "cube": types.SimpleNamespace(category="cube", body=0),
                    "marker": types.SimpleNamespace(category="cube_marker", body=-1),
                }
            ),
        )
        tools.bus = types.SimpleNamespace(
            snapshot=types.SimpleNamespace(body_q=np.array([[0, 0, 0, 1, 0, 0, 0]], dtype=np.float64))
        )
        try:
            import newton  # noqa: F401
        except Exception as error:  # pragma: no cover
            self.skipTest(f"newton unavailable: {error}")
        with mock.patch.object(sim_tools, "shape_meshes", lambda *a, **k: shapes):
            world = tools.get_collision_world(exclude_categories=["cube"])
        self.assertEqual(world["mesh_names"], ["shape_0_collider", "shape_1_collider"])


def _tool(arms: dict, work_arm: str) -> SimTools:
    # __new__ skips __init__'s home-pose probe; check_grasp touches only these.
    tool = SimTools.__new__(SimTools)
    tool.env = SimpleNamespace(handles=SimpleNamespace(arms=arms, work_arm=work_arm))
    tool.bus = SimpleNamespace(call=lambda fn: fn())
    return tool


class TestCheckGraspRefusal(unittest.TestCase):
    def test_unknown_arm_refuses_and_names_the_vocabulary(self) -> None:
        tool = _tool({"left": object(), "right": object()}, "left")
        with self.assertRaises(ToolRefusal) as ctx:
            tool.check_grasp("wrench", arm="work")
        message = str(ctx.exception)
        self.assertIn("'work'", message)
        self.assertIn("left", message)
        self.assertIn("describe_arm", message)

    def test_known_arm_still_answers_without_probes(self) -> None:
        # No contact probe on the env: the documented held=False path.
        tool = _tool({"left": object()}, "left")
        self.assertEqual(
            {"object_name": "wrench", "held": False},
            tool.check_grasp("wrench", arm="left"),
        )

    def test_default_arm_still_answers_without_probes(self) -> None:
        tool = _tool({"left": object()}, "left")
        self.assertFalse(tool.check_grasp("wrench")["held"])


def _perp(a_deg: float, b_deg: float) -> float:
    """Separation of two undirected headings, 0..90 degrees."""
    d = abs(a_deg - b_deg) % 180.0
    return min(d, 180.0 - d)


class GraspHeadingIsSquareToTheLongAxis(unittest.TestCase):
    def test_at_every_staged_yaw(self):
        for yaw_deg in (-175, -90, -5, 0, 5, 44, 90, 175):
            with self.subTest(yaw_deg=yaw_deg):
                self.assertAlmostEqual(_perp(_grasp_heading_deg(math.radians(yaw_deg)), yaw_deg), 90.0, places=9)


if __name__ == "__main__":
    unittest.main()
