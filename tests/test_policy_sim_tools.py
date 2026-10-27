# SPDX-License-Identifier: Apache-2.0
"""``sim.*`` tools: the collision world keeps the statics, ``check_grasp`` refuses an
unknown arm, the grasp heading is square to the long axis, and a question asked of
the scene takes its refs as a graph writes them and comes back refused rather than
raised."""

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


class TestAGraphCanAskTheSceneAQuestion(unittest.TestCase):
    """``sim.query`` and ``sim.list_relations`` as a workflow reaches them: keywords only, refusals as answers."""

    def test_refs_are_read_however_an_author_writes_them(self):
        self.assertEqual(SimTools._refs(None), ())
        self.assertEqual(SimTools._refs("wrench_1.ring, wrench_hook_1.shank"), ("wrench_1.ring", "wrench_hook_1.shank"))
        self.assertEqual(SimTools._refs(["a", ["b, c"], " d "]), ("a", "b", "c", "d"))
        self.assertEqual(SimTools._refs(7), ("7",))

    def test_the_tool_spellings_pass_every_ref_through(self):
        tools = SimTools.__new__(SimTools)
        asked: list[tuple] = []
        tools.query = lambda atom, *refs: asked.append((atom, *refs)) or {}
        tools.list_relations = lambda *refs: asked.append(refs) or {}
        tools.query_tool("covers", refs="towel, table")
        tools.list_relations_tool(refs=["towel", "table"])
        self.assertEqual(asked, [("covers", "towel", "table"), ("towel", "table")])

    def test_a_question_nothing_answers_is_returned_as_a_refusal_not_raised(self):
        tools = SimTools.__new__(SimTools)
        tools.env = SimpleNamespace(make_context=lambda: SimpleNamespace(entities={}), doc=None)
        tools.bus = SimpleNamespace(call=lambda fn, *a, **k: fn(*a, **k))
        out = tools.query("no_such_atom_anywhere", "cube")
        self.assertEqual(list(out), ["refused"])
        self.assertIn("no_such_atom_anywhere", out["refused"])

    def test_a_failure_reading_the_scene_is_a_refusal_too(self):
        def gone():
            raise KeyError("the scene was reset under the question")

        tools = SimTools.__new__(SimTools)
        tools.env = SimpleNamespace(make_context=gone, doc=None)
        tools.bus = SimpleNamespace(call=lambda fn, *a, **k: fn(*a, **k))
        self.assertIn("reset under the question", tools.list_relations("cube")["refused"])


if __name__ == "__main__":
    unittest.main()
