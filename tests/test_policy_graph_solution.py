# SPDX-License-Identifier: Apache-2.0
"""The single-world graph policy: what its close puts back."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

TIER = "fast"
"""A stub env and a scratch graph directory: no GPU, no scene."""


class GraphPolicyCloseTest(unittest.TestCase):
    def _policy(self, env, graph: str):
        from robosimstudio_policy_agent import graph_solution as gs

        with (
            mock.patch.object(gs.SolutionGraph, "_new_connector", lambda self: None),
            mock.patch.object(gs, "solution_param", lambda *a, **k: 0.0, create=True),
        ):
            return gs.SolutionGraph(env, graph=graph)

    @staticmethod
    def _env():
        task = SimpleNamespace(tools=None, graph_path=lambda g: {}[g], full_name="x")
        return SimpleNamespace(renderer=object(), obs_render_enabled=True, task=task)

    def test_on_demand_render_flag_restored(self) -> None:
        root = Path(__file__).resolve().parents[1]
        env = self._env()
        policy = self._policy(env, str(root / "examples/07_policies/pick_place_camera"))
        self.assertTrue(policy._on_demand_observations)
        self.assertFalse(env.obs_render_enabled)
        policy.close()
        self.assertTrue(env.obs_render_enabled)

    def test_staged_graph_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph = Path(tmp) / "g"
            graph.mkdir()
            (graph / "workflow.json").write_text(
                json.dumps({"nodes": {"r": {"type": "router", "script": "scripts/route_perception.py"}}})
            )
            policy = self._policy(self._env(), str(graph))
            staged = Path(policy.graph_path)
            self.assertNotEqual(staged, graph)
            self.assertTrue(staged.exists())
            policy.close()
            self.assertFalse(staged.exists())
            self.assertTrue(graph.exists())


if __name__ == "__main__":
    unittest.main()
