# SPDX-License-Identifier: Apache-2.0
"""Staging a graph: its own tree left alone, and provenance read from a relative directory."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from robosimstudio_policy_agent.graphs import NODES, outside_links, stage_graph

TIER = "fast"
"""Files in a scratch directory: no GPU, no scene."""


class _Scratch(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="rss-fix-policy-"))
        self.addCleanup(shutil.rmtree, self.root, True)


class TestStagingANestedScript(_Scratch):
    def test_the_graph_source_tree_is_not_written_through(self):
        graph = self.root / "graph"
        (graph / "scripts").mkdir(parents=True)
        (graph / "scripts" / "own.py").write_text("# own\n")
        node = sorted(NODES.glob("*.py"))[0].name
        workflow = {
            "nodes": {
                "a": {"type": "script", "script": "scripts/own.py"},
                "b": {"type": "script", "script": f"scripts/sub/{node}"},
            }
        }
        (graph / "workflow.json").write_text(json.dumps(workflow))
        before = sorted(str(p.relative_to(graph)) for p in graph.rglob("*"))
        staged = Path(stage_graph(graph, registries=[]))
        self.addCleanup(shutil.rmtree, staged, True)
        self.assertEqual(sorted(str(p.relative_to(graph)) for p in graph.rglob("*")), before)
        self.assertTrue((staged / "scripts" / "own.py").is_file())
        self.assertTrue((staged / "scripts" / "sub" / node).is_file())


class TestProvenanceOfARelativeGraph(_Scratch):
    def test_the_graphs_own_files_are_not_outside_it(self):
        graph = self.root / "g"
        graph.mkdir()
        (graph / "workflow.json").write_text("{}")
        staged = self.root / "st"
        staged.mkdir()
        os.symlink((graph / "workflow.json").resolve(), staged / "workflow.json")
        cwd = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, cwd)
        self.assertEqual(outside_links("st", "g"), [])


if __name__ == "__main__":
    unittest.main()
