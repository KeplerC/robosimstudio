# SPDX-License-Identifier: Apache-2.0
"""The RGBS bench suite in ``gallery/rgbs`` loads, lowers and keeps the task-directory contract, on no GPU.

The suite is a gallery project, not a shipped task: it loads only under the opt-in root
(``ROBOSIMSTUDIO_TASK_ROOTS=$PWD/gallery/rgbs/tasks``), so nothing else in the tree would notice
a document that stopped parsing, a routine it names that moved, or a module the library renamed
under it. This loads that root in a fresh interpreter, so the root reaches no other test, and
holds three things:

* every variant the gallery filmed lowers, on the one YAM;
* a variant run by a scripted routine names one that its task directory really holds, since
  main has no routine runner and ``routine.py:SolutionRoutine`` resolves it by that name;
* nothing in the suite holds a task object by anything but finger contact: no particle
  dragger, no SuperDex pin, no ``sim.attach``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from robosimstudio_bench.taskbuild.contract import layout_problems

TIER = "fast"

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "gallery" / "rgbs" / "tasks"
SUITE = TASKS / "rgbs"
FILMED = ("handle", "panel", "seam", "strip", "thread", "wipe")
"""The variants the gallery has clips of."""

LOWER = """
import json
from robosimstudio_bench.taskbuild.discovery import task_module
from robosimstudio_bench.taskbuild.documents import document_of
from robosimstudio_bench.taskbuild.registry import get_task, list_tasks
from robosimstudio_bench.taskdoc import read_problem

out = {}
for name in sorted(n for n in list_tasks() if n.startswith("rgbs/")):
    task = get_task(name)
    params = dict(read_problem(document_of(name)).policy_params or {})
    routine = params.get("routine", "")
    found = False
    if routine:
        module, _, cls = str(routine).partition(":")
        found = hasattr(task_module("rgbs", module), cls)
    out[name] = {
        "embodiment": task.embodiment,
        "graphs": [label for label, _ in task.graphs],
        "routine": routine,
        "routine_found": found,
    }
print("RGBS" + json.dumps(out))
"""


class TestTheSuiteLowersUnderItsOptInRoot(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        env = {
            **os.environ,
            "ROBOSIMSTUDIO_TASK_ROOTS": str(TASKS),
            "CUDA_VISIBLE_DEVICES": "",
            "ROBOSIMSTUDIO_DEVICE": "",
        }
        done = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(LOWER)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        line = next((row for row in done.stdout.splitlines() if row.startswith("RGBS")), "")
        if not line:
            raise AssertionError(f"the suite did not lower:\n{done.stdout[-2000:]}\n{done.stderr[-4000:]}")
        cls.variants = json.loads(line[len("RGBS") :])

    def test_every_filmed_variant_lowers_on_the_yam(self) -> None:
        for variant in FILMED:
            with self.subTest(variant=variant):
                self.assertIn(f"rgbs/{variant}", self.variants)
                self.assertEqual("yam", self.variants[f"rgbs/{variant}"]["embodiment"])

    def test_every_variant_has_a_policy_it_can_find(self) -> None:
        for name, facts in self.variants.items():
            if name == "rgbs/decant":
                continue  # the two-stage variant ships no policy, and never did
            with self.subTest(variant=name):
                self.assertTrue(facts["routine"] or facts["graphs"], f"{name} names no routine and no graph")
                if facts["routine"]:
                    self.assertTrue(facts["routine_found"], f"{name} names {facts['routine']}, which is not there")


class TestTheSuiteHoldsThingsOnlyByContact(unittest.TestCase):
    def test_the_task_directory_keeps_the_contract(self) -> None:
        self.assertEqual([], layout_problems(SUITE))

    def test_nothing_latches_pins_or_attaches(self) -> None:
        # Calls, not words: the docstrings say why each of these is gone.
        forbidden = re.compile(
            r"ParticleDragger\(|\.pin_nodes\(|\.set_pin_targets\(|\.attach\(|\"sim\.attach\"|add_attachment_particle_body\("
        )
        found = [
            f"{path.relative_to(ROOT)}:{n}"
            for path in sorted(SUITE.rglob("*"))
            if path.suffix in (".py", ".json")
            for n, line in enumerate(path.read_text().splitlines(), 1)
            if forbidden.search(line) and not line.lstrip().startswith(("#", ";"))
        ]
        self.assertEqual([], found)


if __name__ == "__main__":
    unittest.main()
