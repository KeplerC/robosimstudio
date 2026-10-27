# SPDX-License-Identifier: Apache-2.0
"""The gauntlet: a document read, bound and started the way the loader and the env
do, its thresholds linted, and its reward measure found past the gate words."""

import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from robosimstudio_bench.goals.combinators import AllOf, Leaf
from robosimstudio_bench.taskdoc.ast import Condition, Instance, Ref, TaskDoc

TIER = "fast"
"""Documents in a scratch directory and stub envs: no scene, no GPU."""


# The package re-exports `run` as `gauntlet`, which shadows the module.
gauntlet = importlib.import_module("robosimstudio_bench.goals.gauntlet")


class TestGauntletStartReadsTheActiveSet(unittest.TestCase):
    def test_a_parked_objects_init_clause_is_not_checked(self):
        doc = TaskDoc(
            name="t",
            objects=(Instance("a", "block"), Instance("b", "block")),
            init=(Condition("on-table", (Ref("a"),)), Condition("on-table", (Ref("b"),))),
            goal=Condition("lifted", (Ref("a"),)),
        )

        def fake_bind(c, doc, asset_of, task=""):
            ok = c.head == "on-table" and c.args[0].base == "a"
            return Leaf(fn=lambda ctx: (ok, float(ok), {}))

        env = SimpleNamespace(
            reset=lambda: None,
            make_context=object,
            episode=SimpleNamespace(active=frozenset({"a"})),
        )
        with (
            mock.patch.object(gauntlet, "bind", fake_bind),
            mock.patch.object(gauntlet, "Snapshot", lambda ctx: SimpleNamespace(ctx=ctx)),
        ):
            stage = gauntlet._start(doc, env, "t")
        self.assertEqual(stage.status, "pass", stage.findings)


class TestGauntletReadsTheSuiteAndProfile(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        (root / "demo.suite").write_text(
            "(define (suite demo)\n  (:staging (:profile fake@v1))\n  (:fixtures bin - box))\n"
        )
        self.task = root / "one.task"
        self.task.write_text(
            "(define (problem one)\n  (:suite demo)\n  (:objects tool - hammer)\n  (:goal (In tool bin)))\n"
        )

    def tearDown(self):
        self.dir.cleanup()

    def test_read_merges_the_sibling_suite(self):
        stage, doc = gauntlet._read(self.task)
        self.assertEqual(stage.status, "pass", stage.findings)
        self.assertIn("bin", {f.name for f in doc.objects})
        self.assertEqual(str(dict(doc.staging).get("profile")), "fake@v1")

    def test_bind_falls_back_to_the_documents_profile(self):
        _, doc = gauntlet._read(self.task)
        with mock.patch(
            "robosimstudio_bench.taskbuild.staging.get_profile",
            side_effect=lambda name: SimpleNamespace(bindings=None) if name == "fake@v1" else None,
        ) as get:
            stage, _ = gauntlet._bind(doc, "")
        get.assert_called_once_with("fake@v1")
        self.assertEqual(stage.data.get("profile"), "fake@v1")


class TestVacuousThreshold(unittest.TestCase):
    @staticmethod
    def _flags(op, value):
        return [f.kind for f in gauntlet._vacuous([Condition("distance", (Ref("a"),), op=op, value=value)])]

    def test_strictly_positive_is_not_vacuous(self):
        self.assertEqual(self._flags(">", 0.0), [])

    def test_the_vacuous_ones_are_still_flagged(self):
        self.assertEqual(self._flags(">=", 0.0), ["vacuous-threshold"])
        self.assertEqual(self._flags(">", -0.1), ["vacuous-threshold"])


class TestOneSetOfGateWords(unittest.TestCase):
    def test_the_gauntlet_finds_the_measure_past_the_gate(self):
        section = {"progress": [[":gate-at", 0.1], [":gated_by", ["intact", "film"]], ["depth", "plug"]]}
        self.assertEqual(gauntlet._reward_measure(section), ("depth", ("plug",)))

    def test_the_gauntlet_leaves_are_the_goals(self):
        a, b = Leaf(fn=lambda ctx: (True, 1.0, {})), Leaf(fn=lambda ctx: (True, 1.0, {}))
        self.assertEqual(gauntlet._leaves(AllOf(of=(a, b))), (a, b))
        self.assertEqual(gauntlet._leaves(None), ())


if __name__ == "__main__":
    unittest.main()
