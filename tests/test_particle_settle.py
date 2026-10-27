# SPDX-License-Identifier: Apache-2.0
"""Particle things share one settle per pipeline through ``ParticleMixin.join_settle``.

A run settles once and every thing on the record snapshots its rest after each
join; the "your own kind" tutorial entities join that settle from
``on_scene_ready`` rather than snapshotting alone, or a cloth that settles after
them leaves them with an unsettled rest to reset to.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

TIER = "fast"
"""Stand-in things and a stand-in pipeline, and two example sources read: no GPU."""


def _thing(name: str = "towel", log: list | None = None, run: tuple[int, int] | None = None):
    """A particle thing; ``run`` is the ``(settle_frames, settle_substeps)`` its spec states.

    And the run it joined the settle with, as ``join_settle`` keeps it, until it joins again.
    """
    from robosimstudio.domains.capabilities import ParticleMixin

    class _Stub(ParticleMixin):
        def __init__(self) -> None:
            self.name = name
            self.settles = 0
            self.snapshots = 0
            if run is not None:
                self.spec = SimpleNamespace(settle_frames=run[0], settle_substeps=run[1])
            self.settle_run = run

        def settle(self, pipeline) -> None:
            del pipeline
            self.settles += 1
            if log is not None:
                log.append(f"settle {self.name}")

        def snapshot_rest(self, pipeline) -> None:
            del pipeline
            self.snapshots += 1
            if log is not None:
                log.append(f"snapshot {self.name}")

    return _Stub()


class TestJoinSettle(unittest.TestCase):
    def test_a_run_settles_once_and_every_thing_snapshots_after_each_join(self) -> None:
        pipeline = SimpleNamespace()
        sheet, block, cable = _thing(), _thing(), _thing()
        sheet.join_settle(pipeline, (10, 4))
        block.join_settle(pipeline, (10, 4))
        cable.join_settle(pipeline)
        self.assertEqual((1, 0, 0), (sheet.settles, block.settles, cable.settles))
        self.assertEqual({(10, 4)}, pipeline.particle_settles["runs"])
        self.assertEqual([sheet, block, cable], pipeline.particle_settles["things"])
        self.assertEqual((3, 2, 1), (sheet.snapshots, block.snapshots, cable.snapshots))

    def test_a_different_run_settles_again(self) -> None:
        pipeline = SimpleNamespace()
        a, b = _thing(), _thing()
        a.join_settle(pipeline, (10, 4))
        b.join_settle(pipeline, (20, 4))
        self.assertEqual((1, 1), (a.settles, b.settles))


ROOT = Path(__file__).resolve().parents[1]


TUTORIALS = (
    ROOT / "examples" / "01_cloth" / "04_your_own_cloth.py",
    ROOT / "examples" / "02_soft" / "02_your_own_soft.py",
)


def _called(path: Path) -> set[str]:
    """The ``self.<name>(...)`` calls inside each ``on_scene_ready`` in ``path``."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == "on_scene_ready":
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                    names.add(call.func.attr)
    return names


class TestTutorialsJoinSettle(unittest.TestCase):
    def test_each_tutorial_joins_and_does_not_snapshot_alone(self) -> None:
        for path in TUTORIALS:
            with self.subTest(path=path.name):
                called = _called(path)
                self.assertIn("join_settle", called)
                self.assertNotIn("snapshot_rest", called)


if __name__ == "__main__":
    unittest.main()
