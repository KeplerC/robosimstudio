# SPDX-License-Identifier: Apache-2.0
"""Particle things share one settle per pipeline through ``ParticleMixin.join_settle``.

A run settles once and every thing on the record snapshots its rest after each
join; the "your own kind" tutorial entities join that settle from
``on_scene_ready`` rather than snapshotting alone, or a cloth that settles after
them leaves them with an unsettled rest to reset to.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from tests.fixtures.settle import particle_thing

TIER = "fast"
"""Stand-in things and a stand-in pipeline, and the two tutorial kinds imported without a build: no GPU."""


class TestJoinSettle(unittest.TestCase):
    def test_a_run_settles_once_and_every_thing_snapshots_after_each_join(self) -> None:
        pipeline = SimpleNamespace()
        sheet, block, cable = particle_thing(), particle_thing(), particle_thing()
        sheet.join_settle(pipeline, (10, 4))
        block.join_settle(pipeline, (10, 4))
        cable.join_settle(pipeline)
        self.assertEqual((1, 0, 0), (sheet.settles, block.settles, cable.settles))
        self.assertEqual({(10, 4)}, pipeline.particle_settles["runs"])
        self.assertEqual([sheet, block, cable], pipeline.particle_settles["things"])
        self.assertEqual((3, 2, 1), (sheet.snapshots, block.snapshots, cable.snapshots))

    def test_a_different_run_settles_again(self) -> None:
        pipeline = SimpleNamespace()
        a, b = particle_thing(), particle_thing()
        a.join_settle(pipeline, (10, 4))
        b.join_settle(pipeline, (20, 4))
        self.assertEqual((1, 1), (a.settles, b.settles))


ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = (
    ROOT / "examples" / "01_cloth" / "04_your_own_cloth.py",
    ROOT / "examples" / "02_soft" / "02_your_own_soft.py",
)


def _tutorialparticle_thing(path: Path):
    """The tutorial's own particle kind, made without a build: its ``snapshot_rest`` counts, and nothing else stands in."""
    import importlib.util

    from robosimstudio.domains.capabilities import ParticleMixin

    spec = importlib.util.spec_from_file_location(f"_tutorial_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (kind,) = [
        value
        for value in vars(module).values()
        if isinstance(value, type) and issubclass(value, ParticleMixin) and value.__module__ == module.__name__
    ]
    thing = kind.__new__(kind)
    thing.snapshots = 0

    def snapshot_rest(pipeline) -> None:
        del pipeline
        thing.snapshots += 1

    thing.snapshot_rest = snapshot_rest
    return thing


class TestTutorialsJoinSettle(unittest.TestCase):
    def test_each_tutorial_joins_and_does_not_snapshot_alone(self) -> None:
        """Ready after a settled sheet, the tutorial thing lands on the record and is re-baselined by the next join."""
        for path in TUTORIALS:
            with self.subTest(path=path.name):
                pipeline = SimpleNamespace()
                sheet, mine, later = particle_thing(), _tutorialparticle_thing(path), particle_thing()
                sheet.join_settle(pipeline, (10, 4))
                mine.on_scene_ready(pipeline)
                self.assertIn(mine, pipeline.particle_settles["things"], "on the scene's one settle record")
                later.join_settle(pipeline, (20, 4))
                self.assertEqual(2, mine.snapshots, "re-baselined once on joining and again after the later settle")
                self.assertIsNone(mine.settle_run, "it joins without a run of its own")


if __name__ == "__main__":
    unittest.main()
