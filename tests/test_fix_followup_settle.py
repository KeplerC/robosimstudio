# SPDX-License-Identifier: Apache-2.0
"""Particle things share one settle per pipeline through ``ParticleMixin.join_settle``,
and the studio's re-settle runs the settle of every thing on that record that has one and
re-baselines every thing on it, not only the cloth.

No GPU: stand-in things and a stand-in pipeline.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

TIER = "fast"
"""Stand-ins only: no GPU."""


def _thing(name: str = "towel", log: list | None = None):
    from robosimstudio.domains.capabilities import ParticleMixin

    class _Stub(ParticleMixin):
        def __init__(self) -> None:
            self.name = name
            self.settles = 0
            self.snapshots = 0

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


class TestStudioResettle(unittest.TestCase):
    def _panel(self, pipeline, cloth):
        from robosimstudio_studio.tuning.panels import PanelReadings

        panel = PanelReadings(SimpleNamespace(env=SimpleNamespace(pipeline=pipeline)))
        panel._note = SimpleNamespace(value="")
        panel.resettle_requested = True
        return panel

    def test_every_thing_on_the_record_is_rebaselined(self) -> None:
        cloth, cable = _thing("towel"), _thing("sponge")
        pipeline = SimpleNamespace(particle_settles={"runs": set(), "things": [cloth, cable]})
        panel = self._panel(pipeline, cloth)
        with (
            mock.patch.object(type(panel), "_busy", return_value=""),
            mock.patch.object(type(panel), "_cloth", return_value=cloth),
            mock.patch.object(type(panel), "refresh"),
        ):
            panel.drain()
        self.assertEqual((1, 1), (cloth.settles, cable.settles))
        self.assertEqual((1, 1), (cloth.snapshots, cable.snapshots))
        self.assertEqual("re-settled towel, sponge", panel._note.value)

    def test_without_a_record_the_cloth_alone_is_rebaselined(self) -> None:
        cloth = _thing()
        panel = self._panel(SimpleNamespace(), cloth)
        with (
            mock.patch.object(type(panel), "_busy", return_value=""),
            mock.patch.object(type(panel), "_cloth", return_value=cloth),
            mock.patch.object(type(panel), "refresh"),
        ):
            panel.drain()
        self.assertEqual((1, 1), (cloth.settles, cloth.snapshots))

    def _rope(self, log: list):
        """A SuperDex rope: on the record without a run, and no ``settle`` of its own."""
        from robosimstudio.domains.capabilities import ParticleMixin

        class _Rope(ParticleMixin):
            name = "cable"
            snapshots = 0

            def snapshot_rest(self, pipeline) -> None:
                del pipeline
                self.snapshots += 1
                log.append("snapshot cable")

        return _Rope()

    def _drain(self, things, first):
        pipeline = SimpleNamespace(particle_settles={"runs": set(), "things": things})
        panel = self._panel(pipeline, first)
        with (
            mock.patch.object(type(panel), "_busy", return_value=""),
            mock.patch.object(type(panel), "_cloth", return_value=first),
            mock.patch.object(type(panel), "refresh"),
        ):
            panel.drain()
        return panel

    def test_a_thing_that_does_not_settle_is_rebaselined_where_it_lies(self) -> None:
        """A SuperDex rope joins the record without a run and has no ``settle``: the button raised
        ``AttributeError`` out of the page loop, measured on ``tests/scenes/02_superdex`` and, for
        the same reason, ``examples/01_cloth/04_your_own_cloth``."""
        log: list[str] = []
        rope = self._rope(log)
        panel = self._drain([rope], rope)
        self.assertEqual(["snapshot cable"], log, "nothing stepped")
        self.assertEqual("re-baselined cable where it lies", panel._note.value)

    def test_the_things_that_settle_behind_one_that_does_not_settle(self) -> None:
        """A rope first on the record, then a towel and a sponge that settle: only the first thing was
        asked, so the towel and the sponge were skipped and the page said the rope "does not settle"."""
        log: list[str] = []
        rope, towel, sponge = self._rope(log), _thing("towel", log), _thing("sponge", log)
        panel = self._drain([rope, towel, sponge], rope)
        self.assertEqual((1, 1), (towel.settles, sponge.settles))
        self.assertEqual(
            ["settle towel", "settle sponge", "snapshot cable", "snapshot towel", "snapshot sponge"],
            log,
            "every baseline is taken after the last settle",
        )
        self.assertEqual("re-settled towel, sponge; re-baselined cable where it lies", panel._note.value)


if __name__ == "__main__":
    unittest.main()
