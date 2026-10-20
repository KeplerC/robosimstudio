# SPDX-License-Identifier: Apache-2.0
"""Particle things share one settle per pipeline through ``ParticleMixin.join_settle``,
and the studio's re-settle runs the settle of every thing on that record that has one and
re-baselines every thing on it, not only the cloth -- each distinct run once, as the build does,
and naming as re-settled only the things whose run stepped.

No GPU: stand-in things and a stand-in pipeline.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

TIER = "fast"
"""Stand-ins only: no GPU."""


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


def _panel(pipeline):
    """The readings panel over ``pipeline``, its **Re-settle** pressed."""
    from robosimstudio_studio.tuning.panels import PanelReadings

    panel = PanelReadings(SimpleNamespace(env=SimpleNamespace(pipeline=pipeline)))
    panel._note = SimpleNamespace(value="")
    panel.resettle_requested = True
    return panel


class TestStudioResettle(unittest.TestCase):
    def _panel(self, pipeline, cloth):
        del cloth
        return _panel(pipeline)

    def test_every_thing_on_the_record_is_rebaselined(self) -> None:
        cloth, cable = _thing("towel", run=(60, 10)), _thing("sponge", run=(30, 4))
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
        cloth = _thing(run=(60, 10))
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
        rope, towel, sponge = self._rope(log), _thing("towel", log, (60, 10)), _thing("sponge", log, (30, 4))
        panel = self._drain([rope, towel, sponge], rope)
        self.assertEqual((1, 1), (towel.settles, sponge.settles))
        self.assertEqual(
            ["settle towel", "settle sponge", "snapshot cable", "snapshot towel", "snapshot sponge"],
            log,
            "every baseline is taken after the last settle",
        )
        self.assertEqual("re-settled towel, sponge; re-baselined cable where it lies", panel._note.value)

    def test_two_like_sheets_settle_once(self) -> None:
        """Two sheets on ``(10, 4)`` are one run at the build (``TestJoinSettle``), and a settle steps the
        whole particle array: the button ran both, stepping the scene twice -- 120 steps and 2.53 s of
        page loop where the build took 60 and 1.26 s, measured on the two sheets
        ``tests.test_fix_examples_soft_cloth`` builds."""
        a, b = _thing("a", run=(10, 4)), _thing("b", run=(10, 4))
        panel = self._drain([a, b], a)
        self.assertEqual((1, 0), (a.settles, b.settles), "one run, stepped once")
        self.assertEqual((1, 1), (a.snapshots, b.snapshots))
        self.assertEqual("re-settled a, b", panel._note.value, "b's run stepped: it was a's")

    def test_a_different_run_settles_again(self) -> None:
        a, b = _thing("a", run=(10, 4)), _thing("b", run=(20, 4))
        self._drain([a, b], a)
        self.assertEqual((1, 1), (a.settles, b.settles))

    def test_a_settle_of_no_frames_is_named_as_rebaselined(self) -> None:
        """A tape label's ``settle_frames`` is 0, so its settle returns at once: beside a towel the State line
        read "re-settled towel, label" when nothing had stepped the label's run."""
        towel, label = _thing("towel", run=(60, 10)), _thing("label", run=(0, 10))
        panel = self._drain([towel, label], towel)
        self.assertEqual("re-settled towel; re-baselined label where it lies", panel._note.value)


class TestResettleRunsWhatTheBuildRan(unittest.TestCase):
    """Re-settle runs the run each thing joined the settle with, as the build did.

    It worked each thing's run out again from its spec's ``(settle_frames, settle_substeps)``, which
    is what the shipped kinds hand ``join_settle`` and not what a kind has to: one that joins with no
    run, as the examples' own kinds do, was settled on every Re-settle though the build never settled
    it, and one that joins with a run of its own was settled on its spec's. Here each thing joins at a
    "build", then the button is pressed, and each settles as often as it did at the build.
    """

    def _build_then_resettle(self, joins: list) -> tuple[list[int], list[int], str]:
        pipeline = SimpleNamespace()
        for thing, run in joins:
            thing.join_settle(pipeline, run)
        built = [thing.settles for thing, _run in joins]
        for thing, _run in joins:
            thing.settles = 0
        panel = _panel(pipeline)
        with (
            mock.patch.object(type(panel), "_busy", return_value=""),
            mock.patch.object(type(panel), "_cloth", return_value=joins[0][0]),
            mock.patch.object(type(panel), "refresh"),
        ):
            panel.drain()
        return built, [thing.settles for thing, _run in joins], panel._note.value

    def test_a_thing_that_joins_with_no_run_is_not_settled(self) -> None:
        mine = _thing("mine", run=(60, 10))  # its spec states a run it does not join with
        built, resettled, note = self._build_then_resettle([(mine, None)])
        self.assertEqual(([0], [0]), (built, resettled))
        self.assertEqual("re-baselined mine where it lies", note)

    def test_a_run_of_its_own_is_run_again_and_shared_as_at_the_build(self) -> None:
        a, b = _thing("a", run=(60, 10)), _thing("b", run=(30, 2))
        built, resettled, note = self._build_then_resettle([(a, (30, 2)), (b, (30, 2))])
        self.assertEqual(([1, 0], [1, 0]), (built, resettled), "one run, (30, 2), stepped once both times")
        self.assertEqual("re-settled a, b", note)


if __name__ == "__main__":
    unittest.main()
