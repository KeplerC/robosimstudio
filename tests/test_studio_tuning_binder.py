# SPDX-License-Identifier: Apache-2.0
"""The tuning binders: ``set_spec`` records nothing, a field set back is not pending, one section per backend."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from unittest import mock

from robosimstudio.specs.params import REBUILD, ParamUI, ui
from robosimstudio_studio.tuning.arrangement import ArrangementBinder
from robosimstudio_studio.tuning.binder import SpecBinder
from tests.fixtures.viser_stubs import _StubGui

TIER = "fast"
"""No GPU: a stand-in gui and specs that are dataclasses."""


@dataclass(frozen=True)
class _Spec:
    stiff: float = field(default=13700.0, metadata=ui(tier="live", range=(1.0, 1e6), log=True))
    damp: float = field(default=0.5, metadata=ui(tier="live", range=(0.0, 1.0)))
    mode: str = field(default="a", metadata=ui(tier="live", choices=("a", "b")))


class SetSpecRecordsNothing(unittest.TestCase):
    def test_moving_widgets_to_a_new_spec_is_not_an_edit(self):
        binder = SpecBinder(_Spec())
        binder.build(_StubGui())
        binder.set_spec(replace(_Spec(), stiff=13700.0 * 1.37, damp=0.9, mode="b"))
        self.assertEqual(binder.pending, {})

    def test_a_real_edit_still_records(self):
        gui = _StubGui()
        binder = SpecBinder(_Spec())
        binder.build(gui)
        binder.set_spec(replace(_Spec(), damp=0.9))
        gui.find("damp").value = 0.25
        self.assertEqual(binder.pending, {"damp": 0.25})


class PendingRebuildForgetsWhatWasSetBack(unittest.TestCase):
    def _binder(self):
        spec = SimpleNamespace(with_=lambda **_: spec)
        binder = ArrangementBinder(spec)
        binder._fold = mock.Mock()
        gui = _StubGui()
        binder._add(gui, "rates:substeps", "substeps", 4, ParamUI(tier=REBUILD, range=(1.0, 16.0), step=1.0))
        return binder, gui.find("substeps *")

    def test_set_back_to_the_built_value(self):
        binder, widget = self._binder()
        widget.value = 8
        binder.drain()
        self.assertEqual(binder.pending_rebuild(), ("rates:substeps",))
        widget.value = 4
        binder.drain()
        self.assertEqual(binder.pending_rebuild(), ())

    def _log_binder(self, built: float = 13700.0):
        spec = SimpleNamespace(with_=lambda **_: spec)
        binder = ArrangementBinder(spec)
        binder._fold = mock.Mock()
        gui = _StubGui()
        binder._add(gui, "rates:edge_ke", "edge_ke", built, ParamUI(tier=REBUILD, range=(1.0, 1e6), log=True))
        return binder, gui.find("edge_ke *")

    def test_a_log_slider_dragged_back_onto_its_grid_is_set_back(self):
        import math

        binder, widget = self._log_binder()
        widget.value = 5.0
        binder.drain()
        self.assertEqual(binder.pending_rebuild(), ("rates:edge_ke",))
        # The browser snaps to the 0.01 grid, which the built value is not on.
        widget.value = round(math.log10(13700.0), 2)
        self.assertNotEqual(10.0**widget.value, 13700.0)
        binder.drain()
        self.assertEqual(binder.pending_rebuild(), ())
        # And what is folded is the scene's own value, not the grid point.
        binder._fold.assert_called_with("rates:edge_ke", 13700.0)

    def test_a_log_slider_one_notch_off_is_still_an_edit(self):
        import math

        binder, widget = self._log_binder()
        widget.value = round(math.log10(13700.0), 2) + 0.01
        binder.drain()
        self.assertEqual(binder.pending_rebuild(), ("rates:edge_ke",))


class OneSectionPerBackendTheSceneUses(unittest.TestCase):
    """GENERIC names an mpm backend for three particle domains and a superdex one for soft."""

    def _sections(self, domains):
        from robosimstudio.specs import GENERIC

        gui = _StubGui()
        binder = ArrangementBinder(GENERIC, domains=domains)
        binder.build(gui)
        backends = [
            h
            for h in (m.content for m in gui.markdown)
            if h.startswith("**") and h.endswith(("mujoco**", "vbd**", "mpm**", "superdex**"))
        ]
        return binder, backends

    def test_a_scene_with_no_particles_lists_no_mpm(self):
        _, backends = self._sections({"*", "articulated", "rigid"})
        self.assertEqual(["**articulated: mujoco**", "***: vbd**"], backends)

    def test_the_particle_domains_it_declares_share_one_section(self):
        _, backends = self._sections({"*", "articulated", "granular", "liquid"})
        self.assertEqual(["**articulated: mujoco**", "**granular, liquid: mpm**", "***: vbd**"], backends)

    def test_no_domains_given_lists_every_backend(self):
        binder, _ = self._sections(None)
        self.assertEqual(
            [("articulated",), ("soft",), ("granular", "liquid", "dough"), ("*",)], [d for d, _ in binder.sections()]
        )

    def test_an_edit_to_a_shared_backend_keeps_it_one_island(self):
        """Folded into ``granular`` alone it split the spec, and a heap and a pool would build two mpm islands."""
        from robosimstudio.specs import GENERIC

        binder, _ = self._sections({"*", "articulated", "granular", "liquid"})
        binder.pending["backend:granular:voxel_size"] = 0.005
        binder.drain()
        spec = binder.spec
        shared = {spec.backend_for(d) for d in ("granular", "liquid", "dough")}
        self.assertEqual(1, len(shared), "the particle domains no longer share one backend")
        self.assertEqual(0.005, spec.backend_for("liquid").get("voxel_size"))
        self.assertEqual(GENERIC.backend_for("*"), spec.backend_for("*"))
        self.assertEqual(GENERIC.backend_for("soft"), spec.backend_for("soft"))

    def _live(self, island_domains, path):
        """Drain one edit into a program of one island per backend, recording what it is asked."""
        from robosimstudio.specs import GENERIC
        from robosimstudio.specs.params import RECAPTURE

        islands = [SimpleNamespace(name=n, backend=GENERIC.backend_for(d), domains=ds) for n, d, ds in island_domains]
        asked = []

        def apply(target, value):
            asked.append(target)
            if target.split(":")[1] not in {i.name for i in islands}:
                raise KeyError(f"no island or domain {target.split(':')[1]!r}")
            return RECAPTURE

        binder, _ = self._sections(None)
        binder.program = SimpleNamespace(apply=apply, islands=islands)
        binder.pending[path] = 40
        binder.drain()
        return binder.applied[path], asked

    def test_a_shared_section_applies_live_on_its_backends_island(self):
        """Its path names ``granular``; a liquid-only program has an mpm island, and the edit lands live there."""
        from robosimstudio.specs.params import RECAPTURE

        tier, asked = self._live([("mpm", "liquid", ("liquid",))], "backend:granular:max_iterations")
        self.assertEqual(RECAPTURE, tier)
        self.assertEqual(["backend:mpm:max_iterations"], asked)

    def test_the_fallback_section_applies_live_too(self):
        """No node's domain is ``*``; the vbd island runs the fallback backend, so ``backend:*:`` reaches it."""
        from robosimstudio.specs.params import RECAPTURE

        tier, asked = self._live([("vbd", "*", ("cloth",))], "backend:*:iterations")
        self.assertEqual(RECAPTURE, tier)
        self.assertEqual(["backend:vbd:iterations"], asked)


if __name__ == "__main__":
    unittest.main()
