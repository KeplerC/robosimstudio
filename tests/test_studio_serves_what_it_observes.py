# SPDX-License-Identifier: Apache-2.0
"""The deformable panels serve what the scene observes, not every particle carrier.

``PanelCloth`` listed every entity that ``carries_particles`` and called
``observe`` on the one selected, so the cutting example's page with
``panels="all"`` -- the launcher's default -- died on its first refresh with
``'EntityCuttable' object has no attribute 'observe'``. A cuttable owns eight
particles, the corners its cage solver writes back after every control step
for the page to draw the cut by, with no probe to summarise them and nothing a
pin on them would move. The ``Readings`` folder picked the same first carrier
and read a footprint and a settle speed off those corners. Both now take
``Sim.deformables``, the things carrying a particle probe, as does the gate
that builds them (the page itself is ``tests.test_cut_tofu``). No device: the
panels over stand-ins for the scene, its draggers and the browser.
"""

from __future__ import annotations

import types
import unittest
import unittest.mock

import numpy as np

from tests.test_studio_cloth_requests import _Dragger

TIER = "fast"
"""Stand-ins only; no model, no viser server."""


def _env():
    """A cuttable's eight corners, then a towel that carries a probe, on one particle array."""
    from robosimstudio.sim import Sim

    tofu = types.SimpleNamespace(name="tofu", carries_particles=True, particles=np.arange(8))
    towel = types.SimpleNamespace(
        name="towel",
        carries_particles=True,
        particles=np.arange(8, 12),
        probe=object(),
        observe=lambda state, step: {"z_min": 0.10, "z_max": 0.12},  # a ParticleMixin's summary, the band of it
        published_states=lambda context: {},
    )
    q = np.zeros((12, 3))
    env = types.SimpleNamespace(
        scene=types.SimpleNamespace(by_name={"tofu": tofu, "towel": towel}, embodiment=None),
        handles=types.SimpleNamespace(deformables={"towel": types.SimpleNamespace(sets={})}),
        pipeline=types.SimpleNamespace(hooks=types.SimpleNamespace(post_solve=[]), invalidate_graph=lambda: None),
        model=None,
        state=types.SimpleNamespace(particle_q=types.SimpleNamespace(numpy=lambda: q)),
        step_count=0,
    )
    env.deformables = lambda: Sim.deformables(env)  # the scene's own answer, not a list written for the test
    return env


class TestTheClothPanelServesWhatTheSceneObserves(unittest.TestCase):
    def _panel(self):
        from robosimstudio_studio.panels import PanelCloth
        from tests.fixtures.viser_stubs import _StubGui, _StubSession

        gui = _StubGui()
        with unittest.mock.patch("robosimstudio.core.draggers.ParticleDragger", _Dragger):
            panel = PanelCloth(_StubSession(_env(), gui), handles=2, chrome=True)
        panel.build(gui)
        return panel, gui

    def test_the_cuttable_is_not_offered(self) -> None:
        panel, gui = self._panel()
        self.assertEqual(["towel"], panel.names)
        self.assertEqual(("towel",), tuple(gui.find("Cloth").options))

    def test_a_click_on_a_corner_finds_no_deformable(self) -> None:
        panel, _ = self._panel()
        self.assertEqual("", panel.owner_of(0), "a corner of the cut block, which no pin moves")
        self.assertEqual("towel", panel.owner_of(9))

    def test_the_page_refreshes(self) -> None:
        panel, _ = self._panel()
        panel.refresh()  # raised AttributeError: 'observe', out of the page loop
        self.assertEqual("z 0.100..0.120 m", panel._state.value)


class TestTheReadingsFolderReadsADeformable(unittest.TestCase):
    def test_it_reads_the_first_thing_with_a_probe(self) -> None:
        from robosimstudio_studio.tuning.panels import PanelReadings

        panel = PanelReadings(types.SimpleNamespace(env=_env()))
        self.assertEqual("towel", panel._cloth().name, "the cuttable's corners came first")


if __name__ == "__main__":
    unittest.main()
