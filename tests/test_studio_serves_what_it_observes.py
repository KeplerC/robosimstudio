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
that builds them (the page itself is ``tests.test_cut_tofu``). The
``Appearance`` folder's ``Cloth look``, which the cutting page showed with no
cloth on it, took ``Sim.deformables`` too, and so offered a look on a SuperDex
rope's page (``tests/scenes/02_superdex``) and an MPM honey's
(``03_materials/12_liquid``) that changed nothing: a garment style changes only
a sheet's draw. It asks the viewer for its sheets now. No device: the panels
over stand-ins for the scene, its draggers and the browser, and a viewer that
never opens a server.
"""

from __future__ import annotations

import types
import unittest
import unittest.mock

import numpy as np

from tests.test_studio_cloth_requests import _Dragger

TIER = "fast"
"""Stand-ins only; no model, no viser server."""


def _env(*, with_towel: bool = True):
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
        scene=types.SimpleNamespace(
            by_name={"tofu": tofu, **({"towel": towel} if with_towel else {})}, embodiment=None
        ),
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


def _garment(name: str, **handle):
    """What ``garment_visuals`` makes of one handle: a sheet's triangles, a tube's radius, a body's points."""
    from robosimstudio_studio.garments import GarmentVisual

    triangles = handle.pop("triangles", np.zeros((0, 3), dtype=np.int64))
    return GarmentVisual(name=name, particles=np.arange(8), triangles=triangles, color=(0.85, 0.85, 0.88), **handle)


TOWEL = {"triangles": np.array([[0, 1, 2], [1, 3, 2]])}
ROPE = {"tube_radius": 0.004}  # a centerline, tests/scenes/02_superdex's cable: a tube radius, no triangles
HONEY = {"point_radius": 0.003}  # a particle body, 03_materials/12_liquid's honey: points, no triangles
TOFU = {"point_radius": 1.0e-3}  # a cuttable: its corners as points, and its surface mesh


class TestTheClothLookIsOfferedForACloth(unittest.TestCase):
    def _labels(self, env, **garments) -> list[str]:
        from robosimstudio_studio.garments import ViewerStudio
        from robosimstudio_studio.session import StudioSession
        from tests.fixtures.viser_stubs import _StubGui

        viewer = object.__new__(ViewerStudio)  # no server: only the garment bookkeeping is asked
        viewer._garment_nodes, viewer._garment_hidden, viewer.model = {}, {}, None
        viewer._garment_style = "fluffy"
        viewer.set_garments([_garment(name, **dict(handle)) for name, handle in garments.items()])
        gui = _StubGui()
        session = types.SimpleNamespace(
            env=env,
            show_scenery=True,
            viewer=viewer,
            lighting=types.SimpleNamespace(build=lambda gui: None),
        )
        StudioSession._build_appearance(session, gui)
        return [label for label, _ in gui.widgets]

    def test_the_cutting_page_has_no_cloth_look(self) -> None:
        labels = self._labels(_env(with_towel=False), tofu=TOFU)
        self.assertNotIn("Cloth look", labels, "the tofu's corners are no cloth")

    def test_a_page_with_a_towel_has_one(self) -> None:
        self.assertIn("Cloth look", self._labels(_env(), towel=TOWEL))

    def test_a_rope_page_has_none(self) -> None:
        """The cable carries a probe, so ``Sim.deformables`` offered it a look; it is drawn as its tube in every style."""
        self.assertNotIn("Cloth look", self._labels(_env(), cable=ROPE))

    def test_a_water_page_has_none(self) -> None:
        self.assertNotIn("Cloth look", self._labels(_env(), honey=HONEY), "a particle body is drawn as its points")

    def test_a_towel_beside_a_rope_has_one(self) -> None:
        self.assertIn("Cloth look", self._labels(_env(), cable=ROPE, towel=TOWEL))


if __name__ == "__main__":
    unittest.main()
