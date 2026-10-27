# SPDX-License-Identifier: Apache-2.0
"""The page's lighting rig: what it takes from a scene's look, and the angles its sliders speak in.

A look states a path tracer's lights, and the page translates rather than
copies: a light's direction of travel becomes a position the light shines
from, an ambient triple becomes one number no brighter than the page's own,
intensities scale the page's tuned key, and ``shadows`` and the environment
stay the page's. The sliders say where the key is as an elevation and an
azimuth, which round-trip through the position the rig keeps.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio_studio.lighting import LightingRig, PanelLighting, _from_angles
from tests.fixtures.viser_stubs import _StubGui, _StubScene

TIER = "fast"
"""No GPU, no browser: a scene stub records the lights the rig installs."""


def _look(**stated) -> SimpleNamespace:
    """A render spec stating only what is given; every other field is the backend's own (``None``)."""
    fields = {"light_direction": None, "fill_lights": None, "ambient": None, "key_intensity": None, "shadows": None}
    return SimpleNamespace(**{**fields, **stated})


class TestALooksLights(unittest.TestCase):
    def test_no_look_is_the_page_as_it_was_tuned(self) -> None:
        self.assertEqual(LightingRig(), LightingRig.from_look(None))

    def test_a_look_that_states_nothing_is_the_rig_matched_to_the_tracer(self) -> None:
        self.assertEqual(LightingRig.matched(), LightingRig.from_look(_look()))

    def test_a_key_travelling_down_shines_from_straight_above_at_the_distance(self) -> None:
        rig = LightingRig.from_look(_look(light_direction=(0.0, 0.0, -2.0)), distance=4.0)
        self.assertEqual((-0.0, -0.0, 4.0), rig.key_position)

    def test_any_key_direction_lands_at_the_distance_against_its_travel(self) -> None:
        direction = np.array([0.3, -1.2, -0.7])
        position = np.array(LightingRig.from_look(_look(light_direction=tuple(direction))).key_position)
        self.assertAlmostEqual(3.0, float(np.linalg.norm(position)))
        self.assertAlmostEqual(-1.0, float(position @ direction / np.linalg.norm(position) / np.linalg.norm(direction)))

    def test_the_first_fill_is_placed_the_same_way_and_scaled_by_the_key(self) -> None:
        rig = LightingRig.from_look(_look(fill_lights=(((1.0, 0.0, 0.0), 0.5), ((0.0, 1.0, 0.0), 9.0))))
        self.assertEqual((-3.0, -0.0, -0.0), rig.fill_position)
        self.assertAlmostEqual(0.5 * LightingRig.matched().key_intensity, rig.fill_intensity)

    def test_ambient_is_the_triples_mean_and_never_brighter_than_the_page(self) -> None:
        base = LightingRig.matched().ambient
        self.assertAlmostEqual(0.2, LightingRig.from_look(_look(ambient=(0.1, 0.2, 0.3))).ambient)
        self.assertEqual(base, LightingRig.from_look(_look(ambient=(1.5, 1.5, 1.5))).ambient)

    def test_the_key_scales_the_pages_own_and_shadows_stay_the_pages(self) -> None:
        rig = LightingRig.from_look(_look(key_intensity=2.0, shadows=True))
        self.assertAlmostEqual(2.0 * LightingRig.matched().key_intensity, rig.key_intensity)
        self.assertFalse(rig.shadows, "a rasteriser that cannot cast the shadow does not claim it")


class TestTheKeysAngles(unittest.TestCase):
    def test_the_position_is_the_distance_away_at_the_elevation_and_azimuth_given(self) -> None:
        x, y, z = _from_angles(30.0, 90.0, distance=2.0)
        self.assertAlmostEqual(0.0, x)
        self.assertAlmostEqual(2.0 * math.cos(math.radians(30.0)), y)
        self.assertAlmostEqual(1.0, z)

    def test_the_sliders_read_back_the_angles_a_key_was_placed_at(self) -> None:
        for elevation, azimuth in ((35.0, -40.0), (80.0, 170.0), (5.0, 0.0)):
            with self.subTest(elevation=elevation, azimuth=azimuth):
                rig = LightingRig(key_position=_from_angles(elevation, azimuth))
                panel = PanelLighting(SimpleNamespace(scene=_StubScene()), rig)
                self.assertAlmostEqual(elevation, panel._elevation())
                self.assertAlmostEqual(azimuth, panel._azimuth())


class TestThePanel(unittest.TestCase):
    def _panel(self) -> tuple[PanelLighting, _StubGui, _StubScene]:
        scene, gui = _StubScene(), _StubGui()
        panel = PanelLighting(SimpleNamespace(scene=scene), LightingRig(key_position=_from_angles(35.0, -40.0)))
        panel.build(gui)
        return panel, gui, scene

    def test_raising_the_key_moves_it_and_installs_it(self) -> None:
        panel, gui, scene = self._panel()
        gui.find("Key elevation").fire(60.0)
        np.testing.assert_allclose(panel.rig.key_position, _from_angles(60.0, -40.0))
        np.testing.assert_allclose(scene.lights["/lighting/key"]["position"], _from_angles(60.0, -40.0))

    def test_newtons_flat_default_is_one_tick_away_and_leaves_the_rig_alone(self) -> None:
        panel, gui, scene = self._panel()
        before = panel.rig
        gui.find("Newton default (flat)").fire(True)
        self.assertEqual({"intensity": 1.0}, scene.lights["ambient_light"])
        self.assertEqual({"hdri": None}, scene.environment)
        self.assertIs(before, panel.rig)

    def test_the_ambient_reuses_the_viewers_own_light_rather_than_adding_one(self) -> None:
        _, _, scene = self._panel()
        self.assertEqual({"ambient_light", "/lighting/key", "/lighting/fill"}, set(scene.lights))


if __name__ == "__main__":
    unittest.main()
