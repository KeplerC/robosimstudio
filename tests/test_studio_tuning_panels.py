# SPDX-License-Identifier: Apache-2.0
"""The tuning panels: the preset dropdown follows reverts, late readings get a row,
and Measure / Re-settle refuse while a policy drives or a take records.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from robosimstudio_studio.tuning import panels as panels_mod
from robosimstudio_studio.tuning.metrics import Reading
from robosimstudio_studio.tuning.panels import PanelReadings, PanelTune
from robosimstudio_studio.tuning.tuner import MaterialTuner
from tests.fixtures.viser_stubs import _StubGui, _StubWidget

TIER = "fast"
"""No GPU: a stand-in gui and tuner."""


class _Tuner:
    """Just the provenance half of :class:`MaterialTuner`."""

    def __init__(self):
        self.origin = ""
        self.material = object()

    def __bool__(self):
        return True

    def adopt(self, preset):
        self.origin = preset
        return True

    def revert(self):
        self.origin = ""

    def preset_name(self):
        return self.origin


def _tune_panel():
    panel = PanelTune.__new__(PanelTune)
    panel.tuner = _Tuner()
    panel.binder = mock.Mock()
    panel.physics = mock.Mock()
    panel.cloth_requested = panel.preset_requested = ""
    panel.revert_requested = panel.rebuild_requested = panel.save_requested = False
    panel._presets = ("(as built)", "denim", "silk")
    panel._preset = _StubWidget(panel._presets[0])
    panel._preset.on_update(lambda _: setattr(panel, "preset_requested", str(panel._preset.value)))
    panel._pending = None
    panel._calibration = None
    return panel


class PresetDropdown(unittest.TestCase):
    def test_revert_resets_dropdown_so_preset_can_be_picked_again(self):
        panel = _tune_panel()
        panel._preset.value = "denim"
        panel.drain()
        self.assertEqual(panel.tuner.origin, "denim")
        panel.revert_requested = True
        panel.drain()
        self.assertEqual(panel._preset.value, "(as built)")
        self.assertEqual(panel.preset_requested, "")
        panel._preset.value = "denim"
        panel.drain()
        self.assertEqual(panel.tuner.origin, "denim")

    def test_picking_as_built_reverts(self):
        panel = _tune_panel()
        panel._preset.value = "silk"
        panel.drain()
        panel._preset.value = "(as built)"
        panel.drain()
        self.assertEqual(panel.tuner.origin, "")
        self.assertFalse(panel.revert_requested)

    def test_tuner_revert_forgets_origin(self):
        tuner = MaterialTuner.__new__(MaterialTuner)
        tuner.selected = "shirt"
        tuner.working = {"shirt": "denim-spec"}
        tuner.built = {"shirt": "built-spec"}
        tuner.origin = {"shirt": "denim"}
        tuner.live = {"shirt": mock.Mock()}
        tuner.apply_contact = mock.Mock()
        tuner.revert()
        self.assertEqual(tuner.preset_name(), "")
        self.assertIsNone(tuner.calibration())


def _readings_panel(**session):
    fields = {"driver": None, "record": None, "env": SimpleNamespace(pipeline=object()), **session}
    panel = PanelReadings(SimpleNamespace(**fields))
    panel._cloth = object
    return panel


class ReadingsRows(unittest.TestCase):
    def test_reading_after_build_gets_a_row(self):
        panel = _readings_panel()
        readings = [Reading("sag", 1.0, "cm")]
        panel._read = lambda: list(readings)
        gui = _StubGui()
        panel.build(gui)
        readings.append(Reading("folded", 0.5))
        panel.refresh()
        self.assertIn("folded", panel._rows)
        self.assertEqual(gui.find("folded").value, "0.5")


class ReadingsRefuseWhileDriven(unittest.TestCase):
    def _check(self, **session):
        panel = _readings_panel(**session)
        panel._read = lambda: []
        panel.build(_StubGui())
        panel.measure_requested = panel.resettle_requested = True
        with mock.patch.object(panels_mod, "relax") as relax:
            panel.drain()
        relax.assert_not_called()
        self.assertFalse(panel.measure_requested or panel.resettle_requested)
        self.assertIn("first", panel._note.value)

    def test_refuses_while_a_policy_drives(self):
        self._check(driver=SimpleNamespace(running=True))

    def test_refuses_while_a_take_records(self):
        self._check(record=SimpleNamespace(recording=True))

    def test_measures_when_idle(self):
        panel = _readings_panel()
        panel._read = lambda: []
        panel.build(_StubGui())
        panel.measure_requested = True
        with mock.patch.object(panels_mod, "relax", return_value=(True, 12)) as relax:
            panel.drain()
        relax.assert_called_once()
        self.assertEqual(panel._note.value, "settled in 12 steps")


if __name__ == "__main__":
    unittest.main()
