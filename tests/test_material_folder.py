# SPDX-License-Identifier: Apache-2.0
"""The ``Material`` folder writes through the entity's own ``material`` tunable.

It had a writer of its own, ``LiveMaterial``, which knows the cloth fields and nothing else. Paper's
plastic trio and tape's adhesive are routed by their entities' tunables, which only the inspector
called, so on the fold page ``yield_angle`` moved, the working spec changed and the crease kernel read
the old number (the verifier's ``yield_probe.py``: kernel array ``[0.35, 1.0, 0.0]`` before and after
setting the slider to 0). These drive the folder the way a slider does -- a widget fires, the panel
drains -- over stub entities, and check where the value went. The same check over the real kinds,
and the arrays their solvers read, is ``tests/test_material_sliders``.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.core.tuning import Tunable
from robosimstudio.scene.tuning import LiveMaterial, locate_cloth
from robosimstudio.specs import ArrangementSpec, backend
from robosimstudio_materials.cloth import ClothMaterialSpec
from robosimstudio_materials.paper import PaperMaterialSpec
from tests.fixtures.viser_stubs import _StubGui

TIER = "fast"
"""No GPU and no scene: stub entities against the viser stubs."""


class _Array:
    def __init__(self, values) -> None:
        self.values = np.asarray(values, dtype=np.float64)

    def numpy(self):
        return self.values

    def assign(self, values) -> None:
        self.values = np.array(values, copy=True)


def _env(**entities):
    model = SimpleNamespace(tri_materials=_Array(np.ones((2, 5))), particle_radius=_Array(np.full(4, 0.002)))
    pipeline = SimpleNamespace(captured=True, invalidated=0)
    pipeline.invalidate_graph = lambda: setattr(pipeline, "invalidated", pipeline.invalidated + 1)
    return SimpleNamespace(
        scene=SimpleNamespace(by_name=entities, program=None),
        model=model,
        pipeline=pipeline,
        request=SimpleNamespace(
            physics=ArrangementSpec(control_hz=60.0, substeps=10, backends=(("*", backend("mujoco")),))
        ),
    )


def _sheet(name, material, offers=True, plastic=None):
    """A membrane over particles 0-3 and triangles 0-1; ``plastic`` collects what its own fields received."""
    entity = SimpleNamespace(
        spec=SimpleNamespace(name=name, material=material),
        range=SimpleNamespace(particles=(0, 4), tris=(0, 2)),
        carries_particles=True,
    )

    def tunables(pipeline=None):
        if not offers:
            return ()
        live = LiveMaterial(entity.env.model, locate_cloth(entity.env.model, entity), pipeline)

        def write(field, value):
            if plastic is not None and field in ("yield_angle", "plastic_rate", "plastic_recovery"):
                plastic[field] = float(value)
                return True
            return live.write_field(field, value)

        return (Tunable(name="material", spec=material, write=write, inert=live.inert()),)

    entity.tunables = tunables
    return entity


def _page(*entities):
    from robosimstudio_studio.tuning.panels import PanelTune

    env = _env(**{e.spec.name: e for e in entities})
    for e in entities:
        e.env = env
    panel, gui = PanelTune(SimpleNamespace(env=env)), _StubGui()
    panel.build(gui)
    return panel, gui, env


def _slider(gui, name):
    return next(w for label, w in gui.widgets if label.split(" ")[0] == name)


class TestAFieldAKindAddsReachesItsWriter(unittest.TestCase):
    def test_yield_angle_set_on_the_folder_lands_where_the_kind_routes_it(self) -> None:
        plastic: dict = {}
        panel, gui, _ = _page(_sheet("paper", PaperMaterialSpec(), plastic=plastic))
        _slider(gui, "yield_angle").fire(0.0)
        panel.drain()
        self.assertEqual({"yield_angle": 0.0}, plastic)

    def test_revert_writes_the_kinds_fields_back_too(self) -> None:
        plastic: dict = {}
        panel, gui, _ = _page(_sheet("paper", PaperMaterialSpec(), plastic=plastic))
        _slider(gui, "yield_angle").fire(0.0)
        panel.drain()
        panel.revert_requested = True
        panel.drain()
        self.assertAlmostEqual(PaperMaterialSpec().yield_angle, plastic["yield_angle"])

    def test_a_cloth_field_still_lands_in_the_cloths_rows(self) -> None:
        panel, gui, env = _page(_sheet("paper", PaperMaterialSpec(), plastic={}))
        _slider(gui, "particle_radius").fire(0.003)
        panel.drain()
        self.assertTrue(np.allclose(env.model.particle_radius.values, 0.003))


class TestTheContactTrio(unittest.TestCase):
    def test_the_inspectors_writer_sets_the_model_scalar_and_drops_a_captured_graph(self) -> None:
        sheet = _sheet("towel", ClothMaterialSpec())
        _page(sheet)
        (material,) = sheet.tunables(sheet.env.pipeline)
        self.assertTrue(material.write("friction", 0.37))
        self.assertAlmostEqual(0.37, sheet.env.model.soft_contact_mu)
        self.assertEqual(1, sheet.env.pipeline.invalidated)


class TestAPresetIsOneTheSheetCanWear(unittest.TestCase):
    """A tape's spec refuses a material with no adhesive: a towel preset adopted onto the label moved its
    sliders, and the rebuild then raised in ``TapeSpec.__post_init__``."""

    def test_each_sheet_is_offered_its_own_material_classs_presets(self) -> None:
        from robosimstudio_materials.tape import PAPER_LABEL

        panel, gui, _ = _page(_sheet("label", PAPER_LABEL), _sheet("towel", ClothMaterialSpec()))
        preset = gui.find("Preset")
        # Every tape preset, the contact adhesion ones too: ``adhesion`` is REBUILD, and a stuck label
        # rebuilds under either law.
        self.assertEqual(("(as built)", "duct_tape", "label", "masking_tape", "paper_label"), preset.options)
        gui.find("Cloth").fire("towel")
        panel.drain()
        self.assertIn("terry_towelling", preset.options)
        self.assertIn("paper_label", preset.options, "a label's material is a cloth's too")
        for other in ("rice", "soft_foam", "nylon_bristle"):
            self.assertNotIn(other, preset.options, "no sheet is made of that")
        self.assertEqual("(as built)", preset.value)

    def test_adopting_a_preset_of_another_class_is_refused(self) -> None:
        from robosimstudio_materials.tape import PAPER_LABEL

        panel, _, _ = _page(_sheet("label", PAPER_LABEL))
        with self.assertRaisesRegex(ValueError, "TapeMaterialSpec"):
            panel.tuner.adopt("terry_towelling")
        panel.tuner.adopt("paper_label")
        self.assertIs(PAPER_LABEL, panel.tuner.material)


class TestAnEditTheSpecRefusesIsNotTaken(unittest.TestCase):
    """A tape's reach dragged under its peak_opening: ``replace`` reruns ``TapeMaterialSpec.__post_init__``,
    which raised out of ``SpecBinder.drain`` and the panel's drain, and ended the page's loop."""

    def test_the_folder_keeps_the_spec_and_says_why(self) -> None:
        from robosimstudio_materials.tape import LABEL

        panel, gui, _ = _page(_sheet("label", LABEL))
        _slider(gui, "reach").fire(4.0e-4)
        panel.drain()
        self.assertEqual(LABEL.reach, panel.binder.spec.reach)
        self.assertEqual(LABEL.reach, panel.tuner.material.reach, "nothing reached the working spec")
        self.assertAlmostEqual(LABEL.reach, _slider(gui, "reach").value, msg="the slider goes back")
        self.assertIn("reach not changed", panel.session.status)
        _slider(gui, "reach").fire(2.0e-3)
        panel.drain()
        self.assertEqual(2.0e-3, panel.tuner.material.reach, "the next edit the spec takes lands")
        self.assertEqual("", panel.binder.refused)

    def test_the_inspector_keeps_its_loop_too(self) -> None:
        from robosimstudio_materials.tape import LABEL
        from robosimstudio_studio.tuning.binder import SpecBinder

        binder, gui = SpecBinder(LABEL), _StubGui()
        binder.build(gui)
        _slider(gui, "peak_opening").fire(2.0e-3)
        self.assertEqual({}, binder.drain())
        self.assertIs(LABEL, binder.spec)
        self.assertIn("peak_opening < reach", binder.refused)


class TestWhatCannotMoveIsNotOffered(unittest.TestCase):
    def test_a_kind_whose_engine_copied_its_material_is_not_on_the_folder(self) -> None:
        panel, gui, _ = _page(_sheet("bag", ClothMaterialSpec(), offers=False))
        self.assertFalse(panel.tuner)
        self.assertNotIn("Material", [f.label for f in gui.folders if f.label is not None])

    def test_a_membranes_core_fields_are_read_only(self) -> None:
        _, gui, _ = _page(_sheet("towel", ClothMaterialSpec()))
        for name in ("core_shear_modulus", "core_lame_lambda", "core_damping"):
            self.assertTrue(_slider(gui, name).disabled, name)
        self.assertFalse(_slider(gui, "tri_ke").disabled)


if __name__ == "__main__":
    unittest.main()
