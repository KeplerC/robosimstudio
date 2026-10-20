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

import math
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from robosimstudio.core.tuning import Tunable
from robosimstudio.scene.tuning import LiveMaterial, locate_cloth
from robosimstudio.specs import ArrangementSpec, backend
from robosimstudio.specs.params import iter_params
from robosimstudio_materials.cloth import ClothMaterialSpec
from robosimstudio_materials.paper import PaperMaterialSpec
from robosimstudio_materials.tape import PAPER_LABEL, TapeMaterialSpec
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
        for other in ("paper_label", "paper", "cardboard"):
            self.assertNotIn(other, preset.options, "a towel shows a cloth's fields, not a label's or paper's")
        for other in ("rice", "soft_foam", "nylon_bristle"):
            self.assertNotIn(other, preset.options, "no sheet is made of that")
        self.assertEqual("(as built)", preset.value)

    def test_adopting_a_preset_of_another_class_is_refused(self) -> None:
        from robosimstudio_materials.tape import PAPER_LABEL

        panel, _, _ = _page(_sheet("label", PAPER_LABEL))
        self.assertFalse(panel.tuner.adopt("terry_towelling"))
        self.assertIn("TapeMaterialSpec", panel.tuner.refused)
        self.assertIs(PAPER_LABEL, panel.tuner.material, "nothing loaded")
        self.assertTrue(panel.tuner.adopt("paper_label"))
        self.assertIs(PAPER_LABEL, panel.tuner.material)

    def test_the_presets_are_listed_once(self) -> None:
        """Not once more a preset: each name comes from the listing, so none is looked up in it again."""
        from unittest import mock

        from robosimstudio_materials.tape import PAPER_LABEL
        from robosimstudio_studio.tuning import tuner

        panel, _, _ = _page(_sheet("label", PAPER_LABEL))
        with mock.patch.object(tuner, "list_materials", wraps=tuner.list_materials) as listed:
            offered = panel.tuner.presets()
        self.assertEqual(("duct_tape", "label", "masking_tape", "paper_label"), offered)
        self.assertEqual(1, listed.call_count)


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


def _tape(material=None, **spec):
    """A real tape whose contact law's parameters sit on the CPU, its cloth fields landing nowhere."""
    from unittest import mock

    import warp as wp

    from robosimstudio_materials.cloth import EntityCloth
    from robosimstudio_materials.tape import CONTACT_FIELDS, LABEL, EntityTape, tape

    material = material or LABEL
    label = EntityTape(tape(name="label", on="box", material=material, **spec))
    label.contact, label.bonds = material.adhesion == "contact", np.arange(126, dtype=np.int32)
    label.owned_particles = lambda: np.arange(4)
    label._contact = {"params": wp.array([getattr(material, f) for f in CONTACT_FIELDS], dtype=float, device="cpu")}
    cloth = Tunable(name="material", spec=material, write=lambda field, value: True)
    with mock.patch.object(EntityCloth, "tunables", return_value=(cloth,)):
        offered = label.tunables()
    label.tunables = lambda pipeline=None: offered
    return label


def _law(label) -> np.ndarray:
    """``peak_opening`` and ``reach`` as the kernels hold them [m]."""
    return label._contact["params"].numpy()[1:3]


class TestAWholeLawIsWrittenAsOneSet(unittest.TestCase):
    """Revert, ``(as built)`` and a preset wrote a tape's law one field at a time, each checked against the
    rest as the kernels held them: with the peak and the reach dragged to 0.1 and 0.4 mm, Revert wrote
    LABEL's 0.5 mm peak first, past the 0.4 mm reach, and the ``ValueError`` ended the page's loop."""

    def _dragged(self):
        label = _tape(stuck=False)
        panel, gui, _ = _page(label)
        _slider(gui, "peak_opening").fire(1.0e-4)
        panel.drain()
        _slider(gui, "reach").fire(4.0e-4)
        panel.drain()
        np.testing.assert_allclose([1.0e-4, 4.0e-4], _law(label), rtol=1e-6)
        return label, panel, gui

    def test_revert(self) -> None:
        from robosimstudio_materials.tape import LABEL

        label, panel, _ = self._dragged()
        panel.revert_requested = True
        panel.drain()
        np.testing.assert_allclose([LABEL.peak_opening, LABEL.reach], _law(label), rtol=1e-6)
        self.assertIs(LABEL, panel.tuner.material)

    def test_a_preset_then_as_built(self) -> None:
        from robosimstudio_materials.tape import LABEL, MASKING_TAPE

        label, panel, gui = self._dragged()
        gui.find("Preset").fire("masking_tape")
        panel.drain()
        self.assertAlmostEqual(MASKING_TAPE.strength, float(label._contact["params"].numpy()[0]))
        np.testing.assert_allclose([MASKING_TAPE.peak_opening, MASKING_TAPE.reach], _law(label), rtol=1e-6)
        _slider(gui, "peak_opening").fire(1.0e-4)
        panel.drain()
        _slider(gui, "reach").fire(4.0e-4)
        panel.drain()
        gui.find("Preset").fire("(as built)")
        panel.drain()
        np.testing.assert_allclose([LABEL.peak_opening, LABEL.reach], _law(label), rtol=1e-6)

    def test_the_inspector_writes_its_edits_as_one_set_and_refuses_one_the_kernels_do(self) -> None:
        """With no Material folder on its own page it writes the tunable directly, and its binder holds the law
        it was opened on, not what was written since (here by another page's folder): a peak it takes can be
        one the kernels refuse, which is refused on the status line, the slider put back."""
        from robosimstudio_materials.tape import LABEL
        from robosimstudio_studio.inspector import PanelInspector

        label, panel, _ = self._dragged()
        session = SimpleNamespace(env=panel.session.env, server=None, selected=None, status="")
        inspector, gui = PanelInspector(session), _StubGui()
        inspector._section_parameters(gui, SimpleNamespace(entity=label))
        (binder, _, sheet), peak = inspector._binders[0], _slider(gui, "peak_opening")
        self.assertEqual("", sheet, "no folder on its page, so a direct write")
        peak.fire(8.0e-4)  # under LABEL's 1 mm reach, past the kernels' 0.4 mm
        inspector.drain()
        np.testing.assert_allclose([1.0e-4, 4.0e-4], _law(label), rtol=1e-6, err_msg="nothing written")
        self.assertIn("peak_opening not changed", session.status)
        self.assertIs(LABEL, binder.spec)
        self.assertAlmostEqual(LABEL.peak_opening, peak.value, msg="the slider goes back")
        peak.fire(1.5e-3)  # with the reach in the same drain: past the kernels' reach, under the new one
        _slider(gui, "reach").fire(2.0e-3)
        inspector.drain()
        np.testing.assert_allclose([1.5e-3, 2.0e-3], _law(label), rtol=1e-6)
        self.assertEqual("", binder.refused)


class TestALooseTapeIsOfferedOnlyWhatItCanBeBuiltWith(unittest.TestCase):
    """A loose tape needs contact adhesion, a rule only ``TapeSpec`` holds: the folder offered it
    ``paper_label``, whose opening adhesion and zero strength let every bond go, and Apply & rebuild then
    raised out of the drain."""

    def test_its_presets_are_those_its_spec_takes(self) -> None:
        _, gui, _ = _page(_tape(stuck=False))
        self.assertEqual(("(as built)", "duct_tape", "label", "masking_tape"), gui.find("Preset").options)
        _, gui, _ = _page(_tape(stuck=True))
        self.assertIn("paper_label", gui.find("Preset").options, "a stuck label rebuilds under either law")

    def test_adopting_another_is_refused_on_the_status_line(self) -> None:
        from robosimstudio_materials.tape import LABEL

        label = _tape(stuck=False)
        panel, gui, _ = _page(label)
        panel.session.status = ""
        gui.find("Preset").fire("paper_label")
        panel.drain()
        self.assertIs(LABEL, panel.tuner.material)
        self.assertAlmostEqual(LABEL.strength, float(label._contact["params"].numpy()[0]), msg="no bond let go")
        self.assertIn("only contact adhesion", panel.session.status)
        self.assertEqual("(as built)", gui.find("Preset").value)
        self.assertEqual("none", gui.find("Pending rebuild").value)

    def test_opening_adhesion_on_its_slider_is_refused_too(self) -> None:
        label = _tape(stuck=False)
        panel, gui, _ = _page(label)
        _slider(gui, "adhesion").fire("opening")
        panel.drain()
        self.assertEqual("contact", panel.tuner.material.adhesion)
        self.assertEqual("contact", _slider(gui, "adhesion").value, "the control goes back")
        self.assertIn("adhesion not changed", panel.session.status)
        self.assertEqual("none", gui.find("Pending rebuild").value)


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


def _on_show(gui) -> dict[str, object]:
    """The controls a browser shows, by field name: every widget no hidden folder holds."""

    def shown(folder) -> bool:
        while folder is not None:
            if not folder.visible:
                return False
            folder = folder.parent
        return True

    return {label.split(" ")[0]: w for label, w in gui.widgets if shown(w.parent)}


class TestEachSheetShowsItsOwnClasssFields(unittest.TestCase):
    """The folder built one set of sliders from the first sheet's material class and re-pointed it at the next.

    With the label first, the towel showed ``tack`` and ``break_opening``, and moving ``tack`` raised a
    ``TypeError`` out of the drain, ``replace(ClothMaterialSpec, tack=...)``; with the towel first, the
    label showed no adhesive at all.
    """

    def _fields(self, spec) -> set[str]:
        return {f.name for f, _ in iter_params(spec)}

    def _check(self, gui, spec) -> None:
        every = self._fields(TapeMaterialSpec)
        self.assertEqual(self._fields(spec), every & set(_on_show(gui)), type(spec).__name__)

    def test_a_towel_after_a_label(self) -> None:
        label, towel = PAPER_LABEL, ClothMaterialSpec()
        panel, gui, _ = _page(_sheet("label", label), _sheet("towel", towel))
        self._check(gui, label)
        gui.find("Cloth").fire("towel")
        panel.drain()
        self._check(gui, towel)
        _on_show(gui)["tri_ke"].fire(3.0)  # a log slider: 1000
        panel.drain()
        self.assertAlmostEqual(1000.0, panel.tuner.working["towel"].tri_ke, places=6)
        self.assertEqual(label.tri_ke, panel.tuner.working["label"].tri_ke, "the label is not the sheet picked")

    def test_a_label_after_a_towel(self) -> None:
        label, towel = PAPER_LABEL, ClothMaterialSpec()
        panel, gui, _ = _page(_sheet("towel", towel), _sheet("label", label))
        self._check(gui, towel)
        gui.find("Cloth").fire("label")
        panel.drain()
        self._check(gui, label)
        self.assertAlmostEqual(math.log10(label.tack), _on_show(gui)["tack"].value, msg="at the label's own tack")
        _on_show(gui)["tack"].fire(7.0)
        panel.drain()
        self.assertAlmostEqual(1.0e7, panel.tuner.working["label"].tack, delta=1.0)
        self.assertIs(towel, panel.tuner.working["towel"], "the towel is not the sheet picked")
        gui.find("Cloth").fire("towel")
        panel.drain()
        self._check(gui, towel)
        _on_show(gui)["tri_ke"].fire(3.0)
        panel.drain()
        self.assertAlmostEqual(1000.0, panel.tuner.working["towel"].tri_ke, places=6)
        self.assertAlmostEqual(1.0e7, panel.tuner.working["label"].tack, delta=1.0, msg="and the label kept its edit")


@dataclass(frozen=True)
class _Laid:
    """A sheet as the task lays it out: what ``Apply & rebuild`` substitutes the working material into."""

    name: str
    material: Any


class _Task(SimpleNamespace):
    def with_entities(self, **views):
        return _Task(**{**vars(self), **views})


def _both(*entities, on: int = 0):
    """One page with the Material folder and the inspector open on sheet ``on``, as the session drains them."""
    from robosimstudio_studio.inspector import PanelInspector

    panel, folder, env = _page(*entities)
    env.task = _Task(
        deformables=tuple(_Laid(e.spec.name, e.spec.material) for e in entities), physics=env.request.physics
    )
    session = panel.session
    session.tune, session.server, session.selected, session.status = panel, None, None, ""
    session.episode_id, session.num_worlds, session.load_request = 7, 1, None
    inspector, gui = PanelInspector(session), _StubGui()
    inspector._section_parameters(gui, SimpleNamespace(entity=entities[on]))
    return panel, folder, inspector, gui


def _tick(panel, inspector) -> None:
    """One loop's drains, in the session's order: the inspector is before the folder."""
    inspector.drain()
    panel.drain()


class TestAnInspectorEditIsTheFoldersEditToo(unittest.TestCase):
    """The inspector wrote a sheet's ``material`` tunable directly and never told the folder's tuner, whose
    working spec is what ``Apply & rebuild``, Revert, a preset and ``Pending rebuild`` read. Over these
    stand-ins: a towel's ``tri_ke`` set to 1000 on the inspector held in the model while the working spec and
    the folder's slider stayed at 10000, so a rebuild built it at 10000 again; ``areal_density`` moved there
    was never pending; the folder's ``edge_ke`` set to 10 left the inspector showing 0.1. The same on
    ``03_materials/08_tape``'s page: the inspector's ``tack`` at 2e6 reached the bonds while the working
    spec stayed at 4e6, its ``thickness`` was never pending, and after Revert it still showed 2e6."""

    def test_a_live_edit_survives_apply_and_rebuild(self) -> None:
        towel = _sheet("towel", ClothMaterialSpec())
        panel, folder, inspector, gui = _both(towel)
        _slider(gui, "tri_ke").fire(3.0)  # a log slider: 1000
        _tick(panel, inspector)
        self.assertTrue(np.allclose(towel.env.model.tri_materials.values[:, 0], 1000.0), "still written live")
        self.assertAlmostEqual(1000.0, panel.tuner.working["towel"].tri_ke, places=6)
        self.assertAlmostEqual(3.0, _slider(folder, "tri_ke").value, msg="the folder's slider follows")
        folder.find("Apply & rebuild").fire()
        panel.drain()
        task, episode = panel.session.load_request[:2]
        self.assertAlmostEqual(1000.0, task.deformables[0].material.tri_ke, places=6)
        self.assertEqual(7, episode)

    def test_a_rebuild_field_is_pending(self) -> None:
        panel, folder, inspector, gui = _both(_sheet("towel", ClothMaterialSpec()))
        _slider(gui, "areal_density").fire(0.45)
        _tick(panel, inspector)
        self.assertEqual(("areal_density",), panel.tuner.pending())
        self.assertEqual("areal_density", folder.find("Pending rebuild").value)

    def test_a_folder_edit_after_it_keeps_it_and_revert_puts_both_back(self) -> None:
        panel, folder, inspector, gui = _both(_sheet("towel", ClothMaterialSpec()))
        _slider(gui, "tri_ke").fire(3.0)
        _tick(panel, inspector)
        _slider(folder, "edge_ke").fire(1.0)  # 10, on the folder
        _tick(panel, inspector)
        self.assertAlmostEqual(1000.0, panel.tuner.working["towel"].tri_ke, places=6, msg="kept beside the folder's")
        self.assertAlmostEqual(10.0, panel.tuner.working["towel"].edge_ke, places=6)
        panel.revert_requested = True
        _tick(panel, inspector)
        inspector.drain()  # the next loop's first drain
        self.assertIs(panel.tuner.built["towel"], panel.tuner.working["towel"])
        self.assertAlmostEqual(math.log10(ClothMaterialSpec().tri_ke), _slider(gui, "tri_ke").value)

    def test_a_folder_edit_shows_on_the_inspector(self) -> None:
        panel, folder, inspector, gui = _both(_sheet("towel", ClothMaterialSpec()))
        _slider(folder, "edge_ke").fire(1.0)
        _tick(panel, inspector)
        inspector.drain()  # the next loop's first drain
        self.assertAlmostEqual(1.0, _slider(gui, "edge_ke").value)
        self.assertAlmostEqual(10.0, inspector._binders[0][0].spec.edge_ke, places=6)
        folder.find("Preset").fire("terry_towelling")
        _tick(panel, inspector)
        inspector.drain()
        self.assertIs(panel.tuner.working["towel"], inspector._binders[0][0].spec)

    def test_an_edit_pending_on_the_inspector_lands_on_the_folders(self) -> None:
        """A folder edit between two of the inspector's drains: following it keeps the inspector's control."""
        panel, folder, inspector, gui = _both(_sheet("towel", ClothMaterialSpec()))
        _slider(folder, "edge_ke").fire(1.0)
        panel.drain()
        _slider(gui, "tri_ke").fire(3.0)
        inspector.drain()
        self.assertAlmostEqual(1000.0, panel.tuner.working["towel"].tri_ke, places=6)
        self.assertAlmostEqual(10.0, panel.tuner.working["towel"].edge_ke, places=6, msg="the folder's edit stays")
        self.assertAlmostEqual(3.0, _slider(gui, "tri_ke").value)
        self.assertAlmostEqual(1.0, _slider(gui, "edge_ke").value)

    def test_a_sheet_the_folder_is_not_showing(self) -> None:
        """The inspector's subject need not be the folder's selected sheet; the folder's controls stay on its own."""
        towel = _sheet("towel", ClothMaterialSpec())
        panel, _, inspector, gui = _both(towel, _sheet("label", PAPER_LABEL), on=1)
        _slider(gui, "tack").fire(7.0)
        _tick(panel, inspector)
        self.assertAlmostEqual(1.0e7, panel.tuner.working["label"].tack, delta=1.0)
        self.assertIs(towel.spec.material, panel.tuner.working["towel"], "the towel is not the sheet edited")
        self.assertIs(panel.tuner.material, panel.binder.spec, "the folder still shows the towel")

    def test_one_the_sheet_refuses_is_refused_on_the_status_line(self) -> None:
        """A loose tape's spec refuses opening adhesion, which only the folder's tuner checks."""
        from robosimstudio_materials.tape import LABEL

        panel, folder, inspector, gui = _both(_tape(stuck=False))
        _slider(gui, "adhesion").fire("opening")
        _tick(panel, inspector)
        self.assertIs(LABEL, panel.tuner.working["label"])
        self.assertEqual("contact", _slider(gui, "adhesion").value, "the control goes back")
        self.assertIn("adhesion not changed", panel.session.status)
        self.assertEqual("none", folder.find("Pending rebuild").value)

    def test_the_inspector_checks_a_peak_against_the_reach_the_folder_dragged(self) -> None:
        """The stale binder: it held LABEL's 1 mm reach after the folder dragged it to 0.4 mm, took a 0.8 mm
        peak and left the kernels to refuse it. Following the folder, its own spec refuses it."""
        label = _tape(stuck=False)
        panel, folder, inspector, gui = _both(label)
        _slider(folder, "peak_opening").fire(1.0e-4)
        _tick(panel, inspector)
        _slider(folder, "reach").fire(4.0e-4)
        _tick(panel, inspector)
        inspector.drain()
        self.assertAlmostEqual(4.0e-4, _slider(gui, "reach").value)
        _slider(gui, "peak_opening").fire(8.0e-4)
        _tick(panel, inspector)
        np.testing.assert_allclose([1.0e-4, 4.0e-4], _law(label), rtol=1e-6, err_msg="nothing written")
        self.assertIn("peak_opening not changed", panel.session.status)
        self.assertAlmostEqual(1.0e-4, _slider(gui, "peak_opening").value)


if __name__ == "__main__":
    unittest.main()
