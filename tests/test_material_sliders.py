# SPDX-License-Identifier: Apache-2.0
"""Every material slider a page shows reaches the array its solver, or its kind's kernel, reads.

A slider can move, change the working spec and leave the simulation exactly as it was. Round 3 of
the phase 4 review found it twice over: on the paper fold's page the ``Material`` folder's
``yield_angle`` wrote nothing the crease kernel reads, and in a coupled Newton 1.7 scene an island
may keep its own copy of a model array, so a write into the model moved nothing there either (the
tape's ``tri_materials``, the rod's Dahl pair). Each test here builds an example, writes each field
the way the page does -- the folder's widgets, or the inspector's writer -- and reads the value back
from the island view the field's solver steps in, or from the kind's own kernel array. A field the
page shows that no test reads fails :meth:`TestTheFoldPagesMaterialFolder.test_no_field_goes_unread`,
and the fields that cannot be live are listed in :data:`NOT_LIVE` with the reason.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from robosimstudio.specs.params import LIVE, REBUILD, RECAPTURE, param_ui
from tests.fixtures.examples import _run
from tests.fixtures.grasp import tick

TIER = "medium"
"""One GPU: five examples built through their pages."""

PORT = 9801

NOT_LIVE = {
    "core_shear_modulus": "a slab's core; a membrane has no tetrahedra, so the field is read-only there",
    "core_lame_lambda": "a slab's core; read-only on a membrane",
    "core_damping": "a slab's core; read-only on a membrane",
    "areal_density": "REBUILD: became particle mass at build",
    "zero_rest_angle": "REBUILD: the rest angles a build starts from",
    "shell": "REBUILD: which engine builds the sheet",
    "thickness": "REBUILD: the slab or shell built from it",
    "shell_young": "REBUILD: a host engine copies it when it builds the shell",
    "shell_poisson": "REBUILD: a host engine copies it when it builds the shell",
}
"""What a cloth-like ``Material`` folder shows that no write can move while the scene runs."""


def _open(rel: str, port: int, panels=None):
    from robosimstudio_studio.session import StudioConfig, StudioSession

    sim, kwargs = _run(rel)
    shown = tuple(panels or kwargs.get("panels") or ("inspector", "cloth", "tune"))
    session = StudioSession(sim, config=StudioConfig(port=port, panels=shown, speed="max", live=False))
    if kwargs.get("page") is not None:
        kwargs["page"](session)
    session.begin_run()
    tick(session, 3)
    return sim, session


def _tune(session):
    return next(p for p in session.panels if type(p).__name__ == "PanelTune")


def _slide(session, name: str, value: float) -> None:
    """Set one ``Material`` slider the way a browser does, then let the page drain it."""
    binder = _tune(session).binder
    widget, described = binder._widgets[name], binder._ui[name]
    widget.value = math.log10(value) if described.log else value
    tick(session, 1)


def _new_value(spec, name: str) -> float:
    """A value the field does not hold yet, inside its slider's range."""
    described = param_ui(spec, name)
    current = float(getattr(spec, name))
    if described.range is None:
        return current * 1.37 if current else 0.1
    low, high = described.range
    value = current * 1.37 if current > 0 else 0.5 * (low + high)
    return value if low <= value <= high else 0.63 * current if current > 0 else 0.5 * (low + high)


def _cloth_readers(view, target):
    """The cloth fields, read from the island view the cloth's solver steps in."""

    def rows(attr, index, column=None):
        values = getattr(view, attr).numpy()
        return values[index] if column is None else values[index, column]

    return {
        "tri_ke": lambda: rows("tri_materials", target.tris, 0),
        "tri_ka": lambda: rows("tri_materials", target.tris, 1),
        "tri_kd": lambda: rows("tri_materials", target.tris, 2),
        "edge_ke": lambda: rows("edge_bending_properties", target.edges, 0),
        "edge_kd": lambda: rows("edge_bending_properties", target.edges, 1),
        "particle_radius": lambda: rows("particle_radius", target.particles),
        "friction": lambda: np.array([view.soft_contact_mu]),
        "contact_ke": lambda: np.array([view.soft_contact_ke]),
        "contact_kd": lambda: np.array([view.soft_contact_kd]),
    }


class TestTheFoldPagesMaterialFolder(unittest.TestCase):
    """``examples/03_materials/05_paper``'s own panels: every enabled slider, through the folder."""

    @classmethod
    def setUpClass(cls) -> None:
        from robosimstudio.scene.tuning import locate_cloth

        cls.sim, cls.session = _open("03_materials/05_paper.py", PORT)
        paper = cls.sim["paper"].entity
        cls.paper = paper
        view = cls.sim.scene.program.view_for("paper")
        cls.readers = _cloth_readers(view, locate_cloth(cls.sim.model, paper))
        slots = {"yield_angle": 0, "plastic_rate": 1, "plastic_recovery": 2}
        cls.readers.update({name: (lambda k=k: paper._params.numpy()[[k]]) for name, k in slots.items()})
        cls.binder = _tune(cls.session).binder

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def _enabled(self) -> list[str]:
        return [n for n, w in self.binder._widgets.items() if not w.disabled]

    def test_no_field_goes_unread(self) -> None:
        for name in self._enabled():
            tier = self.binder._ui[name].tier
            if tier == REBUILD:
                self.assertIn(name, NOT_LIVE, f"{name} is a rebuild field with no reason given")
                continue
            self.assertIn(name, self.readers, f"{name} is shown {tier} and nothing here reads where it lands")
        for name, widget in self.binder._widgets.items():
            if widget.disabled:
                self.assertIn(name, NOT_LIVE, f"{name} is shown read-only with no reason given")

    def test_every_live_and_recapture_slider_lands(self) -> None:
        spec = self.binder.spec
        for name in self._enabled():
            if self.binder._ui[name].tier not in (LIVE, RECAPTURE):
                continue
            with self.subTest(name):
                value = _new_value(spec, name)
                _slide(self.session, name, value)
                np.testing.assert_allclose(self.readers[name](), value, rtol=2e-3, err_msg=name)

    def test_yield_angle_at_zero_reaches_the_crease_kernel(self) -> None:
        """The verifier's case: the folder's yield_angle to 0, read where the crease kernel reads it."""
        _slide(self.session, "yield_angle", 0.0)
        self.assertEqual(0.0, float(self.paper._params.numpy()[0]))


class TestTheTapeOnAnIslandWithItsOwnCopy(unittest.TestCase):
    """``examples/03_materials/08_tape``: the label's VBD island keeps its own copy of the cloth arrays."""

    @classmethod
    def setUpClass(cls) -> None:
        from robosimstudio.scene.tuning import locate_cloth

        cls.sim, cls.session = _open("03_materials/08_tape.py", PORT + 1, ("inspector", "cloth", "tune"))
        cls.label = cls.sim["label"].entity
        cls.view = cls.sim.scene.program.view_for("label")
        cls.readers = _cloth_readers(cls.view, locate_cloth(cls.sim.model, cls.label))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_cloth_fields_reach_the_islands_copy(self) -> None:
        spec = _tune(self.session).binder.spec
        for name in ("tri_ke", "edge_ke", "particle_radius"):
            with self.subTest(name):
                self.assertEqual(self.view.tri_materials.shape, self.sim.model.tri_materials.shape, "a compacted copy")
                value = _new_value(spec, name)
                _slide(self.session, name, value)
                np.testing.assert_allclose(self.readers[name](), value, rtol=2e-3)

    def test_the_adhesive_reaches_the_bonds_and_the_break_kernel(self) -> None:
        across, along = self.label.spec.cell
        _slide(self.session, "tack", 2.0e6)
        stiffness = self.view.attachment_particle_body_stiffness.numpy()[self.label.bonds]
        np.testing.assert_allclose(stiffness, 2.0e6 * across * along, rtol=2e-3)
        _slide(self.session, "break_opening", 0.002)
        self.assertAlmostEqual(0.002, float(self.label._limit.numpy()[0]), places=6)

    def test_the_inspectors_tack_is_the_folders_too(self) -> None:
        """The inspector on the label writes through the folder: the bonds take it and so does the working
        spec a rebuild reads, which on main stayed at LABEL's 4e6 while the bonds took 2e6."""
        session = self.session
        session.select(session.index.by_name("label"))
        tick(session, 2)
        binder, _, sheet = next(row for row in session.inspector._binders if row[1].name == "material")
        self.assertEqual("label", sheet)
        binder._widgets["tack"].value = math.log10(3.0e6)
        tick(session, 1)
        across, along = self.label.spec.cell
        stiffness = self.view.attachment_particle_body_stiffness.numpy()[self.label.bonds]
        np.testing.assert_allclose(stiffness, 3.0e6 * across * along, rtol=2e-3)
        tune = _tune(session)
        self.assertAlmostEqual(3.0e6, tune.tuner.working["label"].tack, delta=10.0)
        self.assertAlmostEqual(math.log10(3.0e6), tune.binder._widgets["tack"].value, places=6)

    def test_the_contact_laws_fields_are_read_only(self) -> None:
        """Opening adhesion runs no contact kernel, so nothing reads strength, reach and the rest."""
        from robosimstudio_materials.tape import CONTACT_FIELDS

        widgets = _tune(self.session).binder._widgets
        for name in CONTACT_FIELDS:
            with self.subTest(name):
                self.assertTrue(widgets[name].disabled, f"{name} is a slider on a label no contact kernel reads")
        self.assertFalse(widgets["tack"].disabled)

    def test_a_contact_thickness_reaches_the_margin_the_box_collides_with(self) -> None:
        box = self.sim["box"].entity
        contact = next(t for t in box.tunables(self.sim.pipeline) if t.name == "contact")
        self.assertTrue(contact.write("thickness", 0.003))
        shapes = np.asarray(list(contact.write.shapes), dtype=np.int64)
        view = self.sim.scene.program.view_for("box")
        np.testing.assert_allclose(view.shape_margin.numpy()[shapes], 0.003, rtol=1e-6)


class TestTheStuckLabelsContactAdhesive(unittest.TestCase):
    """``examples/03_materials/09_adhesion``: the contact law's fields reach the array its kernels read."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim, cls.session = _open("03_materials/09_adhesion.py", PORT + 4, ("inspector", "cloth", "tune"))
        cls.label = cls.sim["label"].entity
        cls.binder = _tune(cls.session).binder

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_every_contact_field_lands_in_the_kernels_parameters(self) -> None:
        from robosimstudio_materials.tape import CONTACT_FIELDS

        for k, name in enumerate(CONTACT_FIELDS):
            with self.subTest(name):
                self.assertFalse(self.binder._widgets[name].disabled, name)
                value = _new_value(self.binder.spec, name)
                _slide(self.session, name, value)
                np.testing.assert_allclose(self.label._contact["params"].numpy()[k], value, rtol=2e-3, err_msg=name)

    def test_the_opening_laws_fields_are_read_only(self) -> None:
        """Contact adhesion sets each bond's spring itself and runs no break kernel."""
        for name in ("tack", "break_opening"):
            with self.subTest(name):
                self.assertTrue(self.binder._widgets[name].disabled, f"{name} is a slider nothing here reads")
        (material,) = self.label.tunables(self.sim.pipeline)
        self.assertFalse(material.write("tack", 2.0e6))


class TestTheRodsDahlPair(unittest.TestCase):
    """``examples/03_materials/01_rope_memory``: the Dahl pair is per cable joint, and VBD reads its own copy."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim, cls.session = _open("03_materials/01_rope_memory.py", PORT + 2, ("inspector",))
        cls.rod = cls.sim["cable"].entity
        cls.drive = next(t for t in cls.rod.tunables(cls.sim.pipeline) if t.name == "drive")
        cls.view = cls.sim.scene.program.view_for("cable")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_both_fields_reach_every_cable_joint_in_the_islands_copy(self) -> None:
        joints = len(self.rod._cable_joints(self.sim.model))
        for name, value in (("dahl_eps_max", 0.0123), ("dahl_tau", 0.0456)):
            with self.subTest(name):
                self.assertTrue(self.drive.write(name, value))
                held = getattr(self.view.vbd, name).numpy()
                self.assertEqual(joints, int(np.isclose(held, value).sum()), f"{name}: {np.round(held, 4)}")


class TestABagOffersNothingItCannotMove(unittest.TestCase):
    """``examples/03_materials/06_bag``: SuperDex copies the shell's material when it builds it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim, cls.session = _open("03_materials/06_bag.py", PORT + 3, ("inspector", "cloth", "tune"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_no_material_folder_and_no_inspector_material(self) -> None:
        self.assertFalse(_tune(self.session).tuner, "a Material folder over a shell nothing here can write")
        self.assertEqual((), self.sim["bag"].entity.tunables(self.sim.pipeline))


if __name__ == "__main__":
    unittest.main()
