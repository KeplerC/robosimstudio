# SPDX-License-Identifier: Apache-2.0
"""``robosimstudio_materials/tape.py``: the sheet a tape is built as, what it bonds, and what it declares.

No GPU: nothing is built or stepped. The tape's physics -- a hand peeling a label off a box by its
pull tab, and a label that never lets go staying stuck -- is driven through the example's page by
``tests/test_peel_a_label.py``. Here:

* **The surface.** The sheet's first ``tab`` metres stand straight up at the fold, the rest lies
  flat; the bonded vertices are the flat ones past the fold row, the clamped ones the upper half of
  the turned-up end, and the two never overlap.
* **The break law**, on positions written by hand: only lifting off the face opens a bond; sliding
  along it and pressing into it do not.
* **The declarations**: the spec refuses what cannot be built, the presets carry the adhesive and
  stay soft enough for a tab to pull them, the contact presets share one sheet and rank by strength,
  ``TAPE_60HZ`` puts every free body on the tape's solver with compliant ALM, the flat names resolve,
  and the atoms, record, materials and domain register. The contact law itself is checked against
  ipc-toolkit by ``tests/test_tape_adhesion_law.py``.
"""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np

TIER = "fast"
"""Numpy and the spec classes only."""


class TestTheSurface(unittest.TestCase):
    """A flat sheet whose end stands up into the tab."""

    def test_the_end_stands_straight_up_at_the_fold(self) -> None:
        from robosimstudio_materials.tape import tape_surface

        vertices, faces, up = tape_surface((0.04, 0.095), (8, 19), 0.015)
        self.assertEqual(((8 + 1) * (19 + 1), 3), vertices.shape)
        self.assertEqual(2 * 8 * 19, len(faces))
        self.assertEqual(3 * 9, int(up.sum()), "three 5 mm rows turned up")
        np.testing.assert_allclose(vertices[up, 1], 0.015, err_msg="the turned-up end stands over the fold")
        np.testing.assert_allclose(sorted(set(np.round(vertices[up, 2], 6))), [0.005, 0.010, 0.015])
        self.assertTrue(np.all(vertices[~up, 2] == 0.0), "the rest lies flat")
        tri = vertices[faces]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        self.assertTrue(np.all(area > 1.0e-7), "no triangle folds flat onto its neighbour")

    def test_the_bonds_and_the_clamps(self) -> None:
        from robosimstudio_materials.tape import bonded_rows, clamped_rows, tape_surface

        vertices, _, _ = tape_surface((0.04, 0.095), (8, 19), 0.015)
        bonded, clamped = bonded_rows(vertices, 0.015, 0.005), clamped_rows(vertices, 0.015)
        self.assertEqual(16 * 9, len(bonded), "every flat row past the fold row")
        self.assertTrue(np.all(vertices[bonded, 1] >= 0.02 - 1.0e-9), "the fold row is free, so the end can lift")
        self.assertEqual(2 * 9, len(clamped), "the turned-up end's upper two rows")
        self.assertFalse(set(bonded) & set(clamped))

    def test_a_dry_margin_leaves_the_rows_past_the_fold_unbonded(self) -> None:
        from robosimstudio_materials.tape import bonded_rows, tape_surface

        vertices, _, _ = tape_surface((0.04, 0.095), (8, 19), 0.015)
        dry = bonded_rows(vertices, 0.015, 0.005, 0.012)
        self.assertEqual(14 * 9, len(dry), "the rows 5 and 10 mm past the fold carry no adhesive")
        self.assertGreaterEqual(float(vertices[dry, 1].min()), 0.015 + 0.012)
        self.assertTrue(set(dry) <= set(bonded_rows(vertices, 0.015, 0.005)))

    def test_each_vertex_carries_its_share_of_the_sheet(self) -> None:
        from robosimstudio_materials.tape import sheet_areas

        areas = sheet_areas((0.04, 0.095), (8, 19))
        self.assertAlmostEqual(0.04 * 0.095, float(areas.sum()), places=12)
        self.assertAlmostEqual(25.0e-6, float(np.median(areas)), places=12, msg="an inner vertex: one 5 mm cell")

    def test_without_a_tab_the_whole_sheet_is_stuck(self) -> None:
        from robosimstudio_materials.tape import bonded_rows, clamped_rows, tape_surface

        vertices, _, up = tape_surface((0.02, 0.05), (4, 10), 0.0)
        self.assertFalse(up.any())
        self.assertEqual(len(vertices), len(bonded_rows(vertices, 0.0, 0.005)))
        self.assertEqual(0, len(clamped_rows(vertices, 0.0)))


class TestTheBreakLaw(unittest.TestCase):
    """A bond opens along the face's normal only."""

    def test_only_lifting_off_the_face_opens_a_bond(self) -> None:
        from robosimstudio_materials.tape import opening

        anchors = np.zeros((4, 3))
        normals = np.tile([0.0, 0.0, 1.0], (4, 1))
        points = np.array([[0.0, 0.0, 0.002], [0.005, 0.0, 0.0], [0.0, 0.0, -0.001], [0.003, 0.004, 0.0015]])
        np.testing.assert_allclose(opening(points, anchors, normals), [0.002, 0.0, -0.001, 0.0015])

    def test_the_normal_is_the_faces_wherever_the_body_is_turned(self) -> None:
        from robosimstudio_materials.tape import opening

        side = np.array([[1.0, 0.0, 0.0]])
        self.assertAlmostEqual(0.002, float(opening([[0.502, 0.0, 0.3]], [[0.5, 0.0, 0.3]], side)[0]))
        self.assertAlmostEqual(0.0, float(opening([[0.5, 0.0, 0.31]], [[0.5, 0.0, 0.3]], side)[0]))


class TestTheSpec(unittest.TestCase):
    def test_the_constructor_places_the_corner_and_takes_the_label_by_default(self) -> None:
        from robosimstudio_materials.tape import PAPER_LABEL, TapeSpec, tape

        spec = tape(name="label", on="box", at=(0.02, -0.025, 0.0505))
        self.assertIsInstance(spec, TapeSpec)
        self.assertEqual((0.02, -0.025, 0.0505), spec.pose)
        self.assertIs(PAPER_LABEL, spec.material)
        self.assertEqual("tape", spec.entity)
        self.assertEqual((0.005, 0.005), spec.cell)
        self.assertEqual(0, spec.settle_frames, "snapshotted where it was built, as a reset puts the body back")

    def test_validation_refuses_what_cannot_be_built(self) -> None:
        from robosimstudio_materials.cloth import ClothMaterialSpec
        from robosimstudio_materials.tape import PAPER_LABEL, TapeMaterialSpec, tape

        with self.assertRaisesRegex(ValueError, "TapeMaterialSpec"):
            tape(name="t", on="box", material=ClothMaterialSpec())
        with self.assertRaisesRegex(ValueError, "the sheet's length"):
            tape(name="t", on="box", size=(0.02, 0.01), tab=0.015)
        with self.assertRaisesRegex(ValueError, "must hold"):
            tape(name="t", on="box", tab=0.03, tab_size=(0.044, 0.008, 0.02))
        with self.assertRaises(ValueError):
            dataclasses.replace(PAPER_LABEL, break_opening=-1.0)
        with self.assertRaises(ValueError):
            TapeMaterialSpec(tack=-1.0)

    def test_the_adhesive_fields_are_live(self) -> None:
        from robosimstudio.specs.params import LIVE
        from robosimstudio_materials.tape import ADHESIVE_FIELDS, TapeMaterialSpec

        fields = {f.name: f for f in dataclasses.fields(TapeMaterialSpec)}
        for name in ADHESIVE_FIELDS:
            with self.subTest(field=name):
                self.assertEqual(LIVE, fields[name].metadata["ui"].tier)


class TestThePresets(unittest.TestCase):
    def test_each_preset_is_soft_enough_for_its_tab_to_pull_it(self) -> None:
        # Paper's edge_ke 30 and tri_ke 1e4 held the tab still under the hand's pull (the module's section 1).
        from robosimstudio_materials.tape import tapes

        for name, material in vars(tapes).items():
            with self.subTest(preset=name):
                self.assertLessEqual(material.edge_ke, 0.1)
                self.assertLessEqual(material.tri_ke, 1.0e3)
                self.assertFalse(material.zero_rest_angle, "the built shape, end turned up, is the rest shape")
                self.assertGreater(material.tack, 0.0)
                self.assertGreater(material.break_opening, 0.0)

    def test_the_contact_presets_share_a_sheet_and_rank_by_strength(self) -> None:
        from robosimstudio_materials.tape import CONTACT_FIELDS, DUCT_TAPE, LABEL, MASKING_TAPE, PAPER_LABEL

        self.assertEqual("opening", PAPER_LABEL.adhesion)
        for material in (LABEL, MASKING_TAPE, DUCT_TAPE):
            with self.subTest(strength=material.strength):
                self.assertEqual("contact", material.adhesion)
                self.assertEqual(PAPER_LABEL.tri_ke, material.tri_ke)
                self.assertEqual(PAPER_LABEL.edge_ke, material.edge_ke)
                self.assertTrue(
                    all(getattr(material, f) == getattr(LABEL, f) for f in CONTACT_FIELDS if f != "strength")
                )
        self.assertLess(LABEL.strength, MASKING_TAPE.strength)
        self.assertLess(MASKING_TAPE.strength, DUCT_TAPE.strength)

    def test_a_label_bond_starts_near_the_opening_laws_stiffness(self) -> None:
        # The ipc pull's slope at contact on a 5 mm cell, within a factor of two of PAPER_LABEL's 100 N/m.
        from robosimstudio_materials.tape import LABEL, contact_stiffness

        per_bond = contact_stiffness(LABEL, LABEL.particle_radius) * 0.005 * 0.005
        self.assertGreater(per_bond, 50.0)
        self.assertLess(per_bond, 200.0)

    def test_a_label_row_lets_go_at_under_a_newton(self) -> None:
        from robosimstudio_materials.tape import PAPER_LABEL

        per_bond = PAPER_LABEL.tack * 0.005 * 0.005 * PAPER_LABEL.break_opening
        self.assertAlmostEqual(0.1, per_bond)
        self.assertAlmostEqual(0.9, 9 * per_bond, msg="the pads hold the tab with about 3 N")


class TestTheArrangement(unittest.TestCase):
    def test_free_bodies_share_the_tapes_solver_with_compliant_alm(self) -> None:
        from robosimstudio.specs import GENERIC
        from robosimstudio_materials.tape import TAPE_60HZ

        vbd = TAPE_60HZ.backend_for("*")
        self.assertEqual("vbd", vbd.solver)
        self.assertTrue(vbd.get("rigid_compliant_alm"))
        self.assertEqual(256, vbd.get("rigid_body_contact_buffer_size"))
        self.assertEqual(vbd, TAPE_60HZ.backend_for("tape"), "one island, or Newton drops the bonds")
        self.assertEqual(GENERIC.substeps, TAPE_60HZ.substeps)
        self.assertEqual(GENERIC.coupling, TAPE_60HZ.coupling)
        self.assertEqual(GENERIC.backend_for("articulated"), TAPE_60HZ.backend_for("articulated"))


class TestTheDeclarations(unittest.TestCase):
    def test_the_flat_names_resolve(self) -> None:
        import robosimstudio as rss
        from robosimstudio_materials import tape as module

        for name in ("TAPE_60HZ", "TapeMaterialSpec", "TapeSpec", "tape", "tapes"):
            with self.subTest(name=name):
                self.assertIs(getattr(module, name), getattr(rss, name))

    def test_the_domain_the_entity_and_the_materials_register(self) -> None:
        from robosimstudio.assetlib.materials import get_material, kind_of
        from robosimstudio.scene.registry import ENTITIES
        from robosimstudio.specs.physics import get_domain
        from robosimstudio_materials.tape import PAPER_LABEL, EntityTape

        domain = get_domain("tape")
        self.assertEqual(("tape", "cloth"), domain.reads)
        self.assertEqual("deformables", domain.role)
        self.assertIs(EntityTape, ENTITIES.get("tape"))
        self.assertIs(PAPER_LABEL, get_material("paper_label").spec)
        self.assertEqual("tape", kind_of(PAPER_LABEL))
        from robosimstudio_materials.tape import DUCT_TAPE, LABEL, MASKING_TAPE

        for name, spec in (("label", LABEL), ("masking_tape", MASKING_TAPE), ("duct_tape", DUCT_TAPE)):
            self.assertIs(spec, get_material(name).spec)

    def test_a_record_carries_the_domain_and_builds_the_spec(self) -> None:
        import robosimstudio as rss
        from robosimstudio_materials.tape import Tape

        spec = rss.tape(name="label", on="box")
        record = Tape.from_spec(spec, name="paper_label")
        self.assertEqual("tape", record.sim.domain)
        self.assertIs(spec, record.build())
        with self.assertRaises(TypeError):
            Tape.from_spec(rss.box(name="cube"), name="not_a_tape")

    def test_the_atom_is_catalogued_on_the_tapes_layout(self) -> None:
        from robosimstudio.assetlib.semantics.atoms import ATOMS
        from robosimstudio.assetlib.semantics.handlers import dispatch

        spec = ATOMS.get("peeled_fraction")
        self.assertEqual(("tape",), spec.args[0].substrates)
        self.assertEqual(("fraction", "front"), spec.returns)
        self.assertEqual([], spec.problems())
        self.assertEqual("peeled_fraction", dispatch("peeled_fraction", ("",), ("tape",)).atom)

    def test_the_bonded_atom_is_catalogued_on_the_tapes_layout(self) -> None:
        from robosimstudio.assetlib.semantics.atoms import ATOMS
        from robosimstudio.assetlib.semantics.handlers import dispatch

        spec = ATOMS.get("bonded_fraction")
        self.assertEqual(("tape",), spec.args[0].substrates)
        self.assertEqual(("fraction", "held"), spec.returns)
        self.assertEqual([], spec.problems())
        self.assertEqual("bonded_fraction", dispatch("bonded_fraction", ("",), ("tape",)).atom)

    def test_the_bonded_handler_reads_the_bonds_holding(self) -> None:
        from types import SimpleNamespace

        from robosimstudio_materials.tape import _bonded_fraction

        entity = SimpleNamespace(bonded=lambda: 0.75, held=lambda: 3)
        got = _bonded_fraction(SimpleNamespace(entities={"label": entity}), "label")
        self.assertEqual({"fraction": 0.75, "held": 3.0, "label/bonded_fraction": 0.75}, got)
        with self.assertRaises(KeyError):
            _bonded_fraction(SimpleNamespace(entities={}), "label")

    def test_the_handler_reads_the_entitys_bonds(self) -> None:
        from types import SimpleNamespace

        from robosimstudio_materials.tape import _peeled_fraction

        entity = SimpleNamespace(peeled=lambda: 0.25, peel_front=lambda: 0.02)
        got = _peeled_fraction(SimpleNamespace(entities={"label": entity}), "label")
        self.assertEqual({"fraction": 0.25, "front": 0.02, "label/peeled_fraction": 0.25}, got)
        with self.assertRaises(KeyError):
            _peeled_fraction(SimpleNamespace(entities={}), "label")


if __name__ == "__main__":
    unittest.main()
