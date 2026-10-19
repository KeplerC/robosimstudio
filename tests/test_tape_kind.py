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

    def test_without_a_tab_a_dry_margin_runs_from_the_sheets_end(self) -> None:
        from robosimstudio_materials.tape import bonded_rows, tape, tape_surface

        vertices, _, _ = tape_surface((0.02, 0.05), (4, 10), 0.0)
        dry = bonded_rows(vertices, 0.0, 0.005, 0.012)
        self.assertEqual(8 * 5, len(dry), "the rows 0, 5 and 10 mm from the end carry no adhesive")
        self.assertGreaterEqual(float(vertices[dry, 1].min()), 0.012)
        with self.assertRaisesRegex(ValueError, "leaves nothing to stick"):
            tape(name="t", on="box", size=(0.02, 0.05), cells=(4, 10), tab=0.0, dry=0.05)


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


class TestTheContactLawsLiveWrites(unittest.TestCase):
    """A live write keeps the law the spec allows: a peak at or past the reach is refused, unwritten."""

    def setUp(self) -> None:
        import warp as wp

        from robosimstudio_materials.tape import CONTACT_FIELDS, LABEL, EntityTape, tape

        self.label = EntityTape(tape(name="label", on="box", material=LABEL))
        built = np.array([getattr(LABEL, f) for f in CONTACT_FIELDS], dtype=np.float32)
        self.label._contact = {"params": wp.array(built, dtype=float, device="cpu")}
        self.built = built.copy()

    def params(self) -> np.ndarray:
        return self.label._contact["params"].numpy()

    def test_a_peak_at_or_past_the_reach_is_refused_and_not_written(self) -> None:
        # At peak == reach the law's spring is strength / 0 times 0, NaN in every held bond.
        for name, value in (("reach", 5.0e-4), ("peak_opening", 1.0e-3), ("peak_opening", 2.0e-3), ("reach", 1.0e-4)):
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ValueError, "peak_opening < reach"):
                    self.label.write_adhesive(name, value)
                np.testing.assert_array_equal(self.built, self.params())
        with self.assertRaisesRegex(ValueError, "stiction_speed"):
            self.label.write_adhesive("stiction_speed", 0.0)
        np.testing.assert_array_equal(self.built, self.params())

    def test_the_reach_then_the_peak_moves_both_past_where_they_were(self) -> None:
        self.assertTrue(self.label.write_adhesive("reach", 2.0e-3))
        self.assertTrue(self.label.write_adhesive("peak_opening", 1.5e-3))
        np.testing.assert_allclose([1.5e-3, 2.0e-3], self.params()[1:3], rtol=1e-6)

    def test_a_set_written_together_moves_the_peak_past_the_reach_the_kernels_held(self) -> None:
        """What a whole material's write takes: one field at a time, the peak would land past the old reach."""
        self.assertTrue(self.label.write_adhesive_all({"peak_opening": 1.0e-4, "reach": 4.0e-4}))
        with self.assertRaisesRegex(ValueError, "peak_opening < reach"):
            self.label.write_adhesive("peak_opening", 5.0e-4)
        self.assertTrue(self.label.write_adhesive_all({"strength": 1.0e4, "peak_opening": 5.0e-4, "reach": 1.0e-3}))
        np.testing.assert_allclose([1.0e4, 5.0e-4, 1.0e-3], self.params()[:3], rtol=1e-6)
        with self.assertRaisesRegex(ValueError, "peak_opening < reach"):
            self.label.write_adhesive_all({"strength": 2.0e4, "peak_opening": 2.0e-3})
        self.assertAlmostEqual(1.0e4, float(self.params()[0]), msg="a set refused writes none of it")

    def test_a_write_before_the_kernels_are_bound_says_it_did_not_land(self) -> None:
        """The material tunable passes the write's word on, so a panel does not show a field it never moved."""
        from unittest import mock

        from robosimstudio.core.tuning import Tunable
        from robosimstudio_materials.cloth import EntityCloth
        from robosimstudio_materials.tape import LABEL

        self.label.contact, self.label.bonds = True, np.arange(126, dtype=np.int32)
        cloth = Tunable(name="material", spec=LABEL, write=lambda name, value: True)
        with mock.patch.object(EntityCloth, "tunables", return_value=(cloth,)):
            (material,) = self.label.tunables()
        self.assertTrue(material.write("strength", 1.0e3))
        self.assertAlmostEqual(1.0e3, float(self.params()[0]))
        self.label._contact = {}
        self.assertFalse(self.label.write_adhesive("strength", 2.0e3))
        self.assertFalse(material.write("strength", 2.0e3))


class TestThePressThatMakesABond(unittest.TestCase):
    """The stick kernel reads a press as VBD's penalty for the pair: the mean of the sheet's and the box's ke."""

    def _bonds(self, depth: float, box_ke: float) -> bool:
        """One loose bond over a 100 x 120 x 50 mm box, its particle ``depth`` into the contact, on the CPU."""
        from types import SimpleNamespace

        import warp as wp

        from robosimstudio_materials.tape import CONTACT_FIELDS, LABEL, EntityTape, tape

        label = EntityTape(tape(name="label", on="box", material=LABEL, stuck=False))
        label.body, label.contact = 0, True
        label.bonds, label.bond_particles = np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32)
        one = lambda value, dtype: wp.array([value], dtype=dtype, device="cpu")  # noqa: E731
        identity = wp.transform_identity()
        model = SimpleNamespace(
            shape_transform=one(identity, wp.transform),
            shape_scale=one(wp.vec3(0.05, 0.06, 0.025), wp.vec3),
            shape_margin=one(0.0, float),
            particle_radius=one(5.0e-4, float),
            soft_contact_ke=1.0e4,
            shape_material_ke=one(box_ke, float),
            body_com=one(wp.vec3(0.0), wp.vec3),
        )
        label._arrays = {
            "enabled": [one(False, wp.bool)],
            "body_point": [one(wp.vec3(0.0), wp.vec3)],
            "stiffness": [one(0.0, float)],
        }
        label._rows, label._particle, label.broken = one(0, wp.int32), one(0, wp.int32), one(0, wp.int32)
        params = [8.0e3 if f == "press" else float(getattr(LABEL, f)) for f in CONTACT_FIELDS]
        label._contact = {
            "model": model,
            "boxes": one(0, wp.int32),
            "area": one(2.5e-5, float),
            "normal": one(wp.vec3(0.0, 0.0, 1.0), wp.vec3),
            "dmin": one(5.0e-4, float),
            "params": wp.array(params, dtype=float, device="cpu"),
        }
        state = SimpleNamespace(
            particle_q=one(wp.vec3(0.0, 0.0, 0.025 + 5.0e-4 - depth), wp.vec3),
            particle_qd=one(wp.vec3(0.0), wp.vec3),
            body_q=one(identity, wp.transform),
            body_qd=one(wp.spatial_vector(), wp.spatial_vector),
        )
        label._stick(state, 1.0 / 960.0)
        return bool(label._arrays["enabled"][0].numpy()[0])

    def test_a_soft_box_takes_a_deeper_press(self) -> None:
        # 8 kPa on a 5 mm cell is 0.2 N; the pair's 6.25 kN/m reaches it at 32 um, the sheet's 10 kN/m at 20.
        self.assertFalse(self._bonds(0.9 * 0.2 / 6250.0, 2.5e3), "0.18 N as VBD pushes back: no bond")
        self.assertTrue(self._bonds(1.1 * 0.2 / 6250.0, 2.5e3))

    def test_a_stiff_box_takes_a_shallower_one(self) -> None:
        self.assertTrue(self._bonds(1.1 * 0.2 / 55.0e3, 1.0e5), "0.22 N as VBD pushes back against a 100 kN/m box")
        self.assertFalse(self._bonds(0.9 * 0.2 / 55.0e3, 1.0e5))

    def test_the_label_presses_as_deep_as_it_was_measured(self) -> None:
        """LABEL's press was set on the example's 2.5 kN/m box: the same 20 um the sheet's ke alone gave 8 kPa."""
        from robosimstudio_materials.tape import LABEL

        self.assertAlmostEqual(8.0e3 / 1.0e4, LABEL.press / (0.5 * (1.0e4 + 2.5e3)))


class TestAContactBondsSpring(unittest.TestCase):
    """A contact bond starts at the law's slope at the contact thickness the kernels use: the radius plus the
    margin of the box nearest it, as the model holds them now."""

    def _label(self, margin: float, *, stuck: bool):
        """A label over a 100 x 120 x 50 mm box whose shape has ``margin`` [m], built, finalized and bound on the CPU."""
        from types import SimpleNamespace

        import newton
        import warp as wp

        from robosimstudio_materials.tape import LABEL, EntityTape, tape

        builder = newton.ModelBuilder()
        box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.025), wp.quat_identity()), label="box")
        cfg = builder.default_shape_cfg.copy()
        cfg.margin = margin
        builder.add_shape_box(box, hx=0.05, hy=0.06, hz=0.025, cfg=cfg)
        label = EntityTape(tape(name="label", on="box", at=(-0.02, -0.05, 0.0505), material=LABEL, stuck=stuck))
        label.build(builder, None)
        model = builder.finalize(device="cpu")
        label._bind_bonds(SimpleNamespace(model=model))
        return label, model

    def test_the_bind_counts_the_boxs_margin(self) -> None:
        from robosimstudio_materials.tape import LABEL, contact_stiffness

        label, model = self._label(0.002, stuck=False)
        want = label.bond_area * contact_stiffness(LABEL, LABEL.particle_radius + 0.002)
        np.testing.assert_allclose(model.attachment_particle_body_stiffness.numpy()[label.bonds], want, rtol=1e-5)
        np.testing.assert_allclose(label._contact["dmin"].numpy(), LABEL.particle_radius + 0.002, rtol=1e-6)

    def test_a_reset_after_the_margin_moved_gives_the_springs_the_kernels_make(self) -> None:
        """The box's margin written live, as its contact tunable's ``thickness`` writes it: a reset seats every
        bond at the thickness the stick kernel makes it with now, not the one the bind found."""
        label, model = self._label(0.002, stuck=False)
        model.shape_margin.assign(np.full(model.shape_count, 0.004, dtype=np.float32))
        label._reset_bonds()
        seated = model.attachment_particle_body_stiffness.numpy()[label.bonds].copy()
        thick = label._contact["dmin"].numpy().copy()
        # Lying 4 mm inside the margin, every bond is pressed: the kernels make each and set its spring.
        label._stick(model.state(), 1.0 / 960.0)
        self.assertTrue(model.attachment_particle_body_enabled.numpy()[label.bonds].all())
        np.testing.assert_allclose(thick, label._contact["dmin"].numpy(), rtol=1e-6)
        np.testing.assert_allclose(seated, model.attachment_particle_body_stiffness.numpy()[label.bonds], rtol=1e-5)

    def test_a_stuck_bond_is_anchored_where_the_margin_rests_its_particle(self) -> None:
        """Built a radius off the face, a stuck label is rested a margin further out by VBD's contact: each
        bond is anchored there, built over a box with a margin or reset after one moved live, so it starts
        at no lift rather than the margin into it. At 1.5 mm that is past ``reach``: every bond let go at
        rest. Stepped by VBD on the CPU, a 0.8 mm margin let all 180 go within two substeps."""
        from robosimstudio_materials.tape import LABEL

        margin, up = 1.5e-3, np.array([0.0, 0.0, 1.0])
        for when in ("bind", "reset"):
            with self.subTest(when=when):
                label, model = self._label(margin if when == "bind" else 0.0, stuck=True)
                if when == "reset":
                    model.shape_margin.assign(np.full(model.shape_count, margin, dtype=np.float32))
                    label._reset_bonds()
                np.testing.assert_allclose(
                    model.attachment_particle_body_body_point.numpy()[label.bonds],
                    label.anchors + margin * up,
                    atol=1e-6,
                )
                state = model.state()
                rested = state.particle_q.numpy()
                rested[label.bond_particles] += margin * up
                state.particle_q.assign(rested)
                label._stick(state, 1.0 / 960.0)
                self.assertGreater(margin, LABEL.reach)
                self.assertEqual(len(label.bonds), label.held(), "every bond holds at rest")

    def test_a_reset_puts_back_the_spring_a_bond_starts_at(self) -> None:
        """Not the secant the last episode's lift left, under the law as the kernels hold it now, in the
        island's copy and the model's alike."""
        import warp as wp

        from robosimstudio_materials.tape import CONTACT_FIELDS, LABEL, contact_stiffness

        label, _ = self._label(0.0, stuck=True)
        n = len(label.bonds)
        island = {key: wp.clone(arrays[0]) for key, arrays in label._arrays.items()}
        label._arrays = {key: [island[key], arrays[0]] for key, arrays in label._arrays.items()}
        for array in label._arrays["stiffness"]:
            array.fill_(7.0)
        live = dataclasses.replace(LABEL, strength=1.0e4)
        label._contact["params"].assign(np.array([getattr(live, f) for f in CONTACT_FIELDS], dtype=np.float32))
        label.broken.fill_(1)
        label._reset_bonds()
        want = label.bond_area * contact_stiffness(live, LABEL.particle_radius)
        for array in label._arrays["stiffness"]:
            got = array.numpy()[label.bonds]
            np.testing.assert_allclose(got, want, rtol=1e-5, err_msg="the law as the kernels hold it now")
        for array in label._arrays["enabled"]:
            self.assertTrue(array.numpy()[label.bonds].all(), "a stuck label is stuck again")
        np.testing.assert_allclose(label._contact["dmin"].numpy(), np.full(n, LABEL.particle_radius), rtol=1e-6)
        self.assertFalse(label.broken.numpy().any())


class TestNothingUnderItMakesNoBond(unittest.TestCase):
    """A vertex past the box's edge has nothing under it: stuck at the build or pressed, it is not bonded."""

    HALF = (0.05, 0.06, 0.025)
    """The example's 100 x 120 x 50 mm box, its centre 25 mm up."""

    def _label(self, material, *, stuck: bool, at: tuple[float, float, float]):
        """A 40 x 95 mm label without a tab, its corner at ``at``, over the box, built and bound on the CPU."""
        from types import SimpleNamespace

        import newton
        import warp as wp

        from robosimstudio_materials.tape import EntityTape, tape

        builder = newton.ModelBuilder()
        box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.025), wp.quat_identity()), label="box")
        builder.add_shape_box(box, hx=self.HALF[0], hy=self.HALF[1], hz=self.HALF[2])
        label = EntityTape(tape(name="label", on="box", at=at, material=material, stuck=stuck, tab=0.0))
        label.build(builder, None)
        model = builder.finalize(device="cpu")
        label._bind_bonds(SimpleNamespace(model=model))
        return label, model

    def test_over_a_face_is_within_its_extent(self) -> None:
        from robosimstudio_materials.tape import over_a_face

        points = [
            (0.0, 0.0, 0.0255),  # over the top
            (0.049, 0.059, 0.0255),  # over the top, by its corner
            (0.055, 0.0, 0.0255),  # 5 mm past the top's +x edge
            (0.055, 0.065, 0.0255),  # past its corner
            (0.0505, 0.0, 0.01),  # over the +x side, below the top
            (0.0, 0.0, 0.02),  # inside, under the top
        ]
        over, box = over_a_face(points, [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)], [self.HALF], [0.0])
        np.testing.assert_array_equal([True, True, False, False, True, True], over)
        np.testing.assert_array_equal(np.zeros(6), box)
        over, box = over_a_face(points[:1], np.zeros((0, 7)), np.zeros((0, 3)), np.zeros(0))
        self.assertFalse(over[0], "no box: nothing is under it")
        self.assertEqual(-1, box[0])

    def test_a_stuck_label_over_the_edge_bonds_only_over_the_face(self) -> None:
        """Half over the +x edge: four of its nine columns, 80 vertices, past it. Bonded there, a contact bond
        was seated on the edge along ``(0.995, 0, 0.1)`` and let go on the first substep, reading 44% peeled
        untouched, and an opening bond held the overhang level in the air, 18 mm over where it hangs free
        (``t2_overhang.py``, CPU VBD)."""
        from robosimstudio_materials.tape import LABEL, PAPER_LABEL

        for material in (PAPER_LABEL, LABEL):
            with self.subTest(adhesion=material.adhesion):
                label, model = self._label(material, stuck=True, at=(0.03, -0.05, 0.0505))
                q = model.particle_q.numpy()[label.bond_particles]
                enabled = model.attachment_particle_body_enabled.numpy()[label.bonds]
                over = q[:, 0] <= self.HALF[0] + 1.0e-6
                if material.adhesion == "opening":
                    self.assertEqual(100, len(label.bonds), "a vertex with nothing under it gets no bond")
                else:
                    self.assertEqual(180, len(label.bonds), "each still bonds where it is pressed onto a face")
                np.testing.assert_array_equal(over, enabled)
                np.testing.assert_array_equal(over, label.made)
                self.assertEqual(100, label.held())
                label.broken.fill_(1)
                label._reset_bonds()
                np.testing.assert_array_equal(over, model.attachment_particle_body_enabled.numpy()[label.bonds])
                self.assertEqual(0.0, label.peeled())

    def test_a_label_pressed_past_the_edge_bonds_only_over_the_face(self) -> None:
        """Pressed 0.1 mm into the top, its last column 0.1 mm past the edge: the 20 vertices there bonded to
        the edge along ``(0.243, 0, 0.970)``, 0.4 mm into the contact (``t2_stick.py``)."""
        from robosimstudio_materials.tape import LABEL

        label, model = self._label(LABEL, stuck=False, at=(0.0101, -0.05, 0.0504))
        label._stick(model.state(), 1.0 / 960.0)
        made = model.attachment_particle_body_enabled.numpy()[label.bonds]
        past = model.particle_q.numpy()[label.bond_particles][:, 0] > self.HALF[0]
        self.assertEqual(20, int(past.sum()))
        self.assertTrue(made[~past].all(), "pressed over the face, every vertex bonds")
        self.assertFalse(made[past].any(), "past the edge, none does")
        np.testing.assert_allclose(label._contact["normal"].numpy()[made], np.tile([0.0, 0.0, 1.0], (160, 1)))


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

    def test_a_strip_stuck_to_nothing_shows_its_adhesive_read_only(self) -> None:
        from unittest import mock

        import robosimstudio as rss
        from robosimstudio.core.tuning import Tunable
        from robosimstudio_materials.cloth import EntityCloth
        from robosimstudio_materials.tape import ADHESIVE_FIELDS, CONTACT_FIELDS, EntityTape

        cloth = Tunable(
            name="material", spec=rss.tapes.PAPER_LABEL, write=lambda name, value: True, inert=("core_damping",)
        )
        cases = (
            (False, 0, ("core_damping", *ADHESIVE_FIELDS)),
            (False, 144, ("core_damping", *CONTACT_FIELDS)),
            (True, 0, ("core_damping", *ADHESIVE_FIELDS)),
            (True, 126, ("core_damping", "tack", "break_opening")),
        )
        for contact, bonds, inert in cases:
            with self.subTest(contact=contact, bonds=bonds):
                entity = EntityTape(rss.tape(name="label", on="box" if bonds else ""))
                entity.contact = contact
                entity.bonds = np.arange(bonds, dtype=np.int32)
                with mock.patch.object(EntityCloth, "tunables", return_value=(cloth,)):
                    (material,) = entity.tunables()
                self.assertEqual(inert, material.inert)
                lands = bool(bonds) and not contact
                self.assertEqual(lands, material.write("tack", 2.0e6))
                self.assertEqual(lands, material.write("break_opening", 0.002))
                if not bonds:
                    self.assertFalse(material.write("strength", 1.0e3), "a loose strip's contact law lands nothing")
                self.assertTrue(material.write("tri_ke", 500.0), "the cloth's own fields still land")

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
