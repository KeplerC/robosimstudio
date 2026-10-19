# SPDX-License-Identifier: Apache-2.0
"""``robosimstudio_materials/tape.py``: the sheet a tape is built as, what it bonds, and what it declares.

No GPU: a label is built on the CPU, and stepped there by VBD only to show its bonds hold at rest.
The tape's physics -- a hand peeling a label off a box by its pull tab, and a label that never lets
go staying stuck -- is driven through the example's page by ``tests/test_peel_a_label.py``. Here:

* **The surface.** The sheet's first ``tab`` metres stand straight up at the fold, the rest lies
  flat; the bonded vertices are the flat ones past the fold row, the clamped ones the upper half of
  the turned-up end, and the two never overlap.
* **The break law**, on positions written by hand: only lifting off the face opens a bond; sliding
  along it and pressing into it do not.
* **Where a stuck sheet lies**: the box's margin past a particle radius off the face, at the build,
  at a reset after the margin or the radius moved and at the next control step after a live write,
  so every bond holds at rest over a 2 mm margin and through a 1 mm write; and bonded only over a
  face of the box, none past its edge, stuck at the build or pressed, by the same rule in numpy and
  in warp.
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
"""Numpy, the spec classes and warp on the CPU: four labels stepped 40 substeps, two 80 and two 32, about 7 s."""


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
        label._model = model
        label._contact = {
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


class TestAStuckSheetIsLaidWhereTheContactRestsIt(unittest.TestCase):
    """Placed a particle radius off the face, a stuck sheet is built, reset and anchored the box's margin
    further out, where VBD's contact rests it, so every bond starts at no lift whatever the margin.

    Built a radius off the face, the contact pushed it out through the margin and it overshot: stepped
    by VBD on the CPU, 178 of :data:`LABEL`'s 180 bonds held at a 0.8 mm margin and none 2-5 substeps in
    at 1 mm and up, and :data:`PAPER_LABEL`'s rested 0.47 mm lifted at 0.5 mm and all broke on the
    first substep at 1.2 mm (``z2b.py``, ``z2_opening.py``); reset after the margin moved live, the same
    (``z2_reset.py``)."""

    def _label(self, material, margin: float):
        """A label stuck on a 100 x 120 x 50 mm box resting on the ground, the box's shape at ``margin`` [m],
        built, bound, snapshotted as the scene would and ready to reset, on the CPU."""
        from types import SimpleNamespace

        import newton
        import warp as wp

        from robosimstudio_materials.tape import EntityTape, tape

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.025), wp.quat_identity()), label="box")
        cfg = builder.default_shape_cfg.copy()
        cfg.margin = margin
        self.box_shape = builder.add_shape_box(box, hx=0.05, hy=0.06, hz=0.025, cfg=cfg)
        label = EntityTape(tape(name="label", on="box", at=(-0.02, -0.05, 0.0505), material=material))
        label.build(builder, None)
        builder.color()
        model = builder.finalize(device="cpu")
        label._bind_bonds(SimpleNamespace(model=model))
        label.particles = np.arange(label.surface_count)  # what the bind takes off the thing's range
        label.built = model.particle_q.numpy()[label.particles].copy()  # as on_scene_ready takes it
        state_0, state_1 = model.state(), model.state()
        label.rest = {"particle_q": state_0.particle_q.numpy().copy(), "particle_qd": state_0.particle_qd.numpy()}
        return label, model, SimpleNamespace(model=model, state_0=state_0, state_1=state_1)

    def _margin(self, model, margin: float) -> None:
        """The box's margin written live, as its contact tunable's ``thickness`` writes it."""
        margins = model.shape_margin.numpy().copy()
        margins[self.box_shape] = margin
        model.shape_margin.assign(margins)

    def _lift(self, label, state) -> np.ndarray:
        """Each bond's particle past its anchor, off the box's top [m]: the box is not turned."""
        anchors = label._arrays["body_point"][0].numpy()[label.bonds] + state.body_q.numpy()[label.body][:3]
        return (state.particle_q.numpy()[label.bond_particles] - anchors)[:, 2]

    def test_built_over_a_margin_the_sheet_and_its_tab_lie_that_much_further_off(self) -> None:
        from robosimstudio_materials.tape import LABEL, PAPER_LABEL

        for material in (PAPER_LABEL, LABEL):
            with self.subTest(adhesion=material.adhesion):
                flat, flat_model, _ = self._label(material, 0.0)
                label, model, pipeline = self._label(material, 0.002)
                up = np.array([0.0, 0.0, 0.002])
                np.testing.assert_allclose(label.built, flat.built + up, atol=1e-6)
                np.testing.assert_allclose(
                    model.body_q.numpy()[label.tab_body][:3],
                    flat_model.body_q.numpy()[flat.tab_body][:3] + up,
                    atol=1e-6,
                )
                self.assertAlmostEqual(0.002, label.lift)
                np.testing.assert_allclose(self._lift(label, pipeline.state_0), 0.0, atol=1e-6)
                if label.contact:
                    np.testing.assert_allclose(label._contact["dmin"].numpy(), material.particle_radius + 0.002)

    def test_a_reset_lays_it_at_the_margin_the_box_holds_now(self) -> None:
        """Built at none, the margin written to 2 mm and back: each reset lays the sheet, its tab and every
        anchor where the margin is now, and the opening bonds' break kernel measures from there."""
        from robosimstudio_materials.tape import LABEL, PAPER_LABEL

        for material in (PAPER_LABEL, LABEL):
            with self.subTest(adhesion=material.adhesion):
                label, model, pipeline = self._label(material, 0.0)
                built, tab = label.built.copy(), label.tab_q0.copy()
                for margin in (0.002, 0.0):
                    self._margin(model, margin)
                    label.reset(pipeline, None)
                    up = np.array([0.0, 0.0, margin])
                    for state in (pipeline.state_0, pipeline.state_1):
                        np.testing.assert_allclose(state.particle_q.numpy()[label.particles], built + up, atol=1e-6)
                        c = label.coord_start
                        np.testing.assert_allclose(state.joint_q.numpy()[c : c + 3], tab[:3] + up, atol=1e-6)
                    np.testing.assert_allclose(self._lift(label, pipeline.state_0), 0.0, atol=1e-6)
                    if not label.contact:
                        np.testing.assert_allclose(label._anchor.numpy(), label.anchors + up, atol=1e-6)

    def test_every_bond_holds_at_rest_over_a_2_mm_margin(self) -> None:
        """Stepped by VBD on the CPU as the probes step it, built over the margin or reset after it moved: 40
        substeps, where before the fix :data:`LABEL`'s 144 had let go by the second and :data:`PAPER_LABEL`'s
        broke on the first."""
        import newton

        from robosimstudio_materials.tape import LABEL, PAPER_LABEL

        for material in (PAPER_LABEL, LABEL):
            for when in ("build", "reset"):
                with self.subTest(adhesion=material.adhesion, when=when):
                    label, model, pipeline = self._label(material, 0.002 if when == "build" else 0.0)
                    s0, s1 = pipeline.state_0, pipeline.state_1
                    if when == "reset":
                        self._margin(model, 0.002)
                        label.reset(pipeline, None)
                        for state in (s0, s1):
                            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                    solver = newton.solvers.SolverVBD(model, iterations=10, rigid_compliant_alm=True)
                    collide = newton.CollisionPipeline(model, soft_contact_gap=0.005)
                    contacts, control, dt = collide.contacts(), model.control(), 1.0 / 960.0
                    for _ in range(40):
                        collide.collide(s0, contacts)
                        s0.clear_forces()
                        solver.step(s0, s1, control, contacts, dt)
                        (label._stick if label.contact else label._break)(s1, dt)
                        s0, s1 = s1, s0
                    self.assertEqual(len(label.bonds), label.held(), "every bond holds at rest")
                    self.assertEqual(0.0, label.peeled())
                    # Within 0.11 mm of its anchor, the tab's weight on the fold; the reach and the break are 1 mm.
                    self.assertLess(float(np.abs(self._lift(label, s0)).max()), 2.0e-4)

    def test_a_live_margin_write_lays_the_held_bonds_again_where_the_contact_rests_them(self) -> None:
        """The margin written live, no reset: the relay, once a control step before its substeps, moves every
        held bond's particle and anchor 1 mm off the face, once, and the break kernel's anchors with them; a
        bond let go stays where it was."""
        from robosimstudio.core.schedule import SubstepHooks
        from robosimstudio_materials.tape import LABEL, PAPER_LABEL

        for material in (PAPER_LABEL, LABEL):
            with self.subTest(adhesion=material.adhesion):
                label, model, pipeline = self._label(material, 0.0)
                hooks = SubstepHooks()
                label.contribute_hooks(hooks)
                self.assertIn(("before", "relay:label"), hooks.schedule.names())
                state, rows = pipeline.state_0, label.bonds
                gone = np.arange(len(rows)) < 9  # the row by the fold, let go
                for array in label._arrays["enabled"]:
                    array.assign(np.where(np.isin(np.arange(len(array)), rows[gone]), False, array.numpy()))

                def points(label=label, state=state):
                    return state.particle_q.numpy()[label.bond_particles]

                def anchors(label=label):
                    return label._arrays["body_point"][0].numpy()[label.bonds]

                q0, a0 = points(), anchors()
                self._margin(model, 0.001)
                label._relay(state, 1.0 / 60.0)
                up = np.array([0.0, 0.0, 0.001])
                np.testing.assert_allclose(points()[~gone] - q0[~gone], np.tile(up, (len(rows) - 9, 1)), atol=1e-7)
                np.testing.assert_allclose(anchors()[~gone] - a0[~gone], np.tile(up, (len(rows) - 9, 1)), atol=1e-7)
                np.testing.assert_array_equal(q0[gone], points()[gone])
                np.testing.assert_array_equal(a0[gone], anchors()[gone])
                if label.contact:
                    np.testing.assert_allclose(label._contact["dmin"].numpy()[~gone], material.particle_radius + 0.001)
                else:
                    np.testing.assert_array_equal(anchors(), label._anchor.numpy())
                q1 = points()
                label._relay(state, 1.0 / 60.0)
                np.testing.assert_array_equal(q1, points(), "once: the margin has not moved since")

    def test_every_bond_holds_through_a_live_margin_write(self) -> None:
        """Stepped by VBD on the CPU in control steps of 16 substeps, the relay at the start of each, the margin
        written to 1 mm between two and no reset: all 144 of either law's bonds hold. Without the relay
        :data:`PAPER_LABEL` kept 2 and :data:`LABEL` none (``a2_cpu2.py``)."""
        import newton

        from robosimstudio_materials.tape import LABEL, PAPER_LABEL

        for material in (PAPER_LABEL, LABEL):
            with self.subTest(adhesion=material.adhesion):
                label, model, pipeline = self._label(material, 0.0)
                s0, s1 = pipeline.state_0, pipeline.state_1
                solver = newton.solvers.SolverVBD(model, iterations=10, rigid_compliant_alm=True)
                collide = newton.CollisionPipeline(model, soft_contact_gap=0.005)
                contacts, control, dt = collide.contacts(), model.control(), 1.0 / 960.0
                for step in range(5):
                    if step == 2:
                        self._margin(model, 0.001)
                    label._relay(s0, 16 * dt)
                    for _ in range(16):
                        collide.collide(s0, contacts)
                        s0.clear_forces()
                        solver.step(s0, s1, control, contacts, dt)
                        (label._stick if label.contact else label._break)(s1, dt)
                        s0, s1 = s1, s0
                self.assertEqual(len(label.bonds), label.held(), "every bond holds through the write")
                self.assertLess(float(np.abs(self._lift(label, s0)).max()), 5.0e-4)

    def _radius(self, label, model, radius: float) -> None:
        """The sheet's particle radius written live, as its material's ``particle_radius`` writes it."""
        radii = model.particle_radius.numpy().copy()
        radii[label.particles] = radius
        model.particle_radius.assign(radii)

    def test_a_reset_lays_it_at_the_radius_the_sheet_holds_now(self) -> None:
        """Written from 0.5 to 1.5 mm and reset: the sheet and every anchor 1 mm further off the face, the tab and the
        end it clamps not, each bond laid at the thickness the relay reads, so the relay moves nothing after. Reset at
        the radius it was built at, a contact bond was seated 1 mm over its particle, and the relay, finding it already
        laid there, left the contact to push the sheet out through it on the first substep."""
        from robosimstudio_materials.tape import LABEL, PAPER_LABEL

        for material in (PAPER_LABEL, LABEL):
            with self.subTest(adhesion=material.adhesion):
                label, model, pipeline = self._label(material, 0.0)
                built, tab, now = label.built.copy(), label.tab_q0.copy(), material.particle_radius + 0.001
                self._radius(label, model, now)
                label.reset(pipeline, None)
                np.testing.assert_allclose(label.radius_moved(), 0.001, atol=1e-7)
                clamped = np.isin(label.particles, label.clamp_particles)
                up = np.where(clamped[:, None], 0.0, np.array([[0.0, 0.0, 0.001]]))
                for state in (pipeline.state_0, pipeline.state_1):
                    np.testing.assert_allclose(state.particle_q.numpy()[label.particles], built + up, atol=1e-6)
                    c = label.coord_start
                    np.testing.assert_allclose(state.joint_q.numpy()[c : c + 7], tab, atol=1e-6)
                np.testing.assert_allclose(self._lift(label, pipeline.state_0), 0.0, atol=1e-6)
                laid = label._contact["dmin"] if label.contact else label._laid
                np.testing.assert_allclose(laid.numpy(), now, atol=1e-7)
                q = pipeline.state_0.particle_q.numpy().copy()
                label._relay(pipeline.state_0, 1.0 / 60.0)
                np.testing.assert_array_equal(q, pipeline.state_0.particle_q.numpy(), "laid where the relay lays it")

    def test_every_bond_holds_through_a_live_radius_write_and_a_reset(self) -> None:
        """The radius written to 1.5 mm, the scene reset, then two control steps of VBD on the CPU: all 144 of either
        law's bonds hold. Reset at the radius it was built at, :data:`LABEL` kept 8 (``r2_probe.py``)."""
        import newton

        from robosimstudio_materials.tape import LABEL, PAPER_LABEL

        for material in (PAPER_LABEL, LABEL):
            with self.subTest(adhesion=material.adhesion):
                label, model, pipeline = self._label(material, 0.0)
                s0, s1 = pipeline.state_0, pipeline.state_1
                solver = newton.solvers.SolverVBD(model, iterations=10, rigid_compliant_alm=True)
                collide = newton.CollisionPipeline(model, soft_contact_gap=0.005)
                contacts, control, dt = collide.contacts(), model.control(), 1.0 / 960.0
                self._radius(label, model, material.particle_radius + 0.001)
                label.reset(pipeline, None)
                for state in (s0, s1):
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                for _ in range(2):
                    label._relay(s0, 16 * dt)
                    for _ in range(16):
                        collide.collide(s0, contacts)
                        s0.clear_forces()
                        solver.step(s0, s1, control, contacts, dt)
                        (label._stick if label.contact else label._break)(s1, dt)
                        s0, s1 = s1, s0
                self.assertEqual(len(label.bonds), label.held(), "every bond holds through the write and the reset")
                self.assertLess(float(np.abs(self._lift(label, s0)).max()), 5.0e-4)

    def test_every_bond_holds_through_a_radius_written_to_half_the_sheet_and_a_reset(self) -> None:
        """The radius written to 1.5 mm over the half of the sheet past its middle, the scene reset, then two control
        steps of VBD on the CPU: each particle laid by its own radius's change, so all 144 of either law's bonds hold.
        Lifted by the largest change, the half left at 0.5 mm was laid 1 mm over the thickness each bond is seated at."""
        import newton

        from robosimstudio_materials.tape import LABEL, PAPER_LABEL

        for material in (PAPER_LABEL, LABEL):
            with self.subTest(adhesion=material.adhesion):
                label, model, pipeline = self._label(material, 0.0)
                s0, s1 = pipeline.state_0, pipeline.state_1
                built, along = label.built.copy(), label.built[:, 1]
                far = along > 0.5 * (along.min() + along.max())
                radii = model.particle_radius.numpy().copy()
                radii[label.particles[far]] += 0.001
                model.particle_radius.assign(radii)
                label.reset(pipeline, None)
                clamped = np.isin(label.particles, label.clamp_particles)
                up = np.where((far & ~clamped)[:, None], np.array([[0.0, 0.0, 0.001]]), 0.0)
                np.testing.assert_allclose(s0.particle_q.numpy()[label.particles], built + up, atol=1e-6)
                np.testing.assert_allclose(self._lift(label, s0), 0.0, atol=1e-6)
                for state in (s0, s1):
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                solver = newton.solvers.SolverVBD(model, iterations=10, rigid_compliant_alm=True)
                collide = newton.CollisionPipeline(model, soft_contact_gap=0.005)
                contacts, control, dt = collide.contacts(), model.control(), 1.0 / 960.0
                for _ in range(2):
                    label._relay(s0, 16 * dt)
                    for _ in range(16):
                        collide.collide(s0, contacts)
                        s0.clear_forces()
                        solver.step(s0, s1, control, contacts, dt)
                        (label._stick if label.contact else label._break)(s1, dt)
                        s0, s1 = s1, s0
                self.assertEqual(len(label.bonds), label.held(), "every bond holds through the half write and reset")
                self.assertLess(float(np.abs(self._lift(label, s0)).max()), 5.0e-4)


class TestNothingUnderItMakesNoBond(unittest.TestCase):
    """A vertex past the box's edge has nothing under it: stuck at the build or pressed, it is not bonded."""

    HALF = (0.05, 0.06, 0.025)
    """The example's 100 x 120 x 50 mm box, its centre 25 mm up."""

    def _label(self, material, *, stuck: bool, at: tuple[float, float, float], centre=(0.0, 0.0, 0.025), **kwargs):
        """A label without a tab, 40 x 95 mm unless ``kwargs`` say otherwise, its corner at ``at``, over the box
        centred at ``centre``, built and bound on the CPU."""
        from types import SimpleNamespace

        import newton
        import warp as wp

        from robosimstudio_materials.tape import EntityTape, tape

        builder = newton.ModelBuilder()
        box = builder.add_body(xform=wp.transform(centre, wp.quat_identity()), label="box")
        builder.add_shape_box(box, hx=self.HALF[0], hy=self.HALF[1], hz=self.HALF[2])
        label = EntityTape(tape(name="label", on="box", at=at, material=material, stuck=stuck, tab=0.0, **kwargs))
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
            (0.05 + 1.0e-8, 0.06 + 1.0e-8, 0.0255),  # on the top's corner, a float32 rounding past it
            (0.05002, 0.0, 0.0255),  # 0.02 mm past the +x edge
        ]
        over, box = over_a_face(points, [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)], [self.HALF], [0.0])
        np.testing.assert_array_equal([True, True, False, False, True, True, True, False], over)
        np.testing.assert_array_equal(np.zeros(8), box)
        over, box = over_a_face(points[:1], np.zeros((0, 7)), np.zeros((0, 3)), np.zeros(0))
        self.assertFalse(over[0], "no box: nothing is under it")
        self.assertEqual(-1, box[0])

    def test_the_builds_rule_is_the_kernels(self) -> None:
        """:func:`over_a_face` is numpy and the stick kernel's :func:`_seat_on` warp: the same box and the same
        answer on the edge cases, on the example's box, one turned and off its body's origin, and two boxes
        with margins, whichever is nearest. No box, neither has one under any point."""
        import warp as wp

        from robosimstudio.core.math import quat_to_mat
        from robosimstudio_materials.tape import _FLUSH, _seat_on, over_a_face

        @wp.kernel
        def seats(
            points: wp.array(dtype=wp.vec3),
            boxes: wp.array(dtype=wp.int32),
            shape_transform: wp.array(dtype=wp.transform),
            shape_scale: wp.array(dtype=wp.vec3),
            shape_margin: wp.array(dtype=float),
            over: wp.array(dtype=wp.bool),
            shape: wp.array(dtype=wp.int32),
        ):
            i = wp.tid()
            seat = _seat_on(points[i], 0.0, boxes, shape_transform, shape_scale, shape_margin)
            over[i] = seat.over
            shape[i] = seat.shape

        def kernel(points, transforms, halves, margins):
            cpu, n = "cpu", len(margins)
            over = wp.zeros(len(points), dtype=wp.bool, device=cpu)
            shape = wp.zeros(len(points), dtype=wp.int32, device=cpu)
            inputs = [
                wp.array(np.asarray(points, dtype=np.float32), dtype=wp.vec3, device=cpu),
                wp.array(np.arange(n, dtype=np.int32), dtype=wp.int32, device=cpu),
                wp.array(np.asarray(transforms, dtype=np.float32).reshape(-1, 7), dtype=wp.transform, device=cpu),
                wp.array(np.asarray(halves, dtype=np.float32).reshape(-1, 3), dtype=wp.vec3, device=cpu),
                wp.array(np.asarray(margins, dtype=np.float32), dtype=float, device=cpu),
            ]
            wp.launch(seats, dim=len(points), inputs=inputs, outputs=[over, shape], device=cpu)
            return over.numpy().astype(bool), shape.numpy().astype(np.int64)

        h, f = np.array(self.HALF), float(_FLUSH)
        local = np.array(
            [
                (0.0, 0.0, h[2] + 5e-4),  # over the top
                (h[0] - 1e-3, h[1] - 1e-3, h[2] + 5e-4),  # by its corner
                (h[0] + 0.5 * f, 0.0, h[2] + 5e-4),  # half the allowance past an edge: on it
                (h[0] + 2.0 * f, 0.0, h[2] + 5e-4),  # twice it: past
                (h[0] + 0.5 * f, h[1] + 0.5 * f, h[2] + 5e-4),  # on the corner
                (h[0] + 2.0 * f, h[1] + 0.5 * f, h[2] + 5e-4),  # past one edge, on the other: over neither face
                (h[0] + 2.0 * f, h[1] + 2.0 * f, h[2] + 5e-4),  # past the corner
                (h[0] + 5e-4, 0.0, 0.0),  # over the +x side
                (h[0] + 5e-4, 0.0, h[2] + 5e-4),  # past the top's edge, over the side's: past two
                (0.0, 0.0, 0.5 * h[2]),  # inside
            ]
        )
        turn = np.array([0.0, 0.0, np.sin(0.2), np.cos(0.2)])  # 23 degrees about z
        # A second box 4 mm past the first's +x side, with a 2 mm margin: a point 1.5 mm off the first's side
        # is 2.5 mm off the second's, 0.5 mm past its margin, so the second is the nearer, and it lies over
        # that one's -x side.
        beside = 2 * h[0] + 0.004
        cases = {
            "the example's box": (local, [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)], [h], [0.0]),
            "turned, off the origin": (
                local @ quat_to_mat(turn).T + (0.41, -0.083, 0.025),
                [(0.41, -0.083, 0.025, *turn)],
                [h],
                [0.001],
            ),
            "two boxes with margins": (
                np.array([*local, *(local + np.array([beside, 0.0, 0.0])), (h[0] + 0.0015, 0.0, 0.0)]),
                [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (beside, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
                [h, h],
                [0.0, 0.002],
            ),
            "no box": (local, np.zeros((0, 7)), np.zeros((0, 3)), np.zeros(0)),
        }
        for name, (points, transforms, halves, margins) in cases.items():
            with self.subTest(name):
                over, box = over_a_face(points, transforms, halves, margins)
                want_over, want_box = kernel(points, transforms, halves, margins)
                np.testing.assert_array_equal(want_box, box)
                np.testing.assert_array_equal(want_over, over)
                if name == "the example's box":
                    np.testing.assert_array_equal([1, 1, 1, 0, 1, 0, 0, 1, 0, 1], over)
                if name == "two boxes with margins":
                    np.testing.assert_array_equal([0] * len(local) + [1] * len(local), box[:-1])
                    self.assertEqual((True, 1), (bool(over[-1]), int(box[-1])), "the nearer counting its margin")
                if name == "no box":
                    self.assertFalse(over.any())

    def test_the_build_launches_no_kernel(self) -> None:
        """A stuck sheet's bonds are found in numpy at the build: through the warp function on the CPU, the build
        compiled the tape's kernels for the CPU inside a GPU process, 1.8 s on a cold kernel cache, where it
        takes 8 ms now (``a4_build_time.py``)."""
        from unittest import mock

        import newton
        import warp as wp

        from robosimstudio_materials.tape import LABEL, PAPER_LABEL, EntityTape, tape

        for material in (PAPER_LABEL, LABEL):
            with self.subTest(adhesion=material.adhesion):
                builder = newton.ModelBuilder()
                box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.025), wp.quat_identity()), label="box")
                builder.add_shape_box(box, hx=self.HALF[0], hy=self.HALF[1], hz=self.HALF[2])
                label = EntityTape(tape(name="label", on="box", at=(0.03, -0.05, 0.0505), material=material, tab=0.0))
                with mock.patch.object(wp, "launch", side_effect=AssertionError("a kernel launched at the build")):
                    label.build(builder, None)
                self.assertEqual(100, int(label.made.sum()), "the bonds over the top, none past its edge")

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

    def test_both_laws_read_a_stuck_label_over_the_edge_by_the_bonds_made_at_the_build(self) -> None:
        """Untouched it reads 1.0 bonded and 0.0 peeled, and lifted 2 mm, past the break and the reach, 0.0 and
        1.0, under either law. Counted over every bond, contact adhesion's 80 left unmade past the edge read it
        100/180 bonded untouched and 100/180 peeled at most, where opening adhesion, which builds none there,
        read 1.0 and 0.0."""
        from robosimstudio_materials.tape import LABEL, PAPER_LABEL

        for material in (PAPER_LABEL, LABEL):
            with self.subTest(adhesion=material.adhesion):
                label, model = self._label(material, stuck=True, at=(0.03, -0.05, 0.0505))
                self.assertEqual((1.0, 0.0), (label.bonded(), label.peeled()))
                state = model.state()
                state.particle_q.assign(state.particle_q.numpy() + np.array([0.0, 0.0, 0.002], dtype=np.float32))
                (label._stick if label.contact else label._break)(state, 1.0 / 960.0)
                self.assertEqual(0, label.held())
                self.assertEqual((0.0, 1.0), (label.bonded(), label.peeled()))

    def test_a_stuck_labels_overhang_pressed_onto_the_side_is_counted_once_made(self) -> None:
        """Contact adhesion leaves the 80 vertices past the +x edge unmade at the build, and makes them where they
        are pressed onto the box's side: wrapped down it 0.1 mm into the contact, all 180 hold and it reads 1.0 and
        0.0; the build's 100 then lifted 2 mm, past the reach, 80/180 and 100/180, one less the other. Counted over
        the build's bonds alone, the wrap read nothing: 0.0 bonded with all 80 holding."""
        from robosimstudio_materials.tape import LABEL

        label, model = self._label(LABEL, stuck=True, at=(0.03, -0.05, 0.0505))
        past, p = ~label.made, label.bond_particles
        self.assertEqual(80, int(past.sum()))
        state = model.state()
        q = state.particle_q.numpy().copy()
        beyond = q[p[past], 0] - self.HALF[0]  # how far past the edge each lies along the top [m]
        q[p[past], 0] = self.HALF[0] + LABEL.particle_radius - 1.0e-4
        q[p[past], 2] = 2.0 * self.HALF[2] - beyond
        state.particle_q.assign(q)
        label._stick(state, 1.0 / 960.0)
        self.assertEqual(180, label.held())
        self.assertEqual((1.0, 0.0), (label.bonded(), label.peeled()))
        q[p[label.made], 2] += 0.002
        state.particle_q.assign(q)
        label._stick(state, 1.0 / 960.0)
        self.assertEqual(80, label.held())
        self.assertAlmostEqual(80 / 180, label.bonded(), places=6)
        self.assertAlmostEqual(100 / 180, label.peeled(), places=6)

    def test_a_label_laid_flush_with_the_edges_bonds_on_them(self) -> None:
        """Covering the box's top exactly: its edge vertices lie on the edges, up to float32's rounding past them,
        and every one bonds, as on the example's box and on one off the origin. Tested past the extent with no
        allowance, a row of 11 went unbonded on the example's box, a row and a column (23) on the other
        (``flush.py``, ``flush_step.py``); bonded, all 143 held 60 substeps of CPU VBD."""
        from robosimstudio_materials.tape import LABEL, PAPER_LABEL

        for material in (PAPER_LABEL, LABEL):
            for centre in ((0.0, -0.08, 0.025), (0.41, -0.083, 0.025)):
                with self.subTest(adhesion=material.adhesion, centre=centre):
                    corner = (centre[0] - self.HALF[0], centre[1] - self.HALF[1], centre[2] + self.HALF[2] + 0.0005)
                    size = (2 * self.HALF[0], 2 * self.HALF[1])
                    label, model = self._label(
                        material, stuck=True, at=corner, centre=centre, size=size, cells=(10, 12)
                    )
                    self.assertEqual(143, len(label.bonds))
                    self.assertTrue(label.made.all(), f"{int((~label.made).sum())} vertices on an edge went unbonded")
                    self.assertTrue(model.attachment_particle_body_enabled.numpy()[label.bonds].all())

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
        self.assertAlmostEqual(160 / 180, label.bonded(), msg="a loose label counts every bond", places=6)
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

    @staticmethod
    def _flagged(holding, gone, along=(0.0, 0.0, 0.0, 0.0)):
        """A label stuck by four bonds, all made at the build, its solver's flags ``holding`` and ``gone``: device
        arrays whose copies to the host are counted, ``copies["holding"]`` and ``copies["gone"]``."""
        import warp as wp

        from robosimstudio_materials.tape import EntityTape, tape

        copies = {"holding": 0, "gone": 0}

        class Counted:
            def __init__(self, name, values, dtype) -> None:
                self.name, self.array = name, wp.array(values, dtype=dtype, device="cpu")

            def numpy(self):
                copies[self.name] += 1
                return self.array.numpy()

        entity = EntityTape(tape(name="label", on="box"))
        entity.bonds, entity.made, entity.along = np.arange(4, dtype=np.int32), np.ones(4, dtype=bool), np.array(along)
        entity._arrays = {"enabled": [Counted("holding", holding, wp.bool)]}
        entity.broken = Counted("gone", gone, wp.int32)
        return entity, copies

    def test_the_bonded_handler_reads_the_bonds_holding(self) -> None:
        """Its share and its count off one copy of each flag, the share as :meth:`bonded` reads it."""
        from types import SimpleNamespace

        from robosimstudio_materials.tape import _bonded_fraction

        entity, copies = self._flagged([True, True, True, False], [0, 0, 0, 1])
        got = _bonded_fraction(SimpleNamespace(entities={"label": entity}), "label")
        self.assertEqual({"fraction": 0.75, "held": 3.0, "label/bonded_fraction": 0.75}, got)
        self.assertEqual({"holding": 1, "gone": 1}, copies, "each flag copied once")
        self.assertEqual((0.75, 3), (entity.bonded(), entity.held()))
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
        """Its share and its front off one copy of each flag, as :meth:`peeled` and :meth:`peel_front` read them."""
        from types import SimpleNamespace

        from robosimstudio_materials.tape import _peeled_fraction

        entity, copies = self._flagged([False, True, True, True], [1, 0, 0, 0], along=(0.02, 0.01, 0.0, 0.03))
        got = _peeled_fraction(SimpleNamespace(entities={"label": entity}), "label")
        self.assertEqual({"fraction": 0.25, "front": 0.02, "label/peeled_fraction": 0.25}, got)
        self.assertEqual({"holding": 1, "gone": 1}, copies, "each flag copied once")
        self.assertEqual((0.25, 0.02), (entity.peeled(), entity.peel_front()))
        with self.assertRaises(KeyError):
            _peeled_fraction(SimpleNamespace(entities={}), "label")

    def test_a_loose_strips_shares_copy_one_flag_each(self) -> None:
        """Counting every bond, a loose strip's :meth:`peeled` copies only the let-go flags off the device, and its
        :meth:`bonded` and :meth:`held` only the holding ones."""
        for call, want, copied in (("peeled", 0.25, (0, 1)), ("bonded", 0.75, (1, 0)), ("held", 3, (1, 0))):
            with self.subTest(call):
                entity, copies = self._flagged([False, True, True, True], [1, 0, 0, 0])
                entity.made = np.zeros(4, dtype=bool)
                self.assertEqual(want, getattr(entity, call)())
                self.assertEqual(copied, (copies["holding"], copies["gone"]))


if __name__ == "__main__":
    unittest.main()
