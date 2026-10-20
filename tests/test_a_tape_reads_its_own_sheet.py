# SPDX-License-Identifier: Apache-2.0
"""A tape reads its own sheet: its own radii once a reset, a sheet with no bond laid where it rests, and its handlers
refuse what is not a tape.

No GPU: labels built on the CPU, one stepped 8 substeps by VBD. ``tests/test_tape_kind.py`` holds where a stuck
sheet is laid; here, what that laying reads and what it does with a sheet stuck by no bond:

* **Its own radii, once.** A reset copied the model's whole ``particle_radius`` off the device three times under
  opening adhesion and twice under contact, and a bound tape kept a copy of every particle's radius in the model
  for its life. A label built after 500 other particles reads its own 180 radii once a reset, and lays itself, its
  bonds and their anchors as the whole array laid them.
* **A sheet with no bond** is laid by its radius change as a bonded one is: it rests on the face, stuck to nothing.
* **The handlers** ask for a tape, not for something offering ``peeled()`` or ``bonded()``.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "fast"
"""Three labels built on the CPU, one stepped 8 substeps by VBD, about 2 s."""

BEFORE = 500
"""Particles in the model before the label: another material's, whose radii the label has no use for."""


def _label(material, *, before: int = 0, drum: bool = False, **kwargs):
    """A label stuck on a 100 x 120 x 50 mm box, or on a cylinder 160 mm across and 50 mm tall with no box (``drum``),
    resting on the ground, built after ``before`` loose particles, bound, snapshotted as the scene would and ready to
    reset, on the CPU."""
    from types import SimpleNamespace

    import newton
    import warp as wp

    from robosimstudio_materials.tape import EntityTape, tape

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for i in range(before):
        builder.add_particle(wp.vec3(0.5 + 0.01 * (i % 25), 0.5 + 0.01 * (i // 25), 0.3), wp.vec3(0.0), 0.0, 0.002)
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.025), wp.quat_identity()), label="box")
    if drum:
        builder.add_shape_cylinder(body, radius=0.08, half_height=0.025)
    else:
        builder.add_shape_box(body, hx=0.05, hy=0.06, hz=0.025)
    label = EntityTape(tape(name="label", on="box", at=(-0.02, -0.05, 0.0505), material=material, **kwargs))
    label.build_into(builder, None)
    builder.color()
    model = builder.finalize(device="cpu")
    label.particles = label.owned_particles()  # what the bind takes off the thing's range
    label._bind_bonds(SimpleNamespace(model=model))
    label.built = model.particle_q.numpy()[label.particles].copy()  # as on_scene_ready takes it
    state_0, state_1 = model.state(), model.state()
    label.rest = {"particle_q": state_0.particle_q.numpy().copy(), "particle_qd": state_0.particle_qd.numpy()}
    return label, model, SimpleNamespace(model=model, state_0=state_0, state_1=state_1)


def _write_radius(label, model, by: float) -> None:
    """The sheet's particle radius written ``by`` [m] further, as its material's ``particle_radius`` writes it live."""
    radii = model.particle_radius.numpy().copy()
    radii[label.particles] += by
    model.particle_radius.assign(radii)


class TestAResetReadsItsOwnRadiiOnce(unittest.TestCase):
    def _counting(self, array):
        """A patch that counts each copy of ``array``, or of a slice of it, off the device, and the copies' sizes."""
        from unittest import mock

        import warp as wp

        sizes: list[int] = []
        real = wp.array.numpy
        low, high = int(array.ptr), int(array.ptr) + 4 * int(array.size)

        def numpy(self):
            if self.ptr is not None and low <= int(self.ptr) < high:
                sizes.append(int(self.size))
            return real(self)

        return mock.patch.object(wp.array, "numpy", numpy), sizes

    def test_a_label_after_another_material_reads_its_own_radii_once_a_reset(self) -> None:
        """Written 1 mm further and reset, after 500 particles of something else: one copy of its 180 radii, where
        three of all 680 were taken under opening adhesion and two under contact; laid 1 mm further off the face, the
        tab and the end it clamps not, each bond at the radius it now holds."""
        from robosimstudio_materials.tape import LABEL, PAPER_LABEL

        for material in (PAPER_LABEL, LABEL):
            with self.subTest(adhesion=material.adhesion):
                label, model, pipeline = _label(material, before=BEFORE)
                self.assertEqual((BEFORE, 180), (int(label.particles[0]), len(label.particles)))
                self.assertEqual(180, len(label._radius), "the sheet's radii, not the model's")
                _write_radius(label, model, 0.001)
                patch, sizes = self._counting(model.particle_radius)
                with patch:
                    label.reset(pipeline, None)
                self.assertEqual([180], sizes, "one copy, of the sheet's radii alone")
                np.testing.assert_allclose(label.radius_moved(), 0.001, atol=1e-7)
                clamped = np.isin(label.particles, label.clamp_particles)
                up = np.where(clamped[:, None], 0.0, np.array([[0.0, 0.0, 0.001]]))
                for state in (pipeline.state_0, pipeline.state_1):
                    np.testing.assert_allclose(state.particle_q.numpy()[label.particles], label.built + up, atol=1e-6)
                laid = label._contact["dmin"] if label.contact else label._laid
                np.testing.assert_allclose(laid.numpy(), material.particle_radius + 0.001, atol=1e-7)
                if not label.contact:
                    anchors = label.anchors + np.array([0.0, 0.0, 0.001])
                    np.testing.assert_allclose(label._anchor.numpy(), anchors, atol=1e-6)


class TestASheetWithNoBondRestsWhereItsContactRestsIt(unittest.TestCase):
    """A tabbed label whose tab ends within half a cell of its far end has no bonded row, so a body with no box
    (every shape its face, :meth:`EntityTape._lay`) leaves it laid on the face with clamps and no bond."""

    def test_a_reset_lays_it_by_its_radius_change_and_it_stays_there(self) -> None:
        """Written from 0.5 to 1.5 mm and reset: laid 1 mm further, the tab's end not, and over 8 substeps of VBD the
        face row stays within 0.5 mm of where the contact rests it (0.13 mm, and 0.21 mm over 32, ``f6_probe.py``).
        Left where it was built, as a reset left it while the radius was held per bond, the row started 1 mm inside
        the contact and was thrown 1.4 mm past it by the fourth substep and 2.3 mm by the eighth."""
        import newton

        from robosimstudio_materials.tape import PAPER_LABEL

        label, model, pipeline = _label(PAPER_LABEL, drum=True, tab=0.0935, tab_size=(0.044, 0.008, 0.1))
        self.assertEqual((0, 90), (len(label.bonds), len(label.clamps)), "clamps and no bond")
        self.assertGreater(len(label.lift_shapes), 0, "laid on the drum's face")
        _write_radius(label, model, 0.001)
        label.reset(pipeline, None)
        s0, s1 = pipeline.state_0, pipeline.state_1
        clamped = np.isin(label.particles, label.clamp_particles)
        up = np.where(clamped[:, None], 0.0, np.array([[0.0, 0.0, 0.001]]))
        np.testing.assert_allclose(s0.particle_q.numpy()[label.particles], label.built + up, atol=1e-6)
        for state in (s0, s1):
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        solver = newton.solvers.SolverVBD(model, iterations=10, rigid_compliant_alm=True)
        collide = newton.CollisionPipeline(model, soft_contact_gap=0.005)
        contacts, control, dt = collide.contacts(), model.control(), 1.0 / 960.0
        face = label.built[:, 2] < 0.052
        radius = PAPER_LABEL.particle_radius + 0.001
        for _ in range(8):
            collide.collide(s0, contacts)
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0
            top = float(s0.body_q.numpy()[label.body][2]) + 0.025
            gap = s0.particle_q.numpy()[label.particles][face, 2] - top - radius
            self.assertLess(float(np.abs(gap).max()), 5.0e-4, "the face row stays where the contact rests it")


class TestTheHandlersAskForATape(unittest.TestCase):
    def test_an_entity_offering_peeled_and_bonded_is_not_a_tape(self) -> None:
        """Offering ``peeled()`` and ``bonded()`` and none of a tape's own reads, it raised AttributeError
        (``_let_go``, ``_holding``), not the KeyError each handler says."""
        from types import SimpleNamespace

        from robosimstudio_materials.tape import _bonded_fraction, _peeled_fraction

        lookalike = SimpleNamespace(peeled=lambda: 0.5, bonded=lambda: 0.5)
        snapshot = SimpleNamespace(entities={"label": lookalike})
        for handler in (_peeled_fraction, _bonded_fraction):
            with self.subTest(handler.__name__), self.assertRaisesRegex(KeyError, "not a tape"):
                handler(snapshot, "label")


if __name__ == "__main__":
    unittest.main()
