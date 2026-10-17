# SPDX-License-Identifier: Apache-2.0
"""A cable built with ``dahl_eps_max`` keeps a bend; the same cable built at 0 springs back.

Newton's VBD carries a Dahl hysteresis on the bend and twist of every rod
joint, armed per joint by two model attributes ``model.vbd.dahl_eps_max`` and
``dahl_tau``, and the rod entity authors them from :class:`RodSpec` in the
window after finalize and before the solver exists. Three things have to hold:

* **The rows land where they should**: 2.0 on the rod's own cable joints and 0
  on its anchor weld, and the VBD solver reports its Dahl kernels armed.
* **The memory is real**: two cables side by side, clamped at one end, the
  free end carried round a hairpin and let go -- the plastic one stays bent and
  the elastic one straightens.
* **A reset forgets**: ``sim.reset()`` puts the plastic cable back on its build
  pose and the solver rebaselines its Dahl state, so the next episode starts
  from a straight cable and not from the last one's bend.

The hairpin is the shape that shows the memory, and why is the Dahl law itself:
the friction moment reverses over ``dahl_tau`` radians of unbending as it built
up over ``dahl_tau`` radians of bending, so a joint keeps a bend in proportion
to how far past ``tau`` it was taken. A shallow hook bends each of twenty
joints by a few hundredths of a radian -- under ``tau`` -- and in the
pure-Newton probe this test was designed on the cable kept a fifth of it (tip
0.27 m out during the drive, 0.05 m after release); a hairpin of 50 mm radius
bends each joint it passes through by ~0.3 rad. Measured here, twice, to the
same digits: straightness 0.974 / 0.987 (elastic / plastic) settled, 0.693 /
0.700 at the end of the drive, 0.980 / 0.704 after 240 released steps (0.948 /
0.703 after 120 -- the elastic cable is still ringing then, which is why the
release is 240), and both back on their settled values after ``sim.reset()``.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "medium"
"""Builds two cables on VBD and drives one capsule by hand: a GPU, and a scene."""

LENGTH = 0.30
RADIUS = 0.006
SEGMENTS = 20
HEIGHT = 0.30
"""A light cable in the air, clamped at its start, in the empty cell. Light so
its own sag does not spend the friction envelope: a 0.5 m cable of 8 mm radius at
350 kg/m^3 weighs 35 g and its root moment under gravity (0.086 N*m) is about
the envelope ``bend_ke * dahl_eps_max`` (0.08 N*m) -- that cable, measured in the
probe, kept a tenth of a push. This one weighs 12 g."""

EPS_MAX = 2.0
TAU = 0.1
HAIRPIN = 0.10
"""The free end is carried along a semicircle of this diameter [m], ending
back beside the cable pointing the other way."""

DRIVE_STEPS = 60
RELEASE_STEPS = 240


def _scene(*, num_worlds: int = 1):
    import robosimstudio as rss

    scene = rss.Scene(cell="void", name="dahl_memory")
    for name, x, eps in (("elastic", -0.10, 0.0), ("plastic", 0.10, EPS_MAX)):
        scene.add(
            rss.rod(
                name=name,
                length=LENGTH,
                radius=RADIUS,
                segments=SEGMENTS,
                at=(x, -LENGTH / 2, HEIGHT),
                along=(0.0, 1.0, 0.0),
                anchors=("start",),
                dahl_eps_max=eps,
                dahl_tau=TAU,
            )
        )
    return scene.build(num_worlds=num_worlds)


def _straightness(sim, rod) -> float:
    from robosimstudio_materials.rod.measures import straightness

    return straightness(rod.positions(sim.pipeline.state_0.body_q.numpy()))


def _carry_tip(sim, rods, rest: np.ndarray, frac: float) -> None:
    """Put each cable's tip capsule on the hairpin at ``frac`` of the way round, at rest.

    A kinematic write, as a hand that has the capsule in a hard latch would do:
    the pose is set and the velocity zeroed in both state buffers before the
    step, so the solver sees the tip where the drive put it.
    """
    for state in (sim.pipeline.state_0, sim.pipeline.state_1):
        q = state.body_q.numpy()
        qd = state.body_qd.numpy()
        for rod in rods:
            tip = int(rod.bodies[-1])
            centre = rest[tip, :3] + np.array([0.0, -HAIRPIN / 2, 0.0])
            angle = np.pi * frac
            q[tip, 0] = centre[0] + (HAIRPIN / 2) * np.sin(angle)
            q[tip, 1] = centre[1] + (HAIRPIN / 2) * np.cos(angle)
            qd[tip] = 0.0
        state.body_q.assign(q)
        state.body_qd.assign(qd)


class TestTheRowsLandOnTheCableJoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = _scene()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def _rows(self, name: str):
        import newton

        rod = self.sim.scene.by_name[name]
        model = self.sim.model
        types = model.joint_type.numpy()[rod.joints]
        cable = rod.joints[types == int(newton.JointType.ROD)]
        welds = rod.joints[types == int(newton.JointType.FIXED)]
        return model.vbd.dahl_eps_max.numpy(), model.vbd.dahl_tau.numpy(), cable, welds

    def test_the_plastic_cable_has_its_rows(self) -> None:
        eps, tau, cable, welds = self._rows("plastic")
        self.assertEqual(len(cable), SEGMENTS - 1)
        self.assertEqual(len(welds), 1, "one anchor weld, at the start")
        np.testing.assert_allclose(eps[cable], EPS_MAX, rtol=1e-6)
        np.testing.assert_allclose(tau[cable], TAU, rtol=1e-6)
        np.testing.assert_array_equal(eps[welds], 0.0)
        np.testing.assert_array_equal(tau[welds], 0.0)

    def test_the_elastic_cable_stays_at_zero(self) -> None:
        eps, tau, cable, _ = self._rows("elastic")
        np.testing.assert_array_equal(eps[cable], 0.0)
        np.testing.assert_array_equal(tau[cable], 0.0)

    def test_the_solver_armed_its_dahl_kernels(self) -> None:
        solvers = [s for s in self.sim.pipeline.program.solvers().values() if hasattr(s, "enable_dahl_friction")]
        self.assertEqual(len(solvers), 1, "one VBD solver holds both cables")
        self.assertTrue(solvers[0].enable_dahl_friction)

    def test_the_drive_tunable_reads_and_writes_the_rows(self) -> None:
        rod = self.sim.scene.by_name["plastic"]
        drive = next(t for t in rod.tunables(self.sim.pipeline) if t.name == "drive")
        self.assertAlmostEqual(drive.spec.dahl_eps_max, EPS_MAX, places=5)
        self.assertAlmostEqual(drive.spec.dahl_tau, TAU, places=5)
        self.assertIn("inert when built at 0", drive.summary)
        try:
            self.assertTrue(drive.write("dahl_tau", 0.05))
            _, tau, cable, welds = self._rows("plastic")
            np.testing.assert_allclose(tau[cable], 0.05, rtol=1e-6)
            np.testing.assert_array_equal(tau[welds], 0.0)
        finally:
            drive.write("dahl_tau", TAU)


class TestTheRowsLandInEveryWorld(unittest.TestCase):
    def test_two_worlds_both_carry_the_rows(self) -> None:
        import newton

        sim = _scene(num_worlds=2)
        try:
            rod = sim.scene.by_name["plastic"]
            model = sim.model
            eps = model.vbd.dahl_eps_max.numpy()
            types = model.joint_type.numpy()
            cable = rod.joints[types[rod.joints] == int(newton.JointType.ROD)]
            worlds = rod.worlds
            self.assertIsNotNone(worlds)
            for world in range(2):
                rows = [worlds.joint(int(j), world) for j in cable]
                np.testing.assert_allclose(eps[rows], EPS_MAX, rtol=1e-6, err_msg=f"world {world}")
            self.assertEqual(int(np.count_nonzero(eps)), 2 * len(cable), "rows on the cable joints and nowhere else")
        finally:
            sim.close()


class TestThePlasticCableKeepsTheHairpin(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = _scene()
        cls.rods = [cls.sim.scene.by_name[n] for n in ("elastic", "plastic")]
        cls.sim.run(60).sync()
        cls.rest = cls.sim.pipeline.state_0.body_q.numpy().copy()
        cls.at_rest = [_straightness(cls.sim, r) for r in cls.rods]
        for i in range(DRIVE_STEPS):
            _carry_tip(cls.sim, cls.rods, cls.rest, (i + 1) / DRIVE_STEPS)
            cls.sim.run(1)
        cls.sim.sync()
        cls.driven = [_straightness(cls.sim, r) for r in cls.rods]
        cls.sim.run(RELEASE_STEPS).sync()
        cls.released = [_straightness(cls.sim, r) for r in cls.rods]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def test_both_hang_straight_before_the_drive(self) -> None:
        for name, value in zip(("elastic", "plastic"), self.at_rest, strict=True):
            self.assertGreater(value, 0.95, name)

    def test_the_drive_bent_both(self) -> None:
        for name, value in zip(("elastic", "plastic"), self.driven, strict=True):
            self.assertLess(value, 0.8, name)

    def test_the_elastic_cable_springs_back(self) -> None:
        self.assertGreater(self.released[0], 0.95, f"elastic: {self.released}")

    def test_the_plastic_cable_stays_bent(self) -> None:
        self.assertLess(self.released[1], 0.9, f"plastic: {self.released}")

    def test_reset_forgets_the_bend(self) -> None:
        self.sim.reset()
        self.sim.run(60).sync()
        for name, rod in zip(("elastic", "plastic"), self.rods, strict=True):
            self.assertGreater(_straightness(self.sim, rod), 0.95, name)


if __name__ == "__main__":
    unittest.main()
