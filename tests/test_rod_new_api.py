# SPDX-License-Identifier: Apache-2.0
"""The capsule chain is built through Newton 1.6's ``newton.Rod`` API, and it is the same chain.

``ModelBuilder.add_rod(positions=...)`` and
``newton.utils.create_straight_cable_points_and_quaternions`` are deprecated in
Newton 1.6 (removal slated); the rod entity now builds a :class:`newton.Rod`
and passes ``add_rod(rod=...)`` with all eight stiffness and damping numbers.
Three claims, on the CPU builder with no scene:

* **No deprecation warning** mentioning either old form comes out of a build.
* **The straight spawn is bit-identical** to what the deprecated path built:
  ``body_q``, ``joint_X_p``/``joint_X_c``, masses, ``shape_scale``,
  ``joint_target_ke``/``kd``. The old path is restated here from the entity as
  it stood before the migration (``git show
  materials/sibling:robosimstudio_materials/rod/entity.py``) rather than
  imported, since it is gone.
* **The twist slot is its own**: ``joint_target_ke`` rows 2 and 3 of each cable
  joint equal ``bend_ke`` and ``twist_ke`` when the spec splits them, and are
  equal when ``twist_ke`` is ``None``; the same for shear against stretch.

And one on a GPU: the free ends of two hanging cables, one twisting ten times
more easily than it bends and one ten times less, are rolled a full turn and
let go. Rolled, both carry the same twist in their middle third -- a clamped
cable driven at its tip twists uniformly, 7 of 19 joints' worth of the turn,
whatever its stiffness -- and both stay straight. Released, the stiff cable
springs its roll out within half a second and the soft one keeps most of it
for seconds: the relaxation time is ``twist_kd / twist_ke`` (0.06 / 0.4 =
0.15 s against 0.06 / 0.004 = 15 s), which is where the twist stiffness shows
and the thing a plug on the end of a cable cares about.
"""

from __future__ import annotations

import unittest
import warnings
from types import SimpleNamespace

import numpy as np

TIER = "medium"
"""The first three classes need only the CPU builder; the roll needs a GPU and a scene."""

ARRAYS = (
    "body_q",
    "body_com",
    "body_mass",
    "body_inertia",
    "shape_scale",
    "joint_X_p",
    "joint_X_c",
    "joint_target_ke",
    "joint_target_kd",
    "joint_qd_start",
)


def _episode():
    return SimpleNamespace(overrides={})


def _entity(**fields):
    from robosimstudio_materials.rod import EntityRod, RodSpec

    return EntityRod(RodSpec(name="cable", **fields))


def _build(entity):
    """The entity's own ``build`` on a bare CPU builder, finalized on the CPU."""
    import newton

    builder = newton.ModelBuilder()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        entity.build(builder, _episode())
        model = builder.finalize(device="cpu")
    return model, [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]


def _build_the_old_way(spec):
    """The chain as the entity built it before the migration, with the deprecated calls.

    Wrapped by hand, as :func:`~robosimstudio_materials.rod.entity.add_capsule_chain`
    wraps it: Newton 1.7's own wrapping roots either form with a free joint.
    """
    import newton
    import warp as wp
    from newton.solvers import SolverVBD
    from newton.utils import create_straight_cable_points_and_quaternions

    builder = newton.ModelBuilder()
    SolverVBD.register_custom_attributes(builder)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        points, quats = create_straight_cable_points_and_quaternions(
            start=wp.vec3(*[float(v) for v in spec.start]),
            direction=wp.vec3(*[float(v) for v in spec.direction]),
            length=float(spec.length),
            num_segments=int(spec.segments),
        )
        _, joints = builder.add_rod(
            positions=points,
            quaternions=quats,
            radius=float(spec.radius),
            body_frame_origin="com",
            cfg=newton.ModelBuilder.ShapeConfig(
                density=float(spec.density),
                ke=float(spec.contact_ke),
                kd=float(spec.contact_kd),
                mu=float(spec.mu_base),
                margin=0.0,
                gap=float(spec.contact_gap),
            ),
            color=spec.color,
            stretch_stiffness=float(spec.stretch_ke),
            stretch_damping=float(spec.stretch_kd),
            bend_stiffness=float(spec.bend_ke),
            bend_damping=float(spec.bend_kd),
            label=spec.name,
            wrap_in_articulation=False,
        )
        builder.add_articulation(list(joints), label=f"{spec.name}_articulation")
    return builder.finalize(device="cpu")


def _slots(model, joints):
    """``(ke, kd)`` as ``[n_joints, 4]`` over the cable joints' four dof rows."""
    starts = model.joint_qd_start.numpy()
    ke, kd = model.joint_target_ke.numpy(), model.joint_target_kd.numpy()
    rows = np.asarray([[int(starts[j]) + k for k in range(4)] for j in joints])
    return ke[rows], kd[rows]


class TestNoDeprecatedCall(unittest.TestCase):
    def test_a_build_warns_of_neither_old_form(self) -> None:
        _, warned = _build(_entity())
        for message in warned:
            self.assertNotIn("add_rod(positions=", message)
            self.assertNotIn("create_straight_cable", message)

    def test_a_polyline_build_warns_of_neither_either(self) -> None:
        arc = tuple((0.2 * np.cos(t), 0.2 * np.sin(t), 0.1) for t in np.linspace(0.0, np.pi, 13))
        _, warned = _build(_entity(nodes=arc, radius=0.006))
        for message in warned:
            self.assertNotIn("add_rod(positions=", message)
            self.assertNotIn("create_straight_cable", message)


class TestTheStraightSpawnIsUnchanged(unittest.TestCase):
    def test_every_array_matches_the_deprecated_path(self) -> None:
        entity = _entity()
        new, _ = _build(entity)
        old = _build_the_old_way(entity.spec)
        self.assertEqual(new.body_count, old.body_count)
        self.assertEqual(new.joint_count, old.joint_count)
        for name in ARRAYS:
            with self.subTest(array=name):
                a = np.asarray(getattr(old, name).numpy(), dtype=np.float64)
                b = np.asarray(getattr(new, name).numpy(), dtype=np.float64)
                self.assertEqual(a.shape, b.shape)
                self.assertLessEqual(float(np.max(np.abs(a - b))) if a.size else 0.0, 1e-6, name)

    def test_a_polyline_keeps_its_nodes_and_drops_the_leg_twist(self) -> None:
        """Positions as stated; frames parallel-transported, where the old form aligned each leg on its own.

        Measured on this arc: capsule positions identical to the old build,
        quaternion components up to 0.75 apart -- a rotation about each leg's
        own axis, which the old per-leg frames carried as a rest twist between
        legs that turn and the transported frames do not.
        """
        from robosimstudio_materials.rod.measures import twist_turns

        arc = tuple((0.2 * np.cos(t), 0.2 * np.sin(t), 0.1) for t in np.linspace(0.0, np.pi, 13))
        model, _ = _build(_entity(nodes=arc, radius=0.006))
        q = model.body_q.numpy()
        self.assertEqual(len(q), len(arc) - 1)
        centres = 0.5 * (np.asarray(arc[:-1]) + np.asarray(arc[1:]))
        np.testing.assert_allclose(q[:, :3], centres, atol=1e-6)
        self.assertLess(abs(twist_turns(q[:, 3:7])), 1e-6, "an untwisted polyline is built without rest twist")


class TestTheTwistSlotIsItsOwn(unittest.TestCase):
    def test_none_pairs_twist_with_bend_and_shear_with_stretch(self) -> None:
        entity = _entity()
        model, _ = _build(entity)
        ke, kd = _slots(model, entity.joints if len(entity.joints) else range(*entity._rod_spans[1]))
        spec = entity.spec
        np.testing.assert_allclose(ke[:, 0], spec.stretch_ke)
        np.testing.assert_allclose(ke[:, 1], ke[:, 0])
        np.testing.assert_allclose(ke[:, 2], spec.bend_ke)
        np.testing.assert_allclose(ke[:, 3], ke[:, 2])
        np.testing.assert_allclose(kd[:, 1], kd[:, 0])
        np.testing.assert_allclose(kd[:, 3], kd[:, 2])

    def test_a_stated_twist_and_shear_land_in_their_slots(self) -> None:
        entity = _entity(twist_ke=0.4, twist_kd=0.6, shear_ke=3.0e5, shear_kd=2.0)
        model, _ = _build(entity)
        ke, kd = _slots(model, range(*entity._rod_spans[1]))
        np.testing.assert_allclose(ke[:, 2], 0.04)
        np.testing.assert_allclose(ke[:, 3], 0.4)
        np.testing.assert_allclose(kd[:, 3], 0.6)
        np.testing.assert_allclose(ke[:, 1], 3.0e5)
        np.testing.assert_allclose(kd[:, 1], 2.0)
        np.testing.assert_allclose(ke[:, 0], 2.0e5)

    def test_a_stiffness_given_alone_keeps_its_pairs_damping(self) -> None:
        """Newton pairs a slot's damping with its stiffness only while both are ``None``; the spec resolves that."""
        entity = _entity(twist_ke=0.4)
        model, _ = _build(entity)
        _, kd = _slots(model, range(*entity._rod_spans[1]))
        np.testing.assert_allclose(kd[:, 3], entity.spec.bend_kd)

    def test_drive_gains_spells_the_eight(self) -> None:
        from robosimstudio_materials.rod import RodMaterialSpec, RodSpec

        gains = RodSpec(twist_ke=0.4).drive_gains()
        self.assertEqual(
            set(gains), {f.name for f in RodMaterialSpec.__dataclass_fields__.values()} - {"dahl_eps_max", "dahl_tau"}
        )
        self.assertEqual((gains["shear_ke"], gains["twist_ke"], gains["twist_kd"]), (2.0e5, 0.4, 0.06))


# ---------------------------------------------------------------- the roll
LENGTH = 0.30
RADIUS = 0.006
SEGMENTS = 20
TOP = 0.45
"""The cables hang from a clamp at this height [m], straight down, so gravity
tensions them instead of bending them. Clamped level, a 0.3 m cable at
``bend_ke`` 0.04 droops 156 mm at the tip (straightness 0.974), and rolling the
tip of a drooped cantilever kinks it rather than twisting it: measured
straightness 0.83 for both cables and whole-cable turn counts flipping sign
step to step. Hanging, the roll is a roll."""
BEND_KE = 0.04
ROLL_STEPS = 120
"""A full turn put into the tip over two seconds of scene, as a hand rolling a cable end would."""
UNIFORM_TURNS = 7.0 / (SEGMENTS - 1)
"""What a full turn at the tip leaves in the middle third of a uniformly
twisted chain: the 7 joints between capsules 6 and 13 of 19. Measured 0.368 and
0.369 for the soft and the stiff cable, against 0.3684."""
RELEASE_STEPS = 60
"""Measured after the release, middle-third turns soft / stiff: 0.367 / 0.243
at 5 steps, 0.365 / 0.125 at 10, 0.357 / 0.007 at 30, 0.345 / 0.000 at 60,
0.283 / 0.000 at 240; straightness 1.000 throughout."""


def _roll_scene():
    import robosimstudio as rss

    scene = rss.Scene(cell="void", name="twist")
    for name, x, twist_ke in (("soft_twist", -0.10, BEND_KE / 10), ("stiff_twist", 0.10, BEND_KE * 10)):
        scene.add(
            rss.rod(
                name=name,
                length=LENGTH,
                radius=RADIUS,
                segments=SEGMENTS,
                at=(x, 0.0, TOP),
                along=(0.0, 0.0, -1.0),
                anchors=("start",),
                bend_ke=BEND_KE,
                twist_ke=twist_ke,
            )
        )
    return scene.build()


def _roll_tips(sim, rods, angle: float, *, max_step: float) -> None:
    """Turn each cable's tip capsule about its own axis to ``angle``, through the entity's kinematic hold.

    :meth:`EntityRod.roll` pins the tip -- inverse mass to zero and the kinematic
    flag in the model and every island -- and writes its pose before each step.
    A dynamic tip teleported each step instead injects a twist impulse at a
    capsule whose polar inertia is 1e-8 kg*m^2, and both cables whipped into
    loops within a quarter turn (measured straightness 0.78 at 30 steps).
    """
    for rod in rods:
        rod.roll(sim.pipeline, angle, max_step=max_step)


def _middle_turns(sim, rod) -> float:
    from robosimstudio_materials.rod.measures import twist_turns

    bodies = rod.bodies
    middle = bodies[len(bodies) // 3 : 2 * len(bodies) // 3 + 1]
    return twist_turns(sim.pipeline.state_0.body_q.numpy()[middle, 3:7])


class TestARollTravelsAndAStiffTwistSpringsItOut(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from robosimstudio_materials.rod.measures import straightness

        cls.sim = _roll_scene()
        cls.rods = [cls.sim.scene.by_name[n] for n in ("soft_twist", "stiff_twist")]
        cls.sim.run(60).sync()
        cls.before = [_middle_turns(cls.sim, r) for r in cls.rods]
        for _ in range(ROLL_STEPS):
            _roll_tips(cls.sim, cls.rods, 2.0 * np.pi, max_step=2.0 * np.pi / ROLL_STEPS)
            cls.sim.run(1)
        cls.sim.sync()
        body_q = cls.sim.pipeline.state_0.body_q.numpy()
        cls.rolled = [_middle_turns(cls.sim, r) for r in cls.rods]
        cls.straight_rolled = [straightness(r.positions(body_q)) for r in cls.rods]
        for rod in cls.rods:
            rod.pin(cls.sim.pipeline, on=False)
        cls.sim.run(RELEASE_STEPS).sync()
        body_q = cls.sim.pipeline.state_0.body_q.numpy()
        cls.released = [_middle_turns(cls.sim, r) for r in cls.rods]
        cls.straight_released = [straightness(r.positions(body_q)) for r in cls.rods]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def test_both_start_untwisted(self) -> None:
        for name, turns in zip(("soft", "stiff"), self.before, strict=True):
            self.assertLess(abs(turns), 0.02, name)

    def test_the_roll_reaches_the_middle_of_both(self) -> None:
        """A clamped cable driven at its tip twists uniformly: 7 of 19 joints' worth, whatever its stiffness."""
        for name, turns in zip(("soft", "stiff"), self.rolled, strict=True):
            self.assertAlmostEqual(abs(turns), UNIFORM_TURNS, delta=0.05, msg=f"{name}: {self.rolled}")

    def test_the_stiff_cable_springs_the_roll_out_and_the_soft_one_keeps_it(self) -> None:
        soft, stiff = (abs(t) for t in self.released)
        self.assertLess(stiff, 0.05, f"stiff twist after {RELEASE_STEPS} released steps: {stiff:.3f}")
        self.assertGreater(soft, 0.2, f"soft twist after {RELEASE_STEPS} released steps: {soft:.3f}")

    def test_both_stay_straight(self) -> None:
        for name, value in zip(("soft", "stiff"), [*self.straight_rolled, *self.straight_released], strict=False):
            self.assertGreater(value, 0.98, f"{name}: rolled {self.straight_rolled}, released {self.straight_released}")


if __name__ == "__main__":
    unittest.main()
