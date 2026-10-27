# SPDX-License-Identifier: Apache-2.0
"""The capsule chain is built through Newton's ``newton.Rod`` API, and it is the same chain.

``ModelBuilder.add_rod(positions=...)`` and
``newton.utils.create_straight_cable_points_and_quaternions`` are deprecated in
Newton 1.6 (removal slated); the rod entity now builds a :class:`newton.Rod`
and passes ``add_rod(rod=...)`` with all eight stiffness and damping numbers.
Three claims, on the CPU builder with no scene:

* **No deprecation warning** mentioning either old form comes out of a build.
* **The straight spawn is the chain its spec states**: capsules laid end to
  end from ``start`` along ``direction``, each ``length / segments`` long, of
  the spec's radius and weighing its density times its volume, joined by one
  joint fewer than there are capsules. A polyline keeps its nodes.
* **The twist slot is its own**: ``joint_target_ke`` rows 2 and 3 of each cable
  joint equal ``bend_ke`` and ``twist_ke`` when the spec splits them, and are
  equal when ``twist_ke`` is ``None``; the same for shear against stretch.

The roll of two hanging cables, which needs a GPU, is in ``test_rod_twist``.
"""

from __future__ import annotations

import unittest
import warnings
from types import SimpleNamespace

import numpy as np

TIER = "fast"
"""The CPU builder and a CPU finalize: no scene and no GPU."""


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


class TestTheStraightSpawn(unittest.TestCase):
    """What :class:`RodSpec` states, read back off the built model."""

    def test_the_capsules_lie_end_to_end_along_the_direction(self) -> None:
        from robosimstudio.core.math import quat_to_mat

        for direction in ((0.0, 1.0, 0.0), (0.6, 0.0, -0.8)):
            with self.subTest(direction=direction):
                entity = _entity(start=(0.1, -0.2, 0.3), direction=direction, length=0.5, segments=10)
                model, _ = _build(entity)
                q = model.body_q.numpy().astype(np.float64)
                step = 0.5 / 10
                centres = np.asarray((0.1, -0.2, 0.3)) + np.outer(np.arange(10) + 0.5, step * np.asarray(direction))
                np.testing.assert_allclose(q[:, :3], centres, atol=1e-6)
                for quat in q[:, 3:7]:
                    np.testing.assert_allclose(quat_to_mat(quat)[:, 2], direction, atol=1e-6)
                np.testing.assert_allclose(
                    model.shape_scale.numpy()[:, :2], [[entity.spec.radius, step / 2]] * 10, rtol=1e-5
                )
                self.assertEqual(9, len(range(*entity._rod_spans[1])), "one joint between each pair of capsules")

    def test_each_capsule_weighs_its_density_times_its_volume(self) -> None:
        entity = _entity(length=0.4, segments=8, radius=0.005, density=900.0)
        model, _ = _build(entity)
        r, half = 0.005, 0.4 / 8 / 2
        volume = np.pi * r * r * 2.0 * half + 4.0 / 3.0 * np.pi * r**3
        np.testing.assert_allclose(model.body_mass.numpy(), 900.0 * volume, rtol=1e-5)


class TestTheStraightSpawnIsUnchanged(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
