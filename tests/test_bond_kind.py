# SPDX-License-Identifier: Apache-2.0
"""``robosimstudio_materials/bond.py``: the rule a bond breaks by, the rows a set of bonds writes, and what it refuses.

No GPU and no scene. The physics -- blocks drawn apart at N times the strength, a strip peeled row by
row, a joint held at 80% for ten seconds -- is ``tests/test_bond_pull``, ``test_bond_peel`` and
``test_bond_hold``, on a GPU. Here:

* **The rule**, on forces written by hand: tension along the normal and shear across it are weighed
  against their own strengths, a push along the normal is no load at any size, and a bond with no
  normal breaks on the size of the force.
* **The kernels**, on Warp's CPU device over a constraint list written by hand: they read each bond's
  three rows wherever the solver put them, agree with the rule to float32, switch off the row of a
  bond past its strength in all three places a row is held, and never switch one back on.
* **The rows**, on a bare ``ModelBuilder``: a bond's anchor and normal are in body ``a``'s frame, the
  bonded pair stops colliding, and the pieces take the impedance the spec states.
* **The declarations**: what a spec refuses, what ``rss.bond`` makes, and the flat names.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "fast"
"""Numpy, the spec classes and two small kernels on Warp's CPU device."""


class TestTheRule(unittest.TestCase):
    """``bond_load``: what is compared with a bond's strength."""

    def test_tension_is_the_pull_against_the_normal(self) -> None:
        from robosimstudio_materials.bond import bond_load

        # The force is the constraint's on body a; a bond in tension pulls a back, against the normal.
        self.assertAlmostEqual(1.5, bond_load((0.0, 0.0, -1.5), (0.0, 0.0, 1.0), 2.0))

    def test_a_push_along_the_normal_is_no_load(self) -> None:
        from robosimstudio_materials.bond import bond_load

        for push in (0.1, 5.0, 5.0e3):
            with self.subTest(push=push):
                self.assertEqual(0.0, bond_load((0.0, 0.0, push), (0.0, 0.0, 1.0), 2.0))

    def test_shear_is_weighed_against_its_own_strength(self) -> None:
        from robosimstudio_materials.bond import bond_load

        # 3 N of shear on a bond that holds 0.35 N of tension and 3 N of shear is exactly at its strength.
        self.assertAlmostEqual(0.35, bond_load((3.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.35, 3.0))
        self.assertAlmostEqual(
            3.0, bond_load((0.0, 3.0, 0.0), (0.0, 0.0, 1.0), 2.0), msg="no shear strength: the tension's"
        )

    def test_tension_and_shear_add_in_quadrature(self) -> None:
        from robosimstudio_materials.bond import bond_load

        self.assertAlmostEqual(5.0, bond_load((3.0, 0.0, -4.0), (0.0, 0.0, 1.0), 2.0))
        self.assertAlmostEqual(
            3.0, bond_load((3.0, 0.0, 4.0), (0.0, 0.0, 1.0), 2.0), msg="pushed together: shear alone"
        )

    def test_a_bond_with_no_normal_breaks_on_the_size_of_the_force(self) -> None:
        from robosimstudio_materials.bond import bond_load

        self.assertAlmostEqual(13.0, bond_load((3.0, 4.0, 12.0)))
        self.assertAlmostEqual(
            13.0, bond_load((-3.0, -4.0, -12.0)), msg="compression counts where nothing says which way is off"
        )


def _rows(forces: np.ndarray, equalities: np.ndarray, order: np.ndarray, extra: int = 2):
    """A solver's constraint list holding ``forces`` ``(n, 3)`` for the equalities ``equalities``, bonds in ``order``.

    ``extra`` rows of another type follow, and stale rows sit past the count: what the kernel must not read.
    """
    count = 3 * len(order)
    width = count + extra + 6
    row_type = np.full((1, width), 3, dtype=np.int32)
    row_id = np.zeros((1, width), dtype=np.int32)
    row_force = np.full((1, width), 99.0, dtype=np.float32)
    for slot, bond in enumerate(order):
        row_type[0, 3 * slot : 3 * slot + 3] = 0
        row_id[0, 3 * slot : 3 * slot + 3] = equalities[bond]
        row_force[0, 3 * slot : 3 * slot + 3] = forces[bond]
    # past the count: rows a longer list left behind, typed as equalities of a bond
    row_type[0, count + extra :] = 0
    row_id[0, count + extra :] = equalities[0]
    return np.array([count], dtype=np.int32), row_type, row_id, row_force


class _Rig:
    """The two kernels over hand-written arrays, on Warp's CPU device."""

    def __init__(self, normals, strength, shear, body_q=None, equalities=None) -> None:
        import warp as wp

        self.wp = wp
        n = len(normals)
        self.n = n
        self.equalities = np.arange(n, dtype=np.int32) + 2 if equalities is None else np.asarray(equalities, np.int32)
        bond_of = np.full(int(self.equalities.max()) + 3, -1, dtype=np.int32)
        bond_of[self.equalities] = np.arange(n, dtype=np.int32)
        body_q = np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (2, 1)) if body_q is None else np.asarray(body_q)

        def array(values, dtype):
            return wp.array(np.asarray(values), dtype=dtype, device="cpu")

        self.bond_of = array(bond_of, wp.int32)
        self.body_q = array(body_q.astype(np.float32), wp.transform)
        self.body = array(np.ones(n, dtype=np.int32), wp.int32)
        self.normal = array(np.asarray(normals, dtype=np.float32).reshape(-1, 3), wp.vec3)
        self.strength = array(np.asarray(strength, dtype=np.float32), wp.float32)
        self.shear = array(np.asarray(shear, dtype=np.float32), wp.float32)
        self.equality = array(self.equalities, wp.int32)
        self.row = array(np.arange(n, dtype=np.int32) + 5, wp.int32)
        self.force = wp.zeros((n, 3), dtype=wp.float32, device="cpu")
        self.load = wp.zeros(n, dtype=wp.float32, device="cpu")
        self.peak = wp.zeros(n, dtype=wp.float32, device="cpu")
        self.broken = wp.zeros(n, dtype=wp.int32, device="cpu")
        self.broke_at = wp.zeros(n, dtype=wp.float32, device="cpu")
        self.active = wp.array(np.ones((1, len(bond_of)), dtype=bool), dtype=wp.bool, device="cpu")
        self.enabled = wp.array(np.ones(n + 5, dtype=bool), dtype=wp.bool, device="cpu")
        self.enabled_too = wp.array(np.ones(n + 5, dtype=bool), dtype=wp.bool, device="cpu")

    def step(self, forces, order=None) -> None:
        from robosimstudio_materials.bond import _break_kernel, _gather_kernel

        wp = self.wp
        forces = np.asarray(forces, dtype=np.float32).reshape(-1, 3)
        held = [b for b in (range(self.n) if order is None else order) if not self.broken.numpy()[b]]
        count, row_type, row_id, row_force = _rows(forces, self.equalities, np.asarray(held, dtype=np.int64))
        self.force.zero_()
        wp.launch(
            _gather_kernel,
            dim=row_force.shape[1],
            inputs=[
                wp.array(count, dtype=wp.int32, device="cpu"),
                wp.array(row_type, dtype=wp.int32, device="cpu"),
                wp.array(row_id, dtype=wp.int32, device="cpu"),
                wp.array(row_force, dtype=wp.float32, device="cpu"),
                self.bond_of,
            ],
            outputs=[self.force],
            device="cpu",
        )
        wp.launch(
            _break_kernel,
            dim=self.n,
            inputs=[
                self.force,
                self.body_q,
                self.body,
                self.normal,
                self.strength,
                self.shear,
                self.equality,
                self.row,
            ],
            outputs=[self.load, self.peak, self.broken, self.broke_at, self.active, self.enabled, self.enabled_too],
            device="cpu",
        )


class TestTheKernels(unittest.TestCase):
    """``_gather_kernel`` and ``_break_kernel`` over a constraint list written by hand."""

    def test_each_bond_reads_its_own_rows_wherever_they_lie(self) -> None:
        """The solver writes the rows in whatever order its threads arrive: the id says whose each is."""
        forces = np.array([[1.0, 2.0, -3.0], [4.0, 5.0, -6.0], [7.0, 8.0, -9.0]], dtype=np.float32)
        rig = _Rig(np.zeros((3, 3)), [100.0] * 3, [100.0] * 3)
        rig.step(forces, order=[2, 0, 1])
        np.testing.assert_allclose(rig.force.numpy(), forces)
        np.testing.assert_allclose(rig.load.numpy(), np.linalg.norm(forces, axis=1), rtol=1e-6)
        self.assertEqual([0, 0, 0], rig.broken.numpy().tolist())

    def test_the_kernel_agrees_with_the_rule(self) -> None:
        """Random forces, normals and body turns: the load is ``bond_load``'s to float32."""
        from robosimstudio_materials.bond import bond_load

        rng = np.random.default_rng(7)
        n = 40
        forces = rng.normal(size=(n, 3)) * 3.0
        normals = rng.normal(size=(n, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        normals[::5] = 0.0  # every fifth bond has no normal
        strength = rng.uniform(50.0, 60.0, n)
        shear = rng.uniform(5.0, 500.0, n)
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        body_q = np.array([[0, 0, 0, 0, 0, 0, 1], [0.3, -0.2, 0.5, *quat]], dtype=np.float64)
        rig = _Rig(normals, strength, shear, body_q=body_q)
        rig.step(forces)
        x, y, z, w = quat
        turn = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        expected = [
            bond_load(f, None if not n_.any() else turn @ n_, s, sh)
            for f, n_, s, sh in zip(forces, normals, strength, shear, strict=True)
        ]
        np.testing.assert_allclose(rig.load.numpy(), expected, rtol=2e-5, atol=1e-6)

    def test_a_bond_past_its_strength_is_switched_off_in_all_three_places(self) -> None:
        normals = np.tile([0.0, 0.0, 1.0], (3, 1))
        rig = _Rig(normals, [2.0, 2.0, 2.0], [2.0, 2.0, 2.0])
        rig.step([[0.0, 0.0, -1.9], [0.0, 0.0, -2.1], [0.0, 0.0, 50.0]])
        self.assertEqual([0, 1, 0], rig.broken.numpy().tolist(), "1.9 N holds, 2.1 N breaks, 50 N of push holds")
        self.assertAlmostEqual(2.1, float(rig.broke_at.numpy()[1]), places=5)
        active = rig.active.numpy()[0]
        self.assertEqual([True, False, True], active[rig.equalities].tolist())
        self.assertEqual(1, int((~active).sum()), "no other equality of the solver's was touched")
        for held in (rig.enabled.numpy(), rig.enabled_too.numpy()):
            self.assertEqual([True, False, True], held[5:8].tolist())
            self.assertEqual(1, int((~held).sum()))

    def test_a_broken_bond_never_comes_back(self) -> None:
        """Nothing in the kernels writes a row on: unloaded, pushed together or pulled again, it stays gone."""
        normals = np.tile([0.0, 0.0, 1.0], (2, 1))
        rig = _Rig(normals, [2.0, 2.0], [2.0, 2.0])
        rig.step([[0.0, 0.0, -3.0], [0.0, 0.0, -1.0]])
        self.assertEqual([1, 0], rig.broken.numpy().tolist())
        for forces in ([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]], [[0.0, 0.0, 9.0], [0.0, 0.0, -1.0]]):
            rig.step(forces)
            self.assertEqual([1, 0], rig.broken.numpy().tolist())
            self.assertFalse(bool(rig.active.numpy()[0][rig.equalities[0]]))
            self.assertFalse(bool(rig.enabled.numpy()[5]))
            self.assertEqual(0.0, float(rig.load.numpy()[0]), "a broken bond carries no load")
        np.testing.assert_allclose(rig.peak.numpy(), [3.0, 1.0], rtol=1e-6)

    def test_a_row_past_the_count_is_not_read(self) -> None:
        """The solver's list keeps what a longer step left behind; only the first ``ne`` rows are this step's."""
        rig = _Rig(np.tile([0.0, 0.0, 1.0], (1, 1)), [2.0], [2.0])
        rig.step([[0.0, 0.0, -1.0]])  # the stale rows past the count hold 99 N for this bond
        self.assertEqual([0], rig.broken.numpy().tolist())
        np.testing.assert_allclose(rig.force.numpy(), [[0.0, 0.0, -1.0]])


class TestTheRows(unittest.TestCase):
    """What ``EntityBond.build`` writes on a bare builder."""

    def _built(self, **kwargs):
        import newton
        import warp as wp
        from newton.solvers import SolverMuJoCo

        from robosimstudio_materials.bond import EntityBond, bond

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * np.pi)
        lid = builder.add_body(xform=wp.transform((0.1, 0.2, 0.33), turn), label="scene/lid")
        builder.add_shape_box(lid, hx=0.02, hy=0.02, hz=0.01)
        box = builder.add_body(xform=wp.transform((0.1, 0.2, 0.30), wp.quat_identity()), label="scene/box")
        builder.add_shape_box(box, hx=0.02, hy=0.02, hz=0.02)
        builder.add_shape_sphere(box, radius=0.01)
        spec = bond(
            name="glue",
            a="lid",
            b="box",
            at=[(0.11, 0.2, 0.32), (0.09, 0.2, 0.32)],
            normal=(0.0, 0.0, 1.0),
            strength=2.0,
            **kwargs,
        )
        entity = EntityBond(spec)
        entity.build(builder, None)
        return builder, entity, lid, box

    def test_a_bond_is_a_connect_row_anchored_in_body_a(self) -> None:
        builder, entity, lid, box = self._built()
        table = builder.custom_attributes
        rows = entity.rows.tolist()
        self.assertEqual(2, len(rows))
        for row, expected in zip(rows, ([0.0, -0.01, -0.01], [0.0, 0.01, -0.01]), strict=True):
            self.assertEqual(0, int(table["mujoco:equality_constraint_type"].values[row]), "CONNECT")
            self.assertEqual(lid, int(table["mujoco:equality_constraint_body1"].values[row]))
            self.assertEqual(box, int(table["mujoco:equality_constraint_body2"].values[row]))
            # The lid is turned a quarter about z: a point 10 mm along world x from its centre is -y in its frame.
            np.testing.assert_allclose(
                np.asarray(table["mujoco:equality_constraint_anchor"].values[row]), expected, atol=1e-6
            )
            self.assertTrue(bool(table["mujoco:equality_constraint_enabled"].values[row]))
        self.assertEqual(["glue/0", "glue/1"], entity.labels)
        np.testing.assert_allclose(entity.normals, [[0.0, 0.0, 1.0]] * 2, atol=1e-6)

    def test_a_normal_is_kept_in_body_a_frame(self) -> None:
        import newton
        import warp as wp
        from newton.solvers import SolverMuJoCo

        from robosimstudio_materials.bond import EntityBond, bond

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * np.pi)
        builder.add_body(xform=wp.transform((0.0, 0.0, 0.3), turn), label="tab")
        entity = EntityBond(bond(name="pin", a="tab", at=(0.0, 0.0, 0.3), normal=(2.0, 0.0, 0.0)))
        entity.build(builder, None)
        np.testing.assert_allclose(
            entity.normals, [[0.0, -1.0, 0.0]], atol=1e-6, err_msg="world x, in a frame turned a quarter"
        )
        self.assertEqual(-1, int(builder.custom_attributes["mujoco:equality_constraint_body2"].values[0]), "the world")

    def test_bonded_bodies_stop_colliding_unless_asked(self) -> None:
        builder, _, lid, box = self._built()
        pairs = {tuple(sorted(pair)) for pair in builder.shape_collision_filter_pairs}
        for a in builder.body_shapes[lid]:
            for b in builder.body_shapes[box]:
                self.assertIn(tuple(sorted((a, b))), pairs)
        builder, _, lid, box = self._built(collide=True)
        pairs = {tuple(sorted(pair)) for pair in builder.shape_collision_filter_pairs}
        self.assertFalse(
            any(tuple(sorted((a, b))) in pairs for a in builder.body_shapes[lid] for b in builder.body_shapes[box])
        )

    def test_the_pieces_take_the_stated_impedance(self) -> None:
        from robosimstudio_materials.bond import PIECE_IMPEDANCE

        builder, _, lid, box = self._built(piece_impedance=PIECE_IMPEDANCE, piece_mix=50.0)
        solimp, solmix = (
            builder.custom_attributes["mujoco:geom_solimp"],
            builder.custom_attributes["mujoco:geom_solmix"],
        )
        shapes = [*builder.body_shapes[lid], *builder.body_shapes[box]]
        self.assertEqual(3, len(shapes))
        for shape in shapes:
            np.testing.assert_allclose(np.asarray(solimp.values[shape]), PIECE_IMPEDANCE)
            self.assertEqual(50.0, float(solmix.values[shape]))
        builder, _, _, _ = self._built()
        self.assertFalse(builder.custom_attributes["mujoco:geom_solimp"].values, "left as built unless stated")

    def test_a_name_that_is_not_one_body_is_refused(self) -> None:
        import newton

        from robosimstudio_materials.bond import EntityBond, bond

        builder = newton.ModelBuilder()
        builder.add_body(label="lid")
        entity = EntityBond(bond(name="glue", a="lid", b="box", at=(0.0, 0.0, 0.0)))
        with self.assertRaisesRegex(KeyError, "'box' names 0 bodies"):
            entity.build(builder, None)


class TestTheDeclarations(unittest.TestCase):
    def test_a_point_refuses_what_cannot_be_a_bond(self) -> None:
        from robosimstudio_materials.bond import BondPoint

        for kwargs, why in (
            ({"a": ""}, "the body that comes away"),
            ({"a": "lid", "b": "lid"}, "both ends name"),
            ({"a": "lid", "strength": 0.0}, "strength must be > 0"),
            ({"a": "lid", "shear_strength": -1.0}, "shear_strength must be > 0"),
            ({"a": "lid", "normal": (0.0, 0.0, 0.0)}, "normal must have a length"),
        ):
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, why):
                BondPoint(**kwargs)

    def test_the_constructor_makes_one_bond_a_point(self) -> None:
        from robosimstudio_materials.bond import BondPoint, BondSpec, bond

        extra = BondPoint(a="cap", b="band", at=(0.0, 0.0, 0.1), strength=0.35, shear_strength=3.0, group="seam")
        spec = bond(
            name="glue",
            a="lid",
            b="box",
            at=[(0.01, 0.0, 0.1), (-0.01, 0.0, 0.1)],
            normal=(0, 0, 1),
            strength=2.0,
            points=[extra],
        )
        self.assertIsInstance(spec, BondSpec)
        self.assertEqual("bond", spec.entity)
        self.assertEqual(3, len(spec.points))
        self.assertEqual([("lid", "box", 2.0)] * 2, [(p.a, p.b, p.strength) for p in spec.points[:2]])
        self.assertIs(extra, spec.points[2])
        self.assertEqual(("lid", "box", "cap", "band"), spec.bodies)
        self.assertEqual(1, len(bond(name="pin", a="tab", at=(0.0, 0.0, 0.3)).points), "one point is one bond")
        self.assertEqual(("tab",), bond(name="pin", a="tab", at=(0.0, 0.0, 0.3)).bodies, "the world is no body")

    def test_a_set_refuses_what_is_not_a_point(self) -> None:
        from robosimstudio_materials.bond import BondSpec

        with self.assertRaisesRegex(TypeError, "points are BondPoint"):
            BondSpec(name="glue", points=((0.0, 0.0, 0.0),))
        with self.assertRaisesRegex(ValueError, "needs a name"):
            BondSpec(name="")

    def test_it_composes_in_the_rigid_domain_and_builds_after_the_objects(self) -> None:
        from robosimstudio.specs.asset import build_stage
        from robosimstudio_materials.bond import EntityBond, bond

        spec = bond(name="glue", a="lid", at=(0.0, 0.0, 0.0))
        self.assertEqual("extras", build_stage(spec))
        self.assertEqual("rigid", EntityBond(spec).sim_requirements().domain)

    def test_the_flat_names_are_the_modules(self) -> None:
        import robosimstudio as rss
        from robosimstudio.scene.registry import ENTITIES
        from robosimstudio_materials import bond as module

        for name in ("BondPoint", "BondSpec", "bond"):
            self.assertIs(getattr(module, name), getattr(rss, name))
        self.assertIs(module.EntityBond, ENTITIES.get("bond"))


if __name__ == "__main__":
    unittest.main()
