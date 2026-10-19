# SPDX-License-Identifier: Apache-2.0
"""The brush kind as registered knowledge: kind, entity, materials, flat names, spec and measure, on the CPU.

``robosimstudio_materials/brush.py`` is one module built on the rod kind's
capsule chain. What the registries and the library know of it before any scene
is built is checked here, with no GPU: the kind and its fallback to the rod
layout, where ``GENERIC`` and ``BRUSH_60HZ`` route it, the presets, the flat
names, the spec's validation and arithmetic, the coverage measure on tips laid
out by hand, the atom and its handler, and the import law (the brush names the
rod kind, the rod kind never names the brush). ``tests/test_brush_kind``
builds it on a GPU; ``tests/test_scrub_bench`` drives the examples' task.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

TIER = "fast"
"""No GPU: registries, dataclasses and lattices on the host."""

REPO = Path(__file__).resolve().parents[1]
MATERIALS = REPO / "robosimstudio_materials"


class TestTheKindRegisters(unittest.TestCase):
    def test_the_domain_reads_as_a_brush_then_a_rod(self) -> None:
        from robosimstudio.specs.physics import get_domain
        from robosimstudio.specs.registries import layout_chain, reads_as

        info = get_domain("brush")
        self.assertEqual(("brush", "rod"), info.reads)
        self.assertEqual(("bodies", "joints", "shapes"), info.state)
        self.assertEqual("deformables", info.role)
        self.assertEqual(("brush", "rod"), layout_chain("brush"))
        self.assertTrue(reads_as("brush", "rod"))
        self.assertFalse(reads_as("rod", "brush"))

    def test_the_entity_is_registered_under_its_name(self) -> None:
        from robosimstudio.scene.registry import get_entity
        from robosimstudio_materials.brush import EntityBrush

        self.assertIs(EntityBrush, get_entity("brush"))
        self.assertEqual("brush", EntityBrush.domain)
        self.assertTrue(EntityBrush.world_aware)

    def test_generic_puts_the_bristles_on_vbd_and_the_brush_table_the_handle_on_the_arms_solver(self) -> None:
        from robosimstudio.specs.physics import GENERIC
        from robosimstudio_materials.brush import BRUSH_60HZ

        self.assertEqual("vbd", GENERIC.backend_for("brush").solver)
        self.assertEqual(GENERIC.backend_for("brush"), BRUSH_60HZ.backend_for("brush"))
        self.assertEqual(BRUSH_60HZ.backend_for("articulated"), BRUSH_60HZ.backend_for("rigid"))
        self.assertEqual("mujoco", BRUSH_60HZ.backend_for("rigid").solver)
        self.assertEqual((60.0, 16), (BRUSH_60HZ.control_hz, BRUSH_60HZ.substeps))

    def test_the_materials_and_presets(self) -> None:
        from robosimstudio.assetlib.materials import MATERIAL_KINDS, get_material
        from robosimstudio_materials.brush import NYLON, SOFT, BrushMaterialSpec, presets, resolve_material

        self.assertIs(BrushMaterialSpec, MATERIAL_KINDS.get("brush"))
        self.assertEqual({"nylon": NYLON, "soft": SOFT}, presets)
        self.assertIs(NYLON, get_material("nylon_bristle"))
        self.assertIs(SOFT.spec, resolve_material("soft"))
        self.assertIs(NYLON.spec, resolve_material(NYLON))
        self.assertIs(SOFT.spec, resolve_material("soft_bristle"))
        self.assertGreater(NYLON.spec.bend_ke, 5.0 * SOFT.spec.bend_ke, "a dish brush is stiffer than a paint brush")
        with self.assertRaises(TypeError):
            resolve_material(3.0)

    def test_the_drive_gains_pair_shear_with_stretch_and_twist_with_bend(self) -> None:
        from robosimstudio_materials.brush import BrushMaterialSpec

        gains = BrushMaterialSpec(bend_ke=0.02, bend_kd=1e-4, stretch_ke=5e3, stretch_kd=0.2).drive_gains()
        self.assertEqual(
            {"stretch_ke", "stretch_kd", "shear_ke", "shear_kd", "bend_ke", "bend_kd", "twist_ke", "twist_kd"},
            set(gains),
        )
        self.assertEqual((gains["shear_ke"], gains["shear_kd"]), (5e3, 0.2))
        self.assertEqual((gains["twist_ke"], gains["twist_kd"]), (0.02, 1e-4))


class TestTheFlatNames(unittest.TestCase):
    def test_rss_carries_the_brush(self) -> None:
        import robosimstudio as rss
        from robosimstudio_materials import brush as module

        self.assertIs(module.brush, rss.brush)
        self.assertIs(module.BrushSpec, rss.BrushSpec)
        self.assertIs(module.BrushMaterialSpec, rss.BrushMaterialSpec)
        self.assertIs(module.BRUSH_60HZ, rss.BRUSH_60HZ)

    def test_the_rod_kind_exports_the_chain_builder_the_brush_uses(self) -> None:
        from robosimstudio_materials import rod
        from robosimstudio_materials.rod.entity import add_capsule_chain

        self.assertIs(add_capsule_chain, rod.add_capsule_chain)

    def test_the_slots_the_brush_and_the_cable_tune_are_the_rod_joints_alone(self) -> None:
        """A free joint, a chain and the weld under it in one span: the four slots are the chain's rows."""
        import newton
        import warp as wp
        from newton.solvers import SolverVBD

        from robosimstudio_materials.rod import add_capsule_chain, rod_slot_dofs

        builder = newton.ModelBuilder()
        SolverVBD.register_custom_attributes(builder)
        grip = builder.add_body(xform=wp.transform_identity())
        builder.add_shape_box(grip, hx=0.01, hy=0.01, hz=0.01)
        chain = newton.Rod.create_straight(
            start=wp.vec3(0.0, 0.0, -0.01),
            direction=wp.vec3(0.0, 0.0, -1.0),
            length=0.02,
            segment_count=4,
            radius=0.001,
        )
        first = builder.body_count
        gains = dict.fromkeys(("stretch", "shear", "bend", "twist"), 1.0)
        add_capsule_chain(
            builder,
            chain,
            cfg=builder.default_shape_cfg,
            color=(1.0, 1.0, 1.0),
            gains={f"{slot}_{k}": v for slot, v in gains.items() for k in ("ke", "kd")},
            label="bristle",
        )
        builder.add_joint_fixed(parent=grip, child=first)
        model = builder.finalize(device="cpu")
        types, starts = model.joint_type.numpy(), model.joint_qd_start.numpy()
        rods = [j for j in range(model.joint_count) if int(types[j]) == int(newton.JointType.ROD)]
        self.assertEqual(3, len(rods))
        slots = rod_slot_dofs(model, range(model.joint_count))
        for slot, rows in enumerate(slots):
            np.testing.assert_array_equal(starts[rods] + slot, rows)


class TestTheSpec(unittest.TestCase):
    def test_the_defaults_count_their_bodies(self) -> None:
        import robosimstudio as rss

        spec = rss.brush()
        self.assertEqual((20, 20, 4), (spec.rows, spec.columns, spec.bristle_segments))
        self.assertEqual(400, spec.bristle_count)
        self.assertEqual(400 * 4 + 20 + 1, spec.body_count, "every capsule, a strip a row and the own grip")
        self.assertEqual(400 * 4 + 20, rss.brush(handle="handle").body_count, "a named handle is not the brush's")
        self.assertEqual(6 * 3 * 4 + 3 + 1, rss.brush(rows=3, columns=6).body_count)

    def test_the_roots_are_a_lattice_inset_half_a_pitch(self) -> None:
        import robosimstudio as rss

        spec = rss.brush(rows=4, columns=5, sole=(0.10, 0.04, 0.008), handle="h")
        roots = spec.roots()
        self.assertEqual((20, 2), roots.shape)
        np.testing.assert_allclose((0.02, 0.01), spec.pitch)
        np.testing.assert_allclose((0.0, 0.0), roots.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose((-0.04, -0.015), roots.min(axis=0))
        np.testing.assert_allclose((0.04, 0.015), roots.max(axis=0))
        self.assertTrue(np.all(roots[:5, 1] == roots[0, 1]), "row-major: the first row shares one y")

    def test_the_heights_stack(self) -> None:
        import robosimstudio as rss

        spec = rss.brush(bristle_length=0.02, bristle_radius=0.0005, sole=(0.08, 0.05, 0.006), handle_radius=0.012)
        self.assertAlmostEqual(0.0205, spec.sole_height)
        self.assertAlmostEqual(0.0265, spec.sole_top)
        self.assertAlmostEqual(0.0385, spec.grip_height)

    def test_a_bad_brush_is_refused_where_it_is_written(self) -> None:
        import robosimstudio as rss

        for kwargs, why in (
            ({"rows": 0}, "no rows"),
            ({"bristle_segments": 1}, "a chain needs two links"),
            ({"bristle_length": 0.004, "bristle_segments": 4}, "segments closer than a diameter"),
            ({"sole": (0.08, 0.05, 0.0)}, "a flat sole"),
            ({"handle_length": 0.02, "handle_radius": 0.011}, "a grip shorter than its own caps"),
            ({"handle_length": 0.10}, "a grip longer than the sole"),
            ({"bristle_radius": 0.0}, "no radius"),
        ):
            with self.subTest(why=why), self.assertRaises(ValueError):
                rss.brush(**kwargs)
        with self.assertRaises(TypeError):
            rss.BrushSpec(material="nylon")
        rss.brush(handle="handle", handle_length=0.5)  # a named handle is not the brush's to size

    def test_the_record_a_document_places(self) -> None:
        import robosimstudio as rss
        from robosimstudio_materials.brush import Brush

        spec = rss.brush(handle="handle")
        record = Brush.from_spec(spec, name="dish_brush")
        self.assertEqual("brush", record.sim.domain)
        self.assertIs(spec, record.build())
        with self.assertRaises(TypeError):
            Brush.from_spec(rss.box(name="b"), name="nope")


class TestTheScrubMeasure(unittest.TestCase):
    """``scrubbed`` and the lattice the entity keeps, on tips laid out by hand."""

    LO = np.array([0.0, 0.0])
    HI = np.array([0.1, 0.1])

    def test_tips_everywhere_in_the_patch_scrub_all_of_it(self) -> None:
        from robosimstudio_materials.brush import scrubbed

        xs, ys = np.meshgrid(np.arange(0.0025, 0.1, 0.005), np.arange(0.0025, 0.1, 0.005))
        self.assertEqual(1.0, scrubbed(np.stack([xs.ravel(), ys.ravel()], axis=1), self.LO, self.HI))

    def test_tips_beside_the_patch_scrub_none_of_it(self) -> None:
        from robosimstudio_materials.brush import scrubbed

        beside = np.array([[0.15, 0.05], [-0.01, 0.05], [0.05, 0.11]])
        self.assertEqual(0.0, scrubbed(beside, self.LO, self.HI))
        self.assertEqual(0.0, scrubbed(np.zeros((0, 2)), self.LO, self.HI))

    def test_a_stroke_scrubs_the_cells_it_crossed(self) -> None:
        from robosimstudio_materials.brush import scrubbed

        # One tip drawn along y at x = 12 mm: one column of twenty cells out of 400.
        line = np.stack([np.full(200, 0.012), np.linspace(0.0, 0.0999, 200)], axis=1)
        self.assertAlmostEqual(20 / 400, scrubbed(line, self.LO, self.HI))
        # Ten such tips 10 mm apart: ten columns, half of the patch's twenty.
        lines = np.concatenate([line + np.array([0.01 * k - 0.007, 0.0]) for k in range(10)])
        self.assertAlmostEqual(0.5, scrubbed(lines, self.LO, self.HI))

    def test_the_entitys_lattice_reads_a_patch_the_same_way(self) -> None:
        from robosimstudio_materials.brush import ScrubGrid, scrubbed

        grid = ScrubGrid.around((0.05, 0.05))
        self.assertEqual((120, 120), grid.cells.shape, "600 mm a side at 5 mm")
        line = np.stack([np.full(200, 0.012), np.linspace(0.0, 0.0999, 200)], axis=1)
        grid.mark(line)
        grid.mark(np.array([[5.0, 5.0]]))  # off the lattice: ignored
        self.assertEqual(400, grid.patch_cells(self.LO, self.HI), "the cells whose centres lie in the 100 mm patch")
        self.assertAlmostEqual(scrubbed(line, self.LO, self.HI), grid.fraction(self.LO, self.HI))
        self.assertEqual(0.0, grid.fraction((2.0, 2.0), (2.1, 2.1)), "a patch off the lattice")
        grid.clear()
        self.assertEqual(0, int(grid.cells.sum()))


class _WholeLatticeRefused:
    """A device lattice that serves slices and refuses a read or write of the whole of it."""

    def __init__(self, array) -> None:
        self.array = array

    def __getitem__(self, key):
        return self.array[key]

    def numpy(self):
        raise AssertionError("read every world's lattice back to use one")

    def assign(self, values):
        raise AssertionError("wrote every world's lattice back to clear one")

    def zero_(self) -> None:
        self.array.zero_()


class TestOneWorldsLattice(unittest.TestCase):
    """``scrub_grid(world)`` and ``clear_scrub(world)`` touch that world's slab on the device and no other."""

    def test_a_world_is_read_and_cleared_on_its_own(self) -> None:
        import warp as wp

        from robosimstudio_materials.brush import EntityBrush, ScrubGrid

        grid = ScrubGrid(origin=(0.0, 0.0), cell=0.005, nx=3, ny=2)
        brush = EntityBrush.__new__(EntityBrush)
        brush._grid = grid
        values = np.repeat(np.arange(1, 4, dtype=np.int32), 6)
        brush._grid_device = _WholeLatticeRefused(wp.array(values, dtype=wp.int32, device="cpu"))
        for world in range(3):
            with self.subTest(world=world):
                np.testing.assert_array_equal(np.full((2, 3), world + 1), brush.scrub_grid(world).cells)
        brush.clear_scrub(1)
        np.testing.assert_array_equal([1] * 6 + [0] * 6 + [3] * 6, brush._grid_device.array.numpy())


class TestTheTipsAreRead(unittest.TestCase):
    """The tips' axes over the whole ``(N, 4)`` array agree with ``quat_to_mat`` taken one tip at a time."""

    def test_the_axes_match_one_quaternion_at_a_time(self) -> None:
        from robosimstudio.core.math import quat_to_mat
        from robosimstudio_materials.brush import _z_axes

        quats = np.random.default_rng(7).normal(size=(64, 4))
        quats[1] = 0.0  # no length: quat_to_mat's identity
        quats[2] *= 1.0e-3  # far off the unit sphere
        expected = np.stack([quat_to_mat(q)[:, 2] for q in quats])
        np.testing.assert_allclose(_z_axes(quats), expected, atol=1e-12)
        self.assertEqual((0, 3), _z_axes(np.zeros((0, 4))).shape)

    def test_a_tip_is_half_a_capsule_along_its_axis(self) -> None:
        from robosimstudio.core.math import quat_to_mat
        from robosimstudio_materials.brush import EntityBrush

        rng = np.random.default_rng(3)
        body_q = np.concatenate([rng.normal(size=(10, 3)), rng.normal(size=(10, 4))], axis=1).astype(np.float32)
        brush = EntityBrush.__new__(EntityBrush)
        brush.worlds, brush.half_height = None, 0.0025
        brush.bristle_bodies = np.array([[0, 2], [5, 7]], dtype=np.int32)
        expected = [body_q[b, :3] + quat_to_mat(body_q[b, 3:7])[:, 2] * 0.0025 for b in (2, 7)]
        np.testing.assert_allclose(brush.tip_positions(body_q), expected, atol=1e-6)


class TestCarriedWithTheHandle(unittest.TestCase):
    def test_each_pose_turns_by_the_library_product(self) -> None:
        """``_carried`` moves poses rigidly; its orientations are ``quat_multiply(turn, q)`` one pose at a time."""
        from robosimstudio.core.math import mat_to_quat, quat_multiply, quat_to_mat
        from robosimstudio_materials.brush import _carried

        rng = np.random.default_rng(11)

        def pose() -> np.ndarray:
            q = rng.normal(size=4)
            return np.concatenate([rng.normal(size=3), q / np.linalg.norm(q)])

        start, end = pose(), pose()
        poses = np.stack([pose() for _ in range(32)])
        out = _carried(poses, start, end)
        turn = quat_to_mat(end[3:7]) @ quat_to_mat(start[3:7]).T
        q = mat_to_quat(turn)
        np.testing.assert_allclose(out[:, 3:7], [quat_multiply(q, p[3:7]) for p in poses], atol=1e-12)
        np.testing.assert_allclose(out[:, :3], (poses[:, :3] - start[:3]) @ turn.T + end[:3], atol=1e-12)
        # The start pose itself lands on the end pose (up to the quaternion's sign).
        (back,) = _carried(start[None], start, end)
        np.testing.assert_allclose(back[:3], end[:3], atol=1e-12)
        self.assertAlmostEqual(1.0, abs(float(np.dot(back[3:7], end[3:7]))), places=12)


class TestTheCatalogue(unittest.TestCase):
    def test_the_atom_and_its_handler(self) -> None:
        from robosimstudio.assetlib.semantics.atoms import get_atom
        from robosimstudio.assetlib.semantics.handlers import handlers_for

        atom = get_atom("scrubbed")
        self.assertEqual(("brush",), atom.args[0].substrates)
        self.assertEqual(("fraction",), atom.returns)
        self.assertTrue(atom.reads_scene)
        self.assertTrue(any("brush" in h.substrates for h in handlers_for("scrubbed")))


class TestTheImportLaw(unittest.TestCase):
    @staticmethod
    def _imports(path: Path) -> set[str]:
        out: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                out.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                out.add(node.module)
        return out

    def test_the_brush_names_the_library_and_the_rod_kind_only(self) -> None:
        names = self._imports(MATERIALS / "brush.py")
        sibling = {n for n in names if n.startswith("robosimstudio_materials")}
        self.assertEqual({"robosimstudio_materials.rod"}, sibling)
        self.assertFalse({n for n in names if n.split(".")[0].startswith("robosimstudio_") and n not in sibling})

    def test_the_rod_kind_never_names_the_brush(self) -> None:
        for path in sorted((MATERIALS / "rod").glob("*.py")):
            with self.subTest(module=path.name):
                self.assertFalse(any("brush" in n for n in self._imports(path)))


if __name__ == "__main__":
    unittest.main()
