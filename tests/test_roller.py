# SPDX-License-Identifier: Apache-2.0
"""``rss.roller`` owns the handle the jaws close on, and refuses a handle it cannot build.

The roller is two bodies, the drum and the handle on its axle, and the handle's
four boxes are what a gripper touches: a contact measure between the hand and the
roller, the **contact** tunable and the segmentation all go through the shapes
the entity collects. Collected off the drum's body alone, the handle was in none
of them. The handle's shoulder stands ``grip`` minus 18 mm over the axle, so a
grip too low for the drum puts the shoulder inside it, or under the axle. The
mass, stated or tuned, is the whole tool's, and the grasp box is the handle's
web. No device: a builder, the entity and a CPU model.
"""

from __future__ import annotations

import math
import types
import unittest

import newton
import numpy as np

from robosimstudio_materials.dough import EntityRoller, handle_boxes, roller

TIER = "fast"
"""Builds one roller into a builder and finalizes it on the CPU."""


def _built(spec):
    entity = EntityRoller(spec)
    builder = newton.ModelBuilder()
    entity.build_into(builder, episode=None)
    model = builder.finalize(device="cpu")
    entity.bind(model)
    return entity, builder, model


class TestTheRollerOwnsItsHandle(unittest.TestCase):
    def test_the_handles_boxes_are_the_rollers_shapes(self) -> None:
        for spins in (True, False):
            with self.subTest(spins=spins):
                entity, builder, model = _built(roller(at=(0.0, 0.0, 0.02), spins=spins))
                labels = [builder.shape_label[int(s)] for s in entity.handle(model).collision_shapes]
                self.assertIn("roller", labels, "the drum")
                for k in range(4):
                    self.assertIn(f"roller_handle_{k}", labels, "what the jaws close on")
                self.assertEqual(len(entity.shapes), builder.shape_count)

    def test_the_contact_tunable_reaches_the_handle(self) -> None:
        entity, builder, _ = _built(roller(at=(0.0, 0.0, 0.02)))
        contact = next(t for t in entity.tunables() if t.name == "contact")
        self.assertIn(f"{builder.shape_count} shape(s)", contact.summary)
        self.assertEqual(entity.body, int(np.asarray(builder.shape_body)[0]), "the pose is still the drum's")


class TestAHandleThatCannotBeBuiltIsRefused(unittest.TestCase):
    def test_the_defaults_build(self) -> None:
        roller()
        roller(radius=0.04, grip=0.07)

    def test_a_grip_that_puts_the_shoulder_under_the_axle(self) -> None:
        with self.assertRaisesRegex(ValueError, "grip"):
            roller(grip=0.015)

    def test_a_drum_that_buries_the_shoulder(self) -> None:
        with self.assertRaisesRegex(ValueError, "grip"):
            roller(radius=0.04)

    def test_a_static_roller(self) -> None:
        """It built a free joint and an axle all the same, a static that rolled."""
        with self.assertRaisesRegex(ValueError, "static"):
            roller(static=True)

    def test_a_roller_not_built_from_a_capsule(self) -> None:
        """An asset, or no shape, built no handle, and the roller collected the bench's shapes as its own."""
        from robosimstudio.specs.asset import AssetSpec, ShapeSpec
        from robosimstudio_materials.dough import RollerSpec

        cylinder = ShapeSpec(kind="cylinder", radius=0.02, half_height=0.05)
        for shape, asset in ((None, AssetSpec(path="roller.obj")), (None, None), (cylinder, None)):
            with self.subTest(shape=shape, asset=asset), self.assertRaisesRegex(ValueError, "capsule"):
                RollerSpec(name="roller", shape=shape, asset=asset)
        builder = newton.ModelBuilder()
        builder.add_shape_box(-1, hx=0.5, hy=0.5, hz=0.01, label="bench")
        entity = EntityRoller(roller())
        entity.collect_shapes(builder)  # before its bodies are built: both -1, the world's
        self.assertEqual([], entity.shapes.tolist(), "the bench is not the roller's")


def _tool(entity) -> list[int]:
    return sorted({entity.body, entity.handle_body})


class TestTheMassIsTheWholeTools(unittest.TestCase):
    """``mass`` set the drum and the handle's 65 g came on top; the tunable reached the drum alone."""

    def test_a_stated_mass_is_drum_and_handle_together(self) -> None:
        for spins in (True, False):
            with self.subTest(spins=spins):
                _, _, free = _built(roller(at=(0.0, 0.0, 0.02), spins=spins))
                entity, _, model = _built(roller(at=(0.0, 0.0, 0.02), spins=spins, mass=0.5))
                masses, unstated = model.body_mass.numpy(), free.body_mass.numpy()
                tool = _tool(entity)
                self.assertAlmostEqual(float(masses[tool].sum()), 0.5, places=5)
                np.testing.assert_allclose(masses[tool] / 0.5, unstated[tool] / unstated[tool].sum(), rtol=1e-5)

    def test_the_tunable_reads_and_writes_the_whole_tool(self) -> None:
        entity, _, model = _built(roller(at=(0.0, 0.0, 0.02)))
        tool = _tool(entity)
        masses, inertia = model.body_mass.numpy().copy(), model.body_inertia.numpy().copy()
        mass = next(t for t in entity.tunables() if t.name == "mass")
        self.assertAlmostEqual(mass.spec.mass, float(masses[tool].sum()), places=6)
        self.assertTrue(mass.write("mass", 2.0 * mass.spec.mass))
        np.testing.assert_allclose(model.body_mass.numpy()[tool], 2.0 * masses[tool], rtol=1e-5)
        np.testing.assert_allclose(model.body_inertia.numpy()[tool], 2.0 * inertia[tool], rtol=1e-5)


class TestTheGraspBoxIsTheHandles(unittest.TestCase):
    """The inherited box was the drum's, which put the grasp point on the axle, inside the handle's shoulder."""

    def test_the_box_is_the_web_on_the_handles_body(self) -> None:
        from robosimstudio.solvers.grasp.obb import grasp_from_obb

        for spins in (True, False):
            with self.subTest(spins=spins):
                spec = roller(at=(0.0, 0.0, 0.02), spins=spins)
                entity, _, model = _built(spec)
                centre, half = handle_boxes(spec.grip)[1]
                self.assertEqual(entity.handle_body, entity.obb.body)
                np.testing.assert_allclose(entity.obb.center_local, centre, atol=1e-6)
                np.testing.assert_allclose(entity.obb.half, half, atol=1e-6)
                body_q = model.body_q.numpy().astype(np.float64)
                point, _ = grasp_from_obb(entity.obb, body_q[entity.obb.body])
                over = float(point[2] - body_q[entity.body][2]) - spec.grip
                self.assertLess(abs(over), 0.005, f"the grasp point is {over * 1000:+.1f} mm off the grip")
                top = spec.grip + 0.025 + spec.shape.radius
                np.testing.assert_allclose(entity.bbox, (0.04, 0.14, top), atol=1e-6, err_msg="the tool's own")

    def test_a_handle_tilted_a_fraction_of_a_degree_keeps_its_heading_and_width(self) -> None:
        """The heuristic's long axis was the web's height: tilted 0.2 degrees, its heading read -90, 90, 0 or 180."""
        from robosimstudio.solvers.grasp.obb import grasp_from_obb

        entity, _, model = _built(roller(at=(0.0, 0.0, 0.02)))
        pose = model.body_q.numpy().astype(np.float64)[entity.obb.body]
        for axis in (0, 1):
            for degrees in (0.2, -0.2):
                with self.subTest(about="xy"[axis], degrees=degrees):
                    quat = np.array([0.0, 0.0, 0.0, math.cos(math.radians(degrees) / 2)])
                    quat[axis] = math.sin(math.radians(degrees) / 2)
                    _, yaw = grasp_from_obb(entity.obb, np.concatenate([pose[:3], quat]))
                    self.assertAlmostEqual(0.0, yaw, places=6, msg="the long axis is the web's x, across the drum")
        self.assertAlmostEqual(0.020, entity.obb.grasp_width, places=6, msg="the 20 mm the jaws close across")
        self.assertTrue(entity.obb.fits_gripper)

    def test_the_tools_place_it_by_the_handle_while_the_drum_rolls(self) -> None:
        from robosimstudio_policy_agent.sim_tools import SimTools

        spec = roller(at=(0.0, 0.0, 0.02))
        entity, _, model = _built(spec)
        handle = entity.handle(model)
        body_q = model.body_q.numpy().astype(np.float64)
        body_q[entity.body, 3:7] = (0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))  # rolled a quarter turn
        tools = types.SimpleNamespace(
            env=types.SimpleNamespace(handles=types.SimpleNamespace(entity=lambda name: handle)),
            _body_q=lambda: body_q,
        )
        bounds = SimTools.get_object_bounds(tools, "roller")
        top = float(body_q[entity.handle_body][2]) + spec.grip + 0.025
        self.assertAlmostEqual(bounds["highest_z"], top, places=5, msg="the web's top, over the axle on the handle")


if __name__ == "__main__":
    unittest.main()
