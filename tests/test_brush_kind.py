# SPDX-License-Identifier: Apache-2.0
"""The brush built and stepped on a GPU beside the YAM, on its own grip and on a named handle.

What ``tests/test_brush_pack`` claims of the registries, this measures on built
scenes:

* on its own grip under ``GENERIC`` the whole brush is one VBD island beside the
  arm's, captured, standing on its bristles where it was built; a soft field
  cannot hold the same grip up; the bristles' stiffness is a live tunable; a
  reset puts every body back and clears the lattice;
* on a named box under ``BRUSH_60HZ`` the box is on the arm's solver and holds
  itself up, the strips and bristles are on VBD, and the hand closed on the box
  and lowered 4 mm through its IK takes the strips down with it, bows the field
  without a tip going through the bench or past the sole and marks the lattice;
  lifted, the bristles spring back on their strips, and a reset restores them.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import numpy as np

TIER = "medium"
"""A GPU: three brushes built beside the YAM and stepped a few hundred times."""

AT = (0.025, -0.31, 0.0)
"""Where the brushes stand, in the bench's frame [m]: under the right hand, as the examples put them."""

HANDLE_HALF = (0.04, 0.011, 0.015)

PRESS = 0.004
"""How far below where it closed the hand presses [m]: the scrub's own press."""

OVER_BRUSH = (0.0, 2.149, 1.879, -1.3, 0.0, 0.0)
"""The right arm's joints [rad]: tool centre 100 mm over the bench above the brush, pointing down, as in the example."""


def _build(scene, **kwargs):
    """MuJoCo prints its linesearch notes to stdout during a build; a test log does not need them."""
    with redirect_stdout(io.StringIO()):
        return scene.build(control_mode="joint_position", **kwargs)


def _drive(sim, ik, gripper, quat, start, goal, seconds: float) -> None:
    """Move the tool centre from ``start`` to ``goal`` over ``seconds`` through the IK, one control step at a time."""
    frames = max(1, int(seconds * float(sim.request.physics.control_hz)))
    for k in range(frames):
        a = (k + 1) / frames
        ik.set_target(start * (1.0 - a) + goal * a, quat)
        ik.solve_into(sim.state.joint_q, sim.control.joint_target_q)
        gripper.apply(sim.control.joint_target_q)
        sim.run(1)


def _body_q(sim) -> np.ndarray:
    sim.pipeline.sync()
    return np.asarray(sim.pipeline.state_0.body_q.numpy(), dtype=np.float64)


def _in_strips(entity, body_q, *, every: bool = False) -> np.ndarray:
    """Every tip (or every capsule centre) in its own strip's frame: the bristles' shape, whatever the strips do."""
    from robosimstudio.core.math import quat_to_mat

    points = (
        body_q[entity.bristle_bodies.ravel()][:, :3].reshape(*entity.bristle_bodies.shape, 3)
        if every
        else entity.tip_positions(body_q)[:, None, :]
    )
    per = len(points) // len(entity.strip_bodies)
    rows = []
    for r, strip in enumerate(entity.strip_bodies):
        pose = body_q[strip]
        rows.append((points[r * per : (r + 1) * per] - pose[:3]) @ quat_to_mat(pose[3:7]))
    return np.concatenate(rows).reshape(-1, 3)


def _on_handle(body_q, handle: int, bodies) -> np.ndarray:
    """Each of ``bodies``' centres in the handle's frame: where the field sits on the handle, wherever it stands."""
    from robosimstudio.core.math import quat_to_mat

    pose = body_q[handle]
    return (body_q[bodies][:, :3] - pose[:3]) @ quat_to_mat(pose[3:7])


class TestTheBrushOnItsOwnGrip(unittest.TestCase):
    """The whole brush on VBD: the field holds the grip up, as a brush on a bench is held."""

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        scene = rss.Scene(cell="bench_075", name="brush_own_grip")
        scene.add(rss.robot("yam_bimanual"))
        cls.brush = scene.add(rss.brush(name="brush", at=AT))
        cls.sim = _build(scene)
        cls.built = _body_q(cls.sim)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def test_two_islands_captured_with_the_whole_brush_on_vbd(self) -> None:
        self.sim.run(2)
        program = self.sim.pipeline.program
        islands = {i.name: i for i in program.islands}
        self.assertEqual({"mujoco", "vbd"}, set(islands))
        self.assertTrue(self.sim.pipeline.captured, self.sim.pipeline.capture_error)
        entity = self.brush.entity
        self.assertLessEqual(set(entity.bodies.tolist()), set(islands["vbd"].bodies.tolist()))
        self.assertEqual(entity.spec.body_count, len(entity.bodies))
        self.assertEqual((400, 4), entity.bristle_bodies.shape)

    def test_it_stands_on_its_bristles_where_it_was_built(self) -> None:
        entity = self.brush.entity
        rest = entity.tip_positions(self.built)
        self.sim.run(60)
        now = _body_q(self.sim)
        self.assertTrue(np.isfinite(now).all())
        self.assertLess(float(np.linalg.norm(entity.tip_positions(now) - rest, axis=1).max()), 1.0e-3)
        self.assertLess(abs(float(now[entity.handle_body][2] - self.built[entity.handle_body][2])), 5.0e-4)
        floor = entity.floor_z
        tips = entity.tip_positions(now)
        np.testing.assert_allclose(tips[:, 2] - floor, entity.spec.bristle_radius, atol=4.0e-4)

    def test_the_bristles_stiffness_is_tunable(self) -> None:
        tunables = {t.name: t for t in self.brush.entity.tunables(self.sim.pipeline)}
        self.assertEqual({"material", "contact"}, set(tunables))
        material = tunables["material"]
        self.assertAlmostEqual(0.01, material.spec.bend_ke)
        self.assertTrue(material.write("bend_ke", 0.02))
        self.assertAlmostEqual(0.02, self.brush.entity.tunables(self.sim.pipeline)[0].spec.bend_ke)
        self.assertTrue(material.write("bend_ke", 0.01))

    def test_a_reset_puts_it_back_and_forgets_where_it_was(self) -> None:
        entity = self.brush.entity
        state = self.sim.pipeline.state_0
        moved = _body_q(self.sim)
        moved[entity.bodies, 0] += 0.01
        state.body_q.assign(moved.astype(np.float32))
        self.sim.run(10)
        self.sim.reset()
        np.testing.assert_allclose(_body_q(self.sim)[entity.bodies], self.built[entity.bodies], atol=1.0e-6)
        self.assertEqual(0, int(entity.scrub_grid().cells.sum()))


class TestASoftFieldCannotHoldItsGrip(unittest.TestCase):
    def test_a_paint_brush_settles_under_its_own_grip(self) -> None:
        import robosimstudio as rss

        scene = rss.Scene(cell="bench_075", name="brush_soft")
        scene.add(rss.robot("yam_bimanual"))
        brush = scene.add(rss.brush(name="brush", at=AT, material="soft", rows=12, columns=12))
        sim = _build(scene)
        try:
            grip = brush.entity.handle_body
            before = _body_q(sim)[grip][2]
            sim.run(90)
            after = _body_q(sim)
            self.assertTrue(np.isfinite(after).all())
            self.assertGreater(float(before - after[grip][2]), 0.005, "the soft field held the grip up")
        finally:
            sim.close()


class TestTheBrushOnANamedHandle(unittest.TestCase):
    """The examples' brush: a box on the arm's solver, the field on VBD under it, pressed by the hand."""

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss
        from robosimstudio.robots.registry import get_embodiment

        yam = get_embodiment("yam_bimanual")
        left, right = yam.arms
        yam = rss.replace(yam, arms=(left, rss.replace(right, home_q=OVER_BRUSH)))
        scene = rss.Scene(cell="bench_075", name="brush_named", physics=rss.BRUSH_60HZ)
        scene.add(rss.robot(yam))
        spec = rss.brush(name="brush", at=AT, handle="handle")
        scene.add(rss.box(name="handle", half=HANDLE_HALF, at=(AT[0], AT[1], spec.sole_top + HANDLE_HALF[2])))
        cls.brush = scene.add(spec)
        cls.sim = _build(scene)
        cls.built = _body_q(cls.sim)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def test_the_handle_is_on_the_arms_solver_and_the_field_on_vbd(self) -> None:
        self.sim.run(2)
        islands = {i.name: i for i in self.sim.pipeline.program.islands}
        self.assertEqual({"mujoco", "vbd"}, set(islands))
        self.assertTrue(self.sim.pipeline.captured, self.sim.pipeline.capture_error)
        entity = self.brush.entity
        self.assertIn(entity.handle_body, islands["mujoco"].bodies.tolist())
        self.assertLessEqual(set(entity.bodies.tolist()), set(islands["vbd"].bodies.tolist()))
        self.assertEqual(-1, entity.coord_start, "the handle's coordinates are its own entity's")

    def test_press_bows_the_field_marks_the_bench_and_the_bristles_spring_back(self) -> None:
        sim, entity = self.sim, self.brush.entity
        floor = entity.floor_z
        ik, gripper = sim.arm("right"), sim.gripper("right")
        sim.run(30)
        standing = _body_q(sim)
        handle = entity.handle_body
        self.assertLess(abs(float(standing[handle][2] - self.built[handle][2])), 5.0e-4, "the handle holds itself up")
        shape, field = _in_strips(entity, standing), _in_strips(entity, standing, every=True)
        pos, quat = ik.ee_pose(sim.state)
        grasp = np.array([standing[handle][0], standing[handle][1], floor + entity.spec.sole_top + 0.008 + 0.0165])
        _drive(sim, ik, gripper, quat, pos, grasp + np.array([0.0, 0.0, 0.03]), 1.0)
        _drive(sim, ik, gripper, quat, grasp + np.array([0.0, 0.0, 0.03]), grasp, 1.0)
        gripper.command = 1.0
        _drive(sim, ik, gripper, quat, grasp, grasp, 1.0)
        self.assertAlmostEqual(0.022, float(gripper.width(sim.state)), delta=0.003, msg="the jaws are on the handle")
        entity.clear_scrub()
        pressed_at = grasp - np.array([0.0, 0.0, PRESS])
        _drive(sim, ik, gripper, quat, grasp, pressed_at, 0.5)
        _drive(sim, ik, gripper, quat, pressed_at, pressed_at, 0.5)
        pressed = _body_q(sim)
        self.assertTrue(np.isfinite(pressed).all())
        strip = entity.strip_bodies[0]
        self.assertGreater(
            float(standing[strip][2] - pressed[strip][2]), 0.8 * PRESS, "the strips went down with the hand"
        )
        bow = np.linalg.norm((_in_strips(entity, pressed, every=True) - field)[:, :2], axis=1)
        self.assertGreater(float(bow.max()), 0.002, "the field took the press without bowing")
        tips = entity.tip_positions(pressed)
        self.assertGreater(float((tips[:, 2] - floor).min()), 0.0, "a tip went a whole radius into the bench")
        sole = pressed[strip][2]
        self.assertLess(float(tips[:, 2].max()), sole - 0.005, "a tip folded up toward the sole")
        self.assertGreater(int(entity.scrub_grid().cells.sum()), 50, "the pressed tips marked the bench")
        lifted_at = grasp + np.array([0.0, 0.0, 0.025])
        _drive(sim, ik, gripper, quat, pressed_at, lifted_at, 1.0)
        _drive(sim, ik, gripper, quat, lifted_at, lifted_at, 1.0)
        lifted = _body_q(sim)
        back = np.linalg.norm(_in_strips(entity, lifted) - shape, axis=1)
        self.assertLess(float(back.max()), 5.0e-4, "the bristles did not spring back on their strips")
        self.assertGreater(float((entity.tip_positions(lifted)[:, 2] - floor).min()), 0.015, "the brush is up")

    def test_z_a_reset_restores_the_field(self) -> None:
        """Back where they were built on the reset itself, still there a few steps on, and the lattice cleared.

        The strips' welds hang off another entity's body, and the
        forward-kinematics pass during the reset places them from wherever that
        body's pose then reads, so the reset puts the handle's body where its own
        reset put its coordinates. Left where the press test lifted it, the
        strips came back 25 mm off and the welds pulled them back as the scene
        stepped, the bristles after them.
        """
        entity = self.brush.entity
        self.sim.reset()
        at_once = _body_q(self.sim)
        np.testing.assert_allclose(at_once[entity.bodies, :3], self.built[entity.bodies, :3], atol=1.0e-5)
        self.assertEqual(0, int(entity.scrub_grid().cells.sum()))
        self.assertEqual(
            {"n_bristles": 400.0, "scrubbed_cells": 0.0}, entity.published_states(self.sim.pipeline.state_0)
        )
        self.sim.run(10)
        now = _body_q(self.sim)
        drift = np.linalg.norm(now[entity.bodies, :3] - self.built[entity.bodies, :3], axis=1)
        self.assertLess(float(drift.max()), 5.0e-4)

    def test_z_b_an_episode_that_moves_the_handle_carries_the_field_with_it(self) -> None:
        """An episode may put the handle elsewhere; the field goes back onto it there, not where it was built.

        The handle's reset writes the episode's pose, and the brush's reset after
        it must neither undo that (a solver reset to the model's coordinates did)
        nor leave the strips and bristles where the handle was built.
        """
        sim, entity = self.sim, self.brush.entity
        handle = entity.handle_body
        z = float(entity.spec.sole_top + HANDLE_HALF[2])
        sim.episode = {"object_poses": {"handle": (AT[0] + 0.03, AT[1] + 0.02, z, 0.3)}}
        try:
            sim.reset()
            at_once = _body_q(sim)
            sim.run(10)
            later = _body_q(sim)
        finally:
            sim.episode = None
            sim.reset()
        np.testing.assert_allclose(at_once[handle, :3] - self.built[handle, :3], (0.03, 0.02, 0.0), atol=1.0e-4)
        built = _on_handle(self.built, handle, entity.bodies)
        for when, now in (("at once", at_once), ("10 steps on", later)):
            self.assertTrue(np.isfinite(now).all(), when)
            off = np.linalg.norm(_on_handle(now, handle, entity.bodies) - built, axis=1)
            self.assertLess(float(off.max()), 5.0e-4, f"{when}: a body of the field is {off.max() * 1e3:.2f} mm off")


if __name__ == "__main__":
    unittest.main()
