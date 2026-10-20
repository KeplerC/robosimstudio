# SPDX-License-Identifier: Apache-2.0
"""The three particle kinds built and stepped on a GPU: a heap, a pool, a lump, beside the YAM.

What ``tests/test_mpm_pack`` claims of the registries, this measures on built
scenes, each item on the smallest scene that shows it:

* a heap alone on ``void`` under ``GENERIC`` falls, rests on the surface, is
  reset with the MPM's history, and replicates to two worlds;
* two worlds of four particles, one a float ulp under a cell face, take one
  identical step;
* a heap, the robot and a free paddle under ``GRANULAR_60HZ`` compose as two
  islands with the MPM entry in place and no main collision pipeline, capture,
  take a hand into the heap, answer the five measures, and take a live material
  write;
* a cup of honey under ``LIQUID_120HZ`` keeps its particles above its floor;
* a puck of dough under ``DOUGH_60HZ`` holds its height at rest, keeps most of a
  press after the hand lifts, and offers a reconstructed surface;
* each example script loads under the stub studio and steps finite.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np

TIER = "medium"
"""A GPU: five small scenes stepped a few hundred times, and the six examples built once each."""

HEAP = {"name": "rice", "size": (0.08, 0.08, 0.03), "at": (0.0, 0.0, 0.0), "material": "rice"}
"""A 12,288-particle block of the rice preset at the default 2.5 mm spacing."""


def _quiet_build(scene, **kwargs):
    """MuJoCo prints its linesearch notes to stdout during a build; a test log does not need them."""
    with redirect_stdout(io.StringIO()):
        return scene.build(**kwargs)


def _particles(sim) -> np.ndarray:
    sim.pipeline.sync()
    return np.asarray(sim.pipeline.state_0.particle_q.numpy(), dtype=np.float64)


def _drive(sim, ik, gripper, quat, start, goal, seconds: float) -> None:
    """Move the tool centre from ``start`` to ``goal`` over ``seconds`` through the IK, one control step at a time."""
    frames = max(1, int(seconds * float(sim.request.physics.control_hz)))
    for k in range(frames):
        a = (k + 1) / frames
        ik.set_target(start * (1.0 - a) + goal * a, quat)
        ik.solve_into(sim.state.joint_q, sim.control.joint_target_q)
        gripper.apply(sim.control.joint_target_q)
        sim.run(1)


class TestAHeapAlone(unittest.TestCase):
    """The uncoupled MPM island: one solver, no robot, the carry between the pipeline's two states."""

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        scene = rss.Scene(cell="void", name="heap_alone")
        cls.heap = scene.add(rss.heap(**HEAP))
        cls.sim = _quiet_build(scene)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def test_one_island_captured_and_resting_on_the_surface(self) -> None:
        from robosimstudio.engines.newton.mpm import SolverMPM

        program = self.sim.pipeline.program
        self.assertEqual([("mpm", "mpm")], [(i.name, i.backend.solver) for i in program.islands])
        self.assertTrue(self.sim.pipeline.captured, self.sim.pipeline.capture_error)
        self.assertIsInstance(self.heap.entity.solver, SolverMPM)
        self.sim.run(90)
        q = _particles(self.sim)
        self.assertTrue(np.isfinite(q).all())
        self.assertGreater(float(q[:, 2].min()), -0.005)
        self.assertLess(float(q[:, 2].min()), 0.005, "the bottom layer rests within 5 mm of the surface")
        self.assertLess(float(q[:, 2].max()), 0.03, "the block relaxed rather than standing as built")

    def test_the_handle_is_a_point_body(self) -> None:
        handle = self.heap.handle
        self.assertEqual(0, len(handle.triangles))
        self.assertAlmostEqual(0.00125, handle.point_radius)
        self.assertIsNone(handle.surface, "points by default; a surface is asked for")

    def test_reset_restores_the_particles_and_the_history(self) -> None:
        entity = self.heap.entity
        self.sim.run(30)
        self.sim.reset()
        np.testing.assert_allclose(_particles(self.sim), entity.rest["particle_q"], atol=1.0e-6)
        jp = self.sim.pipeline.state_0.mpm.particle_Jp.numpy()
        np.testing.assert_allclose(jp, 1.0, atol=1.0e-5, err_msg="the plastic volume history was not reset")

    def test_two_worlds_isolate(self) -> None:
        import robosimstudio as rss

        scene = rss.Scene(cell="void", name="heap_worlds")
        scene.add(rss.heap(**{**HEAP, "size": (0.06, 0.06, 0.024), "particle_spacing": 0.004}))
        sim = _quiet_build(scene, num_worlds=2)
        try:
            solver = sim.pipeline.program.solver_of_island("mpm")
            self.assertTrue(solver._separate_worlds)
            self.assertEqual(2, solver._environment_count)
            per_world = sim.pipeline.model.particle_count // 2
            sim.run(60)
            q = _particles(sim)
            self.assertTrue(np.isfinite(q).all())
            for world in range(2):
                block = q[world * per_world : (world + 1) * per_world]
                self.assertGreater(float(block[:, 2].min()), -0.005, f"world {world} fell through the ground")
                self.assertLess(float(block[:, 2].min()), 0.005, f"world {world} is not resting on the ground")
        finally:
            sim.close()


class TestTwoWorldsBinAlike(unittest.TestCase):
    """Two identical worlds take one identical step, with a particle a hair under a cell face.

    The smallest case of the fault ``robosimstudio.engines.newton.mpm`` works around: world 1 packs
    its cells 34 along x, so a particle at local index -2.0000007 floors to cell -3 when the grid is
    built and rounds to -2 when it is looked up (``-2.0000007 + 34`` is 32.0 in float32); Warp 1.18's
    fallback search then leaves cell -3 out and bins it into cell (-2, 5, 2), 0.42 of a cell away.
    The moving fourth particle makes the misbinning show in one step: with Warp's own lookup world 1
    ended up to 28.3 mm from world 0, 5 runs in 5; with the workaround within 1e-6 m, 5 in 5.
    """

    def test_world_one_steps_as_world_zero_does(self) -> None:
        import newton
        from newton.solvers import SolverImplicitMPM

        from robosimstudio.engines.newton import MPM
        from robosimstudio.engines.newton.mpm import SolverMPM

        voxel = float(dict(MPM.defaults)["voxel_size"])
        self.assertEqual(0.004, voxel, "the positions below are in 4 mm cells")
        points = [
            (-0.040, 0.0, 0.0),  # cell x -10 and ...
            (0.040, 0.0, 0.0),  # ... x +10: world 0's tile spans 21 cells and 3 of guard, so world 1 packs at +34
            (-0.010000002570450306, 0.01632, 0.00888),  # local index (-2.0000007, 4.58, 2.72): cell (-3, 4, 2)
            (-0.008, 0.020, 0.008),  # the middle of cell (-2, 5, 2), the one the faulty search picks
        ]
        # The fourth moves, so a node's velocity is a mass-weighted mean worth getting right: under
        # gravity alone every node falls alike, whatever mass the binning gave it.
        velocities = [(0.0, 0.0, 0.0)] * 3 + [(1.0, 0.0, 0.0)]
        template = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(template)
        template.add_particles(
            pos=points,
            vel=velocities,
            mass=[1500.0 * voxel**3] * len(points),
            radius=[0.5 * voxel] * len(points),
            flags=[int(newton.ParticleFlags.ACTIVE)] * len(points),
        )
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        builder.replicate(template, 2)
        model = builder.finalize()
        solver = SolverMPM(model, **dict(MPM.defaults))
        self.assertTrue(solver._separate_worlds)
        state_in, state_out = model.state(), model.state()
        solver.step(state_in, state_out, model.control(), None, 1.0 / float(solver.step_hz))
        q = state_out.particle_q.numpy()
        self.assertTrue(
            np.isfinite(q).all(),
            f"non-finite particles per world: {(~np.isfinite(q)).any(axis=1).reshape(2, -1).sum(axis=1)}",
        )
        np.testing.assert_allclose(
            q[len(points) :], q[: len(points)], atol=1.0e-6, err_msg="world 1 stepped unlike world 0"
        )


class TestTheHeapBesideTheRobot(unittest.TestCase):
    """``GRANULAR_60HZ``: the robot's island holds a free paddle, the MPM's holds the rice, and the two couple."""

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        scene = rss.Scene(cell="bench_075", name="heap_robot", physics=rss.GRANULAR_60HZ)
        scene.add(rss.robot("yam_bimanual"))
        cls.heap = scene.add(rss.heap(**{**HEAP, "at": (0.0, -0.31, 0.0)}))
        cls.paddle = scene.add(rss.box(name="paddle", half=(0.03, 0.05, 0.004), at=(-0.12, -0.31, 0.004)))
        cls.sim = _quiet_build(scene, control_mode="joint_position")
        cls.program = cls.sim.pipeline.program

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def test_two_islands_the_mpm_in_place_and_no_main_pipeline(self) -> None:
        names = [(i.name, i.backend.solver) for i in self.program.islands]
        self.assertEqual([("mujoco", "mujoco"), ("mpm", "mpm")], names)
        configs = self.program.solver._entry_configs_by_name
        self.assertTrue(configs["mpm"].in_place)
        self.assertFalse(configs["mujoco"].in_place)
        self.assertIsNone(self.program.collision)
        (channel,) = self.program.channels
        self.assertEqual(("mujoco", "mpm"), (channel.source, channel.destination))
        self.assertIn("paddle", channel.entities, "the free tool is mirrored into the MPM")
        self.assertTrue(self.sim.pipeline.captured, self.sim.pipeline.capture_error)
        self.assertEqual(1000.0, self.sim.request.physics.coupling.get("mass_scale"))

    def test_the_paddle_rests_and_the_heap_settles_finite(self) -> None:
        self.sim.run(60)
        q = _particles(self.sim)[self.heap.particles]
        self.assertTrue(np.isfinite(q).all())
        self.assertLess(abs(float(q[:, 2].min()) - 0.75), 0.005)
        self.assertLess(abs(float(self.paddle.position[2]) - 0.754), 0.003, "an 8 mm box lies flat on the bench")

    def test_the_measures_answer_from_a_snapshot(self) -> None:
        from robosimstudio_materials.mpm.measures import (
            _fill_fraction,
            _heap_height,
            _mass_in_region,
            _spill_fraction,
            _spread,
        )

        entity = self.heap.entity
        entity.update_probe(self.sim.pipeline.state_0)
        snapshot = SimpleNamespace(particles={"rice": entity.probe}, entities={"rice": entity})
        # The bench frame is at (0.6, 0, 0.75); the heap stands on (0, -0.31) of it.
        inside = {"xy_ranges": (0.5, -0.41, 0.7, -0.21), "z_range": (0.75, 0.80)}
        beside = {"xy_ranges": (0.5, -0.19, 0.7, 0.01), "z_range": (0.75, 0.80)}
        self.assertGreater(_fill_fraction(entity.probe.points("rice"), inside)["fraction"], 0.99)
        self.assertEqual(0.0, _fill_fraction(entity.probe.points("rice"), beside)["fraction"])
        mass = _mass_in_region(snapshot, "rice", inside)
        self.assertAlmostEqual(entity.particle_mass * len(entity.particles) * mass["fraction"], mass["mass"])
        self.assertGreater(mass["mass"], 0.25, "12,288 grains of rice at 2.5 mm weigh 288 g")
        self.assertEqual(0.0, _spill_fraction(snapshot, "rice", inside)["fraction"])
        spilled = _spill_fraction(snapshot, "rice", beside)["fraction"]
        self.assertTrue(0.2 < spilled < 0.9, f"the bottom two spacings of the heap lie outside that box: {spilled:.2f}")
        height = _heap_height(snapshot, "rice")["height"]
        self.assertTrue(0.015 < height < 0.03, f"a 30 mm block slumps to a heap, read {height * 1e3:.1f} mm")
        self.assertAlmostEqual(1.0, _spread(snapshot, "rice")["spread"], delta=0.15)

    def test_a_hand_driven_into_the_heap_displaces_it(self) -> None:
        sim = self.sim
        ik, gripper = sim.arm("right"), sim.gripper("right")
        q = _particles(sim)[self.heap.particles]
        centre, top = q[:, :2].mean(axis=0), float(q[:, 2].max())
        pos, quat = ik.ee_pose(sim.state)
        gripper.command = 1.0
        hover = np.array([centre[0], centre[1], top + 0.05])
        _drive(sim, ik, gripper, quat, pos, hover, 1.5)
        before = _particles(sim)[self.heap.particles]
        _drive(sim, ik, gripper, quat, hover, np.array([centre[0], centre[1], top - 0.015]), 1.5)
        after = _particles(sim)[self.heap.particles]
        self.assertTrue(np.isfinite(after).all())
        moved = np.linalg.norm(after - before, axis=1)
        spacing = float(self.heap.spec.particle_spacing)
        self.assertGreater(int((moved > spacing).sum()), 20)
        self.assertTrue(self.sim.pipeline.captured, "stepping did not drop the graph")

    def test_a_material_write_reaches_the_solver_and_a_reset_restores_the_rest(self) -> None:
        entity = self.heap.entity
        (tunable,) = self.heap.tunables()
        self.assertEqual("material", tunable.name)
        self.assertAlmostEqual(0.68, tunable.spec.friction, places=5)
        self.assertAlmostEqual(1500.0, tunable.spec.density, places=2)
        self.assertTrue(tunable.write("friction", 0.2))
        self.assertFalse(tunable.write("density", 1000.0), "density became particle mass at build")
        first = int(entity.particles[0])
        self.assertAlmostEqual(0.2, float(self.sim.pipeline.model.mpm.friction.numpy()[first]), places=5)
        self.assertAlmostEqual(0.2, float(entity.solver.model.mpm.friction.numpy()[first]), places=5)
        self.sim.run(5)
        self.assertTrue(self.sim.pipeline.captured, "the write dropped the graph and the next step recaptured")
        self.sim.reset()
        np.testing.assert_allclose(_particles(self.sim), entity.rest["particle_q"], atol=1.0e-6)
        jp = self.sim.pipeline.state_0.mpm.particle_Jp.numpy()[entity.particles]
        np.testing.assert_allclose(jp, 1.0, atol=1.0e-5)


class TestTheHoneyCup(unittest.TestCase):
    """``LIQUID_120HZ`` and a free cup: the margin and the heavy proxies keep the honey where it was poured."""

    def test_the_cup_holds_its_honey_at_rest(self) -> None:
        import robosimstudio as rss
        from robosimstudio_materials.liquid import cup

        wall = 0.008
        scene = rss.Scene(cell="bench_075", name="honey_cup", physics=rss.LIQUID_120HZ)
        scene.add(rss.robot("yam_bimanual"))
        scene.add(cup(name="cup", inner=(0.05, 0.05, 0.07), wall=wall, at=(0.0, -0.31, 0.0)))
        honey = scene.add(
            rss.liquid(
                name="honey",
                size=(0.05, 0.05, 0.05),
                at=(0.0, -0.31, wall + 0.002),
                material="honey",
                containers=("cup",),
            )
        )
        sim = _quiet_build(scene, control_mode="joint_position")
        try:
            model = sim.pipeline.model
            labels = list(model.shape_label)
            margins = [float(model.shape_margin.numpy()[i]) for i, l in enumerate(labels) if l.startswith("cup")]
            self.assertEqual(6, len(margins), "the mesh and the shell's five boxes")
            self.assertTrue(all(abs(m - 0.002) < 1e-9 for m in margins), margins)
            self.assertFalse(sim.request.physics.backend_for("liquid").get("project_outside"))
            sim.run(120)
            q = _particles(sim)[honey.particles]
            self.assertTrue(np.isfinite(q).all())
            kept = float((q[:, 2] > 0.75 + wall - 0.001).mean())
            self.assertGreaterEqual(kept, 0.97, f"{kept:.3f} of the honey is above the cup's floor after 2 s")
        finally:
            sim.close()


class TestTheDoughPress(unittest.TestCase):
    """``DOUGH_60HZ``: the lump holds its height alone, keeps most of a press, and reconstructs a surface."""

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        scene = rss.Scene(cell="bench_075", name="dough_press", physics=rss.DOUGH_60HZ)
        scene.add(rss.robot("yam_bimanual"))
        cls.lump = scene.add(
            rss.dough(name="dough", shape="cylinder", size=(0.06, 0.06, 0.04), at=(0.0, -0.31, 0.0), render="surface")
        )
        cls.sim = _quiet_build(scene, control_mode="joint_position")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def _local_height(self, centre: np.ndarray, radius: float = 0.012) -> float:
        """The 95th-percentile height of the particles under the blade, over the bench [m]."""
        q = _particles(self.sim)[self.lump.particles]
        near = q[np.hypot(q[:, 0] - centre[0], q[:, 1] - centre[1]) < radius]
        return float(np.quantile(near[:, 2], 0.95)) - self.lump.entity.floor_z

    def test_the_surface_is_offered_and_meshes_the_lump(self) -> None:
        """A closed surface round the lump, where it is: the particles sit at negative y on this bench."""
        from robosimstudio.core.render.particles import surface_mesh

        handle = self.lump.handle
        self.assertIsNotNone(handle.surface)
        q = np.asarray(self.lump.particle_q, dtype=np.float64)
        self.assertLess(float(q[:, 1].max()), 0.0)
        vertices, triangles, normals = handle.surface(q)
        self.assertGreater(len(triangles), 100)
        self.assertEqual((len(triangles), 3), tuple(triangles.shape))
        self.assertEqual(len(vertices), len(normals))
        vertices, triangles, _ = surface_mesh(handle.surface, q)
        edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
        _, shared = np.unique(edges, axis=0, return_counts=True)
        self.assertTrue((shared == 2).all(), f"{int((shared != 2).sum())} of {len(shared)} edges border one triangle")
        np.testing.assert_array_less(q.min(axis=0) - 0.01, vertices.min(axis=0))
        np.testing.assert_array_less(vertices.max(axis=0), q.max(axis=0) + 0.01)

    def test_the_puck_holds_its_height_at_rest(self) -> None:
        from robosimstudio_materials.mpm.measures import heap_height

        entity = self.lump.entity
        rest = entity.rest["particle_q"][entity.particles]
        settled = heap_height(rest, entity.floor_z)
        self.sim.reset()  # the press test may have run first; a reset is the rest state, the MPM's history with it
        self.sim.run(120)
        held = heap_height(_particles(self.sim)[entity.particles], entity.floor_z) / settled
        self.assertGreater(held, 0.95, f"the puck kept {held:.3f} of its height over 2 s alone (150 iterations)")

    def test_a_press_is_mostly_kept_after_the_hand_lifts(self) -> None:
        from robosimstudio_materials.mpm.measures import spread

        sim, entity = self.sim, self.lump.entity
        ik, gripper = sim.arm("right"), sim.gripper("right")
        q = _particles(sim)[entity.particles]
        centre, top = q[:, :2].mean(axis=0), float(np.quantile(q[:, 2], 0.95))
        pos, quat = ik.ee_pose(sim.state)
        gripper.command = 1.0
        hover = np.array([centre[0], centre[1], top + 0.05])
        _drive(sim, ik, gripper, quat, pos, hover, 1.5)
        before = self._local_height(centre)
        press = np.array([centre[0], centre[1], top - 0.015])
        _drive(sim, ik, gripper, quat, hover, press, 1.5)
        _drive(sim, ik, gripper, quat, press, press, 0.5)
        pressed = self._local_height(centre)
        _drive(sim, ik, gripper, quat, press, hover, 1.5)
        sim.run(60)
        after = self._local_height(centre)
        self.assertLess(pressed, before - 0.005, f"the blade took the dough down: {before:.4f} -> {pressed:.4f} m")
        kept = (before - after) / (before - pressed)
        self.assertGreaterEqual(kept, 0.6, f"kept {kept:.2f} of the compression after the hand lifted")
        rest = entity.rest["particle_q"][entity.particles]
        self.assertGreater(spread(_particles(sim)[entity.particles], rest), 1.0, "a press spreads the lump")
        self.assertTrue(sim.pipeline.captured)


class TestTheExamplesLoad(unittest.TestCase):
    """Each numbered script under the stub studio: it builds, opens one page and steps finite."""

    SCRIPTS = (
        "03_materials/11_granular.py",
        "06_tasks/03_granular_scored.py",
        "03_materials/12_liquid.py",
        "tests/scenes/01_wipe_spill.py",
        "tests/scenes/00_roll_dough.py",
        "03_materials/13_dough.py",
    )

    def test_each_example_builds_and_steps(self) -> None:
        from tests.fixtures.examples import _run

        for rel in self.SCRIPTS:
            with self.subTest(example=rel):
                with redirect_stdout(io.StringIO()):
                    opened, kwargs = _run(rel)
                sim = getattr(opened, "sim", opened)
                try:
                    self.assertTrue(sim.pipeline.captured, sim.pipeline.capture_error)
                    self.assertGreater(sim.model.particle_count, 1000)
                    with redirect_stdout(io.StringIO()):
                        sim.run(20)
                    q = _particles(sim)
                    self.assertTrue(np.isfinite(q).all())
                    if "scored" not in rel:
                        self.assertTrue(callable(kwargs.get("page")), "the page adds its readout")
                finally:
                    sim.close()


if __name__ == "__main__":
    unittest.main()
