# SPDX-License-Identifier: Apache-2.0
"""Real scenes on a CPU, picked rather than forced: what ``robosimstudio[cpu]`` promises, built and stepped.

Every scene here is built with ``device`` left at ``auto`` on a machine whose
GPUs are hidden (``scripts/testrun.py --tier cpu`` empties
``CUDA_VISIBLE_DEVICES``), so what passes is what a laptop without a GPU gets
from the same script: MuJoCo's C library for the arm and the cube, SuperDex for
the sponge, the towel and the cable, Newton's tiled tracer for a picture. Run by
hand on a GPU box without the runner, the module sets ``ROBOSIMSTUDIO_DEVICE=cpu``
for itself instead, which is the same rule one step up.

The numbers each test holds are the GPU's own where there is one to compare --
the cube rests at the height it reaches under MuJoCo-Warp, the contact probe
reads its weight -- because a CPU path that ran and gave different physics would
pass every test that only asked whether it ran.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

TIER = "cpu"
"""No GPU, real scenes; needs the ``[cpu]`` extra for the deformables. 48 s with a cold Warp cache, 23 s warm."""

ROOT = Path(__file__).resolve().parents[1]
HAS_SUPERDEX = importlib.util.find_spec("superdex") is not None
NO_SUPERDEX = "needs the [cpu] extra (SuperDex)"

_hidden = unittest.mock.patch.dict(os.environ, {"ROBOSIMSTUDIO_DEVICE": "cpu"})
"""What hides the GPU when the runner did not: the rule's own variable, for this module only."""


_device_before: list = []
"""Warp's default device before this module ran: the variable moves it, and a run of several modules must get it back."""


def setUpModule() -> None:
    """Hide the GPU by the rule's own variable when the runner did not hide it by the driver's."""
    import warp as wp

    _device_before.append(wp.get_device())
    if wp.is_cuda_available():
        _hidden.start()


def tearDownModule() -> None:
    import warp as wp

    with contextlib.suppress(RuntimeError):  # never started
        _hidden.stop()
    if _device_before:
        wp.set_device(_device_before.pop())


def _arm_and_cube(physics=None, *, at=(-0.143, -0.115, 0.1)):
    import robosimstudio as rss

    scene = rss.Scene(cell="bench_075", **({"physics": physics} if physics is not None else {}))
    cube = scene.add(rss.box(name="cube", half=(0.025, 0.025, 0.025), at=at))
    scene.add(rss.robot("panda"))
    return scene, cube


class TestTheArmAndACubeOnTheDefaultArrangement(unittest.TestCase):
    """The hello-world scene with nothing said about physics or device: GENERIC, on a CPU, becomes CPU."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scene, cls.cube = _arm_and_cube(at=(0.2, 0.1, 0.05))
        cls.sim = cls.scene.build()
        cls.sim.run(150).sync()
        cls.solver = cls.sim.pipeline.program.mujoco_solver

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def test_it_picked_the_cpu_and_said_what_it_changed(self) -> None:
        import robosimstudio as rss

        plat = self.sim.platform
        self.assertTrue(plat.is_cpu, plat)
        self.assertIs(self.sim.request.physics, rss.CPU)
        self.assertTrue(any("GENERIC → CPU" in note for note in plat.notes), plat.notes)
        self.assertEqual(plat.renderer, "tiled")

    def test_every_mujoco_island_is_the_c_library_and_nothing_is_captured(self) -> None:
        program = self.sim.pipeline.program
        self.assertEqual([island.backend.solver for island in program.islands], ["mujoco"])
        self.assertTrue(program.islands[0].backend.get("use_mujoco_cpu"))
        self.assertTrue(self.solver.use_mujoco_cpu)
        self.assertFalse(self.sim.pipeline.captured)
        self.assertIn("mujoco (C)", program.summary())

    def test_the_cube_rests_where_it_does_on_a_gpu(self) -> None:
        surface = float(self.sim.scene.handles.surface_z)
        self.assertAlmostEqual(float(self.cube.position[2]), surface + 0.025, delta=0.002)
        self.assertTrue(np.isfinite(self.sim.pipeline.state_0.body_q.numpy()).all())

    def test_the_contact_probe_reads_the_cubes_weight(self) -> None:
        """From MuJoCo's C data, in MuJoCo-Warp's sign: the table pushes the cube up with m*g."""
        entity = self.sim.scene.objects["cube"]
        shape_body = self.sim.model.shape_body.numpy()
        cube = [int(i) for i in np.flatnonzero(shape_body == int(entity.body))]
        table = [int(i) for i in np.flatnonzero(shape_body < 0)]
        probe = self.sim.contacts
        self.assertIsNotNone(probe)
        probe.update()
        mass = float(self.sim.model.body_mass.numpy()[int(entity.body)])
        weight = mass * 9.81 * self.sim.pipeline.sim_dt
        impulse = probe.pairwise_impulse(cube, table)
        self.assertAlmostEqual(float(impulse[2]), weight, delta=0.1 * weight)
        self.assertTrue(probe.touching(cube))

    def test_live_tuning_reaches_the_model_mj_step_reads(self) -> None:
        self.sim.pipeline.program.apply("backend:mujoco:iterations", 7)
        self.assertEqual(self.solver.mj_model.opt.iterations, 7)

    def camera(self):
        from robosimstudio.specs import CameraSpec

        z = float(self.sim.scene.handles.surface_z)
        return CameraSpec(name="cpu_still", width=320, height=180, fov_y=45.0, eye=(-0.4, -0.9, z + 0.75),
                          look_at=(0.6, 0.0, z + 0.05), up=(0.0, 0.0, 1.0), near=0.01, far=8.0)  # fmt: skip

    def test_a_tiled_still_comes_from_the_cpu(self) -> None:
        rgb = self.sim.render(self.camera())["rgb"]
        self.assertEqual((rgb.shape, rgb.dtype), ((180, 320, 3), np.uint8))
        self.assertGreater(len(np.unique(rgb.reshape(-1, 3), axis=0)), 50, "a single-colour frame drew nothing")

    def test_a_video_is_written_from_the_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            self.sim.record(out, 6, camera=self.camera(), fps=10)
            (video,) = Path(out).glob("*.mp4")
            self.assertGreater(video.stat().st_size, 1000)

    def test_the_arm_solves_a_pose(self) -> None:
        ik = self.sim.arm()
        seed = self.sim.state.joint_q.numpy().astype(np.float64)
        position, quat = ik.ee_pose(self.sim.state)
        target = np.asarray(position, dtype=np.float64) + np.array([0.0, 0.0, -0.05])
        q = ik.solve_q_full(seed, target, quat)
        self.assertLess(ik.pose_errors(q, target, quat)[0], 0.002)


class TestLiveTuningOnACpu(unittest.TestCase):
    """A friction or a gain changed while the scene runs reaches the model ``mj_step`` reads.

    Newton's notify regenerates both into MuJoCo-Warp's model, which a CPU never
    steps, and copies neither into the C one; measured before the copy, the cube's
    friction stayed 1.0 and the arm's gain 650.
    """

    def test_friction_and_gains_reach_the_c_model(self) -> None:
        import newton

        import robosimstudio as rss

        scene, _ = _arm_and_cube(rss.MUJOCO_500HZ, at=(0.2, 0.1, 0.05))
        sim = scene.build()
        try:
            solver = sim.pipeline.program.mujoco_solver
            shapes = [int(s) for s in sim.scene.objects["cube"].shapes]
            geom_to_shape = solver.mjc_geom_to_newton_shape.numpy()[0]
            geoms = [g for g, s in enumerate(geom_to_shape) if int(s) in shapes]
            mu = sim.model.shape_material_mu.numpy()
            mu[shapes] = 0.05
            sim.model.shape_material_mu.assign(mu)
            sim.pipeline.program.notify(int(newton.ModelFlags.SHAPE_PROPERTIES))
            np.testing.assert_allclose(solver.mj_model.geom_friction[geoms, 0], 0.05)

            gain = float(solver.mj_model.actuator_gainprm[0, 0])
            ke = sim.model.joint_target_ke.numpy()
            sim.model.joint_target_ke.assign(ke * 2.0)
            sim.pipeline.program.notify(int(newton.ModelFlags.JOINT_DOF_PROPERTIES))
            self.assertAlmostEqual(float(solver.mj_model.actuator_gainprm[0, 0]), 2.0 * gain, places=3)
        finally:
            sim.close()


class TestParkingOnACpu(unittest.TestCase):
    """A parked body is held by the C model's gravity compensation, the row ``mj_step`` reads."""

    def test_without_a_gravity_compensated_body_parking_refuses(self) -> None:
        """MuJoCo's C library skips the pass when it compiled none; a parked cube would fall reading 1.0.

        Refused where a body is parked; reading a row stays fine.
        """
        import robosimstudio as rss
        from robosimstudio.scene.parking import GravcompParking

        scene = rss.Scene(cell="bench_075")
        scene.add(rss.box(name="cube", half=(0.025, 0.025, 0.025), at=(0.2, 0.1, 0.05)))
        sim = scene.build()
        try:
            parking = GravcompParking(sim.pipeline.program.mujoco_solver, sim.model, sim.scene.worlds)
            body = int(sim.scene.objects["cube"].body)
            self.assertEqual(parking.row(0, body), 0.0)
            parking.apply()  # nothing parked: nothing to refuse
            parking.set(0, [body])
            with self.assertRaisesRegex(RuntimeError, "no gravity-compensated body"):
                parking.apply()
        finally:
            sim.close()

    def test_a_parked_cube_hangs_where_it_was_put(self) -> None:
        import newton

        import robosimstudio as rss
        from robosimstudio.scene.parking import GravcompParking

        scene, cube = _arm_and_cube(rss.MUJOCO_500HZ, at=(0.2, 0.1, 0.05))
        sim = scene.build()
        try:
            sim.run(50).sync()
            entity = sim.scene.objects["cube"]
            solver = sim.pipeline.program.mujoco_solver
            parking = GravcompParking(solver, sim.model, sim.scene.worlds)
            q = sim.pipeline.state_0.joint_q.numpy().copy()
            q[int(entity.coord_start) + 2] += 0.10
            for state in (sim.pipeline.state_0, sim.pipeline.state_1):
                state.joint_q.assign(q)
                newton.eval_fk(sim.model, state.joint_q, state.joint_qd, state)
            parking.set(0, [int(entity.body)])
            parking.apply()
            self.assertEqual(parking.row(0, int(entity.body)), 1.0)
            before = float(cube.position[2])
            sim.run(100).sync()
            self.assertLess(abs(float(cube.position[2]) - before), 0.001)
        finally:
            sim.close()


class TestTheStudioOnACpu(unittest.TestCase):
    """The page with no browser on it: the hello-world pick, through the arm panels, held by friction.

    The medium tier's routine test, on a CPU: the gizmo leads, IK follows, the
    fingers close and the cube comes up because they hold it -- the claim the
    page makes on a GPU, made by MuJoCo's C library here.
    """

    def test_the_pick_lifts_the_cube_by_its_fingers(self) -> None:
        from tests.fixtures.examples import _run
        from tests.fixtures.grasp import open_session, tick
        from tests.test_studio_routine import _pick

        sim, kwargs = _run("00_getting_started/00_hello_world.py")
        session = open_session(sim, 8977, kwargs.get("page"))
        try:
            self.assertTrue(sim.platform.is_cpu)
            session.add_routines({"Pick": _pick})
            panel = session.panels[-1]
            tick(session, 50)
            lying = np.asarray(sim["cube"].position, dtype=np.float64)
            panel.requested = "Pick"
            tick(session, 1)
            steps = 0
            while panel.gen is not None and steps < 3000:
                tick(session, 1)
                steps += 1
            self.assertIn("done", str(panel._status.value))
            self.assertGreater(float(sim["cube"].position[2] - lying[2]), 0.13, "the cube stayed on the bench")
        finally:
            session.teardown()


@unittest.skipUnless(HAS_SUPERDEX, NO_SUPERDEX)
class TestTheDeformableExamplesUnchanged(unittest.TestCase):
    """The shipped sponge, towel and cable, as a reader runs them: VBD is replaced, SuperDex takes them."""

    def run_example(self, rel: str, kind: str) -> tuple[np.ndarray, float]:
        """Twenty control steps of one example; its particles after them, and the bench height."""
        from tests.fixtures.examples import _run

        sim, _ = _run(rel)
        try:
            program = sim.pipeline.program
            engines = {island.backend.solver for island in program.islands}
            self.assertEqual(engines, {"mujoco", "superdex"}, program.summary())
            self.assertTrue(sim.platform.is_cpu)
            owner = next(i for i in program.islands if i.backend.solver == "superdex")
            self.assertGreater(owner.particles.size, 0, f"the {kind} is not on SuperDex")
            sim.run(20).sync()
            now = sim.pipeline.state_0.particle_q.numpy().copy()
            self.assertTrue(np.isfinite(now).all(), f"the {kind} diverged on a CPU")
            self.assertTrue(np.isfinite(sim.pipeline.state_0.body_q.numpy()).all())
            return now, float(sim.scene.handles.surface_z)
        finally:
            sim.close()

    def test_the_sponge(self) -> None:
        particles, surface = self.run_example("02_soft/00_sponge.py", "sponge")
        self.assertAlmostEqual(float(particles[:, 2].min()), surface, delta=0.003, msg="the sponge is off the bench")

    def test_the_towel(self) -> None:
        """Built 20 mm up, it has come down by the time the build hands it over, and lies on the bench."""
        particles, surface = self.run_example("01_cloth/00_towel.py", "towel")
        self.assertAlmostEqual(float(particles[:, 2].min()), surface, delta=0.003, msg="the towel is off the bench")
        self.assertLess(float(particles[:, 2].max()), surface + 0.015, "the towel never came down")

    def test_the_cable(self) -> None:
        self.run_example("03_rope/00_cable.py", "cable")


@unittest.skipUnless(HAS_SUPERDEX, NO_SUPERDEX)
class TestTheCableComesUpInTheHand(unittest.TestCase):
    """``03_rope/00_cable``'s grasp on a CPU, as its page makes it: down, **Close**, up, and the cable comes too.

    The GPU's rope test (:mod:`tests.test_example_grasps_rope`) on the capsule
    chain, made here on the SuperDex centerline the CPU puts the rope on --
    friction alone, no latch -- with its nodes tracked where that test tracks
    capsules. It holds because ``rss.CPU``'s rope is one-way; two-way, this
    gesture lifted the cable in under half its runs (see :data:`rss.CPU`).
    """

    def test_the_hand_lifts_it_and_it_stays_in_the_fingers(self) -> None:
        from tests.fixtures.examples import _run
        from tests.fixtures.grasp import drag_to_grasp, open_session, tick

        sim, kwargs = _run("03_rope/00_cable.py")
        session = open_session(sim, 8979, kwargs.get("page"))
        try:
            self.assertEqual(sim.request.physics.backend_for("rod").get("feedback"), "off")
            nodes = np.asarray(sim["cable"].particles, dtype=int)

            def points() -> np.ndarray:
                return sim.state.particle_q.numpy()[nodes].astype(np.float64)

            tick(session, 30)  # the cable lands on the bench
            lying = points()
            home = session.arms["right"].ik.home_pose[0]
            under = int(np.argmin(np.linalg.norm(lying[:, :2] - home[:2], axis=1)))
            grasp = drag_to_grasp(
                session,
                "right",
                lambda: points()[under],
                at=lying[under] + np.array([0.0, 0.0, 0.010]),
                hover=0.03,
                lift=0.12,
                speed=0.2,
            )
            middle = slice(len(nodes) // 2 - 1, len(nodes) // 2 + 1)
            rise = float(points()[middle].mean(axis=0)[2] - lying[middle].mean(axis=0)[2])
            self.assertTrue(grasp.finite)
            self.assertGreater(rise, 0.10, f"the cable's middle rose {rise * 1e3:.1f} mm of a 120 mm lift")
            self.assertLess(grasp.drift, 0.005, "the cable slid in the fingers")
        finally:
            session.teardown()


class TestWhatACpuRefuses(unittest.TestCase):
    """Said by name, before anything is built, and pointing at what works instead."""

    def test_several_worlds(self) -> None:
        from robosimstudio.scene.compiler import CannotReplicate

        scene, _ = _arm_and_cube()
        with self.assertRaisesRegex(CannotReplicate, "build num_worlds=1 on a CPU"):
            scene.build(num_worlds=2)

    def test_the_path_tracer(self) -> None:
        scene, _ = _arm_and_cube()
        sim = scene.build()
        try:
            previous = os.environ.get("ROBOSIMSTUDIO_RENDERER")
            os.environ["ROBOSIMSTUDIO_RENDERER"] = "rtx"
            try:
                with self.assertRaisesRegex(ValueError, "runs on cuda only"):
                    sim.render()
            finally:
                if previous is None:
                    os.environ.pop("ROBOSIMSTUDIO_RENDERER", None)
                else:
                    os.environ["ROBOSIMSTUDIO_RENDERER"] = previous
        finally:
            sim.close()


class TestWithoutSuperDex(unittest.TestCase):
    """A base install on a CPU: the arm and rigid bodies run, and a deformable names the extra."""

    SCRIPT = """
        import importlib.abc, sys
        class _Refuse(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "superdex" or name.startswith("superdex."):
                    raise ModuleNotFoundError(f"No module named {name!r}", name=name)
                return None
        sys.meta_path.insert(0, _Refuse())

        import robosimstudio as rss
        scene = rss.Scene(cell="bench_075")
        cube = scene.add(rss.box(name="cube", half=(0.025, 0.025, 0.025), at=(0.2, 0.1, 0.05)))
        scene.add(rss.robot("panda"))
        sim = scene.build()
        sim.run(20).sync()
        print("ARM_OK", sim.platform.is_cpu)
        scene = rss.Scene(cell="bench_075")
        scene.add(rss.robot("panda"))
        scene.add(rss.sheet(name="towel", size=(0.2, 0.2), cells=(8, 8), at=(-0.1, -0.1, 0.02)))
        try:
            scene.build()
        except ImportError as exc:
            print("SHEET_REFUSED", "robosimstudio[cpu]" in str(exc))
    """

    def test_the_arm_runs_and_the_sheet_names_the_extra(self) -> None:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        done = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        out = done.stdout + done.stderr
        self.assertIn("ARM_OK True", out, out[-3000:])
        self.assertIn("SuperDex is not installed", out)
        self.assertIn("SHEET_REFUSED True", out, out[-3000:])


if __name__ == "__main__":
    unittest.main()
