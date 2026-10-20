# SPDX-License-Identifier: Apache-2.0
"""A static ``rss.box`` stays where it was put, under ``GENERIC`` and on MuJoCo, beside the YAM or not.

``static=True`` built a bare ``add_link``: a body with no joint, flagged DYNAMIC.
``GENERIC`` puts a rigid object on VBD, which integrated it like any free body.
Beside the YAM, a static 10 cm cube put 0.2 m over the bench fell 148 mm to it in
a second. A static 160 x 160 x 40 mm block resting on the bench sank 3.1 mm, and
under the right hand, shut and driven to 20 mm below the block's top, it slid
(+10.0, -5.9, -4.4) mm, as a free block did (+10.0, -5.8, -4.4). The fingertips
stopped 8.6 mm over the top, where they stop 16.7 mm over an ``rss.slab`` of the
same size. A towel on VBD laid over a static cube with no robot did not build:
its model had no joint, and VBD's first reset refused the empty
``joint_dof_dim``. With that alone mended, the cube sank 14 mm under the towel in
two seconds, the towel 12 mm under where the cube's top had been. And on MuJoCo
the cube did not convert at all: "cannot convert bodies that are outside
articulations and have no standalone joint to world". Newton 1.6 did the same
on both solvers (``s2_newton.py``), so none of it is the 1.7 bump.

A static object is welded now (``EntityRigid._add_welded_link``), so each of
these scenes has a joint and builds. In all of them the static body moved
0.00 mm, the towel lay at 0.852 m on the cube's 0.850 m top, and the hand stopped
16.7 mm over the block's top, as over the slab. Probes: ``s1_repro.py``,
``s3_press.py``.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import numpy as np

TIER = "medium"
"""A GPU: three scenes built, two beside the YAM, each stepped a second or two; then four task builds."""

OVER_BRUSH = (0.0, 2.149, 1.879, -1.3, 0.0, 0.0)
"""The right arm's joints [rad]: tool centre 100 mm over the bench at (0.025, -0.31), pointing down."""

UNDER_HAND = (0.025, -0.31)
"""Where the block rests, in the bench's frame [m]: under the right hand at :data:`OVER_BRUSH`."""

BLOCK_HALF = (0.08, 0.08, 0.02)
"""Wide enough that the shut fingertips land on its top, not beside it [m]."""

SURFACE = (0.6, 0.0, 0.75)
"""``bench_075``'s work-surface origin in world [m]."""


def _yam_scene(physics, name: str):
    import robosimstudio as rss
    from robosimstudio.robots.registry import get_embodiment

    yam = get_embodiment("yam_bimanual")
    left, right = yam.arms
    yam = rss.replace(yam, arms=(left, rss.replace(right, home_q=OVER_BRUSH)))
    scene = rss.Scene(cell="bench_075", name=name, physics=physics)
    scene.add(rss.robot(yam))
    return scene


def _build(scene, **kwargs):
    """MuJoCo prints its linesearch notes to stdout during a build; a test log does not need them."""
    with redirect_stdout(io.StringIO()):
        return scene.build(**kwargs)


def _at(ref) -> np.ndarray:
    return np.asarray(ref.position, dtype=np.float64)


MOVES = (0.0, 0.08)
"""Where the two episodes of :data:`BLOCK_TASK` stand the static block in x, from its declared place [m]."""

BLOCK_TASK = """
(define (problem static_block_{name})
  (:staging (:scene bench_075) (:embodiment panda) (:physics mujoco_500hz@v1) (:max-steps 50)
            (:episodes (:count 2) (:sampler {sampler}){keys}))
  (:objects block - static_block_40mm)
  (:goal (<= (speed block) 0.02)))
"""
"""A 40 mm static block beside the Panda, on MuJoCo, over two episodes."""


def _moves_the_block(task, episode_id, rng) -> dict:
    del task, rng
    return {"object_poses": {"block": (MOVES[episode_id % 2], -0.1, 0.02, 0.0)}}


CUBE_MOVES = (-0.1, 0.1)
"""Where the two episodes of :data:`CUBE_AND_BLOCK_TASK` stand the free cube in x [m]."""

CUBE_AND_BLOCK_TASK = """
(define (problem static_block_after_a_cube)
  (:staging (:scene bench_075) (:embodiment panda) (:physics mujoco_500hz@v1) (:max-steps 50)
            (:episodes (:count 2) (:sampler static_block_and_cube_move)))
  (:objects cube - free_cube_40mm block - static_block_40mm)
  (:goal (<= (speed block) 0.02)))
"""
"""A free cube declared before the static block, both moved by each episode."""


def _moves_the_cube_and_the_block(task, episode_id, rng) -> dict:
    del task, rng
    k = episode_id % 2
    return {"object_poses": {"cube": (CUBE_MOVES[k], 0.1, 0.02, 0.0), "block": (MOVES[k], -0.1, 0.02, 0.0)}}


def _held_x(sim, entity, world: int = 0) -> float:
    """Where MuJoCo holds a static body in x, in one world: its mocap position [m]."""
    solver = sim.pipeline.program.mujoco_solver
    worlds = getattr(sim, "worlds", None)
    body = worlds.body(entity.body, world) if worlds is not None else entity.body
    mjc = int(np.flatnonzero(solver.mjc_body_to_newton.numpy()[world] == body)[0])
    return float(solver.mjw_data.mocap_pos.numpy()[world, int(solver.mj_model.body_mocapid[mjc])][0])


class TestAStaticBoxBesideTheYamUnderGeneric(unittest.TestCase):
    """On VBD beside the arm's MuJoCo: a cube in the air, and a block the right hand presses."""

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        scene = _yam_scene(rss.GENERIC, "static_box_generic")
        cls.cube = scene.add(rss.box(name="cube", half=(0.05,) * 3, at=(0.0, 0.0, 0.2), static=True))
        cls.block = scene.add(rss.box(name="block", half=BLOCK_HALF, at=(*UNDER_HAND, BLOCK_HALF[2]), static=True))
        cls.sim = _build(scene, control_mode="joint_position")
        cls.cube_at, cls.block_at = _at(cls.cube), _at(cls.block)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def test_a_static_cube_in_the_air_stays_there(self) -> None:
        self.sim.run(60).sync()
        np.testing.assert_allclose(_at(self.cube), self.cube_at, atol=1.0e-6, err_msg="it fell 148 mm")
        np.testing.assert_allclose(_at(self.block), self.block_at, atol=1.0e-6, err_msg="it sank 3.1 mm")

    def test_the_shut_hand_pressing_down_stops_on_the_block_and_moves_it_nowhere(self) -> None:
        sim = self.sim
        ik, gripper = sim.arm("right"), sim.gripper("right")
        gripper.command = 1.0
        sim.run(30).sync()
        start, quat = ik.ee_pose(sim.state)
        start = np.asarray(start, dtype=np.float64)
        top = SURFACE[2] + 2 * BLOCK_HALF[2]
        goal = np.array([SURFACE[0] + UNDER_HAND[0], UNDER_HAND[1], top - 0.02])
        frames = int(1.5 * float(sim.request.physics.control_hz))
        for k in range(frames):
            a = min(1.0, (k + 1) / (frames * 2 / 3))  # down over 1 s, then held 0.5 s
            ik.set_target(start * (1.0 - a) + goal * a, quat)
            ik.solve_into(sim.state.joint_q, sim.control.joint_target_q)
            gripper.apply(sim.control.joint_target_q)
            sim.run(1)
        sim.sync()
        tcp, _ = ik.ee_pose(sim.state)
        self.assertTrue(sim.state_is_finite())
        np.testing.assert_allclose(_at(self.block), self.block_at, atol=1.0e-5, err_msg="it slid 10 mm")
        self.assertGreater(
            float(tcp[2]) - top, 0.012, "the fingertips rest on its top: 16.7 mm over it, as over a slab; 8.6 mm sunk"
        )


class TestAStaticBoxOnMujoco(unittest.TestCase):
    """The same cube in one MuJoCo with the arm: it converts, as a mocap body welded to the world.

    Read off MuJoCo, not off ``body_q``. ``SolverMuJoCo`` writes ``body_q`` back
    by forward kinematics over the articulations, and a mocap body is in none, so
    the cube's row keeps its build value whatever MuJoCo does with it. With its
    weld moved 100 mm in x on the MuJoCo side, ``mocap_pos`` and ``xpos`` moved
    100 mm and ``body_q`` did not (``d2_probe.py``).
    """

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        scene = _yam_scene(rss.MUJOCO_500HZ, "static_box_mujoco")
        cls.cube = scene.add(rss.box(name="cube", half=(0.05,) * 3, at=(0.0, 0.0, 0.2), static=True))
        cls.sim = _build(scene, control_mode="joint_position")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def _held(self) -> tuple[np.ndarray, np.ndarray]:
        """Where MuJoCo holds the cube: its mocap position and its quaternion as xyzw."""
        solver = self.sim.pipeline.program.mujoco_solver
        mjc = np.flatnonzero(solver.mjc_body_to_newton.numpy()[0] == self.cube.entity.body)
        self.assertEqual(1, len(mjc), "one MuJoCo body is the cube")
        mocap = int(solver.mj_model.body_mocapid[int(mjc[0])])
        self.assertGreaterEqual(mocap, 0, "a mocap body: the world root a kinematic weld converts to")
        w, x, y, z = solver.mjw_data.mocap_quat.numpy()[0, mocap].astype(np.float64)
        return solver.mjw_data.mocap_pos.numpy()[0, mocap].astype(np.float64), np.array([x, y, z, w])

    def test_it_converts_and_stays_in_the_air(self) -> None:
        built = np.asarray(self.cube.pose, dtype=np.float64)
        self.sim.run(50).sync()
        self.assertTrue(self.sim.state_is_finite())
        pos, quat = self._held()
        np.testing.assert_allclose(pos, built[:3], atol=1.0e-6, err_msg="MuJoCo holds it where it was built")
        np.testing.assert_allclose(quat, built[3:7], atol=1.0e-6)
        np.testing.assert_allclose(pos, (SURFACE[0], SURFACE[1], SURFACE[2] + 0.2), atol=1.0e-6)


class TestATowelOverAStaticBox(unittest.TestCase):
    """A towel on VBD dropped over a static 10 cm cube, no robot: one VBD island, nothing else."""

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        scene = rss.Scene(cell="bench_075", physics=rss.GENERIC, name="towel_over_a_static_box")
        cls.towel = scene.add(
            rss.sheet(name="towel", size=(0.3, 0.3), cells=(20, 20), at=(-0.15, -0.15, 0.12), settle_frames=0)
        )
        cls.cube = scene.add(rss.box(name="cube", half=(0.05,) * 3, at=(0.0, 0.0, 0.05), static=True))
        cls.sim = _build(scene)
        cls.cube_at = _at(cls.cube)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def test_the_towel_lies_on_it_and_it_does_not_sink(self) -> None:
        self.sim.run(120).sync()
        np.testing.assert_allclose(_at(self.cube), self.cube_at, atol=1.0e-6, err_msg="it sank 14 mm")
        q = self.sim.pipeline.state_0.particle_q.numpy().astype(np.float64)
        over = q[np.argmin(np.linalg.norm(q[:, :2] - self.cube_at[:2], axis=1))]
        self.assertGreater(over[2], self.cube_at[2] + 0.05, "the towel lies on the cube's top")

    def test_z_it_resets_and_steps_again(self) -> None:
        for _ in range(2):
            self.sim.reset()
            self.sim.run(5).sync()
        self.assertTrue(self.sim.state_is_finite())
        np.testing.assert_allclose(_at(self.cube), self.cube_at, atol=1.0e-6)


class TestAnEpisodeThatMovesAStaticBox(unittest.TestCase):
    """A static block whose two episodes stand it 80 mm apart, beside the Panda on MuJoCo.

    The weld has no coordinates and nothing placed a world's copy of it, so both
    used to run with the block where episode 0 put it, 80 mm short of where
    episode 1 asked: serially, on one model reused because the two episodes' model
    keys are equal, and in world 1 of a two-world fleet (``d3_probe.py``). Both
    are refused by name now, and the way out the refusal names works.
    """

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss
        from robosimstudio.assetlib import register_asset
        from robosimstudio.domains.rigid import Body
        from robosimstudio_bench.taskbuild.lower import register_sampler

        register_sampler("static_block_moves", _moves_the_block)
        register_sampler("static_block_and_cube_move", _moves_the_cube_and_the_block)
        block = rss.box(name="block", half=(0.02,) * 3, at=(0.0, -0.1, 0.02), static=True)
        register_asset(Body.from_spec(block, name="static_block_40mm"))
        cube = rss.box(name="cube", half=(0.02,) * 3, at=(0.0, 0.1, 0.02))
        register_asset(Body.from_spec(cube, name="free_cube_40mm"))

    @staticmethod
    def _task(name: str, sampler: str = "static_block_moves", keys: str = ""):
        import robosimstudio_bench as bench

        return bench.task(BLOCK_TASK.format(name=name, sampler=sampler, keys=keys))

    @staticmethod
    def _quiet(make):
        with redirect_stdout(io.StringIO()):
            return make()

    def test_a_serial_env_refuses_the_episode_that_moves_it(self) -> None:
        import robosimstudio_bench as bench

        env = self._quiet(lambda: bench.Env(self._task("serial")))
        try:
            self._quiet(lambda: env.reset(0))
            with self.assertRaisesRegex(ValueError, "'block' is static.*model-key-fields"):
                self._quiet(lambda: env.reset(1))
        finally:
            env.close()

    def test_a_refused_reset_leaves_every_pose_its_episode_and_its_evaluator(self) -> None:
        """The free cube is reset before the block refuses: it was left at episode 1's x = 0.70, ``env.episode`` 1."""
        import robosimstudio_bench as bench

        env = self._quiet(lambda: bench.Env(bench.task(CUBE_AND_BLOCK_TASK)))
        try:
            self._quiet(lambda: env.reset(0))
            env.run(5)
            states = (env.pipeline.state_0, env.pipeline.state_1)
            fields = ("joint_q", "joint_qd", "body_q", "body_qd")
            before = [getattr(state, f).numpy().copy() for state in states for f in fields]
            episode, evaluator, clock = env.episode, env.evaluator, (env.step_count, env.pipeline.sim_time)
            with self.assertRaisesRegex(ValueError, "'block' is static.*model-key-fields"):
                self._quiet(lambda: env.reset(1))
            after = [getattr(state, f).numpy() for state in states for f in fields]
            for name, was, now in zip([f"state_{i}.{f}" for i in (0, 1) for f in fields], before, after, strict=True):
                np.testing.assert_array_equal(was, now, f"{name} moved")
            self.assertAlmostEqual(SURFACE[0] + CUBE_MOVES[0], float(env["cube"].position[0]), places=2)
            self.assertIs(episode, env.episode)
            self.assertIs(evaluator, env.evaluator)
            self.assertEqual(clock, (env.step_count, env.pipeline.sim_time))
        finally:
            env.close()

    def test_named_a_model_key_field_each_layout_builds_its_own_model(self) -> None:
        import robosimstudio_bench as bench

        env = self._quiet(lambda: bench.Env(self._task("keyed", keys=" (:model-key-fields object_poses)")))
        try:
            for episode, x in enumerate(MOVES):
                self._quiet(lambda episode=episode: env.reset(episode))
                self.assertAlmostEqual(SURFACE[0] + x, _held_x(env, env["block"].entity), places=5)
        finally:
            env.close()

    def test_a_fleet_refuses_the_world_whose_episode_moves_it(self) -> None:
        from robosimstudio.scene.compiler import CannotReplicate
        from robosimstudio_bench.envs.fleet import FleetEnv
        from robosimstudio_bench.run.fleet import plan_waves

        task = self._task("fleet")
        fleet = self._quiet(lambda: FleetEnv(task, num_worlds=2))
        try:
            (wave,) = plan_waves(task, [0, 1], 2)
            with self.assertRaisesRegex(CannotReplicate, "world 1 runs .*'block' is static"):
                fleet.reset_wave(wave)
        finally:
            fleet.close()

    def test_a_fleet_whose_episodes_leave_it_in_place_holds_it_there_in_every_world(self) -> None:
        from robosimstudio_bench.envs.fleet import FleetEnv
        from robosimstudio_bench.run.fleet import plan_waves

        task = self._task("fixed", sampler="fixed")
        fleet = self._quiet(lambda: FleetEnv(task, num_worlds=2))
        try:
            (wave,) = plan_waves(task, [0, 1], 2)
            self._quiet(lambda: fleet.reset_wave(wave))
            entity = fleet.scene.objects["block"]
            for world in range(2):
                self.assertAlmostEqual(SURFACE[0], _held_x(fleet, entity, world), places=5)
        finally:
            fleet.close()


ARENA_BIN = (0.6, 0.4, 0.77)
"""Where :data:`ARENA_BIN_TASK`'s arena stands its ``bin``, on ``bench_075``'s top, in world [m]."""

ARENA_BIN_TASK = """
(define (problem arena_bin_on_another_model)
  (:staging (:scene bench_075_arena_bin) (:embodiment panda) (:physics mujoco_500hz@v1) (:max-steps 50)
            (:episodes (:count 2) (:sampler arena_bin_moves) (:model-key-fields layout)))
  (:fixtures bin - bin)
  (:goal (<= (speed bin) 0.02)))
"""
"""A ``bin`` the scene's arena declares, over two episodes each on a model of its own."""


def _moves_the_arena_bin(task, episode_id, rng) -> dict:
    """Episode 1 is another model's, runs another active set, and names the arena's ``bin`` 50 mm along x."""
    del task, rng
    if episode_id % 2 == 0:
        return {"layout": 0}
    moved = (ARENA_BIN[0] + 0.05, *ARENA_BIN[1:], 0.0, 0.0, 0.0, 1.0)
    return {"layout": 1, "active": ("bin",), "object_xforms": {"bin": moved}}


class TestAnEpisodeOfAnotherModelThatMovesAnArenaStatic(unittest.TestCase):
    """A reset refused on the way to another model leaves the env on the one it had.

    Only a reset that kept its model asked the layout before taking it up. One that rebuilt closed the
    renderers, compiled the new model and took up the episode and its evaluator, and only then was it
    refused: the env was left on a fresh model it had never reset, at the episode it refused, with its
    renderers closed.
    """

    @classmethod
    def setUpClass(cls) -> None:
        import tempfile
        from pathlib import Path

        from robosimstudio.scene.builtin.benches import BENCH_075
        from robosimstudio.scene.registry import register_cell
        from robosimstudio.specs import AssetSpec, StaticSpec, replace
        from robosimstudio_bench.taskbuild.lower import register_sampler

        cls._dir = tempfile.TemporaryDirectory()
        arena = Path(cls._dir.name) / "arena.xml"
        x, y, z = ARENA_BIN
        arena.write_text(
            f'<mujoco><worldbody><body name="bin" pos="{x} {y} {z}"><freejoint/>'
            '<geom type="box" size="0.02 0.02 0.02" mass="0.1"/></body></worldbody></mujoco>'
        )
        statics = (*BENCH_075.statics, StaticSpec(name="arena", asset=AssetSpec(path=str(arena))))
        register_cell(replace(BENCH_075, name="bench_075_arena_bin", statics=statics))
        register_sampler("arena_bin_moves", _moves_the_arena_bin)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._dir.cleanup()

    def test_it_keeps_its_scene_model_renderers_episode_and_evaluator(self) -> None:
        import robosimstudio_bench as bench

        class Renderer:
            closed = False

            def close(self) -> None:
                self.closed = True

            def render(self, state) -> dict:
                return {}

        env = TestAnEpisodeThatMovesAStaticBox._quiet(lambda: bench.Env(bench.task(ARENA_BIN_TASK)))
        try:
            TestAnEpisodeThatMovesAStaticBox._quiet(lambda: env.reset(0))
            env.renderer, env.obs_renderer = Renderer(), Renderer()
            names = ("scene", "model", "pipeline", "renderer", "obs_renderer", "episode", "evaluator")
            before = {name: getattr(env, name) for name in names}
            with self.assertRaisesRegex(ValueError, "'bin' is static, where the scene's arena stood it"):
                TestAnEpisodeThatMovesAStaticBox._quiet(lambda: env.reset(1))
            self.assertEqual([], [name for name in names if getattr(env, name) is not before[name]], "replaced")
            self.assertEqual([], [name for name in names[3:5] if before[name].closed], "closed")
            TestAnEpisodeThatMovesAStaticBox._quiet(lambda: env.reset(0))
            env.run(5)
            self.assertTrue(env.state_is_finite())
            np.testing.assert_allclose(ARENA_BIN, env["bin"].position, atol=1.0e-3)
        finally:
            env.renderer = env.obs_renderer = None
            env.close()


if __name__ == "__main__":
    unittest.main()
