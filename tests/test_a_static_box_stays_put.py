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
"""A GPU: three scenes built, two beside the YAM, each stepped a second or two."""

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


if __name__ == "__main__":
    unittest.main()
