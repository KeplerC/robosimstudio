# SPDX-License-Identifier: Apache-2.0
"""The roll example's gesture, made the way its page makes it, rolls the dough flat.

``tests/scenes/00_roll_dough`` promises that a hand dragged down onto the
roller's handle, closed, lifted out of the rest and rolled across the block
three times leaves the dough wider and lower. This drives the example's own arm
panel through :mod:`tests.fixtures.grasp` -- the cursor the gizmo writes, the
grip the Close button writes, the session's own setpoint law -- and reads the
two numbers the page shows. Nothing is pinned and no handle is dragged: the
roller moves because the fingers hold its handle, and the dough flattens
because the drum presses it.

Three things the roll also has to show, because each was once the failure: the
jaws are on the handle's 20 mm web and stay there (a plain handle slid 30 mm up
through the closed jaws in one pass), the drum turns on its axle (welded, it
slides and leaves the block at 18.6 mm where rolled it leaves 13.2), and the
dough stays on the bench.
"""

from __future__ import annotations

import time
import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""One GPU: the example built once and stepped through its page for about 12 s of scene time."""

PORT = 9634
TOOL_DOWN = (0.0, 1.0, 0.0, 0.0)
"""The YAM's tool pointing down, jaws closing along world y: the example's home already points it so."""

BENCH = np.array([0.6, 0.0, 0.75])
"""``bench_075``'s surface frame in world [m]; the example's numbers are in it."""

GRIP = 0.05
"""The tool centre over the axle where the jaws close on the handle [m]: ``rss.roller``'s default ``grip``."""

ROLL = 0.088
"""Where the cursor is held over the bench while the drum rolls [m].

The tool centre settles about 3 mm under the cursor and the axle 52 mm under
the tool centre, once the dough has pushed the shoulder onto the fingertips, so
the drum's underside runs about 11 mm over the bench."""

NEAR, FAR = -0.085, 0.075
"""Where each pass starts and ends, along the bench's x [m]: the drum's edge 25 mm short of the block, and past its far side."""

SPEED = 0.10
"""The cursor's walk across the block [m/s], under the ~0.36 m/s a 4 mm voxel allows a tool at 60 Hz."""

PASSES = 3


class TestTheRollByHand(unittest.TestCase):
    """Down onto the handle, Close, out of the rest, three passes: the dough ends wider and lower."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim, kwargs = _run("tests/scenes/00_roll_dough.py")
        cls.sim.run(20).sync()
        t = time.perf_counter()
        cls.sim.run(100).sync()
        cls.ms_per_step = 1e3 * (time.perf_counter() - t) / 100
        cls.session = open_session(cls.sim, PORT, kwargs.get("page"))
        cls.dough, cls.roller = cls.sim["dough"], cls.sim["roller"]
        cls.rest = cls.dough.entity.rest["particle_q"][cls.dough.particles]
        cls.floor = float(cls.dough.entity.floor_z)
        tick(cls.session, 30)
        cls.settled = cls._readings()
        cls.result = cls._roll()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    @classmethod
    def _readings(cls) -> tuple[float, float]:
        from robosimstudio_materials.mpm.measures import heap_height, spread

        points = np.asarray(cls.dough.particle_q, dtype=np.float64)
        return spread(points, cls.rest), heap_height(points, cls.floor)

    @classmethod
    def _axle(cls) -> float:
        return float(cls.session.env.state.joint_q.numpy()[cls.roller.entity.axle_coord])

    @classmethod
    def _roll(cls) -> dict:
        session, roller = cls.session, cls.roller
        arm = session.arms["right"]
        arm.cursor_quat = np.asarray(TOOL_DOWN, dtype=np.float64)

        def hand() -> np.ndarray:
            return arm.ik.ee_pose(session.env.state)[0]

        out: dict = {"axle_rest": np.asarray(roller.position, dtype=np.float64)}
        grip = out["axle_rest"] + np.array([0.0, 0.0, GRIP])
        go(session, "right", grip + np.array([0.0, 0.0, 0.04]), speed=0.10)
        out["miss_grip"] = go(session, "right", grip, speed=0.06)
        arm.grip_request = 1.0
        tick(session, 45)
        out["width"] = float(session.env.gripper("right").width(session.env.state))
        offset = np.asarray(roller.position, dtype=np.float64) - hand()
        go(session, "right", grip + np.array([0.0, 0.0, 0.03]), speed=0.06)
        out["rose"] = float(np.asarray(roller.position)[2] - out["axle_rest"][2])
        y = float(grip[1])
        go(session, "right", np.array([BENCH[0] + NEAR, y, BENCH[2] + ROLL + 0.01]), speed=0.08)
        go(session, "right", np.array([BENCH[0] + NEAR, y, BENCH[2] + ROLL]), speed=0.05)
        out["turned"], out["slip"], out["passes"], out["lowest"] = [], 0.0, [], np.inf
        for k in range(PASSES):
            end = BENCH[0] + (FAR if k % 2 == 0 else NEAR)
            start = np.array(arm.cursor_pos, dtype=np.float64)
            target = np.array([end, y, BENCH[2] + ROLL])
            legs = int(np.ceil(np.linalg.norm(target - start) / (SPEED / 60.0)))
            angle = cls._axle()
            for i in range(1, legs + 1):
                arm.cursor_pos, arm.dragging = start + (target - start) * (i / legs), i < legs
                tick(session, 1)
                out["slip"] = max(out["slip"], float(np.linalg.norm(np.asarray(roller.position) - hand() - offset)))
                out["lowest"] = min(out["lowest"], float(np.asarray(cls.dough.particle_q)[:, 2].min()))
            arm.cursor_pos, arm.dragging = target, False
            tick(session, 20)
            out["turned"].append(abs(cls._axle() - angle))
            out["passes"].append(cls._readings())
        go(session, "right", np.array(arm.cursor_pos) + np.array([0.0, 0.0, 0.05]), speed=0.05)
        tick(session, 60)
        out["final"] = cls._readings()
        out["finite"] = bool(session.env.state_is_finite())
        return out

    def test_the_scene_steps_captured_inside_its_budget(self) -> None:
        self.assertTrue(self.sim.pipeline.captured, self.sim.pipeline.capture_error)
        self.assertLess(self.ms_per_step, 40.0, f"{self.ms_per_step:.1f} ms per control step; the docstring says 21.8")

    def test_the_fingers_closed_on_the_handle_and_lifted_it(self) -> None:
        self.assertLess(self.result["miss_grip"], 0.006, "the hand did not get down to the handle")
        self.assertAlmostEqual(0.020, self.result["width"], delta=0.003, msg="the jaws are not on the handle's web")
        self.assertGreater(self.result["rose"], 0.015, "the roller did not come up out of its rest")

    def test_the_handle_stayed_put_in_the_hand(self) -> None:
        self.assertTrue(self.result["finite"])
        self.assertLess(self.result["slip"], 0.005, f"the roller moved {self.result['slip'] * 1e3:.1f} mm in the hand")

    def test_the_drum_rolled(self) -> None:
        """Each 160 mm pass is 8 rad of a pure roll; a drum that slides turns well under one."""
        for k, turned in enumerate(self.result["turned"]):
            self.assertGreater(turned, 2.0, f"pass {k + 1}: the drum turned {turned:.2f} rad")

    def test_the_dough_stays_on_the_bench(self) -> None:
        self.assertGreater(self.result["lowest"], BENCH[2] - 0.006, "dough went under the bench top during the roll")

    def test_the_dough_is_rolled_wider_and_lower(self) -> None:
        footprint, height = self.result["final"]
        _, settled = self.settled
        self.assertGreaterEqual(footprint, 1.2, f"the footprint is {footprint:.2f} of the settled one")
        self.assertLessEqual(height, 0.6 * settled, f"{height * 1e3:.1f} mm of a settled {settled * 1e3:.1f} mm")


if __name__ == "__main__":
    unittest.main()
