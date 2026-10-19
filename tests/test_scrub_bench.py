# SPDX-License-Identifier: Apache-2.0
"""The brush examples' headline task, done the way their pages ask: the hand scrubs the marked square.

``examples/03_materials/07_brush`` and ``06_tasks/04_brush_scored`` promise a
gesture -- the hand down onto the brush's handle, **Close**, press the
bristles to the bench, scrub the square -- and a number, the share of the
square's 5 mm cells a bristle tip has touched while down. This drives the arm
panel through that gesture with no browser on it
(:mod:`tests.fixtures.grasp`): the cursor the gizmo writes, the grip the Close
button writes, the session's own setpoint law. Nothing is pinned: the handle
stays in the fingers because the pads hold it, and the bristles mark the bench
because they are on it. Both scripts are held to the number: the page's
readout passes half, and the task document's goal turns true.

The strokes are three lengths of :data:`STROKE` along the bench's y, 12 mm
apart in x so the 80 mm brush covers the 100 mm square, at :data:`SPEED`, with
the hand :data:`PRESS` below where it closed.
"""

from __future__ import annotations

import math
import time
import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""Two scenes built on one GPU, each stepped through a page for about twelve seconds of scene time."""

TOOL_DOWN = (0.0, 1.0, 0.0, 0.0)
"""The YAM's tool pointing down, jaws closing along world y: the turn a person gives the gizmo first."""

BENCH = np.array([0.6, 0.0, 0.75])
"""``bench_075``'s surface frame in world [m]; the examples' numbers are in it."""

PATCH_LO = BENCH[:2] + np.array([-0.025, -0.27])
PATCH_HI = BENCH[:2] + np.array([0.075, -0.17])
"""The square as the examples state it, in world x, y [m]."""

PAD_TIP = 0.0165
"""How far the YAM's pad tips reach below its tool centre [m], as the wipe test grasps."""

CLOSE_OVER_SOLE = 0.008
"""Where the pad tips stop when the hand closes: this far over the sole, on the handle's lower part [m]."""

PRESS = 0.004
"""How far below the closing height the hand goes to scrub [m]."""

STROKE = 0.15
SPEED = 0.10
"""Each stroke's length [m] and the cursor's walk along it [m/s]."""

LANES = (-0.012, 0.0, 0.012)
"""The strokes' x offsets from the brush's start [m]: out along the first, back along the second, out along the third."""


def _walk(session, arm, target, out: dict, track) -> None:
    """The cursor walked to ``target`` at :data:`SPEED`, held as a drag; ``track`` sees every step."""
    hz = float(session.env.request.physics.control_hz)
    start = np.array(arm.cursor_pos, dtype=np.float64)
    legs = max(1, math.ceil(float(np.linalg.norm(target - start)) / (SPEED / hz)))
    for k in range(1, legs + 1):
        arm.cursor_pos, arm.dragging = start + (target - start) * (k / legs), k < legs
        tick(session, 1)
        track(out)
    arm.cursor_pos, arm.dragging = target, False


def _scrub(session, brush, handle, *, turn: bool) -> dict:
    """Over the handle, down, **Close**, press, three strokes, lift; what each stage did."""
    arm = session.arms["right"]
    entity = brush.entity
    floor = float(entity.floor_z)
    tick(session, 30)
    if turn:
        arm.cursor_quat = np.asarray(TOOL_DOWN, dtype=np.float64)
    standing = np.asarray(handle.position, dtype=np.float64)
    grasp = np.array([standing[0], standing[1], floor + entity.spec.sole_top + CLOSE_OVER_SOLE + PAD_TIP])
    go(session, "right", grasp + np.array([0.0, 0.0, 0.04]), speed=0.10)
    out: dict = {"miss_grip": go(session, "right", grasp, speed=0.05)}
    arm.grip_request = 1.0
    tick(session, 60)
    gripper = session.env.gripper("right")
    out["width"] = float(gripper.width(session.env.state))

    def hand() -> np.ndarray:
        return arm.ik.ee_pose(session.env.state)[0]

    offset = np.asarray(handle.position, dtype=np.float64) - hand()
    out.update(slip=0.0, down=[], low=math.inf)

    def track(row: dict) -> None:
        row["slip"] = max(row["slip"], float(np.linalg.norm(np.asarray(handle.position) - hand() - offset)))
        tips = entity.tip_positions(session.env.state.body_q.numpy())
        row["down"].append(float(((tips[:, 2] - floor) <= entity.spec.bristle_radius + 0.002).mean()))
        row["low"] = min(row["low"], float((tips[:, 2] - floor).min()))

    pressed = grasp - np.array([0.0, 0.0, PRESS])
    go(session, "right", pressed, speed=0.02, steps=60)
    entity.clear_scrub()
    start = pressed + np.array([LANES[0], 0.0, 0.0])
    go(session, "right", start, speed=0.02, steps=60)
    out["before"] = entity.scrubbed(PATCH_LO, PATCH_HI)
    for k, lane in enumerate(LANES):
        there = pressed + np.array([lane, STROKE if k % 2 == 0 else 0.0, 0.0])
        if k:
            _walk(session, arm, np.array([there[0], arm.cursor_pos[1], there[2]]), out, track)
        _walk(session, arm, there, out, track)
    tick(session, 10)
    out["travel"] = float(np.asarray(handle.position)[1] - standing[1])
    out["scrubbed"] = entity.scrubbed(PATCH_LO, PATCH_HI)
    go(session, "right", hand() + np.array([0.0, 0.0, 0.03]), speed=0.05)
    out["finite"] = bool(session.env.state_is_finite())
    return out


class _Scrub:
    """Build the example, open its page, scrub; the subclasses say which script and what it must read."""

    EXAMPLE = ""
    PORT = 0
    TURN = False

    @classmethod
    def setUpClass(cls) -> None:
        cls.opened, kwargs = _run(cls.EXAMPLE)
        cls.sim = getattr(cls.opened, "sim", cls.opened)
        cls.sim.run(20).sync()
        t = time.perf_counter()
        cls.sim.run(100).sync()
        cls.ms_per_step = 1e3 * (time.perf_counter() - t) / 100
        cls.session = open_session(cls.opened, cls.PORT, kwargs.get("page"))
        cls.brush, cls.handle = cls.sim["brush"], cls.sim["handle"]
        cls.standing = np.asarray(cls.handle.position, dtype=np.float64)
        cls.result = _scrub(cls.session, cls.brush, cls.handle, turn=cls.TURN)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_scene_steps_captured_inside_its_budget(self) -> None:
        self.assertTrue(self.sim.pipeline.captured, self.sim.pipeline.capture_error)
        self.assertLess(self.ms_per_step, 40.0, f"{self.ms_per_step:.1f} ms per control step")

    def test_the_fingers_closed_on_the_handle(self) -> None:
        self.assertLess(self.result["miss_grip"], 0.008, "the hand did not get down to the handle")
        self.assertAlmostEqual(0.022, self.result["width"], delta=0.003, msg="the jaws are not on the handle's faces")

    def test_the_handle_stayed_in_the_fingers(self) -> None:
        self.assertTrue(self.result["finite"])
        self.assertLess(self.result["slip"], 0.005, "the handle slid in the fingers")
        self.assertGreater(self.result["travel"], STROKE - 0.02, "the brush did not come along the first stroke")

    def test_the_tips_stayed_on_the_bench(self) -> None:
        down = np.asarray(self.result["down"])
        self.assertGreater(float(down.mean()), 0.9, f"{down.mean():.2f} of the tips were down through the strokes")
        self.assertGreater(self.result["low"], 0.0, "a tip's rounded end went a whole radius into the bench")

    def test_half_the_square_is_scrubbed(self) -> None:
        self.assertEqual(0.0, self.result["before"], "pressed beside the square, the brush has scrubbed none of it")
        self.assertGreaterEqual(self.result["scrubbed"], 0.5, f"{self.result['scrubbed']:.3f} of the square")


class TestThePageScrubsTheSquare(_Scrub, unittest.TestCase):
    """``03_materials/07_brush``: the hand starts over the handle, pointing down; the readout passes half."""

    EXAMPLE = "03_materials/07_brush.py"
    PORT = 9655

    def test_the_hand_starts_over_the_handle_pointing_down(self) -> None:
        pos, quat = self.session.arms["right"].ik.home_pose
        self.assertLess(
            float(np.linalg.norm(pos[:2] - self.standing[:2])), 0.01, "the home tool centre is over the handle"
        )
        self.assertGreater(float(pos[2] - self.standing[2]), 0.03, "and above it")
        x, y, z, w = quat
        tool_z = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        self.assertLess(float(tool_z[2]), -0.95, f"the tool points down: {tool_z.round(3)}")


class TestTheTaskGoalTurnsTrue(_Scrub, unittest.TestCase):
    """``06_tasks/04_brush_scored``: the same scrub on the task's page, from the robot's own home, and the goal holds."""

    EXAMPLE = "06_tasks/04_brush_scored.py"
    PORT = 9656
    TURN = True

    def test_the_goal_is_met(self) -> None:
        _, _, info = self.opened.evaluate()
        self.assertTrue(info["success"], f"metrics: {info['metrics']}")
        self.assertGreaterEqual(float(info["metrics"]["fraction"]), 0.5)


if __name__ == "__main__":
    unittest.main()
