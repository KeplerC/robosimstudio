# SPDX-License-Identifier: Apache-2.0
"""The cutting example's headline task, done by the robot's hand the way its page asks.

``examples/20_cutting/00_cut_tofu`` promises a gesture and a number. This drives the example's own
arm panel through :mod:`tests.fixtures.grasp` -- the cursor the gizmo writes, the grip the **Close**
button writes -- and reads what the page's **Tofu** folder reads. The right hand is put down on the
knife's handle, closes, lifts the knife 70 mm out of its rest, carries it over the tofu and pushes it
down in 4 mm steps until the edge meets the board. Nothing pins the knife to the hand: the pads
squeeze its handle. Two controls ride on the same drive: over the tofu, and stopped 15 mm over the
board, the tofu is still one piece. A second drive does the same with **Dull blade** ticked, a blade
a thousand times duller: the arm stalls on the tofu and it does not part.
"""

from __future__ import annotations

import time
import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""One GPU: the example built twice and driven through its page, about 11 ms a control step."""

PORT = 9841
DULL_PORT = 9842
WORK = np.array([0.6, 0.0, 0.75])
"""``bench_075``'s work-surface origin in world [m]."""
TOFU_Y = -0.16
"""The tofu's middle across the bench, as the example places it [m]."""
BOARD_TOP = 0.012
"""The board's top over the bench [m]."""
EDGE = 0.04
"""The edge's depth under the handle's middle, the knife's frame [m]."""
LIFT = 0.07
"""How far the hand lifts the knife out of its rest [m]: the edge clears the 52 mm tofu top by 18 mm."""


def cut(session, tofu, knife, *, stop_over: float = 0.0) -> dict:
    """Down onto the handle, **Close**, up, across, then down in 4 mm steps to ``stop_over`` above the board.

    ``slip`` is how far the knife has moved from where it sat in the hand when the pads closed [m].
    """
    arm = session.arms["right"]

    def hand() -> np.ndarray:
        return arm.ik.ee_pose(session.env.state)[0]

    def where() -> np.ndarray:
        return np.asarray(knife.position, dtype=np.float64)

    handle = where()
    out: dict = {"home_to_handle": float(np.linalg.norm(arm.ik.home_pose[0][:2] - handle[:2]))}
    out["miss"] = go(session, "right", handle, speed=0.04)
    arm.grip_request = 1.0
    tick(session, 50)
    offset = where() - hand()
    out["closed_width"] = float(session.env.gripper("right").width(session.env.state))
    up = handle + np.array([0.0, 0.0, LIFT])
    go(session, "right", up, speed=0.04)
    over = np.array([up[0], WORK[1] + TOFU_Y, up[2]])
    go(session, "right", over, speed=0.05)
    out["over"] = {"pieces": tofu.pieces(), "fraction": tofu.cut_fraction()}
    cursor, pushes = over.copy(), []
    bottom = WORK[2] + BOARD_TOP + EDGE - 0.005 + stop_over
    t, steps = time.perf_counter(), session.env.step_count
    for z in np.arange(over[2], bottom - 1.0e-9, -0.004):
        cursor[2] = z
        go(session, "right", cursor.copy(), speed=0.02, steps=20)
        pushes.append(tofu.push)
    go(session, "right", cursor.copy(), steps=100)
    out["drive_ms"] = 1e3 * (time.perf_counter() - t) / max(session.env.step_count - steps, 1)
    out.update(
        pieces=tofu.pieces(),
        fraction=tofu.cut_fraction(),
        depth=tofu.cut_depth(),
        pushes=pushes,
        edge=float(where()[2] - WORK[2] - EDGE),
        slip=float(np.linalg.norm(where() - hand() - offset)),
        finite=bool(session.env.state_is_finite()),
    )
    return out


def _drive(port: int, *, dull: bool) -> dict:
    """Build the example, drive its page through the cut, stopping 15 mm over the board on the way."""
    sim, kwargs = _run("20_cutting/00_cut_tofu.py")
    tofu, knife = sim["tofu"].entity, sim["knife"]
    session = open_session(sim, port, kwargs.get("page"))
    try:
        if dull:
            tofu.set_sharpness(0.001)  # what the page's Dull blade box writes
        tick(session, 30)
        got = cut(session, tofu, knife, stop_over=0.015)
        got["stopped"] = {"pieces": got["pieces"], "fraction": got["fraction"], "edge": got["edge"]}
        if not dull:
            arm = session.arms["right"]
            cursor = np.array(arm.cursor_pos, dtype=np.float64)
            for z in np.arange(cursor[2] - 0.004, WORK[2] + BOARD_TOP + EDGE - 0.005 - 1.0e-9, -0.004):
                cursor[2] = z
                go(session, "right", cursor.copy(), speed=0.02, steps=20)
            go(session, "right", cursor.copy(), steps=100)
            got.update(
                pieces=tofu.pieces(),
                fraction=tofu.cut_fraction(),
                edge=float(np.asarray(knife.position)[2] - WORK[2] - EDGE),
                cuts=tofu.cuts(),
                finite=bool(sim.state_is_finite()),
            )
        got["tick_ms"] = tofu.timing.get("total_ms", 0.0)
    finally:
        session.teardown()
    return got


class TestTheHandCutsTheTofu(unittest.TestCase):
    """Take the knife, lift it, carry it over the tofu and push it down to the board: two pieces."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.got = _drive(PORT, dull=False)

    def test_the_right_hand_starts_over_the_handle_and_gets_there(self) -> None:
        self.assertLess(self.got["home_to_handle"], 0.006, "one straight drag down reaches the handle")
        self.assertLess(self.got["miss"], 0.008, f"the hand ended {self.got['miss'] * 1e3:.1f} mm short")

    def test_the_pads_close_on_the_handle(self) -> None:
        # The handle is 20 mm across the jaws: they stop at 20.4 mm.
        self.assertGreater(self.got["closed_width"], 0.018, f"{self.got['closed_width'] * 1e3:.2f} mm")

    def test_over_the_tofu_nothing_is_cut(self) -> None:
        self.assertEqual(1, self.got["over"]["pieces"])
        self.assertEqual(0.0, self.got["over"]["fraction"])

    def test_stopped_short_of_the_board_the_tofu_is_still_one_piece(self) -> None:
        stopped = self.got["stopped"]
        self.assertEqual(1, stopped["pieces"], f"edge {stopped['edge'] * 1e3:.1f} mm over the bench")
        self.assertGreater(stopped["fraction"], 0.5, "the cut ran part way")
        self.assertLess(stopped["fraction"], 1.0)

    def test_down_to_the_board_the_tofu_parts_in_two(self) -> None:
        self.assertEqual(2, self.got["pieces"], f"edge {self.got['edge'] * 1e3:.1f} mm over the bench")
        self.assertEqual(1.0, self.got["fraction"])
        self.assertEqual(1, self.got["cuts"])
        self.assertTrue(self.got["finite"])

    def test_the_knife_feels_the_cut_and_stays_in_the_hand(self) -> None:
        # Tofu's 3 N/m over the edge in the block: about 0.13 N while cutting.
        self.assertGreater(max(self.got["pushes"]), 0.05)
        self.assertLess(self.got["slip"], 0.005, f"knife slip {self.got['slip'] * 1e3:.1f} mm")

    def test_the_scene_steps_inside_an_interactive_budget(self) -> None:
        self.assertLess(self.got["drive_ms"], 60.0, f"{self.got['drive_ms']:.1f} ms per control step cutting")


class TestADullBladeDoesNotPartIt(unittest.TestCase):
    """The same drive with the blade a thousand times duller: the arm stalls on the tofu, one piece."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.got = _drive(DULL_PORT, dull=True)

    def test_nothing_is_cut(self) -> None:
        self.assertEqual(1, self.got["pieces"])
        self.assertEqual(0.0, self.got["fraction"])

    def test_the_tofu_pushes_the_knife_back_short_of_the_board(self) -> None:
        # The threshold, 3000 N/m over the edge, is far past what the pads and the arm push.
        self.assertGreater(max(self.got["pushes"]), 5.0, f"pushes {np.round(self.got['pushes'], 2)}")
        self.assertGreater(self.got["edge"], BOARD_TOP + 0.01, f"edge {self.got['edge'] * 1e3:.1f} mm over the bench")


if __name__ == "__main__":
    unittest.main()
