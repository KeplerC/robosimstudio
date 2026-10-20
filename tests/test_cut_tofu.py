# SPDX-License-Identifier: Apache-2.0
"""The cutting example's headline task, done by the robot's hand the way its page asks.

``examples/03_materials/10_cutting`` promises a gesture and a number. This drives the example's own
arm panel through :mod:`tests.fixtures.grasp` -- the cursor the gizmo writes, the grip the **Close**
button writes -- and reads what the page's **Tofu** folder reads. The right hand is put down on the
knife's handle, closes, lifts the knife 70 mm out of its rest, carries it over the tofu and pushes it
down in 4 mm steps until the edge meets the board. Nothing pins the knife to the hand: the pads
squeeze its handle. Two controls ride on the same drive: over the tofu, and held 15 mm higher on the
way down, the tofu is still one piece. The drive's first and last steps, replayed the way the page's
**Export video** replays them, draw the block each of them drew. A second drive does the same with
**Dull blade** ticked, a blade a thousand times duller: the arm stalls with the edge a few
millimetres into the tofu, read off the knife's pose, and it does not part. A third build opens the
whole page, every panel, and refreshes each of them.
"""

from __future__ import annotations

import time
import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""One GPU: the example built three times, twice driven through its page at about 11 ms a control step."""

PORT = 9841
DULL_PORT = 9842
ALL_PORT = 9844
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


def cut(session, tofu, knife, *, pause_over: float | None = None) -> dict:
    """Down onto the handle, **Close**, up, across, then down in 4 mm steps until the edge meets the board.

    With ``pause_over`` the hand holds that much higher on the way down for 100 steps, and what the tofu
    reads there is ``out["stopped"]``. ``pushes`` is the push on the edge every control step from the
    first step down, what the page's readout shows [N]; ``push_held`` its mean over the 100 steps held
    at the bottom. ``slip`` is how far the knife has moved from where it sat in the hand when the pads
    closed [m], and ``tilt`` how far its edge leans from level [deg].
    """
    from robosimstudio_materials.cutting.entity import knife_edge

    arm = session.arms["right"]

    def hand() -> np.ndarray:
        return arm.ik.ee_pose(session.env.state)[0]

    def where() -> np.ndarray:
        return np.asarray(knife.position, dtype=np.float64)

    def readings() -> dict:
        return {"pieces": tofu.pieces(), "fraction": tofu.cut_fraction(), "edge_depth": tofu.edge_depth()}

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
    out["over"] = readings()
    pushes: list[float] = []
    session.each_step(lambda: pushes.append(tofu.push))
    cursor, bottom = over.copy(), WORK[2] + BOARD_TOP + EDGE - 0.005
    t, steps = time.perf_counter(), session.env.step_count
    for z in np.arange(over[2], bottom - 1.0e-9, -0.004):
        cursor[2] = z
        go(session, "right", cursor.copy(), speed=0.02, steps=20)
        if pause_over is not None and "stopped" not in out and z - 0.004 < bottom + pause_over - 1.0e-9:
            go(session, "right", cursor.copy(), steps=100)
            out["stopped"] = readings()
    go(session, "right", cursor.copy(), steps=100)
    out["drive_ms"] = 1e3 * (time.perf_counter() - t) / max(session.env.step_count - steps, 1)
    e0, e1, _ = knife_edge(tofu.spec.knife, session.env.state.body_q.numpy()[tofu.knife_body])
    out.update(
        readings(),
        depth=tofu.cut_depth(),
        cuts=tofu.cuts(),
        pushes=list(pushes),
        push_held=float(np.mean(pushes[-100:])),
        tilt=float(np.degrees(np.arcsin(abs(e1[2] - e0[2]) / np.linalg.norm(e1 - e0)))),
        slip=float(np.linalg.norm(where() - hand() - offset)),
        finite=bool(session.env.state_is_finite()),
    )
    return out


def replay(sim, tofu, frames: list) -> dict:
    """The first and last steps' particles written into a fresh state, as **Export video** replays them.

    ``frames`` holds each step's ``particle_q``, copied as the page's video buffer copies it, beside the
    block's vertices that step drew. ``first`` and ``last`` pair what the handle draws at the replayed
    corners with what the step drew; ``nowhere`` is how many vertices it draws at corners no step wrote.
    """
    from robosimstudio.core.render.offline import StateReplay
    from robosimstudio.core.render.particles import surface_mesh

    handle = sim.handles.deformables[tofu.name]
    state = StateReplay(sim.model)
    out: dict = {}
    for name, (q, drawn) in (("first", frames[0]), ("last", frames[-1])):
        at = state.write({"particle_q": q}).particle_q.numpy()[handle.particles]
        out[name] = (surface_mesh(handle.surface, at)[0], drawn)
    out["nowhere"] = len(surface_mesh(handle.surface, at + 1.0e-3)[0])
    return out


def _drive(port: int, *, dull: bool) -> dict:
    """Build the example and drive its page through the cut, holding 15 mm over the board on the way down."""
    sim, kwargs = _run("03_materials/10_cutting.py")
    tofu, knife = sim["tofu"].entity, sim["knife"]
    session = open_session(sim, port, kwargs.get("page"))
    frames: list = []
    session.each_step(
        lambda: frames.append((np.array(session.env.state.particle_q.numpy(), dtype=np.float32), tofu._surface[0]))
    )
    try:
        if dull:
            tofu.set_sharpness(0.001)  # what the page's Dull blade box writes
        tick(session, 30)
        got = cut(session, tofu, knife, pause_over=0.015)
        got["tick_ms"] = tofu.timing.get("total_ms", 0.0)
        got["replay"] = replay(sim, tofu, frames)
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
        self.assertEqual(1, stopped["pieces"], f"edge {stopped['edge_depth'] * 1e3:.1f} mm into the tofu")
        self.assertGreater(stopped["fraction"], 0.5, "the cut ran part way")
        self.assertLess(stopped["fraction"], 1.0)

    def test_down_to_the_board_the_tofu_parts_in_two(self) -> None:
        self.assertEqual(2, self.got["pieces"], f"edge {self.got['edge_depth'] * 1e3:.1f} mm into the tofu")
        self.assertEqual(1.0, self.got["fraction"])
        self.assertEqual(1, self.got["cuts"])
        self.assertTrue(self.got["finite"])

    def test_the_knife_feels_the_cut_and_stays_in_the_hand(self) -> None:
        # Tofu's 3 N/m over the edge in the block: about 0.13 N while cutting.
        self.assertGreater(max(self.got["pushes"]), 0.05)
        self.assertLess(self.got["slip"], 0.005, f"knife slip {self.got['slip'] * 1e3:.1f} mm")

    def test_the_scene_steps_inside_an_interactive_budget(self) -> None:
        self.assertLess(self.got["drive_ms"], 60.0, f"{self.got['drive_ms']:.1f} ms per control step cutting")

    def test_a_replay_draws_the_block_each_step_drew(self) -> None:
        """Whole at the first step, before the knife went in, and in two at the last; nothing elsewhere."""
        (first, drawn_first), (last, drawn_last) = self.got["replay"]["first"], self.got["replay"]["last"]
        self.assertLess(len(first), len(last), "the first step replays with the cut's faces")
        np.testing.assert_allclose(first, drawn_first, atol=1.0e-6)
        np.testing.assert_allclose(last, drawn_last, atol=1.0e-6)
        self.assertEqual(0, self.got["replay"]["nowhere"], "corners no step wrote draw another step's block")


class TestADullBladeDoesNotPartIt(unittest.TestCase):
    """The same drive with the blade a thousand times duller: the arm stalls on the tofu, one piece."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.got = _drive(DULL_PORT, dull=True)

    def test_nothing_is_cut(self) -> None:
        self.assertEqual(1, self.got["pieces"])
        self.assertEqual(0.0, self.got["fraction"])

    def test_the_tofu_pushes_the_knife_back_short_of_the_board(self) -> None:
        # The threshold, 3000 N/m over the edge, is far past what the pads and the arm push: the edge
        # stops 7-11 mm into the 40 mm block, the knife tipped 14-17 degrees in the pads (13 drives).
        got = self.got
        self.assertGreater(max(got["pushes"]), 5.0, f"pushes peak {max(got['pushes']):.2f} N")
        depth = f"edge {got['edge_depth'] * 1e3:.1f} mm into the tofu, tilted {got['tilt']:.1f} deg"
        self.assertGreater(got["edge_depth"], 0.002, depth)
        self.assertLess(got["edge_depth"], 0.015, depth)


class TestTheWholePageOpens(unittest.TestCase):
    """The page with every panel, ``panels="all"``: the launcher's default, where ``rss.view``'s is ``"drive"``.

    The cloth panel listed the tofu, which owns eight particles -- the corners its cage writes back for
    the page to draw the cut by -- and called ``observe`` on it: ``AttributeError`` on the first refresh.
    The tofu carries no particle probe, so it is no deformable the panels serve, and the page has none.
    """

    def test_it_refreshes_without_a_deformable_fold(self) -> None:
        from robosimstudio_studio.session import StudioConfig, StudioSession, panel_key

        sim, kwargs = _run("03_materials/10_cutting.py")
        session = StudioSession(sim, config=StudioConfig(port=ALL_PORT, panels="all", speed="max", live=False))
        try:
            kwargs["page"](session)
            session.begin_run()
            tick(session, 20)
            for panel in session.panels:  # what each refresh does, on every panel, whatever the draw clock says
                panel.refresh()
            keys = {panel_key(p) for p in session.panels}
            self.assertEqual(set(), keys & {"cloth", "tune", "readings"})
            self.assertEqual({}, sim.deformables(), "nothing on the bench carries a particle probe")
        finally:
            session.teardown()


if __name__ == "__main__":
    unittest.main()
