# SPDX-License-Identifier: Apache-2.0
"""The paper fold example's headline task, done by both of the robot's hands the way its page asks.

``examples/13_paper/00_fold_paper`` promises a gesture and a number. This drives
the example's own arm panels through :mod:`tests.fixtures.grasp` -- the cursor
the gizmo writes, the grip the **Close** button writes, the session's own
setpoint law -- and reads the same ``crease_angle`` the page shows. Both hands
closed, the left presses the far end of the sheet down, the right is put on the
near edge and the page's **Edge on the right hand** box is ticked -- the one
attachment grasp here, the edge row pinned at the hand's frame, since the pads
cannot pinch a sheet lying flat. The right hand carries the edge up, across and
down onto the middle third and draws it 15 mm on; the left hand stamps the loop
flat at three places while the edge is still held; the box is unticked and both
hands lift. The fold keeps 2.3 to 3.1 rad and the sheet lies in thirds.
"""

from __future__ import annotations

import time
import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""One GPU: the example built once and driven through its page by both arm panels."""

BENCH = np.array([0.6, 0.0, 0.728])
"""``bench_0728``'s surface frame in world [m]; the example's numbers are in it."""
HELD = 150
"""Control steps the sheet is watched after the hands have gone [2.5 s]."""
FOLD_PORT = 9734
LETTER = {"crease": {"axis": 1, "at": -0.20 + 0.297 / 3}}
"""The fold example's crease line: a third of the way along the sheet from its near edge [bench y, m]."""
NEAR_EDGE = BENCH[1] - 0.20
"""The sheet's near edge [world y, m]."""
HOLD = (BENCH[0] + 0.05, BENCH[1] + 0.075, BENCH[2] + 0.005)
"""Where the left hand presses the far end [world, m]: its flat underside 2 mm into the bench, so the sheet is clamped."""
GRAB = (BENCH[0] + 0.05, NEAR_EDGE + 0.005, BENCH[2] + 0.010)
"""Where the right hand is put on the near edge [world, m]: the tool centre over the edge row, its underside on the paper."""
LAND = NEAR_EDGE + 2 * 0.297 / 3
"""Where the right hand carries the edge [world y, m]: twice the flap's length on, so the fold falls at a third."""
STAMPS = (0.56, 0.63, 0.70)
"""The left hand's tool centre along x for the three stamps [world, m]: its 77 mm underside covers the fold's width."""


def _measures(sim) -> tuple[float, float]:
    from robosimstudio.assetlib.semantics.snapshot import Snapshot
    from robosimstudio.assetlib.vocabulary.measures import cloth_fold
    from robosimstudio_materials.paper import crease_angle

    sim["paper"].entity.update_probe(sim.state, sim.step_count)
    snapshot = Snapshot(sim.make_context())
    angle = float(crease_angle(snapshot, "paper", LETTER)["angle"])
    return angle, float(cloth_fold(snapshot.ctx, "paper", axis=1)[0])


class TestTheHandsFoldTheSheet(unittest.TestCase):
    """Hold the far end, carry the near edge over on the attachment, stamp the fold, let go: it stays folded."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim, kwargs = _run("13_paper/00_fold_paper.py")
        sim = cls.sim
        sim.run(10).sync()
        t = time.perf_counter()
        sim.run(60).sync()
        cls.ms_per_step = 1e3 * (time.perf_counter() - t) / 60
        cls.session = session = open_session(sim, FOLD_PORT, kwargs.get("page"))
        inputs = session.server.gui._gui_input_handle_from_uuid.values()
        cls.box = next(h for h in inputs if getattr(h, "label", "") == "Edge on the right hand")
        points = lambda: np.asarray(sim["paper"].particle_q, dtype=np.float64)  # noqa: E731
        tick(session, 30)
        cls.far0 = float(points()[:, 1].max())
        for arm in ("left", "right"):
            session.arms[arm].grip_request = 1.0
        tick(session, 30)
        cls.miss = {"hold": go(session, "left", (*HOLD[:2], 0.80), speed=0.15)}
        cls.miss["hold"] = max(cls.miss["hold"], go(session, "left", HOLD, speed=0.06))
        go(session, "right", (*GRAB[:2], 0.80), speed=0.15)
        cls.miss["grab"] = go(session, "right", GRAB, speed=0.06)
        cls.box.value = True
        tick(session, 5)
        cls.held = [h.label for h in session.deformables.handles.live]
        x, lift = GRAB[0], BENCH[2] + 0.10
        cls.miss["carry"] = max(
            go(session, "right", (x, GRAB[1], lift), speed=0.08),
            go(session, "right", (x, LAND, lift), speed=0.08),
            go(session, "right", (x, LAND, BENCH[2] + 0.014), speed=0.05),
        )
        tick(session, 20)
        cls.miss["draw"] = go(session, "right", (x, LAND + 0.015, BENCH[2] + 0.014), speed=0.03)
        tick(session, 20)
        cls.far_carried = float(points()[:, 1].max())
        cls.laid = _measures(sim)
        fold = float(points()[:, 1].min())
        y = fold + 0.02
        stamps = [go(session, "left", (*HOLD[:2], BENCH[2] + 0.06), speed=0.08)]
        for px in STAMPS:
            stamps.append(go(session, "left", (px, y, BENCH[2] + 0.05), speed=0.10))
            stamps.append(go(session, "left", (px, y, BENCH[2] + 0.009), speed=0.04))
            tick(session, 15)
            stamps.append(go(session, "left", (px, y, BENCH[2] + 0.05), speed=0.06))
        cls.miss["stamps"] = max(stamps)
        cls.box.value = False
        tick(session, 20)
        go(session, "right", (x, LAND + 0.015, lift), speed=0.08)
        go(session, "left", (STAMPS[-1], y, lift), speed=0.08)
        cls.let_go = list(session.held())
        cls.after = []
        for _ in range(HELD // 15):
            tick(session, 15)
            cls.after.append(_measures(sim))
        cls.top = float(points()[:, 2].max())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_scene_steps_captured_inside_its_budget(self) -> None:
        self.assertTrue(self.sim.pipeline.captured, self.sim.pipeline.capture_error)
        self.assertLess(self.ms_per_step, 60.0, f"{self.ms_per_step:.1f} ms per control step; 25 measured on GPU 7")

    def test_the_hands_got_where_they_were_dragged(self) -> None:
        for leg, miss in self.miss.items():
            self.assertLess(miss, 0.012, f"the {leg} ended {miss * 1e3:.1f} mm short")

    def test_the_attachment_took_the_edge_row_and_let_it_go(self) -> None:
        self.assertEqual(["edge (21)"], self.held, "the box pins the 21 particles of the near edge row and no more")
        self.assertEqual([], self.let_go, "unticked, nothing is held")

    def test_the_left_hand_kept_the_sheet_from_sliding(self) -> None:
        # Held, the far end moved 0.28 mm under the carry; with the left hand kept off it, 9.0 mm
        # (fixprobe.py, GPUs 1 and 6). The bound sits between, so an unheld sheet fails it.
        moved = abs(self.far_carried - self.far0)
        self.assertLess(moved, 0.003, f"the far end moved {moved * 1e3:.1f} mm under the left hand")

    def test_the_stamped_fold_stays(self) -> None:
        angles = [a for a, _ in self.after]
        self.assertGreaterEqual(min(angles), 1.5, f"the fold kept {np.round(angles, 3)} rad after the hands left")
        self.assertLess(max(angles) - min(angles), 0.05, "and does not relax")
        self.assertGreater(min(angles), self.laid[0] + 0.3, f"the stamps creased it past the {self.laid[0]:.2f} laid")

    def test_the_sheet_lies_folded_in_thirds(self) -> None:
        extent = self.after[-1][1]
        self.assertLess(extent, 0.75, f"its span is {extent:.3f} of the flat sheet's")
        self.assertGreater(extent, 0.6, "a third over, not more")
        self.assertLess(self.top - BENCH[2], 0.02, "and it lies flat: nothing stands up off the bench")
        self.assertTrue(self.sim.state_is_finite())


class TestTheBoxTakesItsOwnSlot(unittest.TestCase):
    """A sheet already held by a click keeps that grip, on the mouse, through a tick and an untick of the box.

    The box used to pin slot 0 to the right hand and untick with **Release all**:
    with a click grab already in slot 0, the clicked patch rode the right hand,
    the edge landed in slot 1 on no hand, and unticking let go of both.
    """

    def test_a_click_grab_made_first(self) -> None:
        sim, kwargs = _run("13_paper/00_fold_paper.py")
        session = open_session(sim, 9735, kwargs.get("page"))
        try:
            inputs = session.server.gui._gui_input_handle_from_uuid.values()
            box = next(h for h in inputs if getattr(h, "label", "") == "Edge on the right hand")
            grips, paper = session.deformables, sim["paper"]
            q = np.asarray(sim.state.particle_q.numpy(), dtype=np.float64)
            grips.click_at("paper", q[int(paper.particles[np.argmax(q[paper.particles, 1])])])  # the far edge
            tick(session, 1)
            (click,) = [h.index for h in grips.handles.live]
            box.value = True
            tick(session, 1)
            (edge,) = [h.index for h in grips.handles.live if h.index != click]
            self.assertEqual("right", grips.follow_arm[edge], "the edge rides the right hand")
            self.assertEqual("", grips.follow_arm[click], "and the click grab stays on the mouse")
            box.value = False
            tick(session, 1)
            self.assertEqual([click], [h.index for h in grips.handles.live], "unticked, only the edge is let go")
            self.assertEqual("", grips.follow_arm[edge], "and its slot is off the hand")
        finally:
            session.teardown()


if __name__ == "__main__":
    unittest.main()
