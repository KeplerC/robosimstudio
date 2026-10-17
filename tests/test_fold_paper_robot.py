# SPDX-License-Identifier: Apache-2.0
"""The paper examples' headline tasks, done by the robot's hand the way their pages ask.

``examples/13_paper`` promises two gestures and a number each. This drives the
examples' own arm panels through :mod:`tests.fixtures.grasp` -- the cursor the
gizmo writes, the grip the **Close** button writes, the session's own setpoint
law -- and reads the same ``crease_angle`` the pages show. No handle is put down
and nothing is pinned or seized at the hand, so whatever creases was pushed or
pressed there by the fingers' colliders.

* **The flap** (``01_box_flap``): the left hand, closed and pointing down,
  pushes the standing flap over the box, presses it down near the hinge until
  its free edge is well under the rim, and lifts away. The hinge keeps enough
  bend that the flap comes back to rest shut.
* **The fold** (``00_fold_paper``): both hands closed, the left presses the
  far end of the sheet down, the right is put on the near edge and the page's
  **Edge on the right hand** box is ticked -- the one attachment grasp here,
  the edge row pinned at the hand's frame, since the pads cannot pinch a sheet
  lying flat. The right hand carries the edge up, across and down onto the
  middle third and draws it 15 mm on; the left hand stamps the loop flat at
  three places while the edge is still held; the box is unticked and both
  hands lift. The fold keeps 2.3 to 3.1 rad and the sheet lies in thirds.
"""

from __future__ import annotations

import math
import time
import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""One GPU: each example built once and stepped through its page for about ten seconds of scene time."""

BENCH = np.array([0.6, 0.0, 0.728])
"""``bench_0728``'s surface frame in world [m]; the examples' numbers are in it."""

FLAP_PORT = 9733
SPEED = 0.10
"""The cursor's walk [m/s]; the panel's setpoint law paces the hand under it."""

RIM = BENCH[2] + 0.10
"""The box's rim [world z, m]: its walls are 100 mm tall."""
HINGE = {"crease": {"axis": 2, "at": 0.10 + 0.003 + 0.02 * math.cos(math.radians(15.0))}}
"""The example's crease line: the flap's hinge rows, by height in the bench's frame."""
PUSH_Z = BENCH[2] + 0.162
"""The tool centre's height for the push [m]: its fingertips 145 mm over the bench, on the flap's upper half."""
PRESS = (BENCH[1] - 0.01, BENCH[2] + 0.062)
"""Where the press ends [world y, z, m]: over the box 55 mm in from the hinge, the fingertips 55 mm under the rim."""
HELD = 150
"""Control steps the flap is watched after the hand has gone [2.5 s]."""


def _crease(sim, name: str, line: dict) -> float:
    from robosimstudio.assetlib.semantics.snapshot import Snapshot
    from robosimstudio_materials.paper import crease_angle

    sim[name].entity.update_probe(sim.state, sim.step_count)
    return float(crease_angle(Snapshot(sim.make_context()), name, line)["angle"])


class TestTheHandClosesTheFlap(unittest.TestCase):
    """Close, down beside the flap, across to push it over, down to press it, up: it stays shut."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim, kwargs = _run("13_paper/01_box_flap.py")
        sim = cls.sim
        sim.run(10).sync()
        t = time.perf_counter()
        sim.run(60).sync()
        cls.ms_per_step = 1e3 * (time.perf_counter() - t) / 60
        cls.session = session = open_session(sim, FLAP_PORT, kwargs.get("page"))
        flap = sim["flap"]
        rest = np.asarray(flap.entity.rest["particle_q"], dtype=np.float64)[flap.particles]
        top = np.flatnonzero(rest[:, 2] > rest[:, 2].max() - 0.003)
        cls.edge = lambda: np.asarray(flap.particle_q, dtype=np.float64)[top].mean(axis=0)
        tick(session, 20)
        cls.edge_open = cls.edge()
        cls.home = session.arms["left"].ik.ee_pose(session.env.state)
        session.arms["left"].grip_request = 1.0
        tick(session, 40)
        x, y = float(cls.home[0][0]), float(cls.home[0][1])
        cls.miss = {"down": go(session, "left", (x, y, PUSH_Z), speed=SPEED)}
        cls.miss["push"] = go(session, "left", (x, PRESS[0], PUSH_Z), speed=SPEED)
        cls.pushed = _crease(sim, "flap", HINGE)
        cls.edge_pushed = cls.edge()
        cls.miss["press"] = go(session, "left", (x, PRESS[0], PRESS[1]), speed=SPEED)
        tick(session, 30)
        cls.pressed = _crease(sim, "flap", HINGE)
        go(session, "left", (x, PRESS[0], PUSH_Z), speed=SPEED)
        cls.hand_gone = session.arms["left"].ik.ee_pose(session.env.state)[0]
        cls.after, cls.edges = [], []
        for _ in range(HELD // 15):
            tick(session, 15)
            cls.after.append(_crease(sim, "flap", HINGE))
            cls.edges.append(cls.edge())
        cls.held = list(session.held())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_hand_starts_pointing_down_outside_the_open_flap(self) -> None:
        pos, quat = self.home
        x, y, z, w = quat
        tool_z = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        self.assertLess(float(tool_z[2]), -0.95, f"the tool points down: {tool_z.round(3)}")
        self.assertGreater(float(pos[1] - self.edge_open[1]), 0.02, "the hand is outside the flap's top edge")
        self.assertGreater(float(self.edge_open[2]), RIM + 0.10, "the flap stands open over the rim")

    def test_the_scene_steps_captured_inside_its_budget(self) -> None:
        self.assertTrue(self.sim.pipeline.captured, self.sim.pipeline.capture_error)
        self.assertLess(self.ms_per_step, 80.0, f"{self.ms_per_step:.1f} ms per control step; the docstring says 40")

    def test_the_hand_got_where_it_was_dragged(self) -> None:
        for leg, miss in self.miss.items():
            self.assertLess(miss, 0.01, f"the {leg} ended {miss * 1e3:.1f} mm short")

    def test_the_push_turned_the_flap_over_the_box(self) -> None:
        self.assertLess(float(self.edge_pushed[1]), BENCH[1], "the flap's top edge is over the box")
        self.assertGreater(self.pushed, 0.3, f"the push kept {self.pushed:.2f} rad at the hinge")

    def test_the_press_creased_the_hinge_and_it_holds_after_the_hand_left(self) -> None:
        self.assertGreater(float(self.hand_gone[2] - RIM), 0.05, "the hand is off the flap")
        self.assertEqual([], self.held, "nothing is held by a handle")
        self.assertGreaterEqual(min(self.after), 1.8, f"the hinge kept {np.round(self.after, 3)} rad after release")
        self.assertLess(max(self.after) - min(self.after), 0.02, "and does not relax")

    def test_the_flap_rests_shut(self) -> None:
        edge = self.edges[-1]
        self.assertLess(float(edge[2]), RIM + 0.015, f"its free edge is {1e3 * (edge[2] - RIM):+.0f} mm off the rim")
        self.assertGreater(float(edge[2]), RIM - 0.04, "and not hanging into the box")
        self.assertLess(float(edge[1]), BENCH[1] - 0.04, "lying over the opening")
        self.assertTrue(self.sim.state_is_finite())


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
        self.assertLess(abs(self.far_carried - self.far0), 0.01, "the far end stayed under the left hand")

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


if __name__ == "__main__":
    unittest.main()
