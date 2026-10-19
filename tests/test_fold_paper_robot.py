# SPDX-License-Identifier: Apache-2.0
"""The paper fold example's headline task, done by both of the robot's hands the way its page asks.

``examples/13_paper/00_fold_paper`` promises a gesture and a number. This drives
the example's own arm panels through :mod:`tests.fixtures.grasp` -- the cursor
the gizmo writes, the grip the **Close** button and the **Grip amount** slider
write, the session's own setpoint law -- and reads the same ``crease_angle`` the
page shows. Nothing is pinned to a hand: the right hand's pads pinch the corner of
the sheet where it hangs over the bench's edge, carry the edge up and over while
the hand turns over, and lay it on the middle; the left hand stamps the fold while
the edge is still held; the right hand opens and leaves past the edge. The fold
kept 2.52 to 2.67 rad over 22 drives and the sheet lies in thirds.

The test turns the coupler's contact forces on, so it can also say *how* the edge
was held: the sheet pushes on both fingers while it is carried, and on neither
once the hand has gone -- what a pinch is and an attachment is not.
"""

from __future__ import annotations

import time
import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""One GPU: the example built once and driven through its page by both arm panels."""

FOLD_PORT = 9734
BENCH = np.array([0.6, 0.0, 0.728])
"""``bench_0728``'s surface frame in world [m]; the example's numbers are in it."""
EDGE_Y = BENCH[1] - 0.435
"""The sheet's near edge [world y, m]: 35 mm past the bench's right edge."""
CREASE_Y = EDGE_Y + 0.297 / 3
LETTER = {"crease": {"axis": 1, "at": CREASE_Y}}
"""The fold example's crease line: a third of the way along the sheet from its near edge."""
SHEET_X = BENCH[0] - 0.055
"""The sheet's near-left corner [world x, m]."""
PINCH_X = SHEET_X + 0.035
"""Where the jaws' middle closes along the overhang [world x, m]: 35 mm in from the corner. Further
in, the palm reaches the sheet's end before the jaws do, and pushes it round (11 degrees at 75 mm)."""
PINCH_Y = EDGE_Y + 0.0135
"""Where the jaws' middle closes across the overhang [world y, m]: 13.5 mm in from the edge."""
HOLD = (BENCH[0] + 0.05, EDGE_Y + 0.262, BENCH[2] + 0.003)
"""Where the left hand presses the far end [world, m]: its flat underside 3 mm into the bench."""
STAMPS = (0.60, 0.655, 0.71)
"""The left hand's tool centre along x for the three stamps [world, m], clear of the right hand."""
HELD = 150
"""Control steps the sheet is watched after the hands have gone [2.5 s]."""


def _quat(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _about(axis, degrees):
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    half = np.radians(degrees) / 2
    return np.array([*(axis * np.sin(half)), np.cos(half)])


JAWS = _quat(_quat(_about((0, 1, 0), 10.0), np.array([0.0, 0.70710678, 0.0, 0.70710678])), _about((0, 0, 1), 90.0))
"""The right hand's pinch: the tool along +x tipped 10 degrees down, the jaws closing vertically."""
STAMP = _quat(_about((0, 0, 1), -30.0), np.array([0.0, 0.70710678, 0.0, 0.70710678]))
"""The left hand's stamp: the tool along +x turned 30 degrees toward the fold, which it reaches that way."""


def _measures(sim) -> tuple[float, float]:
    from robosimstudio.assetlib.semantics.snapshot import Snapshot
    from robosimstudio.assetlib.vocabulary.measures import cloth_fold
    from robosimstudio_materials.paper import crease_angle

    sim["paper"].entity.update_probe(sim.state, sim.step_count)
    snapshot = Snapshot(sim.make_context())
    angle = float(crease_angle(snapshot, "paper", LETTER)["angle"])
    return angle, float(cloth_fold(snapshot.ctx, "paper", axis=1)[0])


class _Hand:
    """The right hand's jaws: where their middle is, read off the finger boxes, and the fingers' contact forces."""

    def __init__(self, sim, arm: str) -> None:
        import newton

        self.sim = sim
        index = sim.scene.embodiment.arms[arm]
        model = sim.model
        shape_body = model.shape_body.numpy()
        kinds = model.shape_type.numpy()
        self.fingers = [np.flatnonzero(np.isin(shape_body, side)) for side in index.finger_bodies]
        self.boxes = [s for side in self.fingers for s in side if int(kinds[s]) == int(newton.GeoType.BOX)]

    def middle(self) -> np.ndarray:
        from robosimstudio.core.math import quat_to_mat

        body_q = self.sim.state.body_q.numpy()
        xform = self.sim.model.shape_transform.numpy()
        body = self.sim.model.shape_body.numpy()
        return np.mean(
            [body_q[body[s]][:3] + quat_to_mat(body_q[body[s]][3:7]) @ xform[s][:3] for s in self.boxes], axis=0
        )

    def pushes(self) -> list[float]:
        """The sheet's push on each finger [N]."""
        forces = self.sim.pipeline.program.solver.shape_contact_forces()
        return [float(np.linalg.norm(forces[side].sum(axis=0))) for side in self.fingers]


class TestTheHandsFoldTheSheet(unittest.TestCase):
    """Pinch the overhang, carry the edge over by contact, stamp the fold, let go: it stays folded."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim, kwargs = _run("13_paper/00_fold_paper.py")
        sim = cls.sim
        sim.pipeline.program.solver.enable_contact_forces()  # before the first step captures the graph
        sim.run(10).sync()
        t = time.perf_counter()
        sim.run(60).sync()
        cls.ms_per_step = 1e3 * (time.perf_counter() - t) / 60
        cls.session = session = open_session(sim, FOLD_PORT, kwargs.get("page"))
        points = lambda: np.asarray(sim["paper"].particle_q, dtype=np.float64)  # noqa: E731
        hand = _Hand(sim, "right")
        right = session.arms["right"]
        tick(session, 30)
        rest = points()
        surface = float(np.median(rest[rest[:, 1] < BENCH[1] - 0.40][:, 2]))
        cls.far0 = float(rest[:, 1].max())
        session.arms["left"].grip_request = 1.0
        tick(session, 30)
        cls.miss = {
            "hold": max(
                go(session, "left", (*HOLD[:2], 0.80), speed=0.15), go(session, "left", HOLD, speed=0.06, steps=60)
            )
        }

        def jaws_to(point, speed, steps=120):  # the cursor is the tool centre; put the jaws' middle at ``point``
            target = np.asarray(point) - (hand.middle() - right.ik.ee_pose(sim.state)[0])
            return go(session, "right", target, speed=speed, steps=steps, tolerance=0.008)

        right.cursor_quat = JAWS
        go(session, "right", right.ik.ee_pose(sim.state)[0], speed=0.05)
        jaws_to((SHEET_X - 0.06, PINCH_Y, surface), 0.06)
        cls.miss["in"] = jaws_to((PINCH_X, PINCH_Y, surface), 0.05)
        right.grip_request = 1.0
        tick(session, 60)
        cls.pushes = {"closed": hand.pushes()}
        cls.far_closed = float(points()[:, 1].max())
        # The carry: the edge on an arc about the fold line, the hand turning over with it.
        radius = CREASE_Y - PINCH_Y
        arc = []
        for k in range(1, 13):
            turn = np.pi * k / 12
            right.cursor_quat = _quat(JAWS, _about((0, 0, 1), -np.degrees(turn)))
            # The last waypoint is on the sheet, where the fingers stop: step it briefly.
            arc.append(
                jaws_to(
                    (PINCH_X, CREASE_Y - radius * np.cos(turn), surface + (radius + 0.015) * np.sin(turn)),
                    0.06,
                    120 if k < 12 else 60,
                )
            )
            if k == 6:
                cls.pushes["carried"] = hand.pushes()
                cls.top_of_arc = float(points()[:, 2].max() - BENCH[2])
        cls.miss["arc"] = max(arc[:-1])
        cls.far_carried = float(points()[:, 1].max())
        jaws_to((PINCH_X, CREASE_Y + radius, surface + 0.03), 0.03)
        # 20 mm up, the turned hand's lowest collider is 3 mm over the sheet (fold_contact.py, hand_bottom).
        cls.miss["lay"] = jaws_to((PINCH_X, CREASE_Y + radius, surface + 0.020), 0.02, 60)
        tick(session, 30)
        # Drawn on 30 mm and up 10, which tightens the loop, then up 20 more: held where it was laid,
        # 3 mm over the sheet, the open hand drawn out took the whole sheet 58 to 66 mm with it
        # (fold_counter.py, two drives, GPU 3).
        jaws_to((PINCH_X, CREASE_Y + radius + 0.03, surface + 0.03), 0.03)
        tick(session, 20)
        cls.laid = _measures(sim)
        cls.far_laid = float(points()[:, 1].max())
        y = float(points()[:, 1].min()) + 0.02
        jaws_to((PINCH_X, CREASE_Y + radius + 0.03, surface + 0.05), 0.02)
        session.arms["left"].cursor_quat = STAMP
        left = {"steps": 120, "tolerance": 0.008}
        stamps = [go(session, "left", (*HOLD[:2], BENCH[2] + 0.06), speed=0.08, **left)]
        for x in STAMPS:
            stamps.append(go(session, "left", (x, y, BENCH[2] + 0.06), speed=0.10, **left))
            go(session, "left", (x, y, BENCH[2] + 0.009), speed=0.04, steps=40)  # onto the fold, where it stops
            tick(session, 15)
            stamps.append(go(session, "left", (x, y, BENCH[2] + 0.06), speed=0.06, **left))
        cls.miss["stamps"] = max(stamps)
        cls.stamped = _measures(sim)
        cls.shoved = float(points()[:, 1].max()) - cls.far_laid
        # Let go: up 25 mm, open to half (the lower pad comes back down the 25 mm), out past the edge, up.
        jaws_to(hand.middle() + np.array([0.0, 0.0, 0.025]), 0.02)
        right.grip_request = 0.5
        tick(session, 40)
        before = points()
        jaws_to(hand.middle() + np.array([0.0, 0.04, 0.0]), 0.05)
        cls.dragged = float(np.abs(points() - before).max())
        jaws_to(hand.middle() + np.array([0.0, 0.0, 0.08]), 0.08)
        go(session, "left", (STAMPS[-1], -0.15, BENCH[2] + 0.12), speed=0.08, **left)
        cls.pushes["gone"] = hand.pushes()
        cls.after = []
        for _ in range(HELD // 15):
            tick(session, 15)
            cls.after.append(_measures(sim))
        cls.top = float(points()[:, 2].max() - BENCH[2])
        cls.far_end = float(points()[:, 1].max())
        print(
            f"\n[fold] miss {cls.miss} pushes {cls.pushes} laid {cls.laid} stamped {cls.stamped} "
            f"after {cls.after[-1]} top {cls.top:.4f} dragged {cls.dragged:.4f} shoved {cls.shoved:.4f} "
            f"{cls.ms_per_step:.1f} ms",
            flush=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_scene_steps_captured_inside_its_budget(self) -> None:
        self.assertTrue(self.sim.pipeline.captured, self.sim.pipeline.capture_error)
        self.assertLess(self.ms_per_step, 60.0, f"{self.ms_per_step:.1f} ms per control step")

    def test_the_hands_got_where_they_were_dragged(self) -> None:
        for leg, miss in self.miss.items():
            self.assertLess(miss, 0.015, f"the {leg} ended {miss * 1e3:.1f} mm short")

    def test_the_pads_carry_the_edge_by_contact(self) -> None:
        for moment in ("closed", "carried"):
            left, right = self.pushes[moment]
            self.assertGreater(
                min(left, right), 1.0, f"{moment}: the sheet pushes the fingers {left:.1f} and {right:.1f} N"
            )
        self.assertGreater(self.top_of_arc, 0.08, "half way round the arc the edge is up over the fold")
        self.assertEqual(
            [0.0, 0.0], [round(p, 3) for p in self.pushes["gone"]], "nothing touches the hand once it has gone"
        )

    def test_the_left_hand_kept_the_sheet_from_sliding(self) -> None:
        # Held, the far end moved 4.1 mm over the carry in four drives of this path; with the left hand
        # kept off it, 31.3 to 32.3 mm in four, and the edge was laid with no crease (0 rad), which
        # stamping afterwards took only to 0.47 and 0.53 rad (fold_counter.py, GPU 3). The bound sits
        # between, 11 mm over a held sheet and 16 mm under an unheld one, so an unheld sheet fails it.
        moved = abs(self.far_carried - self.far0)
        self.assertLess(moved, 0.015, f"the far end moved {moved * 1e3:.1f} mm under the left hand")

    def test_the_stamps_shove_the_folded_sheet_but_leave_it_on_the_bench(self) -> None:
        # The left hand's three stamps push the whole folded sheet along the bench, away from the edge
        # it overhung, and the fold rides with it: 42.2 to 55.2 mm in 11 drives of this path over three
        # review rounds (this test and fold3.py, GPUs 1, 3 and 6).
        # The bound keeps the sheet's far end on the bench.
        self.assertGreater(self.shoved, 0.0, f"the far end moved {self.shoved * 1e3:.1f} mm")
        self.assertLess(self.shoved, 0.08, f"the stamps shoved the sheet {self.shoved * 1e3:.1f} mm along the bench")

    def test_the_hand_leaves_without_the_sheet(self) -> None:
        self.assertLess(self.dragged, 0.01, f"drawing the open hand out moved the sheet {self.dragged * 1e3:.1f} mm")

    def test_the_stamped_fold_stays(self) -> None:
        angles = [a for a, _ in self.after]
        self.assertGreaterEqual(min(angles), 2.2, f"the fold kept {np.round(angles, 3)} rad after the hands left")
        self.assertLess(max(angles) - min(angles), 0.05, "and does not relax")
        self.assertGreater(min(angles), self.laid[0] + 0.5, f"the stamps creased it past the {self.laid[0]:.2f} laid")

    def test_the_sheet_lies_folded_in_thirds(self) -> None:
        # A flat fold in thirds spans 0.67 of the sheet; the loop the stamps leave at the fold, and
        # what they push the sheet, make it 0.69 to 0.79 over 22 drives (this test, its take, the
        # take's path in this harness, fold_contact.py; GPUs 3 and 5), where the attachment's carry
        # left 0.6 to 0.75.
        extent = self.after[-1][1]
        self.assertLess(extent, 0.83, f"its span is {extent:.3f} of the flat sheet's")
        self.assertGreater(extent, 0.6, "a third over, not more")
        self.assertLess(self.top, 0.03, "and it lies flat: nothing stands up off the bench")
        self.assertTrue(self.sim.state_is_finite())


if __name__ == "__main__":
    unittest.main()
