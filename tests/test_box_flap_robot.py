# SPDX-License-Identifier: Apache-2.0
"""The box flap example's headline task, done by the robot's hand the way its page asks.

``tests/scenes/01_box_flap`` promises a gesture and a number. This drives the
example's own arm panel through :mod:`tests.fixtures.grasp` -- the cursor the
gizmo writes, the grip the **Close** button writes, the session's own setpoint
law -- and reads the same ``crease_angle`` the page shows. No handle is put down
and nothing is pinned or seized at the hand, so whatever creases was pushed or
pressed there by the fingers' colliders: the left hand, closed and pointing down,
pushes the standing flap over the box, presses it down near the hinge until its
free edge is well under the rim, and lifts away. The hinge keeps enough bend that
the flap comes back to rest shut.
"""

from __future__ import annotations

import math
import time
import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""One GPU: the example built once and driven through its page by the arm panel."""

BENCH = np.array([0.6, 0.0, 0.728])
"""``bench_0728``'s surface frame in world [m]; the example's numbers are in it."""

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
        cls.sim, kwargs = _run("tests/scenes/01_box_flap.py")
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


if __name__ == "__main__":
    unittest.main()
