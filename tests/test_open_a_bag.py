# SPDX-License-Identifier: Apache-2.0
"""The bag example's headline task, done by the robot's hands the way its page asks.

``examples/16_bags/00_open_a_bag`` promises a gesture and two numbers. This drives the
example's own arm panels through :mod:`tests.fixtures.grasp` -- the cursor the gizmo
writes, the grip the **Close** button writes, the session's own setpoint law -- and reads
the measures the page shows. The left hand, closed, is put down on the bag's closed end;
the right hand is put down until its open jaws straddle the top layer's lip, closes,
and is dragged up 120 mm and back 50 mm over the bag. Nothing is pinned: the lip comes up
because the pads hold it. The mouth opens, and the two layers never pass through each
other. Then the same gesture on the same bag with self-contact off: the layers do.
"""

from __future__ import annotations

import time
import unittest
from unittest import mock

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""One GPU: the example built twice, self-contact on and off, each driven through its page for ~10 s of scene."""

PORT = 9761
CONTROL_PORT = 9762
Z = 0.728
"""``bench_0728``'s work surface [m]; the example's bag lies on it, at the bench's middle (0.6, 0)."""
PRESS = (0.6, 0.08, Z + 0.019)
"""Where the left hand's tool centre goes [world, m]: its closed fingertips on the top layer over the closed end."""
GRIP = 0.02
"""The right tool centre's height over the bench to close at [m]: the pads then span the lip's 13-37 mm."""
LIFT, BACK = 0.12, 0.05
"""How high the right hand is dragged after the close, and how far back over the bag [m]."""


def _drive(port: int, *, self_contact: bool) -> dict:
    """Build the example (self-contact as asked), drive its page through the gesture, read the measures."""
    import robosimstudio as rss
    from robosimstudio_materials.bag import layer_gap, mouth_opening

    real = rss.bag
    with mock.patch.object(rss, "bag", lambda **kw: real(**{**kw, "self_contact": self_contact})):
        sim, kwargs = _run("16_bags/00_open_a_bag.py")
    bag = sim["bag"]
    points = lambda: np.asarray(bag.particle_q, dtype=np.float64)  # noqa: E731
    out: dict = {"sim": sim, "spec": bag.entity.spec}
    sim.run(10).sync()
    t = time.perf_counter()
    sim.run(60).sync()
    out["idle_ms"] = 1e3 * (time.perf_counter() - t) / 60
    out["rest_gap"] = layer_gap(points(), bag.entity)
    session = open_session(sim, port, kwargs.get("page"))
    try:
        lip = bag.entity.top[int(bag.entity.spec.cells[0]) + 1 :].ravel()
        p = points()[lip]
        order = np.argsort(p[:, 2])
        x, y = float(np.median(p[:, 0])), float(np.interp(Z + GRIP, p[order, 2], p[order, 1]))
        out["home_to_lip"] = float(np.linalg.norm(session.arms["right"].ik.home_pose[0][:2] - (x, y)))
        session.arms["left"].grip_request = 1.0
        tick(session, 30)
        t, steps = time.perf_counter(), sim.step_count
        miss = [go(session, "left", PRESS, speed=0.06), go(session, "right", (x, y, Z + GRIP), speed=0.06)]
        session.arms["right"].grip_request = 1.0
        gaps = []
        for _ in range(4):
            tick(session, 15)
            gaps.append(layer_gap(points(), bag.entity))
        miss.append(go(session, "right", (x, y, Z + LIFT), speed=0.06))
        gaps.append(layer_gap(points(), bag.entity))
        miss.append(go(session, "right", (x, y + BACK, Z + LIFT), speed=0.04))
        heights, areas = [], []
        for _ in range(6):
            tick(session, 15)
            q = points()
            mouth = mouth_opening(q, bag.entity)
            heights.append(mouth["height"])
            areas.append(mouth["area"])
            gaps.append(layer_gap(q, bag.entity))
        out["drive_ms"] = 1e3 * (time.perf_counter() - t) / max(sim.step_count - steps, 1)
        out.update(miss=max(miss), heights=heights, areas=areas, gaps=gaps, finite=sim.state_is_finite())
    finally:
        session.teardown()
    return out


class TestTheHandsOpenTheBag(unittest.TestCase):
    """Left hand on the closed end, right hand on the lip, Close, up and back: the mouth opens and the layers stay apart."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.got = _drive(PORT, self_contact=True)

    def test_the_scene_steps_inside_an_interactive_budget(self) -> None:
        # 8.6-11 ms idle and 39-52 ms a control step while the pads press the cloth, on GPUs 3 and 5.
        self.assertLess(self.got["idle_ms"], 30.0, f"{self.got['idle_ms']:.1f} ms per control step at rest")
        self.assertLess(self.got["drive_ms"], 90.0, f"{self.got['drive_ms']:.1f} ms per control step in contact")

    def test_the_right_hand_starts_over_the_lip_and_gets_where_it_is_dragged(self) -> None:
        self.assertLess(self.got["home_to_lip"], 0.006, "one straight drag down reaches the lip")
        self.assertLess(self.got["miss"], 0.008, f"a hand ended {self.got['miss'] * 1e3:.1f} mm short")

    def test_the_layers_rest_a_contact_radius_apart(self) -> None:
        radius = self.got["spec"].material.particle_radius
        self.assertGreater(self.got["rest_gap"], 0.8 * radius, f"{self.got['rest_gap'] * 1e3:.2f} mm")

    def test_the_mouth_opens(self) -> None:
        # 54-91 mm and 59-86 cm^2 over eight drives (grip heights 20-30 mm, the lip 4 mm either side),
        # 33-74 mm in four filmed takes; a bag lying shut reads 7.7 mm and 10.2 cm^2.
        self.assertGreater(min(self.got["heights"]), 0.03, f"mouth {np.round(self.got['heights'], 3)} m")
        self.assertGreater(min(self.got["areas"]), 0.003, f"rim area {np.round(self.got['areas'], 4)} m^2")
        self.assertTrue(self.got["finite"])

    def test_the_layers_never_pass_through_each_other(self) -> None:
        # 2.9-3.7 mm at worst over the same drives: the pads squeeze the lip, not the layers.
        self.assertGreater(min(self.got["gaps"]), 0.0015, f"least gap {min(self.got['gaps']) * 1e3:.2f} mm")


class TestWithoutSelfContactTheLayersPass(unittest.TestCase):
    """The same bag and gesture with the shell's self-contact off: the mouth still opens, through itself."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.got = _drive(CONTROL_PORT, self_contact=False)

    def test_the_layers_cross(self) -> None:
        # -0.24 mm at rest and -5.2 mm held open, against +3.6 with self-contact (two drives).
        self.assertLess(self.got["rest_gap"], 0.001, "the top layer settles into the bottom one")
        self.assertLess(min(self.got["gaps"]), -0.002, f"least gap {min(self.got['gaps']) * 1e3:.2f} mm")


if __name__ == "__main__":
    unittest.main()
