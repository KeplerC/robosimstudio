# SPDX-License-Identifier: Apache-2.0
"""The bag example's headline task, done by the robot's hand the way its page asks.

``examples/03_materials/06_bag`` promises a gesture and two numbers. This drives the
example's own arm panel through :mod:`tests.fixtures.grasp` -- the cursor the gizmo
writes, the grip the **Close** button writes, the session's own setpoint law -- and reads
the measures the page shows. The right hand is put down until its open jaws straddle the
top layer's lip, closes, and is dragged up to 80 mm and back 60 mm over the bag. Nothing is
pinned and nothing holds the bag down: the lip comes up because the pads squeeze it. The
mouth opens, the bag stays where it lay, and the two layers never pass through each other.
Then the same gesture on the same bag with self-contact off: the layers lie in each other.
"""

from __future__ import annotations

import importlib.util
import time
import unittest
from unittest import mock

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""One GPU: the example built twice, self-contact on and off, each driven through its page for ~8 s of scene."""

HAS_SUPERDEX = importlib.util.find_spec("superdex") is not None

PORT = 9761
CONTROL_PORT = 9762
Z = 0.728
"""``bench_0728``'s work surface [m]; the example's bag lies on it, at the bench's middle (0.6, 0)."""
GRIP = 0.025
"""The right tool centre's height over the bench to close at [m]: the pads then span the lip's 13-37 mm.
At 20 mm the same drive read 76-90 mm from one run to the next; at 25 mm seven drives read 75-77 mm."""
LIFT, BACK = 0.08, 0.06
"""How high over the bench the right hand is dragged after the close, and how far back over the bag [m].
Lifted to 120 mm the mouth's corners came off the bench and the drives read 36-107 mm."""


def _drive(port: int, *, self_contact: bool) -> dict:
    """Build the example (self-contact as asked), drive its page through the gesture, read the measures."""
    import robosimstudio as rss
    from robosimstudio_materials.bag import layer_gap, mouth_opening

    real = rss.bag
    with mock.patch.object(rss, "bag", lambda **kw: real(**{**kw, "self_contact": self_contact})):
        sim, kwargs = _run("03_materials/06_bag.py")
    bag = sim["bag"]
    points = lambda: np.asarray(bag.particle_q, dtype=np.float64)  # noqa: E731
    out: dict = {"sim": sim, "spec": bag.entity.spec}
    session = open_session(sim, port, kwargs.get("page"))
    try:
        tick(session, 30)
        t = time.perf_counter()
        tick(session, 60)
        out["idle_ms"] = 1e3 * (time.perf_counter() - t) / 60
        rest = points()
        out["rest_gap"] = layer_gap(rest, bag.entity)
        end = rest[bag.entity.bottom[0]].mean(axis=0)
        lip = bag.entity.top[int(bag.entity.spec.cells[0]) + 1 :].ravel()
        p = rest[lip]
        order = np.argsort(p[:, 2])
        x, y = float(np.median(p[:, 0])), float(np.interp(Z + GRIP, p[order, 2], p[order, 1]))
        out["home_to_lip"] = float(np.linalg.norm(session.arms["right"].ik.home_pose[0][:2] - (x, y)))
        t, steps = time.perf_counter(), sim.step_count
        miss = [go(session, "right", (x, y, Z + GRIP), speed=0.06)]
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
        out["end_moved"] = float(np.linalg.norm(points()[bag.entity.bottom[0]].mean(axis=0)[:2] - end[:2]))
        out.update(miss=max(miss), heights=heights, areas=areas, gaps=gaps, finite=sim.state_is_finite())
    finally:
        session.teardown()
    return out


@unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed; nothing steps the bag")
class TestTheHandOpensTheBag(unittest.TestCase):
    """Right hand down onto the lip, Close, up and back: the mouth opens and the layers stay apart."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.got = _drive(PORT, self_contact=True)

    def test_the_scene_steps_inside_an_interactive_budget(self) -> None:
        # Through the page, alone on GPU 1: 11.4 ms at rest and 41.0 ms while the pads hold the cloth.
        self.assertLess(self.got["idle_ms"], 30.0, f"{self.got['idle_ms']:.1f} ms per control step at rest")
        self.assertLess(self.got["drive_ms"], 90.0, f"{self.got['drive_ms']:.1f} ms per control step in contact")

    def test_the_right_hand_starts_over_the_lip_and_gets_where_it_is_dragged(self) -> None:
        self.assertLess(self.got["home_to_lip"], 0.006, "one straight drag down reaches the lip")
        self.assertLess(self.got["miss"], 0.008, f"the hand ended {self.got['miss'] * 1e3:.1f} mm short")

    def test_the_layers_rest_a_contact_radius_apart(self) -> None:
        radius = self.got["spec"].material.particle_radius
        self.assertGreater(self.got["rest_gap"], 0.8 * radius, f"{self.got['rest_gap'] * 1e3:.2f} mm")

    def test_the_mouth_opens(self) -> None:
        # 74.9-76.6 mm and 66-67 cm^2 over seven drives (f_bag.py, GPUs 1 and 3); a bag lying shut, or one
        # whose hand closed over the lip's top and lifted nothing, reads 7.7 mm and 10.2 cm^2.
        self.assertGreater(min(self.got["heights"]), 0.05, f"mouth {np.round(self.got['heights'], 3)} m")
        self.assertGreater(min(self.got["areas"]), 0.004, f"rim area {np.round(self.got['areas'], 4)} m^2")
        self.assertTrue(self.got["finite"])

    def test_the_bag_stays_where_it_lay(self) -> None:
        # Nothing holds it down: its closed end moved 4.9 mm (f_bag.py); lifted to 120 mm it slid 17-38 mm.
        self.assertLess(self.got["end_moved"], 0.015, f"the closed end moved {self.got['end_moved'] * 1e3:.1f} mm")

    def test_the_layers_never_pass_through_each_other(self) -> None:
        # 3.5 mm at worst over the same drives: the pads squeeze the lip, not the layers.
        self.assertGreater(min(self.got["gaps"]), 0.0015, f"least gap {min(self.got['gaps']) * 1e3:.2f} mm")


@unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed; nothing steps the bag")
class TestWithoutSelfContactTheLayersLieInEachOther(unittest.TestCase):
    """The same bag and gesture with the shell's self-contact off: the mouth still opens, its layers not held apart."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.got = _drive(CONTROL_PORT, self_contact=False)

    def test_the_layers_are_not_held_apart(self) -> None:
        # -0.24 mm at rest and -0.29 mm held open, against 3.9 and 3.5 mm with self-contact (two drives).
        self.assertLess(self.got["rest_gap"], 0.001, "the top layer settles into the bottom one")
        self.assertLess(min(self.got["gaps"]), 0.001, f"least gap {min(self.got['gaps']) * 1e3:.2f} mm")


if __name__ == "__main__":
    unittest.main()
