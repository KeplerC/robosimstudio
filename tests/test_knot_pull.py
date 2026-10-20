# SPDX-License-Identifier: Apache-2.0
"""A loose overhand knot pulled tight by both of the robot's hands, through the arm panels.

``examples/03_materials/03_knot`` says a person can drag each hand down
onto a tail of a knotted rope, slide its **Grip amount** to close on it, drag
both hands up and then out, and that the knot draws up and stays a knot. This
drives exactly that through the two arm panels (:mod:`tests.fixtures.grasp`),
one hand at a time as a mouse can, the slider dragged rather than jumped.
Nothing pins the rope: the pads hold it by friction.

A knot is a rope crossing itself and held apart where it crosses by its own
contact (``RodSpec.self_contact``). The control is the same drive on the same
rope with self-contact off: the strands pass through each other, the crossings
go, and the rope comes out straight.
"""

from __future__ import annotations

import importlib.util
import time
import unittest
from unittest import mock

import numpy as np

TIER = "medium"
"""Two builds of the example on one GPU, each driven through its page for about 13 s of scene time."""

HAS_SUPERDEX = importlib.util.find_spec("superdex") is not None
PORTS = (9785, 9786)
BENCH = np.array([0.6, 0.0, 0.75])
RADIUS = 0.006
GRIP_AMOUNT = 0.90
"""What the example asks the slider for on the 12 mm rope."""
INSET = 0.04
"""Where each hand closes, in from its tail's end [m]: the example's homes are over these points."""
ABOVE_AXIS = 0.010
"""Where the tool centre closes over the rope's axis [m], as on the cable: the pads then span its equator."""
LIFT, PULL, SPEED = 0.12, 0.085, 0.04
"""Each hand up [m], then out along the rope [m], at this cursor speed [m/s]."""
TOLERANCE = RADIUS * 2 / 3
"""How far one strand's surface may be inside another's and still count as touching [m]: a third of the
diameter, the give of the skin's samples. Passing through puts one axis on the other: a whole diameter."""


def _drive(port: int, *, self_contact: bool = True) -> dict:
    import robosimstudio as rss
    from robosimstudio_materials.rod.knot import crossings, knot_length, self_clearance
    from tests.fixtures.examples import _run
    from tests.fixtures.grasp import go, open_session, tick

    real = rss.rod

    def rod(**kwargs):
        return real(**{**kwargs, "self_contact": self_contact})

    with mock.patch.object(rss, "rod", rod):
        sim, kwargs = _run("03_materials/03_knot.py")
    rope = sim["rope"]
    session = open_session(sim, port, kwargs.get("page"))
    arms = session.arms

    def points() -> np.ndarray:
        return np.asarray(rope.particle_q, dtype=np.float64)

    try:
        tick(session, 60)  # the knot settles
        rest = points()
        before = {"crossings": crossings(rest), "knot_length": knot_length(rest)}
        ends = {"right": rest[0] + np.array([0.0, INSET, 0.0]), "left": rest[-1] - np.array([0.0, INSET, 0.0])}
        for arm, at in ends.items():
            go(session, arm, at + np.array([0.0, 0.0, ABOVE_AXIS]), speed=0.05)
        for k in range(1, 49):  # both sliders to 0.90 over 0.8 s
            for arm in arms:
                arms[arm].grip_request = GRIP_AMOUNT * k / 48
            tick(session, 1)
        tick(session, 30)
        widths = {arm: float(sim.gripper(arm).width(sim.state)) for arm in arms}
        offsets = {arm: points()[0 if arm == "right" else -1] - arms[arm].ik.ee_pose(sim.state)[0] for arm in arms}
        worst = np.inf
        for leg in ("up", "out"):
            for arm in ("right", "left"):
                # The cursor walked at SPEED as a drag of the gizmo's arrow, the strands' clearance read each step.
                panel = arms[arm]
                start = np.array(panel.cursor_pos, dtype=np.float64)
                step = (
                    np.array([0.0, 0.0, LIFT])
                    if leg == "up"
                    else np.array([0.0, PULL if arm == "left" else -PULL, 0.0])
                )
                count = int(np.ceil(np.linalg.norm(step) / SPEED * 60))
                for k in range(1, count + 1):
                    panel.cursor_pos, panel.dragging = start + step * k / count, k < count
                    tick(session, 1)
                    worst = min(worst, self_clearance(points(), RADIUS))
                go(session, arm, start + step)
        tick(session, 120)
        held = points()
        t = time.perf_counter()
        tick(session, 60)  # pulled tight, the costliest the page gets
        ms = 1e3 * (time.perf_counter() - t) / 60
        slid = max(
            float(
                np.linalg.norm(
                    points()[0 if arm == "right" else -1] - arms[arm].ik.ee_pose(sim.state)[0] - offsets[arm]
                )
            )
            for arm in arms
        )
        return {
            "before": before,
            "crossings": crossings(held),
            "knot_length": knot_length(held),
            "clearance": min(worst, self_clearance(held, RADIUS)),
            "span": float(np.linalg.norm(held[-1] - held[0])),
            "widths": widths,
            "slid": slid,
            "ms": ms,
            "finite": bool(sim.state_is_finite()),
            "sim": sim,
        }
    finally:
        session.teardown()


@unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed; nothing steps the rope")
class TestBothHandsPullTheKnotTight(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.run_ = _drive(PORTS[0])

    def test_it_starts_a_loose_overhand_knot(self) -> None:
        self.assertEqual(3, self.run_["before"]["crossings"])

    def test_the_rope_touches_itself_on_superdex(self) -> None:
        sim = self.run_["sim"]
        self.assertEqual("superdex", sim.request.physics.backend_for("rod").solver)
        self.assertTrue(sim["rope"].entity.spec.self_contact)

    def test_the_pads_close_on_the_rope_and_keep_it(self) -> None:
        for arm, width in self.run_["widths"].items():
            self.assertTrue(0.004 < width < 0.012, f"the {arm} jaws closed to {width * 1e3:.1f} mm on a 12 mm rope")
        self.assertLess(self.run_["slid"], 0.01, f"a tail slid {self.run_['slid'] * 1e3:.1f} mm in its hand")

    def test_it_stays_a_knot(self) -> None:
        self.assertTrue(self.run_["finite"])
        self.assertEqual(3, self.run_["crossings"], "the knot came undone")
        self.assertGreater(
            self.run_["clearance"], -TOLERANCE, f"a strand went {-self.run_['clearance'] * 1e3:.1f} mm in"
        )

    def test_the_knot_draws_up(self) -> None:
        before, after = self.run_["before"]["knot_length"], self.run_["knot_length"]
        self.assertLess(after, 0.95 * before, f"the knot holds {after * 1e3:.0f} mm of rope, {before * 1e3:.0f} before")

    def test_a_control_step_fits_a_live_page(self) -> None:
        """Pulled tight, 45 ms a step alone on the machine (GPU 6); over 60 with a second SuperDex test
        stepping beside it, since each engine takes every core. The bound leaves room for a sharded run."""
        self.assertLess(self.run_["ms"], 90.0, f"{self.run_['ms']:.1f} ms a control step")


@unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed; nothing steps the rope")
class TestWithoutSelfContactItPullsThroughItself(unittest.TestCase):
    """The control: the same rope and drive with self-contact off comes undone, a strand through a strand."""

    def test_the_rope_passes_through_itself_and_the_knot_goes(self) -> None:
        run = _drive(PORTS[1], self_contact=False)
        self.assertLess(run["crossings"], 3, "the knot held without self-contact")
        self.assertLess(run["clearance"], -TOLERANCE, f"closest strands {run['clearance'] * 1e3:.1f} mm")


if __name__ == "__main__":
    unittest.main()
