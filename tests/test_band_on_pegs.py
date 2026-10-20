# SPDX-License-Identifier: Apache-2.0
"""A rubber band hooked from one peg onto a second by the robot's shut jaws, through the arm panel.

``examples/03_materials/04_band`` says a person can press **Close**, drag the
right hand's shut jaws into a band hanging on the upper peg of a pegboard, down
past the lower peg, in under it, up and back out, and leave the band stretched
round both. This drives that through the arm panel (:mod:`tests.fixtures.grasp`):
the cursor moved as the gizmo's arrows move it, the grip as **Close** asks.
Nothing pins the band: the jaws carry it by contact, and the pegs hold it.

The band is a closed SuperDex rod (``RodSpec.closed``). The control is the same
cord cut: the same nodes, the same length, two ends where the loop was. Driven
the same way it does not stay on the lower peg, so what holds the stretch is the
loop being closed, not friction on the pegs.
"""

from __future__ import annotations

import importlib.util
import time
import unittest
from unittest import mock

import numpy as np

TIER = "medium"
"""Three builds of the two band examples on one GPU, each driven through its page for about 14 s of scene time."""

HAS_SUPERDEX = importlib.util.find_spec("superdex") is not None
PORTS = (9783, 9784, 9787)
BENCH = np.array([0.6, 0.0, 0.75])
"""``bench_075``'s surface frame in world [m]."""
FACE, Y = 0.10, -0.20
UPPER, LOWER = (0.26, 0.08), (0.135, 0.05)
"""Each peg's height over the bench and its length out of the board [m], as the example states them."""
SPEED = 0.05
"""How fast the cursor is dragged [m/s]."""


def _path(bottom_x: float, bottom_z: float) -> list[tuple[str, np.ndarray]]:
    """Where the tool centre goes, in the bench's frame [m]: the gesture the example's docstring asks for."""
    zb = LOWER[0]
    return [
        # The tool centre 5 mm short of the band, the fingertips 12 mm through it and 3 mm short of the
        # lower peg's upturned tip, then the bottom strand down to 30 mm under that peg.
        ("in", np.array([bottom_x - 0.005, Y, bottom_z + 0.02])),
        ("down", np.array([bottom_x - 0.005, Y, zb - 0.03])),
        ("under", np.array([FACE - 0.018, Y, zb - 0.03])),  # toward the board, under it; the tips 1 mm off the board
        ("up", np.array([FACE - 0.018, Y, zb - 0.015])),
        ("out", np.array([bottom_x - 0.06, Y, zb - 0.015])),  # back out along the jaws
    ]


def _drive(port: int, *, cut: bool = False) -> dict:
    from robosimstudio_materials.rod.band import encircles
    from tests.fixtures.examples import _load, _run
    from tests.fixtures.grasp import go, open_session, tick

    module = _load("03_materials/04_band.py")
    real = module.band

    def band(**kwargs):
        import robosimstudio as rss

        spec = real(**kwargs)
        return rss.replace(spec, closed=False, nodes=(*spec.nodes, spec.nodes[0])) if cut else spec

    with mock.patch.object(module, "band", band):
        sim, kwargs = _run("03_materials/04_band.py")
    loop = sim["band"]
    rest = float(loop.entity.spec.length)
    session = open_session(sim, port, kwargs.get("page"))
    arm = session.arms["right"]

    def points() -> np.ndarray:
        return np.asarray(loop.particle_q, dtype=np.float64)

    def on(peg) -> bool:
        z, length = peg
        return encircles(points(), BENCH + np.array([FACE, Y, z]), (-1.0, 0.0, 0.0), length)

    try:
        tick(session, 90)  # the band hangs
        hung = {"upper": on(UPPER), "lower": on(LOWER)}
        bottom = points()[np.argmin(points()[:, 2])] - BENCH
        arm.grip_request = 1.0
        tick(session, 45)
        reached = {}
        for tag, target in _path(float(bottom[0]), float(bottom[2])):
            go(session, "right", BENCH + target, speed=SPEED)
            reached[tag] = float(np.linalg.norm(arm.ik.ee_pose(sim.state)[0] - BENCH - target))
        tick(session, 120)
        length = float(np.linalg.norm(np.diff(points(), axis=0), axis=1).sum())
        t = time.perf_counter()
        tick(session, 60)  # stretched round both pegs, the costliest the page gets
        ms = 1e3 * (time.perf_counter() - t) / 60
        return {
            "hung": hung,
            "upper": on(UPPER),
            "lower": on(LOWER),
            "stretch": length / rest,
            "settled_stretch": float(np.linalg.norm(np.diff(points(), axis=0), axis=1).sum()) / rest,
            "reached": reached,
            "ms": ms,
            "finite": bool(sim.state_is_finite()),
            "sim": sim,
        }
    finally:
        session.teardown()


@unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed; nothing steps the band")
class TestTheHandHooksTheBandOnBothPegs(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.run_ = _drive(PORTS[0])

    def test_the_band_starts_on_the_upper_peg_alone(self) -> None:
        self.assertEqual({"upper": True, "lower": False}, self.run_["hung"])

    def test_the_band_is_a_closed_superdex_rod(self) -> None:
        sim = self.run_["sim"]
        self.assertEqual("superdex", sim.request.physics.backend_for("rod").solver)
        self.assertTrue(sim["band"].entity.spec.closed)

    def test_it_ends_round_both_pegs(self) -> None:
        self.assertTrue(self.run_["finite"])
        self.assertTrue(self.run_["upper"], "the band left the upper peg")
        self.assertTrue(self.run_["lower"], "the band is not round the lower peg")

    def test_the_pegs_hold_it_stretched(self) -> None:
        """Round both pegs it is longer than at rest, and stays so with the hand gone."""
        self.assertGreater(self.run_["settled_stretch"], 1.10, f"stretch {self.run_['settled_stretch']:.3f}")

    def test_the_hand_went_where_the_gesture_asks(self) -> None:
        for tag, miss in self.run_["reached"].items():
            self.assertLess(miss, 0.01, f"the hand ended {miss * 1e3:.1f} mm from the {tag!r} target")

    def test_a_control_step_fits_a_live_page(self) -> None:
        self.assertLess(self.run_["ms"], 60.0, f"{self.run_['ms']:.1f} ms a control step")


@unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed; nothing steps the band")
class TestTheScoredTaskTurnsTrue(unittest.TestCase):
    """``06_tasks/07_band_scored``: the same gesture on the task's page, from the YAM's own home, and the goal.

    The task's pegs are the example's cylinder posts with a ``rod`` feature (their two ends), so
    ``encircles`` holds the band to each peg's length, not to the line through it.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from tests.fixtures.examples import _run
        from tests.fixtures.grasp import go, open_session, tick

        cls.env, kwargs = _run("06_tasks/07_band_scored.py")
        sim = getattr(cls.env, "sim", cls.env)
        session = open_session(cls.env, PORTS[2], kwargs.get("page"))
        arm = session.arms["right"]
        try:
            tick(session, 90)
            _, _, cls.before = cls.env.evaluate()
            loop = np.asarray(sim["band"].particle_q, dtype=np.float64)
            bottom = loop[np.argmin(loop[:, 2])] - BENCH
            arm.cursor_quat = np.array([0.0, 0.70710678, 0.0, 0.70710678])  # level, pointing at the board
            go(session, "right", BENCH + np.array([bottom[0] - 0.05, Y, bottom[2] + 0.02]), speed=0.1)
            arm.grip_request = 1.0
            tick(session, 45)
            for _, target in _path(float(bottom[0]), float(bottom[2])):
                go(session, "right", BENCH + target, speed=SPEED)
            tick(session, 120)
            _, _, cls.after = cls.env.evaluate()
        finally:
            session.teardown()

    def test_the_goal_starts_half_met(self) -> None:
        self.assertFalse(self.before["success"])
        self.assertTrue(self.before["metrics"]["goal.0"], "the band does not start round the upper peg")

    def test_the_goal_is_met(self) -> None:
        self.assertTrue(self.after["success"], f"metrics: {self.after['metrics']}")


@unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed; nothing steps the band")
class TestACutBandDoesNotStay(unittest.TestCase):
    """The control: the same cord with two ends, driven the same way, is not left round the lower peg."""

    def test_the_cut_band_is_not_left_stretched_round_both(self) -> None:
        run = _drive(PORTS[1], cut=True)
        self.assertFalse(run["lower"] and run["settled_stretch"] > 1.10, f"stretch {run['settled_stretch']:.3f}")


if __name__ == "__main__":
    unittest.main()
