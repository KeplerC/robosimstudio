# SPDX-License-Identifier: Apache-2.0
"""The granular examples' headline task, done the way their pages ask: the hand sweeps the rice into the tray.

``examples/03_materials/11_granular`` and ``06_tasks/03_granular_scored`` promise
a gesture -- turn the hand down, drag it onto the pusher, close, sweep -- and a
number, the share of the rice in the tray. This drives the arm panel through that
gesture with no browser on it (:mod:`tests.fixtures.grasp`) and holds the two
scripts to the number: the page's readout passes half, and the task document's
goal turns true. Nothing is pinned and no joint target is written, so what
reaches the tray was pushed there by a pusher the fingers closed on.

Two things the sweep also has to show, because each was once the failure: the
rice stays on the bench (rice dropped or squeezed at 60 Hz used to fall through
the bench and the ground by the thousand, which
:data:`~robosimstudio_materials.mpm.physics.PROJECT_GAP` answers; what is left
is a stray grain in some runs, :data:`STRAYS`), and the pusher is inside the
tray's mouth, which a hand that stalls against the heap's push does not reach.
"""

from __future__ import annotations

import math
import time
import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""Two scenes built on a GPU and each stepped through a page for about eight seconds of scene time."""

TOOL_DOWN = (0.0, 1.0, 0.0, 0.0)
"""The YAM's tool pointing down, jaws closing along world y: the turn a person gives the gizmo first."""

SPEED = 0.15
"""The sweep's speed [m/s]: a walking pace, under the ~0.36 m/s at which a tool outruns the MPM's grid at 60 Hz."""

BENCH = np.array([0.6, 0.0, 0.75])
"""``bench_075``'s surface frame in world [m]; the examples' numbers are in it."""

TRAY_LO = BENCH + np.array([-0.14, -0.22, -0.005])
TRAY_HI = BENCH + np.array([0.14, -0.02, 0.03])
"""The tray's inside as the examples state it: 280 by 200 mm, 30 mm walls, 5 mm under the top to the walls' height."""

SWEEP_END = BENCH[1] - 0.16
"""Where the tool centre stops [world y, m]: the pusher's face 130 mm short of the back wall, so the heap has room."""

STRAYS = 3
"""How many of the 16,000 grains may be under the bench top when the sweep ends, or at any step of it.

The grain that gets there is not pushed through the bench: it is thrown. Once
in a while the heap's own squeeze under the pusher shoots a grain sideways
along the top, and it leaves over the bench's edge (first seen below the top
103 mm past the edge, 1.2 s after the hand stopped: ``sweep_strays.py``,
GPU 5), then falls 0.74 m, reaches 3.8 m/s and crosses the ground's collider
band and the 30 mm projection gap in one 60 Hz step, so the ground never sees
it. It happened in two sweeps of thirty (``e2e3.py``'s six and
``sweep_strays.py``'s twenty-four, GPUs 1, 3, 5 and 6, RTX PRO 6000, Newton
1.6.0), one grain each, and the share in the tray was 0.927-0.929 in those
two. Three is above that and below anything the fill could hide."""


def _sweep(session, pusher, points) -> dict:
    """Hover over the pusher, down, **Close**, then the cursor walks toward the tray at :data:`SPEED`."""
    arm = session.arms["right"]
    hz = float(session.env.request.physics.control_hz)
    tick(session, 30)
    arm.cursor_quat = np.asarray(TOOL_DOWN, dtype=np.float64)
    lying = np.asarray(pusher.position, dtype=np.float64)
    go(session, "right", lying + np.array([0.0, 0.0, 0.10]))
    grip = np.array([lying[0], lying[1], BENCH[2] + 0.02])  # fingertips 8 mm over the bench, on the pusher's lower half
    out = {"miss_grip": go(session, "right", grip, speed=0.10)}
    arm.grip_request = 1.0
    tick(session, round(hz))
    out["width"] = float(session.env.gripper("right").width(session.env.state))
    start = np.array(arm.cursor_pos, dtype=np.float64)
    end = np.array([grip[0], SWEEP_END, grip[2]])
    legs = max(1, math.ceil(float(np.linalg.norm(end - start)) / (SPEED / hz)))
    strays = 0
    for k in range(1, legs + 1):
        arm.cursor_pos, arm.dragging = start + (end - start) * (k / legs), k < legs
        tick(session, 1)
        strays = max(strays, int((points()[:, 2] < BENCH[2] - 0.006).sum()))
    arm.cursor_pos, arm.dragging = end, False
    for _ in range(round(1.5 * hz)):
        tick(session, 1)
        strays = max(strays, int((points()[:, 2] < BENCH[2] - 0.006).sum()))
    out["strays"] = strays
    out["pusher"] = np.asarray(pusher.position, dtype=np.float64)
    out["finite"] = bool(session.env.state_is_finite())
    return out


def _fill(points: np.ndarray) -> float:
    from robosimstudio_materials.mpm.measures import fill_fraction

    return fill_fraction(points, TRAY_LO, TRAY_HI)


class _Sweep:
    """Build the example, open its page, sweep; the subclasses say which script and what it must read."""

    EXAMPLE = ""
    PORT = 0

    @classmethod
    def setUpClass(cls) -> None:
        cls.opened, kwargs = _run(cls.EXAMPLE)
        cls.sim = getattr(cls.opened, "sim", cls.opened)
        cls.sim.run(20).sync()
        t = time.perf_counter()
        cls.sim.run(100).sync()
        cls.ms_per_step = 1e3 * (time.perf_counter() - t) / 100
        cls.session = open_session(cls.opened, cls.PORT, kwargs.get("page"))
        cls.rice = cls.sim["rice"]
        cls.result = _sweep(cls.session, cls.sim["pusher"], lambda: np.asarray(cls.rice.particle_q, dtype=np.float64))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_scene_steps_captured_inside_its_budget(self) -> None:
        self.assertTrue(self.sim.pipeline.captured, self.sim.pipeline.capture_error)
        self.assertLess(self.ms_per_step, 40.0, f"{self.ms_per_step:.1f} ms per control step; the docstring says 16.5")

    def test_the_fingers_closed_on_the_pusher(self) -> None:
        self.assertLess(self.result["miss_grip"], 0.008, "the hand did not get down to the pusher")
        self.assertAlmostEqual(0.020, self.result["width"], delta=0.004, msg="the jaws are not on the pusher's faces")

    def test_the_pusher_ended_inside_the_tray(self) -> None:
        self.assertTrue(self.result["finite"])
        pusher = self.result["pusher"]
        self.assertGreater(float(pusher[1]), TRAY_LO[1] + 0.01, f"the pusher stalled at y = {pusher[1]:.3f}")
        self.assertLess(
            abs(float(pusher[2]) - (BENCH[2] + 0.02)), 0.006, "the pusher is no longer standing on the bench"
        )

    def test_the_rice_stays_on_the_bench(self) -> None:
        """Nothing under the top but the few grains the pusher's own margin holds under its base, and a stray.

        The heap names the pusher in ``containers``, so the grid sees it 4 mm
        fatter than it is, 4 mm into the bench included; a grain projected out of
        its underside can sit in that sliver until the pusher moves. Everywhere
        else the top is the floor, and no grain is deeper than the sliver but
        the :data:`STRAYS` the squeeze may throw off the bench's edge.
        """
        points = np.asarray(self.rice.particle_q, dtype=np.float64)
        pusher = self.result["pusher"]
        under = (np.abs(points[:, 0] - pusher[0]) < 0.09) & (np.abs(points[:, 1] - pusher[1]) < 0.015)
        low = points[:, 2] < BENCH[2] - 0.001
        self.assertLessEqual(int((low & ~under).sum()), STRAYS, "rice below the bench top away from the pusher")
        self.assertLessEqual(int((points[:, 2] < BENCH[2] - 0.005).sum()), STRAYS, "rice deeper than the margin")
        self.assertLessEqual(self.result["strays"], STRAYS, "grains left the collider band during the sweep")

    def test_half_the_rice_is_in_the_tray(self) -> None:
        fill = _fill(np.asarray(self.rice.particle_q, dtype=np.float64))
        self.assertGreaterEqual(fill, 0.5, f"{fill:.3f} of the rice is in the tray")


class TestThePageSweepsTheRice(_Sweep, unittest.TestCase):
    """``03_materials/11_granular``: the page's readout passes half once the hand has swept."""

    EXAMPLE = "03_materials/11_granular.py"
    PORT = 9620


class TestTheTaskGoalTurnsTrue(_Sweep, unittest.TestCase):
    """``06_tasks/03_granular_scored``: the same sweep on the task's page, and ``evaluate`` says success."""

    EXAMPLE = "06_tasks/03_granular_scored.py"
    PORT = 9621

    def test_the_goal_is_met(self) -> None:
        env = self.opened
        _, _, info = env.evaluate()
        self.assertTrue(info["success"], f"metrics: {info['metrics']}")
        self.assertGreaterEqual(float(info["metrics"]["fraction"]), 0.5)


if __name__ == "__main__":
    unittest.main()
