# SPDX-License-Identifier: Apache-2.0
"""The stick-a-label example's task, done by the robot's hand the way its page asks.

``examples/03_materials/09_adhesion`` promises a gesture and a number. This drives the example's own
arm panel through :mod:`tests.fixtures.grasp` -- the cursor the gizmo writes, the grip the **Close**
button and the **Grip amount** slider write, the session's own setpoint law -- and reads what the
page's **Label** folder reads. The right hand takes the pull tab of a loose label off the bench,
carries it over the box and lays it down, then rubs it with its shut fingertips, four strokes along
the label 10 mm apart; the label sticks where they pressed. Then the hand takes the tab again and
draws it straight up, and the label peels at the pull the folder reads. Two controls: the label laid
and not rubbed makes almost no bond, and with **Adhesive off** the rub makes none.
"""

from __future__ import annotations

import time
import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""One GPU: the example built twice and driven through its page."""

PORT = 9795
CONTROL_PORT = 9796
WORK = np.array([0.6, 0.0, 0.728])
"""``bench_0728``'s work-surface origin in world [m]."""
TOP = 0.05
"""The example's box top over the bench [m]."""
AJAR = 0.6
"""The **Grip amount** the hand opens to around the tab: 38 mm, so the far jaw stays off the box."""
CLOSE = 0.004
"""Where the tool centre closes, over the tab's middle [m]."""
TIP = 0.0165
"""The tool centre over the shut fingertips, pointing down [m]."""
PUSH = 0.003
"""How far below the label the rub puts the shut fingertips' cursor [m]. At 2 mm four strokes made 63%
of the bonds, twice; at 5 mm the squeeze dragged the loose label along with the fingers in two drives
of three (``p18_film.py``)."""
FAR = WORK + np.array([0.0, -0.125, 0.146])
"""Where the carried label's free end touches the box top, its tab 80 mm over it [m]."""
NEAR = WORK + np.array([0.0, -0.040, TOP + 0.019])
"""Where the tab is set down: on the box's near end, the shut-jaw tips 2.5 mm clear of the top [m]."""
HOVER = 0.05
"""How high the shut fingertips move between strokes [m]: clear of the 24 mm tab with the fingers'
flare, 40 mm across 30 mm up them (``p7_fingers.py``)."""
CLEAR = 0.011
"""Where a stroke starts, past the tab's furthest corner along the label [m]: the shut fingers' 7.2 mm
half-width 25 mm up, and 3.8 mm for the tab to creep in while the hand comes down. The label's pull
leans the tab over the strokes; started 8 mm past its middle, the fingers pushed it over
(``p15_descend.py``), and 19 mm past its middle, on the third stroke (``p18_film.py``)."""
COLUMNS = (1, 3, 5, 7)
"""The label's grid columns the four strokes run along: 5, 15, 25 and 35 mm across its 40."""
LIFT = 0.12
"""How high over the box top the hand draws the tab, straight up, to peel the label [m]."""
PEAK_CAP = 5.0
"""The most the tab may be pulled with while the label peels [N]: the highest peak measured, 4.37 N, and 15%
over it. A drive is stepped, one control step a tick whatever the wall clock, but no two drives step alike:
two of them part at the first step (joint targets 1e-11 apart), with the page's wall-clock refresh off or on.
Over 28 drives of this drive the peak was 2.16-2.51 N, median 2.29 N, 16 alone on a GPU and 12 beside a
second drive on the same GPU at 69-70 ms a step against 35 ms, which moved it nowhere (mean 2.30 N against
2.31 N). Seven earlier drives of the same tree read 2.28-2.45 N but one, run among other tests' load, at 4.37 N,
and seven of the tree before it 2.23-2.68 N but one at 3.54 N. 4.0 N failed the 4.37 N drive (``trace.py``,
``sweep.sh``). 24 more, three at once on one GPU beside a fourth process looping ``test_peel_a_label``, at
137-140 ms a step, read 2.17-2.55 N, median 2.27 N, none over 3.0 N, each peaking 101-127 steps into the
206-step draw: that load did not bring the 4.37 N back (``r5/drive.py``, ``r5/sweep.sh``)."""


def column(label, state, c: int, along: float) -> np.ndarray:
    """Where the label's grid column ``c`` is, ``along`` metres from its tab end, now [m, world].

    Past the last row it carries on along the last cell.
    """
    nx, ny = label.spec.cells
    cell = label.spec.cell[1]
    q = state.particle_q.numpy()[np.asarray(label.particles)]
    j = min(int(along // cell), ny - 1)
    t = along / cell - j
    return (1.0 - t) * q[j * (nx + 1) + c] + t * q[(j + 1) * (nx + 1) + c]


def lay(session, label) -> dict:
    """Take the tab off the bench, carry the label over the box and lay it down along it; let go."""
    arm = session.arms["right"]
    state = lambda: session.env.state  # noqa: E731
    tab0 = label.tab_pose(state())[:3]
    arm.grip_request = AJAR
    tick(session, 40)
    go(session, "right", tab0 + np.array([0.0, 0.0, 0.05]), speed=0.08)
    miss = go(session, "right", tab0 + np.array([0.0, 0.0, CLOSE]), speed=0.04)
    arm.grip_request = 1.0
    tick(session, 60)
    out = {"tab_miss": miss, "closed_width": float(session.env.gripper("right").width(state()))}
    go(session, "right", np.array([tab0[0], tab0[1], WORK[2] + 0.15]), speed=0.04)
    out["carried"] = float(label.tab_pose(state())[2] - tab0[2])
    go(session, "right", FAR, speed=0.03)
    tick(session, 60)
    for k in range(1, 31):  # down and toward the near end together: the label lies down from its free end
        go(session, "right", FAR + (NEAR - FAR) * k / 30, speed=0.03, steps=5)
    tick(session, 20)
    arm.grip_request = AJAR
    tick(session, 40)
    go(session, "right", NEAR + np.array([0.0, 0.0, 0.05]), speed=0.05)
    tick(session, 30)
    q = state().particle_q.numpy()[label.bond_particles]
    box = state().body_q.numpy()[label.body][:3]
    on_top = (np.abs(q[:, 0] - box[0]) < 0.05) & (np.abs(q[:, 1] - box[1]) < 0.06)
    flat = on_top & (q[:, 2] - box[2] - 0.025 < 0.0015)
    out.update(laid=label.bonded(), flat=float(flat.mean()))
    return out


def tab_reach(label, state, away: np.ndarray) -> float:
    """How far along ``away`` the tab reaches, its corners counted, lean and all [m]."""
    from robosimstudio.core.math import quat_to_mat

    pose = label.tab_pose(state)
    half = 0.5 * np.asarray(label.spec.tab_size, dtype=np.float64)
    corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * half
    return float(np.max((corners @ quat_to_mat(pose[3:7]).T + pose[:3]) @ away))


def rub(session, label) -> list[float]:
    """Shut the hand and rub along four of the label's columns, where it lies now: bonded after each.

    The label lies 9 degrees off the bench's axes after the lay (``p12_lay.py``, three lays), so each
    stroke runs along one of its columns, from :data:`CLEAR` past the tab's middle to 3 mm past the
    label's end. The jaws stay square to the bench: turned 9 degrees to the label, pressing 5 mm under
    it, the shut fingertips dragged the whole loose label 35 mm along and made 7% of its bonds, where
    square they made 25% in the same stroke (``p17_stroke.py``, one drive each).
    """
    arm = session.arms["right"]
    state = session.env.state
    a, b = column(label, state, 4, 0.03), column(label, state, 4, 0.09)
    away = np.array([b[0] - a[0], b[1] - a[1], 0.0])
    away /= np.linalg.norm(away)
    arm.grip_request = 1.0
    tick(session, 40)
    out = []
    lift = np.array([0.0, 0.0, TIP - PUSH])

    def ends(c: int) -> tuple[np.ndarray, np.ndarray]:
        state = session.env.state
        first = column(label, state, c, 0.03)
        start = first - away * (float(np.dot(first, away)) - tab_reach(label, state, away) - CLEAR) + lift
        return start, column(label, state, c, 0.098) + lift

    for c in COLUMNS:
        start, end = ends(c)
        go(session, "right", start + np.array([0.0, 0.0, HOVER + PUSH]), speed=0.08, steps=60)
        start, end = ends(c)  # the loose label and its tab creep on the box, up to 1.9 mm/s (p16_drift.py)
        go(session, "right", start, speed=0.05, steps=0)  # held a second at 5 mm, it dragged the label (p14_drive.py)
        tick(session, 10)
        go(session, "right", end, speed=0.03, steps=30)
        tick(session, 10)
        go(session, "right", end + np.array([0.0, 0.0, HOVER + PUSH]), speed=0.05, steps=30)
        out.append(label.bonded())
    return out


def pull(session, label) -> dict:
    """Take the tab again and draw it straight up: how hard the label pulled on it, and what let go."""
    arm = session.arms["right"]
    state = lambda: session.env.state  # noqa: E731
    arm.grip_request = AJAR
    tick(session, 40)
    tab0 = label.tab_pose(state())[:3]
    go(session, "right", tab0 + np.array([0.0, 0.0, 0.05]), speed=0.08)
    go(session, "right", tab0 + np.array([0.0, 0.0, CLOSE]), speed=0.04)
    arm.grip_request = 1.0
    tick(session, 60)
    box0 = np.asarray(state().body_q.numpy()[label.body][:3], dtype=np.float64)
    start = np.array(arm.cursor_pos, dtype=np.float64)
    top = np.array([start[0], start[1], WORK[2] + TOP + LIFT])
    legs = int(np.ceil(np.linalg.norm(top - start) / (0.03 / 60.0)))  # 30 mm/s, as a drag of the gizmo
    pulls, bonded = [], []
    for k in range(1, legs + 1):
        arm.cursor_pos, arm.dragging = start + (top - start) * (k / legs), k < legs
        tick(session, 1)
        pulls.append(label.pull(state()))
        bonded.append(label.bonded())
    tick(session, 60)
    moved = float(np.linalg.norm(state().body_q.numpy()[label.body][:3] - box0))
    return {"peak": float(max(pulls)), "bonded": [*bonded[::30], label.bonded()], "box_moved": moved}


def _drive(port: int, *, adhesive: bool) -> dict:
    """Build the example, lay and rub the label through its page, then pull it by the tab."""
    sim, kwargs = _run("03_materials/09_adhesion.py")
    label = sim["label"].entity
    out: dict = {"bonds": len(label.bonds), "before": label.bonded()}
    session = open_session(sim, port, kwargs.get("page"))
    try:
        tick(session, 20)
        t, steps = time.perf_counter(), sim.step_count
        out.update(lay(session, label))
        if not adhesive:
            label.write_adhesive("strength", 0.0)  # what the page's Adhesive off box writes
        out["rubbed"] = rub(session, label)
        out["ms"] = 1e3 * (time.perf_counter() - t) / max(sim.step_count - steps, 1)
        if adhesive:
            out.update(pull(session, label))
        out["finite"] = bool(sim.state_is_finite())
    finally:
        session.teardown()
    return out


class TestTheHandSticksTheLabel(unittest.TestCase):
    """Carried, laid, rubbed: the label sticks where the fingertips pressed, then peels when pulled."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.got = _drive(PORT, adhesive=True)

    def test_the_scene_steps_inside_an_interactive_budget(self) -> None:
        self.assertLess(self.got["ms"], 60.0, f"{self.got['ms']:.1f} ms per control step")

    def test_the_pads_take_the_tab_and_carry_the_label(self) -> None:
        self.assertLess(self.got["tab_miss"], 0.008, f"the hand ended {self.got['tab_miss'] * 1e3:.1f} mm short")
        self.assertGreater(self.got["closed_width"], 0.006, f"{self.got['closed_width'] * 1e3:.2f} mm")
        self.assertGreater(self.got["carried"], 0.10, f"the tab rose {self.got['carried'] * 1e3:.0f} mm")

    def test_laid_and_not_pressed_it_barely_sticks(self) -> None:
        self.assertEqual(0.0, self.got["before"])
        self.assertGreater(self.got["flat"], 0.9, f"{self.got['flat']:.2f} of the label lies flat on the box")
        self.assertLess(self.got["laid"], 0.1, f"bonded {self.got['laid']:.3f} before the rub")

    def test_the_rub_sticks_it(self) -> None:
        rubbed = self.got["rubbed"]
        self.assertGreaterEqual(rubbed[-1], 0.8, f"bonded after each stroke {np.round(rubbed, 3)}")
        self.assertTrue(np.all(np.diff(rubbed) > 0.05), f"each stroke adds its own band: {np.round(rubbed, 3)}")
        self.assertTrue(self.got["finite"])

    def test_the_pull_peels_it_at_the_force_the_folder_reads(self) -> None:
        self.assertLess(self.got["bonded"][-1], 0.3, f"bonded through the pull {np.round(self.got['bonded'], 3)}")
        self.assertGreater(self.got["peak"], 0.5, f"peak pull {self.got['peak']:.2f} N")
        self.assertLess(self.got["peak"], PEAK_CAP, f"peak pull {self.got['peak']:.2f} N")
        self.assertLess(self.got["box_moved"], 0.01, f"the box moved {self.got['box_moved'] * 1e3:.1f} mm")


class TestWithTheAdhesiveOffNothingSticks(unittest.TestCase):
    """The same carry, lay and rub with strength 0, what the page's Adhesive off box writes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.got = _drive(CONTROL_PORT, adhesive=False)

    def test_the_rub_makes_no_bond(self) -> None:
        self.assertEqual(0.0, self.got["rubbed"][-1], f"bonded after each stroke {self.got['rubbed']}")


if __name__ == "__main__":
    unittest.main()
