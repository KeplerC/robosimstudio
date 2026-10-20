# SPDX-License-Identifier: Apache-2.0
"""The tape example's headline task, done by the robot's hand the way its page asks.

``examples/03_materials/08_tape`` promises a gesture and a number. This drives the example's own
arm panel through :mod:`tests.fixtures.grasp` -- the cursor the gizmo writes, the grip the **Close**
button writes, the session's own setpoint law -- and reads what the page's **Label** folder reads.
The right hand is put down until its open jaws straddle the pull tab, closes, is dragged up 30 mm,
then seven times back 20 mm toward the right arm and up 10 mm. Nothing pins the tab to the hand: the
pads squeeze it. The label peels row by row, the tab stays in the hand while it does and the box
stays where it lay. Then the same label with ``break_opening`` set to 0, what the page's **Never let
go** box writes: nothing peels, and the pull goes into the box instead. Either the tab slides out of
the pads or, when they hold it, the hand drags the whole box along by its label.
"""

from __future__ import annotations

import time
import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""One GPU: the example built twice and driven through its page, about 38 ms a control step."""

PORT = 9791
CONTROL_PORT = 9792
WORK = np.array([0.6, 0.0, 0.728])
"""``bench_0728``'s work-surface origin in world [m]."""
TOP = 0.05
"""The example's box top over the bench [m]."""
STAIRS = (("z", 0.03), *(leg for k in range(1, 8) for leg in (("y", -0.04 - 0.02 * k), ("z", 0.03 + 0.01 * k))))
"""The peel the page asks for, one gizmo arrow at a time, as the take films it [m, work-surface frame;
a ``z`` over the box top]: up to 30 mm, then seven times back 20 mm toward the right arm and up 10 mm,
to 180 mm along and 100 mm up. With the example's damped jaws, in 24 drives every bond let go and the
tab stayed within 3.4 mm of where the pads took it (``peel_probe.py``, two drives to a GPU, GPUs 1 and
3). With the jaws undamped (``target_kd`` 60) the drive failed 2 times in 75, the pads closing with no
tab between them (:meth:`TestTheHandPeelsTheLabel.test_the_pads_close_on_the_tab`). In 80 mm legs
(back 80 mm, up 25 mm, twice), with the 10 g tab and 8 N grip of an earlier draft, the tab slid out of
the pads in three drives of seven, with 44%, 63% and 63% of the label off (``m4_drives.py``)."""


CLOSE = 0.004
"""Where the tool centre closes, over the tab's middle [m]: the pads' lower edges 4 mm over the box top."""


def peel(session, label, stairs=STAIRS) -> dict:
    """Down onto the tab, **Close**, then the stairs: the peel and the tab's slip after each leg.

    ``slip`` is how far the tab has moved from where it sat in the hand when the pads closed [m].
    """
    arm = session.arms["right"]

    def hand() -> np.ndarray:
        return arm.ik.ee_pose(session.env.state)[0]

    def tab() -> np.ndarray:
        return label.tab_pose(session.env.state)[:3]

    tab0 = tab()
    out: dict = {"home_to_tab": float(np.linalg.norm(arm.ik.home_pose[0][:2] - tab0[:2]))}
    miss = [go(session, "right", tab0 + np.array([0.0, 0.0, CLOSE]), speed=0.04)]
    # Where the tab stood against the hand as Close was pressed, and how far it leaned [mm, deg]:
    # what a failed close needs to say about itself.
    pose = label.tab_pose(session.env.state)
    out["at_close"] = (*np.round((hand() - pose[:3]) * 1e3, 2), round(float(np.degrees(_lean(pose[3:7]))), 2))
    arm.grip_request = 1.0
    tick(session, 60)
    offset = tab() - hand()
    out["closed_width"] = float(session.env.gripper("right").width(session.env.state))
    slips, peeled, cursor = [], [], hand().copy()
    for axis, to in stairs:
        i = "xyz".index(axis)
        cursor[i] = WORK[i] + (TOP + to if axis == "z" else to)
        miss.append(go(session, "right", cursor.copy(), speed=0.03, steps=90))
        slips.append(float(np.linalg.norm(tab() - hand() - offset)))
        peeled.append(label.peeled())
    out.update(miss=max(miss[:2]), slips=slips, peeled=peeled, front=label.peel_front(), held=label.held())
    return out


def _lean(quat) -> float:
    """How far a body's own z axis leans off world z [rad], from its ``xyzw`` quaternion."""
    x, y = float(quat[0]), float(quat[1])
    return float(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


def _drive(port: int, *, unbreakable: bool, stairs=STAIRS) -> dict:
    """Build the example, drive its page through the peel, read the label, the tab and the box."""
    sim, kwargs = _run("03_materials/08_tape.py")
    label = sim["label"].entity
    out: dict = {"bonds": len(label.bonds)}
    session = open_session(sim, port, kwargs.get("page"))
    try:
        if unbreakable:
            label.write_break_opening(0.0)
        tick(session, 30)
        t = time.perf_counter()
        tick(session, 60)
        out["idle_ms"] = 1e3 * (time.perf_counter() - t) / 60
        box0 = np.asarray(sim["box"].position, dtype=np.float64)
        t, steps = time.perf_counter(), sim.step_count
        out.update(peel(session, label, stairs))
        out["drive_ms"] = 1e3 * (time.perf_counter() - t) / max(sim.step_count - steps, 1)
        out["box_moved"] = float(np.linalg.norm(np.asarray(sim["box"].position, dtype=np.float64) - box0))
        out["finite"] = bool(sim.state_is_finite())
    finally:
        session.teardown()
    return out


def while_peeling(got: dict) -> list[float]:
    """The tab's slips up to the leg after which the whole label was off."""
    peeled = got["peeled"]
    done = next((i for i, value in enumerate(peeled) if value >= 1.0), len(peeled))
    return got["slips"][:done]


class TestTheHandPeelsTheLabel(unittest.TestCase):
    """Down onto the tab, Close, up and back: the label peels and the tab stays in the hand."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.got = _drive(PORT, unbreakable=False)

    def test_the_scene_steps_inside_an_interactive_budget(self) -> None:
        # Through the page, alone on GPU 1 or 6: about 38.5 ms a control step at rest, 37.5-37.8 peeling (m10_screen.py).
        self.assertLess(self.got["idle_ms"], 60.0, f"{self.got['idle_ms']:.1f} ms per control step at rest")
        self.assertLess(self.got["drive_ms"], 60.0, f"{self.got['drive_ms']:.1f} ms per control step peeling")

    def test_the_right_hand_starts_over_the_tab_and_gets_where_it_is_dragged(self) -> None:
        self.assertLess(self.got["home_to_tab"], 0.006, "one straight drag down reaches the tab")
        self.assertLess(self.got["miss"], 0.008, f"the hand ended {self.got['miss'] * 1e3:.1f} mm short")

    def test_the_pads_close_on_the_tab(self) -> None:
        # The tab is 8 mm thick across the jaws: they stop at 8.33 to 8.34 mm (24 drives, peel_probe.py);
        # closed on nothing, over the box, they read 0.04 mm after the same 60 steps (peel_air.py). Undamped
        # (target_kd 60) the jaws shut onto the tab 8 mm a step, bounced up to 14 mm open again and, in 3
        # closes of 116, knocked it aside and shut on nothing: 0.25 and 0.04 mm, with the tab upright and
        # centred at Close (peel_probe.py, peel_close.py, the round-3 review). The message carries the
        # hand's offset from the tab at Close and the tab's lean: (-0.22 to -0.14, -0.14 to 0.39, 5.7 to
        # 6.8) mm and at most 0.4 deg in the 24 drives.
        width = self.got["closed_width"]
        self.assertGreater(width, 0.006, f"{width * 1e3:.2f} mm; hand - tab, lean at Close: {self.got['at_close']}")

    def test_the_whole_label_comes_off(self) -> None:
        # 1.0 in 24 drives of 24, every one of the 144 bonds (see STAIRS); the label that never lets go below reads 0.
        self.assertGreaterEqual(self.got["peeled"][-1], 0.95, f"peeled {np.round(self.got['peeled'], 3)}")
        self.assertGreater(self.got["front"], 0.07, f"front {self.got['front'] * 1e3:.1f} mm")
        self.assertTrue(self.got["finite"])

    def test_the_peel_front_travels_as_the_hand_pulls(self) -> None:
        peeled = np.asarray(self.got["peeled"])
        self.assertTrue(np.all(np.diff(peeled) >= 0.0), "a broken bond stays broken")
        self.assertLess(peeled[0], 0.2, f"the first lift takes the first rows, not the label: {np.round(peeled, 3)}")

    def test_the_tab_stays_in_the_hand_while_the_label_comes_off(self) -> None:
        # 3.35 to 3.42 mm off where the pads took it while the label came off, in 24 drives (peel_probe.py).
        slips = while_peeling(self.got)
        self.assertLess(max(slips), 0.01, f"tab slip {np.round(np.array(self.got['slips']) * 1e3, 1)} mm")

    def test_the_box_stays_where_it_lay(self) -> None:
        # Nothing holds it down: it moved 1.46 to 1.72 mm in 24 drives (peel_probe.py).
        self.assertLess(self.got["box_moved"], 0.005, f"the box moved {self.got['box_moved'] * 1e3:.1f} mm")


class TestALabelThatNeverLetsGoStaysStuck(unittest.TestCase):
    """The same label with ``break_opening`` 0, what Never let go writes, pulled the same way four steps."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.got = _drive(CONTROL_PORT, unbreakable=True, stairs=STAIRS[:9])

    def test_nothing_peels_and_the_pull_goes_into_the_box(self) -> None:
        # Every bond held in 24 drives of 24 (peel_probe.py, damped jaws, GPUs 1 and 3). In 10 the label
        # dragged the box 4.6 to 15.3 mm until the tab slid out of the pads, 84 to 90 mm off where they
        # took it. In 14 the pads held and the hand dragged the 0.5 kg box 45 to 46 mm by its label: two
        # pads holding the tab with about 5.8 N each at friction 0.8, 9.3 N of grip, can pull more than the
        # box's 4.9 N weight (12 N is the finger's cap, met only by Close's first push). Either way
        # the hand does not take the tab away from a box that stays put, which is what peeling is.
        self.assertEqual(0.0, self.got["peeled"][-1])
        self.assertEqual(self.got["bonds"], self.got["held"])
        slips = np.round(np.array(self.got["slips"]) * 1e3, 1)
        went = max(*self.got["slips"], self.got["box_moved"])
        self.assertGreater(went, 0.03, f"tab slip {slips} mm, box moved {self.got['box_moved'] * 1e3:.1f} mm")


if __name__ == "__main__":
    unittest.main()
