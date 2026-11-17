# SPDX-License-Identifier: Apache-2.0
"""A scripted solution for the peel variants: take the tab between the pads and draw the sheet off.

Contact only. The work hand comes down over the sheet's pull tab with its jaws open and closing along
the sheet, closes on the tab and holds it by squeezing it; nothing pins the tab to the hand and
nothing writes the sheet's state. Then the hand draws the tab up and back over the part still stuck,
and the bonds let go in front of it, row by row.

**The pull follows the peel front.** ``examples/03_materials/08_tape`` peels a 95 mm label with a
fixed staircase -- up 30 mm, then seven times back 20 mm and up 10 mm -- which is about twice the
sheet's length of hand travel. So after the first lift each leg is aimed from where the peel has got
to: the peeled flap, from the front back to the tab, is held at :data:`FLAP_ANGLE` over the stuck
sheet and stretched :data:`STRETCH` past its own length, which is the pull that lets the next row go.
While the front does not move the stretch grows. The legs are short (:data:`LEG`) and slow
(:data:`SPEED`), the pace the example's tab stayed within 3.4 mm of the pads at.

**The hold is chosen by reach.** Of the two headings that close the jaws along the sheet, and of the
tool pitched about that closing axis (:data:`PITCHES`), the routine holds the least-pitched one whose
approach, close and whole pull solve within :data:`REACH_OK`, and keeps it for the whole take.

The sheet is the document's ``(:policy-params (sheet NAME))``, and it runs from its tab along the
sheet frame's ``+y`` (``TapeSpec``): that is the direction the hand pulls.
"""

from __future__ import annotations

import math

import numpy as np

from robosimstudio.arms import arm_slot, arm_slots
from robosimstudio.core.math import quat_from_axis_angle, quat_multiply, quat_to_mat
from robosimstudio.robots.embodiments import top_down_grasp_site_quat, work_gripper
from robosimstudio_bench.solutions.base import solution_param

__all__ = ["RoutineTapePeel"]

OPEN, SHUT = 0.0, 1.0
"""The gripper channel, normalized: 0 open, 1 shut."""

OVER = 0.08
"""How high over the tab the hand comes in [m]."""

CLOSE = 0.004
"""Where the tool centre closes, over the tab's middle [m]: the example's, the pads' lower edges just
over the face."""

GRIP_STEPS = 60
"""Control steps the jaws close for before the hand moves: the example's 1 s."""

FIRST_LIFT = 0.03
"""The first leg: straight up [m], which lifts the fold row off."""

FLAP_ANGLE = math.radians(35.0)
"""The peeled flap's angle over the stuck sheet, from the face [rad]: a 145-degree peel. The shallower
the peel the less it takes to open a row (a peel's force goes as ``1 / (1 - cos)`` of its angle), and
the more of the pull runs along the jaws' closing axis, into a pad, rather than along the pad faces,
where only friction holds the tab. At 60 degrees the tab crept out of the pads 1 mm a leg and was
lost with 84% of the strip off; the example's staircase is about 150 degrees."""

STRETCH = 0.008
"""How far past the flap's own length the hand pulls [m]: what makes the next row let go."""

STRETCH_STEP = 0.002
STRETCH_MAX = 0.03
"""While the front does not move, the pull stretches the flap this much further a leg, up to this [m]."""

LEG = 0.012
"""The most the hand moves in one leg [m]; each leg is re-aimed from the peel front."""

SPEED = 0.03
"""The hand's pace [m/s]: the example's."""

PITCHES = (0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0)
"""Pitches of the tool about the jaws' closing axis the survey tries [deg], least first."""

REACH_OK = 0.002
"""A hold whose worst waypoint over the approach and the pull solves within this reaches [m]."""

LEGS_MAX = 200
STALL_LEGS = 25
"""At most this many pull legs, and this many in a row with the front not moving."""

DONE = 0.99
"""The share peeled at which the routine stops pulling."""

MAX_POS = 0.006
MAX_ROT = math.radians(5.0)
"""A leg whose waypoints solve worse than this is not driven [m, rad]."""


class RoutineTapePeel:
    """Pinch the pull tab, then draw the sheet off its body along the peel front."""

    MODE = "joint_position"

    def __init__(self, env, **kwargs) -> None:
        del kwargs
        self.env = env
        self.work = env.handles.work_arm or arm_slots(env)[0]
        self.dof = len(env.idle_action(self.MODE)) // len(arm_slots(env))
        self.sheet_name = str(solution_param(env, "sheet", "tape"))
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        env = self.env
        # The tape entity itself: a deformable, so not among the handles' rigid entities.
        self.sheet = env.scene.by_name[self.sheet_name]
        self.held = {n: self._q(n) for n in arm_slots(env)}
        self.grip = dict.fromkeys(arm_slots(env), OPEN)
        self.seed = {n: np.asarray(env.state.joint_q.numpy(), dtype=np.float64) for n in arm_slots(env)}
        self.refused: list[str] = []
        self.notes: list[str] = []
        self.phase = 0.0
        self.legs = 0
        self.tab_slip = 0.0
        self._steps = self._choreography()

    def _q(self, name) -> np.ndarray:
        ik = self.env.arm(name)
        return np.asarray(self.env.state.joint_q.numpy(), dtype=np.float64)[ik.arm_coords]

    def row_extras(self) -> dict[str, float]:
        return {
            "phase": float(self.phase),
            "legs": float(self.legs),
            "legs_refused": float(len(self.refused)),
            "peeled": float(self.sheet.peeled()),
            "tab_slip_mm": 1e3 * float(self.tab_slip),
        }

    # ------------------------------------------------------------------ drive
    def act(self) -> np.ndarray:
        try:
            next(self._steps)
        except StopIteration:
            pass
        a = np.asarray(self.env.idle_action(self.MODE), dtype=np.float64)
        for name in arm_slots(self.env):
            s = arm_slot(self.env, name) * self.dof
            a[s : s + self.dof - 1] = self.held[name][: self.dof - 1]
            a[s + self.dof - 1] = self.grip[name]
        return a

    def _hold(self, n: int):
        for _ in range(n):
            yield

    def _leg(self, target, quat, *, speed=SPEED, label="", gate=True, stride=2):
        """Solve the straight line from the hand to ``target`` end to end, gate it, then drive it."""
        name = self.work
        ik = self.env.arm(name)
        start = np.asarray(ik.ee_pose(self.env.state)[0], dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        n = max(2, int(np.linalg.norm(target - start) / speed * 60.0))
        knots = sorted({*range(stride, n + 1, stride), n})
        sols, worst_p, worst_r, seed = {}, 0.0, 0.0, self.seed[name]
        for k in knots:
            point = start + (target - start) * (k / n)
            sol = ik.solve_q_full(seed, point, quat)
            p, r = ik.pose_errors(sol, point, quat)
            worst_p, worst_r = max(worst_p, p), max(worst_r, r)
            sols[k] = np.asarray(sol, dtype=np.float64)
            seed = sol
        if gate and (worst_p > MAX_POS or worst_r > MAX_ROT):
            self.refused.append(f"{label}: {worst_p * 1e3:.1f} mm / {math.degrees(worst_r):.1f} deg")
            return False
        prev_k, prev_q = 0, np.asarray(self.seed[name], dtype=np.float64)
        for k in knots:
            for j in range(prev_k + 1, k + 1):
                q = prev_q + (sols[k] - prev_q) * ((j - prev_k) / (k - prev_k))
                self.held[name] = q[list(ik.arm_coords)]
                yield
            self.seed[name] = sols[k]
            prev_k, prev_q = k, sols[k]
        return True

    # ------------------------------------------------------------- geometry
    def _survey(self) -> None:
        """Where the tab is, which way the sheet runs from it, and the hand's grasp orientation."""
        env = self.env
        spec = self.sheet.spec
        tab = self.sheet.tab_pose(env.state)
        self.tab0 = np.asarray(tab[:3], dtype=np.float64)
        along = quat_to_mat(np.asarray(spec.quat, dtype=np.float64)) @ np.array([0.0, 1.0, 0.0])
        along[2] = 0.0
        self.along = along / np.linalg.norm(along)
        # The fold: where the sheet leaves the face, under the tab, at the face's height.
        self.face_z = float(self.tab0[2]) - 0.5 * float(spec.tab_size[2])
        self.fold = np.array([self.tab0[0], self.tab0[1], self.face_z]) + 0.5 * float(spec.tab_size[1]) * self.along
        self.flap0 = float(spec.tab)
        # Jaws closing along the sheet, across the tab's thickness. Which of the two headings, and how far
        # the tool is pitched about that closing axis, are the arm's to answer: every hold is scored over
        # the approach, the close and the pull's whole line, and the least-pitched one that reaches is held
        # throughout: 30 degrees on the box, 45 on the phone.
        hand = work_gripper(env)
        yaw = math.degrees(math.atan2(float(self.along[1]), float(self.along[0])))
        ik = env.arm(self.work)
        up = np.array([0.0, 0.0, 1.0])
        path = [self.tab0 + np.array([0.0, 0.0, OVER]), self.tab0 + np.array([0.0, 0.0, CLOSE])]
        length = float(spec.size[1])
        for f in (0.25, 0.5, 0.75, 1.0):
            front = f * (length - self.flap0)
            path.append(
                self.fold
                + front * self.along
                + (self.flap0 + front + STRETCH) * (math.cos(FLAP_ANGLE) * self.along + math.sin(FLAP_ANGLE) * up)
            )
        best = None
        for heading in (yaw, yaw + 180.0):
            down = np.asarray(top_down_grasp_site_quat(hand, heading), dtype=np.float64)
            for pitch in PITCHES:
                quat = np.asarray(
                    quat_multiply(quat_from_axis_angle(self.along, math.radians(pitch)), down), dtype=np.float64
                )
                seed, worst = self.seed[self.work], 0.0
                for point in path:
                    sol = ik.solve_q_full(seed, point, quat)
                    p, r = ik.pose_errors(sol, point, quat)
                    worst = max(worst, p + 0.02 * r)
                    seed = sol
                # The least pitch that reaches: the pads close squarely on the upright tab
                # held straight down, and overlap less of it the further they are pitched.
                score = worst if worst > REACH_OK else 0.0
                if best is None or score < best[0] - 1.0e-4:
                    best = (score, quat, heading, pitch, worst)
        self.quat = best[1]
        self.notes.append(
            f"heading {best[2]:.0f} deg, pitched {best[3]:+.0f} deg about the jaws: worst {best[4] * 1e3:.1f} mm "
            f"over the pull; tab at {np.round(self.tab0 * 1e3, 1)} mm"
        )

    def _pull_target(self, extra: float = 0.0) -> np.ndarray:
        """Where the hand goes next: the flap from the front to the tab at FLAP_ANGLE, STRETCHed."""
        front = float(self.sheet.peel_front())
        point = self.fold + front * self.along
        flap = self.flap0 + front + STRETCH + extra
        return point + flap * (math.cos(FLAP_ANGLE) * self.along + math.sin(FLAP_ANGLE) * np.array([0.0, 0.0, 1.0]))

    # ---------------------------------------------------------- choreography
    def _choreography(self):
        self._survey()
        quat = self.quat
        self.phase = 1.0
        yield from self._leg(self.tab0 + np.array([0.0, 0.0, OVER]), quat, speed=0.15, gate=False, label="over the tab")
        yield from self._hold(15)
        yield from self._leg(self.tab0 + np.array([0.0, 0.0, CLOSE]), quat, speed=0.04, gate=False, label="down")
        yield from self._hold(10)
        self.phase = 2.0
        self.grip[self.work] = SHUT
        yield from self._hold(GRIP_STEPS)
        ik = self.env.arm(self.work)
        hand0 = np.asarray(ik.ee_pose(self.env.state)[0], dtype=np.float64)
        offset = self.sheet.tab_pose(self.env.state)[:3] - hand0
        width = float(self.env.gripper(self.work).width(self.env.state))
        self.notes.append(f"closed: tab {np.round(offset * 1e3, 1)} mm from the tool centre, jaws {width * 1e3:.1f} mm")
        self.phase = 3.0
        if not (yield from self._leg(hand0 + np.array([0.0, 0.0, FIRST_LIFT]), quat, label="first lift")):
            return
        # Leg by leg, each from where the peel front has got to.
        extra, last_front, still = 0.0, -1.0, 0
        while float(self.sheet.peeled()) < DONE and self.legs < LEGS_MAX:
            front = float(self.sheet.peel_front())
            if front > last_front + 1.0e-4:
                extra, still, last_front = 0.0, 0, front
            else:
                still += 1
                extra = min(STRETCH_MAX, extra + STRETCH_STEP)
                if still > STALL_LEGS:
                    self.notes.append(f"the front stopped at {front * 1e3:.0f} mm")
                    break
            hand = np.asarray(ik.ee_pose(self.env.state)[0], dtype=np.float64)
            goal = self._pull_target(extra)
            step = goal - hand
            dist = float(np.linalg.norm(step))
            if dist > LEG:
                goal = hand + LEG * step / dist
            if dist > 1.0e-3:
                ok = yield from self._leg(goal, quat, label=f"pull {self.legs}")
                if not ok:
                    break
            else:
                yield from self._hold(6)
            self.legs += 1
            slip = self.sheet.tab_pose(self.env.state)[:3] - np.asarray(ik.ee_pose(self.env.state)[0]) - offset
            self.tab_slip = max(self.tab_slip, float(np.linalg.norm(slip)))
        self.notes.append(
            f"pulled {self.legs} legs: {self.sheet.peeled():.3f} peeled, tab slip {self.tab_slip * 1e3:.1f} mm"
        )
        self.phase = 4.0
        yield from self._hold(30)
        # Let go of the peeled sheet and lift clear.
        self.grip[self.work] = OPEN
        yield from self._hold(40)
        hand = np.asarray(ik.ee_pose(self.env.state)[0], dtype=np.float64)
        yield from self._leg(hand + np.array([0.0, 0.0, 0.05]), quat, speed=0.08, gate=False, label="clear")
        self.phase = 5.0
        while True:
            yield
