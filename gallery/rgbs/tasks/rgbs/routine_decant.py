# SPDX-License-Identifier: Apache-2.0
"""A scripted solution: pick the cutter, pin the carton, seat the blade, draw the tape, put the cutter down, hook the far flap open.

The order is the clip's (``sample_videos/box_decanting.mp4``): the right arm
takes the cutter off the mat; the left arm comes in and pins the carton's far
(+y) end; the right arm seats the blade at the right (-y) end of the tape and
draws it toward the left arm, one way per pass; the cutter goes back down on
the mat where it was; then the right arm puts a fingertip into the join near
the right end and lifts the FAR flap up and over; the left hand lets go.

**Five things this cost measurements to get right**, all of them invisible from
the code that appears to control them:

0. **The pick decides where the BLADE goes.** A top-down grasp with the wrist
   left wherever the IK put it does not pinch a baton where it lies -- the YAM's
   long fingertips sweep it into line with the pads, and the shipped knife came
   out of every pick 40-49 degrees yawed. :meth:`RoutineBoxDecant._across_quat`
   turns the wrist to the tool, and :meth:`RoutineBoxDecant._tip_offset`
   measures where the tip actually ended up instead of trusting the arithmetic.

1. **The tilted hold is reachable one way round only.** The blade has to lead
   the body downward (the body's underside would land on the flaps first held
   level), which is a tilt about the jaw's closing axis; swept over the whole
   draw at the tape's height (``decant/layout.py``), that hold solves to
   0.0 mm / 1.7 deg at every station with the cutter's NOSE TOWARD THE WORKING
   ARM and is 93-194 mm out with the nose leading -- the wrist would have to
   reach past the tape from the far side and hook back. So the nose points at
   the arm, the tip trails the body along the +y draw, and the cut law -- which
   reads speed along the edge -- does not know the difference.

2. **Every leg is interpolated, and planned before it is driven.** A single IK
   solve held for N control steps is one joint target the servo slews to at full
   rate, and a carried knife is flung out of the hand on the way. Worse, an
   unreachable waypoint comes back as a solution tens of millimetres off with
   nothing raised -- so each leg is solved end to end, its orientation slerped
   where it turns, and **gated** on its worst residual before anything moves.

3. **Seating IS the press.** ``_blade_force`` reads the blade-against-flaps
   contact the way a wrist force sensor would and the row carries it, but
   nothing servos on it: a search that keeps descending after contact ejects
   the knife from the hand (:data:`SEAT_CLEAR` has the numbers). The seat is
   planned to land the tip at the cutting depth, and the reported force is a
   measurement of what that bought rather than a set-point.

4. **The jaws close at 1.0 and open at 0.0** (``action_closed_at_high``). Driven
   the other way the hand descends shut, knocks the knife across the mat, and
   opens on empty air -- which reads exactly like a grasp that dropped it.
"""

from __future__ import annotations

import math

import numpy as np

from robosimstudio.arms import arm_slot, arm_slots
from robosimstudio.core.math import quat_from_axis_angle, quat_multiply, quat_to_mat
from robosimstudio_bench.solutions.base import solution_param

from .decant.carton import carton_frame
from .decant.knife import blade_edge_local
from .decant.layout import CARTON, KNIFE, seal_tape

__all__ = ["RoutineBoxDecant"]

OPEN, SHUT = 0.0, 1.0
"""This hand closes at the HIGH end of a normalised gripper command."""

GRIP_X = 0.030
"""Where along the body the jaws close, in the knife's own frame [m]: 12 mm
ahead of the body's centre: the 27 mm pads span x 16.5..43.5 mm on a body
whose nose is at 50, so they sit wholly on the slate front with 6.5 mm of
tool ahead of them and the charcoal block behind.

**The press pitches the cutter in the jaws, and the lever is the knob.** A
vertical press at the tip is reacted across the pads' own patch, and what it
costs goes as ``press * lever`` with the lever the HORIZONTAL run from the
pads to the tip -- ``a*cos(tilt) + b*sin(tilt)``, for a tip ``a`` ahead of the
grip along the body and ``b`` below it.

Measured at ``-0.010`` (the pads half on the charcoal block, the tip 75 mm
ahead, a 58 mm lever at 35 degrees): the cutter rotated in the jaws through
the episode -- its tip read 38 mm below the TCP at the first pass and 2 mm by
the fourth, a pitch of some 35 degrees -- and the press it delivered
alternated 7.3 / 0.0 / 1.7 / 0.0 N along one draw while ``blade_slip`` climbed
to 45 mm, which is the document's own "a shy press skates and cuts exactly
nothing". From here, with :data:`CUT_TILT`'s 50 degrees, the tip is 35 mm
ahead and 13 mm below and the lever is 32 mm -- 45 per cent off what the pads
had to hold. Further forward the pads run onto the nose."""

CUT_TILT = 50.0
"""The nose-down tilt the blade is held at [deg], about the jaw's closing axis.
Read from the document's ``cut-tilt`` when it has one.

Steep, for the two reasons :attr:`.SpecBoxCutter.cut_tilt` states: it puts the
tip 11.2 mm below the body's own nose corner rather than 7.8, so the seat can
drive the blade into the tape without the body landing on it; and it shortens
the press's moment arm on the grasp from 42 mm to 32. The reach sweep on
``layout.py`` covers it."""

GRIP_DZ = 0.004
"""The TCP above the body's centre at the pick [m]. The pads end 10.5 mm past
the TCP, so this leaves them 3.5 mm short of the body's underside -- on the
body, not on the mat."""

CARRY_LIFT = 0.07
"""How far above the tape the knife is carried [m], and how far the hand rises
between passes. A clearance, and a small one on purpose: the higher the carry
the further the tilted wrist is from the pose the draw was swept at."""

SEAT_CLEAR = 0.003
"""Where the seat's approach STOPS, above the tape's top [m], before the
force-checked descent takes over.

The shipped routine seated by depth alone -- a seat commanded 6 mm through the
tape, the arm giving back 4.5 mm of it -- and on this knife that read as a
skate: 5 mm of commanded overdrive left the blade drawing at under 6 N for
48 mm on the first pass and 84 on the second, cut 0.000. So the seat now
descends in millimetre steps from here until the blade-against-tape feed
reads :data:`SEAT_FORCE`, the way ``surface_cleaning`` seats its sponge. The
shipped warning still stands and bounds the search: a descent that continues
past contact ejects the knife from the hand, so the search stops at the first
reading over the target and never exceeds :data:`SEAT_TRAVEL`."""

SEAT_FORCE = 4.5
"""The press the seat descends to [N]. Above the field's 3 N gate by the
margin its ``k_c`` is fitted to cut a cell through in one pass, and inside
what this grasp can deliver -- measured at 3-5 N before the tool's own nose
lands on the tape (:attr:`.SpecSealTape.f_pierce` has the table)."""

BODY_TOUCH = 0.5
"""What counts as the cutter's BODY landing on the flaps [N], and the seat's
hard stop. Past it the tool rides on its own nose corner: measured, the body
went 1.4 -> 6.7 -> 13.6 -> 20.3 N over three millimetres while the blade fell
to 0.00 N and the knife was levered round in the jaws. The descent backs off
two millimetres and stops."""

SEAT_TRAVEL = 0.020
"""How far the seat's search may descend [m] before it gives up on a press."""

DRAW_INSET = 0.010
"""How far inside the strip's two ends the draw keeps its tip [m].

**A blade seated on the strip's own end reads nothing.** The strip is a 100 mm
box and the blade a 15 mm one held nose-down, so its only low point is the
tip's corner; over the strip's END that corner meets a box edge and the pair
is degenerate. Measured: seated exactly at the end the search descended the
full 20 mm reading 0.0 N until the tip was 5 mm through, then 1.8-5.2 N with
the tool's own nose about to land -- and the draw that followed read
3.5 / 0.1 / 0.0 / 0.0 N and cut nothing. Ten millimetres in, the same search
stops at 0.6 mm of penetration with 4.7 N and the first pass cuts 0.81.

The window the law integrates is the blade's EDGE SEGMENT, which runs from the
tip 15 mm toward the tool's NOSE -- and the nose points at the working arm,
against the draw -- so the band swept by a forward pass is
``[first tip, last tip + 15 mm]``. That is why the far inset costs nothing (the
band still reaches 5 mm past the strip) and the near one costs the first
10 mm, which :meth:`RoutineBoxDecant._cut` picks up with a short backward
draw."""

DRAW_BACK = 0.006
"""How far back over the strip's near end each pass draws before it draws
forward [m].

Six, not the whole ten-millimetre inset: the first cell station sits 4 mm from
the strip's end and the law's window reaches 15 mm forward of the tip, so a tip
6 mm back from the seat covers it with 2 mm to spare -- while a tip that backs
up the full inset runs OFF the strip, where the blade reads nothing and a
force-following stroke reads nothing as "press harder"."""

DRAW_POLL = 6
"""How often the draw reads the blade's force and corrects its height, in
control steps -- ten times over the 80 mm stroke."""

DRAW_STEP = 0.0004
"""How far the draw's height correction moves DOWN per reading [m], when the
press is under target. Four tenths of a millimetre against a press that runs
4.5 N per millimetre on a stiff holder and about 1 N per millimetre through
this arm."""

DRAW_BACKOFF = 3.0
"""How much faster the correction retreats than it advances. Asymmetric on
purpose, and the asymmetry is a measurement: a symmetric 0.6 mm servo let one
stroke reach 19 N, which shoved the carton 63 mm and took the episode's cut
from 0.917 to 0.312. Approaching a contact slowly and leaving it quickly is
the right way round for a tool that cuts on a threshold."""

DRAW_CEILING = 2.5
"""What multiple of :data:`SEAT_FORCE` the stroke retreats from.

**Not tighter, and that is measured.** The arithmetic argues for 1.6: the
pin's lateral hold is ``mu * (weight + pin)`` = 7.2 N, the draw's drag is
``0.4 * press``, so a press reaching 17 N drags 6.8 N and is *at* the hold.
But the episode run at 1.6 kept its press to 5.3 N and cut **0.542** against
0.854 at 2.5, and shifted the carton just as far (51 mm against 54) -- so the
drag is not what moves the box, the flap hook is, and the high presses are
what cut the tape. The ceiling is a guard against the tool's nose landing, and
:data:`BODY_TOUCH` is the real one."""

DRAW_SINK = 0.003
"""How far below the seat a stroke's correction may sink [m].

The seat has already found the tape by force, so a stroke that wants more than
three millimetres more is not chasing the arm's sag -- it has lost the tape
altogether, and the reading it is chasing is a zero. Unclamped, a stroke that
ran off the strip's end drove itself to the floor of its range and started the
next one ten millimetres deep."""

PIN_FORCE = 12.0
"""What the pin hand presses the near flap down with [N], found by descending
a millimetre at a time and reading its own pads against the carton.

The pin is a DOWN-force and what it buys is friction: ``mu * (weight + pin)``
on a 0.6 mat, so 12 N buys 7.2 N of lateral hold on top of the 0.9 N the
carton's own 160 g is worth. Against it: a draw drag of 2-5 N, and the flap
hook leaning sideways on the box afterwards. Six newtons was not enough --
measured, the carton finished a whole episode 30 mm adrift -- and the bench's
shove table is what says an unpinned one walks 96 mm at the very press that
opens the tape."""

PIN_INBOARD = 0.011
"""Where across the near flap the pin sits [m]: this far in from the hinge,
so the support hand's carriage stays 25 mm clear in x of the knife's nose at
the seam and of the working hand's carriage over it."""

FINGERTIP_BLADE = 0.007
"""A YAM fingertip pad's thickness [m]: the carton's gap is sized against it
(``carton.FINGERTIP_BLADE``) and the hook is placed with it."""

PAD_OUT = 0.048
"""How far the OUTER face of an open fingertip stands from the TCP [m], across
the jaws' closing axis. **Measured on the built hand, not derived.**

The obvious arithmetic -- the gripper's 47.5 mm of travel plus the 7 mm blade
-- gives 54 mm and is wrong, because the travel is the blade's inner face and
the blade is what the 47.5 is measured to. Read off the fingertip meshes with
the jaws open, the band of vertices at a flap's own height spans TCP + 41 mm
to TCP + 48 mm. Six millimetres of error here is the whole hook: aiming the
outer face 5 mm past the far flap's free edge with the 54 put it 1 mm SHORT of
the edge, and the recorded lift touched the flap for one waypoint out of
twenty-four (``flap_far_open`` 0.036, then back to 0.000 as the hand climbed
away from it)."""

HOOK_DEPTH = 0.012
"""How far below the far flap's UNDERSIDE the fingertip's end is put before it
goes under the flap [m]. The carton's cavity is open under the flaps, so the
only clearance that matters is the 12 mm join the pad descends through."""

HOOK_UNDER = 0.022
"""How far past the far flap's free edge the fingertip's outer face is driven,
at depth, before the hand rises [m]: enough that the pad's END -- a horizontal
face, so its contact normal is VERTICAL -- is under the board rather than
against its edge, and enough that the board does not slide off it as the arc lifts.
Five bought 3-7 mm of purchase and the flap slid off within two waypoints;
this is the pad's own 7 mm thickness clear of the edge again on both sides.

**The edge face is what the first version pushed on, and that is why nothing
opened.** With the hand leaned 22 degrees about the seam so the pad presented
an inclined face under the edge, the recorded contact was a single point on
``carton_flap_far`` through a whole 110-degree arc with ``flap_far_open`` at
exactly 0.000: an inclined face against the flap's 3 mm free-edge face has a
near-HORIZONTAL normal, which pushes the flap toward its own hinge and makes
no opening moment at all. The hand is top-down and unleaned now, and it goes
UNDER."""

HOOK_LEAN = 25.0
"""How far the hooking hand leans about the seam's axis [deg], and WHICH WAY.

The fingertip is a tapered blade, not a parallel one: measured off its mesh
with the jaws open, its outer surface stands at TCP + 48 mm at a shut flap's
own height and at TCP + 43 mm at the pad's bottom, 12 mm lower -- a 16-degree
inward taper. So a VERTICAL finger cannot get its bottom under the flap's free
edge at all: driven in until the bottom is under, its own side is already
jammed against the 3 mm edge face, and the recorded result was the flap (and
the carton with it) shoved 20 mm along +x with ``flap_far_open`` at 0.000
throughout.

Leaning the hand so its approach axis tips toward +x puts the pad's bottom
corner AHEAD of everything above it, which is what lets it pass under the
board -- and that is the rotation which sends world +x UP, so the vector handed
to :meth:`RoutineBoxDecant._lean` is ``-far``. Leaned the other way, tried and
measured, the bottom goes away from the flap and the finger's upper body into
its edge: the flap and the carton with it are shoved 20 mm along +x and
``flap_far_open`` never leaves 0.000.

Once the bottom IS under, a little interference at the flap's own level is the
mechanism rather than a fault: the overlap is 12 mm horizontally and the 3 mm
board vertically, so the least-penetration axis is vertical and the board is
pushed UP. With the bottom short, the only overlap is at the flap's level, the
normal is horizontal, and the flap is pushed sideways -- which is exactly the
pair of recordings above.

The lean is given back over the lift: at 25 degrees the wrist cannot hold the
top of the flap's arc (measured 68 mm of residual at 150 mm over the mat
against 0.0 mm unleaned), and it does not need to -- a flap already up 30
degrees is pushed by a straight finger. So the hand straightens as it rises."""

HOOK_TRIES = 2
"""How many times the fingertip goes back under the board if the lift left the
flap shut. The board slides off a part-lifted fingertip easily, and the box has
usually moved a little by then, so a second attempt is a fresh reading rather
than the same one repeated."""

HOOK_LIFT = 55.0
"""How far round the hinge the fingertip carries the far flap [deg].

Not to the stop: a flat pad end under a plate inclined much past 60 degrees is
a corner contact that slides off it. 55 is where the flap is high enough to be
pushed over from the other side, which is what the flip does."""

FLIP_FROM = 0.030
"""Where the flip's fingertip lands on the raised flap, measured from its free
edge down the flap [m]. The hand comes over the raised flap's -x face and
drives +x, carrying it past vertical; past 90 degrees gravity takes it to its
110-degree stop and it stays there."""

MAX_POS, MAX_ROT = 0.004, math.radians(6.0)
"""What a planned leg may be off by before it is refused."""

DRAW_SPEED = 0.05

PASSES = 4
"""How many one-way draws along the tape. Read from the document's
``draw-passes`` when it has one.

Each pass re-reads the carton, re-measures the tip in hand and follows the
force down the stroke, so a pass that nudged the tool or the box is corrected
for by the next. Four rather than three because the law cuts nothing under its gate: what a
pass leaves behind is wherever its press dipped, and the only way to fill those
is another pass that dips somewhere else. Measured over the ledger: one pass
0.81, four 0.85-0.92, five 0.54 -- and that last is the spread rather than the
pass count, which is why the number stays where the evidence is."""


def _rot(quat) -> np.ndarray:
    """Rotation matrix from an xyzw quaternion."""
    return quat_to_mat(np.asarray(quat, dtype=np.float64))


def _yaw_of(quat) -> float:
    """Heading about +z, from an xyzw quaternion."""
    x, y, z, w = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _half_wrap(angle: float) -> float:
    """``angle`` folded into ``(-pi/2, pi/2]`` -- an axis, not a direction.

    A two-fingered hand's pads are symmetric, so a bar lies the same way in them
    turned by 180 degrees. Folding modulo pi is what keeps the wrist from taking
    the long way round to an orientation it is already in."""
    return (float(angle) + math.pi / 2.0) % math.pi - math.pi / 2.0


def _slerp(q0, q1, t: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64) / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=np.float64) / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1.0 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


class RoutineBoxDecant:
    """Cut the tape open with the safety cutter and hook the far flap up, both hands."""

    MODE = "joint_position"

    def __init__(self, env, **kwargs) -> None:
        del kwargs
        self.env = env
        self.work = env.handles.work_arm or arm_slots(env)[0]
        self.support = env.handles.support_arm
        self.dof = len(env.idle_action(self.MODE)) // len(arm_slots(env))
        self.passes = int(solution_param(env, "draw-passes", PASSES))
        self.grip_x = float(solution_param(env, "grasp-station", GRIP_X))
        self.tilt = math.radians(float(solution_param(env, "cut-tilt", CUT_TILT)))
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        env = self.env
        self.held = {n: self._q(n) for n in arm_slots(env)}
        self.grip = dict.fromkeys(arm_slots(env), OPEN)
        self.seed = {n: np.asarray(env.state.joint_q.numpy(), dtype=np.float64) for n in arm_slots(env)}
        self.refused: list[str] = []
        self.notes: list[str] = []
        self.press_n = 0.0
        self.pin_n = 0.0
        self.flap_lifted = 0.0
        self.flap_flipped = 0.0
        self.phase = 0.0
        self._steps = self._choreography()

    def _q(self, name):
        ik = self.env.arm(name)
        return np.asarray(self.env.state.joint_q.numpy(), dtype=np.float64)[ik.arm_coords]

    def row_extras(self) -> dict[str, float]:
        return {
            "press_n": float(self.press_n),
            "pin_n": float(self.pin_n),
            "legs_refused": float(len(self.refused)),
            "phase": float(self.phase),
            # The flap at its two stages, so a trace says whether the fingertip
            # failed to lift it or the flip failed to carry it over.
            "flap_lifted": float(self.flap_lifted),
            "flap_flipped": float(self.flap_flipped),
        }

    # ------------------------------------------------------------------ drive
    def act(self) -> np.ndarray:
        try:
            next(self._steps)
        except StopIteration:
            pass
        return self._action()

    def _action(self) -> np.ndarray:
        a = np.asarray(self.env.idle_action(self.MODE), dtype=np.float64)
        for name in arm_slots(self.env):
            s = arm_slot(self.env, name) * self.dof
            a[s : s + self.dof - 1] = self.held[name][: self.dof - 1]
            a[s + self.dof - 1] = self.grip[name]
        return a

    # ------------------------------------------------------------------- legs
    def _hold(self, n: int):
        for _ in range(n):
            yield

    def _leg(
        self,
        name,
        target,
        quat,
        *,
        speed=0.06,
        grip=None,
        label="",
        gate=True,
        stride=2,
        max_pos=MAX_POS,
        max_rot=MAX_ROT,
    ):
        """Solve the whole leg, gate it, then drive it one waypoint per step.

        The orientation is slerped from where the hand is to ``quat``, so a leg
        that turns the wrist (the tilt into the cutting hold, the lean into the
        hook) turns it over the leg rather than in one step. The IK is solved
        every ``stride`` waypoints and the joint targets interpolated between:
        a millimetre of straight-line joint interpolation is invisible, and a
        full-pose solve per control step doubled the wall time of every leg.

        `gate` only where something is IN THE HAND or the leg ends in a contact.
        A leg starting from the arm's home pose cannot hold its orientation at
        its first waypoint -- the residual runs to 150 mm and 65 degrees and
        shrinks as the arm converges -- so gating a free approach refuses every
        one of them. With nothing in the hand that convergence is harmless."""
        ik = self.env.arm(name)
        start, q_start = ik.ee_pose(self.env.state)
        start = np.asarray(start, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)
        q_start = np.asarray(q_start, dtype=np.float64)
        turn = 2.0 * math.acos(
            min(1.0, abs(float(np.dot(q_start / np.linalg.norm(q_start), quat / np.linalg.norm(quat)))))
        )
        n = max(2, int(np.linalg.norm(target - start) / speed * 60.0), int(turn / math.radians(60.0) * 60.0))
        knots = sorted({*range(stride, n + 1, stride), n})
        sols, worst_p, worst_r, seed = {}, 0.0, 0.0, self.seed[name]
        for k in knots:
            f = k / n
            step = start + (target - start) * f
            q_k = _slerp(q_start, quat, f)
            sol = ik.solve_q_full(seed, step, q_k)
            p, r = ik.pose_errors(sol, step, q_k)
            worst_p, worst_r = max(worst_p, p), max(worst_r, r)
            sols[k] = np.asarray(sol, dtype=np.float64)
            seed = sol
        if gate and (worst_p > max_pos or worst_r > max_rot):
            # Refuse rather than drive it. An unreachable waypoint is a lurch,
            # and a lurch with something in the hand is a thrown part.
            self.refused.append(f"{label or name}: {worst_p * 1e3:.1f} mm / {math.degrees(worst_r):.1f} deg")
            return
        if grip is not None:
            self.grip[name] = grip
        prev_k, prev_q = 0, np.asarray(self.seed[name], dtype=np.float64)
        for k in knots:
            for j in range(prev_k + 1, k + 1):
                f = (j - prev_k) / (k - prev_k)
                q = prev_q + (sols[k] - prev_q) * f
                self.held[name] = q[list(ik.arm_coords)]
                yield
            self.seed[name] = sols[k]
            prev_k, prev_q = k, sols[k]

    def _across_quat(self, name, at, along: float, wrist_toward=None):
        """Top-down, with the JAWS ACROSS a bar whose long axis is at world yaw
        ``along`` [rad].

        ``_down_quat``-style -- whatever wrist roll the IK happens to return for
        a downward tool axis -- was the shipped task's biggest lie: the YAM's
        fingertips are long blades, closing them on a baton does not pinch it
        where it lies, it SWEEPS it into line with the pads, and the knife left
        every pick 40-49 degrees yawed with the blade off the seam by
        ``edge_ahead * sin(yaw)``. So the wrist is turned to the bar rather than
        left where the solver put it. The pads are symmetric, so alignment is
        modulo 180 degrees and the turn is taken the short way round.
        """
        ik = self.env.arm(name)
        q0, _ = ik.solve_best([self.seed[name]], np.asarray(at), (0.0, 0.0, -1.0))
        _, home = ik.fk_ee(np.asarray(q0, dtype=np.float32))
        turn = _half_wrap(float(along) - _yaw_of(home))
        quat = quat_multiply([0.0, 0.0, math.sin(turn / 2), math.cos(turn / 2)], home)
        if wrist_toward is None:
            return quat
        # Of the two holds that pinch a bar equally well, the one with the
        # wrist on the side of ``wrist_toward``. On ``yam`` the grip point is
        # 44 mm off the wrist axis, so the two holds put the arm in different
        # places, and only one of them can follow the tilted draw: with the
        # wrist on the far side the wrist roll runs into its 2.09 rad stop at
        # the far end of the tape (24-29 mm off over the draw, measured), with
        # it on the carton's side every station solves to 0.0 mm.
        flipped = quat_multiply([0.0, 0.0, 1.0, 0.0], quat)
        toward = np.asarray(wrist_toward, dtype=np.float64) - np.asarray(at, dtype=np.float64)
        toward[2] = 0.0

        def wrist_side(q) -> float:
            offset = _rot(q) @ self._wrist_from_tcp(name)
            offset[2] = 0.0
            return float(np.dot(offset, toward))

        return flipped if wrist_side(flipped) > wrist_side(quat) + 1.0e-6 else quat

    def _wrist_from_tcp(self, name) -> np.ndarray:
        """The wrist body's origin from the TCP, in the TCP's own frame [m]."""
        hand = self.env.scene.embodiment.hands.get(name)
        offset = np.asarray(hand.tcp_offset, dtype=np.float64)
        rot = _rot(np.asarray(hand.tcp_offset_rot, dtype=np.float64))
        return rot.T @ -offset

    def _branch(self, name, targets, quat, tries: int = 24):
        """The arm configuration from which a chain of poses solves best.

        The full-pose solve is local: chained from wherever the arm stands, it
        stays on that arm's branch, and the pick leaves the arm on one from which
        the tilted hold cannot follow the tape to its far end (residuals of
        20-30 mm, measured), while another branch follows all of it at 0.0 mm.
        So the chain is tried from the current configuration and from ``tries``
        deterministic perturbations of it, and the best start is returned with
        its worst residual."""
        ik = self.env.arm(name)
        coords = list(ik.arm_coords)
        base = np.asarray(self.seed[name], dtype=np.float64)
        rng = np.random.default_rng(0)
        scale = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0])[: len(coords)]
        best = None
        for k in range(tries + 1):
            seed = base.copy()
            if k:
                seed[coords] = seed[coords] + rng.uniform(-math.pi, math.pi, len(coords)) * scale
            first, worst, chained = None, 0.0, seed
            for target in targets:
                sol = np.asarray(ik.solve_q_full(chained, target, quat), dtype=np.float64)
                p, r = ik.pose_errors(sol, target, quat)
                worst = max(worst, float(p) + 0.01 * math.degrees(float(r)))
                first = sol if first is None else first
                chained = sol
            if best is None or worst < best[0]:
                best = (worst, first)
        return best[1], best[0]

    def _joint_leg(self, name, q_target, steps: int):
        """Drive the arm to a joint configuration, linearly in joint space."""
        ik = self.env.arm(name)
        q0 = np.asarray(self.seed[name], dtype=np.float64)
        q1 = np.asarray(q_target, dtype=np.float64)
        for j in range(1, steps + 1):
            q = q0 + (q1 - q0) * j / steps
            self.held[name] = q[list(ik.arm_coords)]
            yield
        self.seed[name] = q1

    @staticmethod
    def _lean(quat, axis_world, angle: float, want_down) -> np.ndarray:
        """``quat`` turned about a WORLD axis by ``angle`` either way, whichever
        way sends the world vector ``want_down`` downward.

        The tilt into the cutting hold and the lean into the hook are both
        "rotate the hand about a horizontal axis so that some part of what it
        carries goes down", and which sign does that depends on the roll the
        pick happened to settle on. Measured rather than assumed, per leg.
        """
        best, best_z = None, math.inf
        for s in (1.0, -1.0):
            q = quat_multiply(quat_from_axis_angle(axis_world, s * angle), quat)
            r = quat_from_axis_angle(axis_world, s * angle)
            z = float((_rot(r) @ np.asarray(want_down, dtype=np.float64))[2])
            if z < best_z:
                best, best_z = q, z
        return np.asarray(best, dtype=np.float64)

    # -------------------------------------------------------------- measuring
    def _blade_force(self) -> float:
        """Blade against the flaps' colliders, the way a wrist force sensor
        would read it. The BLADE alone against the FLAPS alone (boards and
        strips): ``pairwise_force`` sums impulse vectors, so folding in the body
        or the carton base brings back the cancellation that makes a balanced
        pair read zero."""
        probe = self.env.contacts
        if probe is None:
            return 0.0
        # A wave hands out a ``ContactView`` -- one world's rows of a read the
        # fleet already took at its snapshot -- which has the query surface and
        # not the refresh: the wave owns when the table is read, and a routine
        # forcing one here would be re-reading the device from a graph thread
        # while the fleet is stepping. The single-world probe still refreshes on
        # demand, which is what it always did.
        refresh = getattr(probe, "update", None)
        if callable(refresh):
            refresh(step=None)
        return float(abs(probe.pairwise_force(self._blade, self._flaps)[2]))

    def _body_force(self) -> float:
        """The cutter's BODY (everything but the blade) against the flaps [N].

        The other end of the press window: the blade can only be driven into
        the tape until the tool's own nose corner reaches it, and past that the
        blade reads nothing while the body reads twenty newtons."""
        probe = self.env.contacts
        if probe is None:
            return 0.0
        probe.update(step=None)
        return float(abs(probe.pairwise_force(self._body_shapes, self._flaps)[2]))

    def _tip_offset(self) -> np.ndarray:
        """World vector from the work TCP to the blade's TIP, measured with the
        knife already in the hand and the arm still.

        The spec's arithmetic says where the tip is relative to the grasp; the
        pads decide where the tool actually settled. Measured on the shipped
        grasp the edge came out up to 11 mm off the seam's centreline with the
        TCP driven exactly down it -- against a 6 mm kerf. So every pass aims
        with the blade it is holding: one reading, taken at the top of the pass
        with the arm stopped, used for every waypoint of it."""
        ik = self.env.arm(self.work)
        tcp, _ = ik.ee_pose(self.env.state)
        pose = np.asarray(self.env.state.body_q.numpy()[self._knife_body], dtype=np.float64)
        tip = pose[:3] + _rot(pose[3:7]) @ self._tip_local
        return np.asarray(tip, dtype=np.float64) - np.asarray(tcp, dtype=np.float64)

    def _pad_end(self, quat) -> np.ndarray:
        """World offset from the TCP to the END of the OUTER FACE of whichever
        fingertip stands at +x, jaws open, for a hand held at ``quat``.

        In the grasp-site frame the jaws close along +-y and the approach is
        +z; measured on the built hand, the pads end 10.5 mm past the TCP along
        the approach (checked against the mesh: a pad commanded to 0.8420 read
        0.8415) and their outer faces stand :data:`PAD_OUT` off the axis. The
        outer face is the one that goes under the far flap's edge, so it is the
        one that is placed. Which of the two tips is the +x one depends on the
        roll, so both are tried."""
        R = _rot(quat)
        ends = [R @ np.array([0.0, s * PAD_OUT, 0.0105]) for s in (1.0, -1.0)]
        return max(ends, key=lambda e: float(e[0]))

    def _resolve(self) -> None:
        """Indices, and the lengths the choreography is written against.

        The lengths come from the ASSETS -- ``blade_edge_local`` for the tip,
        ``carton_frame`` for the seam and the flaps, the seam spec for the
        tape -- because each is already the one place it is derived, and a
        routine that restates them is a routine that keeps aiming where the
        knife used to have a blade.
        """
        env = self.env
        # The model indices are resolved against, which replicated is the
        # TEMPLATE and not the one that steps. A name lookup over the stepping
        # model finds one match per world, and the shape sets below would then
        # name every world's blade -- which the per-world contact view rebases
        # to the template and cannot tell apart from this world's own.
        model = env.scene.index_model
        labels = list(model.shape_label)
        body = model.shape_body.numpy()
        knife = env.handles.entity("knife")
        p0, direction, edge_len = blade_edge_local(KNIFE)
        self._tip_local = np.asarray(p0, dtype=np.float64) + float(edge_len) * np.asarray(direction, dtype=np.float64)
        self._nose_local = np.asarray(direction, dtype=np.float64)
        f = carton_frame(CARTON)
        self._frame = f
        tape = seal_tape()
        self.tape_len = float(tape.tape_len)
        self.tape_thickness = float(tape.tape_thickness)
        self._frame_travel = float(tape.flap_travel)
        self._blade = [i for i, lab in enumerate(labels) if lab.endswith("knife_blade")]
        flap_bodies = [
            i for i, lab in enumerate(list(model.body_label)) if lab.endswith(("carton_flap_near", "carton_flap_far"))
        ]
        self._flaps = [int(s) for s in np.flatnonzero(np.isin(body, flap_bodies))]
        self._knife_body = knife.body
        self._carton_body = env.handles.entity("carton").body
        self._carton_shapes = [int(s) for s in np.flatnonzero(np.isin(body, [self._carton_body, *flap_bodies]))]
        self._knife_shapes = [int(s) for s in np.flatnonzero(body == self._knife_body)]
        self._body_shapes = [s for s in self._knife_shapes if not labels[s].endswith("knife_blade")]
        self._mat_shapes = [i for i, lab in enumerate(labels) if lab.split("/")[-1] == "mat"]
        # The pads by the hand's own finger bodies, not by a link name: the i2rt
        # hand calls them `{arm}_tip_*` and the arena hand `{arm}_link_*_finger`.
        self._tips = {
            name: [
                int(s)
                for s in np.flatnonzero(
                    np.isin(body, [b for side in env.scene.embodiment.arms[name].finger_bodies for b in side])
                )
            ]
            for name in arm_slots(env)
        }

    # ---------------------------------------------------------- choreography
    def _choreography(self):
        """The whole take, in the clip's order. Each phase is its own generator
        so a harness can drive one of them alone against a prepared scene."""
        self._survey()
        yield from self._pick()
        yield from self._pin()
        yield from self._carry()
        yield from self._cut()
        yield from self._stow()
        yield from self._hook_far()
        yield from self._release()
        yield from self._hold(120)
        self.phase = 8.0

    def _survey(self) -> None:
        """Read the carton and the knife as they stand, once, and derive every
        world point the phases aim at from the assets' own frames."""
        env = self.env
        self._resolve()
        knife = np.asarray(env.state.body_q.numpy()[self._knife_body], dtype=np.float64)
        R_k = _rot(knife[3:7])

        self._read_carton()

        # --- the pick, along the KNIFE's own long axis, not along world y ------
        # Writing it this way is what makes the grasp survive a knife that
        # spawned turned, and it is the same yaw the wrist is set to.
        nose_world = R_k @ self._nose_local
        self.knife_yaw = math.atan2(float(nose_world[1]), float(nose_world[0]))
        self.grip_at = knife[:3] + R_k @ np.array([self.grip_x, 0.0, 0.0]) + np.array([0.0, 0.0, GRIP_DZ])
        self.kq = self._across_quat(self.work, self.grip_at, self.knife_yaw, wrist_toward=self._carton[:3])
        # The closing axis in world is the grasp frame's y; the tilt is about it,
        # whichever way sends the nose down.
        self.cq = self._lean(self.kq, _rot(self.kq)[:, 1], self.tilt, nose_world)
        self.pq = None

    def _read_carton(self) -> None:
        """Where the seam, the tape and the far flap are RIGHT NOW, from the
        carton's live pose.

        Called once at the survey and again at the top of every draw pass and
        before the flap is hooked, because the carton moves: the pin is a
        down-force and its lateral hold is only ``mu * (weight + pin)``, so the
        draw's own drag walks the box a few millimetres a pass. Read once at the
        start, a three-pass draw ends up aiming its last stroke at a seam that
        has moved out from under it -- measured, a run that read the carton once
        finished 30 mm adrift with ``cut_fraction`` 0.604, and the shipped
        150 mm knife failed the same way with five passes at 0.795.
        """
        f = self._frame
        carton = np.asarray(self.env.state.body_q.numpy()[self._carton_body], dtype=np.float64)
        R_c = _rot(carton[3:7])
        self._carton, self._R_c = carton, R_c
        seam_mid = carton[:3] + R_c @ (
            np.asarray(f["seam_p0"]) + 0.5 * float(f["seam_len"]) * np.asarray(f["seam_dir"])
        )
        along = R_c @ np.asarray(f["seam_dir"])
        along[2] = 0.0
        along /= np.linalg.norm(along)
        self.draw = along if float(along[1]) > 0.0 else -along  # toward +y: the right end to the left arm
        self.tape_top = float(seam_mid[2]) + self.tape_thickness
        self.seam_mid = seam_mid
        self.tape_start = seam_mid - 0.5 * self.tape_len * self.draw
        self.tape_end = seam_mid + 0.5 * self.tape_len * self.draw
        self.tape_start[2] = self.tape_end[2] = self.tape_top
        far = R_c @ np.array([0.0, 1.0, 0.0])  # the far flap's side, world
        far[2] = 0.0
        self.far = far / np.linalg.norm(far)
        self.hinge_far = carton[:3] + R_c @ np.array([0.0, float(f["hinge_y"]), float(f["rim_z"])])
        self.flap_w = float(f["flap_free_y"])

    def _pick(self):
        """Top-down over the cutter, jaws across its body, shut, lift, tilt."""
        grip, kq, cq = self.grip_at, self.kq, self.cq
        self.phase = 1.0
        yield from self._leg(
            self.work, grip + np.array([0.0, 0.0, 0.10]), kq, grip=OPEN, gate=False, speed=0.18, label="approach"
        )
        yield from self._hold(15)
        yield from self._leg(self.work, grip, kq, speed=0.04, gate=False, label="descend to the body")
        yield from self._hold(15)
        self.grip[self.work] = SHUT
        yield from self._hold(60)
        yield from self._leg(self.work, grip + np.array([0.0, 0.0, 0.10]), kq, label="lift")
        yield from self._hold(20)
        yield from self._leg(self.work, grip + np.array([0.0, 0.0, 0.10]), cq, speed=0.05, label="tilt")
        yield from self._hold(20)

    def _carry(self):
        """Over the right end of the tape, at the carry height, tilted. The tip
        is measured here, in hand and at rest, and that reading is what every
        later reading is checked against: the knife pitches 15-20 degrees
        further nose-down in the jaws than the tilt commanded, and that is a
        settled pose to aim with rather than a slip to refuse."""
        off = self._tip_offset()
        self._off0 = off.copy()
        # The grasp frame's z is the APPROACH -- down, for a top-down pick -- and
        # its x the knife's axis; the knife's own z is up. Measured on the built
        # hand: the knife's x reads (1, 0, 0) in the grasp frame and its z
        # (0, 0, -1), so the spec's tip goes over with its z negated.
        local = self._tip_local - np.array([self.grip_x, 0.0, GRIP_DZ])
        spec_off = _rot(self.cq) @ np.array([local[0], -local[1], -local[2]])
        self.notes.append(f"tip in hand {np.round(off * 1e3, 1)} mm vs the spec's {np.round(spec_off * 1e3, 1)}")
        self.phase = 2.0
        # Onto the arm branch the whole cut solves from, before anything is
        # chained: over the tape's start, down the draw, and up off its end.
        carry = np.array([0.0, 0.0, CARRY_LIFT])
        targets = [self.tape_start - off + carry]
        targets += [self.tape_start + (self.tape_end - self.tape_start) * k / 10 - off for k in range(11)]
        targets.append(self.tape_end - off + carry)
        q_over, worst = self._branch(self.work, targets, self.cq)
        self.notes.append(f"cutting branch: worst {worst * 1e3:.1f} (mm + deg/100)")
        q_now = np.asarray(self.seed[self.work], dtype=np.float64)
        span = float(np.max(np.abs((q_over - q_now)[list(self.env.arm(self.work).arm_coords)])))
        if span > math.radians(5.0):
            yield from self._joint_leg(self.work, q_over, max(60, int(span / math.radians(30.0) * 60.0)))
            yield from self._hold(20)
        yield from self._leg(self.work, self.tape_start - off + carry, self.cq, speed=0.08, label="carry")
        yield from self._hold(20)

    def _pin(self):
        """The support hand's closed jaws pressed onto the near flap, 15 mm in
        from the +y end wall and 11 mm in from the hinge.

        Pick, THEN pin, THEN carry and draw. The pin goes on before the knife
        comes over the tape, and it goes on the near flap's OUTBOARD band rather
        than beside the seam, because the hand that carries the tilted knife
        stands over the tape's middle with its carriage reaching 70 mm toward
        +y and 25 mm toward -x of the seam: a pin planned at the seam's x and
        the flap's end put the support hand's carriage 45 mm into that space,
        and the first thing it did was knock the knife level in the jaws.
        Inboard of the hinge the two carriages are 25 mm apart in x and the
        pinned hand's top is 48 mm under the working hand's fingertips at the
        seat.

        The descent is FORCE-CHECKED, in millimetre steps, because the support
        arm at this reach sags ~10 mm under its own weight at the 1600 gain: a
        pin commanded 4 mm through the flap's top stopped 5 mm above it and held
        nothing."""
        if not self.support:
            return
        f = self._frame
        pin_local = np.array(
            [-0.5 * float(f["seam_len"]) + 0.015, -(float(f["hinge_y"]) - PIN_INBOARD), float(f["seam_z"])]
        )
        pin = self._carton[:3] + self._R_c @ pin_local
        touch_z = float(pin[2]) + 0.0105  # the pads' ends on the flap's top face
        self.pq = self._across_quat(self.support, pin + np.array([0.0, 0.0, 0.06]), math.pi / 2.0)
        self.phase = 3.0
        yield from self._leg(
            self.support,
            np.array([pin[0], pin[1], touch_z + 0.08]),
            self.pq,
            grip=SHUT,
            gate=False,
            speed=0.16,
            label="over the far end",
        )
        yield from self._hold(15)
        yield from self._leg(
            self.support,
            np.array([pin[0], pin[1], touch_z + 0.006]),
            self.pq,
            speed=0.04,
            gate=False,
            label="down to the flap",
        )
        yield from self._hold(10)
        self.pin_n = yield from self._press_down(
            self.support, pin[:2], touch_z + 0.006, self.pq, target_n=PIN_FORCE, max_travel=0.030, label="pin"
        )
        yield from self._hold(15)

    def _press_down(self, name, xy, z_start: float, quat, *, target_n: float, max_travel: float, label: str):
        """Descend in millimetre steps until the hand's pads read ``target_n``
        of contact against the carton, or ``max_travel`` is spent. Returns the
        force it stopped at. surface_cleaning's ``_press``, for the same reason:
        a commanded depth is not a commanded force when the arm sags."""
        z, spent, force = float(z_start), 0.0, 0.0
        while spent < max_travel:
            z -= 0.001
            spent += 0.001
            yield from self._leg(name, [xy[0], xy[1], z], quat, speed=0.02, label="", gate=False)
            yield from self._hold(10)
            force = self._hand_force(name)
            if force >= target_n:
                break
        self.notes.append(f"{label}: {force:.1f} N at {spent * 1e3:.0f} mm down")
        return force

    def _knife_on_mat(self) -> float:
        """The cutter against the mat [N], for the set-down."""
        probe = self.env.contacts
        if probe is None:
            return 0.0
        probe.update(step=None)
        return float(np.linalg.norm(probe.pairwise_force(self._knife_shapes, self._mat_shapes)))

    def _hand_force(self, name) -> float:
        """The named hand's fingertips against the carton (base and flaps), as
        a magnitude. A closed pair of pads pressing straight down has both
        forces pointing the same way, so the net vector is the press."""
        probe = self.env.contacts
        if probe is None:
            return 0.0
        probe.update(step=None)
        return float(np.linalg.norm(probe.pairwise_force(self._tips[name], self._carton_shapes)))

    def _cut(self):
        """Seat, and draw one way at a time.

        On a return stroke the blade rides over what it has already opened and
        reads no force at all -- measured: a first pass cut 0.705, its return
        added 0.022 and 31 mm of skate. So the knife is lifted, carried back
        and re-seated. And the aim is re-measured at the top of EVERY pass: the
        knife swings in the jaws while the arm moves."""
        env = self.env
        ik = env.arm(self.work)
        cq, draw = self.cq, self.draw
        carry = np.array([0.0, 0.0, CARRY_LIFT])
        self.phase = 4.0
        for _ in range(self.passes):
            self._read_carton()
            draw = self.draw
            off = self._tip_offset()
            drift = off - self._off0
            if float(np.max(np.abs(drift))) > 0.025 or float(off[2]) > -0.020:
                # Not a pose to correct for: the knife has moved in the jaws
                # since it was measured after the tilt, or its tip is no longer
                # well below the hand -- a pick that did not take the body.
                self.refused.append(
                    f"grasp: tip moved {np.round(drift * 1e3, 1)} mm in the jaws (now {np.round(off * 1e3, 1)})"
                )
                return
            seat = self.tape_start + DRAW_INSET * draw - off + np.array([0.0, 0.0, SEAT_CLEAR])
            yield from self._leg(self.work, seat, cq, speed=0.03, label="seat the blade")
            yield from self._hold(15)
            # The press, by force: down a millimetre at a time until the BLADE
            # reads SEAT_FORCE against the tape -- or until the tool's own BODY
            # lands on it, which is the other end of a window only a few
            # millimetres wide and the thing that ends the cut if it is missed.
            z, spent, force, landed = float(seat[2]), 0.0, 0.0, False
            while spent < SEAT_TRAVEL:
                z -= 0.001
                spent += 0.001
                yield from self._leg(self.work, [seat[0], seat[1], z], cq, speed=0.02, label="", gate=False)
                # TEN control steps, not four: at the 1600 gain this arm is still
                # sagging into a millimetre step four steps after it was
                # commanded, and a search that reads the force before the arm
                # has arrived walks straight past its target.
                yield from self._hold(10)
                force = self._blade_force()
                if self._body_force() > BODY_TOUCH:
                    landed = True
                    z += 0.002
                    yield from self._leg(self.work, [seat[0], seat[1], z], cq, speed=0.02, label="", gate=False)
                    yield from self._hold(10)
                    force = self._blade_force()
                    break
                if force >= SEAT_FORCE:
                    break
            self.press_n = force
            self.notes.append(
                f"seat: {force:.1f} N at {spent * 1e3:.0f} mm down" + (" (body landed, backed off)" if landed else "")
            )
            # The draw FOLLOWS THE FORCE, every few control steps, rather than
            # being trimmed once a quarter. The arm sags differently along the
            # stroke -- measured, a seat that reads 4.5 N at the near end reads
            # 0.0 N a third of the way along and 7 N two thirds -- and the law
            # cuts nothing at all under the gate, so a press that dips for
            # 20 mm leaves 20 mm of tape. That is the scattered quarter behind
            # a scored `cut_fraction` of 0.75 with a `cut_run` of 0.21: both
            # rows parted equally (near 0.250, far 0.225), so the gaps are
            # along the stroke and not to one side of it.
            # BACK first, then forward. The law's window runs from the tip
            # 15 mm toward the tool's NOSE and the nose points against the
            # draw, so a forward stroke can never reach the tape behind its own
            # start: a run whose passes all began 10 mm in finished at
            # cut_fraction 0.917 with exactly the two near-end stations intact.
            # Backing up from the seat covers them, and the tape there is still
            # whole so the blade has something to read.
            start = np.array([seat[0], seat[1], z])
            run = self.tape_len - 2.0 * DRAW_INSET
            trim = yield from self._stroke(ik, start, -draw, DRAW_BACK, cq, 0.0)
            back = start - draw * DRAW_BACK
            trim = yield from self._stroke(ik, back, draw, run + DRAW_BACK, cq, trim)
            yield from self._leg(self.work, self.tape_end - off + carry, cq, speed=0.06, label="lift clear")
            yield from self._hold(15)
            yield from self._leg(self.work, self.tape_start - off + carry, cq, speed=0.10, label="back to the start")
            yield from self._hold(15)

    def _stroke(self, ik, frm, direction, length: float, quat, trim: float):
        """One force-following stroke of the blade along the tape.

        The height is corrected every :data:`DRAW_POLL` control steps from the
        field's own blade-against-flaps force: down :data:`DRAW_STEP` while the
        press is under target, up while it is over -- or the moment the tool's
        own BODY touches, which is the ceiling. Returns the trim it ended on,
        so the next stroke starts where this one left off.
        """
        worst, best = 99.0, 0.0
        n = max(2, int(length / DRAW_SPEED * 60.0))
        for k in range(n):
            p = frm + direction * length * (k + 1) / n + np.array([0.0, 0.0, trim])
            sol = ik.solve_q_full(self.seed[self.work], p, quat)
            self.seed[self.work] = sol
            self.held[self.work] = np.asarray(sol)[ik.arm_coords]
            yield
            if k % DRAW_POLL:
                continue
            force, body = self._blade_force(), self._body_force()
            worst, best = min(worst, force), max(best, force)
            if body > BODY_TOUCH or force > DRAW_CEILING * SEAT_FORCE:
                trim += DRAW_BACKOFF * DRAW_STEP
            elif force < SEAT_FORCE:
                trim -= DRAW_STEP
            trim = float(np.clip(trim, -DRAW_SINK, 0.006))
        self.press_n = max(self.press_n, float(best))
        self.notes.append(f"stroke {length * 1e3:.0f} mm: {worst:.1f}-{best:.1f} N, trim {trim * 1e3:+.1f} mm")
        return trim

    def _stow(self):
        """The cutter back on the mat where it was: level, down, open, away."""
        grip, kq, cq = self.grip_at, self.kq, self.cq
        self.phase = 5.0
        yield from self._leg(self.work, grip + np.array([0.0, 0.0, 0.10]), cq, speed=0.08, label="back over the mat")
        yield from self._hold(15)
        yield from self._leg(self.work, grip + np.array([0.0, 0.0, 0.10]), kq, speed=0.05, label="level")
        yield from self._hold(15)
        yield from self._leg(self.work, grip + np.array([0.0, 0.0, 0.012]), kq, speed=0.04, label="down to the mat")
        yield from self._hold(10)
        # Force-checked: the cutter may sit lower in the jaws than it was
        # picked (it settles under the press), and a set-down commanded to the
        # pick height then drives it into the mat at a hundred newtons.
        z, spent = float(grip[2]) + 0.012, 0.0
        while spent < 0.020:
            z -= 0.001
            spent += 0.001
            yield from self._leg(self.work, [grip[0], grip[1], z], kq, speed=0.02, label="", gate=False)
            yield from self._hold(4)
            if self._knife_on_mat() >= 0.3:
                break
        self.notes.append(f"set down: {self._knife_on_mat():.1f} N on the mat at {spent * 1e3:.0f} mm down")
        yield from self._hold(10)
        self.grip[self.work] = OPEN
        yield from self._hold(30)
        yield from self._leg(self.work, grip + np.array([0.0, 0.0, 0.10]), kq, speed=0.08, gate=False, label="retreat")
        yield from self._hold(10)

    def _hook_far(self):
        """Hook the far flap up with a fingertip, then flip it past vertical.

        Jaws open and closing along the flaps' axis, so the +x fingertip is a
        7 mm blade that fits the 12 mm join. Top-down and UNLEANED, so its end
        is a horizontal face whose contact normal is vertical -- see
        :data:`HOOK_UNDER` for the leaned version that pushed on the flap's own
        edge face for a whole arc and opened nothing.

        Four legs and then three: down the join to below the board, +x under
        it, up along the flap's own arc to :data:`HOOK_LIFT`, withdraw; then
        over the raised flap's -x face and a drive +x that carries it past
        vertical onto its stop.
        """
        self._read_carton()
        far, seam_mid = self.far, self.seam_mid
        f = self._frame
        self.phase = 6.0
        # 20 mm outside the taped run: the join is open board there, no strip.
        gap_y = float(self.tape_start[1]) - 0.020
        rim_z = float(self._carton[2]) + float(f["rim_z"])
        edge_x = float(seam_mid[0]) + 0.5 * float(f["seam_gap"])
        up = np.array([0.0, 0.0, 1.0])
        hq0 = self._across_quat(self.work, np.array([float(seam_mid[0]), gap_y, rim_z + 0.10]), math.pi / 2.0)
        # Leaned so the pad's BOTTOM corner leads in +x -- see HOOK_LEAN, and
        # note which way: the world +x vector is the one sent downward.
        hq = self._lean(hq0, np.array([0.0, 1.0, 0.0]), math.radians(HOOK_LEAN), -far)
        end0, end = self._pad_end(hq0), self._pad_end(hq)
        over = np.array([float(seam_mid[0]), gap_y, rim_z + 0.09])
        into = np.array([float(seam_mid[0]), gap_y, rim_z - HOOK_DEPTH])
        under = np.array([edge_x + HOOK_UNDER, gap_y, rim_z - HOOK_DEPTH])
        yield from self._leg(self.work, over - end0, hq0, grip=OPEN, gate=False, speed=0.12, label="over the join")
        yield from self._hold(15)
        yield from self._leg(self.work, over - end, hq, gate=False, speed=0.08, label="lean")
        yield from self._hold(10)
        yield from self._leg(self.work, into - end, hq, speed=0.03, label="into the join")
        yield from self._hold(15)
        yield from self._leg(self.work, under - end, hq, speed=0.02, label="under the board")
        yield from self._hold(15)
        # The lift: the pad's bottom corner follows the point of the flap it is
        # under, round the flap's own hinge, while the wrist gives its lean back.
        r = self.flap_w - HOOK_UNDER
        for attempt in range(HOOK_TRIES):
            if attempt:
                # The board can slip off a fingertip part way up, and the box
                # may have moved under it. Re-read, go back down and try again
                # rather than flipping a flap that is still shut.
                self._read_carton()
                far, seam_mid = self.far, self.seam_mid
                edge_x = float(seam_mid[0]) + 0.5 * float(f["seam_gap"])
                rim_z = float(self._carton[2]) + float(f["rim_z"])
                yield from self._leg(
                    self.work,
                    np.array([float(seam_mid[0]), gap_y, rim_z + 0.06]) - end0,
                    hq0,
                    gate=False,
                    speed=0.12,
                    label="back over the join",
                )
                yield from self._leg(
                    self.work,
                    np.array([float(seam_mid[0]), gap_y, rim_z - HOOK_DEPTH]) - end,
                    hq,
                    speed=0.03,
                    gate=False,
                    label="into the join again",
                )
                yield from self._leg(
                    self.work,
                    np.array([edge_x + HOOK_UNDER, gap_y, rim_z - HOOK_DEPTH]) - end,
                    hq,
                    speed=0.02,
                    gate=False,
                    label="under the board again",
                )
                yield from self._hold(10)
            for th in np.linspace(0.0, math.radians(HOOK_LIFT), 16):
                f_th = float(th) / math.radians(HOOK_LIFT)
                q_th = self._lean(hq0, np.array([0.0, 1.0, 0.0]), math.radians(HOOK_LEAN) * (1.0 - f_th), -far)
                c = self.hinge_far + r * (-math.cos(th) * far + math.sin(th) * up)
                yield from self._leg(
                    self.work,
                    np.array([float(c[0]), gap_y, float(c[2])]) - self._pad_end(q_th),
                    q_th,
                    speed=0.04,
                    label="lift the flap",
                    stride=1,
                    max_pos=0.008,
                    max_rot=math.radians(12.0),
                )
            self.flap_lifted = self._flap_far()
            self.notes.append(f"lift {attempt}: flap_far_open {self.flap_lifted:.3f}")
            if self.flap_lifted > 0.15:
                break
        yield from self._hold(20)
        yield from self._leg(
            self.work,
            np.array([float(seam_mid[0]) - 0.05, gap_y, rim_z + 0.12]) - end0,
            hq0,
            speed=0.10,
            gate=False,
            label="withdraw",
        )
        yield from self._hold(15)

        # The flip: onto the raised flap's -x face, then drive +x past vertical.
        def on_flap(deg: float) -> np.ndarray:
            t = math.radians(deg)
            c = self.hinge_far + (self.flap_w - FLIP_FROM) * (-math.cos(t) * far + math.sin(t) * up)
            return np.array([float(c[0]), gap_y, float(c[2])])

        # Where the flap ACTUALLY is, not where the arc asked it to be: the
        # fingertip can slip out from under a board it has lifted part way, and
        # a flip aimed at 55 degrees when the flap is at 12 pushes air.
        lifted = max(12.0, math.degrees(self._flap_far() * float(self._frame_travel)))
        self.notes.append(f"flip from {lifted:.0f} deg")
        push = on_flap(lifted)
        yield from self._leg(
            self.work,
            push - end0 + np.array([-0.06, 0.0, 0.06]),
            hq0,
            gate=False,
            speed=0.12,
            label="over the raised flap",
        )
        yield from self._hold(10)
        yield from self._leg(
            self.work, push - end0 + np.array([-0.014, 0.0, 0.0]), hq0, speed=0.05, gate=False, label="behind the flap"
        )
        yield from self._hold(10)
        for k in range(1, 13):
            yield from self._leg(
                self.work,
                on_flap(lifted + k * (105.0 - lifted) / 12.0) - end0,
                hq0,
                speed=0.05,
                gate=False,
                label="flip it over",
                stride=1,
            )
        self.flap_flipped = self._flap_far()
        self.notes.append(f"flipped: flap_far_open {self.flap_flipped:.3f}")
        yield from self._hold(20)
        yield from self._leg(
            self.work,
            np.array([float(seam_mid[0]) - 0.07, gap_y, rim_z + 0.14]) - end0,
            hq0,
            speed=0.12,
            gate=False,
            label="retreat",
        )

    def _flap_far(self) -> float:
        """How far open the far flap is, as a fraction of its travel -- the
        field's own published number, read back for the routine's notes."""
        seam = getattr(self.env.scene, "by_name", {}).get("seam")
        if seam is None:
            return 0.0
        return float(seam._carton_states(self.env.state.body_q.numpy()).get("flap_far_open", 0.0))

    def _release(self):
        """The support hand lets go and lifts away."""
        env = self.env
        self.phase = 7.0
        if self.support and self.pq is not None:
            pos, _ = env.arm(self.support).ee_pose(env.state)
            yield from self._leg(
                self.support,
                np.array([pos[0], pos[1], pos[2] + 0.12]),
                self.pq,
                grip=OPEN,
                gate=False,
                speed=0.10,
                label="support retreat",
            )
