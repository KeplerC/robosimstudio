# SPDX-License-Identifier: Apache-2.0
"""A scripted solution: pick the cutter, seat the blade, draw the tape, put the cutter down, hook the far flap open.

The order is the clip's (``sample_videos/box_decanting.mp4``): the right arm
takes the cutter off the mat; seats the blade at the right (-y) end of the tape
and draws it toward the left arm, one way per pass, the tip servoed onto the
seam; the cutter goes back down on the mat where it was; then the right arm
puts a fingertip into the join past the tape's right end and lifts the FAR
flap. The clip's left hand steadies the carton; here the left arm stays home,
because on the arm that pin cost the cut (``seam.task`` has the numbers) and
the unpinned carton moved under 2 mm through the whole draw.

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

DRAW_END = 0.003
"""How far inside the strip's FAR end the forward stroke stops its tip [m].

The stroke used to stop at :data:`DRAW_INSET`, and the last cells were reached
only by the window's lead: read by WORLD position rather than by the field's
station index (``cut_run`` pairs a near cell with the far cell at the mirrored
place -- the far flap's frame is turned half round), the episodes the earlier
notes call "a slant" had both rows parted over the same 64 mm and both intact
over the last 26 mm toward the support arm."""

DRAW_BACK = 0.008
"""How far back over the strip's near end each pass draws before it draws
forward [m].

Eight, not the whole ten-millimetre inset: the edge's window runs from the tip
toward the tool's body, about a millimetre behind the tip and ten ahead of it
at this hold, so the first cell, 4 mm from the strip's end, is in it only
while the tip is within about 1.2 mm of it or past it. Backed up six, a pass
gave that cell 1.2 mm of travel and it survived four passes part-cut; backed
up eight it gets 3 mm each way. The full ten runs the tip OFF the strip, where
the blade reads nothing and a force-following stroke reads nothing as "press
harder"."""

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

TIP_SIDE = -0.0005
"""Where across the seam the stroke holds the blade's TIP [m], measured from the
seam's centreline toward the NEAR flap (so negative is toward the far one).

Half a millimetre toward the far row on purpose: the far flap is the one that
gets hooked open, and its hinge hold is what is left of the far row -- so the
far row has to part ALL the way, while the goal leaves the near row two cells
of slack."""

SIDE_GAIN = 0.6
"""How much of the measured sideways error the stroke takes out per reading.

**The tip is servoed across the seam, not aimed once.** Measured on the arm:
aimed in the air with the tip offset read the pass before, the pressed blade
drew the whole first pass 8.2 mm toward the near flap -- the near row parted
end to end and the far row not at all. The arm at this reach gives sideways
under the press; the knife's own pose says by how much, so each reading moves
the stroke that much back."""

SIDE_LIMIT = 0.015
"""The most the sideways correction may move the stroke [m]."""

DRAW_SINK = 0.003
"""How far below the seat a stroke's correction may sink [m].

The seat has already found the tape by force, so a stroke that wants more than
three millimetres more is not chasing the arm's sag -- it has lost the tape
altogether, and the reading it is chasing is a zero. Unclamped, a stroke that
ran off the strip's end drove itself to the floor of its range and started the
next one ten millimetres deep."""

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

HOOK_BELOW = 0.008
"""How far under the far flap's free-edge corner the fingertip's end is held
[m]: the edge then meets the pad's outer FACE, not its end, at every angle the
climb aims at -- including the :data:`HOOK_LEAD` it runs ahead by."""

HOOK_PRESS = 1.2
"""What the far flap's edge has to push back on the fingertip with [N] before
the climb starts, and what the climb holds.

**Not less, and that is measured.** At 0.5 N the drag carried the edge up
5.8 degrees and stopped there for 300 control steps. It is close to the
1.6 N the carton's mat friction holds, and the carton does slide: 18-23 mm
along +x over the hook in the scored episodes."""

HOOK_LEAD = 10.0
"""How far ahead of the flap's MEASURED angle the climb aims across [deg]."""

HOOK_PULL = 0.006
"""How far above its riding height the climb pulls the pad [m]: what the
servo's spring stretches to drag the edge up by friction."""

HOOK_UNDER_AT = 25.0
"""The flap angle past which the climb stops dragging and pushes [deg]: by
then the free edge is 27 mm over the join, and a push across it has a lever of
``w sin(theta)`` about the hinge."""

HOOK_SLOW = 0.015
"""How fast the pad pushes the raised board until it is past 45 degrees
[m/s]. Slow because the hinge's damping is paid at the edge and the lever is
still short there -- an estimate; the speed was not tested on its own."""

HOOK_RADIUS = 0.056
"""How far out from the hinge the push is AIMED [m], 9 mm in from the free
edge. The contact it gets is the edge itself: read off the contact table at
10, 30, 60 and 69 degrees, one fingertip collider on the far board, pushing
it +x at 0.8-1.6 N."""

HOOK_DONE = 98.0
"""The angle at which the flap is past vertical and gravity has it [deg]."""

HOOK_END = 104.0
"""The furthest the climb aims [deg] -- short of the 110-degree stop."""

HOOK_LEAN = 25.0
"""How far the hooking hand leans about the seam, the +x pad's end leading
[deg] -- and why it must.

The fingertip is a tapered blade: its outer face stands 5 mm further in at
the pad's end than 12 mm up it, a 16-degree inward taper, so an UNLEANED pad
pressed on the free edge pushes it along a normal 16 degrees below
horizontal. About the hinge that is ``w N sin(theta - 16)``: it CLOSES the
flap until the flap is past 16 degrees, and only friction lifts it --
measured, the unleaned drag stalled at 5.8, 9 and 29 degrees and once, with
the press raised, slid the carton 5 mm instead. Leaned 25 degrees, with the
press found by touch and the pull on, every run reached the angle where the
pad goes under the board. The lean alone does not open it: pressed across
with no pull, leaned, the flap stayed shut and the carton slid 6.5 mm."""

HOOK_TOUCH = 0.3
"""What counts as the pad touching a flap on the way down the join [N]."""

HOOK_SHORT = 0.0015
"""How far short of the far edge the pad's outer end corner goes down the
join [m], before the press closes the rest."""

HOOK_STEPS = 400
"""Climb steps per attempt; each is a reading and a short leg."""

HOOK_OUT = 0.024
"""How far past the tape's end, along the seam, the fingertip goes into the
join [m]: clear of the far strip's end on one side and of the end wall under
the join on the other."""

HOOK_TRIES = 2
"""How many times the fingertip goes back into the join if the flap is not
over. The edge can slip off a climbing pad and the box moves a little, so a
second attempt re-reads both."""

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
        self.side_trim = 0.0
        self._seat_side = 0.0
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
        stays on that arm's branch, and on ``yam`` the tilted hold's draw is a
        reach where branches differ (the other of the two pick holds runs the
        wrist roll into its stop at the far end of the tape). So the chain is
        tried from the current configuration and from ``tries`` deterministic
        perturbations of it, and the best start is returned with its worst
        residual."""
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

    def _tip_side(self) -> float:
        """How far the blade's tip stands off the seam's centreline right now,
        toward the near flap [m], from the knife's and the carton's live poses."""
        q = self.env.state.body_q.numpy()
        pose = np.asarray(q[self._knife_body], dtype=np.float64)
        tip = pose[:3] + _rot(pose[3:7]) @ self._tip_local
        return float((tip - self.seam_mid) @ self._side())

    def _side(self) -> np.ndarray:
        """The horizontal unit vector across the seam, toward the near flap."""
        side = np.cross(np.array([0.0, 0.0, 1.0]), self.draw)
        return side / np.linalg.norm(side)

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
        self._far_flap = [i for i, lab in enumerate(labels) if lab.endswith("carton_flap_far")]
        self._near_flap = [i for i, lab in enumerate(labels) if lab.endswith("carton_flap_near")]
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
        yield from self._carry()
        yield from self._cut()
        yield from self._stow()
        yield from self._hook_far()
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

    def _knife_on_mat(self) -> float:
        """The cutter against the mat [N], for the set-down."""
        probe = self.env.contacts
        if probe is None:
            return 0.0
        probe.update(step=None)
        return float(np.linalg.norm(probe.pairwise_force(self._knife_shapes, self._mat_shapes)))

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
            seat = (
                self.tape_start
                + DRAW_INSET * draw
                - off
                + self._side() * (TIP_SIDE + self.side_trim)
                + np.array([0.0, 0.0, SEAT_CLEAR])
            )
            self._seat_side = self.side_trim
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
            run = self.tape_len - DRAW_INSET - DRAW_END
            trim = yield from self._stroke(ik, start, -draw, DRAW_BACK, cq, 0.0)
            back = start - draw * DRAW_BACK
            trim = yield from self._stroke(ik, back, draw, run + DRAW_BACK, cq, trim)
            # Up out of the cut BEFORE going anywhere: the blade sits a
            # millimetre into the strips, and the old lift-clear dragged it
            # forward and up at once -- measured, the knife slid 4-5 mm out of
            # the jaws on every one, and the fourth pass was refused for it.
            here = (
                back
                + draw * (run + DRAW_BACK)
                + np.array([0.0, 0.0, trim])
                + self._side() * (self.side_trim - self._seat_side)
            )
            yield from self._leg(self.work, here + np.array([0.0, 0.0, 0.02]), cq, speed=0.02, gate=False, label="")
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
        side = self._side()
        # `frm` already carries the side trim the seat was aimed with; only
        # what the strokes learn on top of it moves the line.
        trim0 = self._seat_side
        errs = []
        for k in range(n):
            p = frm + direction * length * (k + 1) / n + np.array([0.0, 0.0, trim]) + side * (self.side_trim - trim0)
            sol = ik.solve_q_full(self.seed[self.work], p, quat)
            self.seed[self.work] = sol
            self.held[self.work] = np.asarray(sol)[ik.arm_coords]
            yield
            if k % DRAW_POLL:
                continue
            err = self._tip_side() - TIP_SIDE
            errs.append(err)
            self.side_trim = float(np.clip(self.side_trim - SIDE_GAIN * err, -SIDE_LIMIT, SIDE_LIMIT))
            force, body = self._blade_force(), self._body_force()
            worst, best = min(worst, force), max(best, force)
            if body > BODY_TOUCH or force > DRAW_CEILING * SEAT_FORCE:
                trim += DRAW_BACKOFF * DRAW_STEP
            elif force < SEAT_FORCE:
                trim -= DRAW_STEP
            trim = float(np.clip(trim, -DRAW_SINK, 0.006))
        self.press_n = max(self.press_n, float(best))
        self.notes.append(
            f"stroke {length * 1e3:.0f} mm: {worst:.1f}-{best:.1f} N, trim {trim * 1e3:+.1f} mm, "
            f"tip off the line {np.mean(np.abs(errs or [0.0])) * 1e3:.1f} mm mean (side trim {self.side_trim * 1e3:+.1f})"
        )
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
        """Lift the far flap open in two holds: drag its free edge up from the
        join, then push the raised board round its hinge from underneath.

        **Not by driving a fingertip under the shut board.** The join is 12 mm
        and the pad 7, so a pad whose body is still in the join reaches at most
        a millimetre under either flap -- and the shipped hook, driven 22 mm
        "under the board" at a fixed depth, pushed the far flap's edge instead
        and walked the carton 15 mm along +x before the lift began (measured,
        with the seam cut through and the flap holding nothing).

        What a cut flap needs is far less: with the far row parted the tape's
        hold is gone and the flap weighs 10 g, 0.003 N*m about its hinge.

        1. **Down the join by touch**, leaned (:data:`HOOK_LEAN`), stepping
           away from whichever flap the pad meets.
        2. **Press** the pad's outer face on the far edge until the board reads
           :data:`HOOK_PRESS`.
        3. **Drag**: the servo pulls the pad :data:`HOOK_PULL` above where it
           rides, so friction (pad mu 1.0) carries the edge up. A push across
           alone goes along the board, through the hinge: measured, it slid
           the carton 6.5 mm with the flap still shut.
        4. **Push across**: past :data:`HOOK_UNDER_AT` the hand is aimed round
           the arc, :data:`HOOK_RADIUS` out from the hinge and
           :data:`HOOK_LEAD` ahead of the measured angle, giving the lean back
           as it goes. The pad stays on the raised edge and pushes it +x, which
           the edge's height turns into an opening moment. Dragging at the
           edge alone stalled at 29 degrees.

        Each step aims at the flap's MEASURED angle, never past it, so a pad
        that loses the board waits for it rather than climbing away. The press
        still nudges the carton, 18-23 mm along +x in the scored episodes. The
        goal latches as the flap passes 69 degrees under the finger; run on
        past that, this climb took it to 82 and it stood at 79 after the hand
        withdrew -- open, not on its stop.
        """
        self._read_carton()
        f = self._frame
        self.phase = 6.0
        far, up = self.far, np.array([0.0, 0.0, 1.0])
        # Outside the taped run, where the join is open board: the fingertip is
        # some 40 mm across along the seam, and 20 mm out its body stood on the
        # far strip's end (measured: 9.6 N straight down on it).
        gap_y = float(self.tape_start[1]) - HOOK_OUT
        hq0 = self._across_quat(self.work, np.array([float(self.seam_mid[0]), gap_y, 0.95]), math.pi / 2.0)

        def leaned(deg: float) -> np.ndarray:
            # About the seam, the +x pad's end leading: see HOOK_LEAN.
            return self._lean(hq0, self.draw, math.radians(deg), -far)

        hq = leaned(HOOK_LEAN)
        end = self._pad_end(hq)
        for attempt in range(HOOK_TRIES):
            self._read_carton()
            far = self.far
            rim_z = float(self._carton[2]) + float(f["rim_z"])
            edge = self.seam_mid + far * 0.5 * float(f["seam_gap"])
            edge[2] = rim_z
            # The pad's outer bottom corner over the join, 2 mm short of the
            # far edge, then down until it is HOOK_BELOW under the board.
            over = edge - far * HOOK_SHORT + up * 0.06
            over[1] = gap_y
            into = edge - far * HOOK_SHORT - up * HOOK_BELOW
            into[1] = gap_y
            yield from self._leg(self.work, over - end, hq, grip=OPEN, gate=False, speed=0.12, label="over the join")
            yield from self._hold(10)
            # Down the join by touch: a millimetre at a time from 6 mm over the
            # board, stepping away from whichever flap the pad meets. Where the
            # leaned pad's faces are is the hand's mesh, not this arithmetic,
            # and a descent planned on the arithmetic alone drove it into the
            # far board at 87 N.
            z = float(edge[2]) + 0.006
            while z > float(into[2]):
                z = max(float(into[2]), z - 0.001)
                for _ in range(6):
                    target = into.copy()
                    target[2] = z
                    yield from self._leg(self.work, target - end, hq, gate=False, speed=0.02, label="")
                    yield from self._hold(3)
                    far_n, near_n = self._far_flap_force(), self._near_flap_force()
                    if far_n > HOOK_TOUCH:
                        into -= far * 0.0007
                    elif near_n > HOOK_TOUCH:
                        into += far * 0.0007
                    else:
                        break
            # Toward the far flap a half millimetre at a time until its edge
            # pushes back: the press is found, not commanded.
            reach, force = 0.0, 0.0
            while reach < 0.012:
                reach += 0.0005
                yield from self._leg(self.work, into + far * reach - end, hq, gate=False, speed=0.02, label="")
                yield from self._hold(4)
                force = self._far_flap_force()
                if force >= HOOK_PRESS:
                    break
            self.notes.append(f"hook {attempt}: {force:.2f} N on the far edge at {reach * 1e3:.1f} mm")
            # Where the pad met the edge, as an offset along `far` from the
            # edge's own corner: the climb holds the pad that far in.
            face = float((into + far * reach - edge) @ far)
            under = False
            for _ in range(HOOK_STEPS):
                measured = self._flap_far() * float(self._frame_travel)
                ahead = min(math.radians(HOOK_END), measured + math.radians(HOOK_LEAD))
                if measured >= math.radians(HOOK_UNDER_AT):
                    if not under:
                        # The hinge where it is NOW, once: the press has
                        # nudged the carton some 10 mm by here, and a push
                        # aimed at the hinge read before the approach ran that
                        # far behind the board. Once only -- re-read every
                        # step, the hand chased a carton it was itself pushing
                        # and walked it 44 mm.
                        self._read_carton()
                        far, under = self.far, True
                    # The edge is up: aim round the arc, HOOK_RADIUS out from
                    # the hinge, giving the lean back; the pad pushes the
                    # raised edge across.
                    give = min(1.0, (measured - math.radians(HOOK_UNDER_AT)) / math.radians(30.0))
                    q = leaned(HOOK_LEAN * (1.0 - give))
                    point = self.hinge_far + HOOK_RADIUS * (-math.cos(ahead) * far + math.sin(ahead) * up)
                    point[1] = gap_y
                    # Slowly while the lever is short (see HOOK_SLOW).
                    slow = HOOK_SLOW if measured < math.radians(45.0) else 0.04
                    yield from self._leg(self.work, point - self._pad_end(q), q, gate=False, speed=slow, label="")
                    if measured > math.radians(HOOK_DONE):
                        break
                    continue
                w = float(self.flap_w)
                # Across: the edge where it is about to be, held in by the
                # press. Up: a pull HOOK_PULL above where the pad rides now,
                # so the servo is always drawing the pad up the edge and
                # friction drags the edge with it -- aimed at the flap's
                # measured angle, never past it, so the pad cannot climb off
                # the board. Measured without the pull: the press alone pushed
                # along the board, through the hinge, and slid the carton
                # 6.5 mm with the flap still shut.
                corner = self.hinge_far + w * (-math.cos(ahead) * far + math.sin(ahead) * up)
                lift = float(self.hinge_far[2]) + w * math.sin(measured) - HOOK_BELOW + HOOK_PULL
                force = self._far_flap_force()
                if force < HOOK_PRESS:
                    face += 0.0003
                elif force > 4.0 * HOOK_PRESS:
                    face -= 0.0006
                face = float(np.clip(face, -0.006, 0.008))
                target = corner + far * face
                target[1], target[2] = gap_y, lift
                yield from self._leg(self.work, target - end, hq, gate=False, speed=0.04, label="")
                if measured > math.radians(HOOK_DONE):
                    break
            self.flap_lifted = self._flap_far()
            self.notes.append(f"lift {attempt}: flap_far_open {self.flap_lifted:.3f}")
            if self.flap_lifted * float(self._frame_travel) > math.radians(5.0):
                # Over, or part way: going back into the join would drop it.
                break
        yield from self._hold(20)
        pos, _ = self.env.arm(self.work).ee_pose(self.env.state)
        yield from self._leg(
            self.work,
            np.asarray(pos, dtype=np.float64) - far * 0.04 + up * 0.08,
            hq,
            speed=0.08,
            gate=False,
            label="retreat",
        )
        yield from self._hold(30)
        self.flap_flipped = self._flap_far()
        self.notes.append(f"flap left at: flap_far_open {self.flap_flipped:.3f}")

    def _near_flap_force(self) -> float:
        """The work hand's fingertips against the near flap's board [N]."""
        probe = self.env.contacts
        if probe is None:
            return 0.0
        probe.update(step=None)
        return float(np.linalg.norm(probe.pairwise_force(self._tips[self.work], self._near_flap)))

    def _far_flap_force(self) -> float:
        """The work hand's fingertips against the far flap's BOARD [N] -- not
        its strip, which a pad standing on it reads as a press on the edge."""
        probe = self.env.contacts
        if probe is None:
            return 0.0
        probe.update(step=None)
        return float(np.linalg.norm(probe.pairwise_force(self._tips[self.work], self._far_flap)))

    def _flap_far(self) -> float:
        """How far open the far flap is, as a fraction of its travel -- the
        field's own published number, read back for the routine's notes."""
        seam = getattr(self.env.scene, "by_name", {}).get("seam")
        if seam is None:
            return 0.0
        return float(seam._carton_states(self.env.state.body_q.numpy()).get("flap_far_open", 0.0))
