# SPDX-License-Identifier: Apache-2.0
"""The polish suite's privileged tools: state, grip, and strip-pose planning.

Nine tools, for the things the generic ``sim.*`` set cannot know. Five carry
the task and are described below; ``sim.plan_retreat``, ``sim.capture_grasp``,
``sim.release_grasp`` and ``sim.wait_steps`` are one line each at their
definitions.

**``sim.get_polish_state``** -- the mark field and machine read together: per-spot
cleared fractions, the smoothed press force, spindle speed, per-arm holds (the
jaw-stall test run for *each* hand -- the generic ``gripped`` reads the work arm
only), and the frames a plan needs: the drum's working **contact point**, its
**spindle axis** (which is also the direction a rub must run in), the **working
normal** the press runs along, the ledge, and the strip's pose.

**``sim.plan_pick``** -- both hands' approach/descend onto the strip's end
stations while it lies on the mat, as joint-space stage trajectories (the usbc
plan shape: full-pose LM per waypoint, warm-started, densified at the control
rate). Independent single-arm stages: nothing is held yet.

**``sim.plan_strip_move``** -- the heart. A named strip-pose motion (raise to
the bit, press a spot, RUB it along the spindle axis, lift off, shift, to/from
the ledge, back to the mat)
turned into per-holder joint trajectories by carrying each holding hand
RIGIDLY with the strip: hand pose = strip pose composed with the grasp offset
captured from the grasp geometry. One holder returns a streamable single-arm
trajectory; two holders return paired waypoints for ``robot.stream_dual``,
which advances every arm one waypoint per control step inside one tool call.

**``sim.set_grip``** -- calibrated partial widths, ramped (a step input on a
kp=20000 finger servo ejects a 4 mm part), persisting through hold actions. A
*gated* close also arms :class:`GripHold`, which keeps the commanded force on
the pads for as long as the hand holds the strip -- see there for why a
position-servo hand otherwise puts a rigid part on the bench two marks in.

**``sim.wait_polish``** -- the feedback dwell: watch one spot's cleared
fraction while the arms hold the press, exit on target / stall / budget.
The press is *force-verified* rather than positioned blind: the same read
reports the smoothed press force, so a dwell that is pressing air stalls out in
150 steps instead of spending its whole budget on nothing. Against a sideways
drum it is a *poll between rubs* rather than the thing that does the work: the
removal happens during the ``rub`` legs, and this decides whether another one
is worth running.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import warp as wp

from robosimstudio.arms import arm_id_of
from robosimstudio.domains.rigid.measures.jaws import GAP_BAND, STALL_MARGIN
from robosimstudio.solvers import grasp_quat
from robosimstudio.solvers.motion.path import stream_hz

from .polish import STRIP_HALF

__all__ = ["ToolsPolish", "register_polish_tools"]

GRASP_STATION = 0.058
"""Grasp station |x| in the strip's own frame [m].

58 mm on a 75 mm half-length, which leaves only ~13 mm of strip between
pads about 20 mm deep -- not much, and it is the most that fits. Gripping
further inboard was tried at 40, 45 and 50 mm and is worse at every one:
the closing hand sweeps the strip off its rest during the slide in, and the
captured offsets come back as nonsense. 45 mm put it on the floor."""

SINGLE_STATION = 0.045
"""Where a single holding hand would grip to press the far end's spot [m].

Unused by the shipped graph, which never lets go -- and the arithmetic that put
it here was right about the torque and wrong about the pads. 78 mm of lever to a
+-33 mm spot is ~0.3-0.6 N*m about the grasp, and what reacts it is not a
"40 N pinch on 20 mm pads" but a couple **as wide as the strip is thick**: 4 mm.
That needs ~75 N a pad where a hand makes ~35, so a one-handed press rotates the
strip out of contact -- measured at 2.8-3.3 N of press at touchdown, 0.00 N
through the dwell, and slip growing 4 -> 64 mm a cycle, against a two-handed
press that holds 7.6-7.9 N through the same dwell on 1 mm of slip.

What a single hand does *not* have trouble with is holding the strip: released
in free air it fell 0.0-0.2 mm over a thousand control steps at 30-38 N a pad.
(An earlier note here said the opposite, on a whole-hand net force measured
against the fingertip's convex hull; see :meth:`ToolsPolish.pad_forces`.) Kept
with its number for the same reason the ledge motions are kept: a
`strip_handover` variant would need both, and re-deriving them is worse than
reading why they were not enough."""

GRASP_LIFT = 0.0
"""Grasp site off the strip's centre plane, along the strip's normal [m].

ZERO, and that is load-bearing. The hands come in end-on with the jaws open
around the strip, so the ONLY thing that lets a pad pass its end face instead
of hitting it is vertical clearance, and this offset spends it. At +4 mm -- a
leftover from an abandoned crossways approach -- the lower pad sits 4 mm above
the strip's mid-plane with only 2 mm under it, less than the arm's own droop,
so the pads met the strip's END and drove it like a vice: measured 81 N of pad
force while the hands were still 68 mm out and still closing, with the strip
squeezed between them and pressed into the rest at 29 N.

That force reads exactly like a grasp on a contact probe and is nothing of the
kind -- it is axial, so there is no vertical friction in it at all, and every
lift dropped the strip the moment the hands stopped converging.

Centred, the jaws straddle the strip with 6 mm of clearance a side."""

APPROACH_H = 0.06
"""Stage height above a set-down before lowering [m]."""

PICK_STANDOFF = 0.055
"""How far outboard of its station a hand stands off before closing in [m].

Along the strip's own axis: the hand stages clear of the end, then slides in
so the open jaws come to straddle the strip rather than arriving on top of it.
"""

STANDOFF = 0.020
"""Gap between the strip's top face and the bit face at a press's start [m]."""

SHIFT_SPEED = 0.02
"""How fast the strip travels sideways between marks [m/s]."""

# A deeper retreat before sideways moves was tried (50 mm rather than the 20 mm
# standoff, on the theory that a shift was dragging the strip across the running
# drum) and made things WORSE, not better: the centre spot went from clearing to
# being thrown on its second press. The extra travel is extra error, and the
# strip at the standoff is already 20 mm clear of the face.

OVERDRIVE = 0.003
"""How far past first contact the press target runs [m].  ; CAL

4-6 N at the arrangement's mass-scaled contact impedance, and HALF what this
first shipped with, because the press is what costs the grasp. The drum's drag
on the pressed strip is ``mu * press`` in the strip's own plane, so a 9 N press
puts ~9 N of sideways load into a grasp whose whole friction budget is the pads'
normal force -- 10-25 N once the carry has settled. Measured over the three-mark
episode: at 6 mm of overdrive the strip walks across the pads about a
millimetre per press and is on the bench by the third mark (2 of 3 marks
polished, then dropped, in 4 runs of 4); at 3 mm it does not walk and the same
episode finishes, sets the strip down on its rest and scores, 6 runs of 6. The
dwell pays for it in time -- the Preston rate is linear in pressure -- and the
episode has the budget: a scoring run is ~2900 of 12000 control steps.

Turning the drum sideways gave this number a SECOND job, and it wants the
press DEEPER rather than shallower, which is why 3 mm survived the change. A
barrel contact's band is ``2*sqrt(2*R*d - d*d)`` wide for a ride-in ``d``, and
that width runs along the STRIP's length -- the direction a mark is only 8 mm
in. Measured (``rgbs/scripts/rgbs_polish_bench.py --press``): 1 mm of overdrive gives
1.73 N, 0.67 mm of ride-in and an 8.1 mm band, which barely covers a mark; 3 mm
gives 3.31 N and 14.8 mm, which covers it with margin. Below about 2 mm the
mark's own length stops being covered, and then rubbing ALONG the strip would
start to gain cells -- which is the direction that is supposed to be useless.

So the two jobs bracket it: the grasp wants it shallow, the band wants it deep,
and 3 mm is where both hold. What the press does NOT set is the other axis: the
drum's 10 mm LENGTH against the mark's 20 mm width is fixed geometry, so no
depth of press ever clears more than 0.68 of a mark standing still."""

STROKE_AMP = 0.006
"""Half the in-contact rub, along the spindle axis [m].

Bounded at both ends, and not by taste. Below: the drum is 10 mm long against
a mark 18 mm across, so the patch has to travel at least +-4 mm to reach the
mark's far edges at all. Above: the strip is 25 mm wide, so a 10 mm patch may
travel +-7.5 mm before a rim runs off an edge and starts gouging. 6 mm sits
between them with 1.5 mm of margin on the edge and 2 mm of reach to spare --
and the reach margin is spent, not decorative: two compliant arms do not
deliver the whole commanded stroke, and at 1 mm of it one mark in three stalled
short (see ``SpecPolishMarks.spot_half``).

The DIRECTION is the whole point and it is not free choice. The drum's surface
at the contact runs across the spindle axis -- along the strip's own long axis
-- so a stroke that way works against the full ``mu * press`` drag, in the one
direction an end-on pinch cannot hold, and the strip walks out of the jaws. A
stroke ALONG the axis is perpendicular to that drag, and a slow perpendicular
stroke on a contact already sliding at 0.38 m/s costs almost nothing, because
the friction cone is saturated in the sliding direction."""

STROKE_SPEED = 0.02
"""How fast the strip rubs along the spindle axis [m/s].

A shift's speed, and for the same reason -- this is a carry with two hands on a
rigid part and the pace at which one stays a carry is a property of the arms,
not of the errand. It is also twenty times under the drum's own 0.38 m/s
surface speed, which is what keeps ``v_rel`` in Preston's law a property of the
drum rather than of the hand: the rub decides WHICH cells are worked, not how
fast each one clears."""

STROKE_CYCLES = 3
"""Complete out-and-back rubs per ``rub`` node.

One cycle is 4 * STROKE_AMP of travel, 1.2 s at STROKE_SPEED, so a node is
~3.6 s of contact and a spot gets up to three of them before the schedule gives
up on it -- against the ~15 s of dwell the press-and-wait build spent per mark.
Split into nodes rather than run as one long rub so the field is polled in
between and a finished spot is not rubbed on."""

GRIP_SQUEEZE = 0.0006
"""Interference per finger under the hold command [m].

Sized against the 4 mm thickness the jaws actually close on, and kept small
because the jaw test calls a grasp held only when the pads stall within
``GAP_BAND`` (1.5 mm) of the part's own dimension -- squeeze harder and a
perfectly good grip reports as a miss.
"""

GRIP_OVER = 0
"""Extra ramp steps to close AFTER the force gate trips.

The gate alone stops the jaws the instant the pads touch, which is a bite with
no reserve: it holds while both hands share the squeeze, and it was believed to
collapse the moment one let go -- "two hands gated at 30 N carry the strip, then
opening either hand drops the OTHER hand's pad force to 0.1 N and the strip
falls 18 mm". That reading was wrong twice: the grasp was on the fingertip's
convex hull, and 0.1 N was the whole-hand NET contact force, which is ~0 for any
balanced grip. Re-measured per pad on the blade, the remaining hand keeps
30-38 N and the strip does not move.

Set to ZERO anyway, because the reserve costs more than it buys. Forty steps
past the gate is roughly 30 N -> 60 N, and at that grip the PRESS stops
working: the strip shifts in the hands after the offsets are captured, so the
planned press aims at where it used to be and never reaches the drum -- two
episodes at 0.00 reward against 0.33 at the bare gate, and a separate 60 N run
that picked at 71/72 N and still pressed at 0 N.

A one-shot reserve was the wrong shape for the problem anyway: the grip does
not fail at the close, it *bleeds away over the episode*. See
:class:`GripHold`, which spends grip force continuously and only when the pads
report they need it."""

GRIP_POLL = 6
"""Control steps between pad-force reads while a gated grip closes."""

GRIP_HOLD_STEP = 4.0e-5
"""Finger travel per poll while a force hold is restoring its grip [m].

0.04 mm, which is about 1.6 N at the task's 20 kN/m finger servo -- small
enough that the restore is a squeeze rather than a step input, and 6 control
steps apart it recovers a sagging carry inside a tenth of a second."""

GRIP_HOLD_CAP = 0.008
"""How much extra finger travel one armed hold may spend in total [m].

The cap is the safety, not the setpoint. The strip is rigid, so while it IS
between the pads the fingers cannot close through it and the hold simply stops
when the force is back. What the cap stops is the other case: a hand that has
genuinely lost the part reads zero however far it closes, and without a bound it
would drive its jaws the whole 15 mm to the `hold` preset -- which on this
collider is ~14 mm past first contact, i.e. 280 N of servo error thrown into the
arm's pose. 8 mm is five times the erosion a whole episode measured."""

GRIP_FORCE = 30.0
"""What a hand closes to when it is taking the strip [N].

Measured, and the margin is the point. The gate stops the ramp the moment the
pads load the part, so too low a gate leaves a shallow bite that any shift
unloads: at 30 N both hands settle at +-60 mm and carry, but only 19 N a pad
survives the raise, and by the second spot the support hand measured 0 N -- it
had let go without ever slipping visibly. At 60 N the hands settle at +-61 mm,
the strip still moves only 0.3 mm, and 34-38 N a pad survives the carry.

The ceiling is over-squeeze, not friction: run to a commanded width instead
and the ramp walks both hands from +-58 mm out to +-83 mm."""

PRE_OPEN = 0.021
"""How far past the strip's half-thickness each jaw stands while approaching [m].

21 mm a side -- a 46 mm gap on a 4 mm strip -- and the number is measured, not
generous-by-taste. The pads are much deeper than the part is thick, so the jaw
gap, not the strip, sets whether a hand can slide in end-on without touching
it. Swept at the +-58 mm station, first contact during the entry comes at:

    gap 16 mm -> touches at 75.7 mm (the strip's own END), hands stall at 64.5
    gap 26 mm -> touches at 68.4 mm,                        hands stall at 61.1
    gap 36 mm -> touches at 60.0 mm,                        hands reach  58.7
    gap 46 mm -> never touches,                             hands reach  57.9

Below about 40 mm the pads meet the strip's end face and drive it like a vice
-- 81-97 N of contact force with the hands still 65-70 mm out and still
closing, which reads exactly like a grasp on a probe and has no vertical
friction in it whatsoever. At 46 mm the strip does not move at all (0.00 mm)
and both hands arrive where they were asked to.

**That sweep was run against the fingertip's convex HULL**, whose inner face
stands ~13 mm proud of the blade, so every number in it is about a jaw 26 mm
thicker than the one drawn. On the decomposed collider the entry is easier, not
harder, and 46 mm is now generous rather than minimal -- kept because generous
is the safe direction here and re-sweeping it buys the pick nothing. What would
be worth re-measuring is the *lower* bound, if a variant ever wants to enter
somewhere tighter than this cell."""

PAD_ON_PART = 5.0
"""Grip force that counts as a hand being on the strip [N].

Comfortably inside the working range measured on the decomposed collider: the
pads first touch a 4 mm strip at a 3.0 mm finger command and make 13 N by
2.0 mm, so 5 N lands between "touching" and "gripping" and separates them
cleanly. The 70-86 N this was originally set against was a whole-hand NET force
on the hull, which is a different quantity -- see :meth:`ToolsPolish.pad_forces`."""

GRIP_WIDTH = 2.0 * STRIP_HALF[2]
"""What the jaws close on [m]: the strip's 4 mm THICKNESS."""


class GripHold:
    """The finger drive, made to hold a FORCE instead of a position.

    **This is what a bimanual carry needs and a position servo cannot give.**
    The hand's grip force is not a setpoint; it is a side effect -- whatever
    standing position error the part happens to hold the jaws open by, times the
    finger servo's stiffness. So the force is only as good as that error, and
    the error is eroded by every press: the part settles a few tenths of a
    millimetre deeper into the pads, the servo arrives at the target it was
    given, and having arrived it stops pushing. Measured on the two-handed
    carry, gated at 30 N and then left alone: 40 N at the close, ~20 N at the
    bit, 8 N after the first mark, 2 N after the second, and the strip is on the
    bench during the third. Nothing slips visibly first; the pads simply stop
    pressing.

    A real force-controlled hand does not have this failure because it keeps
    closing. This is that, at the poll rate the press dwell already pays for:
    while a hold is armed, any hand whose pads read under the target closes by
    ``GRIP_HOLD_STEP``. Close-only -- the fingers cannot close through a rigid
    part that is still between them, so "keep closing until the force is back"
    converges on its own -- and capped by ``GRIP_HOLD_CAP`` so a hand that has
    actually lost the part cannot drive its jaws shut on nothing.

    It runs as a post-step hook rather than inside a tool call because the
    bleed happens *during* motions, and a graph's tools only run between them.
    :meth:`arm` installs it and :meth:`install` is idempotent, which is what
    survives an episode reset: a reset retires every hook a policy added (see
    ``EnvBase._retire_episode_hooks``), and the next episode's gated close puts
    this one back.
    """

    def __init__(self, tools: ToolsPolish, poll: int = GRIP_POLL) -> None:
        self.tools = tools
        self.poll = int(poll)
        self.targets: dict[str, float] = {}
        """Armed arms, by name, and the pad force each is holding [N]."""
        self.spent: dict[str, float] = {}
        self.ticks = 0

    def install(self) -> None:
        """Put this hook on the environment, once. Cheap enough to call often."""
        hooks = self.tools.env.post_step_hooks
        if self not in hooks:
            hooks.append(self)

    def arm(self, force: float, names) -> None:
        self.install()
        for name in names:
            self.targets[name] = float(force)
            self.spent[name] = 0.0

    def disarm(self, names) -> None:
        for name in names:
            self.targets.pop(name, None)
            self.spent.pop(name, None)

    def __call__(self, env) -> None:
        if not self.targets:
            return
        self.ticks += 1
        if self.ticks % self.poll:
            return
        forces = self.tools.pad_forces(tuple(self.targets))
        for name, target in self.targets.items():
            if self.spent.get(name, 0.0) >= GRIP_HOLD_CAP or forces[name] >= target:
                continue
            gripper = env.gripper(name)
            span = env.handles.arms[name].gripper_open
            now = span * (1.0 - float(gripper.command))
            gripper.command = float(min(1.0, 1.0 - max(0.0, now - GRIP_HOLD_STEP) / span))
            self.spent[name] = self.spent.get(name, 0.0) + GRIP_HOLD_STEP


def register_polish_tools(registry, env, bus) -> None:
    """Register the suite's ``sim.*`` tools."""
    tools = ToolsPolish(env, bus)
    for name, fn, summary in tools.entries():
        registry.register_callable(name, fn, summary=summary, tags=("privileged",))


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _qaxis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = 0.5 * float(angle)
    return np.array([*(axis * math.sin(half)), math.cos(half)])


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _qconj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


class ToolsPolish:
    """The suite's privileged tool implementations."""

    def __init__(self, env, bus) -> None:
        self.env = env
        self.bus = bus
        self._probe = None
        self._hold = GripHold(self)
        """The force hold: see :class:`GripHold`. Armed by a gated close."""
        self._shapes: dict[str, list[int]] = {}
        self._grasp: dict[str, np.ndarray] = {}
        self._grasp_q: dict[str, np.ndarray] = {}
        """Per arm, the TCP's measured offset in the strip's own frame [m].

        Captured when the jaws close, and used for every carry after. The
        nominal station is where the hand was ASKED to grip; this is where
        it actually did, and the two differ by this arm's standing servo
        error. Commanding both hands to nominal stations they are not at
        makes them pull against each other through a rigid strip for the
        length of every carry -- measured letting go two thirds of the way
        into the first raise."""

    def entries(self) -> list[tuple[str, Any, str]]:
        return [
            ("sim.get_polish_state", self.get_polish_state, "Marks, press, spindle, per-arm holds, frames."),
            ("sim.plan_pick", self.plan_pick, "Both hands' approach/descend onto the strip's end stations."),
            ("sim.plan_retreat", self.plan_retreat, "A short straight withdrawal for one hand."),
            (
                "sim.plan_strip_move",
                self.plan_strip_move,
                "A named strip motion (raise/press/liftoff/shift/ledge/mat) as per-holder joint trajectories.",
            ),
            ("sim.set_grip", self.set_grip, "Command a partial jaw width: 'open' | 'hold', or metres. Ramped."),
            ("sim.wait_polish", self.wait_polish, "Dwell on a spot until cleared/stalled while the press holds."),
            ("sim.wait_steps", self.wait_steps, "Dwell for N control steps while the policy loop holds."),
            (
                "sim.capture_grasp",
                self.capture_grasp,
                "Record where each hand actually took the strip, for every carry after.",
            ),
            ("sim.release_grasp", self.release_grasp, "Forget a hand's captured grasp when it lets go."),
        ]

    # -------------------------------------------------------------- which world
    @property
    def _world(self) -> int:
        """Which world these tools are driving. 0 unless a wave replicated it.

        A wave builds one connector per world against a
        :class:`~robosimstudio_bench.envs.world_view.WorldView`, which carries
        the index; a single-world ``Env`` does not, and answers 0.
        """
        return int(getattr(self.env, "world", 0))

    # ------------------------------------------------------------------ grip
    def _pad_shapes(self) -> dict[str, list[int]]:
        """Collider shapes per FINGER, and the strip's, resolved once.

        Per finger, not per hand, and that is the whole point -- see
        :meth:`pad_forces`. Keys are ``"strip"`` and ``f"{arm}/{finger}"``.
        """
        if not self._shapes:
            # The model indices are resolved against, which replicated is the
            # TEMPLATE and not the one that steps: a fleet model carries one
            # ``strip`` label per world, and the per-world contact view rebases
            # its shapes to the template, so a set naming every world's strip is
            # the wrong question asked of the right table.
            labels = list(self.env.scene.index_model.shape_label)
            self._shapes["strip"] = [s for s, l in enumerate(labels) if l.split("/")[-1] == "strip"]
            for name in self.env.scene.embodiment.iks:
                for finger in ("tip_left", "tip_right"):
                    tail = f"{name}_{finger}"
                    self._shapes[f"{name}/{finger}"] = [
                        s for s, l in enumerate(labels) if tail in l and "visual" not in l
                    ]
        return self._shapes

    def pad_force(self, name: str) -> float:
        """How hard this hand's pads press the strip [N].

        What "the hand is on the part" means here, in place of the jaw-stall
        test. That
        test compares the measured gap against the part's declared dimension,
        and on this hand the two disagree by design: the pads stall at ~7.7 mm
        on a 4 mm strip while pressing it with 70-86 N, because the jaw-gap
        derivation and the pad faces are not the same surface. A force is what
        "the hand is on the part" actually means, and it needs no calibration
        against a dimension nobody measured.
        """
        return self.pad_forces((name,))[name]

    def pad_forces(self, names) -> dict[str, float]:
        """Every named hand's GRIP force off one contact read-back [N].

        The weaker of the two pads, and each pad measured on its own. Both
        halves of that are load-bearing, and getting either wrong reads a good
        grasp as no grasp:

        **Per pad.** ``pairwise_force`` sums impulse *vectors*. Asked for a whole
        hand against the strip it returns the hand's NET push, and a grasp that
        is working is exactly the case where the two pads are equal and
        opposite -- so the better the grip, the closer to zero it reads.
        Measured, with each pad pressing the strip at 44-45 N: 0.16 N.

        This is not a subtlety that stayed theoretical. Every grip constant this
        suite shipped was gated on the whole-hand number, and it worked only
        because the hand was gripping on its fingertips' convex HULLS -- whose
        inner faces are not parallel, so the grasp was permanently unbalanced
        and the net force was large enough to gate on. Give the hand its real
        blades (``EmbodimentSpec.decompose_colliders``) and the same close reads
        0.2 N while pressing at 45.

        **The weaker one.** What friction can hold is set by the pad pressing
        least, not by their sum and not by their average -- a hand with one pad
        on the part and one in the air is not half a grasp.

        The read-back is the cost, not the arithmetic: ``get_polish_state`` used
        to call :meth:`pad_force` four times and pull the solver's contact table
        across the bus four times for one answer.
        """
        from robosimstudio.core.probes.contact import ContactProbe, FleetContactProbe

        worlds = getattr(self.env.scene, "worlds", None)
        world = int(getattr(self.env, "world", 0))
        if self._probe is None:
            solver = self.env.pipeline.program.mujoco_solver
            # Replicated, the single-world probe reads world 0's geom table and
            # answers every world with world 0's grip -- so a hand that never
            # closed reads as holding, and a graph's verify node routes on it.
            # The fleet's probe reads every world in the same three copies and
            # rebases each one's shapes to the template.
            self._probe = (
                ContactProbe(solver, self.env.model, self.env.pipeline.sim_dt)
                if worlds is None
                else FleetContactProbe(solver, self.env.model, self.env.pipeline.sim_dt, worlds=worlds)
            )
        shapes = self._pad_shapes()
        if worlds is None:
            self._probe.update()
            view = self._probe
        else:
            self._probe.update_all()
            view = self._probe.view(world)
        strip = shapes["strip"]

        def grip(name: str) -> float:
            pads = [shapes.get(f"{name}/{finger}", []) for finger in ("tip_left", "tip_right")]
            forces = [float(np.linalg.norm(view.pairwise_force(strip, pad))) for pad in pads if pad]
            return min(forces) if forces else 0.0

        return {name: grip(name) for name in names}

    # ------------------------------------------------------------------ frames
    def _machine(self):
        return self.env.scene.by_name["machine"]

    def _marks(self):
        return self.env.scene.by_name["marks"]

    def _arm_name(self, role: str) -> str:
        handles = self.env.handles
        if role == "work":
            return handles.work_arm or next(iter(handles.arms))
        if role == "support":
            return handles.support_arm or next(iter(handles.arms))
        return role  # already a name

    def _approach_dir(self, station: float) -> np.ndarray:
        """Which way a hand travels onto its station, in the strip's frame.

        End-on: inward along the strip's long axis, each hand from its own
        outboard end. What makes that safe is that both hands do it AT THE SAME
        TIME. Entering one at a time, whichever hand touches first shoves the
        strip -- measured moving it 14 mm and leaving both tool centres at
        +-75.4 mm on a strip whose half-length is 75 mm, gripping the literal
        tips. Entering together, the two axial pushes are equal and opposite:
        the strip drifts 3 mm instead of 14, and both hands land within about a
        millimetre of the +-58 mm they were asked for.
        """
        return np.array([-math.copysign(1.0, station), 0.0, 0.0])

    def _station_for(self, name: str) -> float:
        """This hand's station along the strip, captured if we have it."""
        local = self._grasp.get(name)
        if local is not None:
            return float(local[0])
        work = self.env.handles.work_arm or next(iter(self.env.handles.arms))
        return -GRASP_STATION if name == work else GRASP_STATION

    def _hand(self, name: str):
        return self.env.scene.embodiment.hands.get(name)

    def _grip_open(self, name: str) -> float:
        return self.env.handles.arms[name].gripper_open

    def _stream_hz(self) -> float:
        return stream_hz(self.env.task.physics)

    def _site_quat(self, name: str, strip_q: np.ndarray, station: float | None = None) -> np.ndarray:
        """The grasp frame for one hand: in along the strip's axis, jaws on its
        faces.

        Approach runs inward along the strip's long axis from this hand's own
        end of the part; the jaws close along the strip's own normal, so the
        pads land flat on its faces however the strip is tilted.

        A top-down grip on the strip's 25 mm EDGES was built and measured as
        the alternative, and it is genuinely the gentler grasp -- descending
        vertically onto the part never moved it by a millimetre, where every
        sideways approach did. It does not fit this cell. The strip lies
        between the two arm bases, so two hands reaching down onto it bring
        both forearms over its middle, and link4 jams into link4 at 70-77 N
        whatever the stations are (measured at +-58, +-66 and +-72 mm).
        Leaning the hands outboard to splay the arms only trades that for
        link4 against the machine's housing at 87 N. Worth revisiting if the
        cell's layout ever changes; it is not available here.
        """
        hand = self._hand(name)
        station = self._station_for(name) if station is None else station
        approach = _qrot(strip_q, self._approach_dir(station))
        # The strip's own normal. What comes BACK is a grasp closing across
        # its 25 mm width, and the 90-degree discrepancy is this hand's frame
        # convention rather than a solver failure -- asked for the width
        # instead, the wrist rolls the other way and the jaws end up vertical
        # (measured both directions by tip-to-tip separation).
        close = _qrot(strip_q, np.array([0.0, 0.0, 1.0]))
        return np.asarray(
            grasp_quat(hand.approach_axis, hand.close_axis, hand.tcp_offset_rot, approach, close),
            dtype=np.float64,
        )

    def _flat_strip_quat(self) -> np.ndarray:
        """The strip's carry orientation: long axis along world y."""
        return _qaxis(np.array([0.0, 0.0, 1.0]), math.pi / 2.0)

    def _press_strip_quat(self) -> np.ndarray:
        """The press orientation: the flat carry PITCHED so the strip's top
        face is square to the drum's working radial.

        Load-bearing, and more so than it was against the old end face. The
        tool lies fifteen degrees nose-down, so the barrel's lowest generatrix
        leans by the same fifteen degrees: a strip presented flat catches the
        drum's nose rim alone -- 1.3 mm lower than the axial centre -- and
        polishes a gouge across one edge of the mark instead of a spot. Pitched
        to match, the whole 10 mm line lands, and the spindle axis then lies IN
        the strip's own plane, which is what makes the stroke a slide rather
        than a climb.

        The pitch is a roll about the strip's long axis, so it costs the grasp
        nothing: both hands hold at the ends and turn with it.
        """
        machine = self._machine()
        # -w: the strip's top-face normal points back UP the working radial.
        n_top = -np.asarray(machine.work_normal_world, dtype=np.float64)
        z = np.array([0.0, 0.0, 1.0])
        axis = np.cross(z, n_top)
        norm = float(np.linalg.norm(axis))
        if norm < 1.0e-9:
            return self._flat_strip_quat()
        angle = math.asin(min(1.0, norm)) if float(np.dot(z, n_top)) >= 0.0 else math.pi - math.asin(min(1.0, norm))
        return _qmul(_qaxis(axis / norm, angle), self._flat_strip_quat())

    def _hand_pose_on_strip(self, name: str, station: float, strip_p: np.ndarray, strip_q: np.ndarray):
        """The hand target that keeps this grasp rigid on the strip.

        Both halves ride the strip: the site sits at a fixed point in the
        strip's frame, and the frame is rebuilt from the strip's *current*
        orientation -- so a tilted or lifted strip carries its hands with it
        rather than being dragged toward a stale world frame.
        """
        local = self._grasp.get(name)
        if local is None:
            local = np.array([float(station), 0.0, GRASP_LIFT])
        held_q = self._grasp_q.get(name)
        quat = self._site_quat(name, strip_q, station) if held_q is None else _qmul(strip_q, held_q)
        return strip_p + _qrot(strip_q, local), quat

    # ------------------------------------------------------------------- state
    def _held_by(self, ctx, name: str) -> tuple[bool, float]:
        """The jaw-stall test for one named arm -- `gripped`'s arithmetic, run
        per hand (the shipped handler reads the work arm only)."""
        arm = ctx.handles.arms.get(name)
        if arm is None or not arm.gripper_coords or not len(ctx.joint_target_q):
            return False, float("nan")
        joint_q = ctx.state.joint_q.numpy()
        measured = float(np.mean([joint_q[c] for c in arm.gripper_coords]))
        commanded = float(np.mean([ctx.joint_target_q[c] for c in arm.gripper_coords]))
        gap = arm.jaw_gap(measured)
        lag = measured - commanded
        held = lag > STALL_MARGIN and abs(gap - GRIP_WIDTH) < GAP_BAND
        return bool(held), float(gap)

    def get_polish_state(self) -> dict[str, Any]:
        def read() -> dict[str, Any]:
            env = self.env
            ctx = env.make_context()
            marks = self._marks()
            machine = self._machine()
            field = marks.read(self._world)
            work, support = self._arm_name("work"), self._arm_name("support")
            held_w, gap_w = self._held_by(ctx, work)
            held_s, gap_s = self._held_by(ctx, support)
            pose = ctx.pose("strip")
            speed = float(machine.published_states(ctx).get("disc_speed", 0.0))
            target = float(machine.spec.spindle_speed)
            from robosimstudio.core.math import transform_point

            def tcp_of(arm_name: str) -> list[float]:
                arm = ctx.handles.arms.get(arm_name)
                if arm is None or arm.ee_body < 0:
                    return [float("nan")] * 3
                return [float(v) for v in transform_point(ctx.body_q[arm.ee_body], arm.ee_offset)]

            forces = self.pad_forces((work, support))
            centres = marks.spot_centres_world(pose)
            spots = [
                {
                    "index": i,
                    "x": float(machine_x[0]),
                    "cleared": float(field["spot_fractions"][i]),
                    "done": bool(field["spot_fractions"][i] >= 0.95),
                    "centre": centres[i],
                }
                for i, machine_x in enumerate(marks.spec.spots)
            ]
            return {
                "polished_fraction": float(field["cleared_fraction"]),
                "spots": spots,
                "spots_remaining": int(sum(1 for s in spots if not s["done"])),
                "press_force": float(field["press_force"]),
                "strip_on_bit": bool(field["press_force"] >= float(marks.spec.f_min)),
                "disc_speed": speed,
                "spinning": bool(abs(speed - target) <= 0.2 * target),
                "strip_position": [float(v) for v in pose[:3]],
                "strip_quat_xyzw": [float(v) for v in pose[3:7]],
                "held": {"work": held_w, "support": held_s},
                "jaw_gap": {"work": gap_w, "support": gap_s},
                "tcp": {"work": tcp_of(work), "support": tcp_of(support)},
                "pad_force": {"work": forces[work], "support": forces[support]},
                "on_part": {
                    "work": bool(forces[work] >= PAD_ON_PART),
                    "support": bool(forces[support] >= PAD_ON_PART),
                },
                "grip_hold": {
                    role: float(self._hold.targets.get(name, 0.0))
                    for role, name in (("work", work), ("support", support))
                },
                "arms": {
                    "work": {"name": work, "arm_id": arm_id_of(env, work)},
                    "support": {"name": support, "arm_id": arm_id_of(env, support)},
                },
                "bit_contact": [float(v) for v in machine.contact_pose[:3]],
                "spindle_axis": [float(v) for v in machine.spindle_axis_world],
                "work_normal": [float(v) for v in machine.work_normal_world],
                "ledge": [float(v) for v in machine.ledge_pose[:3]],
                "grip": {
                    "open_m": self._grip_open(work),
                    "hold_m": float(GRIP_WIDTH / 2.0 - GRIP_SQUEEZE),
                },
            }

        return self.bus.call(read)

    # ------------------------------------------------------------------- grip
    def _set_grip_both(
        self, width: Any, ramp_steps: int, until_force: float, over_steps: int, hold_force: float
    ) -> dict[str, Any]:
        """Ramp both hands to one width together, one control step per tick."""
        names = [self._arm_name("work"), self._arm_name("support")]
        self._hold.disarm(names)

        def resolve(name: str) -> tuple[float, float]:
            if isinstance(width, str):
                if width == "open":
                    cmd = self._grip_open(name)
                elif width == "ready":
                    cmd = GRIP_WIDTH / 2.0 + PRE_OPEN
                elif width == "hold":
                    cmd = max(0.0, GRIP_WIDTH / 2.0 - GRIP_SQUEEZE)
                else:
                    raise ValueError(f"unknown grip preset {width!r}; have 'open' | 'ready' | 'hold' or metres")
            else:
                cmd = float(np.clip(float(width), 0.0, self._grip_open(name)))
            return float(cmd), float(self.env.gripper(name).command)

        def write(fractions: dict[str, float]) -> None:
            for name, fraction in fractions.items():
                gripper = self.env.gripper(name)
                gripper.command = float(fraction)
                gripper.apply(self.env.pipeline.control.joint_target_q)

        resolved = {name: self.bus.call(resolve, name) for name in names}
        targets = {name: 1.0 - cmd / self._grip_open(name) for name, (cmd, _) in resolved.items()}
        starts = {name: start for name, (_, start) in resolved.items()}
        steps = max(1, int(ramp_steps))
        stopped = 0
        for i in range(1, steps + 1):
            self.bus.call(write, {n: starts[n] + (targets[n] - starts[n]) * (i / steps) for n in names})
            self._wait_for_step(1)
            stopped = i
            if until_force > 0.0 and i % GRIP_POLL == 0:
                if min(self.bus.call(self.pad_forces, names).values()) >= until_force:
                    stopped = min(steps, i + max(0, int(over_steps)))
                    for k in range(i + 1, stopped + 1):
                        self.bus.call(write, {n: starts[n] + (targets[n] - starts[n]) * (k / steps) for n in names})
                        self._wait_for_step(1)
                    break
        self._arm_hold(names, width, until_force, hold_force)
        return {
            "arm": "both",
            "command_m": float(resolved[names[0]][0]),
            "fraction_closed": float(starts[names[0]] + (targets[names[0]] - starts[names[0]]) * (stopped / steps)),
            "ramp_used": int(stopped),
            "ramp_steps": int(steps),
            "pad_force": {n: round(f, 2) for n, f in self.bus.call(self.pad_forces, names).items()},
        }

    def _arm_hold(self, names: list[str], width: Any, until_force: float, hold_force: float) -> None:
        """A gated close is a force hold; anything else is a release of one.

        The default follows the gate rather than adding a second number,
        because they are the same claim: a hand told to take the strip at 30 N
        is a hand that should still be at 30 N two marks later, and the only
        reason it is not is that its drive is a position source. ``hold_force``
        overrides -- 0 opts out, which is what the studio's manual jaw slider
        and the release path want.
        """
        force = float(until_force) if hold_force < 0.0 else float(hold_force)
        if force > 0.0 and not (isinstance(width, str) and width in ("open", "ready")):
            self._hold.arm(force, names)

    def set_grip(
        self,
        width: Any = "open",
        arm: str = "work",
        ramp_steps: int = 24,
        until_force: float = 0.0,
        over_steps: int = GRIP_OVER,
        hold_force: float = -1.0,
    ) -> dict[str, Any]:
        """`width` is 'open', 'hold', or a per-finger command in metres.

        ``arm="both"`` ramps the two hands in LOCKSTEP, which the pick needs.
        Closed one after the other, the first hand's pads shove the free strip
        before the second arrives -- measured skidding it 194 mm off its rest,
        or camming it up onto the closed fingers. Closed together the two
        pushes oppose and the strip stays put (measured drift 0.5 mm).

        ``until_force`` stops the ramp once both pads actually load the part,
        and closing the pick ANY other way does not hold. Run to a commanded
        width instead and the ramp keeps travelling long after the pads are on
        the strip -- ``jaw_gap`` reads 25 mm while the pads already press it at
        66 N, because the gap derivation and the pad faces are not the same
        surface -- and the over-squeeze wedges both hands outboard: measured
        walking them from +-58 mm to +-83 mm, dragging the strip 10 mm, and
        leaving each pad half off the end. Gated at 30 N the hands stay at
        +-60 mm, the strip moves 0.3 mm, and the raise carries it 62 mm with
        19 N still on each pad. Gated at 60 N, +-61 mm and 34 N carried.

        A gated close also ARMS THE FORCE HOLD at the same number, which is what
        keeps the grip for the rest of the episode rather than only for the
        close -- see :class:`GripHold`. ``hold_force`` sets the hold apart from
        the gate when the two want different numbers: the gate is a ramp and
        what bounds it is what the close may do to the strip's pose, while the
        hold only ever nudges. -1 follows the gate; 0 opts out.
        """
        if arm == "both":
            return self._set_grip_both(width, ramp_steps, float(until_force), int(over_steps), float(hold_force))
        name = self._arm_name(arm)
        self._hold.disarm([name])

        def resolve() -> tuple[float, float]:
            if isinstance(width, str):
                if width == "open":
                    cmd = self._grip_open(name)
                elif width == "ready":
                    cmd = GRIP_WIDTH / 2.0 + PRE_OPEN
                elif width == "hold":
                    cmd = max(0.0, GRIP_WIDTH / 2.0 - GRIP_SQUEEZE)
                else:
                    raise ValueError(f"unknown grip preset {width!r}; have 'open' | 'ready' | 'hold' or metres")
            else:
                cmd = float(np.clip(float(width), 0.0, self._grip_open(name)))
            gripper = self.env.gripper(name)
            return float(cmd), float(gripper.command)

        def write_fraction(fraction: float) -> None:
            gripper = self.env.gripper(name)
            gripper.command = float(fraction)
            gripper.apply(self.env.pipeline.control.joint_target_q)

        cmd, start_fraction = self.bus.call(resolve)
        target_fraction = 1.0 - cmd / self._grip_open(name)  # 1.0 is closed
        steps = max(1, int(ramp_steps))
        stopped = 0
        for i in range(1, steps + 1):
            fraction = start_fraction + (target_fraction - start_fraction) * (i / steps)
            self.bus.call(write_fraction, fraction)
            self._wait_for_step(1)
            stopped = i
            if until_force > 0.0 and i % GRIP_POLL == 0 and self.bus.call(self.pad_force, name) >= until_force:
                stopped = min(steps, i + max(0, int(over_steps)))
                for k in range(i + 1, stopped + 1):
                    self.bus.call(write_fraction, start_fraction + (target_fraction - start_fraction) * (k / steps))
                    self._wait_for_step(1)
                break
        self._arm_hold([name], width, until_force, hold_force)
        return {
            "arm": name,
            "command_m": float(cmd),
            "fraction_closed": float(start_fraction + (target_fraction - start_fraction) * (stopped / steps)),
            "ramp_used": int(stopped),
            "ramp_steps": int(steps),
            "pad_force": {name: round(float(self.bus.call(self.pad_force, name)), 2)},
        }

    def capture_grasp(self, arms: list | None = None) -> dict[str, Any]:
        """Record each named hand's TCP pose -- position AND orientation -- in
        the strip's own frame.

        Called once the jaws have closed and settled. Every later carry then
        commands the hand to where it IS on the strip rather than to where the
        plan wished it were, so two hands on one rigid part stop fighting.

        The orientation half is not decoration. Capturing only the position and
        re-deriving the wrist from ``_site_quat`` commands both hands back to a
        nominal frame neither of them is actually at -- each arm holds the
        strip a couple of degrees off nominal, in its own direction -- so the
        first carry twists a rigid part between two stiff wrists. Measured: the
        strip lifts cleanly, clears the rails, is carried 11 mm at 60 N a pad,
        and is then wrenched out of both hands within a fifth of a second, in
        free air, with nothing else touching it.
        """
        roles = arms or ["work", "support"]

        def read() -> dict[str, Any]:
            from robosimstudio.core.math import transform_point

            ctx = self.env.make_context()
            pose = ctx.pose("strip")
            strip_p = np.asarray(pose[:3], dtype=np.float64)
            strip_q = np.asarray(pose[3:7], dtype=np.float64)
            out = {}
            for role in roles:
                name = self._arm_name(role)
                arm = ctx.handles.arms[name]
                tcp = np.asarray(transform_point(ctx.body_q[arm.ee_body], arm.ee_offset), dtype=np.float64)
                local = _qrot(_qconj(strip_q), tcp - strip_p)
                # ee_offset is a point, so the hand's orientation IS the ee
                # body's -- the same frame the IK is handed as its target.
                ee_q = np.asarray(ctx.body_q[arm.ee_body][3:7], dtype=np.float64)
                self._grasp[name] = local
                self._grasp_q[name] = _qmul(_qconj(strip_q), ee_q)
                out[role] = [float(v) for v in local]
            return out

        return {"captured": self.bus.call(read)}

    def release_grasp(self, arms: list | None = None) -> dict[str, Any]:
        """Forget a hand's captured grasp -- it is no longer on the strip."""
        for role in arms or ["work", "support"]:
            name = self._arm_name(role)
            self._grasp.pop(name, None)
            self._grasp_q.pop(name, None)
            self._hold.disarm([name])
        return {"released": list(arms or ["work", "support"])}

    # ------------------------------------------------------------------- wait
    def wait_steps(self, steps: int = 30) -> dict[str, Any]:
        return {"waited": self._wait_for_step(int(steps), timeout_s=60.0)}

    def _wait_for_step(self, steps: int, timeout_s: float = 10.0) -> int:
        start = int(self.bus.snapshot.step)
        deadline = time.monotonic() + timeout_s
        while int(self.bus.snapshot.step) < start + steps and time.monotonic() < deadline:
            time.sleep(0.002)
        return int(self.bus.snapshot.step) - start

    def wait_polish(
        self, spot: int = 1, until: float = 0.95, max_steps: int = 600, stall_steps: int = 150
    ) -> dict[str, Any]:
        """Watch one spot clear while the policy loop holds the press."""

        def read() -> tuple[float, float]:
            marks = self._marks()
            field = marks.read(self._world)
            return float(field["spot_fractions"][int(spot)]), float(field["press_force"])

        cleared, press = self.bus.call(read)
        start = cleared
        waited = 0
        last_progress_at = 0
        presses: list[float] = []
        while waited < int(max_steps):
            self._wait_for_step(6)
            waited += 6
            now, press = self.bus.call(read)
            presses.append(press)
            if now > cleared + 1.0e-6:
                cleared, last_progress_at = now, waited
            if cleared >= float(until):
                break
            if waited - last_progress_at >= int(stall_steps):
                break
        return {
            "spot": int(spot),
            "cleared": float(cleared),
            "gained": float(cleared - start),
            "done": bool(cleared >= float(until)),
            "stalled": bool(cleared < float(until) and waited - last_progress_at >= int(stall_steps)),
            "waited": int(waited),
            "press_mean": float(np.mean(presses)) if presses else 0.0,
        }

    # ------------------------------------------------------------------- plans
    def _solve_path(self, ik, targets: list[tuple[np.ndarray, np.ndarray]]) -> list[list[float]]:
        """Warm-started full-pose LM along a target path -- the usbc discipline.

        The seed is **this world's** coordinate block, copied device to device.
        ``ik.model`` is the template, so its coordinate count is one world's
        stride; the state's array holds every world's, and the offset is what
        picks out the one this connector is driving. Seeding from the array's
        head would warm-start every world's plan from the first world's arm,
        which solves -- to a pose the hand asking is not in.
        """
        q_ik = ik.joint_q_ik
        coords = int(ik.model.joint_coord_count)
        world = int(getattr(self.env, "world", 0))
        wp.copy(
            q_ik.reshape((coords,)),
            self.env.pipeline.state_0.joint_q,
            0,
            world * coords,
            coords,
        )
        out: list[list[float]] = []
        for pos, quat in targets:
            ik.set_target(np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64))
            ik.solver.step(q_ik, q_ik, iterations=ik.iterations)
            q = q_ik.numpy()[0]
            out.append([float(q[c]) for c in ik.arm_coords])
        return out

    def plan_pick(
        self,
        station_work: float = -GRASP_STATION,
        station_support: float = GRASP_STATION,
        correct: dict | None = None,
    ) -> dict:
        """Both hands onto the strip where it lies. Independent single-arm
        stages -- nothing is held yet.

        ``over`` stages each hand high and outboard of its end of the strip,
        ``descend`` drops it to the strip's own height still clear of the part,
        and ``close_in`` slides it inward along the strip's axis onto the
        station with the open jaws straddling the strip's thickness.

        ``close_in`` MUST be executed on both arms at once -- see
        ``_approach_dir``. The stages are returned per arm precisely so a
        caller can stream the matching pair together.
        """

        def build() -> dict[str, Any]:
            env = self.env
            ctx = env.make_context()
            pose = ctx.pose("strip")
            strip_p = np.asarray(pose[:3], dtype=np.float64)
            strip_q = np.asarray(pose[3:7], dtype=np.float64)
            stream_hz = self._stream_hz()
            arms = []
            for role, station in (("work", float(station_work)), ("support", float(station_support))):
                name = self._arm_name(role)
                ik = env.arm(name)
                p, q = self._hand_pose_on_strip(name, station, strip_p, strip_q)
                p = p + np.asarray((correct or {}).get(role, (0.0, 0.0, 0.0)), dtype=np.float64)
                # Back off outboard along the length, the way the hand came in.
                out = -_qrot(strip_q, self._approach_dir(station))
                # THREE stages, and the first one is why. A single "go to the
                # standoff and hold" move leaves the joint-space path from home
                # entirely to the servo, and on this cell that path sweeps
                # straight over the strip: measured flinging it 320 mm off the
                # bench before a finger had closed. Staging high above the
                # standoff first, then descending beside the strip, keeps every
                # sweep out of the part's space.
                up = np.array([0.0, 0.0, APPROACH_H])
                stages = []
                for stage, d0, d1, h0, h1, duration in (
                    ("over", PICK_STANDOFF, PICK_STANDOFF, 1.0, 1.0, 1.0),
                    ("descend", PICK_STANDOFF, PICK_STANDOFF, 1.0, 0.0, 0.8),
                    # 2.6 s, not 1.0: the entry is what sets the strip sliding, and
                    # the slower it goes the less it takes with it. Measured at the
                    # close, both pads at 82 N against 77/5 N when it was quick --
                    # at the quick speed one hand ended up past the strip's end
                    # holding nothing at all.
                    ("close_in", PICK_STANDOFF, 0.0, 0.0, 0.0, 2.6),
                ):
                    n = max(2, int(duration * stream_hz))
                    targets = [
                        (
                            p + out * (d0 + (d1 - d0) * i / (n - 1)) + up * (h0 + (h1 - h0) * i / (n - 1)),
                            q,
                        )
                        for i in range(n)
                    ]
                    stages.append(
                        {
                            "stage": stage,
                            "trajectory": {"waypoints": [{"positions": w} for w in self._solve_path(ik, targets)]},
                            "duration": duration,
                        }
                    )
                arms.append(
                    {
                        "role": role,
                        "arm": name,
                        "arm_id": arm_id_of(env, name),
                        "station": station,
                        "stages": stages,
                        "target_tcp": [float(v) for v in p],
                    }
                )
            return {"arms": arms, "grip_hold_m": float(GRIP_WIDTH / 2.0 - GRIP_SQUEEZE)}

        return self.bus.call(build)

    def plan_retreat(self, role: str = "work", distance: float = 0.13, duration: float = 1.0) -> dict:
        """Withdraw one hand straight back along its own approach axis.

        NOT ``robot.go_home``. Homing swings the arm through a wide arc across
        the workspace while the other hand is still holding the strip and the
        two arms are solid to each other -- measured diverging the simulation
        outright. A hand that has just let go only needs to be out of the way,
        and backing straight out the way it came in is the shortest path there.
        """

        def build() -> dict[str, Any]:
            env = self.env
            ctx = env.make_context()
            name = self._arm_name(role)
            ik = env.arm(name)
            arm = ctx.handles.arms[name]
            from robosimstudio.core.math import transform_point

            tcp = np.asarray(transform_point(ctx.body_q[arm.ee_body], arm.ee_offset), dtype=np.float64)
            pose = ctx.pose("strip")
            strip_q = np.asarray(pose[3:7], dtype=np.float64)
            station = self._station_for(name)
            out = -_qrot(strip_q, self._approach_dir(station))
            quat = self._site_quat(name, strip_q, station)
            stream_hz = self._stream_hz()
            n = max(2, int(duration * stream_hz))
            targets = [(tcp + out * (float(distance) * i / (n - 1)), quat) for i in range(n)]
            return {
                "role": role,
                "arm": name,
                "arm_id": arm_id_of(env, name),
                "waypoints": self._solve_path(ik, targets),
            }

        return self.bus.call(build)

    def _strip_path(self, motion: str, spot_x: float) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """The strip-pose legs of a named motion: (position, quat, duration)."""
        env = self.env
        machine = self._machine()
        ctx = env.make_context()
        pose = ctx.pose("strip")
        p0 = np.asarray(pose[:3], dtype=np.float64)
        q0 = np.asarray(pose[3:7], dtype=np.float64)
        # The drum's WORKING CONTACT POINT and the two directions that matter
        # there: `n` presses (out of the barrel, at the strip) and `s` strokes
        # (along the spindle). They are perpendicular, and `s` lies in the
        # pressed strip's own plane -- see `_press_strip_quat`.
        contact = np.asarray(machine.contact_pose[:3], dtype=np.float64)
        n = np.asarray(machine.work_normal_world, dtype=np.float64)
        s = np.asarray(machine.spindle_axis_world, dtype=np.float64)
        q_press = self._press_strip_quat()
        q_flat = self._flat_strip_quat()
        strip_x = _qrot(q_press, np.array([1.0, 0.0, 0.0]))
        hz = float(STRIP_HALF[2])

        def press_center(depth_along_n: float) -> np.ndarray:
            # The spot under the contact band; the top face `depth_along_n`
            # out from the barrel (negative = ridden into it).
            return contact + n * (depth_along_n + hz) - strip_x * float(spot_x)

        standoff = press_center(STANDOFF)
        pressed = press_center(-OVERDRIVE)
        ledge = np.asarray(machine.ledge_pose[:3], dtype=np.float64) + np.array([0.0, 0.0, hz + 0.001])

        if motion == "raise":
            up = p0 + np.array([0.0, 0.0, APPROACH_H])
            return [(up, q0, 1.0), (standoff, q_press, 1.6)]
        if motion == "press":
            return [(pressed, q_press, 1.0)]
        if motion == "liftoff":
            return [(standoff, q_press, 0.8)]
        if motion in ("rub", "rub_across"):
            # THE RUB, and the whole point of the task: the strip is stroked
            # back and forth while it is pressed, ALONG the spindle axis.
            #
            # `rub_across` is the same stroke turned ninety degrees, along the
            # strip's own long axis -- the direction the drum's drag runs in,
            # which reaches no new cells and which a pinch cannot hold. NOTHING
            # calls it: the shipped schedule must not (tests/test_polish.py
            # refuses a graph that names it) and the measurements have their own
            # rig (rgbs/scripts/rgbs_polish_bench.py --rub, tests/test_polish_scenes.py).
            # It is kept for the same reason `SINGLE_STATION` is: a
            # `strip_lateral` variant would want it, and a probe with real arms
            # is the one place the claim can be put to a GRASP rather than to a
            # rigid holder.
            #
            # Every leg is measured from the PRESS pose, never from wherever the
            # strip has slipped to, so a rub cannot accumulate drift; and each
            # cycle returns to centre so the dwell that follows measures a
            # centred press.
            way = s if motion == "rub" else strip_x
            leg = float(STROKE_AMP) / float(STROKE_SPEED)
            legs: list[tuple[np.ndarray, np.ndarray, float]] = []
            for _ in range(int(STROKE_CYCLES)):
                legs.append((pressed + way * STROKE_AMP, q_press, leg))
                legs.append((pressed - way * STROKE_AMP, q_press, 2.0 * leg))
                legs.append((pressed, q_press, leg))
            return legs
        if motion == "shift":
            # Duration scales with the distance. A fixed 0.8 s was far too fast
            # for the 33 mm move to the next mark -- measured leaving the strip
            # behind entirely, the hands arriving alone. SHIFT_SPEED keeps every
            # shift at one honest carry speed instead. This is the only motion
            # that still travels far, now that a mark is finished by rubbing it
            # rather than by pressing it twice off-centre.
            travel = float(np.linalg.norm(standoff - p0))
            return [(standoff, q_press, max(0.8, travel / SHIFT_SPEED))]
        if motion == "to_ledge":
            above = ledge + np.array([0.0, 0.0, 0.05])
            return [(above, q_flat, 1.6), (ledge, q_flat, 1.0)]
        if motion == "from_ledge":
            return [(ledge + np.array([0.0, 0.0, 0.05]), q_flat, 1.0)]
        # The set-down target is where the strip SETTLED at episode start --
        # the context's own record, no frame arithmetic to get wrong.
        strip_body = ctx.handles.entity("strip").body
        settled = np.asarray(ctx.settled_body_q[strip_body][:3], dtype=np.float64)
        if motion == "to_mat":
            above = settled + np.array([0.0, 0.0, APPROACH_H])
            return [(above, q_flat, 1.8)]
        if motion == "lower":
            return [(settled + np.array([0.0, 0.0, 0.002]), q_flat, 1.2)]
        raise ValueError(f"unknown strip motion {motion!r}")

    def plan_strip_move(
        self,
        motion: str,
        holders: list[str] | None = None,
        stations: dict[str, float] | None = None,
        spot_x: float = 0.0,
        correct: dict | None = None,
    ) -> dict[str, Any]:
        """A named strip motion as per-holder joint trajectories.

        Args:
            motion: raise | press | rub | rub_across | liftoff | shift |
                to_ledge | from_ledge | to_mat | lower.
            holders: which hands hold the strip (roles or names). Two holders
                get PAIRED waypoints for ``robot.stream_dual``; one holder gets
                a streamable single-arm trajectory. Both are densified at the
                cell's full control rate.
            stations: strip-frame grasp x per role, e.g. {"work": -0.058}.
            spot_x: the mark spot's strip-frame x, for press-family motions.
        """
        holders = holders or ["work", "support"]
        stations = stations or {}

        def build() -> dict[str, Any]:
            env = self.env
            legs = self._strip_path(motion, spot_x)
            pair = len(holders) > 1
            # FULL control rate for both cases. `robot.stream_dual`
            # advances every arm on every control step, so a pair path
            # densified at half rate is a path executed at double speed:
            # measured shaking the strip out of both hands during the
            # first carry, which read downstream as a press that never
            # touched the drum.
            stream_hz = self._stream_hz()
            # The strip-pose path, densified.
            path: list[tuple[np.ndarray, np.ndarray]] = []
            ctx = env.make_context()
            pose = ctx.pose("strip")
            prev_p = np.asarray(pose[:3], dtype=np.float64)
            prev_q = np.asarray(pose[3:7], dtype=np.float64)
            path0_p, path0_q = prev_p.copy(), prev_q.copy()
            total = 0.0
            for p1, q1_leg, duration in legs:
                steps = max(2, int(duration * stream_hz))
                q1 = np.asarray(q1_leg, dtype=np.float64)
                # Hemisphere-align before interpolating. A quaternion and its
                # negation are the SAME rotation, so a leg whose endpoints
                # happen to carry opposite signs interpolates straight through
                # zero -- the normalised midpoint is garbage and the commanded
                # strip whips through a half turn. Measured as the carry
                # throwing the strip partway through the second leg, with one
                # hand and with two, at every squeeze and station tried.
                if float(np.dot(prev_q, q1)) < 0.0:
                    q1 = -q1
                for i in range(1, steps + 1):
                    t = i / steps
                    p = prev_p + (p1 - prev_p) * t
                    q = prev_q + (q1 - prev_q) * t
                    path.append((p, q / np.linalg.norm(q)))
                prev_p, prev_q = np.asarray(p1, dtype=np.float64), q1
                total += duration
            arms = []
            for role in holders:
                name = self._arm_name(role)
                ik = env.arm(name)
                station = float(stations.get(role, stations.get(name, GRASP_STATION)))
                shift = np.asarray((correct or {}).get(role, (0.0, 0.0, 0.0)), dtype=np.float64)
                # RELATIVE to where this hand actually is, not absolute through
                # the captured grasp offset. The strip slips inside the latch
                # while it is pressed, so by the next motion that offset is
                # stale -- and an absolute mapping then commands the hand to a
                # pose it is not at, which is a step input. Measured: the shift
                # after a press threw the strip 25 mm sideways and 32 mm down
                # inside ten control steps. Carrying the hand by the strip's
                # planned DISPLACEMENT starts every path exactly where the hand
                # already is, so no motion can begin with a jump.
                cur_p, cur_q = ik.ee_pose(env.state)
                cur_p = np.asarray(cur_p, dtype=np.float64)
                cur_q = np.asarray(cur_q, dtype=np.float64)
                q0_inv = _qconj(np.asarray(path0_q, dtype=np.float64))
                arm_rel = cur_p - np.asarray(path0_p, dtype=np.float64)
                targets = []
                for p, q in path:
                    dq = _qmul(np.asarray(q, dtype=np.float64), q0_inv)
                    targets.append((np.asarray(p) + _qrot(dq, arm_rel) + shift, _qmul(dq, cur_q)))
                arms.append(
                    {
                        "role": role,
                        "arm": name,
                        "arm_id": arm_id_of(env, name),
                        "station": station,
                        "target_tcp": [float(v) for v in targets[-1][0]],
                        "waypoints": self._solve_path(ik, targets),
                    }
                )
            return {"motion": motion, "pair": pair, "duration": float(total), "arms": arms}

        return self.bus.call(build)
