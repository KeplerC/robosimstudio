# SPDX-License-Identifier: Apache-2.0
"""The surface-prep suite's privileged ``sim.*`` tools, and the choreography that spends them.

The half of this suite that answers the task rather than states it: the measured
surface a policy is given (:class:`ToolsSurfacePrep`), the hold that lets a hand
carry a foreign engine's foam (:class:`SpongeHold`), and the take itself
(:class:`Wiper`) -- the support hand steadying the bar at its far elbow, the
work hand taking the sponge, leaning 35 degrees along the bar, pressing to a
force, stroking the stripe off, and putting the sponge back. :class:`Wiper` is
one generator that both the benchmark's scripted policy
(:class:`RoutineSurfacePrep`, which yields actions into ``Env.step``) and the
live page (``rgbs/scripts/surface_cleaning.py``, which steps the sim itself)
play, so there is one choreography and not two copies of it.

Two things here are not polish's, and both are consequences of the abrasive
being a foreign engine's foam rather than a rigid body bolted to a stand.

**The grasp needed an engine fix, not a task fix.** SuperDex builds every
mirrored collider with ``penalty_smoothing_half_distance`` at the engine's
default of 5 mm, which is wider than the ~3 mm at which the contact penalty
engages -- so a fingertip pressing foam stays on the soft part of the ramp for
its whole useful range. Measured here: the pads sat **2.06 mm inside the block**
and pressed it with **700 Pa**, the jaws' 6 mm of travel produced 1.23 mm of
dent, and closing any further extruded the block instead of squeezing it. That
reads exactly like a task-level grasp bug and is not one.

``robosimstudio/engines/superdex/mirror.py`` now builds mirrors at 0.5 mm, where
the pads rest **on** the surface (contact gap +0.35 to +0.62 mm) rather than
inside it. What that changed, same scene, same 39 mm gap:

===================  ==========  ==========
                     before      after
===================  ==========  ==========
deepest contact gap  -2.06 mm    +0.35 mm
peak deformation      1.95 mm     4.73 mm
grip force            0.93 N      5.53 N
===================  ==========  ==========

**And it removed the cliff.** The old band was 37-42 mm with a hard floor -- 36
extruded the block and dropped it -- and that floor was the contact model, not
the foam. Every gap from 41 down to **26 mm** now carries, with grip rising
monotonically 2.28 -> 13.16 N and the block bulging 70 -> 75.6 mm on its long
axis as it is squeezed. A "the foam only takes 7-8 mm of jaw travel" rule was
recorded here from the old numbers; it was an artifact and is retracted.

The one measurement from that period that still stands is where the pads are:
see :data:`PAD_CONTACT_OFFSET`.

**The pins are a latch, not the grasp.** The jaws carry the sponge on their own
through a lift and the strokes; :class:`SpongeHold` only insures that against
the press. So it pins the small patch under the pads rather than the waist set:
the waist is 58% of the lattice, and pinned that way the block stops being foam
-- it deforms 0.06 mm under a press and drives 3.2 mm into the bar instead of
conforming to it.

**``sim.attach`` will lie to you.** It registers against this sponge (the entity
carries particles), reports ``attached: true``, and does nothing: it appends a
``ParticleDragger`` to ``post_solve`` and the bridge's own particle write runs
later, in ``after``, over the whole actor range, every control step. Nothing
raises and the sponge simply stays where it was.

The pins are re-aimed from the **pad** frames and never from the wrist body: a
wrist is 130 mm behind the grip point on this hand, so a hold carried on the
wrist frame rotates the foam about a point it is not held at.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from robosimstudio.arms import arm_slot, arm_slots
from robosimstudio.core.math import quat_multiply

from .prep.layout import LAYOUT, Layout

__all__ = [
    "RoutineSurfacePrep",
    "SpongeHold",
    "ToolsSurfacePrep",
    "Wiper",
    "register_surface_prep_tools",
    "stroke_span",
    "wipe_quat",
]

GRIP_GAP = 0.032
"""Pad separation the jaws close to on the sponge [m].

18 mm of squeeze on the block's 50 mm axis -- 36% strain, which is what taking
hold of a sponge looks like. Measured on the earlier 45 mm block it read 10.16 N
of grip and 8.66 mm of peak deformation, and it survives a lift and every stroke
of the take.

There is no cliff to stay clear of any more, so this is chosen for the grip it
gives rather than to avoid one: 36 mm gave 7.68 N and 6.36 mm, 28 mm gave
12.76 N and 10.92 mm, and everything from 41 down to 26 carried. 32 leaves
margin over the ~4 N the press into the bar needs, without crushing the block
so hard it stops conforming.

This was 0.030 for most of the suite's life -- close to right by accident, and
useless, because the contact underneath it was letting the pads sink into the
foam rather than press it. Then briefly 0.039, chosen to sit mid-band in a band
that only existed because of that bug.

Note the span this is measured against: ``ArmHandle.gripper_span`` is 94 mm and
``gripper_open`` is **per-finger travel** at 47.5 mm -- reading the second as the
first halves every gap commanded, which is how the first grasp probe here closed
to 18 mm believing it had asked for 36.
"""

PAD_CONTACT_OFFSET = (0.001, 0.0, 0.007)
"""Where the pads actually touch a grasped object, in the TOOL frame [m].

**Measured, from MuJoCo's own contact table on a grasp that works** -- the
support hand taking a 30 mm cube and lifting it clear. The five geom pairs it
reports sit at::

    right_tip_left    (-3.9, +15.0, +7.4)   (+5.3, +14.9, +8.6)
    right_tip_right   (-3.2, -14.8, +6.0)   (+1.8, -14.8, +6.2)   (+4.8, -14.9, +6.4)

so the pads straddle the TCP at +-15 mm on the jaw axis and lead it by about
7 mm -- i.e. **the TCP is the grasp point**, which is the convention every other
suite here already aims with.

This constant is a retraction. Two earlier attempts to derive it from the model
both gave large wrong answers, and each one moved a grasp far enough to close on
air: the fingertip **body origins** sit 80 mm behind the TCP (they are at the
finger root, not the pad), and ``model.shape_transform`` gives 210 mm and the
wrong side of the hand -- it is not a per-collider body-local offset here, and
all fifty decomposed pieces of one fingertip report the same value. Take tool
geometry from contacts on a working grasp; do not compute it."""

PIN_RADIUS = 0.014
"""How far from a pad's contact point a node may be and still be pinned [m].

Sized to the pad, not to the block: about a 28 mm disc on each gripped face,
which on this lattice takes roughly a tenth of the nodes instead of the waist
set's 58%. Everything outside it is free foam, which is the whole point -- the
sponge has to conform around a 25 mm tube, and a block pinned through its
middle cannot."""

PIN_STIFFNESS = 4000.0
PIN_DAMPING = 20.0
"""Compliant node constraints, not a weld.

The engine's own constraint, so the pinned patch still argues with contact and
inertia: the foam under the pads deforms against the bar while the band above it
tracks the hand. ``tests/test_superdex.py`` uses 3000/15 for a straight squash;
this is a little stiffer because the hold has to survive a lateral drag."""

SETTLE = 20
"""Control steps to let a close equilibrate before anything is read."""

REACH_TOLERANCE = 0.010
"""How far a planned waypoint may miss before the plan is refused [m].

**A plan nobody checks is a plan that gets executed anyway**, and this tool
computed the error and threw it away for four takes of a film. What that looked
like: the approach height was set 90 mm over the bar, the arm cannot reach the
far station up there, the IK missed by 39 mm, and the hand carried the sponge
32 mm to the side of a 22 mm bar. Every downstream reading was then honest and
useless -- 35 contact samples, 0.02 N, a patch that never cleared, and a press
that a force poll happily drove to 16 mm chasing contact that was never
geometrically possible."""

CLEARANCE = 0.040
"""Height of the sponge's underside above the crown for crossing to and from the bar [m].

Reach, not preference. The work arm solves the wipe at TCP heights of about
z 0.86 to 0.88 world and the reach sweep never cleared anything much higher;
90 mm of clearance put the wrist off the end of the envelope on the earlier
layout. 40 mm lifts the foam clear of the crown, which is all a crossing needs,
and the force loop's travel budget is this plus 30 mm."""

TILT_DEG = 35.0
"""How far the working hand leans along the bar for the wipe [deg].

The real hand comes in leaning along the bar. Measured on the page, 35-50
degrees is the band the arm holds through a whole stroke at the -90 degree jaw
roll; upright (0) the wrist runs out of orientation at the far end of the stripe.
The lean lifts the sponge's far edge, so the contact patch trails the pinned
centre by about a third of the block -- which is why the strokes overrun the
stripe (:data:`OVERRUN_FAR`)."""

PRESS_N = 4.0
"""What the force loop presses to, read off the taint field [N].

A newton clears the crown under the 22 mm calibration (the ±40 deg stations
needed ~1 N and 3 mm of overdrive); 4 N keeps the whole contact patch well over
the field's 300 Pa floor through a stroke, at a load the pinned grasp carries
without the foam creeping out of the jaws."""

STROKE_SPEED = 0.06
"""Along the bar, during a stroke [m/s]. Removal is pressure times sliding
speed, so a stationary press removes nothing; 60 mm/s is what the clip's hand
does and what keeps a segment's IK inside the gate."""

STROKE_CYCLES = 3
"""Out-and-back cycles before the take gives up. The stripe clears in one or two
at 4 N; the third is the budget for a press that seated light."""

STROKE_SEGMENTS = 4
"""Segments per stroke, with a seat trim between them. See :meth:`Wiper._stroke`."""

OVERRUN_NEAR = 0.025
OVERRUN_FAR = 0.045
"""How far a stroke's ends run past the stripe's, near and far [m].

Both ends overrun the stripe: measured on the page, the end tenths of the stripe
were what stayed blue (38 % and 8 % cleared) when the strokes turned at the
block's own edge. The far end runs further because the leaning hand's contact
patch trails the pinned centre toward the near end."""

HAND_CLEAR = 0.025
"""A stroke never comes within this of the steadying hand's pads [m] -- a
jaw's half-width plus a margin. A stroke to the end of a longer stripe shoved
that hand off the bar."""

TRIM_STEP = 0.0015
TRIM_BOUNDS = (-0.015, 0.008)
"""The running seat-height trim [m]: a step per segment, bounded. See :meth:`Wiper._stroke`."""

RESEAT_LIFT = 0.008
RESEAT_TRAVEL = 0.020
"""The re-seat at the far end: start this far above the near seat's height and
spend at most this much travel finding the force again [m]."""

PATCH_CLEARED = 0.96
"""When the take stops stroking. Over the document's 0.95 gate, so an exit
cannot leave the field a fraction short of it."""

LEG_MAX_POS, LEG_MAX_ROT = 0.006, math.radians(8.0)
"""What a planned leg may be off by, with the sponge in the hand, before it is
refused. Tighter than :data:`REACH_TOLERANCE` because a routine leg is
*interpolated* -- every waypoint on it is solved, so the worst is a real worst
rather than the endpoint's. See ``box_decant``: an unreachable waypoint comes
back as a solution tens of millimetres off with nothing raised, and a lurch
with something in the hand is a thrown part."""

CARRY_MAX_ROT = math.radians(25.0)
"""The orientation gate for a leaning leg: a hand leaning 35 degrees along the
bar reads a larger, harmless residual than an upright one, and 8 degrees
refused every carry."""

SUPPORT_STANDOFF = 0.050
SUPPORT_ABOVE_AXIS = 0.004
SUPPORT_GRIP_EXTRA = 0.004
"""The steadying hold: approach the far elbow from 50 mm above, come down so
the pads straddle the tube with the TCP 4 mm over its axis, and close to the
tube's diameter plus 4 mm -- a steadying hand, not a clamp on a welded bar."""

TAIL = 150
"""Settled control steps after both hands have retreated: the goal's
``hold-for`` is two seconds, 120 steps, and this is what makes them happen
inside the take rather than after it."""

OPEN, SHUT = 0.0, 1.0
"""This hand closes at the HIGH end of a normalised gripper command.

The same convention ``box_decant`` records, and the same trap: driven the other
way the hand descends shut, knocks the sponge across the mat, and opens on
empty air -- which reads exactly like a grasp that dropped it."""

DOWN_QUAT = (1.0, 0.0, 0.0, 0.0)
"""Tool +z onto world -z, xyzw: every grasp on this bench is top-down."""


def wipe_quat(tilt_deg: float = 0.0) -> np.ndarray:
    """Top-down, rolled so the jaws close across world x (across the sponge's
    50 mm axis, so its 75 mm axis lies along the bar), then tilted about world x
    by ``tilt_deg`` -- the real hand comes in leaning along the bar.

    Rolled -90 rather than +90: both close the jaws across x, but with the hand
    leaning along the bar only this one lets the wrist hold the pose -- the
    other reads 13-46 deg of orientation residual along the whole run. The
    roll is the jaw axis and it is not free: the pads straddle the TCP on the
    tool's own **y** (:data:`PAD_CONTACT_OFFSET`'s measured contact pairs, at
    +-15 mm), so the roll is what decides which sponge axis the jaws squeeze.
    """
    q = quat_multiply((0.0, 0.0, -math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)), DOWN_QUAT)
    t = math.radians(tilt_deg)
    return np.asarray(quat_multiply((math.sin(t / 2.0), 0.0, 0.0, math.cos(t / 2.0)), q), dtype=np.float64)


def stroke_span(layout: Layout, sponge_half: float) -> tuple[float, float]:
    """Where a stroke's two ends sit along the bar [m, axial], for a sponge of
    ``sponge_half`` half-length: the stripe overrun at both ends, kept on the
    straight run at the near end and a jaw's width clear of the steadying hand
    at the far one."""
    near = max(-layout.stripe_half + sponge_half - OVERRUN_NEAR, -layout.straight_half + sponge_half + 0.005)
    far = min(layout.stripe_half - sponge_half + OVERRUN_FAR, layout.support_u - sponge_half - HAND_CLEAR)
    return float(near), float(far)


def _bridge_of(pipeline, entity: str):
    """The host-engine bridge simulating ``entity``'s island, or ``None``.

    Duck-typed the way ``domains/soft`` finds it, and deliberately not reached
    through the taint field's own handle: two callers wanting the same object
    is not a reason for one of them to own the other.
    """
    program = getattr(pipeline, "program", None)
    if program is None:
        return None
    island_of = getattr(program, "island_of", None)
    if not callable(island_of):
        return None
    try:
        island = island_of(entity)
    except (KeyError, ValueError):
        return None
    solver = program.solver
    lookup = getattr(solver, "solver", None)
    inner = lookup(island.name) if callable(lookup) else solver
    return getattr(inner, "bridge", None)


def _qrot(q, v) -> np.ndarray:
    x, y, z, w = (float(c) for c in q)
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=np.float64)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _qconj(q) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _slerp(q0, q1, t: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64) / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=np.float64) / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    th = math.acos(d)
    return (math.sin((1.0 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


class SpongeHold:
    """The foam, carried by the hand that closed on it.

    A post-step hook rather than something a tool does once, for
    :class:`~tasks.polish.tools.GripHold`'s reason: the hand moves *between*
    tool calls, and a graph's tools only run between motions. Every control step
    this re-aims the pin targets to where the grasped patch would be if it were
    rigidly attached to the pad frame -- so the band the jaws hold follows the
    hand exactly, and everything below it is still foam.

    Installed idempotently and retired with the episode, which is the same
    lifecycle a reset gives it: ``EnvBase`` drops every hook a policy added, and
    the bridge's ``restore_rest`` drops every pin, so the next grasp starts from
    nothing rather than from a stale id.
    """

    def __init__(self, tools: ToolsSurfacePrep) -> None:
        self.tools = tools
        self.pin: int | None = None
        self.arm: str = ""
        self.local: np.ndarray | None = None
        """The pinned nodes in the pad frame, captured at the close."""

    def install(self) -> None:
        hooks = self.tools.env.post_step_hooks
        if self not in hooks:
            hooks.append(self)

    def arm_hold(self, arm: str, pin: int, local: np.ndarray) -> None:
        self.arm, self.pin, self.local = arm, int(pin), np.asarray(local, dtype=np.float64)
        self.install()

    def disarm(self) -> None:
        self.pin, self.local, self.arm = None, None, ""

    def __call__(self, env) -> None:
        if self.pin is None or self.local is None:
            return
        pos, quat = env.arm(self.arm).ee_pose(env.state)
        targets = np.asarray(pos, dtype=np.float64) + np.array(
            [_qrot(np.asarray(quat, dtype=np.float64), row) for row in self.local]
        )
        try:
            self.tools.bridge.set_pin_targets(self.pin, targets)
        except (KeyError, IndexError):
            # The pin went with an episode reset; the next grasp makes another.
            self.disarm()


class ToolsSurfacePrep:
    """The suite's privileged tool implementations.

    ``env`` is anything with the stepping surface -- the benchmark's ``Env`` or
    a bare ``Sim`` -- and the tools never step it themselves: a graph's runtime
    hands every world read to the sim thread through ``bus.call`` and waits on
    the step counter, and a routine on the stepping thread uses
    :class:`_DirectBus`.
    """

    def __init__(self, env, bus, *, layout: Layout = LAYOUT) -> None:
        self.env = env
        self.bus = bus
        self.layout = layout
        self._hold = SpongeHold(self)
        self._bridge = None
        self._sponge_nodes: np.ndarray | None = None
        self._carry: dict[str, Any] | None = None
        """Geometry of the current grasp, captured when the jaws closed."""

    # ------------------------------------------------------------------ wiring
    @property
    def bridge(self):
        if self._bridge is None:
            self._bridge = _bridge_of(self.env.pipeline, "sponge")
            if self._bridge is None:
                raise RuntimeError(
                    "surface_prep's tools need the sponge on a host-engine island: the grasp is "
                    "the engine's own node constraints, and there is no other way to hold a "
                    "foreign engine's deformable in a hand."
                )
        return self._bridge

    def _taint(self):
        return self.env.scene.by_name["taint"]

    def _sponge(self):
        return self.env.scene.by_name["sponge"]

    # -------------------------------------------------------------- which world
    @property
    def _world(self) -> int:
        """Which world these tools are driving. 0 unless a wave replicated it.

        A wave builds one connector per world against a
        :class:`~robosimstudio_bench.envs.world_view.WorldView`, which carries
        the index; a single-world ``Env`` does not, and answers 0.
        """
        return int(getattr(self.env, "world", 0))

    def _particle_offset(self) -> int:
        """Where this world's particle block starts.

        An entity's ``particles`` are the **template's** indices, like every
        index resolved at bind. ``particle_q`` on a wave holds one block per
        world, so a patch chosen from the template indexes world 0's sponge --
        and a pin made from it holds world 0's block while this world's hand
        closes on nothing. Measured: at three worlds on one episode, world 0
        succeeded in 2277 steps and worlds 1 and 2 truncated at 12000 having
        never picked their sponge up.
        """
        worlds = getattr(self.env, "worlds", None) or getattr(getattr(self.env, "fleet", None), "worlds", None)
        return self._world * int(getattr(worlds, "particles_per_world", 0) or 0)

    def _sponge_parts(self) -> np.ndarray:
        """This world's sponge particles, as indices into the fleet's array."""
        return np.asarray(self._sponge().particles, dtype=np.int64) + self._particle_offset()

    def _arm_name(self, role: str) -> str:
        """``"work"``/``"support"`` to this embodiment's arm name."""
        if role in self.env.scene.embodiment.iks:
            return role
        return str(self.env.handles.arm(role).name)

    def pad_offset(self, arm: str = "work") -> np.ndarray:
        """Where the pads sit relative to the TCP, in the TOOL frame [m].

        :data:`PAD_CONTACT_OFFSET`, which is a measured 7 mm -- so aiming the
        TCP straight at a grasp point is very nearly right, and this only trims
        it. ``arm`` is accepted because the two hands are mirrored and a caller
        should not have to know they happen to agree; it is not read.

        This used to return the fingertip **body** midpoint, which is 80 mm
        behind the TCP because those origins are at the finger root rather than
        the pad, and every grasp planned through it closed a hand's length above
        the sponge -- the approach solved, the jaws shut on air, and nothing
        raised.
        """
        del arm
        return np.asarray(PAD_CONTACT_OFFSET, dtype=np.float64)

    def _waist(self) -> np.ndarray:
        if self._sponge_nodes is None:
            # This world's block: the set is the template's, like every index
            # an entity resolves at bind, and the fallback patch is pinned
            # through the same array the disc is.
            self._sponge_nodes = (
                np.asarray(self._sponge().set_indices("sponge/waist"), dtype=np.int64) + self._particle_offset()
            )
        return self._sponge_nodes

    def _pad_patch(self, name: str, gap: float) -> np.ndarray:
        """The nodes under the two pads, chosen in the hand's own frame.

        A disc of radius :data:`PIN_RADIUS` on each gripped face, rather than
        the waist band: the jaws hold the sponge by themselves now, so the pins
        only have to insure that against the press, and pinning 58% of the
        lattice makes the block behave like a rigid one.

        Falls back to the waist set if the discs come up nearly empty -- which
        means the pads are not where this thinks they are, and a latch on the
        wrong nodes is worse than a coarse one.
        """
        parts = self._sponge_parts()
        q = self.env.pipeline.state_0.particle_q.numpy()[parts]
        pos, quat = self.env.arm(name).ee_pose(self.env.state)
        inv = _qconj(np.asarray(quat, dtype=np.float64))
        local = np.array([_qrot(inv, row - np.asarray(pos, dtype=np.float64)) for row in q])
        off = np.asarray(PAD_CONTACT_OFFSET, dtype=np.float64)
        # Distance measured IN the face plane, not in space: the two discs are
        # a jaw-gap apart along y, so a spatial radius would have to span the
        # block to reach either of them.
        in_plane = np.hypot(local[:, 0] - off[0], local[:, 2] - off[2]) <= PIN_RADIUS
        on_face = np.abs(local[:, 1] - off[1]) >= 0.5 * float(gap) - PIN_RADIUS
        chosen = parts[in_plane & on_face]
        return chosen if len(chosen) >= 8 else self._waist()

    def entries(self) -> list[tuple[str, Any, str]]:
        return [
            (
                "sim.get_prep_state",
                self.get_prep_state,
                "Taint field, press, the bar's frame, the stripe and the hold.",
            ),
            (
                "sim.plan_sponge_pick",
                self.plan_sponge_pick,
                "Approach and descend so the PADS land on the sponge, not the TCP.",
            ),
            (
                "sim.grip_sponge",
                self.grip_sponge,
                "Close on the sponge and carry it: jaws for friction, pins for the hold.",
            ),
            ("sim.release_sponge", self.release_sponge, "Let the sponge go and open the jaws."),
            ("sim.set_grip", self.set_grip, "Command a pad separation in metres, or 'open'. Ramped."),
            (
                "sim.plan_support_hold",
                self.plan_support_hold,
                "The support hand onto the bar at its far elbow: over, then down astride the tube.",
            ),
            (
                "sim.plan_sponge_move",
                self.plan_sponge_move,
                "A named sponge motion (lift/over/press/stroke) as the work arm's trajectory.",
            ),
            ("sim.plan_retreat", self.plan_retreat, "A straight withdrawal for one hand."),
            ("sim.wait_wipe", self.wait_wipe, "Dwell on the stripe until it clears or stalls."),
            ("sim.wait_steps", self.wait_steps, "Dwell for N control steps while the policy loop holds."),
        ]

    # --------------------------------------------------------------------- read
    def get_prep_state(self) -> dict[str, Any]:
        """Everything a node needs about the world, in one bus call."""

        def read() -> dict[str, Any]:
            env = self.env
            taint = self._taint()
            field = taint.read(self._world)
            bar_body = int(env.handles.entity("bar").body)
            # ``env.state`` and not ``env.pipeline.state_0``: a wave's view
            # slices the fleet's body array to this world and indexes it like the
            # template, which is the space ``bar_body`` is in. The raw pipeline
            # array would hand every world world 0's bar.
            pose = env.state.body_q.numpy()[bar_body]
            bar = np.asarray(pose[:3], dtype=np.float64)
            quat = np.asarray(pose[3:7], dtype=np.float64)
            along = _qrot(quat, np.array([0.0, 0.0, 1.0]))
            radius = float(taint.spec.bar_radius)
            sponge_q = env.pipeline.state_0.particle_q.numpy()[self._sponge_parts()]
            half_along = float(self._half_along(sponge_q, pose))
            near_u, far_u = stroke_span(self.layout, half_along)
            hold = bar + along * float(self.layout.support_u)
            out: dict[str, Any] = {
                "cleaned_fraction": float(field["cleared_fraction"]),
                "patches": [
                    {"index": i, "cleared": float(f), "done": bool(f >= 0.9)}
                    for i, f in enumerate(field["patch_fractions"])
                ],
                "wipe_force": float(field["wipe_force"]),
                "slide_speed": float(field["slide_speed"]),
                "sponge_on_bar": bool(field["touching"]),
                "sponge_speed": float(field["sponge_speed"]),
                "bar_position": [float(v) for v in bar],
                "bar_quat_xyzw": [float(v) for v in quat],
                "bar_along": [float(v) for v in along],
                "bar_radius": radius,
                "crown_z": float(bar[2]) + radius,
                # The stripe, and the span a stroke may use: overrun at both
                # ends, kept on the straight run and clear of the steadying hand.
                "stripe": {
                    "half": float(taint.spec.patch_half[0]),
                    "near_u": near_u,
                    "far_u": far_u,
                    "near": [float(v) for v in bar + along * near_u],
                    "far": [float(v) for v in bar + along * far_u],
                },
                # Where the support hand takes the bar: the far elbow's tangent
                # point, on the axis; the pads straddle the tube there.
                "support_hold": [float(v) for v in hold],
                "tube_diameter": 2.0 * radius,
                "sponge_centroid": [float(v) for v in sponge_q.mean(axis=0)],
                "sponge_underside": float(sponge_q[:, 2].min()),
                "sponge_half_along_bar": half_along,
                "sponge_held": self._hold.pin is not None,
                "arms": {},
            }
            for role in ("work", "support"):
                name = self._arm_name(role)
                pos, quat_arm = env.arm(name).ee_pose(env.state)
                out["arms"][role] = {
                    "name": name,
                    "arm_id": list(env.scene.embodiment.iks).index(name),
                    "tcp": [float(v) for v in pos],
                    "quat_xyzw": [float(v) for v in quat_arm],
                    "gap": float(env.handles.arms[name].gripper_span * (1.0 - float(env.gripper(name).command))),
                }
            return out

        return self.bus.call(read)

    def _half_along(self, points: np.ndarray, pose) -> float:
        """Half the sponge's reach along the bar's own axis [m], as built.

        Measured off the body rather than read from its spec, so a re-sized
        sponge cannot leave this stale -- and projected onto the bar's axis
        rather than assuming which world axis that is.
        """
        along = _qrot(np.asarray(pose[3:7], dtype=np.float64), np.array([0.0, 0.0, 1.0]))
        u = np.asarray(points, dtype=np.float64) @ along
        return 0.5 * float(u.max() - u.min())

    # --------------------------------------------------------------------- grip
    def set_grip(self, width: Any = "open", arm: str = "work", ramp_steps: int = 40) -> dict[str, Any]:
        """Command a pad separation [m], or ``"open"``. Ramped, never stepped.

        A step input on foam is a punch: the penalty contact's response to a
        millimetre of overlap arriving in one control step is tens of newtons,
        and the sponge leaves before the jaws are closed.
        """
        name = self._arm_name(arm)
        span = float(self.env.handles.arms[name].gripper_span)
        gap = span if width == "open" else float(width)
        target = max(0.0, min(1.0, 1.0 - gap / span))

        def write(fraction: float) -> None:
            gripper = self.env.gripper(name)
            gripper.command = float(fraction)
            # Setting `command` alone leaves the finger targets where they were:
            # `apply` is what writes them into the control the solver reads.
            gripper.apply(self.env.pipeline.control.joint_target_q)

        start = float(self.bus.call(lambda: self.env.gripper(name).command))
        steps = max(1, int(ramp_steps))
        # ONE STEP PER WRITE, interleaved -- the ramp is the waiting. Writing
        # the whole ramp inside a single bus call assigns the last value and
        # nothing else: the sim never advances between the writes, so what the
        # foam sees is the step input this docstring warns about.
        for i in range(1, steps + 1):
            self.bus.call(write, start + (target - start) * (i / steps))
            self._wait_for_step(1)
        self._wait_for_step(SETTLE)
        return {"arm": name, "gap": gap, "command": target, "ramp_steps": steps}

    def grip_sponge(self, arm: str = "work", gap: float = GRIP_GAP) -> dict[str, Any]:
        """Take the sponge: close the jaws, then pin the foam under the pads to the pad frame.

        The jaws are closed first and the pins captured *after*, so what is
        carried is where the foam actually ended up under the pads rather than
        where it was before they touched it.
        """
        name = self._arm_name(arm)
        self.set_grip(gap, arm=name)

        def capture() -> dict[str, Any]:
            nodes = self._pad_patch(name, gap)
            q = self.env.pipeline.state_0.particle_q.numpy()[nodes]
            pos, quat = self.env.arm(name).ee_pose(self.env.state)
            inv = _qconj(np.asarray(quat, dtype=np.float64))
            local = np.array([_qrot(inv, row - np.asarray(pos, dtype=np.float64)) for row in q])
            pin = self.bridge.pin_nodes(
                "sponge", nodes, q.copy(), stiffness=PIN_STIFFNESS, damping=PIN_DAMPING, world=self._world
            )
            self._hold.arm_hold(name, pin, local)
            # How far the block's own underside hangs below the hand, measured
            # at the grasp rather than assumed. Taking it from the PINNED band
            # instead reads 6 mm where the sponge is 30 mm thick and drives
            # every later press 10 mm too deep -- the rig error that turned a
            # 0.6 N press into 37 N in rgbs/scripts/rgbs_prep_bench.py.
            whole = self.env.pipeline.state_0.particle_q.numpy()[self._sponge_parts()]
            self._carry = {
                "underside_drop": float(pos[2]) - float(whole[:, 2].min()),
                "quat": [float(v) for v in quat],
            }
            return {"nodes": int(len(nodes)), "pin": int(pin), "underside_drop": self._carry["underside_drop"]}

        out = self.bus.call(capture)
        self._wait_for_step(SETTLE)
        return {"arm": name, "gap": float(gap), "held": True, **out}

    def release_sponge(self, arm: str = "work") -> dict[str, Any]:
        """Let go: drop the pins first, then open the jaws.

        In that order. Opening first leaves the foam hanging on the pins with
        nothing under it, and the release then reads as a sponge that teleports
        when the pins go.
        """
        name = self._arm_name(arm)
        pin = self._hold.pin

        def drop() -> None:
            if pin is not None:
                self.bridge.unpin(pin)
            self._hold.disarm()

        self.bus.call(drop)
        self._carry = None
        self._wait_for_step(SETTLE)
        self.set_grip("open", arm=name)
        return {"arm": name, "released": pin is not None}

    # --------------------------------------------------------------------- plan
    def _solve_path(self, ik, targets: list[tuple[np.ndarray, np.ndarray]]) -> list[list[float]]:
        """Warm-started full-pose IK along a path -- the usbc/polish discipline.

        Each waypoint seeds the next, so the arm stays in one elbow basin for a
        whole motion instead of choosing a fresh one per point and snapping
        between them.
        """
        seed = self.env.state.joint_q.numpy().copy()
        out: list[list[float]] = []
        for pos, quat in targets:
            seed = np.asarray(
                ik.solve_q_full(seed, np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)),
                dtype=np.float64,
            )
            out.append([float(seed[c]) for c in ik.arm_coords])
        return out

    def _plan(
        self, role: str, path: list[tuple[np.ndarray, np.ndarray]], *, allow_unreachable: bool = False
    ) -> dict[str, Any]:
        """One arm's path as the joint trajectory the library's mover takes.

        **Refuses a path it cannot reach.** See :data:`REACH_TOLERANCE`.
        """
        name = self._arm_name(role)

        def solve() -> dict[str, Any]:
            ik = self.env.arm(name)
            waypoints = self._solve_path(ik, path)
            worst = 0.0
            seed = self.env.state.joint_q.numpy().copy()
            for q, (pos, quat) in zip(waypoints, path, strict=True):
                full = seed.copy()
                for c, v in zip(ik.arm_coords, q, strict=True):
                    full[c] = v
                worst = max(worst, float(ik.pose_errors(full, np.asarray(pos), np.asarray(quat))[0]))
            return {
                "arm": name,
                "arm_id": list(self.env.scene.embodiment.iks).index(name),
                "waypoints": waypoints,
                "worst_position_error": worst,
                "reachable": bool(worst <= REACH_TOLERANCE),
            }

        out = self.bus.call(solve)
        if not out["reachable"] and not allow_unreachable:
            raise RuntimeError(
                f"{name} cannot reach this path: worst waypoint misses by "
                f"{out['worst_position_error'] * 1e3:.1f} mm against a {REACH_TOLERANCE * 1e3:.0f} mm "
                "tolerance. Executing it anyway puts the hand somewhere it was never asked to be "
                "and every measurement downstream is then about that place instead."
            )
        return out

    def plan_support_hold(self, role: str = "support", standoff: float = SUPPORT_STANDOFF) -> dict[str, Any]:
        """The support hand onto the bar at its far elbow: over, then down astride the tube.

        Two legs, top-down with the jaws across the tube. Approached from
        ``standoff`` above -- the trap the page found is a hand that came in
        low and swept the elbow. The hold is a steadying one on a welded bar,
        so there is no lift; :meth:`set_grip` to ``tube_diameter`` plus 4 mm
        closes the pads on it.
        """
        state = self.get_prep_state()
        hold = np.asarray(state["support_hold"], dtype=np.float64)
        quat = wipe_quat(0.0)
        over = hold + np.array([0.0, 0.0, float(standoff)])
        down = hold + np.array([0.0, 0.0, SUPPORT_ABOVE_AXIS])
        out = self._plan(role, [(over, quat), (down, quat)])
        return {
            **out,
            "stations": {"over": over.tolist(), "hold": down.tolist()},
            "grip_gap": state["tube_diameter"] + SUPPORT_GRIP_EXTRA,
        }

    def plan_sponge_pick(self, standoff: float = 0.10) -> dict[str, Any]:
        """Bring the pads onto the sponge, over then down, top-down.

        The target is the sponge's centroid minus :meth:`pad_offset`, which is
        a 7 mm trim: on this hand the pads straddle the TCP, so aiming the TCP
        at the block is already very nearly aiming the pads at it. The offset
        used to be 80 mm and pointed the wrong way, and every pick planned
        through it closed a hand's length above the sponge.
        """
        state = self.get_prep_state()
        centroid = np.asarray(state["sponge_centroid"], dtype=np.float64)
        quat = wipe_quat(0.0)
        grip_point = centroid - _qrot(quat, self.pad_offset("work"))
        over = grip_point + np.array([0.0, 0.0, float(standoff)])
        out = self._plan("work", [(over, quat), (grip_point, quat)])
        return {**out, "grip_point": grip_point.tolist(), "sponge_centroid": centroid.tolist()}

    def plan_sponge_move(
        self,
        motion: str,
        depth: float = 0.004,
        clearance: float = CLEARANCE,
        cycles: int = 2,
        tilt_deg: float = TILT_DEG,
    ) -> dict[str, Any]:
        """A named sponge motion, as the work arm's joint trajectory.

        ``lift``      straight up to the clearance height, wherever it is.
        ``over``      across to the near end of the stripe, AT the clearance
                      height, leaning ``tilt_deg`` along the bar.
        ``press``     down onto the bar until the block's underside is ``depth``
                      into it, geometrically. The shipped take presses by
                      FORCE instead (:meth:`Wiper._press`): the leaning hand
                      puts the foam's lowest point where no flat measurement
                      predicts it, and free-hanging foam sags and creeps.
        ``stroke``    out to the far end of the stripe and back, at the current
                      height, ``cycles`` times.

        **Always cross at clearance.** Crossing at working height drags the
        sponge over whatever is between the mat and the bar -- the near block
        and the elbow -- and over the stripe the last stroke just cleared.
        """
        # The cheap guard first. Reading the world and then discovering there is
        # nothing in hand costs a bus round trip to raise the same error, and
        # buries it under whatever the state read fails on instead.
        if self._carry is None:
            raise RuntimeError(
                "plan_sponge_move needs the sponge in hand: the press height is measured from the "
                "block's underside at the grasp, and there is no such measurement before one."
            )
        if motion not in ("lift", "over", "press", "stroke"):
            raise ValueError(f"unknown sponge motion {motion!r}; have lift | over | press | stroke")
        state = self.get_prep_state()
        name = self._arm_name("work")
        tcp = np.asarray(state["arms"]["work"]["tcp"], dtype=np.float64)
        quat = wipe_quat(float(tilt_deg))
        # From the GRASP, and measuring it live is the mistake that looks like
        # the fix. What matters is not when it was measured but whether the foam
        # was SUPPORTED when it was: hanging free at clearance height the block
        # sags, and it creeps further every press -- measured 17.4, 18.4, then
        # 21.6 mm over three stations. Pressed against the bar it recovers to
        # 17.4, 17.1, 17.2 mm, and the grasp-time capture (16.9 mm, taken with
        # the sponge still resting on the mat) is that same supported number.
        # Planning off the free-hanging value aims the hand up to 4.4 mm high,
        # which turned a commanded 8 mm press into 3.6 mm and 0.00 N.
        drop = float(self._carry["underside_drop"])
        top = float(state["crown_z"])
        near = np.asarray(state["stripe"]["near"], dtype=np.float64)
        far = np.asarray(state["stripe"]["far"], dtype=np.float64)

        if motion == "lift":
            path = [(np.array([tcp[0], tcp[1], top + drop + float(clearance)]), quat)]
        elif motion == "over":
            path = [(np.array([near[0], near[1], top + drop + float(clearance)]), quat)]
        elif motion == "press":
            path = [(np.array([tcp[0], tcp[1], top + drop - float(depth)]), quat)]
        elif motion == "stroke":
            legs = []
            for _ in range(max(1, int(cycles))):
                legs += [np.array([far[0], far[1], tcp[2]]), np.array([near[0], near[1], tcp[2]])]
            path = [(np.asarray(p, dtype=np.float64), quat) for p in legs]
        else:  # pragma: no cover -- the vocabulary is checked on entry
            raise ValueError(f"unknown sponge motion {motion!r}")
        out = self._plan(name, path)
        return {**out, "motion": motion, "waypoint_count": len(path)}

    def plan_retreat(self, role: str = "work", distance: float = 0.13) -> dict[str, Any]:
        """A straight withdrawal upward, for a hand that has let go."""
        state = self.get_prep_state()
        tcp = np.asarray(state["arms"][role]["tcp"], dtype=np.float64)
        quat = np.asarray(state["arms"][role]["quat_xyzw"], dtype=np.float64)
        return self._plan(role, [(tcp + np.array([0.0, 0.0, float(distance)]), quat)])

    # -------------------------------------------------------------------- dwell
    def wait_steps(self, steps: int = 30) -> dict[str, Any]:
        """Dwell for N control steps, holding whatever the arms were last told.

        Reports what it actually waited, which is not always what was asked: the
        wait is bounded by a wall-clock timeout so a stalled sim cannot hang a
        graph node forever.
        """
        return {"waited": self._wait_for_step(int(steps))}

    def wait_wipe(
        self, patch: int = 0, until: float = 0.95, max_steps: int = 600, stall_steps: int = 150
    ) -> dict[str, Any]:
        """Watch the stripe clear while the policy loop holds the press.

        Three exits and they are not the same outcome: reached, stalled, or out
        of budget. A stall on this task usually means the press is too light --
        the pressure floor cuts the low-pressure edge of the contact long
        before the geometric wrap does -- or the strokes are not reaching the
        end of the stripe.
        """

        def read() -> tuple[float, float]:
            field = self._taint().read(self._world)
            return float(field["patch_fractions"][int(patch)]), float(field["wipe_force"])

        cleared, press = self.bus.call(read)
        start, waited, last_gain = cleared, 0, 0
        presses: list[float] = []
        while waited < int(max_steps):
            self._wait_for_step(6)
            waited += 6
            now, press = self.bus.call(read)
            presses.append(press)
            if now > cleared + 1.0e-6:
                cleared, last_gain = now, waited
            if cleared >= float(until):
                break
            if waited - last_gain >= int(stall_steps):
                break
        return {
            "patch": int(patch),
            "cleared": float(cleared),
            "gained": float(cleared - start),
            "done": bool(cleared >= float(until)),
            "stalled": bool(cleared < float(until) and waited - last_gain >= int(stall_steps)),
            "waited": int(waited),
            "press_mean": float(np.mean(presses)) if presses else 0.0,
        }

    def _wait_for_step(self, steps: int, timeout_s: float = 10.0) -> int:
        start = int(self.bus.snapshot.step)
        deadline = time.monotonic() + timeout_s
        while int(self.bus.snapshot.step) < start + steps and time.monotonic() < deadline:
            time.sleep(0.002)
        return int(self.bus.snapshot.step) - start


def register_surface_prep_tools(registry, env, bus) -> None:
    """Register the suite's ``sim.*`` tools."""
    tools = ToolsSurfacePrep(env, bus)
    for name, fn, summary in tools.entries():
        registry.register_callable(name, fn, summary=summary, tags=("privileged",))


# ---------------------------------------------------------------------------
# The take
# ---------------------------------------------------------------------------


class _DirectBus:
    """A stand-in for the tool bus, for a caller already on the stepping thread.

    :class:`ToolsSurfacePrep` exists to be driven by a graph, whose nodes run on
    another thread and therefore hand every world read to the sim thread through
    ``bus.call``. The take is already there -- it runs between steps, from
    inside the loop -- so the marshalling is not just unnecessary, it would
    deadlock.

    Deliberately without a ``snapshot``: the tools' ``_wait_for_step`` waits on
    the step counter *advancing*, which cannot happen while the take is between
    two of its own yields. Anything that dwells belongs in the choreography as
    yielded steps, and reaching for a tool that dwells should fail loudly rather
    than hang.
    """

    def call(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


class Wiper:
    """The take, as a generator: one control step per ``next()``.

    Two ways to play it, one choreography. :meth:`steps` yields once per control
    step with the arms' targets set and never steps the sim -- that is what
    :class:`RoutineSurfacePrep` drives, handing :meth:`action` to ``Env.step``
    between yields. :meth:`play` wraps it and steps ``sim`` itself, which is
    what the live page wants: ``for _ in Wiper(sim).play(): ...`` plays one
    control step per tick.

    ``sim`` is anything with the stepping surface -- ``arm()``, ``state``,
    ``handles``, ``idle_action()``, ``pipeline``, ``scene``,
    ``post_step_hooks`` -- so an ``Env`` and a bare ``Sim`` both serve.

    **The choreography, and the traps each part of it records.**

    * The support hand steadies the bar at the far elbow's tangent point,
      approached from :data:`SUPPORT_STANDOFF` above; low approaches swept the
      elbow. Skipped, with a note, if its IK misses by over 5 mm or 15 deg.
    * The work hand picks the sponge top-down at the -90 deg jaw roll, so the
      75 mm axis lies along the bar, and pins the foam under the pads.
    * It crosses at clearance leaning :data:`TILT_DEG` along the bar and
      presses BY FORCE -- a millimetre at a time until the field reads
      :data:`PRESS_N` -- with a travel budget of the clearance plus 30 mm.
    * Strokes run between :func:`stroke_span`'s ends in
      :data:`STROKE_SEGMENTS` segments with a running seat trim, and after the
      first outward stroke the far end is re-seated by force too: the arm sags
      differently at the far end of its reach.
    * The sponge goes back where it came from, upright; both hands retreat;
      :data:`TAIL` settled steps make the goal's two-second hold happen inside
      the take.

    The live page's twin of the clip (``rgbs/scripts/surface_cleaning.py``)
    plays exactly this on exactly this scene.
    """

    MODE = "joint_position"
    """The action space this emits into, read by ``SolutionRoutine``."""

    def __init__(
        self,
        sim,
        *,
        layout: Layout = LAYOUT,
        tilt_deg: float = TILT_DEG,
        grip_gap: float = GRIP_GAP,
        press_n: float = PRESS_N,
        stroke_speed: float = STROKE_SPEED,
        stroke_cycles: int = STROKE_CYCLES,
        clearance: float = CLEARANCE,
        steady: bool = True,
        verbose: bool = True,
    ) -> None:
        self.sim = sim
        self.layout = layout
        self.tilt = float(tilt_deg)
        self.grip_gap = float(grip_gap)
        self.press_n_target = float(press_n)
        self.speed = float(stroke_speed)
        self.cycles = int(stroke_cycles)
        self.clearance = float(clearance)
        self.steady = bool(steady)
        self.verbose = bool(verbose)
        self.tools = ToolsSurfacePrep(sim, _DirectBus(), layout=layout)
        """Built per take on purpose: it owns the :class:`SpongeHold`, and
        ``EnvBase.reset`` has just dropped every post-step hook a policy added
        and the bridge every pin, so a carried-over hold would re-aim an id that
        no longer exists."""
        self.arms = arm_slots(sim)
        self.work = sim.handles.work_arm or self.arms[0]
        self.support = sim.handles.support_arm or (self.arms[1] if len(self.arms) > 1 else "")
        self.dof = len(sim.idle_action(self.MODE)) // len(self.arms)
        self.held = {name: self._q(name) for name in self.arms}
        self.grip = dict.fromkeys(self.arms, OPEN)
        self.seed = {name: np.asarray(sim.state.joint_q.numpy(), dtype=np.float64) for name in self.arms}
        self.refused: list[str] = []
        self.log: list[str] = []
        self.drop = 0.0
        """How far the sponge's underside hangs below the TCP at the grasp [m]."""
        self.trim = 0.0
        """Running seat-height correction [m], from the press readings."""
        self.cleaned = 0.0
        self.press_n = 0.0
        self.slide = 0.0
        self.pinned_nodes = 0
        self.steadied = False
        self.phase = "idle"

    # ------------------------------------------------------------- helpers
    def _q(self, name: str) -> np.ndarray:
        ik = self.sim.arm(name)
        return np.asarray(self.sim.state.joint_q.numpy(), dtype=np.float64)[ik.arm_coords]

    def _say(self, text: str) -> None:
        self.log.append(text)
        if self.verbose:
            print(f"[wipe] {text}", flush=True)

    def action(self) -> np.ndarray:
        """The action holding every arm where the take last put it."""
        a = np.asarray(self.sim.idle_action(self.MODE), dtype=np.float64)
        for name in self.arms:
            s = arm_slot(self.sim, name) * self.dof
            a[s : s + self.dof - 1] = self.held[name][: self.dof - 1]
            a[s + self.dof - 1] = self.grip[name]
        return a

    def _tick(self):
        """One control step: the targets are set; whoever drives steps."""
        yield

    def _hold_steps(self, n: int):
        for _ in range(int(n)):
            yield from self._tick()

    def _field(self) -> dict:
        field = self.sim.scene.by_name["taint"].read()
        self.cleaned = float(field["cleared_fraction"])
        self.press_n = float(field["wipe_force"])
        self.slide = float(field["slide_speed"])
        return field

    def _sponge_particles(self) -> np.ndarray:
        return np.asarray(self.sim.scene.by_name["sponge"].particles, dtype=np.int64)

    def sponge_points(self) -> np.ndarray:
        return self.sim.pipeline.state_0.particle_q.numpy()[self._sponge_particles()]

    def reach(self, arm: str, target, quat) -> tuple[float, float]:
        """How well ``arm`` can hold ``quat`` at ``target`` [world]: (mm, deg)."""
        ik = self.sim.arm(arm)
        target = np.asarray(target, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)
        sol = ik.solve_q_full(self.seed[arm].copy(), target, quat)
        p, r = ik.pose_errors(sol, target, quat)
        return p * 1e3, math.degrees(r)

    def _leg(
        self,
        arm: str,
        target,
        quat,
        *,
        speed: float = 0.06,
        label: str = "",
        gate: bool = True,
        max_pos: float = LEG_MAX_POS,
        max_rot: float = LEG_MAX_ROT,
        stride: int = 3,
    ):
        """Solve a straight leg, gate it, drive it one waypoint per step.

        The IK is solved every ``stride``-th waypoint and the joint targets are
        interpolated between -- a millimetre or two of straight-line joint
        interpolation is invisible, and a full-pose solve per control step was
        what made the page stall for seconds at the head of every stroke.

        ``gate`` is off for the free approaches only. A leg starting from the
        arm's home pose cannot hold the working orientation at its first
        waypoint -- the residual runs to a hand's length and shrinks as the arm
        converges -- so gating one refuses every approach there is. With
        nothing in the hand that convergence is harmless; with the sponge in
        it, it is a throw.
        """
        ik = self.sim.arm(arm)
        start, q_start = ik.ee_pose(self.sim.state)
        start = np.asarray(start, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)
        q_start = np.asarray(q_start, dtype=np.float64)
        # Long enough for the distance at `speed`, and for the turn at ~60 deg/s.
        cosine = abs(float(np.dot(q_start / np.linalg.norm(q_start), quat / np.linalg.norm(quat))))
        turn = 2.0 * math.acos(min(1.0, cosine))
        n = max(2, int(np.linalg.norm(target - start) / float(speed) * 60.0), int(turn / math.radians(60.0) * 60.0))
        stride = max(1, int(stride))
        knots = sorted({*range(stride, n + 1, stride), n})
        sols, worst_p, worst_r, seed = {}, 0.0, 0.0, self.seed[arm]
        for k in knots:
            f = k / n
            step = start + (target - start) * f
            q_k = _slerp(q_start, quat, f)
            sol = ik.solve_q_full(seed, step, q_k)
            p, r = ik.pose_errors(sol, step, q_k)
            worst_p, worst_r = max(worst_p, p), max(worst_r, r)
            sols[k] = np.asarray(sol, dtype=np.float64)
            seed = sol
        note = f"{label or arm}: {worst_p * 1e3:.1f} mm / {math.degrees(worst_r):.1f} deg over {n} steps"
        if gate and (worst_p > max_pos or worst_r > max_rot):
            self.refused.append(note)
            self._say("REFUSED " + note)
            return
        if label:  # the force loop's millimetre legs are unlabelled, and would drown the log
            self._say(note)
        prev_k, prev_q = 0, np.asarray(self.seed[arm], dtype=np.float64)
        for k in knots:
            for j in range(prev_k + 1, k + 1):
                f = (j - prev_k) / (k - prev_k)
                q = prev_q + (sols[k] - prev_q) * f
                self.held[arm] = q[list(ik.arm_coords)]
                yield from self._tick()
            self.seed[arm] = sols[k]
            prev_k, prev_q = k, sols[k]

    def _reachable(self, arm: str, candidates, *, max_pos: float = 0.005, max_rot_deg: float = 15.0):
        """The first ``(target, quat, label)`` the arm can hold, or ``None``."""
        for target, quat, label in candidates:
            mm, deg = self.reach(arm, target, quat)
            if mm <= max_pos * 1e3 and deg <= max_rot_deg:
                return target, quat, label
            self._say(f"skip {label}: IK {mm:.1f} mm / {deg:.1f} deg")
        return None

    def _ramp_grip(self, arm: str, gap, steps: int = 60):
        """Ease the jaws to a pad separation [m], or ``"open"``, one step at a time.

        Ramped for :meth:`ToolsSurfacePrep.set_grip`'s reason, which is about the
        foam and not about the servo: a step input on a penalty contact is a
        punch, and a millimetre of overlap arriving in one control step is tens
        of newtons -- the sponge leaves before the jaws are closed.
        """
        span = float(self.sim.handles.arms[arm].gripper_span)
        width = span if gap == "open" else float(gap)
        start, target = float(self.grip[arm]), max(0.0, min(1.0, 1.0 - width / span))
        for i in range(1, int(steps) + 1):
            self.grip[arm] = start + (target - start) * (i / int(steps))
            yield from self._tick()

    def _take_hold(self) -> None:
        """Pin the foam under the pads to the pad frame, and carry it there.

        :meth:`ToolsSurfacePrep.grip_sponge`'s capture, without its bus round
        trips: the nodes are chosen AFTER the jaws have closed, so what is
        carried is where the foam actually ended up under the pads rather than
        where it was before they touched it.
        """
        sim, tools, arm = self.sim, self.tools, self.work
        nodes = tools._pad_patch(arm, self.grip_gap)
        q = sim.pipeline.state_0.particle_q.numpy()[nodes]
        pos, quat = sim.arm(arm).ee_pose(sim.state)
        pos = np.asarray(pos, dtype=np.float64)
        inv = _qconj(np.asarray(quat, dtype=np.float64))
        local = np.array([_qrot(inv, row - np.asarray(pos, dtype=np.float64)) for row in q])
        pin = tools.bridge.pin_nodes(
            "sponge", nodes, q.copy(), stiffness=PIN_STIFFNESS, damping=PIN_DAMPING, world=tools._world
        )
        tools._hold.arm_hold(self.work, pin, local)
        # Off the WHOLE block, never off the pinned band: the pinned nodes are a
        # disc at mid-height, so measuring the hang from them reads 6 mm where
        # the sponge is 30 mm thick and drives every press 10 mm too deep.
        whole = sim.pipeline.state_0.particle_q.numpy()[self._sponge_nodes()]
        self.drop = float(pos[2]) - float(whole[:, 2].min())
        self.pinned_nodes = int(len(nodes))
        tools._carry = {"underside_drop": self.drop, "quat": [float(v) for v in quat]}
        self._say(f"holding the sponge: {len(nodes)} nodes pinned, underside {self.drop * 1e3:.1f} mm below the TCP")

    def _let_go(self) -> None:
        """Drop the pins, in that order relative to the jaws.

        Opening first leaves the foam hanging on the pins with nothing under it,
        and the release then reads as a sponge that teleports when they go.
        """
        hold = self.tools._hold
        if hold.pin is not None:
            self.tools.bridge.unpin(hold.pin)
        hold.disarm()
        self.tools._carry = None

    def _sponge_nodes(self) -> np.ndarray:
        """This world's sponge particles, as indices into the fleet's array."""
        return self.tools._sponge_parts()

    def _field(self) -> dict:
        field = self.sim.scene.by_name["taint"].read(self.tools._world)
        self.cleaned = float(field["cleared_fraction"])
        self.press_n = float(field["wipe_force"])
        self.slide = float(field["slide_speed"])
        return field

    def _press(self, xy, z_start: float, quat, *, target_n: float, max_travel: float, label: str = "press"):
        """Descend in millimetre steps until the marker field reads ``target_n`` of
        press, or ``max_travel`` is spent. Returns the seat height.

        A force loop rather than a geometric depth: the hand leans along the
        bar, so the sponge's lowest point is not where a flat-grasp measurement
        put it, and free-hanging foam sags and creeps besides.
        """
        z, spent, force = float(z_start), 0.0, 0.0
        while spent < max_travel:
            z -= 0.001
            spent += 0.001
            yield from self._leg(self.work, [xy[0], xy[1], z], quat, speed=0.02, label="", gate=False)
            yield from self._hold_steps(4)
            field = self._field()
            force = float(field["wipe_force"]) if field["touching"] else 0.0
            if force >= target_n:
                break
        self._say(f"{label}: seated {spent * 1e3:.0f} mm down at {force:.2f} N")
        return z

    def _stroke(self, a, b, quat, label: str, segments: int = STROKE_SEGMENTS):
        """Walk from ``a`` to ``b`` in segments, trimming the seat height between
        them so the press stays near its target: the arm sags differently along
        the bar and the pinned foam creeps in the jaws, so one seat height
        measured at one end reads 1.3 N at the other and 20 N a stroke later.
        """
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        for i in range(int(segments)):
            p1 = a + (b - a) * (i + 1) / segments
            yield from self._leg(
                self.work,
                [p1[0], p1[1], p1[2] + self.trim],
                quat,
                speed=self.speed,
                label=f"{label}.{i}",
                max_rot=CARRY_MAX_ROT,
            )
            field = self._field()
            force = float(field["wipe_force"]) if field["touching"] else 0.0
            if force < 0.6 * self.press_n_target:
                self.trim -= TRIM_STEP
            elif force > 2.5 * self.press_n_target:
                self.trim += TRIM_STEP
            self.trim = float(np.clip(self.trim, *TRIM_BOUNDS))

    # ------------------------------------------------------------- the take
    def steps(self):
        """The whole take, without stepping: one yield per control step."""
        sim, L = self.sim, self.layout
        wq = wipe_quat(self.tilt)  # the wipe: leaning along the bar
        pq = wipe_quat(0.0)  # the pick and the set-down: top-down
        bar_body = int(sim.handles.entity("bar").body)
        pose = sim.pipeline.state_0.body_q.numpy()[bar_body]
        bar = np.asarray(pose[:3], dtype=np.float64)
        along = _qrot(pose[3:7], np.array([0.0, 0.0, 1.0]))  # world +y
        crown = float(bar[2]) + L.tube_radius

        # --- steady the bar with the support hand ------------------------------
        self.steadied = False
        if self.steady and self.support:
            self.phase = "steady"
            sq = wipe_quat(0.0)
            at = bar + along * L.support_u  # the far elbow's tangent point
            tube = 2.0 * L.tube_radius
            mm, deg = self.reach(self.support, at + np.array([0.0, 0.0, SUPPORT_ABOVE_AXIS]), sq)
            self._say(f"support hold at the far elbow: IK {mm:.1f} mm / {deg:.1f} deg")
            if mm < 5.0 and deg < 15.0:
                self.steadied = True
                yield from self._leg(
                    self.support,
                    at + np.array([0.0, 0.0, SUPPORT_STANDOFF]),
                    sq,
                    speed=0.20,
                    label="support over the bar",
                    gate=False,
                )
                yield from self._hold_steps(10)
                # pads straddle the tube: TCP a little above the axis
                yield from self._leg(
                    self.support,
                    at + np.array([0.0, 0.0, SUPPORT_ABOVE_AXIS]),
                    sq,
                    speed=0.05,
                    label="support down on the bar",
                    gate=False,
                )
                yield from self._hold_steps(10)
                yield from self._ramp_grip(self.support, tube + SUPPORT_GRIP_EXTRA, steps=40)
                yield from self._hold_steps(SETTLE)

        # --- the pick ---------------------------------------------------------
        # The PADS onto the block, not the TCP: a 7 mm trim on this hand.
        self.phase = "pick"
        centroid = self.sponge_points().mean(axis=0)
        pick = centroid - _qrot(pq, np.asarray(PAD_CONTACT_OFFSET, dtype=np.float64))
        yield from self._leg(
            self.work, pick + np.array([0.0, 0.0, 0.10]), pq, speed=0.20, label="over the sponge", gate=False
        )
        yield from self._hold_steps(20)
        yield from self._leg(self.work, pick, pq, speed=0.05, label="down onto the sponge", gate=False)
        yield from self._hold_steps(20)
        yield from self._ramp_grip(self.work, self.grip_gap, steps=60)
        yield from self._hold_steps(SETTLE)
        self._take_hold()
        yield from self._hold_steps(SETTLE)

        # --- to the bar --------------------------------------------------------
        self.phase = "carry"
        clear_z = crown + self.drop + self.clearance
        half_sponge = 0.5 * L.sponge_size[1]
        # The hand leans toward +y, which lifts the sponge's far edge: the
        # contact patch trails the pinned centre by a third of the block. So the
        # far end of the stroke runs a block-third past the stripe, up to the
        # elbow; the near end stops where the block's own edge reaches -- and
        # never within a jaw's width of the steadying hand.
        u0, u1 = stroke_span(L, half_sponge)
        way = self._reachable(
            self.work,
            [
                ([pick[0], pick[1], clear_z], pq, "lift the sponge, upright"),
                ([pick[0], pick[1], clear_z], wq, "lift the sponge, leaning"),
                ([pick[0], pick[1], pick[2] + 0.06], pq, "lift the sponge a little"),
            ],
        )
        if way is not None:
            yield from self._leg(self.work, way[0], way[1], speed=0.10, label=way[2], max_rot=CARRY_MAX_ROT)
        yield from self._hold_steps(10)
        start = bar + along * u0
        yield from self._leg(
            self.work,
            [start[0], start[1], clear_z],
            wq,
            speed=0.10,
            label="over the near end of the stripe, leaning",
            max_rot=CARRY_MAX_ROT,
        )
        yield from self._hold_steps(10)
        seat_z = yield from self._press(
            (start[0], start[1]),
            clear_z,
            wq,
            target_n=self.press_n_target,
            max_travel=self.clearance + 0.030,
            label="press onto the bar",
        )
        seat = np.array([start[0], start[1], seat_z])
        yield from self._hold_steps(SETTLE)
        self._field()
        self._say(f"pressing at {self.press_n:.2f} N, cleaned {self.cleaned:.2f}")

        # --- the wipe ----------------------------------------------------------
        self.phase = "wipe"
        far = bar + along * u1
        far_seat = np.array([far[0], far[1], seat_z])
        for cycle in range(self.cycles):
            yield from self._stroke(seat, far_seat, wq, f"stroke {cycle} out")
            self._say(
                f"cycle {cycle} out: cleaned {self.cleaned:.2f}, press {self.press_n:.2f} N, trim {self.trim * 1e3:+.1f} mm"
            )
            if cycle == 0:
                # The arm sags differently at the far end of its reach: re-seat
                # there by force too, and stroke between the two seats.
                far_z = yield from self._press(
                    (far[0], far[1]),
                    far_seat[2] + self.trim + RESEAT_LIFT,
                    wq,
                    target_n=self.press_n_target,
                    max_travel=RESEAT_TRAVEL,
                    label="re-seat at the far end",
                )
                far_seat = np.array([far[0], far[1], far_z - self.trim])
            if self.cleaned >= PATCH_CLEARED:
                break
            yield from self._stroke(far_seat, seat, wq, f"stroke {cycle} back")
            self._say(
                f"cycle {cycle} back: cleaned {self.cleaned:.2f}, press {self.press_n:.2f} N, trim {self.trim * 1e3:+.1f} mm"
            )
            if self.cleaned >= PATCH_CLEARED:
                break
        pos, _ = sim.arm(self.work).ee_pose(sim.state)
        yield from self._leg(
            self.work, [pos[0], pos[1], clear_z], wq, speed=0.06, label="off the bar", max_rot=CARRY_MAX_ROT
        )
        yield from self._hold_steps(10)

        # --- put it back ---------------------------------------------------------
        # Where it came from, upright, which is what the clip does. The goal's
        # clock starts when the sponge is off the bar, still and out of both
        # hands, so the set-down is not raced against it.
        self.phase = "return"
        way = self._reachable(
            self.work,
            [
                ([pick[0], pick[1], pick[2] + 0.06], wq, "back over the mat, leaning"),
                ([pick[0], pick[1], pick[2] + 0.06], pq, "back over the mat, upright"),
                ([pick[0], pick[1], pick[2] + 0.03], wq, "back over the mat, low"),
            ],
        )
        if way is not None:
            yield from self._leg(self.work, way[0], way[1], speed=0.12, label=way[2], max_rot=CARRY_MAX_ROT)
        yield from self._hold_steps(10)
        yield from self._leg(
            self.work,
            [pick[0], pick[1], pick[2] + 0.002],
            pq,
            speed=0.05,
            label="set the sponge down, upright",
            max_rot=CARRY_MAX_ROT,
        )
        yield from self._hold_steps(10)
        self._let_go()
        yield from self._ramp_grip(self.work, "open", steps=30)
        yield from self._hold_steps(15)
        yield from self._leg(self.work, [pick[0], pick[1], pick[2] + 0.16], pq, speed=0.10, label="retreat", gate=False)
        if self.steadied:
            yield from self._ramp_grip(self.support, "open", steps=30)
            yield from self._hold_steps(10)
            pos, _ = sim.arm(self.support).ee_pose(sim.state)
            yield from self._leg(
                self.support,
                [pos[0], pos[1], pos[2] + 0.12],
                wipe_quat(0.0),
                speed=0.10,
                label="support retreat",
                gate=False,
            )
        self._field()
        self.phase = "done"
        self._say(f"done: cleaned {self.cleaned:.3f}; {len(self.refused)} legs refused")
        # Two settled seconds is what the goal's `hold-for` needs, and the
        # episode is scored on the world standing still rather than on the last
        # motion having happened.
        yield from self._hold_steps(TAIL)

    def play(self):
        """The take, stepping ``sim`` itself: ``for _ in wiper.play(): ...``, or
        hand it to the live page, which plays one step per tick."""
        for _ in self.steps():
            self.sim.step(self.action(), mode=self.MODE)
            yield


class RoutineSurfacePrep:
    """The take, as the benchmark's scripted policy drives it.

    ``SolutionRoutine`` asks :meth:`act` for an action once per control step and
    hands it to ``Env.step``; this advances the :class:`Wiper` one yield and
    returns its :meth:`Wiper.action`. Rebuilt per episode by :meth:`reset`,
    because the hold and the pins do not survive one.

    What it does on the shipped scene is recorded in ``assets/prep/PROVENANCE.md``.
    """

    MODE = Wiper.MODE

    def __init__(self, env, **kwargs: Any) -> None:
        self.env = env
        self.kwargs = dict(kwargs)
        self.reset()

    def reset(self) -> None:
        self.wiper = Wiper(self.env, **self.kwargs)
        self._steps = self.wiper.steps()

    @property
    def refused(self) -> list[str]:
        return self.wiper.refused

    @property
    def cleaned(self) -> float:
        return self.wiper.cleaned

    def act(self) -> np.ndarray:
        try:
            next(self._steps)
        except StopIteration:
            pass
        return self.wiper.action()

    def row_extras(self) -> dict[str, float]:
        w = self.wiper
        return {
            "cleaned": float(w.cleaned),
            "press_n": float(w.press_n),
            "slide_mm_s": float(w.slide * 1e3),
            "pinned_nodes": float(w.pinned_nodes),
            "underside_drop_mm": float(w.drop * 1e3),
            "trim_mm": float(w.trim * 1e3),
            "steadied": float(w.steadied),
            "legs_refused": float(len(w.refused)),
        }
