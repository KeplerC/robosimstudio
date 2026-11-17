# SPDX-License-Identifier: Apache-2.0
"""The peel family's code: two privileged tools, and the two-handed film peel.

**The tools** are what is left after the generic surface took the rest. This file used to hold
four tools -- a plan, a bond reading, a latch and a release -- and three of them
were the *generic* thing written once per suite:

============================  ====================================================
``sim.plan_peel``             split along its three seams. Which tab and which way
                              it peels is :meth:`ToolsTapePeel.get_tab` below; the
                              IK legs are ``motion.plan_joint`` /
                              ``motion.plan_linear``; the stage cursor is a graph
                              edge.
``sim.get_bond_state``        ``sim.get_states``, which reads the ten states the
                              ``tape`` record *declares* rather than the four this
                              file computed by hand -- and reads the **film's**
                              ten when ``film_peel`` runs, which this one could
                              not.
``sim.latch_tape``            ``sim.attach``, which three suites had each written.
``sim.release_tape``          ``sim.detach``.
============================  ====================================================

**The oracle.** :meth:`ToolsTapePeel.get_tab` is the privileged body of a
perception subgraph and is labelled as one. It answers the question a camera
would have to answer -- where is the free end of this strip, which way does the
peel run, how wide is the thing to take hold of -- and it answers it by reading
the particles, which is exactly what a real rig cannot do. It plans nothing.

**The anchor, which does not generalise and stays.** A layout with a rigid
pull-tab holds it at its rest position with the task's own ``HandlePin`` -- that
is what makes it a pull-tab rather than a drooping flap. Once a gripper has the
tab, that pin has to let go, and until it does two kinematic holds fight over the
same particles: the pin wins, because it targets a fixed rest position while the
dragger only asks. The tab does not move, the hand lifts away from it, and not a
single bond breaks. That was the whole of this graph's first working run -- every
node green, 220 lift waypoints executed, ``broken_fraction`` 0.0.

``sim.attach`` cannot do it, and should not learn how: releasing a task's own
anchor is a fact about *this* layout's staging, not about holding a deformable.
So it is a tool of its own, and the ordering that used to be welded inside
``latch`` -- attach, prove the hold, *then* release the anchor -- is now three
lines of a node script, where a reader can see it.

**The routine** (:class:`RoutineFilmPeel`, 2026-09-10) is what ``panel`` and
``panel_gentle`` ship since the film moved onto the phone: the clip's
choreography, both hands. The graph ``gt`` still drives ``handle`` and cannot
drive these two -- its haul is one straight line from a latched tab, and a film
on a face leaning back 30 degrees has to be folded back *down the face* in an
arc that the graph's ``lift`` landmark cannot state. Everything the routine
stands on is measured and recorded on its constants: which corner the holding
hand takes and with what tool angle, where the pinch on the tab sits relative
to the pad geometry, the band of tool angles the right arm can hold through
the whole arc, and the orientation the carry has to change to.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from robosimstudio.arms import arm_slot, arm_slots
from robosimstudio.core.math import quat_slerp, quat_to_mat
from robosimstudio.solvers.grasp.antipodal import grasp_quat
from robosimstudio_bench.solutions.base import solution_param

from .peel.phone import PHONE_SIZE
from .peel.tape import GRASP_RADIUS
from .staging import CELL

__all__ = ["HOLD_MARGIN", "LATCH_RADIUS", "RoutineFilmPeel", "ToolsTapePeel", "register_tape_peel_tools"]

LATCH_RADIUS = GRASP_RADIUS
"""How near the tool-centre point a particle must be for the grasp to count [m].

The *miss* test, and only that. Wide enough to catch a tab the grasp landed a few
millimetres off, narrow enough that it cannot reach the far side of the strip --
and the same number the tape's own ``held`` reading uses, because "did a latch
take it" and "is it held" must be one question or the metric describes a grasp
nobody made. It was three copies of 0.02 before the asset owned it.

The rate the hold walks at is no longer this file's business, and the constant
it used to re-export -- ``rgbs/tasks/rgbs/assets/peel/tape.PIN_MAX_STEP``, 0.6 mm a substep --
is now only the *handle pin's*. ``sim.attach`` walks its own at 0.5 mm, slightly
slower, for the same reason: a pin that teleports its particles to the hand
injects the whole grasp error in one substep and detonates the solver. Measured
on ``handle`` the conservative default peels a larger fraction than the tape's
own number did, so the two are not in tension and there is nothing to reconcile.
"""

TAB_EDGE_DROP = 0.005
"""How far below the tab's top edge a flap is taken [m].

A drooping flap is grasped at its *upper* edge, not its centroid. Sinking the
grasp site to the centroid also sinks the gripper chassis -- which rides several
centimetres below the wrist -- onto the strip. Just below the top edge keeps the
palm clear and starts the peel next to the fold, where it should start.
"""

HOLD_MARGIN = 0.01
"""How much wider than the declared column the hold reaches [m].

A real wide film is machine-peeled by a clamp bar across its full edge, and a
sphere around the tool centre is that bar only while the tab is narrower than it:
on the 140 mm film a 2 cm sphere seized 7 of the 15 edge particles and the unheld
wings swung from the held centre -- reading as the film dangling far below a hand
that had it. So the hold takes the whole declared column, and
:data:`LATCH_RADIUS` stays the miss test either way.
"""


def register_tape_peel_tools(registry, env, bus) -> None:
    """Register ``sim.get_tab`` and ``sim.release_anchor``."""
    tools = ToolsTapePeel(env, bus)
    for name, fn, summary in tools.entries():
        registry.register_callable(name, fn, summary=summary, tags=("privileged",))


class ToolsTapePeel:
    """The tape suite's privileged tool implementations."""

    def __init__(self, env, bus) -> None:
        self.env = env
        self.bus = bus

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

        An entity's ``particles`` and its named sets are the **template's**
        indices, like every index resolved at bind, while ``particle_q`` on a
        wave holds one block per world. Unshifted, the hold radius below would be
        measured over world 0's tab -- and every world would be handed the first
        one's geometry to grasp.
        """
        worlds = getattr(self.env.scene, "worlds", None)
        return self._world * int(getattr(worlds, "particles_per_world", 0) or 0)

    def entries(self) -> list[tuple[str, Any, str]]:
        return [
            ("sim.get_tab", self.get_tab, "The strip's free end: where it is, which way it peels, what to hold."),
            ("sim.release_anchor", self.release_anchor, "Let the layout's rigid pull-tab go, once a hand has it."),
        ]

    # ----------------------------------------------------------------- oracle
    def get_tab(self, entity: str = "") -> dict[str, Any]:
        """Where the free end is, which way the peel runs, and what to take hold of.

        The oracle body of ``perceive_tab``. Everything here is a property of the
        *strip as it lies now*, read off the particles -- which is why it is
        privileged, and why the vision body has to earn the same four numbers
        from a camera.

        ``rigid`` is the one that decides the grasp, and it is the layout
        speaking rather than a mode anybody chose: a pinch needs something stiff
        to close on, so a strip with a declared rigid pull-tab is pinched and a
        strip whose tab droops down the box face is taken from above. The
        scripted routine derives it the same way, from the same field.

        The bond front is deliberately *not* here. It is a published state of the
        tape and ``sim.get_states`` reads all ten of them; answering it twice is
        how a suite ends up with a metric that disagrees with its own evaluator.
        """

        def read() -> dict[str, Any]:
            name = entity or self.env.task.deformables[0].name
            spec = next(c for c in self.env.task.deformables if c.name == name)
            body = self.env.scene.by_name[name]
            # This world's probe, whose named sets carry this world's particles.
            # ``update_probe`` refreshes the first world's, which on a wave hands
            # every episode the tab of the first one.
            probe = body.probe_for(self._world)
            probe.update(self.env.state)
            feature = str(getattr(spec, "handle_set", "") or "")
            if feature:
                point = probe.centroid(f"{name}/{feature}")
            else:
                point = probe.centroid(f"{name}/tab")
                point[2] = probe.height_band(f"{name}/tab")[1] - TAB_EDGE_DROP
            held = np.asarray(body.set_indices(feature) if feature else body.particles) + self._particle_offset()
            spread = float(np.linalg.norm(self.env.state.particle_q.numpy()[held] - point, axis=1).max())
            return {
                "entity": name,
                "point": _vec3(point),
                "peel_dir": _vec3(np.asarray(spec.peel_back_dir, dtype=np.float64)),
                "feature": feature,
                "rigid": bool(feature),
                "grasp_radius_m": float(LATCH_RADIUS),
                "hold_radius_m": spread + HOLD_MARGIN,
                "anchored": self._anchored(body),
            }

        return self.bus.call(read)

    # ----------------------------------------------------------------- anchor
    def release_anchor(self, entity: str = "") -> dict[str, Any]:
        """Let the layout's rigid pull-tab go. Harmless where there is no pin.

        Call it *after* a hold is proven, never before: until a gripper has the
        tab this pin is what holds it at rest, and a graph that released it first
        would be lifting against a flap.
        """

        def let_go() -> dict[str, Any]:
            name = entity or self.env.task.deformables[0].name
            body = self.env.scene.by_name[name]
            was = self._anchored(body)
            body.release_handle(self._world)
            return {"released": was, "entity": name}

        return self.bus.call(let_go)

    def _anchored(self, body) -> bool:
        """Whether THIS world's tab is still pinned.

        The pin holds and releases each world's tab on its own -- one hand
        closing on its tab is not a reason for another episode's strip to come
        unpinned -- so the flag has to be asked per world. Falls back to the
        scalar for anything that has no pin at all.
        """
        pin = getattr(body, "pin", None)
        if pin is None:
            return False
        reader = getattr(pin, "enabled_for", None)
        return bool(reader(self._world)) if callable(reader) else bool(getattr(pin, "enabled", False))


def _vec3(v) -> dict[str, float]:
    """A 3-vector in the shape every ``robot.*`` and ``motion.*`` tool speaks.

    ``gap_core.types.Vec3``, so a landmark drops straight into a pose without a
    node having to know which of the two spellings this particular tool chose.
    """
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


# ---------------------------------------------------------------------------
# The two-handed film peel.
# ---------------------------------------------------------------------------

OPEN, SHUT = 0.0, 1.0
"""This hand closes at the HIGH end of a normalised gripper command."""

PEEL_SPEED = 0.05
"""How fast the latched tab travels through the peel [m/s]. ``(peel-speed ...)``.

Not a load knob: the cohesive-zone model is rate-independent, so the tension
budget ``panel_gentle`` scores is set by the peel's geometry and not by this.
It is the episode's clock -- at 0.05 the film reads peeled about 1200 control
steps in (measured 2026-09-10: the gate at step 1211, the sheet fully off at
240 mm of run) and the whole take, both hands back, inside the documents' 1800.
"""

PEEL_RISE = 0.55
"""How far the tab rises off the face per metre it travels back down it.
``(peel-rise ...)``.

The arc's slope in the face's own frame. The pads stay parallel to the glass,
so the film leaves the hand toward the bottom edge and folds back on itself --
a 180-degree peel, which is what the clip shows -- and the rise is what keeps
the folded sheet clear of the hand that is holding the phone as the front
comes down the face. Measured on the right arm: every value from 0.5 to 1.1
solves the arc to 0.0 mm from the tab down to 0.28 m of run; the graph's
straight haul is 0.26 over 0.24, about 1.1.
"""

ARC_LIFT = 0.015
"""The arc's first rise off the face [m], before the slope takes over: the
tab clears the glass before it turns."""

ARC_STEP = 0.04
"""One peel leg's run down the face [m]. Legs are short so the film's states
are read between them and the peel stops when the sheet is off, not at a
distance chosen in advance."""

ARC_RUN_MAX = 0.36
"""Furthest the tab is dragged back down the face [m] before the routine
gives up on the arc. A 180-degree peel moves the hand about twice as far as
the front, so a 140 mm film needs ~0.28 plus the tab; 0.36 is that with
margin."""

PINCH_TILT_DEG = 10.0
"""The tab pinch: the tool axis lies in the face plane, this far below +y
(across the bench, away from the right arm's base), jaws closing along the face
normal [deg].

Measured 2026-09-10 (``rgbs/scripts/rgbs_peel_bench.py phone``) with the phone's
bottom edge anywhere from x = -0.20 to -0.06: at 0 the arc's far end fails by
16-46 mm on half the placements; at 10 and 20 the pre-grasp, the tab, four
arc points and the clearance all solve to 0.0 mm and under 2 degrees; at 30
the pre-grasp is 6-25 mm out. 10 is the near edge of the band with the most
room on the far side. The closing twin matters: closing along +n solves
under 2 degrees, along -n it parks the roll 40-120 degrees out with the same
position score -- the graph's finding on ``handle``, again.
"""

PINCH_OUT = 0.003
"""How far past the tab's outermost row the tool centre sits [m].

The pads are 54 mm wide behind their 38 mm tip band, centred on the tool
centre and lying along the face; at the outermost row (27 mm past the top
edge) the wider band would reach the phone's top-edge corner. 3 mm out puts
it 2 mm clear.
"""

HOLD_TILT_DEG = 10.0
"""The corner pinch: the tool axis lies in the face plane, this far down the
face from -y (into the phone from the left), jaws closing along the face
normal [deg].

Measured 2026-09-10: from the left arm's home the approach, pre-grasp and
pinch solve to 0.0 mm and under 2 degrees at 0 and 15 degrees with the jaws
closing along -n, and 20-25 mm out at 30 on a 30-degree phone; closing along
+n is 5-120 degrees out at every angle. 10 keeps the wrist up off the mat.
"""

HOLD_UP = 0.025
"""Where the corner pinch's tool centre sits up the face from the bottom
edge [m]. The pad's tip band is 38 mm wide along the face: from 6 to 44 mm
up, clear of the bottom edge below and short of the cradle's rest above."""

HOLD_OUT = 0.006
"""How far outboard of the phone's edge the corner pinch's tool centre sits
[m]. The pad tip reaches 10.5 mm past it, so 4.5 mm of pad goes under the
phone's corner -- and the tip lands 1.5 mm outboard of the film's edge
column, which is the whole reason the film is 60 mm wide (layout.py)."""

CARRY_TILT_DEG = 30.0
"""The carry: the tool points +x (away from the arms) and this far down,
jaws closing across y [deg].

The pinch orientation cannot be carried to the drop: at (x - 0.02, -0.20)
it solves 50-115 mm out on every placement, because the tool points +y while
the hand stands beside the right arm's own base. Tool +x and 30 down solves
the carry's high and low points to 0.0 mm everywhere measured; a top-down
grasp does at most placements. The film is on a kinematic pin, so turning
the hand turns the sheet and nothing else.
"""

DROP_AT = (-0.02, -0.20)
"""Where the film is put down, relative to the phone's bottom edge in x and
absolute in y [m, surface frame]: on the mat to the right of the phone, where
the clip drops it."""

DROP_HIGH, DROP_LOW = 0.16, 0.05
"""Carry height and release height over the mat [m]."""

MAX_POS, MAX_ROT_DEG = 0.005, 8.0
"""What a gated leg may be off by before it is refused."""

STALL_STEP = 1.0 / 60.0
"""How much gripper command a step of the stall ramp adds. A finger always
trails its target, so the ramp is slow enough that a stall reads as the
closure stopping and not as lag: a second from open to shut."""

STALL_HOLD = 5
"""Steps of no closure that count as the fingers having met the phone."""

GRIP_BITE = 0.08
"""Command added past the stall: the grip force on a rigid edge."""


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / float(np.linalg.norm(v))


class RoutineFilmPeel:
    """Hold the phone by its corner with one hand, peel the film off it with the other.

    The clip's choreography, in order: the LEFT arm comes in from its side and
    pinches the phone's bottom-left corner across its thickness -- a real
    contact grasp on a welded 8 mm edge, closed until the fingers stall -- and
    holds; the RIGHT arm pinches the film's tab at the phone's raised top edge,
    latches the tab kinematically, lets the layout's anchor go, and folds the
    film back down the face in short gated legs until the bond field reads
    peeled; then lifts clear, turns, carries the sheet to the mat on the right,
    lowers it, lets go and retreats; then the left hand opens and retreats.

    Every leg is solved end to end and gated on its worst residual before it is
    driven (``rgbs/scripts/surface_cleaning.py``'s discipline): an unreachable
    waypoint on this arm comes back as a solution tens of millimetres off with
    nothing raised, and a lurch with a latched sheet drags the film off from
    wherever the hand actually went and calls it a peel.

    The latch is the same mechanism the graph's ``sim.attach`` installs -- a
    pinning :class:`~robosimstudio.core.draggers.ParticleDragger` over the tab,
    re-aimed at the hand every control step -- and in the same order: latch,
    prove the hold, *then* release the anchor. Both ways round fail, and the
    module docstring says how.
    """

    MODE = "joint_position"

    def __init__(self, env, **kwargs) -> None:
        del kwargs
        self.env = env
        slots = arm_slots(env)
        self.work = env.handles.work_arm or slots[0]
        self.support = env.handles.support_arm or (slots[1] if len(slots) > 1 else "")
        self.dof = len(env.idle_action(self.MODE)) // len(slots)
        self.speed = float(solution_param(env, "peel-speed", PEEL_SPEED))
        self.rise = float(solution_param(env, "peel-rise", PEEL_RISE))
        self.film = env.scene.by_name[env.task.deformables[0].name]
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        env = self.env
        self.held = {n: self._q(n) for n in arm_slots(env)}
        self.grip = dict.fromkeys(arm_slots(env), OPEN)
        self.seed = {n: np.asarray(env.state.joint_q.numpy(), dtype=np.float64) for n in arm_slots(env)}
        self.refused: list[str] = []
        self.log: list[str] = []
        self.phase = "idle"
        self.peak_excess = 0.0
        self.latched = 0
        self.hold_closure = 0.0
        self.states: dict[str, float] = {}
        self.dragger = None
        # The previous episode's latch, if this routine is being re-driven in
        # one process: the entity clears its own side at reset and this is the
        # other half, so a stale dragger cannot be re-aimed or re-read.
        self.film.latch = None
        self._steps = self._choreography()

    def _q(self, name):
        ik = self.env.arm(name)
        return np.asarray(self.env.state.joint_q.numpy(), dtype=np.float64)[ik.arm_coords]

    def _say(self, text: str) -> None:
        self.log.append(text)

    def row_extras(self) -> dict[str, float]:
        # The bond field read now, not the last step's cached states: the
        # evaluator ends the episode on the step the gate is met, and a row
        # that reported the step before it would say 0.933 of a 1.000 peel.
        front, _ = self.film.peel_front() if hasattr(self.film, "peel_front") else (0.0, 0)
        return {
            "broken_fraction": float(self._broken()),
            "peel_front": float(front),
            "peel_excess": float(self.states.get("peel_excess", 0.0)),
            "peak_excess": float(self.peak_excess),
            "legs_refused": float(len(self.refused)),
            "latched": float(self.latched),
            "hold_closure": float(self.hold_closure),
            "phase": self.phase,
        }

    # ------------------------------------------------------------------ drive
    def act(self) -> np.ndarray:
        self._read()
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

    def _read(self) -> None:
        """The film's published states this step, and the peak load so far."""
        ctx = getattr(self.env, "last_context", None)
        if ctx is None:
            return
        try:
            self.states = dict(self.film.published_states(ctx))
        except Exception:
            return
        self.peak_excess = max(self.peak_excess, float(self.states.get("peel_excess", 0.0)))

    def _broken(self) -> float:
        bonds = getattr(self.film, "bonds", None)
        return float(bonds.read().s) if bonds is not None else 0.0

    # ------------------------------------------------------------------ legs
    def _hold(self, n: int):
        for _ in range(n):
            yield

    def _leg(
        self, name, target, quat, *, speed=0.06, label="", gate=True, max_pos=MAX_POS, max_rot_deg=MAX_ROT_DEG, stride=3
    ):
        """Solve a straight leg, gate it, drive it one waypoint per step.

        Position and orientation interpolate together (slerp), the IK is solved
        every ``stride``-th waypoint and the joint targets are interpolated
        between. ``gate=False`` only on a free leg from the arm's home posture,
        which cannot hold the target orientation at its first waypoints and
        converges as it goes -- with nothing in the hand that is harmless.
        """
        env, ik = self.env, self.env.arm(name)
        start, q_start = ik.ee_pose(env.state)
        start = np.asarray(start, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        quat = _unit(quat)
        q_start = _unit(q_start)
        turn = 2.0 * math.acos(min(1.0, abs(float(np.dot(q_start, quat)))))
        n = max(2, int(np.linalg.norm(target - start) / float(speed) * 60.0), int(turn / math.radians(60.0) * 60.0))
        stride = max(1, int(stride))
        knots = sorted({*range(stride, n + 1, stride), n})
        sols, worst_p, worst_r, seed = {}, 0.0, 0.0, self.seed[name]
        for k in knots:
            f = k / n
            step = start + (target - start) * f
            q_k = quat_slerp(q_start, quat, f)
            sol = ik.solve_q_full(seed, step, q_k)
            p, r = ik.pose_errors(sol, step, q_k)
            worst_p, worst_r = max(worst_p, p), max(worst_r, r)
            sols[k] = np.asarray(sol, dtype=np.float64)
            seed = sol
        note = f"{label or name}: {worst_p * 1e3:.1f} mm / {math.degrees(worst_r):.1f} deg over {n} steps"
        if gate and (worst_p > max_pos or worst_r > math.radians(max_rot_deg)):
            self.refused.append(note)
            self._say("REFUSED " + note)
            return False
        self._say(note)
        prev_k, prev_q = 0, np.asarray(self.seed[name], dtype=np.float64)
        for k in knots:
            for j in range(prev_k + 1, k + 1):
                f = (j - prev_k) / (k - prev_k)
                q = prev_q + (sols[k] - prev_q) * f
                self.held[name] = q[list(ik.arm_coords)]
                yield
            self.seed[name] = sols[k]
            prev_k, prev_q = k, sols[k]
        return True

    def _reach(self, name, target, quat) -> tuple[float, float]:
        """How well ``name`` can hold ``quat`` at ``target`` [world]: (mm, deg)."""
        ik = self.env.arm(name)
        target = np.asarray(target, dtype=np.float64)
        sol = ik.solve_q_full(self.seed[name].copy(), target, _unit(quat))
        p, r = ik.pose_errors(sol, target, _unit(quat))
        return p * 1e3, math.degrees(r)

    def _grasp(self, name, tool_axis, close_axis) -> np.ndarray:
        """The wrist target, xyzw, that points this hand along ``tool_axis``
        with its jaws closing along ``close_axis`` -- off the hand's own
        measured axes, in the frame ``ArmIK.solve_q_full`` wants."""
        hand = self.env.handles.arms[name]
        return grasp_quat(hand.approach_axis, hand.close_axis, hand.ee_offset_rot, _unit(tool_axis), _unit(close_axis))

    def _ramp_grip(self, name, target, steps: int):
        start = float(self.grip[name])
        for i in range(1, int(steps) + 1):
            self.grip[name] = start + (float(target) - start) * (i / int(steps))
            yield

    def _close_until_stall(self, name):
        """Close the jaws until the fingers stop closing, then bite.

        A stall is a *contact*: the command keeps rising and the measured
        closure does not follow. Tested on the derivative rather than the gap,
        because a finger always trails its target and a lag is not a stall.
        """
        gripper = self.env.gripper(name)
        last = float(gripper.width(self.env.state))
        still = 0
        moved = 0.0
        while self.grip[name] < 1.0:
            self.grip[name] = min(1.0, self.grip[name] + STALL_STEP)
            yield
            width = float(gripper.width(self.env.state))
            moved += max(0.0, last - width)
            still = still + 1 if last - width < 2.0e-4 else 0
            last = width
            if moved > 0.005 and still >= STALL_HOLD:
                break
        self.hold_closure = float(gripper.closed_fraction(self.env.state))
        yield from self._ramp_grip(name, min(1.0, self.grip[name] + GRIP_BITE), 10)

    # ------------------------------------------------------------------ latch
    def _install_latch(self) -> None:
        """The pin that carries the tab, and the follower that re-aims it.

        Installed once per episode from the choreography, so the hooks land
        above the environment's episode-hook floor and are retired at the next
        reset; the CUDA graph is invalidated after the hook is appended, or the
        pin's kernel is simply absent from the captured block and the tool
        reports particles nothing holds (``sim.attach`` learned the same).
        """
        from robosimstudio.core.draggers import ParticleDragger

        env = self.env
        handle = np.asarray(self.film.set_indices(self.film.spec.handle_set), dtype=np.int64)
        q = np.asarray(env.state.particle_q.numpy(), dtype=np.float64)[handle]
        spread = float(np.linalg.norm(q - q.mean(axis=0), axis=1).max())
        self.dragger = ParticleDragger(env.model, radius=spread + HOLD_MARGIN, max_particles=64, mode="pin")
        env.pipeline.hooks.post_solve.append(self.dragger.apply)
        env.post_step_hooks.append(self._follow)
        env.pipeline.invalidate_graph()

    def _follow(self, env) -> None:
        if self.dragger is not None and self.dragger.active:
            position, quat = env.arm(self.work).ee_pose(env.state)
            self.dragger.set_target(position, quat)

    def _latch(self) -> bool:
        """Seize the tab at the hand. Whole declared column, miss test on the nearest."""
        env = self.env
        position, quat = env.arm(self.work).ee_pose(env.state)
        candidates = self.film.set_indices("tab")
        n = int(self.dragger.grab(env.state, position, quat, candidates=candidates, fallback_nearest=False))
        nearest = float(self.dragger.last_distance)
        if n == 0 or nearest > LATCH_RADIUS:
            self.dragger.release()
            self.refused.append(f"latch: nearest tab particle {nearest * 1e3:.1f} mm from the tool centre")
            return False
        self.latched = n
        self.film.latch = self.dragger
        self._say(f"latched {n} tab particles, nearest {nearest * 1e3:.1f} mm")
        return True

    # ---------------------------------------------------------- choreography
    def _phone(self):
        """The phone's axes and foot in WORLD, off the fixture's placed pose."""
        xform = np.asarray(self.env.handles.entity("phone").xform, dtype=np.float64)
        rot = quat_to_mat(xform[3:7])
        u, n = rot[:, 0], rot[:, 2]
        length, width, thick = (float(v) for v in PHONE_SIZE)
        foot = xform[:3] - u * (length / 2.0) - n * (thick / 2.0)
        return u, n, foot, length, width, thick

    def _choreography(self):
        env = self.env
        y_axis = np.array([0.0, 1.0, 0.0])
        u, n, foot, _length, width, thick = self._phone()
        self._install_latch()

        # --- the holding hand: the phone's bottom-left corner ----------------
        self.phase = "hold"
        corner = foot + u * HOLD_UP + n * (thick / 2.0) + y_axis * (width / 2.0 + HOLD_OUT)
        tilt = math.radians(HOLD_TILT_DEG)
        hold_tool = math.cos(tilt) * (-y_axis) + math.sin(tilt) * (-u)
        hq = self._grasp(self.support, hold_tool, -n)
        hold_pre = corner - hold_tool * 0.05 + n * 0.03
        hold_high = corner + np.array([0.0, 0.0, 0.12])
        yield from self._leg(self.support, hold_high, hq, speed=0.18, gate=False, label="support over the corner")
        yield from self._hold(10)
        yield from self._leg(self.support, hold_pre, hq, speed=0.08, label="support to the pre-grasp")
        yield from self._hold(10)
        yield from self._leg(self.support, corner, hq, speed=0.03, label="support onto the corner")
        yield from self._hold(10)
        yield from self._close_until_stall(self.support)
        yield from self._hold(15)
        self._say(f"holding the phone at closure {self.hold_closure:.2f}")

        # --- the peeling hand: the tab at the top edge ------------------------
        self.phase = "pinch"
        handle = np.asarray(self.film.set_indices(self.film.spec.handle_set), dtype=np.int64)
        tab = np.asarray(env.state.particle_q.numpy(), dtype=np.float64)[handle].mean(axis=0) + u * PINCH_OUT
        tilt = math.radians(PINCH_TILT_DEG)
        pinch_tool = math.cos(tilt) * y_axis + math.sin(tilt) * (-u)
        pq = self._grasp(self.work, pinch_tool, n)
        pre = tab - pinch_tool * 0.05 + n * 0.02
        yield from self._leg(self.work, pre + n * 0.06, pq, speed=0.18, gate=False, label="work over the tab")
        yield from self._hold(10)
        yield from self._leg(self.work, pre, pq, speed=0.06, label="work to the pre-grasp")
        yield from self._hold(10)
        yield from self._leg(self.work, tab, pq, speed=0.04, label="work onto the tab")
        yield from self._hold(10)
        yield from self._ramp_grip(self.work, SHUT, 40)
        yield from self._hold(10)
        # Latch, prove the hold, THEN release the anchor. Reversed, the layout's
        # pin and the dragger fight over the same particles and the pin wins:
        # the tab never moves and not one bond breaks.
        if not self._latch():
            return
        yield from self._hold(10)
        handle_now = np.asarray(env.state.particle_q.numpy(), dtype=np.float64)[handle].mean(axis=0)
        tcp, _ = env.arm(self.work).ee_pose(env.state)
        drift = float(np.linalg.norm(handle_now - np.asarray(tcp)))
        if drift > LATCH_RADIUS:
            self.refused.append(f"hold: the tab sits {drift * 1e3:.1f} mm from the tool centre")
            return
        self.film.release_handle()
        yield from self._hold(10)

        # --- the peel: back down the face, folding the film on itself ---------
        self.phase = "peel"
        run = 0.0
        while run < ARC_RUN_MAX and self._broken() < 0.999:
            run_next = min(ARC_RUN_MAX, run + ARC_STEP)
            options = [
                (tab - u * run_next + n * (ARC_LIFT + self.rise * run_next), "peel"),
                (tab - u * run_next + n * (ARC_LIFT + 2.0 * self.rise * run_next), "peel, steeper"),
                (tab - u * run + n * (ARC_LIFT + self.rise * run + ARC_STEP), "peel, lift only"),
            ]
            moved = False
            for target, label in options:
                mm, deg = self._reach(self.work, target, pq)
                if mm > MAX_POS * 1e3 or deg > MAX_ROT_DEG:
                    self._say(f"skip {label} at run {run_next * 1e3:.0f} mm: IK {mm:.1f} mm / {deg:.1f} deg")
                    continue
                ok = yield from self._leg(
                    self.work, target, pq, speed=self.speed, label=f"{label} to run {run_next * 1e3:.0f} mm"
                )
                if ok:
                    moved = True
                    if label != "peel, lift only":
                        run = run_next
                    break
            if not moved:
                self.refused.append(f"peel: no reachable leg past run {run * 1e3:.0f} mm")
                break
            yield from self._hold(5)
            self._say(f"run {run * 1e3:.0f} mm: broken {self._broken():.3f}")
        self._say(f"peel done at run {run * 1e3:.0f} mm: broken {self._broken():.3f}")

        # --- clear, carry to the mat on the right, put it down ----------------
        self.phase = "carry"
        pos, _ = env.arm(self.work).ee_pose(env.state)
        yield from self._leg(self.work, np.asarray(pos) + n * 0.05, pq, speed=0.08, label="lift clear")
        yield from self._hold(10)
        # DROP_AT's x rides on the phone's foot (world), its y and the heights on
        # the cell's surface frame, which is a pure translation of world.
        cx = float(foot[0]) + DROP_AT[0]
        cy = float(CELL.center[1]) + DROP_AT[1]
        cz, lz = float(CELL.surface_z) + DROP_HIGH, float(CELL.surface_z) + DROP_LOW
        tilt = math.radians(CARRY_TILT_DEG)
        cq = self._grasp(self.work, np.array([math.cos(tilt), 0.0, -math.sin(tilt)]), y_axis)
        yield from self._leg(self.work, [cx, cy, cz], cq, speed=0.15, label="carry to the mat", max_rot_deg=12.0)
        yield from self._hold(10)
        yield from self._leg(self.work, [cx, cy, lz], cq, speed=0.10, label="lower the film")
        yield from self._hold(10)
        if self.dragger is not None:
            self.dragger.release()
        self.film.latch = None
        yield from self._ramp_grip(self.work, OPEN, 30)
        yield from self._hold(15)
        yield from self._leg(self.work, [cx, cy, cz], cq, speed=0.10, label="work retreat", gate=False)

        # --- let the phone go ---------------------------------------------------
        self.phase = "release"
        yield from self._ramp_grip(self.support, OPEN, 30)
        yield from self._hold(10)
        yield from self._leg(self.support, hold_pre, hq, speed=0.08, label="support off the corner", gate=False)
        yield from self._hold(5)
        yield from self._leg(self.support, hold_high, hq, speed=0.15, label="support retreat", gate=False)
        self.phase = "done"
        self._say(
            f"done: broken {self._broken():.3f}, peak excess {self.peak_excess:.3f} N, {len(self.refused)} legs refused"
        )
        yield from self._hold(30)
