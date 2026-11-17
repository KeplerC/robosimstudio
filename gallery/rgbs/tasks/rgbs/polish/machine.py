# SPDX-License-Identifier: Apache-2.0
"""The polishing machine: a fixed rotary tool whose spindle is a velocity-driven
revolute, replicating the reference rig (a Dremel-style tool clamped to a
vertical stand over a work mat, a polishing drum on a mandrel in the collet, and
a small rest ledge on the column that the strip is laid across during regrasps).

**The tool is clamped SIDEWAYS, nose forward at the arms.** It hung nose-down
until this was measured against the reference photograph, and the difference is
not cosmetic: a nose-down tool presents the drum's flat END FACE, and nobody
polishes on a drum's end. The real one presents the drum's BARREL -- its
cylindrical side and the rim corners that bound it -- and the work is rubbed
ALONG the tool's own axis across it. Three things follow, and they are the task:

* the contact is a LINE along the spindle axis, not a disc, so there is no
  on-axis dead spot and no reason to press twice off-centre;
* the drum's surface at that line runs PERPENDICULAR to the axis -- across the
  strip's long axis -- so its Coulomb drag pulls the strip straight out of an
  end-on pinch, and a stroke in that direction cannot be held;
* a stroke ALONG the axis is perpendicular to that drag, and a slow
  perpendicular stroke on a fast-sliding contact costs almost nothing, because
  the friction cone is already saturated in the sliding direction.

Three findings from the feasibility probes are still load-bearing here:

* **The drive is Newton's native VELOCITY joint actuator** -- ``target_ke=0,
  target_kd=kv`` on the revolute infers ``JointTargetMode.VELOCITY`` and
  SolverMuJoCo installs a velocity servo whose total force is clamped by
  ``effort_limit``. Measured on this stack: holds the commanded speed exactly
  under contact load, and the torque clamp gives an honest spin-up/stall
  instead of an instability. No custom kernel, nothing host-side, and the
  target is read from ``control.joint_target_qd`` inside the captured graph.
* **The bit collider is a convex 16-gon prism mesh, never a CYLINDER
  primitive.** mujoco_warp's multi-point manifold covers box/mesh pairs under
  ``enable_multiccd`` and never cylinders -- a cylinder-primitive bit gets one
  contact point and visibly rocks/chatters on a flat face (measured; the
  solver itself warns). Sixteen facets keep the footprint honestly circular.
  With a barrel contact the facet count matters more, not less: the pressed
  strip rides on one or two facets at a time.
* **The spun surface really drags.** A free part resting on the running face
  co-rotates in stick at low rim speed and is thrown at high rim speed, so the
  spindle speed below is chosen for a rim speed (~0.4 m/s) that is meaningful
  but inside the regime the probes verified stable at 960 Hz.

The stand (base, column, clamp, tool nose, rest ledge) is world-fixed scenery:
shapes on body ``-1``, the ``PortUsbc`` pattern. Only the spindle is a dynamic
link. The collet, the mandrel and the motor housing are visual-only so that no
stray near-bit contact exists -- the motor deliberately overlaps the column,
which is what a clamp stand looks like and what nothing can collide with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import newton
import numpy as np
import warp as wp

from robosimstudio.core.render.palette import FIXTURE, METAL
from robosimstudio.scene.entity import Entity
from robosimstudio.scene.registry import register_entity

from ..look import BLACK, DREMEL_GREY, DRUM_PINK

__all__ = ["EntityPolishMachine", "SpecPolishMachine", "machine_frame"]

# Six greys and a tan, all arrived at by eye against a bench that rendered black,
# and three of them within 0.02 of a palette entry that already existed. What they
# were really saying is what the roles say: the stand and the tool body are
# machinery, the collet and mandrel are machined metal, the ledge is a fixture,
# and the drum is a warm abrasive. The collet, mandrel and ledge are still
# spelled as the palette; the stand, tool and drum take the reference clip's own
# colours from ``rgbs/tasks/rgbs/look.py``, because that clip is what the suite's
# picture is matched to.
STAND_COLOR = BLACK
"""The stand: a black post and clamp in the reference clip. Was ``MACHINE``."""
TOOL_COLOR = DREMEL_GREY
"""The rotary tool's body: near-black housing in the reference clip. Was
``MACHINE``; before that ``(0.10, 0.10, 0.11)``, which renders at pixel 22
under the tiled tracer -- the black cylinder silhouette in the recorded video."""

COLLET_COLOR = METAL
MANDREL_COLOR = METAL
BIT_COLOR = DRUM_PINK
"""The polishing drum: a used sanding drum's pink-tan. Was ``SUBSTRATE``."""

LEDGE_COLOR = FIXTURE

BIT_SIDES = 16
"""Collider facets for the drum. Enough that the footprint is circular to
~0.1 mm at Ø25; few enough that GJK/EPA on the tiny prism stays crisp."""


def machine_frame(spec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(contact, spindle_axis, work_normal)`` in the MACHINE-LOCAL frame.

    The one place the working geometry is derived. The entity composes it
    through ``at``/``yaw`` and the surface frame; ``rgbs/scripts/rgbs_polish_bench.py``
    builds its press rig against it directly. Restating any of the three
    somewhere else is how a rig ends up measuring a press it is not making.

    * ``contact`` -- the lowest point of the barrel at the drum's axial centre;
    * ``spindle_axis`` -- unit, from the drum back up the tool to the motor;
    * ``work_normal`` -- unit, from the spindle axis out to ``contact``. It is
      the direction the drum presses in, and it is what a strip's top face must
      be square to.
    """
    tilt = math.radians(float(spec.tilt_deg))
    spindle_axis = np.array([0.0, -math.sin(tilt), math.cos(tilt)])
    work_normal = np.array([0.0, -math.cos(tilt), -math.sin(tilt)])
    contact = np.array([0.0, float(spec.contact_reach), float(spec.contact_height)])
    return contact, spindle_axis, work_normal


def _drum_mesh(radius: float, half_height: float, sides: int = BIT_SIDES) -> newton.Mesh:
    """A closed convex prism along local +z -- the drum bit's collider."""
    ang = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    ring = np.stack([np.cos(ang) * radius, np.sin(ang) * radius], axis=1)
    top = np.hstack([ring, np.full((sides, 1), half_height)])
    bot = np.hstack([ring, np.full((sides, 1), -half_height)])
    verts = np.vstack([top, bot, [[0.0, 0.0, half_height]], [[0.0, 0.0, -half_height]]]).astype(np.float32)
    ct, cb = 2 * sides, 2 * sides + 1
    idx: list[int] = []
    for i in range(sides):
        j = (i + 1) % sides
        idx += [i, j, ct]
        idx += [sides + j, sides + i, cb]
        idx += [i, sides + i, j, j, sides + i, sides + j]
    return newton.Mesh(vertices=verts, indices=np.asarray(idx, dtype=np.int32))


@dataclass(frozen=True, kw_only=True)
class SpecPolishMachine:
    """The machine's knobs, at the reference rig's proportions.

    Poses are in the work-surface frame: ``at``/``yaw`` place the stand's base
    on the mat, and every height below is above that surface. The tool leans
    ``tilt_deg`` from vertical, so at 75 degrees it lies 15 degrees nose-down
    with its motor end back over the column and its drum forward over the mat.
    """

    entity: str = "polish_machine"
    name: str = "machine"
    at: tuple[float, float] = (0.0, 0.0)
    yaw: float = 0.0
    tilt_deg: float = 75.0
    """Tool lean from vertical [deg]. **Structural, not styling.**

    At 0 the tool hangs nose-down and presents the drum's flat end face, which
    is what this first shipped with and is not how a polishing drum is used. At
    90 it lies dead horizontal. 75 -- fifteen degrees nose-down -- is the
    reference photograph's mounting, and the fifteen degrees are load-bearing
    twice over: they are what a bench operator gives themselves so the work can
    be seen, and they tilt the barrel's lowest generatrix, so the strip must be
    PITCHED to match (see ``ToolsPolish._press_strip_quat``). Pressed flat
    against a leaning drum the strip would catch the nose rim alone -- 1.6 mm
    lower than the axial centre at this lean -- and polish a gouge instead of a
    spot.

    Everything below derives from it, so it is the one number to change if the
    mounting is ever re-measured. Values near 0 are refused at build: the
    working radial is undefined for a vertical spindle."""
    contact_reach: float = 0.160
    """Working contact point forward of the column axis [m].

    Unchanged from the nose-down build's ``face_reach``, and deliberately so:
    it is the number the arms were measured against and the whole reach
    envelope, the pick and the set-down depend on it. Two things set it. The
    column and its clamp have to sit behind the arms' working corridor -- at
    the 105 mm this first shipped with, the right arm's link4 jammed into the
    bracket on every approach (measured: a steady 0.105 rad joint-0 tracking
    error, both jaws closing on air). And it must NOT hang over the strip's
    rest. Cantilevering it there was tried, to make the lift a straight
    vertical move: it breaks the pick instead, because a gripper stands about
    135 mm above its own tool centre and drives itself into the collet while
    reaching for a strip lying under the drum."""
    contact_height: float = 0.140
    """The working contact point above the work surface [m].

    The lowest point of the BARREL at the drum's axial centre -- not an
    end-face centre, which is what the nose-down build measured here. The value
    is unchanged so that the arms' press stroke and the ledge clearance below
    both stay valid."""
    bit_radius: float = 0.0125
    """Drum radius [m] -- a 25 mm drum, not the 13 mm this first shipped with.

    Under a barrel contact the radius sets the CONTACT BAND: a strip riding
    ``d`` into the drum touches a band ``2*sqrt(2*R*d - d*d)`` wide across the
    axis. At 25 mm a 0.2 mm ride-in gives a 4.4 mm band on a 10 mm spot, which
    is what makes a stroke necessary rather than decorative -- see
    ``SpecPolishMarks.gap_tol`` and ``ToolsPolish.OVERDRIVE``. Polishing drums
    come in many sizes; this one is chosen so the task is about technique."""
    bit_half: float = 0.005
    """Drum half-length along the spindle [m].  ; CAL

    The contact patch is twice this -- 10 mm -- across the strip's 25 mm width,
    and it is deliberately NARROWER THAN THE MARK it has to clear (20 mm). That
    is what makes the rub necessary rather than decorative: half the mark is
    out of reach at any one press, and the other half is reached by traversing
    along the spindle. ``STROKE_AMP`` is 6 mm against it, so the patch sweeps
    22 mm and still stops 1.5 mm short of either edge of the strip."""
    bit_mu: float = 1.2
    """Abrasive drum friction. Orphaned sanding-block asset ships 2.0; 1.2
    with the strip's 1.0 combines (sqrt) to 1.10 -- firm drag without gluing.
    It is also what makes a cross-axis stroke fail: the drag is ``mu * press``
    along the strip's own long axis, which is the direction an end-on pinch
    holds worst."""
    mandrel_len: float = 0.022
    """Exposed shaft between the collet's mouth and the drum's rear face [m].

    Measured off the reference photograph, and it buys the clearance a sideways
    tool needs: with the drum this far proud, the tool nose's collider stands
    32 mm clear of the strip's far edge at the press. Shorten it and the nose
    reaches into the strip's own plane."""
    spindle_speed: float = 30.0
    """Target speed [rad/s] = 286 RPM; rim 0.38 m/s on the 25 mm drum.

    Halved from 60 once the drum was enlarged, because the two multiply: rim
    speed is omega*r, and a 25 mm drum at 60 rad/s runs its edge at 0.75 m/s.
    A strip that catches that edge off-centre takes a large tangential impulse
    -- measured as the simulation diverging outright mid-press, more than once.
    At 30 rad/s the rim speed is back where the 13 mm drum had it, which is
    where the press was stable, and the surface is still moving fast enough
    that the drag story the task is about holds."""
    spindle_kv: float = 0.01
    """Velocity-servo gain [N*m*s/rad]: ~4% droop under an 8 N press -- the
    bit audibly 'bogs down' under pressure, which is honest."""
    spindle_tau_max: float = 0.08
    """Motor torque ceiling [N*m], small-rotary-tool scale. Stalling needs a
    ~24 N press, above anything the task commands: a policy can slow the bit,
    not stop it."""
    spindle_armature: float = 2.0e-4
    """Rotor inertia [kg*m^2]: dominates the gram-scale drum, ~0.15 s spin-up."""
    ledge_height: float = 0.110
    """Rest-ledge top surface above the work surface [m]."""
    ledge_half: tuple[float, float, float] = (0.045, 0.0175, 0.004)
    ledge_mu: float = 0.8
    contact_ke: float = 6.0e4
    contact_kd: float = 500.0
    """Contact stiffness pair = 4 ms solref, the ceiling a 960 Hz arrangement
    admits (tool_sort's measurement). It is also the ride-in knob: a softer
    pair lets the strip sink into the drum and widens the contact band until a
    static press covers a whole spot and the stroke stops mattering."""


# --- the tool's own proportions, forward of the clamp ----------------------
NOSE_RADIUS = 0.0175
NOSE_HALF = 0.030
"""The Dremel's barrel forward of its clamp collar -- the only part of the tool
body that COLLIDES. What is behind the clamp is drawn (see ``MOTOR_*``) and not
collided: it lies behind the column, where no arm reaches, and giving it a
collider would only push the stand off the back of the table."""
MOTOR_RADIUS = 0.019
MOTOR_HALF = 0.055
COLLET_RADIUS = 0.006
COLLET_HALF = 0.0075
MANDREL_RADIUS = 0.0015
COLUMN_HALF = 0.020
COLUMN_TOP = 0.166
"""Column height [m]. Bounded above by the tool: the spindle axis crosses the
column's front face at z = 0.190, so a column taller than 0.1715 would be drawn
through the motor housing. It still stands well over the 0.110 rest ledge."""


@register_entity("polish_machine")
class EntityPolishMachine(Entity):
    """World-fixed stand plus one velocity-driven spindle link."""

    spec: SpecPolishMachine

    def __init__(self, spec: SpecPolishMachine, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.spindle_body: int = -1
        self.spindle_joint: int = -1
        self.spindle_dof: int = -1
        self.spindle_coord: int = -1
        self.bit_shape: int = -1
        self.contact_pose: np.ndarray | None = None
        """World 7-pose of the barrel's working contact point -- the lowest
        point of the drum at its axial centre. Its +z is the outward working
        normal, pointing down-and-back at the pressed strip, so a planner
        presses along this frame's +z exactly as it did against the old end
        face."""
        self.ledge_pose: np.ndarray | None = None
        """World 7-pose of the rest-ledge top center; +z up."""
        self.spindle_axis_world: np.ndarray | None = None
        """World unit vector up the spindle (from the drum toward the motor).

        Also **the stroke direction**: it is the one direction in the pressed
        strip's plane that is perpendicular to the drum's drag."""
        self.work_normal_world: np.ndarray | None = None
        """World unit vector from the spindle axis out to the contact point --
        the direction the drum presses the strip in. Perpendicular to
        ``spindle_axis_world`` by construction."""
        self._published_at: tuple[object, dict[str, float]] | None = None

    world_aware = True
    """Every phase of this machine carries a world axis -- and it owns none.

    Said out loud rather than inherited by luck, because the flag is a claim and
    this one is cheap to check. The stand is world-fixed geometry and the spindle
    is driven by a velocity actuator the compiler replicates with everything
    else, so nothing here allocates per-world state, contributes a substep kernel
    or holds an index it later applies to a fleet array. What it *does* do is
    publish, and :meth:`published_states` reads the speed off
    ``context.body_qd`` -- which a wave has already sliced to the asking world
    and indexed like the template -- so the answer is that world's spindle and
    not the first one's.

    Without this, a wave refuses the whole scene by name (see
    ``robosimstudio_bench.envs.fleet.scoring_refusal``): the refusal is by
    capability and cannot tell a publisher that is already per world from one
    that reads row 0.
    """

    # ------------------------------------------------------------------ build
    def _cfg(self, builder: newton.ModelBuilder, mu: float, density: float = 0.0):
        cfg = builder.default_shape_cfg.copy()
        cfg.mu = float(mu)
        cfg.mu_torsional = 0.0
        cfg.mu_rolling = 0.0
        cfg.ke = float(self.spec.contact_ke)
        cfg.kd = float(self.spec.contact_kd)
        if density:
            cfg.density = float(density)
        return cfg

    def _world(self, t_local: wp.transform) -> wp.transform:
        """Machine-local -> world, through ``at``/``yaw`` and the surface frame."""
        spec = self.spec
        base = wp.transform(
            wp.vec3(float(spec.at[0]), float(spec.at[1]), 0.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(spec.yaw)),
        )
        t = wp.transform_multiply(base, t_local)
        seven = self.placed(np.array([*t.p, *t.q], dtype=np.float64))
        return wp.transform(
            wp.vec3(float(seven[0]), float(seven[1]), float(seven[2])),
            wp.quat(float(seven[3]), float(seven[4]), float(seven[5]), float(seven[6])),
        )

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec = self.spec
        name = spec.name
        tilt = math.radians(float(spec.tilt_deg))
        if not math.radians(30.0) <= tilt <= math.radians(90.0):
            raise ValueError(
                f"polish_machine tilt_deg={spec.tilt_deg} is outside 30..90 degrees; the working "
                "radial is undefined for a near-vertical spindle and the drum would present its "
                "end face, which is not how a polishing drum is used"
            )
        # R_x(tilt) maps +z onto u, the spindle axis pointing from the drum back
        # up the tool toward the motor; w is the outward working radial, the one
        # direction perpendicular to u that points as far down as it can. The
        # two are orthogonal by construction: w = -(0, cos, sin), u = (0, -sin, cos).
        q_tilt = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), tilt)
        contact, u, w = machine_frame(spec)
        R = float(spec.bit_radius)
        axis_c = contact - w * R  # the drum's centre, on the spindle axis

        def along_np(dist: float) -> np.ndarray:
            return axis_c + u * dist

        def along(dist: float) -> wp.vec3:
            p = along_np(dist)
            return wp.vec3(float(p[0]), float(p[1]), float(p[2]))

        stand = self._cfg(builder, mu=0.6)
        ledge_cfg = self._cfg(builder, mu=spec.ledge_mu)
        ident = wp.quat_identity()
        # Nothing near the bit may collide except the drum itself.
        decor = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)

        # --- the tool, laid out back from the drum along +u -----------------
        mandrel_c = float(spec.bit_half) + 0.5 * float(spec.mandrel_len)
        collet_c = float(spec.bit_half) + float(spec.mandrel_len) + COLLET_HALF
        nose_front = float(spec.bit_half) + float(spec.mandrel_len) + 2.0 * COLLET_HALF
        nose_c = nose_front + NOSE_HALF + NOSE_RADIUS
        nose_rear = nose_front + 2.0 * (NOSE_HALF + NOSE_RADIUS)
        motor_c = nose_rear + MOTOR_HALF

        # --- world-fixed scenery: shapes on body -1 ------------------------
        builder.add_shape_box(
            -1,
            xform=self._world(wp.transform(wp.vec3(0.0, 0.0, 0.006), ident)),
            hx=0.060,
            hy=0.060,
            hz=0.006,
            cfg=stand,
            color=STAND_COLOR,
            label=f"{name}_base",
        )
        builder.add_shape_box(
            -1,
            xform=self._world(wp.transform(wp.vec3(0.0, 0.0, 0.5 * (0.012 + COLUMN_TOP)), ident)),
            hx=COLUMN_HALF,
            hy=COLUMN_HALF,
            hz=0.5 * (COLUMN_TOP - 0.012),
            cfg=stand,
            color=STAND_COLOR,
            label=f"{name}_column",
        )
        # The clamp: a collar on the tool's barrel, and the bracket reaching to
        # it from the top of the column. Both are derived from the tool's own
        # layout rather than placed by hand, so a change to `tilt_deg` or
        # `contact_reach` carries them with it.
        clamp = along_np(nose_c + NOSE_HALF * 0.5)
        builder.add_shape_box(
            -1,
            xform=self._world(
                wp.transform(
                    wp.vec3(0.0, 0.5 * (COLUMN_HALF + float(clamp[1])), 0.5 * (COLUMN_TOP + float(clamp[2]))),
                    ident,
                )
            ),
            hx=0.014,
            hy=max(0.010, 0.5 * abs(float(clamp[1]) - COLUMN_HALF)),
            hz=max(0.010, 0.5 * abs(float(clamp[2]) - COLUMN_TOP)),
            cfg=stand,
            color=STAND_COLOR,
            label=f"{name}_bracket",
        )
        builder.add_shape_box(
            -1,
            xform=self._world(wp.transform(wp.vec3(*(float(v) for v in clamp)), q_tilt)),
            hx=0.021,
            hy=0.021,
            hz=0.012,
            cfg=stand,
            color=STAND_COLOR,
            label=f"{name}_clamp",
        )
        builder.add_shape_capsule(
            -1,
            xform=self._world(wp.transform(along(nose_c), q_tilt)),
            radius=NOSE_RADIUS,
            half_height=NOSE_HALF,
            cfg=stand,
            color=TOOL_COLOR,
            label=f"{name}_tool_nose",
        )
        # The motor housing is DRAWN and not collided: it lies behind the column
        # (and through it, which is what a clamp stand looks like), where no arm
        # reaches. Collide it and the stand has to move 60 mm back to fit, off
        # the far edge of the table.
        builder.add_shape_capsule(
            -1,
            xform=self._world(wp.transform(along(motor_c), q_tilt)),
            radius=MOTOR_RADIUS,
            half_height=MOTOR_HALF,
            cfg=decor,
            color=TOOL_COLOR,
            label=f"{name}_tool_body",
        )
        builder.add_shape_cylinder(
            -1,
            xform=self._world(wp.transform(along(collet_c), q_tilt)),
            radius=COLLET_RADIUS,
            half_height=COLLET_HALF,
            cfg=decor,
            color=COLLET_COLOR,
            label=f"{name}_collet",
        )
        builder.add_shape_cylinder(
            -1,
            xform=self._world(wp.transform(along(mandrel_c), q_tilt)),
            radius=MANDREL_RADIUS,
            half_height=0.5 * float(spec.mandrel_len),
            cfg=decor,
            color=MANDREL_COLOR,
            label=f"{name}_mandrel",
        )
        ledge_center = wp.vec3(
            0.0, COLUMN_HALF + float(spec.ledge_half[1]), float(spec.ledge_height) - float(spec.ledge_half[2])
        )
        builder.add_shape_box(
            -1,
            xform=self._world(wp.transform(ledge_center, ident)),
            hx=float(spec.ledge_half[0]),
            hy=float(spec.ledge_half[1]),
            hz=float(spec.ledge_half[2]),
            cfg=ledge_cfg,
            color=LEDGE_COLOR,
            label=f"{name}_ledge",
        )

        # --- the spindle: one dynamic link on a velocity-driven revolute ----
        t_bit = self._world(wp.transform(along(0.0), q_tilt))
        link = builder.add_link(xform=t_bit, label=f"{name}_spindle")
        bit_cfg = self._cfg(builder, mu=spec.bit_mu, density=3000.0)
        self.bit_shape = builder.add_shape_mesh(
            link,
            mesh=_drum_mesh(float(spec.bit_radius), float(spec.bit_half)),
            cfg=bit_cfg,
            color=BIT_COLOR,
            label=f"{name}_bit",
        )
        j = builder.add_joint_revolute(
            parent=-1,
            child=link,
            parent_xform=t_bit,
            child_xform=wp.transform(),
            axis=(0.0, 0.0, 1.0),
            target_vel=float(spec.spindle_speed),
            target_ke=0.0,
            target_kd=float(spec.spindle_kv),
            actuator_mode=newton.JointTargetMode.VELOCITY,
            effort_limit=float(spec.spindle_tau_max),
            armature=float(spec.spindle_armature),
            label=f"{name}_spindle",
        )
        builder.add_articulation([j], label=name)
        self.spindle_body = link
        self.spindle_joint = j
        self.spindle_dof = int(builder.joint_qd_start[j])
        self.spindle_coord = int(builder.joint_q_start[j])

        # --- planner-facing world constants --------------------------------
        # The working frame's +z is w. R_x(pi/2 + tilt) maps +z onto
        # (0, -sin(pi/2+tilt), cos(pi/2+tilt)) = (0, -cos tilt, -sin tilt) = w.
        q_work = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0 + tilt)
        t_contact = self._world(wp.transform(wp.vec3(*(float(v) for v in contact)), q_work))
        self.contact_pose = np.array([*t_contact.p, *t_contact.q], dtype=np.float64)
        t_ledge = self._world(wp.transform(wp.vec3(ledge_center[0], ledge_center[1], float(spec.ledge_height)), ident))
        self.ledge_pose = np.array([*t_ledge.p, *t_ledge.q], dtype=np.float64)
        # Both unit vectors come out of the SAME world transform as the poses
        # above, so a change to `at`/`yaw` cannot move one without the others.
        axis_c_w = np.array([*self._world(wp.transform(along(0.0), ident)).p], dtype=np.float64)
        tip_w = np.array([*self._world(wp.transform(along(1.0), ident)).p], dtype=np.float64)
        self.spindle_axis_world = (tip_w - axis_c_w) / np.linalg.norm(tip_w - axis_c_w)
        radial = self.contact_pose[:3] - axis_c_w
        self.work_normal_world = radial / np.linalg.norm(radial)

        # The clearance arithmetic the arms depend on, asserted rather than
        # hoped. The strip is pressed with its top face in the plane through
        # `contact` normal to w, and it reaches `strip_half` along u from there;
        # the tool nose is the only tool collider that could reach into it.
        # 1e-4, not 1e-12: both vectors come back through `wp.transform`, which
        # is float32, so a perfect right angle measures a few times 1e-7 here.
        # A real mistake in the frame is O(0.1) and this still catches it.
        skew = abs(float(np.dot(self.spindle_axis_world, self.work_normal_world)))
        assert skew < 1.0e-4, f"the working radial is {skew:.2e} off perpendicular to the spindle axis"
        strip_edge = contact + u * 0.0125
        nose_cap = along_np(nose_front + NOSE_RADIUS)
        clearance = float(np.linalg.norm(strip_edge - nose_cap)) - NOSE_RADIUS
        assert clearance >= 0.015, (
            f"the tool nose stands {clearance * 1e3:.1f} mm from the strip's far edge at the press; "
            "lengthen `mandrel_len`"
        )
        assert float(spec.contact_height) - float(spec.ledge_height) >= 0.02, "ledge crowds the bit"
        assert 2.0 * float(spec.bit_half) <= 0.025, "the drum is wider than the strip it polishes"

    # ------------------------------------------------------------------ bind
    def bind(self, model: newton.Model) -> None:
        super().bind(model)
        m = float(model.body_mass.numpy()[self.spindle_body])
        print(
            f"[polish_machine] spindle mass {m * 1e3:.1f} g, drive: "
            f"w*={self.spec.spindle_speed:.0f} rad/s kv={self.spec.spindle_kv} "
            f"tau_max={self.spec.spindle_tau_max} N*m"
        )

    # ------------------------------------------------------------- published
    def published_states(self, snapshot) -> dict[str, float]:
        """Spindle speed off the context's host ``body_qd`` -- no device read.

        The context hands rigid velocities as ``[n, 6]`` (linear, angular);
        the signed spindle speed is the world angular velocity projected on
        the (fixed) spindle axis. Cached per context object, the tape pattern.
        """
        context = getattr(snapshot, "ctx", snapshot)
        cached = self._published_at
        if cached is not None and cached[0] is context:
            return cached[1]
        body_qd = getattr(context, "body_qd", None)
        out: dict[str, float] = {}
        if body_qd is not None and 0 <= self.spindle_body < len(body_qd) and self.spindle_axis_world is not None:
            w_world = np.asarray(body_qd[self.spindle_body], dtype=np.float64)[3:6]
            speed = float(np.dot(w_world, self.spindle_axis_world))
            target = float(self.spec.spindle_speed)
            out = {
                "disc_speed": speed,
                "disc_rim_speed": speed * float(self.spec.bit_radius),
                # Named at_speed, not "spinning": `spinning` is the record's
                # semantic STATE (a threshold over disc_speed), and shadowing it
                # would give one name two resolution paths.
                "at_speed": 1.0 if abs(speed - target) <= 0.2 * abs(target) else 0.0,
            }
        self._published_at = (context, out)
        return out

    # ----------------------------------------------------------------- reset
    def reset(self, pipeline, episode, *, world: int = 0) -> None:
        """Nothing physical to restore: the environment puts the spindle
        coordinate back and zeroes every joint velocity *after* entity resets,
        and the velocity actuator then re-spins the drum from rest in ~0.15 s
        -- before the arms finish homing. Writing ``joint_qd`` here would be
        erased, so it is not written. Only the published-state cache clears.

        ``world`` is accepted and not read, which is the honest form for an
        entity with nothing per world to restore: the cache is one slot keyed on
        the *context object*, and a wave hands every world a different one."""
        del pipeline, episode, world
        self._published_at = None
