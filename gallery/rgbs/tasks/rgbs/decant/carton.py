# SPDX-License-Identifier: Apache-2.0
"""The shipping carton: a free base carrying two top flaps, taped shut in the middle.

**One class, not two.** The carton is an :class:`ObjectSpec` whose ``entity``
field names this :class:`~robosimstudio.domains.rigid.entity.EntityRigid`
subclass, which is the seam ``ObjectSpec.entity``'s docstring points at -- *"a
subclass that needs its own physics overrides one field"*. ``super().build()``
makes the free body, its free joint, the articulation SolverMuJoCo needs in
order to integrate it at all, and the floor slab; this class then adds the four
walls **onto that same body** and hangs the two flap links off it.

The alternative -- an extras entity adding walls to an object somebody else
built -- was rejected for a specific reason: ``EntityRigid.bind`` only refills
``self.shapes`` when it is empty, so the walls would never enter the object's
shape set, segmentation would drop them, and ``_width_across`` would read the
4 mm floor slab as the carton's graspable width.

**The flaps are the entity's, not the ``articulated`` substrate's.** The record
declares ``domain="rigid"``. ``docs/task_language.md`` S8.4 marks the
articulated substrate as declared-and-not-stepped; none of it is needed, because
``add_joint_revolute`` is what ``usbc_insert``'s 32-link cable and ``polish``'s
spindle already run through SolverMuJoCo every step.

**What holds the flaps shut is the tape, and the tape lives next door.**
:mod:`.seam` owns the two tape strips, the severance field and the hinge torque,
because the thing that knows how much tape is left is the thing that is severing
it. This module owns the board and publishes what the carton can measure about
itself.

**The seam runs along the carton's LONG axis, and the long axis is laid along
the bench.** The entity's own frame has the seam along local ``+x``; the layout
yaws the spec a quarter turn so that axis lies along world ``y`` (left-right in
``sample_videos/box_decanting.mp4``), which is how the clip's parcel stands. The
frame was kept and the spec turned, rather than the frame re-cut, so
:func:`carton_frame` stays the one derivation and the seam field's cell layout
stays one formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import newton
import numpy as np
import warp as wp

from robosimstudio.core.render.finish import Finish
from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.scene.build.bodies import set_body_mass
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.asset import ObjectSpec, ShapeSpec

from ..look import KRAFT

__all__ = ["FINGERTIP_BLADE", "EntityShippingCarton", "SpecShippingCarton", "carton_frame"]

GRIP_SPAN = 0.09396
"""YAM ``linear_4310`` fingertip separation at full open [m].

Read here rather than imported: an asset may not name a robot, and this is the
number two build assertions are stated against."""

FINGERTIP_BLADE = 0.007
"""A YAM fingertip pad's thickness along the jaw's closing axis [m]: the low
band of the pad is a 7 x 27 mm blade. The seam gap is sized against it, because
opening a flap means putting one of these into the gap."""


@dataclass(frozen=True, kw_only=True)
class SpecShippingCarton(ObjectSpec):
    """A corrugated parcel carton with two top flaps, a short tape across their join.

    The dimensions are the clip's parcel, measured against the mat's 25 mm grid
    and the arms' own 620 mm mount spacing, and each records what else it is
    fitted against.
    """

    entity: str = "shipping_carton"
    category: str = "shipping_carton"
    name: str = "carton"

    outer: tuple[float, float, float] = (0.200, 0.150, 0.100)
    """Outside L x W x H [m], the floor's underside to the wall tops.

    **The clip's carton, not a reach-limited one.** ``sample_videos/
    box_decanting.mp4`` shows a parcel about 200 mm along the bench, 150 mm
    across it and 100 mm tall: eight grid squares of front face at frame 0, a
    front face 1.08x the height of a square's projected... measured as 242 px
    long against 108 px of front face in a frame where the elevation makes a
    vertical 100 mm read 0.84 of a horizontal one, and a top face whose
    projected depth reads 0.53 of the front face, which at the phone's ~28
    degree elevation and 0.8 m range is 150 mm rather than 100. The shipped
    carton was 180 x 135 x 90 because its seam ran along world x and a wrist
    held along x was reach-limited at 240 mm; the seam now runs along y, the
    draw is the 100 mm of tape rather than the whole seam, and the sweep in
    ``layout.py`` is what says the 200 mm parcel fits.

    The 200 mm is the LENGTH of the seam and the flaps; only the middle 100 mm
    of it is taped (:attr:`~.seam.SpecSealTape.tape_len`). The 150 mm is what
    the two flaps span with their gap, and 150 - 2*wall = 142 mm of opening
    against a 94 mm jaw span leaves 24 mm a side for the decant lift -- asserted
    at build, where the floor is a 20 mm total margin. The 100 mm puts the tape
    at 0.1085 m over the work surface (see :func:`carton_frame`)."""

    wall: float = 0.004
    """Double-wall corrugated [m], and the tolerance the flaps are fitted to:
    the hinge sits on the wall's INNER top edge, so a shut flap lies flush
    across the opening rather than proud of it."""

    flap_thickness: float = 0.003
    seam_gap: float = 0.012
    """The slot between the two shut flaps' free edges [m]. **Twelve, and the
    number is a fingertip.**

    The clip's carton has a visible dark join along the whole seam, and the
    errand's last move is a fingertip put INTO that join to hook the far flap
    up. A YAM fingertip's low band is a 7 mm blade (:data:`FINGERTIP_BLADE`),
    and 12 mm admits it with 2.5 mm a side; asserted at build against 9 mm.

    This shipped at zero, and the zero was measured: at 2 mm against a 1.5 mm
    blade the knife slotted straight through the join and the press read 0.00 N
    at every overdrive from 6 to 30 mm, while the contact table showed two
    healthy blade-to-flap pairs carrying exactly zero impulse. That finding
    still rules, and it is why the gap is now BRIDGED where the blade goes: the
    seam entity lays a rigid tape strip on each flap that reaches from the
    flap's top face across half the gap, so the two strips meet at the
    centreline with zero gap and the blade lands on a continuous plane over the
    taped 100 mm -- the two 0-gap flaps already proved a blade cuts on two boxes
    that touch on a shared face. Outside the tape the join is open, which is
    what the hook needs and the blade never visits."""

    flap_travel: float = 1.92
    """Hinge upper limit [rad] -- 110 degrees.

    Past vertical, which is where a real flap folds back onto the carton's
    outside. At 90 degrees an open flap balances on its limit and the lift path
    fouls it; past 110 the two flaps meet over the seam."""

    flap_armature: float = 5.0e-4
    """Rotor inertia on each hinge [kg*m^2], and it is load-bearing.

    The tape's hold is an **explicit** generalized force, so its gains are
    bounded by ``kp < I/dt^2`` and ``kd < I/dt``. A 10 g flap 65 mm long has
    ``I = m*L^2/3 = 1.4e-5``, which at dt = 1/960 bounds kp at **13 N*m/rad** --
    nowhere near enough to hold a flap shut against an arm. This raises I_eff
    to 5.1e-4 and the bound to 474, which is the difference between the
    mechanism working and ringing. ``polish`` makes the same move for the same
    reason (``spindle_armature``, "dominates the gram-scale drum")."""

    flap_damping: float = 0.02
    """Passive hinge damping [N*m*s/rad]. MuJoCo integrates it implicitly, so
    it costs nothing in stability and stops an opened flap oscillating."""

    flap_mass: float = 0.010
    carton_mass: float = 0.160
    """Shell mass [kg]. A 200x150x100 double-wall shell is about this.

    With the mat at mu 0.6 the sliding budget is ``mu*m*g`` = 0.94 N unloaded,
    against a draw drag of ``mu_blade * press``. **So an unpinned carton moves**,
    which is the failure this task is about -- and it is measured by
    ``rgbs/scripts/rgbs_decant_bench.py --shove`` rather than asserted here."""

    friction: float | None = 0.6
    """Carton against the mat.

    Deliberately below polish's 0.9. Polish's mat exists so a set-down stays
    put; this one exists so the shove is reachable and the support hand has
    something to do. Cardboard on a rubber work mat measures 0.5-0.7."""

    ke: float | None = 6.0e4
    kd: float | None = 500.0
    mass: float | None = None
    shape: ShapeSpec | None = None
    color: tuple[float, float, float] | None = (0.72, 0.58, 0.40)
    flap_color: tuple[float, float, float] = (0.68, 0.54, 0.37)
    finish: Finish | None = KRAFT
    """Corrugated board, textured, for the path tracer; the colours above are
    what every other backend draws."""

    def __post_init__(self) -> None:
        # The floor slab is what `EntityRigid._add_primitive` builds; the walls
        # and the flaps are this entity's. Derived rather than stated so the
        # shape and `outer` cannot disagree.
        if self.shape is None:
            hx, hy = 0.5 * self.outer[0], 0.5 * self.outer[1]
            object.__setattr__(self, "shape", ShapeSpec(kind="box", half_extents=(hx, hy, 0.5 * self.wall)))
        if self.mass is None:
            object.__setattr__(self, "mass", float(self.carton_mass))
        super().__post_init__()


def carton_frame(spec: SpecShippingCarton) -> dict[str, Any]:
    """Every derived length, in the carton's own frame.

    Origin at the floor slab's centre, ``+z`` up, ``+x`` the long axis, and the
    seam along ``+x`` at ``y = 0``. This is the ONE place the geometry is
    derived: the entity builds from it, :mod:`.seam` lays its strips and cells
    out on it and asserts against it, ``routine_decant.py`` aims with it, and
    ``rgbs/scripts/rgbs_decant_bench.py`` reads it rather than restating it --
    ``polish_bench``'s rule, that a bench which measures a press the machine is
    not making must not be constructible.

    Both flap links carry local ``+y`` toward their free edge (the far one
    through ``R_z(pi)``), so ``flap_free_y`` and ``seam_centre_y`` are the same
    number on either flap: the free edge, and the seam's centreline half a gap
    past it.
    """
    L, W, H = (float(v) for v in spec.outer)
    wall = float(spec.wall)
    gap = float(spec.seam_gap)
    half_wall = 0.5 * wall
    # The floor's centre is the origin, so its top face is at +half_wall and the
    # walls stand from there to the rim.
    floor_half = (0.5 * L, 0.5 * W, half_wall)
    wall_z = H - wall  # wall height above the floor's top face
    rim_z = half_wall + wall_z
    # Flaps hinge on the wall's INNER top edge, so a shut flap lies flush.
    hinge_y = 0.5 * W - wall
    flap_hy = 0.5 * (hinge_y - 0.5 * gap)
    flap_half = (0.5 * L, flap_hy, 0.5 * float(spec.flap_thickness))
    seam_z = rim_z + float(spec.flap_thickness)
    return {
        "floor_half": np.array(floor_half),
        "wall": wall,
        "wall_z": wall_z,
        "rim_z": rim_z,
        "hinge_y": hinge_y,
        "flap_half": np.array(flap_half),
        "flap_free_y": 2.0 * flap_hy,
        "seam_gap": gap,
        "seam_centre_y": 2.0 * flap_hy + 0.5 * gap,
        "seam_p0": np.array([-0.5 * L, 0.0, seam_z]),
        "seam_dir": np.array([1.0, 0.0, 0.0]),
        "seam_len": L,
        "seam_z": seam_z,
        "opening": 2.0 * (0.5 * W - wall),
        "cavity": (-0.5 * L + wall, -hinge_y, half_wall, 0.5 * L - wall, hinge_y, rim_z),
    }


@register_entity("shipping_carton")
class EntityShippingCarton(EntityRigid):
    """The carton: floor and walls on one free body, two flap links hung off it."""

    spec: SpecShippingCarton

    def __init__(self, spec: SpecShippingCarton, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.flap_bodies: list[int] = []
        self.flap_joints: list[int] = []
        self.flap_coords: list[int] = []
        self.flap_dofs: list[int] = []
        self.flap_shapes: list[int] = []
        self.flap_yaw: list[float] = []
        self._rest_xy = np.zeros(2)
        self._rest_yaw = 0.0
        self._published_at: tuple[Any, dict[str, float]] | None = None

    # ------------------------------------------------------------------ build
    def build(self, builder: newton.ModelBuilder, episode) -> None:
        # The free body, the free joint, the articulation and the FLOOR.
        super().build(builder, episode)
        spec, f = self.spec, carton_frame(self.spec)
        cfg = self._contact_cfg(builder)
        ident = wp.quat_identity()
        L, W = float(spec.outer[0]), float(spec.outer[1])
        wall, half_wall = f["wall"], 0.5 * f["wall"]
        mid_z = half_wall + 0.5 * f["wall_z"]

        for name, centre, half in (
            ("wall_xlo", (-0.5 * L + half_wall, 0.0, mid_z), (half_wall, 0.5 * W, 0.5 * f["wall_z"])),
            ("wall_xhi", (0.5 * L - half_wall, 0.0, mid_z), (half_wall, 0.5 * W, 0.5 * f["wall_z"])),
            ("wall_ylo", (0.0, -0.5 * W + half_wall, mid_z), (0.5 * L - wall, half_wall, 0.5 * f["wall_z"])),
            ("wall_yhi", (0.0, 0.5 * W - half_wall, mid_z), (0.5 * L - wall, half_wall, 0.5 * f["wall_z"])),
        ):
            builder.add_shape_box(
                self.body,
                xform=wp.transform(wp.vec3(*(float(v) for v in centre)), ident),
                hx=float(half[0]),
                hy=float(half[1]),
                hz=float(half[2]),
                cfg=cfg,
                color=spec.color,
                label=f"{spec.name}_{name}",
            )

        # --- the two flaps, on revolutes to the carton's own body -----------
        # Both links are built with local +y toward the free edge and local +x
        # the hinge axis, the far one carried through R_z(pi). So +q opens BOTH
        # flaps upward, `flap_open` reads the pair symmetrically, and the seam's
        # strip and cell layout is one formula rather than two.
        fh = f["flap_half"]
        for side, sign in (("near", -1.0), ("far", 1.0)):
            # Local +y must point INWARD, over the opening, so the two plates
            # meet across `seam_gap` on the centreline. The near flap hinges on
            # the -y wall and so reaches toward +y (yaw 0); the far one hinges on
            # +y and reaches toward -y (yaw pi). Inverted, the flaps extend
            # outboard into free air and the seam lands 148 mm off the
            # centreline -- which reads as a press that touches nothing.
            yaw = 0.0 if sign < 0 else math.pi
            q_yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
            anchor = wp.transform(wp.vec3(0.0, sign * f["hinge_y"], f["rim_z"]), q_yaw)
            link = builder.add_link(xform=self._flap_world(anchor), label=f"{spec.name}_flap_{side}")
            shape = builder.add_shape_box(
                link,
                xform=wp.transform(wp.vec3(0.0, float(fh[1]), float(fh[2])), ident),
                hx=float(fh[0]),
                hy=float(fh[1]),
                hz=float(fh[2]),
                cfg=cfg,
                color=spec.flap_color,
                label=f"{spec.name}_flap_{side}",
            )
            j = builder.add_joint_revolute(
                parent=self.body,
                child=link,
                parent_xform=anchor,
                child_xform=wp.transform(),
                axis=(1.0, 0.0, 0.0),
                target_ke=0.0,
                target_kd=0.0,
                actuator_mode=newton.JointTargetMode.NONE,
                # `limit_lower = 0` IS the carton's rim: a flap is filtered
                # against its parent, so a shut flap has nothing to rest on and
                # the joint limit has to be the stop. It also means the tape's
                # torque only ever has to resist OPENING, which is what tape does.
                limit_lower=0.0,
                limit_upper=float(spec.flap_travel),
                armature=float(spec.flap_armature),
                damping=float(spec.flap_damping),
                label=f"{spec.name}_flap_{side}",
            )
            self.flap_bodies.append(int(link))
            self.flap_yaw.append(float(yaw))
            self.flap_joints.append(int(j))
            self.flap_shapes.append(int(shape))
            self.flap_coords.append(int(builder.joint_q_start[j]))
            self.flap_dofs.append(int(builder.joint_qd_start[j]))
        builder.add_articulation(self.flap_joints, label=f"{spec.name}_flaps")

        # `_add_primitive` already set the declared mass, on the FLOOR alone --
        # the walls did not exist yet. `set_body_mass` rescales and preserves
        # the distribution, so re-stating it here is what makes the declared
        # mass the last word AND the inertia include the walls.
        set_body_mass(builder, self.body, float(spec.carton_mass))
        for link in self.flap_bodies:
            set_body_mass(builder, link, float(spec.flap_mass))
        self._assert_layout(f)

    def _flap_world(self, anchor: wp.transform) -> wp.transform:
        """A flap link's world xform: the carton's own placed pose, then the
        hinge anchor in its frame."""
        pose = self._placed(self._episode_pose(None))
        base = wp.transform(wp.vec3(*(float(v) for v in pose[:3])), wp.quat(*(float(v) for v in pose[3:7])))
        return wp.transform_multiply(base, anchor)

    def _assert_layout(self, f: dict[str, Any]) -> None:
        """What the arms and the seam both depend on, checked where it is built."""
        spec = self.spec
        opening = float(f["opening"])
        assert opening >= GRIP_SPAN + 0.020, (
            f"the carton's opening is {opening * 1e3:.0f} mm against a {GRIP_SPAN * 1e3:.0f} mm "
            "jaw span; the decant lift puts the whole hand inside it once both flaps are laid "
            "back -- widen `outer[1]`"
        )
        spanned = 4.0 * float(f["flap_half"][1]) + float(spec.seam_gap)
        assert abs(spanned - 2.0 * float(f["hinge_y"])) < 1.0e-9, (
            f"the two flaps span {spanned * 1e3:.2f} mm across a {2 * f['hinge_y'] * 1e3:.2f} mm "
            "opening; they must meet across `seam_gap` exactly or the seam is not where it is drawn"
        )
        assert float(spec.seam_gap) >= FINGERTIP_BLADE + 0.002, (
            f"the seam gap is {spec.seam_gap * 1e3:.0f} mm against a {FINGERTIP_BLADE * 1e3:.0f} mm "
            "fingertip blade; opening the far flap puts one into the gap and it needs a millimetre "
            "a side"
        )
        floor = float(spec.flap_mass) * (2.0 * float(f["flap_half"][1])) ** 2 / 3.0
        assert float(spec.flap_armature) >= 20.0 * floor, (
            f"flap_armature {spec.flap_armature:.1e} is under 20x the flap's own inertia "
            f"{floor:.1e} kg*m^2. The tape's hold is an EXPLICIT generalized force, so kp is "
            f"bounded by I/dt^2 -- at the physical inertia that bound is "
            f"{floor / (1.0 / 960.0) ** 2:.1f} N*m/rad and the tape needs a hundred times it"
        )

    # ------------------------------------------------------------------- bind
    def bind(self, model: newton.Model) -> None:
        super().bind(model)
        pose = model.body_q.numpy()[self.body]
        self._rest_xy = np.asarray(pose[:2], dtype=np.float64).copy()
        self._rest_yaw = _yaw_of(pose[3:7])

    # --------------------------------------------------------------- queries
    def flap_angles(self, body_q) -> tuple[float, float]:
        """Each flap's hinge angle [rad], from the poses alone.

        Read off ``body_q`` rather than off ``joint_q`` because the evaluator's
        context carries body poses and **not** joint coordinates -- a
        ``published_states`` that reaches for ``context.joint_q`` returns nothing
        and the state it declares scores every episode zero, which is what the
        init check caught here.

        The flap's orientation in the base's frame is ``R_z(yaw) @ R_x(q)`` by
        construction, so undoing the anchor yaw leaves a pure hinge rotation and
        ``q`` is read straight off it.
        """
        base = _rot(np.asarray(body_q[self.body], dtype=np.float64)[3:7])
        out: list[float] = []
        for link, yaw in zip(self.flap_bodies, self.flap_yaw, strict=True):
            rel = base.T @ _rot(np.asarray(body_q[link], dtype=np.float64)[3:7])
            c, s_ = math.cos(-yaw), math.sin(-yaw)
            unyaw = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]) @ rel
            out.append(abs(math.atan2(float(unyaw[2, 1]), float(unyaw[1, 1]))))
        return out[0], out[1]

    # ------------------------------------------------------------------ reset
    def reset(self, pipeline, episode) -> None:
        """The base pose, then the hinges, then the rest reference.

        The rest pose is re-recorded from what was just written rather than from
        the spec, so ``carton_shift`` reads 0.0 at t=0 by construction instead of
        by the two happening to agree."""
        super().reset(pipeline, episode)
        for state in (pipeline.state_0, pipeline.state_1):
            if state is None:
                continue
            q = state.joint_q.numpy()
            qd = state.joint_qd.numpy()
            for coord, dof in zip(self.flap_coords, self.flap_dofs, strict=True):
                q[coord] = 0.0
                qd[dof] = 0.0
            state.joint_q.assign(q)
            state.joint_qd.assign(qd)
        pose = pipeline.state_0.body_q.numpy()[self.body]
        self._rest_xy = np.asarray(pose[:2], dtype=np.float64).copy()
        self._rest_yaw = _yaw_of(pose[3:7])
        self._published_at = None


def _rot(quat_xyzw) -> np.ndarray:
    """Rotation matrix from an xyzw quaternion."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _yaw_of(quat_xyzw) -> float:
    """Heading about +z, from an xyzw quaternion."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
