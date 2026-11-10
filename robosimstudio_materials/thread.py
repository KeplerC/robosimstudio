# SPDX-License-Identifier: Apache-2.0
"""The ``thread`` kind: rigid parts that screw together -- a bottle and its cap, a bulb and its socket -- on real thread geometry.

This file is what ``rss.bottle``, ``rss.cap``, ``rss.bulb`` and ``rss.socket``
build, imported as ``robosimstudio_materials.thread``. Run
``examples/08_demos/unscrew_the_cap/unscrew_the_cap.py`` first. Nothing joins a cap to its bottle:
each part is one rigid body whose thread is a watertight mesh, a trapezoid ridge
wound round the neck's outside and another round the cap's inside, and what
holds the cap on, turns a quarter turn into three quarters of a millimetre of
rise and lets it go after two turns is contact between the two ridges.

**It needs Newton's collision pipeline, which is what** :data:`THREAD_60HZ`
**stages.** MuJoCo collides a mesh as its convex hull, and the hull of a cap is
a solid plug. Newton collides two meshes through their signed-distance fields,
so the parts here are cooked into one each, and the table hands MuJoCo the
contacts Newton found (``use_mujoco_contacts=False``): the arm, the hand and the
parts stay on one solver, with the arm's own actuators and gravity compensation.

Measured on the default thread (3 mm pitch, two turns, 27.6 mm over the neck's
crests), a 2.9 g cap on a clamped bottle, 1200 Hz, RTX PRO 6000
(``tests/test_thread_pair``): turned by a torque of 2 mN m it rises 3.001 mm a
turn over the two turns, 8 um rms off the helix; a pull ramped to 20 N in half
a second raises it 0.26 mm, the thread's own play, and it stays on. What sets
how hard a thread holds is the step rate and the parts'
:attr:`~robosimstudio.specs.ObjectSpec.impedance`, not the resolution of either
mesh, and that is why the parts state 0.998. On the same solver and pipeline,
driven outside the scene layer: signed-distance fields of 64, 128 and 256 voxels
and meshes of 17k and 34k triangles turned alike (7 to 10 um rms); at 600 Hz the
flanks gave 12 um a newton at Newton's default impedance and 1.3 at 0.998, and
let go between 40 and 80 N; at 1920 Hz the pair held 100 N.

**A light part tunnels where a heavy one would not.** MuJoCo acts on a contact
once the surfaces overlap, and a 3 g cap moves a ridge's whole 0.9 mm overlap in
one 600 Hz step under a 20 N force applied all at once: it ends half a pitch up,
the two ridges inside each other, and still follows the helix from there. Load
a thread over tenths of a second, as a hand does, and close a jaw on a cap
slowly (the unscrewing demo closes in two stages).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs import ObjectSpec
from robosimstudio.specs.arrangement import ArrangementSpec, CollisionSpec, backend
from robosimstudio.specs.asset import ShapeSpec
from robosimstudio.specs.base import Vec3

__all__ = [
    "THREAD_60HZ",
    "EntityThreaded",
    "ThreadSpec",
    "ThreadedSpec",
    "bottle",
    "bulb",
    "cap",
    "cap_mesh",
    "neck_mesh",
    "screwed",
    "socket",
]

PARTS = ("bottle", "cap", "bulb", "socket")
"""The parts: a bottle and a bulb carry the outside thread, a cap and a socket the inside one."""

SDF_BAND = 0.002
"""How far either side of a part's surface its signed-distance field is kept [m]: past the 0.9 mm
the two ridges overlap, which is the deepest a thread contact gets before it has failed."""


# =============================================================================
# 1. The thread, as numbers and as two meshes
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class ThreadSpec:
    """One right-handed single-start thread [m], shared by the part that carries it outside and the one inside.

    The defaults are a 28 mm bottle finish, close to an E27 lamp base: a
    trapezoid ridge 1.2 mm high, 1.8 mm wide at its root and 0.9 mm at its
    crest, wound two turns at a 3 mm pitch. Seen from above, turning the inside
    part (the cap) counter-clockwise raises it.

    Attributes:
        pitch: Rise per turn.
        turns: Turns each ridge is wound; a cap comes free after this many.
        core_radius: The outside part's radius at the ridge's root.
        ridge_height: The ridge's, so the outside part is
            ``2 * (core_radius + ridge_height)`` over its crests.
        ridge_half_base, ridge_half_crest: Half the ridge's width at its root
            and at its crest.
        taper_deg: The angle over which each end of a ridge runs in and out.
        radial_clearance: Between one part's crests and the other's wall.
        bore_radius: The hole down the outside part.
        finish_length: The outside part's threaded length, rim to collar.
        collar_radius, collar_height: A ring under the outside thread.
        wall, top: The inside part's wall and end thickness.
        skirt: How far the inside part's wall runs past its ridge's end.
        ribs, rib_height: Grip ribs round the inside part.
        segments: Mesh facets round the axis.
        row: Mesh row height along the axis through the ridge.
    """

    pitch: float = 3.0e-3
    turns: float = 2.0
    core_radius: float = 12.6e-3
    ridge_height: float = 1.2e-3
    ridge_half_base: float = 0.9e-3
    ridge_half_crest: float = 0.45e-3
    taper_deg: float = 20.0
    radial_clearance: float = 0.3e-3
    bore_radius: float = 10.5e-3
    finish_length: float = 18.0e-3
    collar_radius: float = 15.0e-3
    collar_height: float = 2.0e-3
    wall: float = 1.4e-3
    top: float = 1.5e-3
    skirt: float = 0.6e-3
    ribs: int = 30
    rib_height: float = 0.35e-3
    segments: int = 180
    row: float = 0.15e-3

    def __post_init__(self) -> None:
        if not 0.0 < self.ridge_half_crest < self.ridge_half_base:
            raise ValueError("a thread's ridge is narrower at its crest than at its root, and neither is zero")
        if self.axial_play <= 0.0:
            raise ValueError(
                f"the two ridges do not fit one pitch: {self.pitch * 1e3:g} mm leaves "
                f"{self.axial_play * 1e3:.2f} mm between their flanks"
            )
        if not 0.0 < self.radial_clearance < self.ridge_height:
            raise ValueError("the radial clearance is positive and under the ridge's height, or nothing engages")

    @property
    def crest_radius(self) -> float:
        """The outside part's radius over its ridge."""
        return self.core_radius + self.ridge_height

    @property
    def inner_radius(self) -> float:
        """The inside part's wall, at its ridge's root."""
        return self.crest_radius + self.radial_clearance

    @property
    def outer_radius(self) -> float:
        """The inside part's outside, between its ribs."""
        return self.inner_radius + self.wall

    @property
    def grip_radius(self) -> float:
        """The inside part's outside over its ribs: where a flat jaw lands."""
        return self.outer_radius + self.rib_height

    @property
    def start(self) -> float:
        """Height of the outside ridge's lower end, measured from the rim (negative: below it)."""
        return -2.5e-3 - self.pitch * self.turns

    @property
    def cap_depth(self) -> float:
        """From the inside part's seat to its open end."""
        return -(self.start - 0.5 * self.pitch - self.ridge_half_base - self.skirt)

    @property
    def axial_play(self) -> float:
        """How far the screwed-on part moves along the axis between its two flank contacts.

        The flanks are parallel, so the gap is the same across the band where the ridges overlap.
        """
        slope = (self.ridge_half_base - self.ridge_half_crest) / self.ridge_height
        return self.pitch - 2.0 * (self.ridge_half_base + self.ridge_half_crest) + 2.0 * slope * self.radial_clearance


def _ridge(theta: np.ndarray, z: float, start: float, thread: ThreadSpec) -> np.ndarray:
    """The ridge's height at each angle and one height ``z``, for a helix whose centre starts at ``start``."""
    out = np.zeros_like(theta)
    total = 2.0 * math.pi * thread.turns
    taper = math.radians(thread.taper_deg)
    for k in range(-1, math.ceil(thread.turns) + 2):
        wound = theta + 2.0 * math.pi * k
        centre = start + thread.pitch * wound / (2.0 * math.pi)
        profile = np.clip(
            (thread.ridge_half_base - np.abs(z - centre)) / (thread.ridge_half_base - thread.ridge_half_crest), 0.0, 1.0
        )
        ends = np.clip(np.minimum(wound / taper, (total - wound) / taper), 0.0, 1.0)
        out = np.maximum(out, profile * ends)
    return thread.ridge_height * out


def _sweep(radii, heights, *, poles: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rings swept round the z axis into one closed, outward-facing surface.

    ``radii[k]`` is ring ``k``'s radius at each angle and ``heights[k]`` its
    height. With ``poles`` the first and last ring close onto the axis at those
    two heights (a cup); without, the last ring joins the first (a tube).
    """
    radii = np.asarray(radii, dtype=np.float64)
    count, around = radii.shape
    theta = np.linspace(0.0, 2.0 * math.pi, around, endpoint=False)
    points = np.stack(
        [radii * np.cos(theta), radii * np.sin(theta), np.repeat(np.asarray(heights)[:, None], around, 1)], -1
    )
    vertices = [points.reshape(-1, 3)]
    i = np.arange(around)
    j = (i + 1) % around
    pairs = [(k, k + 1) for k in range(count - 1)] + ([] if poles else [(count - 1, 0)])
    faces = []
    for a, b in pairs:
        faces += [np.stack([a * around + i, a * around + j, b * around + j], -1)]
        faces += [np.stack([a * around + i, b * around + j, b * around + i], -1)]
    if poles:
        first, last = count * around, count * around + 1
        vertices.append(np.array([[0.0, 0.0, poles[0]], [0.0, 0.0, poles[1]]]))
        faces += [np.stack([np.full(around, first), j, i], -1)]
        faces += [np.stack([np.full(around, last), (count - 1) * around + i, (count - 1) * around + j], -1)]
    v = np.concatenate(vertices)
    f = np.concatenate(faces).astype(np.int32)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    if float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) < 0.0:
        f = f[:, ::-1].copy()
    return v, f


def _rows(low: float, high: float, step: float) -> np.ndarray:
    return np.linspace(low, high, max(1, math.ceil((high - low) / step - 1e-9)) + 1)


def neck_mesh(thread: ThreadSpec | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The outside-threaded part as ``(vertices, faces)``: a tube with its rim at z = 0, running down.

    Watertight, and the same frame as :func:`cap_mesh`'s when the two are
    screwed home, so a cap placed at a neck's rim with no turn is on it.
    """
    t = thread or ThreadSpec()
    theta = np.linspace(0.0, 2.0 * math.pi, t.segments, endpoint=False)
    flat = np.ones(t.segments)
    bottom = -t.finish_length
    low = t.start - t.ridge_half_base - 2 * t.row
    high = t.start + t.pitch * t.turns + t.ridge_half_base + 2 * t.row
    rings = [(t.bore_radius * flat, bottom), (t.collar_radius * flat, bottom)]
    rings += [(t.collar_radius * flat, bottom + t.collar_height), (t.core_radius * flat, bottom + t.collar_height)]
    rings += [(t.core_radius * flat, z) for z in _rows(bottom + t.collar_height, low, 0.6e-3)[1:-1]]
    rings += [(t.core_radius + _ridge(theta, z, t.start, t), z) for z in _rows(low, high, t.row)]
    rings += [(t.core_radius * flat, z) for z in _rows(high, 0.0, 0.4e-3)[1:]]
    rings += [(t.bore_radius * flat, 0.0)]
    rings += [(t.bore_radius * flat, z) for z in _rows(bottom, 0.0, 2.0e-3)[1:-1][::-1]]
    return _sweep([r for r, _ in rings], [z for _, z in rings])


def cap_mesh(thread: ThreadSpec | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The inside-threaded part as ``(vertices, faces)``: a ribbed cup, its seat at z = 0, open downward."""
    t = thread or ThreadSpec()
    theta = np.linspace(0.0, 2.0 * math.pi, t.segments, endpoint=False)
    flat = np.ones(t.segments)
    outer, inner, skirt = t.outer_radius, t.inner_radius, -t.cap_depth
    start = t.start - 0.5 * t.pitch  # half a pitch under the neck's ridge, so the two interleave
    rib = t.rib_height * 0.5 * (1.0 + np.cos(t.ribs * theta))
    low = start - t.ridge_half_base - 2 * t.row
    high = min(start + t.pitch * t.turns + t.ridge_half_base + 2 * t.row, -2 * t.row)
    rings = [(0.5 * outer * flat, t.top), ((outer - 0.5e-3) * flat, t.top), (outer * flat, t.top - 0.5e-3)]
    rings += [(outer + rib, z) for z in _rows(skirt + 0.4e-3, t.top - 1.0e-3, 0.6e-3)[::-1]]
    rings += [(outer * flat, skirt), (inner * flat, skirt)]
    rings += [(inner * flat, z) for z in _rows(skirt, low, 0.3e-3)[1:-1]]
    rings += [(inner - _ridge(theta, z, start, t), z) for z in _rows(low, high, t.row)]
    rings += [(inner * flat, z) for z in _rows(high, 0.0, 0.3e-3)[1:]]
    rings += [(0.5 * inner * flat, 0.0)]
    return _sweep([r for r, _ in rings], [z for _, z in rings], poles=(t.top, 0.0))


# =============================================================================
# 2. The parts
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class ThreadedSpec(ObjectSpec):
    """One threaded part: :class:`~robosimstudio.specs.ObjectSpec` with which part it is and its thread.

    Built by :func:`bottle`, :func:`cap`, :func:`bulb` and :func:`socket`, each
    of which says where its part's origin is. Every part's axis is its own z.

    Attributes:
        part: ``"bottle"``, ``"cap"``, ``"bulb"`` or ``"socket"``.
        thread: The thread it carries; two parts mate when theirs are equal.
        sdf_resolution: Voxels along the longest side of each mesh's
            signed-distance field. 64 to 256 turned alike; the robot runs were
            measured at 256, 0.14 mm cells.
        body: The bottle's body or the bulb's glass tube, ``(radius, height)``.
        shoulder: Height of the bottle's shoulder, body to stem.
        stem: The bottle's plain neck under its thread, ``(radius, height)``.
        contact: The bulb's base contact and the socket's centre pad,
            ``(radius, height)``: the pad stands this high off the socket's
            floor, and the bulb's contact meets it before its rim seats.
    """

    entity: str = "threaded"
    part: str = "cap"
    thread: ThreadSpec = field(default_factory=ThreadSpec)
    sdf_resolution: int = 256
    body: tuple[float, float] = (0.030, 0.180)
    shoulder: float = 0.025
    stem: tuple[float, float] = (0.0125, 0.026)
    contact: tuple[float, float] = (0.004, 0.0003)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.part not in PARTS:
            raise ValueError(f"threaded part {self.name!r}: part must be one of {PARTS}, got {self.part!r}")
        if self.asset is not None:
            raise ValueError(f"threaded part {self.name!r}: built here from its thread, so it takes no asset")
        # The cylinder the part fits in: a stated primitive is what tells the rigid entity it builds this itself.
        t, tall = self.thread, self.thread.top + self.thread.cap_depth
        wide = t.grip_radius
        if self.part == "bottle":
            wide, tall = max(self.body[0], t.collar_radius), self.rim
        elif self.part == "bulb":
            wide, tall = max(self.body[0], t.collar_radius), t.finish_length + self.body[1] + self.body[0]
        object.__setattr__(self, "shape", ShapeSpec(kind="cylinder", radius=wide, half_height=0.5 * tall))
        if int(self.sdf_resolution) % 8:
            raise ValueError(f"threaded part {self.name!r}: sdf_resolution must be a multiple of 8")

    @property
    def rim(self) -> float:
        """Height of the mating point over the part's origin [m].

        A bottle's rim, a socket's floor: where :func:`screwed` puts the other
        part's origin when it is screwed home. A cap's seat and a bulb's tip
        are their origins, 0.
        """
        t = self.thread
        if self.part == "bottle":
            return self.body[1] + self.shoulder + self.stem[1] + t.finish_length
        return t.top if self.part == "socket" else 0.0

    @property
    def grasp_height(self) -> float:
        """Height over the origin of the middle of the band a hand closes on [m]: the ribs, the glass, the body."""
        t = self.thread
        if self.part == "cap":
            return 0.5 * (t.top - t.cap_depth)
        if self.part == "bulb":
            return t.finish_length + 0.55 * self.body[1]
        if self.part == "bottle":
            return 0.5 * self.body[1]
        return t.top + 0.5 * t.cap_depth


_DEFAULTS: dict[str, Any] = {"friction": 0.3, "ke": 1.0e7, "kd": 1.0e4, "impedance": 0.998, "contact_gap": 0.002}
"""What every part states unless told otherwise: a hard plastic's friction, a stiff contact,
and the impedance that makes MuJoCo hold it (:attr:`~robosimstudio.specs.ObjectSpec.impedance`)."""


def bottle(
    *, name: str = "bottle", at: Vec3 = (0.0, 0.0, 0.0), thread: ThreadSpec | None = None, **kwargs
) -> ThreadedSpec:
    """A bottle with a threaded neck, standing on its base.

    ``at`` is the middle of its base, so ``z = 0`` stands it on the bench. A
    60 mm body 180 mm tall, a shoulder and a 25 mm stem carry the rim 249 mm
    up; 30 g unless ``mass`` says otherwise. It is that tall for the hand that
    holds it: a YAM holding the body near its base carries its wrist camera
    183 mm up, under the pads of the hand turning the cap, which hang 56 mm
    below the rim. ``static=True`` clamps it where it stands.
    """
    spec = {"mass": 0.030, "color": (0.35, 0.62, 0.50), **_DEFAULTS, **kwargs}
    return ThreadedSpec(name=name, part="bottle", thread=thread or ThreadSpec(), pose=at, **spec)


def cap(*, name: str = "cap", at: Vec3 = (0.0, 0.0, 0.0), thread: ThreadSpec | None = None, **kwargs) -> ThreadedSpec:
    """A ribbed screw cap. ``at`` is its seat, the inside of its top: a bottle's rim when it is screwed home.

    31.0 mm across, 31.7 mm over its 30 ribs, 13 mm tall and 2.9 g of
    polypropylene at the default thread. :func:`screwed` gives the pose that
    puts it on a bottle. Put down, it rests on a bench or a box either way up;
    set over a post narrower than its mouth it sank through the post's end at
    40 mm/s.
    """
    spec = {"density": 950.0, "color": (0.86, 0.16, 0.14), **_DEFAULTS, **kwargs}
    return ThreadedSpec(name=name, part="cap", thread=thread or ThreadSpec(), pose=at, **spec)


def bulb(*, name: str = "bulb", at: Vec3 = (0.0, 0.0, 0.0), thread: ThreadSpec | None = None, **kwargs) -> ThreadedSpec:
    """A lamp: a threaded base under a short glass tube with a round end. ``at`` is the base's tip.

    The glass is a 34 mm tube 22 mm long, 20 g with its base. A contact in the
    middle of the tip meets the socket's pad as the bulb comes home.
    """
    spec = {"mass": 0.020, "body": (0.017, 0.022), "color": (1.0, 0.96, 0.78), **_DEFAULTS, **kwargs}
    return ThreadedSpec(name=name, part="bulb", thread=thread or ThreadSpec(), pose=at, **spec)


def socket(
    *, name: str = "socket", at: Vec3 = (0.0, 0.0, 0.0), thread: ThreadSpec | None = None, **kwargs
) -> ThreadedSpec:
    """A lamp holder: the inside thread, mouth up, with a contact pad in its floor. ``at`` is its underside.

    Fixed where it stands unless ``static=False``: a lamp holder is screwed to
    something.
    """
    spec = {"static": True, "color": (0.92, 0.90, 0.84), **_DEFAULTS, **kwargs}
    return ThreadedSpec(name=name, part="socket", thread=thread or ThreadSpec(), pose=at, **spec)


def screwed(onto: ThreadedSpec, *, turns_out: float = 0.0, lift: float = 1.0e-4) -> dict[str, tuple[float, ...]]:
    """Where the mating part goes on ``onto``, as the ``at`` and ``quat`` to build it with.

    A cap on a bottle, a bulb in a socket: ``rss.cap(**screwed(flask))``.
    ``turns_out`` backs it off that many turns, raised a pitch for each; past
    the thread's ``turns`` it stands on the ridge's end, not held. ``lift``
    starts it that far clear [m].

    Raises:
        ValueError: If ``onto`` is a cap or a bulb, or is not upright.
    """
    if onto.part not in ("bottle", "socket"):
        raise ValueError(f"{onto.name!r} is a {onto.part}: a cap goes on a bottle and a bulb in a socket")
    if tuple(float(v) for v in onto.quat) != (0.0, 0.0, 0.0, 1.0):
        raise ValueError(f"{onto.name!r} is turned: screwed() places a part on one standing upright")
    x, y, z = (float(v) for v in onto.pose[:3])
    half = math.pi * float(turns_out)
    height = z + onto.rim + onto.thread.pitch * float(turns_out) + float(lift)
    return {"at": (x, y, height), "quat": (0.0, 0.0, math.sin(half), math.cos(half))}


class EntityThreaded(EntityRigid):
    """A :class:`ThreadedSpec`: one body, its thread a mesh with a cooked signed-distance field."""

    spec: ThreadedSpec
    thread_shape: int = -1
    """The shape that carries the thread, once built."""
    contact_shape: int = -1
    """A bulb's base contact or a socket's pad, once built; -1 on a bottle and a cap."""

    def _add_primitive(self, builder, pose) -> None:
        import warp as wp

        from robosimstudio.scene.build.bodies import set_body_mass

        spec, t = self.spec, self.spec.thread
        xform = wp.transform(tuple(float(v) for v in pose[:3]), wp.quat(*[float(v) for v in pose[3:7]]))
        if spec.static:
            body = self._add_welded_link(builder, xform)
        else:
            body = builder.add_body(xform=xform, label=spec.name)
            joint = builder.joint_count - 1
            self.coord_start = int(builder.joint_q_start[joint])
            self.dof_start = int(builder.joint_qd_start[joint])
        self.body = body
        cfg = self._contact_cfg(builder)
        cfg.density = float(spec.density)
        flipped = wp.quat(1.0, 0.0, 0.0, 0.0)  # half a turn about x: the thread stays right-handed
        upright = wp.quat_identity()

        def cylinder(radius: float, low: float, high: float, label: str) -> int:
            at = wp.transform((0.0, 0.0, 0.5 * (low + high)), upright)
            return builder.add_shape_cylinder(
                body, xform=at, radius=radius, half_height=0.5 * (high - low), cfg=cfg, label=f"{spec.name}_{label}"
            )

        def threaded(mesh_of, at: float, quat) -> int:
            mesh = self._cooked(mesh_of(t))
            where = wp.transform((0.0, 0.0, at), quat)
            return builder.add_shape_mesh(body, xform=where, mesh=mesh, cfg=cfg, label=f"{spec.name}_thread")

        if spec.part == "cap":
            self.thread_shape = threaded(cap_mesh, 0.0, upright)
        elif spec.part == "socket":
            self.thread_shape = threaded(cap_mesh, t.top, flipped)
            self.contact_shape = cylinder(spec.contact[0] + 0.001, t.top, t.top + spec.contact[1], "pad")
        elif spec.part == "bulb":
            radius, tube = spec.body
            self.thread_shape = threaded(neck_mesh, 0.0, flipped)
            self.contact_shape = cylinder(spec.contact[0], 0.0, 0.008, "contact")
            cylinder(radius, t.finish_length, t.finish_length + tube, "glass")
            dome = wp.transform((0.0, 0.0, t.finish_length + tube), upright)
            builder.add_shape_sphere(body, xform=dome, radius=radius, cfg=cfg, label=f"{spec.name}_dome")
        else:
            (radius, height), (stem, stem_height) = spec.body, spec.stem
            cylinder(radius, 0.0, height, "body")
            for k in range(4):  # the shoulder, as four steps
                r = radius + (stem - radius) * (k + 0.5) / 4.0
                cylinder(r, height + spec.shoulder * k / 4.0, height + spec.shoulder * (k + 1) / 4.0, f"shoulder_{k}")
            cylinder(stem, height + spec.shoulder, height + spec.shoulder + stem_height, "stem")
            self.thread_shape = threaded(neck_mesh, spec.rim, upright)
        if spec.mass is not None and not spec.static:
            set_body_mass(builder, body, float(spec.mass))

    def _cooked(self, mesh: tuple[np.ndarray, np.ndarray]):
        """The mesh as Newton's, with its signed-distance field cooked and cached on disk by content."""
        import tempfile
        from pathlib import Path

        import newton
        import warp as wp

        if not wp.get_device().is_cuda:
            raise RuntimeError(
                f"threaded part {self.spec.name!r}: a thread collides through a signed-distance field, which "
                "Newton bakes on CUDA only, and this scene is building on a CPU"
            )
        vertices, faces = mesh
        cooked = newton.Mesh(vertices.astype(np.float32), faces.reshape(-1).astype(np.int32))
        directory = Path(tempfile.gettempdir()) / "robosimstudio_sdf_cache"
        directory.mkdir(parents=True, exist_ok=True)
        cooked.build_sdf(
            max_resolution=int(self.spec.sdf_resolution),
            narrow_band_range=(-SDF_BAND, SDF_BAND),
            margin=SDF_BAND,
            cache_dir=directory,
        )
        return cooked


register_entity("threaded")(EntityThreaded)


# =============================================================================
# 3. The physics that carries a thread
# =============================================================================

THREAD_60HZ = ArrangementSpec(
    control_hz=60.0,
    substeps=20,
    backends=(("*", backend("mujoco", use_mujoco_contacts=False, nconmax=2048, njmax=8192, ls_iterations=50)),),
    collision=CollisionSpec(
        rigid_gap=2.0e-3,
        finger_priority=1,
        rigid_contact_max=2048,
        contact_ke=6.0e4,
        contact_kd=500.0,
    ),
)
"""The arm, the hand and every rigid part on one MuJoCo solver, fed the contacts Newton's pipeline finds: 60 Hz over 1200 Hz.

**What a threaded part needs, and what a hand needs to hold one there.** Newton's
pipeline collides the parts' meshes through their signed-distance fields, which
MuJoCo's own collision cannot (a mesh is its hull there). On that pipeline a YAM
closing 50 N a jaw on a 3 g cap used to lose it: it slid out of the fingers at
4 to 5 mm/s. Two settings here are what holds it, with the parts' own
:attr:`~robosimstudio.specs.ObjectSpec.impedance`:

- ``finger_priority=1``: the fingers' own MuJoCo contact class is the law of
  what they touch, where the default mixes it with the part's softer one. Mixed,
  a loose cap closed on and lifted is lost; with it the cap comes up with the
  hand and creeps 0.015 mm/s in the jaws (``tests/test_thread_grasp``).
- ``rigid_gap`` of 2 mm: Newton's default reports a contact from 100 mm away,
  and every finger shape did, 380 of them before the hand had touched anything
  and 951 at the busiest step, at half the speed. With it two arms unscrewing a
  cap peak at 262 contacts of the 2048 this table allows.

1200 Hz is where a hand's ordinary roughness stays inside a thread's play. Two
YAM arms unscrewing a cap run at 0.75 x real time on an RTX PRO 6000,
graph-captured (``examples/08_demos/unscrew_the_cap``).

It is for rigid bodies, as :data:`~robosimstudio.specs.MUJOCO_500HZ` is, and
needs a GPU: MuJoCo's C library takes no contacts from outside, so a CPU build
refuses ``use_mujoco_contacts=False`` by name.
"""
