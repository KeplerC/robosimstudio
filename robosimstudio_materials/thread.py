# SPDX-License-Identifier: Apache-2.0
"""The ``thread`` kind: rigid parts that screw together -- a bottle and its cap, a bulb and its socket -- on real thread geometry.

This file is what ``rss.bottle``, ``rss.cap``, ``rss.bulb`` and ``rss.socket``
build, imported as ``robosimstudio_materials.thread``. Run
``examples/08_demos/unscrew_the_cap/unscrew_the_cap.py`` first. Nothing joins a cap to its bottle:
each part is one rigid body whose thread is a watertight mesh, a trapezoid ridge
wound round the neck's outside and another round the cap's inside, and what
holds the cap on, turns a quarter turn into three quarters of a millimetre of
rise and lets it go after two turns is contact between the two ridges. The
bulb and its socket are a household lamp on the medium Edison screw, whose
thread is rounded and whose numbers are the published ones (:class:`LampSpec`).

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

Measured on the lamp, alone in its holder (``tests/test_lamp_pair``): turned by
a torque of 10 mN m from three turns out it goes down 3.62 mm a turn against
the 3.629 of its pitch, over two and a half turns and never farther off the
helix than the 0.18 mm of its thread's play, until its eyelet is on the holder's
centre contact; 390 contacts, 1.6 x real time. **A round thread clamps harder
than it looks.** Its flanks are arcs, and turned on at a turn in 4 s into a
closed contact the lamp went from under 2 N to as much as 115 N in one control
step: slow the hand as it comes home
(``examples/08_demos/screw_in_the_bulb`` does, and reads 2 to 4 N).

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

from robosimstudio.core.render.finish import GLOSS, Finish
from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs import ObjectSpec
from robosimstudio.specs.arrangement import ArrangementSpec, CollisionSpec, backend
from robosimstudio.specs.asset import ShapeSpec
from robosimstudio.specs.base import Vec3

__all__ = [
    "THREAD_60HZ",
    "EntityThreaded",
    "LampSpec",
    "ThreadSpec",
    "ThreadedSpec",
    "bottle",
    "bulb",
    "cap",
    "cap_mesh",
    "holder_meshes",
    "lamp_meshes",
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
# 2. The lamp: a medium Edison screw under a pear of glass, and its holder
# =============================================================================

INCH = 0.0254


@dataclass(frozen=True, kw_only=True)
class LampSpec:
    """A household lamp and the holder it screws into [m]: an A19 bulb on a medium Edison screw (E26).

    The numbers are published ones, kept in the inches they are printed in:

    - **The thread** is the American Standard rolled thread for the screw
      shells of lamp bases and sockets, medium size, as Machinery's Handbook
      tabulates it (read from Engineers Edge's copy of that table): seven
      threads to the inch, a crest and a root that are arcs of 0.047 in, 0.033
      in deep; the base 1.037 to 1.031 in over its crests and 0.971 to 0.965 in
      at its roots, the socket 1.053 to 1.045 in and 0.987 to 0.979 in. Each
      diameter here is the middle of its two limits, which leaves 0.19 mm
      between the two threads all round.
    - **The base** is GE Lighting Components' data sheet 7675-1A, "Medium Screw
      Bases" (November 1994): 0.937 in tall, 1.075 in over its top rim, a centre
      eyelet of 0.389 in, glass insulated; its thread columns repeat the
      standard's (depth 0.033 in, radius 0.047 in, crest 1.037 in, root 0.971 in).
    - **The bulb** is the A19: 19 eighths of an inch at its widest and about
      4 3/8 in long (Wikipedia, "A-series light bulb", after ANSI C79.1 and
      IEC 60887), inside the 2.38 in by 4.5 in a maker lists as its maximum
      (Satco S2419, a 60 W A19, whose 0.07 lb is the 32 g :func:`bulb` weighs).

    What is not from a drawing: the pear's outline below its sphere (a cone
    tangent to it, into a 26.4 mm neck), how the base's 23.8 mm divide between
    eyelet, insulator, dome, thread and rim, the turns of thread on each part,
    and everything about the holder but its thread.

    Attributes:
        pitch: Rise per turn, a seventh of an inch.
        base_major, base_minor: The base's **diameters** over its crests and at
            its roots.
        holder_major, holder_minor: The holder's, at its roots and over its
            crests.
        base_turns, holder_turns: Turns of thread wound on each.
        taper_deg: The angle over which each end of a thread runs in and out.
        base_height: From the eyelet's face to the top of the base's rim.
        rim_diameter: Over the base's top rim, where the glass is cemented in.
        eyelet_diameter: The centre contact at the base's tip.
        bulb_diameter: The glass at its widest.
        length: The whole lamp, eyelet to the top of the glass.
        well: The holder's depth, its mouth above its centre contact.
        pad: The holder's centre contact, ``(radius, height off the floor)``.
        housing: The holder's body, ``(radius, flange radius, foot)``: ``foot``
            is how far its underside is below the centre contact.
        segments: Mesh facets round the axis.
        row: Mesh row height along the axis through a thread.
    """

    pitch: float = INCH / 7.0
    base_major: float = 0.5 * (1.037 + 1.031) * INCH
    base_minor: float = 0.5 * (0.971 + 0.965) * INCH
    holder_major: float = 0.5 * (1.053 + 1.045) * INCH
    holder_minor: float = 0.5 * (0.987 + 0.979) * INCH
    base_turns: float = 3.5
    holder_turns: float = 2.75
    taper_deg: float = 30.0
    base_height: float = 0.937 * INCH
    rim_diameter: float = 1.075 * INCH
    eyelet_diameter: float = 0.389 * INCH
    bulb_diameter: float = 2.375 * INCH
    length: float = 4.375 * INCH
    well: float = 21.3e-3
    pad: tuple[float, float] = (5.5e-3, 1.0e-3)
    housing: tuple[float, float, float] = (21.0e-3, 30.0e-3, 12.0e-3)
    segments: int = 180
    row: float = 0.12e-3

    def __post_init__(self) -> None:
        if not 0.0 < self.radial_clearance < self.depth:
            raise ValueError(
                "the holder's thread is wider than the base's by less than the thread's depth, and by more than 0"
            )
        if abs(0.5 * (self.holder_major - self.holder_minor) - self.depth) > 1.0e-5:
            raise ValueError("the two threads are one depth: the holder's diameters differ by what the base's do")
        if self.arc_radius < 0.25 * self.pitch:
            raise ValueError(
                "a thread this deep for its pitch has no round profile: its arcs would be past a half circle"
            )

    @property
    def depth(self) -> float:
        """The thread's depth, crest to root."""
        return 0.5 * (self.base_major - self.base_minor)

    @property
    def arc_radius(self) -> float:
        """The radius of the two arcs, crest and root, that meet at half depth and make the profile."""
        return (self.pitch**2 / 16.0 + self.depth**2 / 4.0) / self.depth

    @property
    def radial_clearance(self) -> float:
        """Between the two threads all round, when the lamp is centred in the holder."""
        return 0.5 * (self.holder_major - self.base_major)

    @property
    def axial_play(self) -> float:
        """How far the lamp moves along the axis between its two flank contacts, centred in the holder."""
        across = np.linspace(-0.5, 0.5, 4001)
        low, high = 0.0, 0.5 * self.pitch
        for _ in range(40):
            shift = 0.5 * (low + high)
            gap = self.radial_clearance + _round(across, self) - _round(across - shift / self.pitch, self)
            low, high = (shift, high) if float(gap.min()) > 0.0 else (low, shift)
        return 2.0 * low

    @property
    def thread_foot(self) -> float:
        """Height over the eyelet's face of the lower end of the base's threaded band."""
        return self.base_height - 1.6e-3 - self.pitch * (self.base_turns + 1.0)

    @property
    def crest_start(self) -> float:
        """Height over the eyelet's face at which the base's crest starts winding."""
        return self.thread_foot + 0.5 * self.pitch

    @property
    def equator(self) -> float:
        """Height over the eyelet's face of the glass at its widest: where a hand closes on it."""
        return self.length - 0.5 * self.bulb_diameter

    @property
    def mouth(self) -> float:
        """Height of the holder's mouth over its underside."""
        return self.housing[2] + self.well

    @property
    def catch(self) -> float:
        """How far above home the lamp is as its thread first meets the holder's [m]: about three turns' worth."""
        top = self.crest_start + self.pitch * (self.holder_turns + 1.0)
        return top - self.thread_foot - self.pitch


def _round(offset, lamp: LampSpec) -> np.ndarray:
    """The round profile's height over its root, ``offset`` pitches along the axis from a crest."""
    along = np.abs(offset - np.round(offset)) * lamp.pitch
    radius, depth = lamp.arc_radius, lamp.depth
    crest = depth - (radius - np.sqrt(np.maximum(radius**2 - along**2, 0.0)))
    root = radius - np.sqrt(np.maximum(radius**2 - (0.5 * lamp.pitch - along) ** 2, 0.0))
    return np.where(along <= 0.25 * lamp.pitch, crest, root)


def _wound(theta: np.ndarray, z: float, start: float, turns: float, lamp: LampSpec) -> np.ndarray:
    """A round thread's height over its root at each angle and one height ``z``.

    Its crest starts at ``start`` and winds ``turns``; each end runs in over
    :attr:`LampSpec.taper_deg`.
    """
    offset = (z - start) / lamp.pitch - theta / (2.0 * math.pi)
    turn = np.round(offset)
    along = theta + 2.0 * math.pi * turn
    taper = math.radians(lamp.taper_deg)
    ends = np.clip(np.minimum(along / taper, (2.0 * math.pi * turns - along) / taper), 0.0, 1.0)
    return ends * _round(offset - turn, lamp)


def _outline(lamp: LampSpec) -> tuple[np.ndarray, np.ndarray]:
    """The glass, as ``(radius, height)`` up its outside: a neck, a cone, and the sphere the cone is tangent to."""
    radius = 0.5 * lamp.bulb_diameter
    neck = 0.5 * lamp.rim_diameter - 0.45e-3
    corner = np.array([neck, lamp.base_height + 4.0e-3])
    centre = np.array([0.0, lamp.equator])
    to_centre = centre - corner
    reach = float(np.linalg.norm(to_centre))
    lean = math.atan2(-to_centre[0], to_centre[1]) - math.asin(radius / reach)  # the tangent's angle off the axis
    along = np.array([-math.sin(lean), math.cos(lean)])
    touch = corner + math.sqrt(reach**2 - radius**2) * along
    blend = corner + 3.5e-3 * along
    below = np.array([neck, lamp.base_height + 1.0e-3])
    f = np.linspace(0.0, 1.0, 9)[:, None]
    points = [
        np.array([[neck, lamp.base_height - 0.8e-3]]),
        (1 - f) ** 2 * below + 2 * f * (1 - f) * corner + f**2 * blend,
    ]
    points += [blend + (touch - blend) * np.linspace(0.0, 1.0, 16)[1:, None]]
    first = math.atan2(touch[1] - centre[1], touch[0])
    arc = np.linspace(first, 0.5 * math.pi, 40)[1:-1]
    points += [np.stack([radius * np.cos(arc), centre[1] + radius * np.sin(arc)], -1)]
    outline = np.concatenate(points)
    return outline[:, 0], outline[:, 1]


def lamp_meshes(lamp: LampSpec | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The lamp's three turned parts as ``(vertices, faces)``, each watertight, the eyelet's face at z = 0.

    ``"base"`` is the metal: a dome, the threaded shell and the rim.
    ``"insulator"`` is the ring of glass between it and the eyelet, and
    ``"glass"`` the bulb. The base and :func:`holder_meshes`' shell are one
    helix, so a lamp standing on the holder's centre contact is screwed home.
    """
    m = lamp or LampSpec()
    theta = np.linspace(0.0, 2.0 * math.pi, m.segments, endpoint=False)
    flat = np.ones(m.segments)
    root, rim = 0.5 * m.base_minor, 0.5 * m.rim_diameter
    eyelet, shoulder = 0.5 * m.eyelet_diameter, 9.0e-3
    foot, band = m.thread_foot, m.thread_foot + m.pitch * (m.base_turns + 1.0)
    dome = foot - 2.6e-3
    rings = [
        (shoulder + (root - shoulder) * math.sin(0.5 * math.pi * f) * flat, dome + (foot - dome) * f)
        for f in np.linspace(0, 1, 10)[:-1]
    ]
    rings += [(root + _wound(theta, z, m.crest_start, m.base_turns, m), z) for z in _rows(foot, band, m.row)]
    rings += [
        ((root + (rim - root) * (0.5 - 0.5 * math.cos(math.pi * f))) * flat, band + (m.base_height - band) * f)
        for f in np.linspace(0, 1, 8)[1:]
    ]
    base = _sweep([r for r, _ in rings], [z for _, z in rings], poles=(dome, m.base_height))
    low = 0.6e-3
    rings = [
        ((eyelet + 0.5e-3 + (shoulder + 0.2e-3 - eyelet - 0.5e-3) * f) * flat, low + (dome + 0.2e-3 - low) * f)
        for f in np.linspace(0, 1, 8)
    ]
    insulator = _sweep([r for r, _ in rings], [z for _, z in rings], poles=(low, dome + 0.2e-3))
    radii, heights = _outline(m)
    glass = _sweep([r * flat for r in radii], list(heights), poles=(float(heights[0]), m.length))
    return {"base": base, "insulator": insulator, "glass": glass}


def holder_meshes(lamp: LampSpec | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The holder's two turned parts as ``(vertices, faces)``, each watertight, its underside at z = 0, mouth up.

    ``"shell"`` is the threaded metal cup, its floor a pad's height under the
    centre contact; ``"housing"`` is the body round it.
    """
    m = lamp or LampSpec()
    theta = np.linspace(0.0, 2.0 * math.pi, m.segments, endpoint=False)
    flat = np.ones(m.segments)
    body, flange, home = m.housing  # ``home`` is the centre contact's face, where the lamp's eyelet comes to rest
    bore, outer = 0.5 * m.holder_major, 0.5 * m.holder_major + 1.0e-3
    floor, under, mouth = home - m.pad[1], home - m.pad[1] - 1.2e-3, m.mouth
    start = home + m.crest_start + 0.5 * m.pitch  # the holder's ridge runs half a pitch above the base's crest
    low, high = start - 0.5 * m.pitch, start + m.pitch * (m.holder_turns + 0.5)
    rings = [(outer * flat, z) for z in _rows(under, mouth, 2.0e-3)]
    rings += [(bore * flat, z) for z in _rows(high, mouth, 0.3e-3)[::-1]]
    rings += [(bore - _wound(theta, z, start, m.holder_turns, m), z) for z in _rows(low, high, m.row)[::-1][1:]]
    rings += [(bore * flat, z) for z in _rows(floor, low, 0.5e-3)[::-1][1:]]
    shell = _sweep([r for r, _ in rings], [z for _, z in rings], poles=(under, floor))
    fit = outer + 2.0e-5  # the housing's bore, a hair off the shell so no two faces share a place
    lip = mouth + 0.5e-3
    turned = [(flange, 0.0), (flange, 4.0e-3), (body + 1.5e-3, 7.0e-3), (body, 9.0e-3)]
    turned += [(body, z) for z in _rows(9.0e-3, mouth - 1.0e-3, 3.0e-3)[1:]]
    turned += [(body - 0.4e-3, mouth), (body - 1.4e-3, lip), (fit + 0.6e-3, lip), (fit, mouth)]
    turned += [(fit, z) for z in _rows(under - 2.0e-5, mouth, 3.0e-3)[::-1][1:]]
    housing = _sweep([r * flat for r, _ in turned], [z for _, z in turned], poles=(0.0, under - 2.0e-5))
    return {"shell": shell, "housing": housing}


# =============================================================================
# 3. The parts
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class ThreadedSpec(ObjectSpec):
    """One threaded part: :class:`~robosimstudio.specs.ObjectSpec` with which part it is and its thread.

    Built by :func:`bottle`, :func:`cap`, :func:`bulb` and :func:`socket`, each
    of which says where its part's origin is. Every part's axis is its own z.

    Attributes:
        part: ``"bottle"``, ``"cap"``, ``"bulb"`` or ``"socket"``.
        thread: The thread a bottle or a cap carries; two mate when theirs are equal.
        lamp: The lamp a bulb or a socket is cut to; two mate when theirs are equal.
        sdf_resolution: Voxels along the longest side of the threaded mesh's
            signed-distance field. 64 to 256 turned alike; the robot runs were
            measured at 256, 0.14 mm cells on a cap and 0.11 mm on a lamp's base.
        body: The bottle's body, ``(radius, height)``.
        shoulder: Height of the bottle's shoulder, body to stem.
        stem: The bottle's plain neck under its thread, ``(radius, height)``.
    """

    entity: str = "threaded"
    part: str = "cap"
    thread: ThreadSpec = field(default_factory=ThreadSpec)
    lamp: LampSpec = field(default_factory=LampSpec)
    sdf_resolution: int = 256
    body: tuple[float, float] = (0.030, 0.180)
    shoulder: float = 0.025
    stem: tuple[float, float] = (0.0125, 0.026)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.part not in PARTS:
            raise ValueError(f"threaded part {self.name!r}: part must be one of {PARTS}, got {self.part!r}")
        if self.asset is not None:
            raise ValueError(f"threaded part {self.name!r}: built here from its thread, so it takes no asset")
        # The cylinder the part fits in: a stated primitive is what tells the rigid entity it builds this itself.
        t, m = self.thread, self.lamp
        wide, tall = t.grip_radius, t.top + t.cap_depth
        if self.part == "bottle":
            wide, tall = max(self.body[0], t.collar_radius), self.rim
        elif self.part == "bulb":
            wide, tall = 0.5 * m.bulb_diameter, m.length
        elif self.part == "socket":
            wide, tall = m.housing[1], m.mouth
        object.__setattr__(self, "shape", ShapeSpec(kind="cylinder", radius=wide, half_height=0.5 * tall))
        if int(self.sdf_resolution) % 8:
            raise ValueError(f"threaded part {self.name!r}: sdf_resolution must be a multiple of 8")

    @property
    def pitch(self) -> float:
        """What one turn raises or lowers this part [m]: its thread's pitch, or its lamp's."""
        return self.lamp.pitch if self.part in ("bulb", "socket") else self.thread.pitch

    @property
    def rim(self) -> float:
        """Height of the mating point over the part's origin [m].

        A bottle's rim, the face of a socket's centre contact: where
        :func:`screwed` puts the other part's origin when it is screwed home. A
        cap's seat and a bulb's eyelet are their origins, 0.
        """
        if self.part == "bottle":
            return self.body[1] + self.shoulder + self.stem[1] + self.thread.finish_length
        return self.lamp.housing[2] if self.part == "socket" else 0.0

    @property
    def grasp_height(self) -> float:
        """Height over the origin of the middle of the band a hand closes on [m]: the ribs, the glass at its widest, the body."""
        t = self.thread
        if self.part == "cap":
            return 0.5 * (t.top - t.cap_depth)
        if self.part == "bulb":
            return self.lamp.equator
        if self.part == "bottle":
            return 0.5 * self.body[1]
        return 0.5 * self.lamp.mouth


GLASS: Vec3 = (0.96, 0.95, 0.92)
"""A lamp's glass, dark."""
GLASS_FINISH = Finish(roughness=0.12, opacity=0.35, ior=1.5)
"""Soda-lime glass, lightly frosted: the path tracer draws it translucent. The studio's page draws a
shape that repaints itself flat, so there the bulb is an opaque pearl shell."""
LIT: Vec3 = (1.0, 0.92, 0.55)
"""The glass with the filament on: :meth:`EntityThreaded.light`."""
SHELL: Vec3 = (0.80, 0.80, 0.79)
"""The base's and the holder's screw shells: aluminium."""
BRASS: Vec3 = (0.80, 0.62, 0.28)
"""The eyelet and the holder's centre contact."""
VITRITE: Vec3 = (0.035, 0.035, 0.04)
"""The black glass that insulates the eyelet from the shell."""
PORCELAIN: Vec3 = (0.93, 0.92, 0.88)
"""The holder's body."""
LAMP_DENSITY = (127.0, 1240.0)
"""What the lamp's glass and its base are built at [kg/m^3], before the whole is scaled to the lamp's
mass: a bulb is a 0.6 mm wall, drawn here solid, so the 149 cm^3 it encloses carry 19 g and the 10.5 cm^3
of base 13 g, and the lamp's centre of mass is 52 mm up, under its widest part."""

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


def bulb(*, name: str = "bulb", at: Vec3 = (0.0, 0.0, 0.0), lamp: LampSpec | None = None, **kwargs) -> ThreadedSpec:
    """A household lamp, base down: an A19 bulb on a medium Edison screw. ``at`` is the face of its eyelet.

    60.3 mm of glass at its widest, 81 mm up, and 111 mm tall; the base under
    it is 26.3 mm over its thread's crests, a rounded thread of 3.63 mm pitch
    (:class:`LampSpec` has where each number comes from). 32 g, 19 of them
    glass, so it hangs base down from a hold on its widest part. The eyelet in
    the middle of its tip is what meets a socket's centre contact:
    ``sim["bulb"].entity.contact_shape``.

    It is drawn as glass, on a metal base with a black insulator, and
    :meth:`EntityThreaded.light` draws it lit or dark.
    """
    spec = {"mass": 0.032, "color": GLASS, "finish": GLASS_FINISH, **_DEFAULTS, **kwargs}
    return ThreadedSpec(name=name, part="bulb", lamp=lamp or LampSpec(), pose=at, **spec)


def socket(*, name: str = "socket", at: Vec3 = (0.0, 0.0, 0.0), lamp: LampSpec | None = None, **kwargs) -> ThreadedSpec:
    """A lamp holder, mouth up: a threaded metal shell in a porcelain body, a centre contact in its floor. ``at`` is its underside.

    60 mm over its flange and 34 mm tall; the mouth is 21.3 mm above the centre
    contact, so a lamp's thread first meets the shell's 11.8 mm, three and a
    quarter turns, above home. Fixed where it stands unless ``static=False``:
    a lamp holder is screwed to something.
    """
    spec = {"static": True, "color": PORCELAIN, **_DEFAULTS, **kwargs}
    return ThreadedSpec(name=name, part="socket", lamp=lamp or LampSpec(), pose=at, **spec)


def screwed(onto: ThreadedSpec, *, turns_out: float = 0.0, lift: float = 1.0e-4) -> dict[str, tuple[float, ...]]:
    """Where the mating part goes on ``onto``, as the ``at`` and ``quat`` to build it with.

    A cap on a bottle, a bulb in a socket: ``rss.cap(**screwed(flask))``.
    ``turns_out`` backs it off that many turns, raised a pitch for each; past
    the thread's ``turns`` (a lamp's :attr:`LampSpec.catch`, 3.25 turns) it
    stands on the thread's end, not held. ``lift`` starts it that far clear [m].

    Raises:
        ValueError: If ``onto`` is a cap or a bulb, or is not upright.
    """
    if onto.part not in ("bottle", "socket"):
        raise ValueError(f"{onto.name!r} is a {onto.part}: a cap goes on a bottle and a bulb in a socket")
    if tuple(float(v) for v in onto.quat) != (0.0, 0.0, 0.0, 1.0):
        raise ValueError(f"{onto.name!r} is turned: screwed() places a part on one standing upright")
    x, y, z = (float(v) for v in onto.pose[:3])
    half = math.pi * float(turns_out)
    height = z + onto.rim + onto.pitch * float(turns_out) + float(lift)
    return {"at": (x, y, height), "quat": (0.0, 0.0, math.sin(half), math.cos(half))}


class EntityThreaded(EntityRigid):
    """A :class:`ThreadedSpec`: one body, its thread a mesh with a cooked signed-distance field."""

    spec: ThreadedSpec
    thread_shape: int = -1
    """The shape that carries the thread, once built."""
    contact_shape: int = -1
    """A bulb's eyelet or a socket's centre contact, once built; -1 on a bottle and a cap."""
    glass_shape: int = -1
    """A bulb's glass, once built; -1 on every other part."""
    lit: bool = False
    """Whether a bulb is drawn lit: :meth:`light`."""

    def _add_primitive(self, builder, pose) -> None:
        import warp as wp

        from robosimstudio.scene.build.bodies import set_body_mass

        spec, t, m = self.spec, self.spec.thread, self.spec.lamp
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
        upright = wp.quat_identity()
        self._look = {}

        def cylinder(radius: float, low: float, high: float, label: str, look=None, shape_cfg=None) -> int:
            at = wp.transform((0.0, 0.0, 0.5 * (low + high)), upright)
            shape = builder.add_shape_cylinder(
                body,
                xform=at,
                radius=radius,
                half_height=0.5 * (high - low),
                cfg=shape_cfg or cfg,
                label=f"{spec.name}_{label}",
            )
            self._look[shape] = look
            return shape

        def turned(mesh, label: str, *, at: float = 0.0, look=None, resolution: int = 0, shape_cfg=None) -> int:
            cooked = self._cooked(mesh, resolution or int(spec.sdf_resolution))
            where = wp.transform((0.0, 0.0, at), upright)
            shape = builder.add_shape_mesh(
                body, xform=where, mesh=cooked, cfg=shape_cfg or cfg, label=f"{spec.name}_{label}"
            )
            self._look[shape] = look
            return shape

        metal = Finish(roughness=0.3, metallic=1.0)
        if spec.part == "cap":
            self.thread_shape = turned(cap_mesh(t), "thread")
        elif spec.part == "socket":
            meshes = holder_meshes(m)
            self.thread_shape = turned(meshes["shell"], "thread", look=(SHELL, metal))
            turned(meshes["housing"], "housing", resolution=128)
            radius, height = m.pad
            self.contact_shape = cylinder(radius, spec.rim - height, spec.rim, "pad", look=(BRASS, metal))
        elif spec.part == "bulb":
            meshes = lamp_meshes(m)
            light, heavy = cfg.copy(), cfg.copy()
            light.density, heavy.density = LAMP_DENSITY
            self.thread_shape = turned(meshes["base"], "thread", look=(SHELL, metal), shape_cfg=heavy)
            turned(meshes["insulator"], "insulator", look=(VITRITE, GLOSS), resolution=64, shape_cfg=heavy)
            self.contact_shape = cylinder(
                0.5 * m.eyelet_diameter, 0.0, 1.0e-3, "eyelet", look=(BRASS, metal), shape_cfg=heavy
            )
            self.glass_shape = turned(meshes["glass"], "glass", resolution=128, shape_cfg=light)
        else:
            (radius, height), (stem, stem_height) = spec.body, spec.stem
            cylinder(radius, 0.0, height, "body")
            for k in range(4):  # the shoulder, as four steps
                r = radius + (stem - radius) * (k + 0.5) / 4.0
                cylinder(r, height + spec.shoulder * k / 4.0, height + spec.shoulder * (k + 1) / 4.0, f"shoulder_{k}")
            cylinder(stem, height + spec.shoulder, height + spec.shoulder + stem_height, "stem")
            self.thread_shape = turned(neck_mesh(t), "thread", at=spec.rim)
        if spec.mass is not None and not spec.static:
            set_body_mass(builder, body, float(spec.mass))

    # ------------------------------------------------------------------ looks
    def apply_display(self, builder) -> None:
        """The spec's colour and finish on every shape, and a lamp's metal, brass and insulator on theirs."""
        super().apply_display(builder)
        for shape, look in getattr(self, "_look", {}).items():
            if look is not None:
                builder.shape_color[shape], self.finishes[shape] = look

    @property
    def recolors_live(self) -> bool:
        """A bulb repaints its glass when it is lit; no other part repaints."""
        return self.spec.part == "bulb"

    def light(self, on: bool) -> None:
        """Draw a bulb lit or dark. Display only: the solver never reads it.

        A lamp is lit when its eyelet presses on the holder's centre contact,
        and that force is the scene's to read (``sim.contacts``); this is what
        the reader calls with the answer. It writes the glass's row of
        ``model.shape_color``, which every viewer reads again for a shape of an
        entity that :attr:`recolors_live`.

        Raises:
            ValueError: If this part is not a bulb.
        """
        if self.spec.part != "bulb":
            raise ValueError(f"threaded part {self.spec.name!r} is a {self.spec.part}: only a bulb lights")
        if bool(on) == self.lit or self.model is None:
            return
        colors = self.model.shape_color.numpy()
        colors[self.glass_shape][:3] = LIT if on else tuple(float(v) for v in self.spec.color or GLASS)
        self.model.shape_color.assign(colors)
        self.lit = bool(on)

    def reset(self, pipeline, episode) -> None:
        super().reset(pipeline, episode)
        if self.spec.part == "bulb":
            self.light(False)

    def _cooked(self, mesh: tuple[np.ndarray, np.ndarray], resolution: int):
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
            max_resolution=int(resolution),
            narrow_band_range=(-SDF_BAND, SDF_BAND),
            margin=SDF_BAND,
            cache_dir=directory,
        )
        return cooked


register_entity("threaded")(EntityThreaded)


# =============================================================================
# 4. The physics that carries a thread
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
