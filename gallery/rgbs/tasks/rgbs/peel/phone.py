# SPDX-License-Identifier: Apache-2.0
"""The phone the ``panel`` variants peel a film off, and the cradle it leans in.

Measured off ``sample_videos/film peeling.mp4`` (2026-09-10): a phone -- black
glass face, pale silver frame, 147 x 72 x 8 mm -- leans back in a small white
3D-printed cradle at the bench's centreline, its bottom edge toward the arms and
low, its top edge away and raised, the face carrying a crinkled protective film.
The left YAM pinches the phone's bottom-left corner across its thickness and
holds it; the right YAM pinches the film at the top edge and folds it back down
the face and off, then drops it on the mat to the right.

**Where it stands is a reach measurement first and a transcription second.**
The clip's inset puts the phone about four grid rows (100 mm) in front of the
arm-base brackets' tops, which with the inset camera's parallax on a bracket
150 mm tall is a phone centre near surface x = -0.18. The tab pinch, the whole
peel arc and the carry to the mat solve to 0.0 mm for the right arm with the
bottom edge anywhere from x = -0.20 to -0.06 (``rgbs/scripts/rgbs_peel_bench.py phone``),
so the phone sits at :data:`PHONE_FOOT` -- bottom edge at x = -0.160, centre
at -0.096 -- a little further from the arms than the clip, where the arc's far
end (back over the phone at x = -0.29) still clears the right arm's own base.

**The cradle is narrower than the clip's, and the pad geometry decides it.** A
corner pinch across an 8 mm edge puts one YAM pad *under* the phone, and that
pad's tip band is 38 mm wide, 9 mm thick and reaches 10.5 mm past the tool
centre (measured on the decomposed colliders in the grasp-site frame, all three
numbers). Under the phone's corner it occupies |y| >= 31.5 mm, so everything the
phone rests on stays inside |y| <= 21 mm: a 42 mm-wide stand where the clip's
is about 58. Same reason the phone's bottom edge sits 20 mm off the mat rather
than on it -- the yoke behind the pad tip is 15 mm thick and would otherwise be
in the table -- and why the film is 60 mm wide rather than the 66 a 3 mm rim
would give: the pad tip is 1.5 mm outboard of the film's edge column, so no
film particle is ever pressed under the holding hand's pad.

**Five boxes, three of them colliding, and the count is the arrangement's.**
``cloth_vbd@ref`` gives MuJoCo 2048 contact slots, and mujoco-warp fills a slot
for every hull piece whose bounding box overlaps a box's -- measured with the
left hand closed on the corner: 622 stored pairs at a median separation of
69 mm, 91 fingertip pieces against each of six boxes, seven of them within
5 mm. Two hands over six boxes ran past 2048 as the right hand arrived and the
process died on an illegal memory access. So the glass slab and the angled rest
are drawn and not collided with (the film hangs on its bonds 1.2 mm above the
glass and never touches it; nothing reaches the rest), and the stand is one
block rather than a lip and a block: three colliding boxes near the hands --
the frame, the block, the table -- and about 360 slots per hand.

**Leaning back at 30 degrees, and welded there.** The clip's phone ends up
lying flat once the film is off (the holding hand lets it settle); a welded
phone cannot, and leaning is the pose that makes the tab reachable -- the
film's top edge stands 94 mm up, the right arm takes it with its tool 10
degrees below +y in the face plane, closing along the face normal, which is the
same band the flat ``handle`` layout's 35-degree pinch lives in once the face's
own 30 degrees are taken off.

Everything below is in the cell's work-surface frame unless it says world.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import newton
import numpy as np
import warp as wp

from robosimstudio.core.render.finish import MATTE, Finish
from robosimstudio.domains.static.entity import EntityStatic
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs import StaticSpec

from ..look import PANEL_BLACK, PANEL_GLOSS

__all__ = [
    "CRADLE_HALF_WIDTH",
    "PHONE_BLACK",
    "PHONE_FOOT",
    "PHONE_SILVER",
    "PHONE_SIZE",
    "PHONE_TILT_DEG",
    "PhoneFrame",
    "SpecPhone",
    "SpecPhoneCradle",
    "phone_frame",
    "tilt_quat",
]

PHONE_SIZE = (0.147, 0.072, 0.008)
"""Length, width, thickness [m]. A 6.1-inch phone: 147 x 72 x 8 mm."""

PHONE_TILT_DEG = 30.0
"""How far back from horizontal the phone leans [deg], about world y.

The clip's phone reads 30-40 degrees before the peel and flatter after; 30 is
the shallow end, chosen because the peel arc drags the film back over the
phone toward the arms and every degree of lean puts the tab higher and the
arc's far end further back over the right arm's own base.
"""

PHONE_FOOT = (-0.160, 0.0, 0.025)
"""The phone's bottom-back edge, at its midpoint [m]: on the cradle block's top.

x is the reach compromise the module docstring records; y = 0 is the clip's
centreline; z = 25 mm is the block's height, and it is the holding hand's, not
the clip's -- the pad that goes under the phone's corner needs 20 mm of air
below the back face for the yoke behind its tip, and the clip's phone sits on
the mat.
"""

FACE_RIM = 0.0015
"""The glass stops this far short of the frame's edge on every side [m]: the
bezel. 1.5 mm rather than 2 so the film's end rows sit 2 mm inside the glass
box's footprint at both ends (layout.py's ``FILM_RIM_TOP``), where float
rounding cannot decide whether they bond."""

FACE_PROUD = 0.0005
"""Half the glass slab's thickness [m]: the face stands 0.5 mm proud of the
frame, which is what draws a bezel under the path tracer."""

CRADLE_HALF_WIDTH = 0.021
"""Half the cradle's width across the bench [m]. See the module docstring: the
pad under the phone's corner starts 31.5 mm out, and this keeps 10 mm of air."""

CRADLE_BLOCK_HALF = (0.035, CRADLE_HALF_WIDTH, 0.0125)
"""The block the phone stands on [m]: 70 deep, 42 wide, 25 tall, from 20 mm
in front of the bottom edge to 50 behind it -- the clip's block less the width
the pinch takes off it, and the one colliding piece of the stand."""

CRADLE_BLOCK_AHEAD = 0.020
"""How far the block's front face stands in front of the phone's bottom edge [m]."""

CRADLE_REST_HALF = (0.035, CRADLE_HALF_WIDTH, 0.006)
"""The angled back rest [m]: a 70 x 42 x 12 mm plate under the phone's back,
from 23 to 93 mm up the back, leaning with the phone. Drawn only."""

CRADLE_REST_UP = 0.058
"""Where the back rest's centre sits along the phone's back, from the bottom
edge [m]. Its lower corner lands 1 mm above the block's top."""

PHONE_SILVER = (0.70, 0.71, 0.70)
"""The frame: sampled light grey off the clip's bezel, satin rather than mirror."""

PHONE_BLACK = PANEL_BLACK
"""The glass, under the film: the same glossy black the old panel block was."""

CRADLE_WHITE = (0.86, 0.86, 0.84)
"""White PLA, a touch warm, matte."""

PHONE_SATIN = Finish(roughness=0.45, metallic=0.6)
"""Brushed aluminium, but not a mirror: the clip's frame catches the light
without reflecting the room."""

CRADLE_MATTE = MATTE


def tilt_quat(tilt_deg: float = PHONE_TILT_DEG) -> tuple[float, float, float, float]:
    """The phone's orientation, xyzw: a rotation about world y that puts local
    +x up the face (bottom edge to top edge, rising toward +x) and local +z
    along the face normal, toward the arms and up.

    About -y, because a positive rotation about y would drop local +x below
    the bench. Identity when the phone lies flat.
    """
    half = math.radians(float(tilt_deg)) / 2.0
    return (0.0, -math.sin(half), 0.0, math.cos(half))


@dataclass(frozen=True)
class PhoneFrame:
    """The phone's own axes and the points a layout or a policy aims at.

    Attributes:
        u: Unit vector up the face, from the bottom edge to the top edge.
        n: The face's outward normal -- toward the arms and up.
        foot: The bottom-back edge's midpoint [m], :data:`PHONE_FOOT`.
        quat: The phone's orientation, xyzw.
    """

    u: np.ndarray
    n: np.ndarray
    foot: np.ndarray
    quat: tuple[float, float, float, float]

    @property
    def length(self) -> float:
        return float(PHONE_SIZE[0])

    @property
    def width(self) -> float:
        return float(PHONE_SIZE[1])

    @property
    def thickness(self) -> float:
        return float(PHONE_SIZE[2])

    @property
    def centre(self) -> np.ndarray:
        """The phone body's centre [m]."""
        return self.foot + self.u * (self.length / 2.0) + self.n * (self.thickness / 2.0)

    @property
    def glass_centre(self) -> np.ndarray:
        """Centre of the glass slab [m], on the frame's face plane."""
        return self.centre + self.n * (self.thickness / 2.0)

    @property
    def glass_half(self) -> tuple[float, float, float]:
        """The glass slab's half-extents [m] in the phone's frame: the box the
        film bonds to. Its +z face is the bond plane."""
        return (self.length / 2.0 - FACE_RIM, self.width / 2.0 - FACE_RIM, FACE_PROUD)

    @property
    def face_centre(self) -> np.ndarray:
        """Centre of the glass's outer face [m]: what the rig looks at."""
        return self.glass_centre + self.n * FACE_PROUD

    def on_face(self, along: float, across: float, above: float = 0.0) -> np.ndarray:
        """A point ``along`` metres up the face from the bottom edge, ``across``
        metres toward +y, ``above`` metres off the glass [m]."""
        return (
            self.foot
            + self.u * float(along)
            + np.array([0.0, float(across), 0.0])
            + self.n * (self.thickness + FACE_PROUD + float(above))
        )

    def in_body(self, along: float, across: float, depth: float = 0.0) -> np.ndarray:
        """A point ``along`` metres up the phone from its bottom edge, ``across``
        toward +y, at ``depth`` into the phone from its mid-plane (negative is
        toward the back) [m]."""
        return (
            self.foot
            + self.u * float(along)
            + np.array([0.0, float(across), 0.0])
            + self.n * (self.thickness / 2.0 + float(depth))
        )

    def along(self, point) -> float:
        """How far up the face a point is, from the bottom edge [m]."""
        return float(np.dot(np.asarray(point, dtype=np.float64) - self.foot, self.u))


def phone_frame(foot=PHONE_FOOT, tilt_deg: float = PHONE_TILT_DEG) -> PhoneFrame:
    """The phone's frame as placed, in the work-surface frame."""
    theta = math.radians(float(tilt_deg))
    return PhoneFrame(
        u=np.array([math.cos(theta), 0.0, math.sin(theta)]),
        n=np.array([-math.sin(theta), 0.0, math.cos(theta)]),
        foot=np.asarray(foot, dtype=np.float64),
        quat=tilt_quat(tilt_deg),
    )


# ---------------------------------------------------------------------------
# The specs and the entities that build them: two shapes for the phone, three
# for the cradle, all on the world body -- welded scenery with collision.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class SpecPhone(StaticSpec):
    """A phone, welded where it leans. ``xform`` is its body centre and tilt."""

    entity: str = "phone"
    name: str = "phone"
    foot: tuple[float, float, float] = PHONE_FOOT
    tilt_deg: float = PHONE_TILT_DEG
    color: tuple[float, float, float] | None = PHONE_SILVER
    finish: Finish | None = PHONE_SATIN


@dataclass(frozen=True, kw_only=True)
class SpecPhoneCradle(StaticSpec):
    """The stand: a block and an angled rest, welded under the phone.
    ``xform`` is the block's centre, the landmark a framing check reads."""

    entity: str = "phone_cradle"
    name: str = "cradle"
    foot: tuple[float, float, float] = PHONE_FOOT
    tilt_deg: float = PHONE_TILT_DEG
    color: tuple[float, float, float] | None = CRADLE_WHITE
    finish: Finish | None = CRADLE_MATTE


def _xform(entity: EntityStatic, centre, quat) -> wp.transform:
    placed = entity.placed((*[float(v) for v in centre], *[float(v) for v in quat]))
    return wp.transform(tuple(float(v) for v in placed[:3]), wp.quat(*[float(v) for v in placed[3:7]]))


_DRAWN_ONLY = ~(int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES))
"""Flag mask for a shape the cameras see and nothing touches."""


@register_entity("phone")
class EntityPhone(EntityStatic):
    """The frame box and the glass slab on its face, both on the world body."""

    spec: SpecPhone

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec = self.spec
        frame = phone_frame(spec.foot, spec.tilt_deg)
        quat = frame.quat
        self._glass = builder.add_shape_box(
            body=-1,
            xform=_xform(self, frame.glass_centre, quat),
            hx=frame.glass_half[0],
            hy=frame.glass_half[1],
            hz=frame.glass_half[2],
            label=f"{spec.name}_glass",
        )
        # Drawn, not collided with: the film's bonds hold it 1.2 mm above the
        # glass (measured at reset) and the hands never touch the face, so the
        # slab's only physical effect was a second box's worth of contact
        # slots under every fingertip piece near it.
        builder.shape_flags[self._glass] &= _DRAWN_ONLY
        builder.add_shape_box(
            body=-1,
            xform=_xform(self, frame.centre, quat),
            hx=frame.length / 2.0,
            hy=frame.width / 2.0,
            hz=frame.thickness / 2.0,
            label=f"{spec.name}_frame",
        )
        if spec.friction is not None:
            for s in range(self.range.shapes[0], builder.shape_count):
                builder.shape_material_mu[s] = float(spec.friction)

    def apply_display(self, builder: newton.ModelBuilder) -> None:
        # The spec's colour and finish are the frame's; the glass is its own.
        super().apply_display(builder)
        builder.shape_color[self._glass] = tuple(float(v) for v in PHONE_BLACK)
        self.finishes[self._glass] = PANEL_GLOSS


@register_entity("phone_cradle")
class EntityPhoneCradle(EntityStatic):
    """A block the phone stands on and, drawn only, the angled rest it leans against."""

    spec: SpecPhoneCradle

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec = self.spec
        frame = phone_frame(spec.foot, spec.tilt_deg)
        parts = (
            ("block", cradle_block_centre(spec.foot), (0.0, 0.0, 0.0, 1.0), CRADLE_BLOCK_HALF, True),
            (
                "rest",
                frame.foot + frame.u * CRADLE_REST_UP - frame.n * CRADLE_REST_HALF[2],
                frame.quat,
                CRADLE_REST_HALF,
                False,
            ),
        )
        for part, centre, quat, half, collide in parts:
            shape = builder.add_shape_box(
                body=-1,
                xform=_xform(self, centre, quat),
                hx=float(half[0]),
                hy=float(half[1]),
                hz=float(half[2]),
                label=f"{spec.name}_{part}",
            )
            if not collide:
                builder.shape_flags[shape] &= _DRAWN_ONLY
        if spec.friction is not None:
            for s in range(self.range.shapes[0], builder.shape_count):
                builder.shape_material_mu[s] = float(spec.friction)


def cradle_block_centre(foot=PHONE_FOOT) -> tuple[float, float, float]:
    """The cradle block's centre [m]: the fixture's landmark. Its top is the
    foot's height, so the phone stands on it."""
    fx = float(foot[0])
    return (fx - CRADLE_BLOCK_AHEAD + CRADLE_BLOCK_HALF[0], 0.0, CRADLE_BLOCK_HALF[2])
