# SPDX-License-Identifier: Apache-2.0
"""The cable box: a free plastic cable-management box with a lidded channel along its front-top edge.

What ``channel routing.mp4`` actually shows on the bench, sized off the clip
rather than authored: a cream box the length of the working span, a black
dimpled anti-slip mat over its top, a cream lid strip standing proud along the
top-front edge with a dark slot behind it, a white power strip lying on the mat
toward the far edge, and a notch in the front face. The physics stays the
conduit's -- a closed 17 mm square bore a cable is fed through end-first -- but
the bore is now *part of a body*, and the body is free. That is the one thing
the reference clip was said to differ on ("the channel is not a fixture, so the
left arm needs to hold it"), and it is why this is an :class:`ObjectSpec`
subclass built by an :class:`~robosimstudio.domains.rigid.entity.EntityRigid`
subclass, the pattern ``decant/carton.py`` set.

**Boxes, not a mesh, still.** MuJoCo collides a mesh geom as its convex hull and
the hull of a box with a channel in it is a solid block -- the bore is exactly
the feature a hull erases. Every shape below is an ``add_shape_box`` on one
free body, so the bore is real to the solver. Overlapping boxes on one body cost
nothing (a body does not collide with itself), which is what lets the body be
stated as a few solid blocks rather than a hollow shell nobody can see into.

**The bore is 100 mm up now, on the box's top, and that changes the feed.** The
conduit's floor was the bench, so a straight rod fed in without a lift. Here
the bore's floor is the box's own top -- the shelf the channel stands on, the
ledge the plug emerges onto and the rubber mat are one flat surface -- the
cable drapes off the entry end down to the bench mat, and the hand works on
that hang rather than on a table. What it costs the routine is measured on
``routine_tunnel.py``; what it costs the cable's spawn is
``layout.py::cable_nodes``, which seeds the hang rather than a straight line.

**The lid strip is shorter than the box, as the clip's is.** The bore runs
:data:`RUN` from the right (-y) end face; beyond its exit the top-front strip is
an open ledge at the bore floor's level, and the connector emerges onto it where
the second hand can pinch it. The rod is what sets :data:`RUN` (see
``layout.py::slack``), not the box.

Frame: the body's origin is on the **bore axis, at the bore floor, at the centre
of the run** -- exactly where the conduit's was -- so every region in
``thread.task`` is measured from the same place it always was, and the box's
body extends from that origin: front face at ``-(BORE/2 + WALL)``, right end
face at ``-RUN/2``, mat top level with the floor. A region targeting this object is
offset by this origin with no rotation applied
(:class:`~robosimstudio.scene.handles.RegionHandle`), which is why the box
stands square to the bench with its run along +y.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import newton
import numpy as np
import warp as wp

from robosimstudio.core.render.finish import GLOSS, SATIN, Finish, register_texture
from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.scene.build.bodies import set_body_mass
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.asset import ObjectSpec, ShapeSpec

from ..look import BLACK

__all__ = [
    "BORE",
    "BOX_CREAM",
    "BOX_DEPTH",
    "BOX_GAP",
    "BOX_HEIGHT",
    "BOX_LENGTH",
    "CUTAWAY",
    "DECALS",
    "FLARE",
    "FLARE_ANGLE",
    "LID_PROUD",
    "LINER",
    "MAT_TOP",
    "PARTS",
    "ROOF",
    "RUN",
    "SKINS",
    "SLOT",
    "WALL",
    "EntityCableBox",
    "SpecCableBox",
    "box_frame",
    "flare_box",
    "flare_floor_box",
    "flare_inner_face",
    "flare_roof_box",
    "mouth_width",
    "shape_labels",
]

# ---------------------------------------------------------------- the bore

BORE = 0.017
"""The square bore, side [m].

**Sized to the connector, not to the cable.** What has to clear the run is
:data:`~tasks.rgbs.assets.tunnel.plug.PLUG_W`, a 15 mm overmold, so 17 mm leaves
1 mm a side on the head and 3.5 mm on the cable behind it. 17 mm is also what
keeps the head's *roll* honest: a 15 x 10 mm box has an 18 mm diagonal, so it
cannot turn over inside the bore, only tilt. Open this past 18 mm and the
connector is free to rotate and jam broadside; close it under about 16 mm and
the head does not go through at all. Unchanged from the conduit.
"""

RUN = 0.250
"""Entry face to exit face [m]: the lidded part of the channel.

**Set by the rod, not by the box.** ``layout.py::slack`` needs
``LENGTH >= RUN + EXIT_CLEAR + MIN_TAIL`` -- a route longer than its rod scores
zero for ever -- and the rod's length is itself capped by the bench: it spawns
straight behind the entry and its tail has to land on the table. 0.44 m of rod
at the entry this layout puts at y -0.21 lands its tail at about y -0.60 against
a table edge at -0.65, and 0.44 - 0.08 - 0.10 leaves 0.26 for the run. 0.25 is
that with 10 mm of slack. The clip's lid strip covers about two thirds of its
box; this covers 60 % of :data:`BOX_LENGTH`, which is as close as the rod gets.
"""

WALL = 0.008
LINER = 0.002
"""How much of each bore wall is the LINER the cable rubs [m]; the rest is the
skin the hand grips.

**One label cannot serve both hands, and this is the split that lets it.**
``shape_labels`` names a shape as a cable surface and the rod's material pass
then writes ``mu_surface`` to the SHAPE -- so a wall named for the cable is
0.0011 to everything, and the support hand's blade at ``mu_gripper`` 60 pinches
it at ``sqrt(60 * 0.0011)`` = 0.257. Measured, that is not a hold: with both
pads closed on the lid, the support fingers went from 24.6/24.8 mm to
28.5/18.8 mm in six steps as the feed loaded the cable -- ten millimetres of box
sliding BETWEEN stationary jaws -- and the box ended a stroke 17 mm out of
place with 3 degrees of yaw.

So each wall is two boxes: a 2 mm liner on the bore side, named, which is what
the cable slides on, and a 6 mm skin outside it, unnamed, which is what the
hand grips at ``sqrt(60 * 2.0)`` = 10.9. That is what a moulding is anyway --
the channel's inside and the box's outside are different surfaces -- and it is
the difference between a steadying hand and a decoration.

**What the skin costs, stated because it is the same trap facing the other
way:** a cable that touched a skin would meet it at mu 10.9 and stop dead. The
skins are the outer faces at ``+-(BORE/2 + WALL)``, 16.5 mm off the bore axis;
the cable lives on the axis inside the bore and hangs off the entry end outside
it, and the furthest the routine ever takes it off the axis is the 10 mm
sideways step it uses to free a stuck jaw. The margin is 6.5 mm."""

ROOF = 0.006
"""Bore wall and lid thickness [m]. The lid is the bore's roof: with the mat top
at :data:`MAT_TOP` it stands :data:`LID_PROUD` proud, which is the clip's
strip."""

FLARE = 0.015
FLARE_ANGLE = math.radians(20.0)
"""The lead-in: 15 mm of wall yawed 20 degrees out of the run, each side, plus a
floor and a roof panel tilted the same 20 degrees.

The conduit's flare was two yawed walls and **no roof**, deliberately: its cable
lay on the bench and was pushed horizontally, so the only way out of the mouth
was upward and that was the difficulty the task was made of. Neither half of
that survives the box. The cable now arrives from BELOW -- it hangs off the
entry end and the hand pushes it up a 50-degree line -- so an open-topped funnel
is not a difficulty, it is an exit: measured, the cable rode up over the lip and
landed across the lid, 22 mm off a bore axis with 3.5 mm of clearance, and
levered the box 16 mm sideways when the hand let go.

So the mouth is a funnel on all four sides: walls that gather in x, a floor that
lifts a cable arriving low, and a roof that stops one arriving steep from riding
over. That is also what a cable box's channel entry is."""

BOX_GAP = 0.004
"""Contact detection gap for the box's shapes [m]. Twice the bore's per-side
clearance and above the cell's 2 mm ``rigid_gap``: a capsule crossing 2 mm of
clearance inside one substep is still paired. A *detection* margin, not a
standoff -- ``margin`` stays 0."""

# ----------------------------------------------------------------- the box

BOX_LENGTH = 0.400
"""The box along the bench (y) [m].

Measured off ``channel routing.mp4`` two ways and they disagree. Counting the
mat's 25 mm grid along the box's front-bottom edge in frame 0 gives 18-20
squares (450-500 mm); projecting the same edge through the suite's ``photo``
camera gives 390, and in the rendered ``photo`` still a 420 mm box ran 65 mm
past the clip's left end while its right end landed where the clip's does. The
still is what the suite is matched against, so 400: its length-to-height is
the clip's 4.0, its centre lands 5 mm right of the bench's centreline as the
clip's inset has it, and the rod's tail keeps 45 mm of table (see :data:`RUN`
and ``layout.py::TUNNEL_Y``)."""

BOX_DEPTH = 0.200
"""Front face to far face (x) [m]. The clip's overhead inset shows a top at
least 200 mm deep (the power strip has mat on both sides of it); the phone's
foreshortening reads it as half that. Nothing reaches the far half."""

BOX_HEIGHT = 0.100
"""Bench to the mat top [m]. The box's front face is 235 px tall in frame 0
against 55 px grid squares, which is 4.3 squares at the phone's 25-degree look-
down: 100-115 mm. 100, with the length, keeps the clip's proportions in the
``photo`` still. The lid stands another :data:`LID_PROUD` above this."""

MAT_TOP = 0.0
"""The mat top's height above the bore floor [m]: level with it.

The bore floor IS the box's top: the shelf the channel stands on, the ledge the
plug emerges onto and the mat are one flat surface, and the lid strip stands the
whole bore plus its own thickness proud of it. It was 13 mm above the floor
(a 10 mm lid) and the ledge past the lid rendered as a recessed notch in the
box's top-front corner, which the clip's box does not have -- its mat runs to
the front edge beside the strip. Frame 0 puts the strip's front face at about
a third of a grid square plus its foreshortened top: 15-25 mm, which 23 is."""

LID_PROUD = BORE + ROOF - MAT_TOP
"""How far the lid strip stands above the mat top [m]: 23 mm, derived."""

MAT_THICK = 0.006
"""The rubber mat slab [m]."""

SLOT = 0.024
"""The slot behind the lid strip [m], between the bore's back wall and the mat.

The clip's slot is a dark line a few millimetres wide; this one is sized for a
fingertip, because that is what it is for. The support hand pinches the lid
across x with one pad in front of the box's front face and the other down this
slot, and a YAM blade is 7 mm thick along its closing direction at the pad
(measured on the built model, ``rgbs/scripts/rgbs_tunnel_bench.py pads``). The jaws come
down 9 mm wider than the lid a side (``routine_tunnel.py::LID_CLEAR``) and
the arm settles up to 3 mm off its commanded x at this reach, so the back pad
lands between 22 and 35 mm behind the bore axis; the slot runs from 16.5 to
40.5. It was 20 mm with a 4 mm clearance, and the 3 mm of settle put the front
pad on the lid's edge: the jaws closed on air above the strip and the box was
pressed rather than held."""

SLOT_DEPTH = 0.005
"""How far below the bore floor the slot goes [m]. The support hand's pads reach
``_pad_drop`` (10.7 mm) under its grasp site and the site sits half a lid above
the floor, so the tips come to rest about 1 mm above the floor level; the slot
floor sits 5 mm under that so a settled hand never bottoms out on it."""

SHELF = 0.005
"""The strip of box top the bore stands on [m], from the slot floor up to the
bore floor. The shelf runs the whole length: under the bore it is the bore's
floor, beyond the exit it is the open ledge the connector emerges onto."""

BASE_INSET, BASE_THICK = 0.005, 0.004
"""The foot the box stands on: a slab inset from every face, so the one shape
whose friction is against the *table* is not one the cable can reach. See
:func:`shape_labels`."""

STRIP_SIZE = (0.270, 0.050, 0.028)
"""The power strip [m], L x W x H: six outlets plus USB, lying along the box on
the mat, centred on the mat's depth -- the clip's inset shows about as much mat
beyond it as between it and the lid. Its length is 64 % of the box's there."""

NOTCH = (0.060, 0.022)
"""The cable exit notch in the front face [m], width x height, drawn as a
decal at the bottom of the face just left of the run's centre."""

SKINS = ("skin_front", "skin_back")
"""The bore walls' outer halves: built, collided, and deliberately NOT named as
cable surfaces, so the support hand can hold the box. See :data:`LINER`."""

PARTS = (
    "wall_front",
    "wall_back",
    "roof",
    "flare_left",
    "flare_right",
    "flare_floor",
    "flare_roof",
    "shelf",
    "front_block",
    "deck",
    "mat",
    "strip",
)
"""Every **colliding** shape the entity builds that a cable can touch, in build
order -- the friction surfaces. The rod restates the whole scene's friction at
bind and finds a shape by its label's tail, so these strings are the same ones
the cable's ``surface_labels`` carries; see :func:`shape_labels`.

``base`` and the two :data:`SKINS` are built and collide and are deliberately
not here: the foot's friction is against the table and the skins' is against the
support hand's pads, and naming either as a cable surface would put the box on
the rod's ice -- which for the skins is measured (see :data:`LINER`)."""

DECALS = ("notch", "outlet", "usb", "cord")
"""Display-only shapes: no collision, no mass. Named so a reader of the model's
labels knows what not to count."""

CUTAWAY: tuple[str, ...] = ()
"""Which of :data:`PARTS` to build but not *draw*, for looking inside the bore.
Shipped empty. Set it to ``("roof",)`` while debugging what the cable does in
there; nothing about the physics changes, which is the point of the flag."""

BOX_CREAM = (0.92, 0.82, 0.68)
"""The box's cream: the clip's own hue, at the brightest albedo that is still a
painted surface.

**The ratio is what was wrong, not the brightness.** This shipped at
(0.78, 0.63, 0.44) -- ``look.DUCT_CREAM`` divided by an estimated lighting
factor -- and rendered the front face at sRGB 136/120/95, a channel ratio of
1 : 0.88 : 0.70 against the clip's 1 : 0.93 : 0.80. A warm tan, where the clip
shows a cream box. Measured over a 360 x 55 px patch of the face in the
``photo`` view against the same patch of ``channel routing.mp4`` frame 0.

**And the brightness cannot be fixed here, which is worth writing down so the
next reader does not try.** Solving albedo = target / lighting from that same
pair gives (2.23, 1.98, 1.65): the front face receives about 30 % of the light
the clip's does, so no albedo reaches 218/202/175 -- a *pure white* face
renders at 148. The number below is the clip's hue scaled to a 0.92 peak,
which renders the face at 148/139/124: the right colour, two thirds of the
brightness.

**The shortfall is the lighting and it is vertical-only.** The mat, which is
horizontal, renders at 182/180/175 against the clip's 196/203/207 -- 93 % --
while this face reaches 68 %, because ``lab`` is a panel overhead and a dome a
vertical surface sees half of, and the lid overhangs this face besides. Raising
the preset's ambient 0.30 -> 0.60 takes the face to 174/165/152 and the mat to
192/190/186, both closer; it is not done here because it relights ten variants
across two suites to fix one box, and because the clip is a phone's
auto-exposed frame, so its absolute levels are partly an ISP's opinion. The
*ratio* between the box and the mat beside it is not, and that is what this
constant is fitted to."""

MAT_BLACK = (0.068, 0.072, 0.078)
"""The rubber mat, for the backends that draw one colour: the texture's mean,
sRGB ~73/76/80 as frame 0 averages it across dimples and valleys."""

STRIP_WHITE = (1.0, 0.98, 0.95)
"""A white power strip. Frame 0 renders its top at sRGB 236/229/218; at
(0.91, 0.91, 0.90) the lab preset reached 214/212/208. Kept off the palette's
``DEFORMABLE`` (0.88) on purpose: ``tests/test_palette.py`` refuses a literal
within 2e-3 of an entry."""

OUTLET_GREY = (0.22, 0.22, 0.24)
"""The outlet faces on the strip."""

CORD_RADIUS = 0.0035
"""The strip's own cord, a 7 mm white flex."""

DIMPLE_MAT = Finish(roughness=0.5, texture="rgbs_dimple", uv_scale=(6.0, 6.0))
"""The anti-slip mat: a hex-dot rubber texture at a 7 mm dot pitch (24 dots
across a tile that repeats every 168 mm), shinier than chalk because the
dimple crowns catch the light in the clip."""


def _dimple(size: int = 512, across: int = 24, rows: int = 28) -> np.ndarray:
    """Black rubber with a hexagonal lattice of raised dimples.

    Frame 0 samples the mat at sRGB 40/40/42 in the valleys and about 115 on
    the dimple crowns, averaging 73/76/80 over the dimpled area. Painted at
    those values the lab preset rendered it at 104/101/99 -- the light adds --
    so the texture is drawn 0.75x darker than the sample, valleys 32 and
    crowns 92, and the crowns are domes over about half the mat's area. The
    render is what is matched. ``rows`` is even and ``rows / across`` is 1.17
    against a true hex lattice's 1.155, which is what lets a hex pattern tile a
    square seamlessly; the 1 % squash is invisible.
    """
    valley = np.array([32.0, 32.0, 34.0]) / 255.0
    crown = np.array([92.0, 94.0, 98.0]) / 255.0
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64) / size
    pitch, row_h = 1.0 / across, 1.0 / rows
    best = np.full((size, size), np.inf)
    j0 = np.floor(ys / row_h)
    for dj in (0.0, 1.0):
        j = j0 + dj
        shift = (j % 2.0) * 0.5 * pitch
        cx = np.round((xs - shift) / pitch) * pitch + shift
        cy = j * row_h
        best = np.minimum(best, np.hypot(xs - cx, ys - cy))
    dome = np.clip(1.0 - (best / (0.46 * pitch)) ** 2, 0.0, 1.0)
    grain = (np.random.default_rng(17).random((size, size)) - 0.5) * 0.05
    img = valley[None, None, :] + (crown - valley)[None, None, :] * dome[..., None]
    img = img * (1.0 + grain)[..., None]
    return np.clip(img, 0.0, 1.0)


register_texture("rgbs_dimple", _dimple, version=3)


def shape_labels(name: str) -> tuple[str, ...]:
    """Every friction-surface label a box called ``name`` will carry.

    **The one place this set is written.** ``EntityRod.apply_model_materials``
    assigns friction by the tail of a shape's label -- a gripper, a named
    surface, or ``mu_base`` at 2.0 -- and 2.0 combines with the cable's own 60
    as ``sqrt(mu_a * mu_b)`` to mu ~= 10.9. Nothing raises; the cable simply
    never slides on that shape. So the task's rod spec builds its
    ``surface_labels`` from this call, and the test that every colliding shape
    the entity emits is in it reads the built model rather than this tuple.

    **What naming a shape here costs, stated because it is not obvious and it is
    paid by the other hand.** The pass writes ``mu_surface`` to the SHAPE, not
    to a pair -- so the lid strip the support hand pinches is 0.0011 to
    everything, and a gripper blade at ``mu_gripper`` 60 holds it at
    ``sqrt(60 * 0.0011)`` = **0.257**, against the 60.0 that same blade gets on
    the cable. The hand grips this box 233x more weakly than it grips the wire
    running through it.

    That is survivable here, and measured rather than assumed: a 350 g box on a
    foot that meets the table at 0.047 is held by its own weight at 161 mN
    against the feed's steady bore drag of 17, so the pinch is not what keeps it
    still in the ordinary case -- it is what keeps it still when the feeding hand
    does something sudden, which is the case the routine's release exists for.
    If a variant ever needs a firm hold on this box, the fix is to split each
    gripped face into a liner the cable rubs (named here) and an outer skin the
    hand grips (not named), which is what a real moulding is anyway; it costs
    four boxes and is not worth it until something needs it.
    """
    return tuple(f"{name}_{part}" for part in PARTS)


def mouth_width() -> float:
    """How wide the flared entry is at its outer lip [m]."""
    return BORE + 2.0 * FLARE * math.tan(FLARE_ANGLE)


def flare_inner_face(side: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Where one flare panel's inner face must run, in the box's frame.

    ``(at the entry, at the lip)`` as ``(x, y)`` pairs, for ``side`` -1 or +1.
    The intent, stated apart from the box that realises it, so
    :func:`flare_box` can be checked against what it is for. The face starts
    flush with the bore's wall and opens by ``FLARE * tan(FLARE_ANGLE)``;
    anything else is a step the cable catches on, or a funnel that narrows.
    """
    spread = FLARE * math.tan(FLARE_ANGLE)
    return (
        (side * BORE / 2, -RUN / 2),
        (side * (BORE / 2 + spread), -(RUN / 2 + FLARE)),
    )


def flare_box(side: float) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """One flare panel as ``(centre, half_extents, yaw)`` in the box's frame.

    A box whose inner face is the segment :func:`flare_inner_face` names, pushed
    ``WALL / 2`` along that segment's outward normal and yawed to lie along it.
    Derived rather than written down: the first conduit anchored this on the
    wall's centreline and built a funnel 3 mm wider at the entry than at the
    lip, which nothing raised and no picture showed.
    """
    near, far = flare_inner_face(side)
    along = (near[0] - far[0], near[1] - far[1])
    length = math.hypot(*along)
    unit = (along[0] / length, along[1] / length)
    normal = (unit[1], -unit[0])
    if normal[0] * side < 0:
        normal = (-normal[0], -normal[1])
    centre = (
        0.5 * (near[0] + far[0]) + (WALL / 2) * normal[0],
        0.5 * (near[1] + far[1]) + (WALL / 2) * normal[1],
        BORE / 2,
    )
    return centre, (WALL / 2, length / 2, BORE / 2), side * FLARE_ANGLE


def flare_roof_box() -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """The lead-in roof as ``(centre, half_extents, pitch)`` in the box's frame.

    :func:`flare_floor_box`'s mirror about the bore's mid-height: its underside
    starts flush with the bore's ceiling at the entry and rises outward at
    :data:`FLARE_ANGLE`, so a cable pushed up a steep line is turned down into
    the run instead of over the lip.
    """
    along = (0.0, -math.cos(FLARE_ANGLE), math.sin(FLARE_ANGLE))
    normal = (0.0, math.sin(FLARE_ANGLE), math.cos(FLARE_ANGLE))
    mid = (0.0, -RUN / 2 + 0.5 * FLARE * along[1], BORE + 0.5 * FLARE * along[2])
    centre = (mid[0], mid[1] + (WALL / 2) * normal[1], mid[2] + (WALL / 2) * normal[2])
    half = (mouth_width() / 2 + WALL, FLARE / 2, WALL / 2)
    return centre, half, -FLARE_ANGLE


def flare_floor_box() -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """The lead-in floor as ``(centre, half_extents, pitch)`` in the box's frame.

    Its top face starts flush with the bore floor at the entry (``y = -RUN/2,
    z = 0``) and falls away outward at :data:`FLARE_ANGLE` for :data:`FLARE`
    along its slope, so a cable arriving under the floor's level rides up into
    the mouth. ``pitch`` is the rotation about +x that lays the box's local +y
    along that slope; the box sits ``WALL / 2`` under the face.
    """
    along = (0.0, -math.cos(FLARE_ANGLE), -math.sin(FLARE_ANGLE))
    normal = (0.0, -math.sin(FLARE_ANGLE), math.cos(FLARE_ANGLE))
    mid = (0.0, -RUN / 2 + 0.5 * FLARE * along[1], 0.5 * FLARE * along[2])
    centre = (mid[0], mid[1] - (WALL / 2) * normal[1], mid[2] - (WALL / 2) * normal[2])
    half = (mouth_width() / 2 + WALL, FLARE / 2, WALL / 2)
    return centre, half, FLARE_ANGLE


def box_frame() -> dict[str, float]:
    """Every derived level and face of the box, in its own frame [m].

    The ONE place the box's geometry is derived: the entity builds from it, the
    document's regions are checked against it, and the routine reads the lid's
    top, the mat's top and the ledge off it rather than restating them.
    """
    x_front = -(BORE / 2 + WALL)
    x_slot0 = BORE / 2 + WALL
    return {
        "x_front": x_front,
        "x_back": x_front + BOX_DEPTH,
        "x_slot0": x_slot0,
        "x_slot1": x_slot0 + SLOT,
        "y_right": -RUN / 2,
        "y_left": -RUN / 2 + BOX_LENGTH,
        "y_exit": RUN / 2,
        "y_centre": -RUN / 2 + BOX_LENGTH / 2,
        "z_lid": BORE + ROOF,
        "z_mat": MAT_TOP,
        "z_deck": MAT_TOP - MAT_THICK,
        "z_shelf": 0.0,
        "z_slot": -SLOT_DEPTH,
        "z_bottom": MAT_TOP - BOX_HEIGHT,
        "x_strip": 0.5 * (x_slot0 + SLOT + x_front + BOX_DEPTH),
        "z_strip": MAT_TOP + STRIP_SIZE[2] / 2,
    }


@dataclass(frozen=True, kw_only=True)
class SpecCableBox(ObjectSpec):
    """The cable box: one free body, a dozen boxes, a bore along its front-top edge.

    Every dimension is a module constant rather than a field, because the
    document's regions transcribe the bore and ``tests/test_rgbs_tunnel.py``
    holds the two together by reading the module; a spec that could restate
    the run would be a second place for the number to drift.
    """

    entity: str = "cable_box"
    category: str = "cable_box"
    name: str = "box"

    mass: float | None = 0.35
    """A 420 x 200 x 105 plastic box with a rubber mat on it [kg]. Light enough
    that the support hand's hold is what keeps it still under the feed, and
    heavy enough that it does not walk under the cable's own weight."""

    friction: float | None = 0.5
    """Plastic on a foam mat. **Read by nobody in this suite**, and that is worth
    stating: the rod's ``scene_friction`` pass rewrites every shape's friction
    at bind (``mu_surface`` on the labelled shapes, ``mu_base`` on the foot), so
    what the box actually meets the table with is that pass's number. It is
    kept so the box built alone behaves like a box."""

    color: tuple[float, float, float] | None = BOX_CREAM
    finish: Finish | None = SATIN
    """Cream plastic, satin, on every shape not painted otherwise by
    :meth:`EntityCableBox.apply_display`. See :data:`BOX_CREAM` for why it is
    not ``look.DUCT_CREAM``."""

    def __post_init__(self) -> None:
        # The foot slab is what the base class would build at the origin; it is
        # stated here so the spec reads as a primitive object (which is what
        # `after_build` keys on), and built by this entity at its real offset.
        if self.shape is None:
            f = box_frame()
            hx = 0.5 * (f["x_back"] - f["x_front"]) - BASE_INSET
            hy = 0.5 * BOX_LENGTH - BASE_INSET
            object.__setattr__(self, "shape", ShapeSpec(kind="box", half_extents=(hx, hy, BASE_THICK / 2)))
        super().__post_init__()


@register_entity("cable_box")
class EntityCableBox(EntityRigid):
    """The box: a free body carrying the bore, the shelf, the mat, the strip and the decals."""

    spec: SpecCableBox

    def __init__(self, spec: SpecCableBox, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.part_shapes: dict[str, int] = {}
        self.decal_shapes: list[int] = []
        self.notch_shapes: list[int] = []
        self.outlet_shapes: list[int] = []
        self.mat_shapes: list[int] = []
        self.strip_shapes: list[int] = []
        self.cord_shapes: list[int] = []
        self.cable_shapes: list[int] = []

    # ------------------------------------------------------------------ build
    def build(self, builder: newton.ModelBuilder, episode) -> None:
        """The free body, then every box on it.

        Not ``super().build()``: the base class draws the spec's primitive at
        the body's origin, and this body's origin is the bore axis -- the
        shapes all sit at offsets from it, the foot included.
        """
        spec, f = self.spec, box_frame()
        pose = self._placed(self._episode_pose(episode))
        xform = wp.transform(tuple(float(v) for v in pose[:3]), wp.quat(*[float(v) for v in pose[3:7]]))
        body = builder.add_body(xform=xform, label=spec.name)
        joint = builder.joint_count - 1
        self.coord_start = int(builder.joint_q_start[joint])
        self.dof_start = int(builder.joint_qd_start[joint])
        self.body = body

        cfg = self._contact_cfg(builder)
        # Overwritten for a scene with a rod in it -- EntityRod restates every
        # shape's material at bind -- but `gap` is not part of that pass.
        cfg.margin, cfg.gap = 0.0, BOX_GAP
        ident = wp.quat_identity()
        y_mid = f["y_centre"]

        def box(part: str, centre, half, quat=ident, *, cfg=cfg) -> int:
            shape = builder.add_shape_box(
                body,
                xform=wp.transform(wp.vec3(*[float(v) for v in centre]), quat),
                hx=float(half[0]),
                hy=float(half[1]),
                hz=float(half[2]),
                cfg=cfg,
                label=f"{spec.name}_{part}",
            )
            self.part_shapes[part] = int(shape)
            if part in CUTAWAY:
                builder.shape_flags[shape] = int(builder.shape_flags[shape]) & ~int(newton.ShapeFlags.VISIBLE)
            return int(shape)

        def slab(part: str, x0, x1, z0, z1, y0=None, y1=None) -> int:
            y0 = f["y_right"] if y0 is None else y0
            y1 = f["y_left"] if y1 is None else y1
            return box(
                part,
                (0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)),
                (0.5 * (x1 - x0), 0.5 * (y1 - y0), 0.5 * (z1 - z0)),
            )

        # --- the bore, RUN long, on the shelf ---------------------------------
        # Each wall is a liner the cable rubs and a skin the hand grips: one
        # shape cannot carry both frictions. See LINER.
        liner_off = 0.5 * (BORE + LINER)
        skin_off = 0.5 * BORE + LINER + 0.5 * (WALL - LINER)
        box("wall_front", (-liner_off, 0.0, BORE / 2), (LINER / 2, RUN / 2, BORE / 2))
        box("wall_back", (liner_off, 0.0, BORE / 2), (LINER / 2, RUN / 2, BORE / 2))
        box("skin_front", (-skin_off, 0.0, BORE / 2), ((WALL - LINER) / 2, RUN / 2, BORE / 2))
        box("skin_back", (skin_off, 0.0, BORE / 2), ((WALL - LINER) / 2, RUN / 2, BORE / 2))
        box("roof", (0.0, 0.0, BORE + ROOF / 2), (BORE / 2 + WALL, RUN / 2, ROOF / 2))
        for part, side in (("flare_left", -1.0), ("flare_right", 1.0)):
            centre, half, yaw = flare_box(side)
            box(part, centre, half, wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(yaw)))
        for part, geometry in (("flare_floor", flare_floor_box()), ("flare_roof", flare_roof_box())):
            centre, half, pitch = geometry
            box(part, centre, half, wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(pitch)))

        # --- the box body: shelf, front block, deck, mat, foot ----------------
        slab("shelf", f["x_front"], f["x_slot0"], f["z_slot"], f["z_shelf"])
        slab("front_block", f["x_front"], f["x_slot1"], f["z_bottom"] + BASE_THICK, f["z_slot"])
        slab("deck", f["x_slot1"], f["x_back"], f["z_bottom"] + BASE_THICK, f["z_deck"])
        self.mat_shapes.append(slab("mat", f["x_slot1"], f["x_back"], f["z_deck"], f["z_mat"]))
        base = slab(
            "base",
            f["x_front"] + BASE_INSET,
            f["x_back"] - BASE_INSET,
            f["z_bottom"],
            f["z_bottom"] + BASE_THICK,
            f["y_right"] + BASE_INSET,
            f["y_left"] - BASE_INSET,
        )
        del base

        # --- the power strip, lying on the mat --------------------------------
        sl, sw, sh = STRIP_SIZE
        self.strip_shapes.append(box("strip", (f["x_strip"], y_mid, f["z_strip"]), (sw / 2, sl / 2, sh / 2)))

        # --- decals: no collision, no mass -------------------------------------
        decal = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        lift = 0.0004
        nw, nh = NOTCH
        notch = box(
            "notch",
            (f["x_front"] - lift, y_mid - 0.02, f["z_bottom"] + BASE_THICK + nh / 2 + 0.002),
            (lift, nw / 2, nh / 2),
            cfg=decal,
        )
        self.notch_shapes.append(notch)
        self.decal_shapes.append(notch)
        # Six outlets in a row and two USB ports at the -y end, on the strip's top.
        top = f["z_strip"] + sh / 2 + lift
        for k in range(6):
            y = y_mid - sl / 2 + 0.075 + k * 0.032
            self.outlet_shapes.append(box(f"outlet_{k}", (f["x_strip"], y, top), (0.010, 0.011, lift), cfg=decal))
        for k in range(2):
            y = y_mid - sl / 2 + 0.022 + k * 0.016
            self.outlet_shapes.append(box(f"usb_{k}", (f["x_strip"], y, top), (0.006, 0.004, lift), cfg=decal))
        self.decal_shapes.extend(self.outlet_shapes)
        # The strip's cord: off its -y end, curving away toward the far edge and
        # down the back of the box. Capsules extend along their local +z.
        cord = [
            ((f["x_strip"], y_mid - sl / 2 - 0.020, MAT_TOP + CORD_RADIUS), (0.0, -1.0, 0.0), 0.020),
            ((f["x_strip"] + 0.024, y_mid - sl / 2 - 0.052, MAT_TOP + CORD_RADIUS), (0.6, -0.8, 0.0), 0.022),
            ((f["x_strip"] + 0.058, y_mid - sl / 2 - 0.062, MAT_TOP + CORD_RADIUS), (1.0, -0.15, 0.0), 0.020),
            ((f["x_back"] + CORD_RADIUS, y_mid - sl / 2 - 0.064, MAT_TOP - 0.045), (0.0, 0.0, -1.0), 0.050),
            ((f["x_back"] + 0.045, y_mid - sl / 2 - 0.064, f["z_bottom"] + CORD_RADIUS), (1.0, 0.0, 0.0), 0.045),
        ]
        for k, (centre, axis, half) in enumerate(cord):
            axis_v = np.asarray(axis, dtype=np.float64)
            axis_v /= np.linalg.norm(axis_v)
            quat = _quat_z_to(axis_v)
            shape = builder.add_shape_capsule(
                body,
                xform=wp.transform(wp.vec3(*[float(v) for v in centre]), quat),
                radius=CORD_RADIUS,
                half_height=float(half),
                cfg=decal,
                label=f"{spec.name}_cord_{k}",
            )
            self.cord_shapes.append(int(shape))
            self.decal_shapes.append(int(shape))

        set_body_mass(builder, body, float(spec.mass))
        self._assert_layout(f)

    def _assert_layout(self, f: dict[str, float]) -> None:
        """What the routine and the document both lean on, checked where it is built."""
        assert f["z_lid"] - f["z_mat"] > 0.0, "the lid must stand proud of the mat"
        assert f["x_slot1"] - f["x_slot0"] >= 0.016, "the slot must pass a fingertip"
        assert 0.0 < LINER < WALL, "a wall is a liner the cable rubs plus a skin the hand grips"
        ledge = f["y_left"] - f["y_exit"]
        assert ledge >= 0.120, (
            f"the ledge past the exit is {ledge * 1e3:.0f} mm; the connector is grabbed 26 mm past the "
            "exit and drawn 70 mm along it by a hand 20 mm wide, which is 116, and a head drawn off "
            "the box's end drops 100 mm and leaves the goal's z window"
        )

    # ---------------------------------------------------------------- display
    def apply_display(self, builder: newton.ModelBuilder) -> None:
        """Cream satin everywhere, then the parts that are not cream.

        The mat and the strip are painted for every backend and given their
        finishes for the path tracer; the decals get theirs. Painted here rather
        than through the spec because a spec carries one colour and this body
        wears four.
        """
        super().apply_display(builder)
        for s in self.mat_shapes:
            builder.shape_color[s] = MAT_BLACK
            self.finishes[s] = DIMPLE_MAT
        for s in self.strip_shapes:
            builder.shape_color[s] = STRIP_WHITE
            self.finishes[s] = SATIN
        for s in self.cord_shapes:
            builder.shape_color[s] = STRIP_WHITE
            self.finishes[s] = GLOSS
        for s in self.notch_shapes:
            builder.shape_color[s] = BLACK
            self.finishes[s] = SATIN
        for s in self.outlet_shapes:
            builder.shape_color[s] = OUTLET_GREY
            self.finishes[s] = SATIN


def _quat_z_to(axis: np.ndarray) -> wp.quat:
    """The rotation taking local +z onto ``axis`` (unit), as a warp quaternion."""
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(z @ axis, -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return wp.quat_identity()
    if dot < -1.0 + 1e-9:
        return wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)
    cross = np.cross(z, axis)
    cross /= np.linalg.norm(cross)
    return wp.quat_from_axis_angle(wp.vec3(*[float(v) for v in cross]), math.acos(dot))
