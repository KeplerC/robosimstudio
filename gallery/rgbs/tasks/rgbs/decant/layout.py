# SPDX-License-Identifier: Apache-2.0
"""Where ``seam`` and ``decant`` stand their carton: the placements the records build.

Layout (surface frame; the bench top's centre is the origin, world (0.6, 0,
0.75)). The YAM i2rt pair mounts at x = -0.3475, y = +-0.31, facing each other
across y.

**THE CARTON STANDS WHERE THE CLIP HAS IT, and the clip's placement is
reachable -- measured, not assumed.** ``sample_videos/box_decanting.mp4``
stands its parcel on the bench's centreline (its centre reads within a grid
square of y = 0 in both the phone frame and the overhead inset) with its long
axis along the bench, a little on the arms' side of the bench's middle. The
seam therefore runs along world y and the 100 mm of tape is drawn along it
toward +y, from the right (-y) end to the left, which is what the clip shows.

Swept here with a chained full-pose solve (21 stations along the tape at its
own height, the knife's tilted hold, 4 mm / 6 deg gate) for the working arm,
worst residual over the draw:

    carton x    tilt  0    20     30     40     45  deg   (nose toward the arm)
      -0.10        0.0    0.0    0.0    0.0    0.0  mm  (1.5-1.7 deg)
      -0.06        0.0    0.0    0.0    0.0    0.0
      -0.02        0.0    0.0    0.0    0.0    0.0
      +0.02       44.9    2.0    0.0    0.0    0.0  (6.0 deg at 20)
      +0.06       52.2   26.4    0.1    0.0    0.0  (5.4 deg at 30)

and the SAME hold with the nose LEADING the draw (the blade tip ahead of the
hand, the wrist over the tip from the +y side) is 93-194 mm out at every
station -- the wrist would have to reach past the tape and hook back. So the
knife is held with its nose toward the working arm, tilted 35 degrees so the
tip is the lowest thing on it, and drawn +y with the tip trailing the body;
the cut law reads speed along the edge and does not care which end leads.

**Which x the carton stands at is set by the flap, not by the draw.** The draw
solves anywhere from x -0.10 to +0.02; the arc that hooks the far flap open --
a top-down hand whose +x fingertip follows the flap's free edge round its hinge
to 110 degrees -- solves at 0.0 mm for x <= -0.06 and is 33 mm out at -0.02,
56-69 mm at +0.02 and beyond. So -0.06 is the far end of the band both moves
share, and it is as near the clip's placement as the second move allows.

**And the pin goes on the near flap's +y end**, closed jaws pressed straight
down 15 mm in from the +y end wall and 11 mm in from the hinge: the far flap is
the one that gets lifted, so a hand on it would hold it shut, and the near flap
stays shut with its half of the tape on it anyway. Inboard of the hinge rather
than beside the seam, so the support hand's carriage stays clear of the working
hand's over the tape. The support arm holds that pose at 0.0 mm / 1.0-1.1 deg
from its home seed at every candidate swept.

The cutter lies flat on the mat to the right (-y) of the carton and a little
beyond it (+x), nose toward the arm that takes it, so a top-down pick across
its body already holds it the way the draw wants it. It goes back down where it
came from.

**What this module is, structurally.** Every factory here is zero-argument and
returns the placed spec under the name the documents use, because a document
names a *type* and the type's record builds it (:attr:`AssetRecord.build`).
``records.py`` names these factories; this module holds the arithmetic they
share, so the seam field's restatement of the carton's frame and the knife's
edge is derived from the same two specs the scene is built from.
"""

from __future__ import annotations

import math

from robosimstudio.specs import ObjectSpec, StaticSpec
from robosimstudio.specs.asset import ShapeSpec

from ..look import grid_finish_at
from ..staging import rgbs_views
from .carton import SpecShippingCarton, carton_frame
from .knife import SpecBoxCutter, blade_edge_local
from .seam import SpecSealTape

__all__ = [
    "CARTON",
    "CARTON_AT",
    "CARTON_YAW",
    "CONTENTS",
    "CONTENTS_HALF",
    "DROP_CENTER",
    "DROP_HALF",
    "KNIFE",
    "KNIFE_AT",
    "KNIFE_YAW",
    "MAT_CENTER",
    "MAT_HALF",
    "SEAM_AIM",
    "SEAM_VIEWS",
    "TAPE_TOP",
    "box_cutter",
    "carton_contents",
    "decant_mat",
    "seal_tape",
    "shipping_carton",
]

CARTON_AT = (-0.06, 0.0)
"""Where the carton stands, in the surface frame -- world (0.58, 0).

On the bench's centreline, as the clip has it, and 20 mm on the arms' side of
the bench's middle: inside the band the draw sweep in the module docstring
solves at every tilt, and where the support arm's pin and the working arm's
flap hook both solve (``[pin]``, ``[flap]`` in the same sweep). Stated on its
own rather than taken from ``MAT_CENTER``: the mat is furniture, this is the
number the whole layout is fitted against."""

CARTON_YAW = -0.5 * math.pi
"""The carton's yaw [rad]: its long axis -- and the seam, the entity's local
``+x`` -- along world ``y``. A quarter turn this way rather than the other so
the entity's ``near`` flap (its local ``-y`` hinge) is the one on the arms'
side (world ``-x``) and ``far`` is the one at the slat wall: the names the
documents use and the routine hooks."""

MAT_CENTER = (CARTON_AT[0], -0.085)
MAT_HALF = (0.14, 0.225, 0.002)
"""The work mat: what the carton stands on, the surface it slides across, what
the knife lies on and -- in `decant` -- what the contents are set down on.

Longer toward -y than the carton needs because the knife lies there (out to
y = -0.31 since the knife moved for the YAM's wrist camera) and the set-down
goes there too; both bands are on the working arm's side, which is the
one that can reach them top-down. Under the path tracer it wears the bench's
own 25 mm grid and disappears into it, which is what the clip shows -- a carton
on the mat, no pad."""

KNIFE_AT = (CARTON_AT[0] + 0.05, -0.235)
"""Where the cutter lies, in the surface frame.

To the right (-y) of the carton and a little beyond it (+x), as the clip has
it. Its body runs from y = -0.285 to -0.185, 85 mm clear of the carton's end
wall. It lay 15 mm clear on ``yam_i2rt``; on ``yam`` the hold that can follow
the draw has the wrist toward the carton, 44 mm off the grip point with a D405
camera 80 mm further out, and 15 mm of clearance put that camera on the far
flap 38 mm before the jaws reached the knife."""

KNIFE_YAW = -0.5 * math.pi
"""The cutter's yaw [rad]: its long axis along world y with the NOSE toward the
working arm (-y). A top-down pick with the jaws across the body then already
holds it the way the draw sweep says the tilted hold has to be -- nose toward
the arm -- with no turn in the carry."""

DROP_CENTER = (-0.06, -0.12)
DROP_HALF = (0.050, 0.040)
"""Where the contents are set down, in the MAT's own frame [m].

Surface (-0.13..-0.03, -0.21..-0.13): on the mat, between the carton's -y end
wall and the working arm's base, clear of the carton's footprint in y and of
the knife (which lies at x >= +0.065 in the mat's frame) in x."""

CONTENTS_HALF = (0.045, 0.030, 0.040)
"""The packed part: 90 x 60 x 80 mm, in the CARTON's frame (90 along the seam).

Its height is the number that matters. It stands on the carton floor, so its
top face sits 16 mm below the rim -- inside a YAM fingertip's 35 mm of usable
depth, which is what makes the lift a grasp rather than a fishing expedition."""

MAT_COLOR = (0.24, 0.24, 0.26)
"""What every backend but the path tracer draws the mat as; under RTX it wears
the bench's own grid (:func:`~look.grid_finish_at`) and disappears into it."""

_MAT_TOP = 2.0 * MAT_HALF[2]

CARTON = SpecShippingCarton(
    name="carton",
    # AT REST on the mat, with half a millimetre of settling clearance. Derived
    # rather than typed: a spawn that is not where the carton comes to rest puts
    # the seam somewhere the bench does not park the blade, and every press then
    # reads exactly 0.00 N -- which is how this was found.
    pose=(CARTON_AT[0], CARTON_AT[1], _MAT_TOP + 0.5 * SpecShippingCarton().wall + 0.0005),
    quat=(0.0, 0.0, math.sin(0.5 * CARTON_YAW), math.cos(0.5 * CARTON_YAW)),
)

KNIFE = SpecBoxCutter(
    name="knife",
    # Flat on the mat on its underside, the same half millimetre of clearance.
    pose=(KNIFE_AT[0], KNIFE_AT[1], _MAT_TOP + SpecBoxCutter().body_half[2] + 0.0005),
    quat=(0.0, 0.0, math.sin(0.5 * KNIFE_YAW), math.cos(0.5 * KNIFE_YAW)),
)

CONTENTS = ObjectSpec(
    name="part",
    category="carton_contents",
    shape=ShapeSpec(kind="box", half_extents=CONTENTS_HALF),
    pose=(
        CARTON_AT[0],
        CARTON_AT[1] - 0.02,
        float(CARTON.pose[2]) + 0.5 * float(CARTON.wall) + CONTENTS_HALF[2] + 0.0005,
    ),
    quat=CARTON.quat,
    mass=0.20,
    friction=0.6,
    ke=6.0e4,
    kd=500.0,
    color=(0.30, 0.42, 0.62),
)
"""The packed part, standing on the carton floor, turned with the carton. `seam`
ships the carton EMPTY on purpose -- a part rattling inside changes the base's
inertia and so changes what the pin has to react, and a cut measured on one
carton would not transfer to the other -- so only `decant` declares it."""

TAPE_TOP = float(CARTON.pose[2]) + float(carton_frame(CARTON)["seam_z"]) + SpecSealTape().tape_thickness
"""The tape's top face [m, surface frame]: 0.1085. The mat, the carton's floor
slab, its walls, a flap and a strip, added up once."""

SEAM_AIM = (CARTON_AT[0], CARTON_AT[1], TAPE_TOP)
"""What the rig looks at [m, surface frame]: the middle of the tape on the seam.

One aim for both variants. ``decant``'s set-down band is 170 mm away in -y,
inside the overhead's footprint and inside the plan view's, so nothing about
the second variant asks for a second aim."""

SEAM_VIEWS = rgbs_views(SEAM_AIM)
"""The suite's rig aimed at the tape: what both documents stage as
``(:cameras assets.decant.layout:SEAM_VIEWS)``. Five cameras now -- the two
that reproduce the clip's handheld phone and its inset are the ones the look is
matched in. A constant rather than a registration, because a rig aimed at a
layout is the layout's arithmetic."""


def decant_mat() -> StaticSpec:
    """The work mat. The bench itself is the cell's.

    The mat's friction is 0.9 -- the CARTON's own 0.6 is what decides whether it
    slides, and stating a low number on both would be saying the same thing
    twice in a place where only one of them is read."""
    return StaticSpec(
        name="mat",
        half_extents=MAT_HALF,
        xform=(MAT_CENTER[0], MAT_CENTER[1], MAT_HALF[2], 0.0, 0.0, 0.0, 1.0),
        friction=0.9,
        color=MAT_COLOR,
        finish=grid_finish_at(MAT_CENTER),
    )


def shipping_carton() -> SpecShippingCarton:
    """The carton, at rest on the mat, long axis along the bench."""
    return CARTON


def box_cutter() -> SpecBoxCutter:
    """The cutter, flat on the mat beside the carton."""
    return KNIFE


def carton_contents() -> ObjectSpec:
    """The packed part, on the carton floor."""
    return CONTENTS


def seal_tape() -> SpecSealTape:
    """The tape: two strips and the field on them. Built last by stage: it
    resolves both flap link labels and the blade shape, all of which must
    already exist.

    Its geometry is RESTATED from the two frame functions rather than reached
    for at build time, which is polish's discipline -- a spec must build with no
    sibling entity in hand, and the field asserts the restatement against what
    was actually built when it binds."""
    frame = carton_frame(CARTON)
    p0, direction, length = blade_edge_local(KNIFE)
    return SpecSealTape(
        name="seam",
        carton=CARTON.name,
        knife_object=KNIFE.name,
        flap_half=tuple(float(v) for v in frame["flap_half"]),
        seam_gap=float(CARTON.seam_gap),
        seam_len=float(frame["seam_len"]),
        flap_travel=float(CARTON.flap_travel),
        blade_p0=tuple(float(v) for v in p0),
        blade_dir=tuple(float(v) for v in direction),
        blade_len=float(length),
    )
