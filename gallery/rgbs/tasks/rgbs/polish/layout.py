# SPDX-License-Identifier: Apache-2.0
"""Where ``strip`` stands its machine, its rest and its strip: the placements the records build.

Layout (surface frame; the bench top's center is the origin). The YAM pair
mounts at x = -0.3475, y = +-0.31, facing each other across y -- the composed
workcell's own bases. The stand therefore goes deep (+x) and the tool is
clamped SIDEWAYS across it, nose forward over the mat (yaw pi/2 turns the
stand's local forward, +y, onto world -x) and fifteen degrees down. That puts
the **spindle axis along world x** -- pointing at the arms -- and the drum's
BARREL, not its end face, over the work. The stand is set back far enough that
its column and clamp sit behind the arms' working corridor: at the 105 mm reach
this first shipped with, the right arm's link4 jammed into the clamp on every
approach.

**Which axis is which is the whole task.** The strip lies with its long axis
along world y, arm to arm; the spindle runs along world x. So the rub that
polishes -- along the spindle, toward and away from the arms -- is a move
ACROSS the strip's width, and the rub that does not is a move along its length.
The second one fails twice over: the contact band is already wider than a mark
in that direction, and it is the direction the drum's surface runs in, so it
costs the full mu * press drag through a pinch that cannot carry it.

The strip starts on a low parts rest with its long axis along y, one graspable
end toward each arm, and BOTH hands hold it from the pick to the set-down: the
three marks sit at 0 and +-33 mm along a strip the hands take at +-58 mm, so a
mark is brought to the fixed bit by translating the whole rigid body rather than
by putting it down and regrasping. That was measured against the alternative --
see ``graphs/polish_gt/README.md``; a single hand holds the strip perfectly
well, but cannot react the press torque -- the mark is 78 mm from the grasp and
the grip's couple is only as wide as the strip is thick -- and a rub loads that
couple for its whole length rather than for the moment of a press.

**Load-bearing, not scenery.** The hands take the strip END-ON -- in along its
own long axis, jaws closing on its faces -- because that is the only grasp
whose arms fit. A top-down grip on the strip's 25 mm edges is the gentler
grasp and was built and measured: descending vertically never moved the part
at all, where every sideways approach did. But the strip lies BETWEEN the two
arm bases, so two hands reaching down onto it bring both forearms over its
middle, and link4 jams into link4 at 70-77 N at every station tried (+-58,
+-66, +-72 mm); leaning the hands outboard to splay the arms only trades that
for link4 against the machine's housing at 87 N.

An end-on grasp needs room for the lower pad *under* the strip, which a strip
lying flat on the mat does not have -- hence the rest. 30 mm is where both arms
first solve on-target and clear (swept-sphere clearance to the machine +29 mm
right, +74 mm left; nothing at 20 mm or below solves at all). The rest is kept
SHORT along the strip -- 32 mm -- so the hands can take it well inboard of its
ends: grip too close to a tip and there is barely a centimetre of strip between
pads 20 mm deep, and the carry levers it out; grip inboard of a wide rest and
the pads hit the rest instead.

The rest carries two locating rails, and their height is a measured decision
rather than a detail -- see ``RAIL_HALF``.

**The rig.** ``strip`` stages the suite's default rig, ``rgbs_views@v1``,
unaimed: the calibration behind that rig was transplanted onto *this* layout's
drum contact, so the default IS this variant's aim. :data:`STRIP_AIM` derives
the same point from the machine's own spec and holds it against the staging
module's restatement, so neither can drift from the other unnoticed.
"""

from __future__ import annotations

import math

from robosimstudio.core.render.palette import DEFORMABLE, FIXTURE
from robosimstudio.specs import ObjectSpec, StaticSpec
from robosimstudio.specs.asset import ShapeSpec

from ..look import BLACK, BLACK_PLASTIC, grid_finish_at
from ..staging import POLISH_AIM
from .machine import SpecPolishMachine
from .marks import SpecPolishMarks

__all__ = [
    "MACHINE_AT",
    "MACHINE_YAW",
    "MAT_CENTER",
    "MAT_HALF",
    "RAIL_HALF",
    "RAIL_OFFSET",
    "REST_CENTER",
    "REST_HALF",
    "STRIP_AIM",
    "STRIP_HALF",
    "STRIP_QUAT",
    "STRIP_SPAWN",
    "parts_rest",
    "polish_machine",
    "polish_marks",
    "polish_mat",
    "polish_rail_far",
    "polish_rail_near",
    "polish_strip",
]

STRIP_HALF = (0.075, 0.0125, 0.002)
"""The strip's half extents [m] -- one constant, read by the record's grasp
box, the placed ObjectSpec and the mark field's face geometry alike.

4 mm thick, the reference part's own dimension. Thickening it to 8 mm was
tried, on the theory that the jaws simply could not close far enough, and it
changed nothing: the pads still reach their commanded gap with the strip
nowhere between them."""

MACHINE_AT = (0.155, 0.0)
"""Where the machine's stand sits on the bench [m, surface frame]."""

MACHINE_YAW = math.pi / 2
"""Turns the stand's local forward (+y) onto world -x, nose over the mat."""

MAT_CENTER = (-0.06, 0.0)
MAT_HALF = (0.12, 0.10, 0.002)
"""The work mat: what the rest and the set-down both sit on."""

REST_CENTER = (-0.05, 0.0)
REST_HALF = (0.020, 0.016, 0.0250)
"""The parts rest: a 40 x 32 x 50 mm block the strip lies across."""

STRIP_SPAWN = (
    REST_CENTER[0],
    REST_CENTER[1],
    2.0 * MAT_HALF[2] + 2.0 * REST_HALF[2] + STRIP_HALF[2] + 0.0005,
)
"""On the rest, with half a millimetre of settling clearance."""

STRIP_QUAT = (0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0))
"""Long axis along y: one graspable end toward each arm."""

RAIL_HALF = (0.002, 0.014, 0.006)
"""Locating rails either side of the strip's width [m].

They are load-bearing, and their HEIGHT is the measured part. Both hands enter
end-on and their axial pushes cancel, but what is left over is a sideways
nudge, and with no rails at all it puts the strip 91 mm off the rest before a
finger closes. At 5 mm the rails stopped that and created a worse failure: they
stood barely a millimetre proud of a 4 mm strip, so the entering pads walked it
up ONTO them -- measured riding 6.8 mm, which leaves the jaws closing through
air the strip has left. At 12 mm they stand 8 mm above its top face and it
cannot climb out.

They flank the strip's middle only (28 mm apart, over the rest's 32 mm), so
nothing here is in the pads' way: the hands take the strip at +-58 mm, well
outboard of the rails, straddling its thickness in free air."""

RAIL_OFFSET = 0.016
"""Half the gap between the rails [m]: 28 mm, on a 25 mm strip."""

MAT_COLOR = FIXTURE
"""The work mat. A pad on the bench, which is what FIXTURE names.

Was ``(0.24, 0.24, 0.26)``, chosen when the bench under it was 0.07 and anything
above black read as a surface. Against the bench's own value it now has to be the
lighter of the two, because the strip is read against the mat and the mat is read
against the bench."""

REST_COLOR = BLACK
"""The parts rest and its rails: black prints, as the reference clip's fixtures
are. Was ``FIXTURE``; before that ``(0.30, 0.30, 0.33)`` -- FIXTURE to two
decimal places, arrived at independently, which is the duplication the palette
exists to stop."""

STRIP_COLOR = DEFORMABLE
"""The strip. Not deformable, but it is the thing whose *surface* is the
observation, and it is read the way a sheet is: pale, against everything else.
Was ``(0.92, 0.92, 0.94)``, which pyrender clipped to pure white."""

_MACHINE = SpecPolishMachine(name="machine", at=MACHINE_AT, yaw=MACHINE_YAW)
"""The one machine spec every reader shares: the factory returns it, and the aim below reads it."""

STRIP_AIM = (MACHINE_AT[0] - _MACHINE.contact_reach, MACHINE_AT[1], _MACHINE.contact_height)
"""Where the rig looks [m, surface frame]: the drum's working contact.

``MACHINE_AT[0]`` less the machine's ``contact_reach`` (the yaw turns its
forward onto world -x), at its ``contact_height``. The staging module restates
this as ``POLISH_AIM`` because it is what the default rig is registered at and
``staging`` may not import the assets; the assertion below is what holds the
two together.
"""

if not all(math.isclose(a, b, abs_tol=1.0e-9) for a, b in zip(STRIP_AIM, POLISH_AIM, strict=True)):
    raise ValueError(
        f"staging.POLISH_AIM {POLISH_AIM} is not the drum contact this layout puts at {STRIP_AIM}; "
        "the default rig rgbs_views@v1 is aimed at the wrong place"
    )


def polish_mat() -> StaticSpec:
    """The work mat. The bench itself is the cell's."""
    return StaticSpec(
        name="mat",
        half_extents=MAT_HALF,
        xform=(MAT_CENTER[0], MAT_CENTER[1], MAT_HALF[2], 0.0, 0.0, 0.0, 1.0),
        friction=0.9,
        color=MAT_COLOR,
        finish=grid_finish_at(MAT_CENTER),
    )


def parts_rest() -> StaticSpec:
    """The parts rest the strip lives on."""
    return StaticSpec(
        name="rest",
        half_extents=REST_HALF,
        xform=(
            REST_CENTER[0],
            REST_CENTER[1],
            2.0 * MAT_HALF[2] + REST_HALF[2],
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        friction=0.9,
        color=REST_COLOR,
        finish=BLACK_PLASTIC,
    )


def _rail(side: str, sign: float) -> StaticSpec:
    return StaticSpec(
        name=f"rail_{side}",
        half_extents=RAIL_HALF,
        xform=(
            REST_CENTER[0] + sign * RAIL_OFFSET,
            REST_CENTER[1],
            2.0 * MAT_HALF[2] + 2.0 * REST_HALF[2] + RAIL_HALF[2],
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        friction=0.6,
        color=REST_COLOR,
        finish=BLACK_PLASTIC,
    )


def polish_rail_near() -> StaticSpec:
    """The locating rail on the -x side of the strip."""
    return _rail("near", -1.0)


def polish_rail_far() -> StaticSpec:
    """The locating rail on the +x side of the strip."""
    return _rail("far", 1.0)


def polish_strip() -> ObjectSpec:
    """The strip: a procedural box, no mesh. Contact pair (6e4, 500) = the
    arrangement's own ceiling, because Newton's bare default is ~400x too soft
    under a position-controlled press.

    ``friction`` is 1.0, and it is load-bearing rather than generous. What holds
    the strip through a press is the friction budget at the pads against the
    drum's drag, and the drag is ``mu * press`` at the *other* contact -- same
    order as the whole budget. The number was also 1.0 in the built model all
    along without being declared: until the primitive path was fixed to read
    this spec (see ``EntityRigidObject._contact_cfg``), a procedural box took
    the builder's default and the 0.5 written here reached nothing. Stated at
    what the measurements were taken on."""
    return ObjectSpec(
        name="strip",
        category="polish_strip",
        shape=ShapeSpec(kind="box", half_extents=STRIP_HALF),
        pose=STRIP_SPAWN,
        quat=STRIP_QUAT,
        mass=0.016,
        friction=1.0,
        ke=6.0e4,
        kd=500.0,
        color=STRIP_COLOR,
    )


def polish_machine() -> SpecPolishMachine:
    """The rotary tool on its stand."""
    return _MACHINE


def polish_marks() -> SpecPolishMarks:
    """The mark field. Built after the machine by stage: it resolves the
    spindle's body label, which must already exist."""
    return SpecPolishMarks(name="marks", strip_object="strip", machine="machine", strip_half=STRIP_HALF)
