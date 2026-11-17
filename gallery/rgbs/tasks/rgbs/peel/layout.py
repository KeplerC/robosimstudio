# SPDX-License-Identifier: Apache-2.0
"""Where the peel variants lay their sheets: the placed records ``handle``,
``panel`` and ``panel_gentle`` build from.

``tape`` and ``film`` (beside this module) say what a bonded sheet *is* -- the
bond field, the readouts, the material -- and deliberately not where one lies.
A document, though, names a type and gets a placed spec back, so a type has to
know its placement: that is what :attr:`~robosimstudio.assetlib.AssetRecord.build`
is for, and it is why the two layouts here are two *records* rather than
two functions only a loader could call. Each is the material record with a
factory attached, under a name that says which layout it is.

**One tape and one film, because two different problems.** ``handle`` is the
manipulation reference -- a short strip along a long, low box, with a tab pinned
rigid at box-top height that a hand can take. ``panel`` is the de-filming one --
the clip's protective film on a phone leaning back in a cradle, an order of
magnitude less bonded per unit area, peeled two-handed. One record could not
place both, and a parameterised factory is what a document cannot call.

**The film lies on a tilted face (2026-09-10).** ``panel`` used to lay its
sheet on a 150 x 223 x 73 mm glossy block standing where the handle box does;
the clip it reproduces has a phone leaning back 30 degrees on a stand, so the
film is now laid in the phone's own frame -- ``surface_quat`` on the spec puts
the bond field's box on the glass and its opening axis along the face normal --
with the pull-tab overhanging the phone's raised top edge. The phone and the
stand are ``phone.py``'s; this module places the sheet on them.

The suite also carried ``along_x``, the physics reference AutoPeel's own strip
was transcribed onto: 26 x 6 cm along a 200 x 160 x 120 mm box with 8 cm of tab
drooping off the far edge, where the bond count of 133 and the 2.0 N aggregate
hold were fitted. It was withdrawn with its document; the constants it fitted
are the ones ``tape.py`` still states, and ``handle`` is what states them now.

**Every constant below is stated in the cell's work-surface frame.** The cell
is :data:`~staging.CELL`, ``bench_075``; it was ``bench_0728`` -- the same table
22 mm lower -- and the move cost these layouts nothing, because both YAM mounts
are derived against the surface frame too. What the move *did* cost is recorded
on :data:`HANDLE_GRASP_XY`, and it is about the robot rather than the table.

**The particle-set selectors stay numpy.** The tab and the handle column are
selected against coordinates the box's placement fixes, which is exactly what a
layout is, and a rule language that could say "everything past this edge"
would be harder to read than the three lines that do.

**The rig is aimed here too.** The suite has one camera rig
(:func:`~staging.rgbs_views`) and it differs per variant only in where it
points; the aim is the box's centre lifted just above its top face, which is
layout arithmetic and so lives with the layout. Each aimed rig is a constant
of this module, and a document stages it by artifact reference --
``(:cameras assets.peel.layout:HANDLE_VIEWS)`` -- the way it names a routine.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from robosimstudio.assetlib import AssetRecord, register_asset
from robosimstudio.assetlib.semantics import FeatureSpec
from robosimstudio.assetlib.sim import SimRequirements, contacts_with
from robosimstudio.core.render.palette import SUBSTRATE
from robosimstudio.specs import StaticSpec
from robosimstudio_bench.taskbuild.arrangements import register_arrangement

from ..look import CLEAR_FILM, KRAFT, TAPE
from ..staging import CLOTH_VBD, rgbs_views
from .film import FILM, SpecFilm
from .phone import (
    PHONE_FOOT,
    PHONE_SIZE,
    PHONE_TILT_DEG,
    PhoneFrame,
    SpecPhone,
    SpecPhoneCradle,
    cradle_block_centre,
    phone_frame,
)
from .tape import TAPE as TAPE_RECORD
from .tape import SpecTape

__all__ = [
    "AIM_RISE",
    "FILM_FINISH",
    "FILM_PHONE",
    "FILM_TAB",
    "FILM_WIDTH",
    "HANDLE_AIM",
    "HANDLE_BOX_CENTER",
    "HANDLE_VIEWS",
    "PANEL_AIM",
    "PANEL_VIEWS",
    "PEEL_BOX_HANDLE",
    "PHONE",
    "PHONE_CRADLE",
    "PHONE_PHYSICS",
    "TAPE_HANDLE",
    "handle_box",
    "handle_tape",
    "panel_film",
    "phone",
    "phone_cradle",
]

# ---- handle: the demo-replay layout ----------------------------------------
# A long, low box (top at 0.801 m) with the strip along its long axis and a
# short tab pointing at the peel arm. The box is
# placed so the handle tip lands on a point that serves both grasp styles: from
# there the left arm solves a top-down grasp to 0.4 mm and a 20-degree
# down-tilted horizontal pinch to about a millimetre at every waypoint.
HANDLE_BOX_HALF = (0.0413, 0.1113, 0.0365)
HANDLE_TAPE_LEN, HANDLE_TAPE_WIDTH, HANDLE_TAB = 0.18, 0.025, 0.04
HANDLE_CELLS = (18, 2)
HANDLE_GRASP_XY = (-0.10, 0.20)
"""Where the handle tip lands [m], in the cell's work-surface frame (world 0.50, 0.20)."""

TAB_INSET = 0.005
"""How far past the box edge the tab selection starts [m].

Not zero, and the reason is the column that straddles the edge: it rests on the
box, it bonds, and calling it part of the free end would put bonded particles in
the set a grasp is measured against.
"""

HANDLE_BAND = 0.006
"""Depth of the pinned handle column, from the tab tip inward [m].

The reference pins everything within 6 mm of the tip, which is one grid column
at this cell size -- 0.18 m over 18 cells is 10 mm, so 6 mm takes exactly the
outermost row and nothing behind it.
"""

TAPE_COLOR = (0.80, 0.62, 0.34)
"""Display RGB of the strip: kraft packing-tape amber.

The spec default near-white read as a paper strip. Amber over the box's
cardboard tan is what a taped carton actually looks like, and the translucency
below is what makes it tape rather than ribbon.
"""

TAPE_OPACITY = 0.62
"""Display alpha of the strip. BOPP tape is translucent -- you read the box
through it -- and it is also how the peeled-back part stays legible against
the part still bonded. Display only."""

# ---- panel: the film on the phone --------------------------------------
FILM_LEN, FILM_WIDTH, FILM_TAB = 0.140, 0.060, 0.030
"""The sheet on the phone's glass: 140 of the face's 147 mm, from 3.5 mm above
the bottom edge to 3.5 mm short of the top edge, 60 of its 72 mm of width, and
a 30 mm tab running on past the top edge into the air.

The rim is not cosmetic. 60 rather than the 66 a 3 mm rim would give because
the holding hand pinches the phone's bottom-left corner across its thickness
and its pad tip lands 31.5 mm out from the centreline -- 1.5 mm outboard of a
60 mm film's edge column, so no film particle is pressed onto the glass under
that pad (a pressed particle never opens, and five of them cap the broken
fraction at 0.96 against a 0.95 gate). The 3.5 mm at either end keeps the
end rows 2 mm inside the glass slab's footprint (its bezel is 1.5 mm), where
float rounding cannot decide whether they bond.
"""

FILM_CELLS = (17, 6)
"""18 x 7 = 126 particles at a 10 mm cell along the face and across it. The
same cell the tape's calibration was taken at, which is what lets its fitted
``edge_ke`` be scaled rather than refitted."""

FILM_RIM_TOP = 0.0035
"""How far short of the phone's top edge the bonded sheet stops [m]."""

FILM_LIFT = 0.001
"""How far above the glass the sheet is authored [m], for the settle to drop
onto -- the same millimetre every layout here uses, and the one
``EntityFilm.rest_load`` measures as the standing bond load."""

FILM_COLOR = (0.78, 0.80, 0.83)
"""Display RGB of the film: clear, as the reference clip's is, and a shade
greyer than ``look.FILM_CLEAR``.

Was protective-film blue ``(0.47, 0.67, 0.87)`` -- surface-protection film is
often tinted so an operator can see which parts still carry it -- and drawing
this one the spec default near-white made it a strip of paper. The clip's film
is untinted over a glossy black face, and what makes the peel read there is
the film's creased highlights against the glass, which the crinkle map of
:data:`~look.CLEAR_FILM` carries for the path tracer. Under that tracer a
translucent sheet scatters the lab's light rather than alpha-blending it, so a
near-white tint at any opacity reads as a frosted slab over the glass
(measured 2026-09-10: face 193/255 at 0.35, 178 at 0.18, against the clip's
~60 with bright creases); the greyer tint is what keeps the haze down.
"""

FILM_OPACITY = 0.10
"""Display alpha of the film. A 60 um PE film is a thing you see the substrate
through; opaque was the single biggest reason the scene read as paper-on-wood.
Display only, like the colour. Was 0.35 on the black panel block."""

FILM_FINISH = replace(
    CLEAR_FILM,
    opacity=FILM_OPACITY,
    ior=1.1,
    texture="rgbs_crinkle",
    uv_scale=(20.0, 20.0),
)
"""The film's finish for the path tracer, and the three departures from
:data:`~look.CLEAR_FILM` are all display compensations for one missing feature.

**What the clip shows at the photo camera's grazing angle** is a black phone
seen THROUGH a bright, visibly crinkled film: mean 125/255 over the face with a
spread of 52 and highlights at 229. What ``CLEAR_FILM`` renders there is one
flat light-grey slab: 166 with a spread of **1.4**. Measured over nine
settings on this scene (2026-09-11, the photo view's face patch):

===========================================  ======  =====
setting                                        mean    std
===========================================  ======  =====
opaque (opacity 1.0)                            216    7.2
shipped CLEAR_FILM (opacity 0.35 -> 0.10)       166    1.4
+ roughness 0.55, creases 3x finer              166    1.8
+ a dark (0.20) tint                            156    1.7
ior 1.2                                         107    5.4
ior 1.05                                         83    7.5
ior 1.05 + the crinkle as a DIFFUSE map          39   15.7
**ior 1.1 + diffuse crinkle at 20/m (this)**   **48** **15**
===========================================  ======  =====

The reading: the face's brightness there is **Fresnel specular** off a smooth
dielectric under the lab's big soft dome, and nothing about the film's own
albedo, opacity or roughness touches it -- which is why the first four rows are
one number. A real crinkled sheet breaks that reflection because its wrinkles
are steep facets pointing every way; ``Finish`` has no normal map, so this
renderer cannot say that. So the sheet is given a weaker mirror (``ior`` 1.1
against polyethylene's real 1.51) and the crinkle is carried as a **diffuse**
pattern instead of a roughness one -- ``rgbs_crinkle`` is the same registered
map either way, at 20 repeats per metre, which puts its ridge network at the
5-6 mm of the clip's air bubbles (the UVs a deformable is drawn with are in
METRES along its principal axes, ``core.render.uv.planar_uvs``).

Two consequences worth stating. A bound diffuse texture REPLACES the colour for
this backend, so :data:`FILM_COLOR` now paints the film for the tiled and GL
backends only -- which is where a policy's observations come from, and there a
clear film over black glass should stay legible. And the roughness scalar was
already inert: ``rtx.py`` connects a ``roughness_texture`` to the roughness
input outright, so every ``CLEAR_FILM`` roughness number is the map's 0.12-0.72
and never the 0.15 beside it."""

AIM_RISE = 0.02
"""How far above the substrate's top face the rig is aimed [m].

The bond line is on the top face and the tab lifts off it, so the aim goes a
little above the surface rather than on it -- which is where the peel front
actually is once the haul starts.
"""


# ---------------------------------------------------------------------------
# Where things stand, derived once.
# ---------------------------------------------------------------------------

HANDLE_BOX_CENTER = (
    HANDLE_GRASP_XY[0],
    # Push the box away along the strip so the handle tip lands on the grasp
    # point -- the reference's recipe, mirrored for the +y (left) arm.
    HANDLE_GRASP_XY[1] - HANDLE_BOX_HALF[1] - HANDLE_TAB / 1.2,
    HANDLE_BOX_HALF[2],
)
"""The handle box [m, surface frame], placed so the tab tip lands on the grasp point."""

PHONE_FRAME: PhoneFrame = phone_frame(PHONE_FOOT, PHONE_TILT_DEG)
"""The phone as placed: its axes, its glass, the points the film is laid from."""

HANDLE_AIM = (HANDLE_BOX_CENTER[0], HANDLE_BOX_CENTER[1], HANDLE_BOX_CENTER[2] + HANDLE_BOX_HALF[2] + AIM_RISE)
"""What the rig looks at on the handle layout [m, surface frame]: just above
the box top, where the bond line is and where the peel front travels."""

PANEL_AIM = tuple(float(v) for v in PHONE_FRAME.face_centre)
"""What the rig looks at on the phone layout [m, surface frame]: the centre
of the glass, in the face's own plane -- the bond line lies on it and the
peel front travels down it."""


# ---------------------------------------------------------------------------
# Selectors: resolved on the REST layout, so they stay the tab wherever the
# sheet later goes -- which is the whole reason particle sets are recorded flat.
# ---------------------------------------------------------------------------


def _tab_selector(axis: int, threshold: float):
    """Everything hanging past the substrate's tab-side edge, in the sheet's own frame."""

    def select(rest: np.ndarray) -> np.ndarray:
        return rest[:, axis] > threshold

    return select


def _handle_selector(axis: int, band: float = HANDLE_BAND):
    """The tab's outermost column: everything within ``band`` of the free end."""

    def select(rest: np.ndarray) -> np.ndarray:
        return rest[:, axis] > rest[:, axis].max() - band

    return select


def _along_face(frame: PhoneFrame, rest: np.ndarray) -> np.ndarray:
    """How far up the phone's face each rest position lies [m], from the bottom
    edge. The selectors see the work-surface frame, which is the frame the
    phone's foot is stated in, so this is a dot product and nothing else."""
    return (rest - frame.foot) @ frame.u


def _tab_selector_on_face(frame: PhoneFrame, threshold: float):
    """Everything past the phone's top edge, measured up the tilted face."""

    def select(rest: np.ndarray) -> np.ndarray:
        return _along_face(frame, rest) > threshold

    return select


def _handle_selector_on_face(frame: PhoneFrame, band: float = HANDLE_BAND):
    """The tab's outermost row up the face: everything within ``band`` of the free end."""

    def select(rest: np.ndarray) -> np.ndarray:
        along = _along_face(frame, rest)
        return along > along.max() - band

    return select


# ---------------------------------------------------------------------------
# The factories. Zero-argument, because a document names a type and nothing
# else; each returns the spec under the name the documents use.
# ---------------------------------------------------------------------------


def _box(name: str, center, half) -> StaticSpec:
    """The box the tape is stuck to, in the cell's work-surface frame.

    The table it stands on is the cell's. This is the task's, because without
    something for the strip to stick to there is no peel.
    """
    return StaticSpec(
        name=name,
        half_extents=tuple(float(v) for v in half),
        xform=(*center, 0.0, 0.0, 0.0, 1.0),
        color=SUBSTRATE,
        finish=KRAFT,
    )


def handle_box() -> StaticSpec:
    """The demo-replay layout's longer, lower box."""
    return _box("box", HANDLE_BOX_CENTER, HANDLE_BOX_HALF)


def phone() -> SpecPhone:
    """The phone the film is stuck to, leaning in its cradle.

    Its ``xform`` is the body centre with the tilt, which is the landmark a
    framing check reads and the pose ``handles.entity("phone")`` reports; the
    shapes are built from the foot and the tilt by the entity.
    """
    frame = PHONE_FRAME
    return SpecPhone(
        foot=PHONE_FOOT,
        tilt_deg=PHONE_TILT_DEG,
        xform=(*(float(v) for v in frame.centre), *frame.quat),
    )


def phone_cradle() -> SpecPhoneCradle:
    """The stand under it: the block the bottom edge stands on, and its angled rest."""
    return SpecPhoneCradle(
        foot=PHONE_FOOT,
        tilt_deg=PHONE_TILT_DEG,
        xform=(*cradle_block_centre(PHONE_FOOT), 0.0, 0.0, 0.0, 1.0),
    )


def handle_tape() -> SpecTape:
    """The demo-replay strip: along the box's y axis, with its rigid pull-tab."""
    box_center = HANDLE_BOX_CENTER
    box_top = box_center[2] + HANDLE_BOX_HALF[2]
    box_y1 = box_center[1] + HANDLE_BOX_HALF[1]
    # The grid's local +x maps to world +y under a +90 degree rotation about z,
    # so the strip runs along y with its tab overhanging the +y edge.
    origin = (
        box_center[0] + HANDLE_TAPE_WIDTH / 2.0,
        box_y1 + HANDLE_TAB - HANDLE_TAPE_LEN,
        box_top + 0.001,
    )
    s = float(np.sqrt(0.5))
    return SpecTape(
        name="tape",
        mesh="grid",
        size=(HANDLE_TAPE_LEN, HANDLE_TAPE_WIDTH),
        cells=HANDLE_CELLS,
        pose=origin,
        quat=(0.0, 0.0, s, s),
        surface_center=box_center,
        surface_half=HANDLE_BOX_HALF,
        peel_back_dir=(0.0, -1.0, 0.0),
        tab_axis=1,
        handle_set="handle",
        handle_pin_step=0.0006,
        particle_sets=(
            ("tab", _tab_selector(1, box_y1 + TAB_INSET)),
            ("handle", _handle_selector(1)),
        ),
        settle_frames=60,
        settle_substeps=20,
        color=TAPE_COLOR,
        opacity=TAPE_OPACITY,
        finish=TAPE,
    )


def panel_film() -> SpecFilm:
    """The film laid up the phone's glass, its tab running on past the top edge.

    Laid in the phone's own frame: the grid's local +x is the phone's ``u``
    (up the face) and its local +z the face normal, so the sheet starts
    ``FILM_LIFT`` off the glass and the settle drops it on. The bond field's
    box is the glass slab itself, at the phone's tilt, and the peel runs back
    down the face -- ``peel_back_dir`` is ``-u`` -- with the rows counted along
    the face (``tab_axis`` 0 is the surface frame's x, which is ``u``).
    """
    frame = PHONE_FRAME
    length = float(PHONE_SIZE[0])
    bottom = length - FILM_RIM_TOP - FILM_LEN
    origin = frame.on_face(bottom, -FILM_WIDTH / 2.0, FILM_LIFT)
    return SpecFilm(
        name="film",
        mesh="grid",
        size=(FILM_LEN + FILM_TAB, FILM_WIDTH),
        cells=FILM_CELLS,
        pose=tuple(float(v) for v in origin),
        quat=frame.quat,
        color=FILM_COLOR,
        opacity=FILM_OPACITY,
        finish=FILM_FINISH,
        surface_center=tuple(float(v) for v in frame.glass_centre),
        surface_half=frame.glass_half,
        surface_quat=frame.quat,
        peel_back_dir=tuple(float(v) for v in -frame.u),
        tab_axis=0,
        handle_set="handle",
        particle_sets=(
            ("tab", _tab_selector_on_face(frame, length + TAB_INSET)),
            ("handle", _handle_selector_on_face(frame)),
        ),
        settle_frames=60,
        settle_substeps=20,
    )


# ---------------------------------------------------------------------------
# The records: the material declarations, placed. Each is registered AFTER the
# factory it names, and each keeps the declaration's home so its provenance
# reads as the tape's or the film's rather than as this module's.
# ---------------------------------------------------------------------------

_HANDLE_TAB_RECIPE = FeatureSpec(
    name="tab",
    kind="set",
    tier="recipe",
    payload={"select": {"op": "extreme", "axis": "y", "band": HANDLE_TAB - TAB_INSET, "high": True}},
    provenance=(
        "The free end hanging past the box edge on the HANDLE layout: 35 mm back from the "
        "strip's own far end along world y -- the 40 mm overhang less the 5 mm inset that keeps "
        "the column straddling the box edge out of it. The reference record's recipe reads "
        "along x with a 75 mm band and says itself that a strip laid along y with a 40 mm tab "
        "cannot share it; this is that layout's own statement of the same selection, and the "
        "spec's particle_sets are what actually select."
    ),
)
"""The handle layout's tab, as the reference record states its own."""

TAPE_HANDLE = register_asset(
    replace(
        TAPE_RECORD,
        name="tape_handle",
        summary=(
            "The adhesive strip on the demo-replay layout: 18 x 2.5 cm along a longer, lower box's "
            "y axis, with a 4 cm pull-tab pinned rigid at box-top height where a hand can take it."
        ),
        features=tuple(_HANDLE_TAB_RECIPE if f.name == "tab" else f for f in TAPE_RECORD.features),
        build=handle_tape,
    )
)

FILM_PHONE = register_asset(
    replace(
        FILM,
        name="film_phone",
        summary=(
            "The protective film on the phone's glass: 14 x 6 cm up a face leaning back 30 degrees, "
            "with a 3 cm tab running on past the raised top edge, pinned rigid in the air until a hand takes it."
        ),
        build=panel_film,
    )
)

PEEL_BOX_HANDLE = register_asset(
    AssetRecord(
        name="peel_box_handle",
        summary=(
            "The 83 x 223 x 73 mm box the pull-tab strip is stuck to, welded to the bench where its "
            "handle tip lands on the measured grasp point."
        ),
        tags=("box", "carton", "substrate", "fixture"),
        sim=SimRequirements(domain="static", contacts=contacts_with("deformables", "robot")),
        provenance=(
            "The demo-replay layout's box: longer and lower than the reference's, placed by the "
            "reference's own recipe so the tab tip lands where the left arm solves both grasp styles."
        ),
        build=handle_box,
    )
)

PHONE = register_asset(
    AssetRecord(
        name="phone",
        summary=(
            "A 147 x 72 x 8 mm phone -- black glass face in a pale silver frame -- welded leaning "
            "back 30 degrees in a cradle at the bench's centreline, bottom edge toward the arms."
        ),
        tags=("phone", "substrate", "fixture", "glass"),
        features=(
            FeatureSpec(
                name="face",
                kind="face",
                payload={
                    "center": [float(v) for v in PHONE_FRAME.glass_centre],
                    "half": [float(v) for v in PHONE_FRAME.glass_half],
                    "quat": [float(v) for v in PHONE_FRAME.quat],
                },
                provenance=(
                    "The glass slab, as the box the film's bond field is built on: its +z face is "
                    "the bond plane and the axis a bond opens along. Stated in the work-surface "
                    "frame, the frame every peel constant is in."
                ),
            ),
        ),
        sim=SimRequirements(domain="static", contacts=contacts_with("deformables", "robot")),
        provenance=(
            "Measured off sample_videos/film peeling.mp4 (phone.py has the numbers); placed where "
            "both grasps and the whole peel arc solve to 0.0 mm on yam_i2rt, which is a little "
            "further from the arms than the clip's."
        ),
        build=phone,
    )
)

PHONE_CRADLE = register_asset(
    AssetRecord(
        name="phone_cradle",
        summary=(
            "The white printed stand the phone leans in: a 42 mm-wide block with an angled rest, "
            "welded under the phone, narrow enough that the holding hand's pad clears it at the corner."
        ),
        tags=("cradle", "stand", "fixture"),
        sim=SimRequirements(domain="static", contacts=contacts_with("deformables", "robot")),
        provenance=(
            "The clip's block is about 58 x 40 x 25 mm; this one is 42 wide because the pad that goes "
            "under the phone's corner starts 31.5 mm from the centreline (phone.py)."
        ),
        build=phone_cradle,
    )
)

# ---------------------------------------------------------------------------
# The film variants' arrangement: the tape suite's, with a contact gap.
# ---------------------------------------------------------------------------

PHONE_PHYSICS = register_arrangement(
    "cloth_vbd_bimanual@v1",
    # ``cloth_vbd@ref`` -- the same 60 Hz over 20 substeps, the same 80 VBD
    # iterations, MuJoCo owning the arms, one-way coupling: every number that
    # is a stability floor is the reference's -- with ONE change, and it is
    # the difference between a two-handed take and a dead process.
    #
    # Newton's default contact gap is 0.1 m (``ModelBuilder.rigid_gap``), and
    # the reference arrangement leaves it there: every rigid pair within ten
    # centimetres is a stored contact, and every stored contact takes one of
    # MuJoCo's ``nconmax`` = 2048 slots. A YAM fingertip is 50-53 decomposed
    # hull pieces. Measured on the phone layout, 2026-09-10: the left hand
    # closed on the phone's corner stored 313 pairs; the right hand arriving
    # at the tab 15 cm away added 609 hand-to-hand pairs before it touched
    # anything, 1253 in all by the pinch and 5900 with the jaws shut -- past
    # 2048 the overflow is an illegal memory access, not a dropped contact,
    # and the run dies mid-peel. Raising the ceiling to 8192 only moved the
    # death to the first peel leg. At a 2 mm gap -- the cable tunnel's own --
    # the same hold stores 4 pairs. ``handle`` keeps the reference
    # arrangement untouched; one hand on one box never reaches the ceiling.
    replace(CLOTH_VBD, collision=replace(CLOTH_VBD.collision, rigid_gap=0.002)),
)
"""``(:physics cloth_vbd_bimanual@v1)``: the reference cloth arrangement for a
scene with two hands over one small fixture."""

# ---------------------------------------------------------------------------
# The suite's rig, aimed at each layout. One geometry, three aims -- each a
# constant a document stages as an artifact, since a rig aimed at a layout is
# the layout's arithmetic and not something a versioned registration holds.
# ---------------------------------------------------------------------------

HANDLE_VIEWS = rgbs_views(HANDLE_AIM)
"""``(:cameras assets.peel.layout:HANDLE_VIEWS)``: the rig, just above the handle box."""

PANEL_VIEWS = rgbs_views(PANEL_AIM)
"""``(:cameras assets.peel.layout:PANEL_VIEWS)``: the rig, aimed at the glass.

Its own aim now: the phone leans at the centreline where the clip's does, so
this rig no longer shares the handle layout's point -- the film's face centre
is what the photo and inset views are framed on when they are put beside the
clip.
"""
