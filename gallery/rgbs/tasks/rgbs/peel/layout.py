# SPDX-License-Identifier: Apache-2.0
"""Where the peel variants lay their sheets: a tape strip on a box (``handle``) and a film on a phone (``panel``).

Both sheets are the library's tape (:mod:`robosimstudio_materials.tape`): a cloth held to a free
body by bonds that let go once the sheet lifts a millimetre off it, its end turned up into a rigid
pull tab the pads close on. That tab is the reason the variants were re-authored on it. The October
peel held its sheet's end with a particle latch that followed the hand and released a kinematic pin
on the tab when the latch had it; under the library's tape the hand holds the tab by squeezing it,
and nothing moves the sheet but the pads and the bonds.

What that costs is the box and the phone being free bodies. A tape sticks only to a body on its own
VBD island (``TAPE_60HZ``), so neither is welded any more: the box is a 0.8 kg carton standing on the
bench and the phone a 0.2 kg slab lying face up on a printed stand. The phone used to lean back 30
degrees in a welded cradle; a free phone leaning in a cradle is rocked off its rest by the peel's
pull, and a body that settles after it is built breaks every bond it carries (the tape builds its
bonds on the body's spawn pose), so it lies flat on the stand, where it rests exactly as it is built.

**Every constant below is stated in the cell's work-surface frame**, the frame of
:data:`~staging.CELL`. Both sheets lie along x with their tab at the far (+x) end, on the left arm's
side of the bench, and the left arm takes the tab straight down, jaws closing along the sheet, and
draws it back over the sheet toward the robot: the gesture ``examples/03_materials/08_tape``
measures, with a longer sheet. That is where the reach is. Swept with the arm's own IK over the
approach, the close and the whole pull (a 140 mm peel at 35 degrees), a straight-down hold solves
within 2 mm for a tab from x -0.12 to 0.0 at y +0.12 to +0.20, pulling toward the robot. Pulling
across the bench instead, as the October strip ran, it is 18-290 mm short for every tab swept, and
the tab slid in the pads of the 45-60-degree-pitched hold that does reach.

The rig is aimed here too, as in every family of the suite: the aim is the work's own centre lifted
over its top face, and each aimed rig is a constant a document stages as an artifact.
"""

from __future__ import annotations

from dataclasses import replace

from robosimstudio.assetlib import register_asset
from robosimstudio.core.render.finish import MATTE
from robosimstudio.domains.rigid import Body
from robosimstudio.domains.rigid.make import box
from robosimstudio_bench.taskbuild.arrangements import register_arrangement
from robosimstudio_materials.tape import PAPER_LABEL, TAPE_60HZ, Tape, tape

from ..look import CLEAR_FILM, KRAFT, PANEL_BLACK, PANEL_GLOSS, TAPE
from ..staging import rgbs_views

__all__ = [
    "BOX_AT",
    "BOX_HALF",
    "FILM",
    "HANDLE_VIEWS",
    "PANEL_VIEWS",
    "PEEL_PHYSICS",
    "PHONE_AT",
    "PHONE_HALF",
    "STRIP",
]

PEEL_PHYSICS = register_arrangement("rgbs_tape_60hz@v1", TAPE_60HZ)
"""``(:physics rgbs_tape_60hz@v1)``: the library tape's own arrangement -- ``GENERIC`` with compliant
ALM on the VBD island the sheet shares with what it is stuck to."""

# ---- handle: a tape strip along a long, low box ------------------------------

BOX_HALF = (0.1113, 0.0413, 0.0365)
"""The box [m, half-extents]: the October layout's 83 x 223 x 73 mm, its long side along x."""

BOX_AT = (-0.08, 0.16, BOX_HALF[2])
"""The box's centre [m]: the strip's tab at x 0.0, y +0.16, inside the band the left arm reaches the
whole pull from top-down."""

BOX_MASS = 0.8
"""[kg]. A packed carton rather than an empty one: the peel draws its tab up and back, and the
box's own weight on the bench is what keeps it from following."""

STRIP_SIZE = (0.025, 0.155)
"""The strip [m, (across, along)], the turned-up end included: 25 mm packing tape with 140 mm stuck
down, the October strip's bonded length (it was 180 mm with 40 mm of tab hanging past the box)."""

STRIP_CELLS = (5, 31)
"""A 5 mm cell, the one the library tape's constants were measured at."""

STRIP_INSET = 0.008
"""How far inside the box's +x edge the sheet's tab end lies [m]: the tab's 8 mm clear of the edge."""

QUARTER_TURN = (0.0, 0.0, 0.70710678, 0.70710678)
"""A sheet's own +y (from its tab) to world -x, toward the robot."""

TAB_SIZE = (0.040, 0.008, 0.024)
"""The pull tab [m]: wider than the strip, the handle the variant is named for."""

TAPE_COLOR = (0.80, 0.62, 0.34)
"""Kraft packing-tape amber over the box's tan. Display only."""

TAPE_OPACITY = 0.62
"""BOPP tape is translucent: the box reads through it. Display only."""

BOX_TOP = 2.0 * BOX_HALF[2]

STRIP = tape(
    name="tape",
    on="box",
    size=STRIP_SIZE,
    cells=STRIP_CELLS,
    # The tab end's corner, a particle radius over the box top; turned a quarter turn so the strip runs
    # toward -x (and its across axis along +y), the way the hand pulls it.
    at=(BOX_AT[0] + BOX_HALF[0] - STRIP_INSET, BOX_AT[1] - 0.5 * STRIP_SIZE[0], BOX_TOP + 0.0005),
    quat=QUARTER_TURN,
    material=PAPER_LABEL,
    tab_size=TAB_SIZE,
    color=TAPE_COLOR,
    opacity=TAPE_OPACITY,
    finish=TAPE,
)
"""The strip, stuck down along the box's top with its tab at the +x end."""

HANDLE_BOX = box(
    name="box",
    half=BOX_HALF,
    at=BOX_AT,
    mass=BOX_MASS,
    friction=0.8,
    color=(0.72, 0.56, 0.38),
    finish=KRAFT,
)

PEEL_BOX_HANDLE = register_asset(
    Body.from_spec(
        HANDLE_BOX,
        name="peel_box_handle",
        summary="The 83 x 223 x 73 mm box the strip is stuck to, free on the bench.",
        tags=("box", "carton", "substrate"),
    )
)
TAPE_HANDLE = register_asset(
    Tape.from_spec(
        STRIP,
        name="tape_handle",
        summary="A 25 x 180 mm packing-tape strip along the box's top, its end turned up into a pull tab.",
        tags=("tape", "adhesive", "strip"),
    )
)

# ---- panel: a protective film on a phone ------------------------------------

PHONE_HALF = (0.0735, 0.036, 0.004)
"""The phone [m, half-extents]: 147 x 72 x 8 mm, face up, its long side along x."""

STAND_HALF = (0.060, 0.030, 0.0325)
"""The printed stand the phone lies on [m, half-extents]: 120 x 60 x 65 mm, white, welded to the bench.
It puts the glass at the box top's height, where the arm reaches the tab straight down and the hand's
carriage clears the bench; on the bench itself the pitched hand stopped 45 mm over the tab."""

STAND_AT = (-0.07, 0.16, STAND_HALF[2])

PHONE_AT = (STAND_AT[0], STAND_AT[1], 2.0 * STAND_HALF[2] + PHONE_HALF[2])
"""The phone's centre [m]: lying on the stand, its film's tab where the strip's is."""

PHONE_MASS = 0.2
"""[kg]. A phone's."""

FILM_SIZE = (0.060, 0.155)
"""The film [m, (across, along)], the turned-up end included: 60 of the face's 72 mm across and 140 of
its 147 mm along, the tab's 15 mm on top."""

FILM_CELLS = (12, 31)

FILM_INSET = 0.008
"""How far inside the phone's top (+x) edge the film's tab end lies [m]."""

FILM_TACK = 2.0e6
"""The film's bond stiffness per square metre [N/m^3]: half the paper label's. Protective film is a
weak adhesive by design -- it comes off clean -- and it is laid over a phone a fifth the box's weight."""

FILM_COLOR = (0.78, 0.80, 0.83)
"""Clear film, a shade greyer than ``look.FILM_CLEAR`` so it does not read as frosted under the
path tracer. Display only."""

FILM_OPACITY = 0.10

FILM_FINISH = replace(CLEAR_FILM, opacity=FILM_OPACITY, ior=1.1, texture="rgbs_crinkle", uv_scale=(20.0, 20.0))
"""The October film's finish: a weak mirror and a diffuse crinkle, which is what reads as a crinkled
clear sheet over black glass under the path tracer."""

PHONE_TOP = PHONE_AT[2] + PHONE_HALF[2]

FILM = tape(
    name="film",
    on="phone",
    size=FILM_SIZE,
    cells=FILM_CELLS,
    at=(PHONE_AT[0] + PHONE_HALF[0] - FILM_INSET, PHONE_AT[1] - 0.5 * FILM_SIZE[0], PHONE_TOP + 0.0005),
    quat=QUARTER_TURN,
    tab_size=(0.062, 0.008, 0.024),
    material=replace(PAPER_LABEL, tack=FILM_TACK, areal_density=0.06, friction=0.2),
    color=FILM_COLOR,
    opacity=FILM_OPACITY,
    finish=FILM_FINISH,
)
"""The film, stuck down along the phone's face with its tab at the top (+x) edge, as wide as the film."""

PHONE_BODY = box(
    name="phone",
    half=PHONE_HALF,
    at=PHONE_AT,
    mass=PHONE_MASS,
    friction=0.8,
    color=PANEL_BLACK,
    finish=PANEL_GLOSS,
)

PHONE = register_asset(
    Body.from_spec(
        PHONE_BODY,
        name="phone",
        summary="A 147 x 72 x 8 mm phone, black glass up, free on the bench.",
        tags=("phone", "substrate", "glass"),
    )
)
PHONE_STAND = register_asset(
    Body.from_spec(
        box(name="stand", half=STAND_HALF, at=STAND_AT, static=True, color=(0.86, 0.86, 0.84), finish=MATTE),
        name="phone_stand",
        summary="The white printed stand the phone lies on, welded to the bench.",
        tags=("stand", "fixture"),
    )
)
FILM_PHONE = register_asset(
    Tape.from_spec(
        FILM,
        name="film_phone",
        summary="Protective film over the phone's glass, its top end turned up into a pull tab.",
        tags=("film", "protective", "adhesive"),
    )
)

# ---- the rig, aimed at each layout -----------------------------------------

AIM_RISE = 0.02
"""How far above the stuck face the rig is aimed [m]: the peel front runs just over it."""

HANDLE_AIM = (BOX_AT[0], BOX_AT[1], BOX_TOP + AIM_RISE)
PANEL_AIM = (PHONE_AT[0], PHONE_AT[1], PHONE_TOP + AIM_RISE)

HANDLE_VIEWS = rgbs_views(HANDLE_AIM)
"""``(:cameras peel.layout:HANDLE_VIEWS)``: the rig, just above the box top."""

PANEL_VIEWS = rgbs_views(PANEL_AIM)
"""``(:cameras peel.layout:PANEL_VIEWS)``: the rig, just above the phone's glass."""
