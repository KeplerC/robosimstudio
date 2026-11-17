# SPDX-License-Identifier: Apache-2.0
"""Where ``wipe`` stands its bar, its blocks, its mat and its sponge: the clip, measured.

The simulated twin of the corpus's *Surface Preparing* clip, and the second
suite in it whose scoring quantity is an irreversible per-cell surface field.
What the clip shows, measured off its frames and its overhead inset (grid pitch
25 mm on 300 mm interlocking mats -- the tube is one square across, the sponge
three squares long, the end blocks three wide):

* a chrome **grab bar** -- a straight tube on two 90-degree elbows whose legs
  drop into two black machined blocks -- lying across the mat between the arms;
* a **blue marker stripe** drawn along the crown of the straight run;
* a white kitchen **sponge** on the mat just past the right-hand block;
* the right YAM takes the sponge, presses it onto the bar and wipes the stripe
  off along the bar's length, while the left YAM steadies the bar at its far
  end; then the sponge goes back where it came from.

**The bar is a fixture, bolted into its blocks as the real one is.** It used to
be a free 22 mm bar with a cube on each end resting on two pads, so that the
support arm *could* take it up -- and the record of trying says one hand at one
end never got it airborne (``PROVENANCE.md``). The clip's bar is not held up by
anything but its blocks, so neither is this one: it is a ``static`` object
welded to the world, the whole press is reacted by the bar, and the foam's wrap
round the tube is what a press depth buys.

**The abrasive is foam, so the bar can be round.** A rigid pad on a cylinder
touches a line. Foam wraps it, and how far round is a function of press depth:
43 degrees at first touch, 87 at 3 mm, 126 at 8 mm, measured on the 22 mm pair
this field was calibrated on. The stripe here is a 4 mm marker line on the
crown, so a press need not reach round at all -- what it has to do is travel:
the stripe is 200 mm long and the sponge 75.

**The press cannot be read from MuJoCo.** The sponge is co-simulated by SuperDex
and has no geom in MuJoCo's contact table, so the removal law is fed by the
bridge's own per-sample contact records instead. See ``taint.py``.

Everything below is in the cell's work-surface frame unless it says world:
``bench_075`` centres at world (0.6, 0, 0.75) with half-extents (0.2975, 0.65),
the YAM bases stand at surface (-0.3475, +-0.31) facing each other across y, +x
is away from the arms and the working (right) arm is at -y.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from robosimstudio.core.render.room import LAB_MAT_GREY
from robosimstudio.specs import SimRequirements, StaticSpec, contacts_with

from ..look import BLACK, BLACK_PLASTIC, grid_finish_at
from ..staging import CELL, rgbs_views

__all__ = [
    "BAR_QUAT",
    "LAYOUT",
    "WIPE_AIM",
    "WIPE_VIEWS",
    "Layout",
    "prep_block_far",
    "prep_block_near",
    "prep_mat",
]


@dataclass(frozen=True)
class Layout:
    """Where everything stands [m, surface frame]. One object, read by everything.

    Every constant is a measurement off the clip, and the derived properties at
    the bottom are the only place the arithmetic between them is done -- the
    records, the taint field, the routine and the live page all read this and
    none of them retypes a number.
    """

    mat_thickness: float = 0.010
    """EVA puzzle mat, 10 mm, over the whole bench top."""

    tube_radius: float = 0.0125
    """A 25 mm tube: one grid square across in every frame."""

    straight_half: float = 0.170
    """Half the straight run between the elbows' tangent points: 13.5 squares
    of the 25 mm grid in the overhead inset, 340 mm."""

    bend_radius: float = 0.032
    """Centreline radius of the elbows."""

    leg_drop: float = 0.050
    """From the bar's axis down to the block's top face, at the leg: two tube
    diameters in every frame."""

    block_half: tuple[float, float, float] = (0.038, 0.040, 0.035)
    """The end blocks: 76 deep, 80 wide (along the bar), 70 tall."""

    block_inset: tuple[float, float] = (0.018, 0.020)
    """The leg enters a block's top face near its front-inner corner: this is
    the block centre's offset from the leg, (+x away from the arms, outward
    along the bar)."""

    bar_x: float = 0.0
    bar_y: float = -0.02
    """The bar's axis: x from the arms, y along the bench. Seven to eight grid
    rows in front of the arm-base brackets in the inset, once the crown's
    parallax is taken off; a shade toward the working (right, -y) arm."""

    sponge_size: tuple[float, float, float] = (0.050, 0.075, 0.030)
    """A kitchen sponge -- three squares by two in the inset -- long axis ALONG
    the bar (y)."""

    sponge_cells: tuple[int, int, int] = (9, 13, 6)
    """~5.5 mm lattice, 980 particles; the pads are 15 mm wide."""

    sponge_gap: float = 0.070
    """Clear mat between the right block's outer face and the sponge: 84 mm in
    the near-orthographic inset, 40 in the oblique phone frame; split."""

    sponge_forward: float = 0.035
    """The sponge lies a little beyond the bar's line (away from the arms): it
    sits level with the blocks' far edge in the inset."""

    stripe_half: float = 0.100
    """Half-length of the marker stripe along the crown: the middle 200 mm of
    the 340 mm run. Shorter than the photograph's stripe on purpose -- the
    working hand's far stroke has to stay clear of the hand steadying the far
    elbow, and a stroke to the end of a longer stripe shoved that hand off."""

    stripe_half_angle_deg: float = 9.0
    """Half-width of the stripe round the tube: a ~4 mm wide marker line."""

    support_u: float = 0.170
    """Where the steadying hand takes the bar: at the far elbow's tangent
    point, as far from the wipe as the round tube allows."""

    # ------------------------------------------------------------ derived
    @property
    def mat_half(self) -> tuple[float, float, float]:
        """The whole bench top, half the mat's thickness."""
        return (float(CELL.half[0]), float(CELL.half[1]), 0.5 * self.mat_thickness)

    @property
    def mat_top(self) -> float:
        return self.mat_thickness

    @property
    def block_top(self) -> float:
        return self.mat_top + 2.0 * self.block_half[2]

    @property
    def bar_z(self) -> float:
        """The bar's axis height: the blocks' top plus the leg drop.

        Derived rather than typed, and it is where the bar rests because the bar
        is welded there: a spawn that is not the resting pose puts the stripe
        somewhere the bench does not park the sponge, and every press then reads
        0.00 N -- which is how box_decant found the same bug."""
        return self.block_top + self.leg_drop

    @property
    def crown_z(self) -> float:
        return self.bar_z + self.tube_radius

    @property
    def leg_u(self) -> float:
        """Axial distance from the bar's centre to a leg's axis."""
        return self.straight_half + self.bend_radius

    def block_centre(self, sign: float) -> tuple[float, float, float]:
        """A block's centre; ``sign`` is +1 for the far (+y, support) end, -1 for the near."""
        return (
            self.bar_x + self.block_inset[0],
            self.bar_y + sign * (self.leg_u + self.block_inset[1]),
            self.mat_top + self.block_half[2],
        )

    @property
    def sponge_at(self) -> tuple[float, float, float]:
        """The sponge's bottom-face centre, past the right (-y) block."""
        _, by, _ = self.block_centre(-1.0)
        outer = by - self.block_half[1]
        return (self.bar_x + self.sponge_forward, outer - self.sponge_gap - 0.5 * self.sponge_size[1], self.mat_top)

    @property
    def aim(self) -> tuple[float, float, float]:
        """What the cameras look at: the crown of the bar, mid-run."""
        return (self.bar_x, self.bar_y, self.crown_z)


LAYOUT = Layout()

BAR_QUAT = (-0.5, -0.5, -0.5, 0.5)
"""Lays the bar along the bench's y with its crown up, xyzw.

The bar's own axis is its local z (that is where ``ShapeSpec`` puts a cylinder)
and its taint angle is measured from its local x, so this rotation has to send
**local z onto world y and local x onto world z** -- a cyclic permutation of the
axes, which is the 120-degree turn about (1, 1, 1) and has all four components
equal. It is stated rather than composed from two turns because the four halves
are exact and a product of two rotations is not, and
``tests/test_rgbs_prep.py`` checks the basis it produces rather than the
numbers, so a wrong sign fails on what it means. The legs drop along local -x,
which this sends to world -z."""

WIPE_AIM = LAYOUT.aim
"""What the rig looks at [m, surface frame]: the bar's crown, mid-run.

Derived rather than typed. The overhead cannot hold the whole scene at this
aim -- the blocks sit at 0.77 of its half-frame and the sponge past its edge at
1.18, because the clip's bar is 404 mm leg to leg and the calibrated overhead
frames 540 mm at the crown's depth -- and moving the aim toward the sponge would
move the ``inset`` view off the bar it was measured against. The ``top``,
``photo`` and ``inset`` views hold everything; the overhead holds the work.
"""

WIPE_VIEWS = rgbs_views(WIPE_AIM)
"""The suite's rig aimed at the crown: what ``wipe`` stages as
``(:cameras assets.prep.layout:WIPE_VIEWS)``."""

_STATIC_CONTACTS = SimRequirements(domain="static", contacts=contacts_with("objects", "robot", "deformables"))
"""What the mat and the blocks touch. ``deformables`` is stated on this side as
well as on the sponge's (``fixtures``): either alone makes the edge, and stating
both is how the pair survives one of them being edited."""


def prep_mat(layout: Layout = LAYOUT) -> StaticSpec:
    """The interlocking mat over the whole bench top, carrying the 25 mm grid.

    The lab scenery's own 0.5 mm grid mat is buried inside this one; the grid
    finish is the library's measured tile, so the lines here and on the scenery
    land on the same 25 mm multiples."""
    hx, hy, t = layout.mat_half
    return StaticSpec(
        name="mat",
        half_extents=(hx, hy, t),
        xform=(0.0, 0.0, t, 0.0, 0.0, 0.0, 1.0),
        friction=0.9,
        color=LAB_MAT_GREY,
        finish=grid_finish_at((0.0, 0.0)),
        sim=_STATIC_CONTACTS,
    )


def _block(which: str, layout: Layout) -> StaticSpec:
    sign = -1.0 if which == "near" else 1.0
    cx, cy, cz = layout.block_centre(sign)
    return StaticSpec(
        name=f"block_{which}",
        half_extents=layout.block_half,
        xform=(cx, cy, cz, 0.0, 0.0, 0.0, 1.0),
        friction=0.9,
        color=BLACK,
        finish=BLACK_PLASTIC,
        sim=_STATIC_CONTACTS,
    )


def prep_block_near(layout: Layout = LAYOUT) -> StaticSpec:
    """The block under the near (-y, working-arm) leg. The sponge lies past it."""
    return _block("near", layout)


def prep_block_far(layout: Layout = LAYOUT) -> StaticSpec:
    """The block under the far (+y) leg, beside which the support hand steadies the bar."""
    return _block("far", layout)


assert math.isclose(sum(v * v for v in BAR_QUAT), 1.0, abs_tol=1e-12), "BAR_QUAT is not a unit quaternion"
assert LAYOUT.leg_drop > LAYOUT.bend_radius, "the leg has no length: the elbow alone spans the drop"
assert LAYOUT.block_inset[0] + LAYOUT.tube_radius <= LAYOUT.block_half[0], "the leg misses the block's top face (x)"
assert LAYOUT.block_inset[1] + LAYOUT.tube_radius <= LAYOUT.block_half[1], "the leg misses the block's top face (y)"
assert LAYOUT.stripe_half <= LAYOUT.straight_half, "the stripe runs into the elbows"
assert LAYOUT.support_u <= LAYOUT.straight_half, "the support hold is past the straight run"
assert abs(LAYOUT.sponge_at[1]) + 0.5 * LAYOUT.sponge_size[1] <= LAYOUT.mat_half[1], "the sponge hangs off the mat"
