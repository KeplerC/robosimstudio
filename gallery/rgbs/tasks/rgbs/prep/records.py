# SPDX-License-Identifier: Apache-2.0
"""The surface-prep suite's declarations: what exists and what may be said of it.

Six records. The bar is the workpiece -- an object, so it has a handle, but a
welded one -- and the sponge is the abrasive; the mat and the two blocks are
fixtures; the taint field is the bar's *state*, declared as its own record
because everything a goal says about cleaning resolves through its published
states -- polish's mark field pattern, which is peel's bond field pattern
before it.

Every record names its ``build`` -- the factory in ``layout.py``, ``bar.py`` or
``taint.py`` (the sponge's is in ``sponge.py``, beside its own record) that
returns it placed on this suite's bench. The mat is ``prep_mat`` rather than a
``work_mat`` shared with the other families, and the two blocks are two records
rather than one ``bar_block``, because a record that builds a thing builds it in
one place.

Nothing here reaches for a sibling task's copy of anything, and the sponge is
deliberately ``prep_sponge`` rather than ``sponge``: the redundancy rule accepts
a second registration under a taken name only when it says the *same thing*, and
this one is a different block of foam.
"""

from __future__ import annotations

from robosimstudio.assetlib import AssetRecord, register_asset
from robosimstudio.assetlib.semantics import PublishedState, StateSpec
from robosimstudio.specs import SimRequirements, contacts_with

from .bar import EntityGrabBar, grab_bar  # noqa: F401 -- registers the entity kind
from .layout import prep_block_far, prep_block_near, prep_mat
from .sponge import PREP_SPONGE  # noqa: F401 -- registers the sponge's own record
from .taint import PATCH_DONE, EntityPrepTaint, prep_taint  # noqa: F401 -- registers the entity kind

_STATIC = SimRequirements(domain="static", contacts=contacts_with("objects", "robot", "deformables"))

GRAB_BAR = register_asset(
    AssetRecord(
        name="grab_bar",
        summary=(
            "A 25 mm chrome grab bar: 340 mm of straight tube on two elbows whose legs drop into "
            "the blocks it is bolted to. A fixture with a handle -- the surface a wipe is scored "
            "on, and a thing the support hand can steady, but not a thing that leaves the bench."
        ),
        tags=("bar", "metal", "workpiece", "fixture"),
        features=(),
        # `deformables` is what puts the bar in reach of the foam. The sponge
        # declares the mirror side of the same rule; either alone is enough to
        # make the edge, and stating both is how the pair survives one of them
        # being edited.
        sim=SimRequirements(domain="rigid", contacts=contacts_with("surface", "fixtures", "robot", "deformables")),
        build=grab_bar,
    )
)

PREP_MAT = register_asset(
    AssetRecord(
        name="prep_mat",
        summary="The interlocking foam mat over the whole bench top, carrying the 25 mm grid.",
        tags=("mat", "fixture"),
        sim=_STATIC,
        build=prep_mat,
    )
)

PREP_BLOCK_NEAR = register_asset(
    AssetRecord(
        name="prep_block_near",
        summary="The black machined block the bar's near (working-arm) leg drops into; the sponge lies past it.",
        tags=("block", "fixture"),
        sim=_STATIC,
        build=prep_block_near,
    )
)

PREP_BLOCK_FAR = register_asset(
    AssetRecord(
        name="prep_block_far",
        summary="The block under the bar's far leg, the near block's twin; the support hand steadies the bar beside it.",
        tags=("block", "fixture"),
        sim=_STATIC,
        build=prep_block_far,
    )
)

PREP_TAINT = register_asset(
    AssetRecord(
        name="prep_taint",
        summary=(
            "The marker stripe along the bar's crown, as an irreversible per-cell Preston-law "
            "removal field: cells lose intensity under measured pressure times sliding speed where "
            "the sponge touches them, and never regain it."
        ),
        tags=("taint", "stain", "field"),
        states=(
            StateSpec(
                name="tainted",
                measure="cleaned_fraction",
                op="<=",
                value=0.05,
                summary="Essentially untouched: no more than a twentieth of the stripe's cells cleared.",
                provenance=(
                    "Designed here; the init-state mirror of `cleaned`. Safe to verify at the "
                    "reset instant because it reads the field directly -- unlike a smoothed "
                    "contact force or a jaw-stall test, both of which polish and box_decant "
                    "measured false at reset and true a dozen steps later."
                ),
            ),
            StateSpec(
                name="cleaned",
                measure="cleaned_fraction",
                op=">=",
                value=0.95,
                summary="Essentially gone: at least 95% of the stripe's cells cleared.",
                provenance=(
                    "0.95 rather than 1.0 for polish's reason: cells on the stripe's rim can sit "
                    "just inside the removal threshold while the stain is plainly gone, and "
                    "waiting for them measures patience."
                ),
            ),
        ),
        published=(
            PublishedState(
                name="cleaned_fraction",
                type="float",
                summary="Share of taint cells cleared, over the whole stripe.",
                tier="perceivable",
                range=(0.0, 1.0),
            ),
            PublishedState(
                name="patches_remaining",
                type="float",
                summary=f"How many taint patches are not yet {PATCH_DONE:.0%} cleared: one stripe, so 1.0 or 0.0.",
                tier="perceivable",
            ),
            PublishedState(
                name="patch0_cleared",
                type="float",
                summary="Cleared fraction of the stripe (patch 0, the only one: the middle 200 mm of the crown).",
                tier="perceivable",
                range=(0.0, 1.0),
            ),
            PublishedState(
                name="wipe_force",
                type="float",
                summary="Smoothed sponge-against-bar force [N] (10-step mean of nonzero).",
                tier="internal",
                unit="N",
            ),
            PublishedState(
                name="slide_speed",
                type="float",
                summary="Fastest tangential sliding speed in the live contact patch [m/s].",
                tier="internal",
                unit="m/s",
            ),
            PublishedState(
                name="sponge_on_bar",
                type="bool",
                summary="Some part of the contact patch is pressing above the removal floor.",
                tier="measurable",
            ),
            PublishedState(
                name="sponge_speed",
                type="float",
                summary=(
                    "The sponge's centroid speed [m/s], off the node positions the bridge writes back "
                    "each control step. The library's `settle-speed` reads a host-island body as "
                    "0.0 forever; this is the instrument that says the sponge has been put down."
                ),
                tier="measurable",
                unit="m/s",
            ),
        ),
        build=prep_taint,
    )
)
