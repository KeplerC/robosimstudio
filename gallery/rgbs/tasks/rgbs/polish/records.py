# SPDX-License-Identifier: Apache-2.0
"""The polish suite's declarations: what exists, what it exposes, what may be
said about it.

Seven records. The strip is the one manipulable body; the mat, the rest, its
two rails and the machine are fixtures; the mark field is the strip's *state*,
declared as its own record because everything a goal says about polishing
resolves through its published states (the peel tape's pattern -- the field
publishes, the document names).

Every record names its ``build`` -- the zero-argument factory in ``layout.py``
that returns it placed on this suite's bench. The mat is ``polish_mat`` rather
than a ``work_mat`` shared with the other families, and the rails are two
records rather than one, for the same reason: a record that builds a thing
builds it in one place.
"""

from __future__ import annotations

from robosimstudio.assetlib import AssetRecord, register_asset
from robosimstudio.assetlib.semantics import FeatureSpec, PublishedState, StateSpec
from robosimstudio.specs import SimRequirements, contacts_with

from .layout import (
    STRIP_HALF,
    parts_rest,
    polish_machine,
    polish_marks,
    polish_mat,
    polish_rail_far,
    polish_rail_near,
    polish_strip,
)
from .machine import EntityPolishMachine  # noqa: F401  -- registers the entity kind
from .marks import SPOT_DONE, EntityPolishMarks  # noqa: F401  -- registers the entity kind

POLISH_STRIP = register_asset(
    AssetRecord(
        name="polish_strip",
        summary="A 150 x 25 x 4 mm rigid white plastic strip with blue marks on its top face.",
        tags=("strip", "plastic", "workpiece"),
        features=(
            FeatureSpec(
                name="body",
                kind="grasp_box",
                payload={"half": [float(v) for v in STRIP_HALF], "center_local": [0.0, 0.0, 0.0]},
                provenance=(
                    "The procedural box itself: the jaws close across the 25 mm width, and the "
                    "grip measure reads that width off this declaration rather than the mesh."
                ),
            ),
        ),
        sim=SimRequirements(domain="rigid", contacts=contacts_with("surface", "fixtures", "robot")),
        build=polish_strip,
    )
)

POLISH_MAT = register_asset(
    AssetRecord(
        name="polish_mat",
        summary="The rubber work mat the rest stands on, in front of the machine.",
        tags=("mat", "fixture"),
        sim=SimRequirements(domain="static", contacts=contacts_with("objects", "robot")),
        build=polish_mat,
    )
)

PARTS_REST = register_asset(
    AssetRecord(
        name="parts_rest",
        summary=("A low block the strip lies across, holding its ends clear of the mat so a hand can take it end-on."),
        tags=("rest", "cradle", "fixture"),
        sim=SimRequirements(domain="static", contacts=contacts_with("objects", "robot")),
        build=parts_rest,
    )
)

POLISH_RAIL_NEAR = register_asset(
    AssetRecord(
        name="polish_rail_near",
        summary="The locating rail on the -x side of the strip: 12 mm tall, 8 mm proud of the strip's top face.",
        tags=("rail", "fixture"),
        sim=SimRequirements(domain="static", contacts=contacts_with("objects", "robot")),
        provenance="Measured: at 5 mm the entering pads walked the strip up onto it; at 12 mm it cannot climb out.",
        build=polish_rail_near,
    )
)

POLISH_RAIL_FAR = register_asset(
    AssetRecord(
        name="polish_rail_far",
        summary="The locating rail on the +x side of the strip, the near rail's mirror.",
        tags=("rail", "fixture"),
        sim=SimRequirements(domain="static", contacts=contacts_with("objects", "robot")),
        provenance="The near rail's twin, 28 mm across from it on a 25 mm strip.",
        build=polish_rail_far,
    )
)

ROTARY_POLISHER = register_asset(
    AssetRecord(
        name="rotary_polisher",
        summary=(
            "A rotary tool clamped SIDEWAYS to a stand, nose forward and fifteen degrees "
            "down, spinning a 25 mm polishing drum that is worked on its barrel; its column "
            "carries a small rest ledge."
        ),
        tags=("polisher", "rotary_tool", "machine", "fixture"),
        states=(
            StateSpec(
                name="spinning",
                measure="disc_speed",
                op=">=",
                value=24.0,
                summary="The drum is at working speed: within 20% of the commanded 30 rad/s.",
                provenance=(
                    "80% of the commanded speed. The velocity servo droops ~4% under a firm "
                    "press and re-spins from rest in ~0.15 s after a reset; 24 rad/s is below "
                    "neither transient and well above a stalled or unpowered drum. It read "
                    "48.0 -- 80% of a 60 rad/s command -- for as long as the drum was 13 mm; "
                    "the spindle was halved to 30 rad/s when the drum was enlarged and this "
                    "was not carried with it, leaving a state that could never be true. "
                    "Nothing on the shipped path read it (the graph routes on "
                    "sim.get_polish_state's own `spinning`, computed from the live spec), "
                    "which is exactly why it survived."
                ),
            ),
        ),
        published=(
            PublishedState(
                name="disc_speed",
                type="float",
                summary="Spindle speed [rad/s], signed along the spindle axis.",
                tier="measurable",
                unit="rad/s",
            ),
            PublishedState(
                name="disc_rim_speed",
                type="float",
                summary="Surface speed at the drum's rim [m/s].",
                tier="internal",
                unit="m/s",
            ),
            PublishedState(
                name="at_speed",
                type="bool",
                summary="Within 20% of the commanded spindle speed.",
                tier="measurable",
            ),
        ),
        sim=SimRequirements(domain="rigid", contacts=contacts_with("objects", "robot", "surface")),
        build=polish_machine,
    )
)

POLISH_MARKFIELD = register_asset(
    AssetRecord(
        name="polish_markfield",
        summary=(
            "The blue marks on the strip's top face, as an irreversible per-cell Preston-law "
            "removal field: cells lose intensity under pressure times sliding speed at the "
            "spinning bit, and never regain it."
        ),
        tags=("marks", "stain", "field"),
        states=(
            StateSpec(
                name="marked",
                measure="polished_fraction",
                op="<=",
                value=0.05,
                summary="Essentially untouched: no more than a twentieth of the mark cells cleared.",
                provenance="Designed here; the init-state mirror of `polished`.",
            ),
            StateSpec(
                name="polished",
                measure="polished_fraction",
                op=">=",
                value=0.95,
                summary="Essentially gone: at least 95% of the mark cells cleared.",
                provenance=(
                    "0.95 rather than 1.0 for peel's reason: rim cells of a spot can sit just "
                    "inside the removal threshold while the mark is plainly gone, and waiting "
                    "for them measures patience."
                ),
            ),
        ),
        published=(
            PublishedState(
                name="polished_fraction",
                type="float",
                summary="Share of mark cells cleared, over every spot.",
                tier="perceivable",
                range=(0.0, 1.0),
            ),
            PublishedState(
                name="spots_remaining",
                type="float",
                summary=f"How many mark spots are not yet {SPOT_DONE:.0%} cleared.",
                tier="perceivable",
            ),
            PublishedState(
                name="spot0_cleared",
                type="float",
                summary="Cleared fraction of spot 0 (toward the strip's -x end).",
                tier="perceivable",
                range=(0.0, 1.0),
            ),
            PublishedState(
                name="spot1_cleared",
                type="float",
                summary="Cleared fraction of spot 1 (the strip's center).",
                tier="perceivable",
                range=(0.0, 1.0),
            ),
            PublishedState(
                name="spot2_cleared",
                type="float",
                summary="Cleared fraction of spot 2 (toward the strip's +x end).",
                tier="perceivable",
                range=(0.0, 1.0),
            ),
            PublishedState(
                name="press_force",
                type="float",
                summary="Smoothed strip-against-bit normal force [N] (10-step mean of nonzero).",
                tier="internal",
                unit="N",
            ),
            PublishedState(
                name="strip_on_bit",
                type="bool",
                summary="The smoothed press force is above the removal gate.",
                tier="measurable",
            ),
        ),
        build=polish_marks,
    )
)
