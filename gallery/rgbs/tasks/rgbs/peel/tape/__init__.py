# SPDX-License-Identifier: Apache-2.0
"""Adhesive tape: the asset whose physics is the thing itself.

The module this package absorbed makes the argument better than a summary can. It
introduces a discrete cohesive-zone bond model -- every tape particle resting on
the box is a penalty spring to a fixed anchor, and a spring breaks *permanently*
once the particle has lifted past a threshold -- and then says why it lives where
it does:

    This lives in the suite rather than in core or in the generic cloth entity
    because it is not general deformable physics: it is what adhesive tape *is*.
    A cloth entity that knew about bonds would be carrying one task's model into
    every scene that drapes a shirt.

Both halves of that are right, and together they name a layer that did not exist.
Bonds are not general cloth physics and they are not a fact about *peeling* -- a
second task using the same tape would need every one of them. They are a fact
about the tape, and this is where a fact about the tape goes. The fear the
original expresses is not violated by moving it here; it is *satisfied*. Bonds
attach to the tape asset, and the generic cloth entity stays clean.

So this package is one asset in three files: the declaration here,
:mod:`~rgbs/tasks/rgbs/peel/tape/adhesion.py` for the bond field and the handle pin, and
:mod:`~rgbs/tasks/rgbs/peel/tape/entity.py` for the spec and the entity that reads them.

**The published states are the tape's own instrument panel**, and their tiers are
vocabulary rather than plumbing. ``bond_fraction`` and ``broken_fraction`` are
``internal`` -- no instrument reads the force between a particle and its anchor,
so a policy may never be handed either, while an evaluator may score on them.
``peel_front`` is ``measurable``: how far the peel has travelled is something a
real rig could instrument with a camera looking along the strip, so it is a
candidate for de-privileging later. That distinction is the entire content of the
observability tier, and it only becomes statable once states are declared instead
of stuffed into a dict.
"""

from __future__ import annotations

from robosimstudio.assetlib.record import AssetRecord
from robosimstudio.assetlib.registry import register_asset
from robosimstudio.assetlib.semantics.features import FeatureSpec
from robosimstudio.assetlib.semantics.published import PublishedState
from robosimstudio.assetlib.semantics.states import StateSpec
from robosimstudio.assetlib.sim import SimRequirements, contacts_with

from .adhesion import (
    BOND_BREAK,
    BOND_MARGIN,
    BOND_STIFFNESS,
    PIN_MAX_STEP,
    AdhesionBonds,
    BondState,
    HandlePin,
)
from .entity import (
    CLEAR_HEIGHT,
    GRASP_RADIUS,
    ROW_BROKEN_FRACTION,
    TAB_SET,
    TAPE_MATERIAL,
    EntityTape,
    SpecTape,
    bond_broken_fraction,
    peel_front_progress,
)

__all__ = [
    "BOND_BREAK",
    "BOND_MARGIN",
    "BOND_STIFFNESS",
    "CLEAR_HEIGHT",
    "GRASP_RADIUS",
    "PIN_MAX_STEP",
    "ROW_BROKEN_FRACTION",
    "TAB_SET",
    "TAPE",
    "TAPE_MATERIAL",
    "AdhesionBonds",
    "BondState",
    "EntityTape",
    "HandlePin",
    "SpecTape",
    "bond_broken_fraction",
    "peel_front_progress",
]

TAB_OVERHANG = 0.075
"""How far back from the strip's free end the tab reaches [m].

The reference layout's own: 80 mm of the 260 mm strip hangs past the box edge,
and the 5 mm nearest the edge is inset so the boundary column -- which straddles
the edge and does bond -- is not called tab. Stated as a distance from the
strip's *own* end rather than as a plane in the world, which is what makes the
recipe below survive both a re-mesh and a box that moved.
"""

TAPE = register_asset(
    AssetRecord(
        name="peel_tape",
        summary="An adhesive strip bonded to a surface, which peels when pulled off it.",
        tags=("tape", "adhesive", "strip"),
        features=(
            FeatureSpec(
                name="adhesion",
                kind="bond_field",
                payload={"stiffness": BOND_STIFFNESS, "break_distance": BOND_BREAK},
                provenance=(
                    "AutoPeel's discrete cohesive-zone model (ICRA 2025), at the reference's own "
                    "stiffness. Reproduces its 133 bonds and its settled droop."
                ),
            ),
            FeatureSpec(
                name="tab",
                kind="set",
                tier="recipe",
                payload={"select": {"op": "extreme", "axis": "x", "band": TAB_OVERHANG, "high": True}},
                provenance=(
                    "The free end hanging past the box edge, on the REFERENCE LAYOUT: 75 mm back "
                    "from the strip's own far end along world x, which is the 80 mm overhang less "
                    "the 5 mm inset that keeps the column straddling the box edge out of it. "
                    "AN 'extreme' AND NOT A 'beyond': a recipe is resolved against the rest layout "
                    "in WORLD coordinates, so the threshold this used to carry -- 0.10, the box's "
                    "own half-extent, a body-frame number -- was compared against world x and "
                    "selected all 189 particles of a 56-particle tab. Every particle of a strip at "
                    "x = 0.52 to 0.78 is past 0.10 m. Measured from the strip's own far end there "
                    "is no origin left to get wrong, and the selection survives a box that moved. "
                    "WHAT IT DOES NOT SURVIVE IS ANOTHER AXIS OR ANOTHER OVERHANG, and that is a "
                    "property of the recipe language rather than an oversight here: the closed "
                    "selector set states a band along a named world axis, so a strip laid along y "
                    "with a 40 mm tab -- which is exactly tape_peel/handle -- cannot share this "
                    "one. Such a layout states its own selection in its particle_sets, and does."
                ),
            ),
            FeatureSpec(
                name="anchor",
                kind="anchor",
                payload={"release_on": "grasp"},
                provenance=(
                    "The pull-tab is held at its as-laid pose until a gripper takes it. A property of "
                    "how this tape is staged, not of the peeling task -- a drooping flap and a "
                    "presented one are different starting worlds."
                ),
            ),
        ),
        states=(
            StateSpec(
                name="adhered",
                measure="bond_fraction",
                op=">=",
                value=0.95,
                summary="Still stuck down: essentially every bond intact.",
                provenance="Designed here. A strip that has lost 5% of its bonds is being peeled, not stuck.",
            ),
            StateSpec(
                name="peeled",
                measure="bond_fraction",
                op="<=",
                value=0.1,
                summary="Off the surface: essentially every bond broken.",
                provenance=(
                    "Designed here, and deliberately not 0.0 -- a strip can be lifted clear while a "
                    "couple of bonds at the far corner never opened, and calling that unpeeled would "
                    "measure patience rather than skill."
                ),
            ),
        ),
        published=(
            PublishedState(
                name="bond_fraction",
                type="float",
                summary="Share of adhesive bonds still intact.",
                tier="internal",
                range=(0.0, 1.0),
            ),
            PublishedState(
                name="broken_fraction",
                type="float",
                summary="Share of adhesive bonds that have let go. One minus bond_fraction.",
                tier="internal",
                range=(0.0, 1.0),
                # Both directions of one number, and it is not redundancy. A peel
                # GROWS toward success, so a progress reward divides by this one;
                # "still stuck down" SHRINKS, so the adhered state thresholds the
                # other. Deriving either from the other in a document would mean
                # a goal reading `1 - x <= 0.05` where the reference reads
                # `x >= 0.95`, and those two disagree in binary floating point
                # the moment the fraction lands exactly on the gate.
            ),
            PublishedState(
                name="peel_front",
                type="float",
                summary="How far along the strip the peel front has travelled, as a share of its rows.",
                tier="measurable",
                range=(0.0, 1.0),
            ),
            PublishedState(
                name="peel_front_rows",
                type="float",
                summary="The same front, counted in fully-broken width-rows from the tab.",
                tier="measurable",
            ),
            PublishedState(
                name="bond_load",
                type="float",
                summary="Tension the intact bonds are currently carrying.",
                tier="internal",
                unit="N",
            ),
            PublishedState(
                name="anchored",
                type="bool",
                summary="Whether the pull-tab is still held at its as-laid pose.",
                tier="internal",
            ),
            PublishedState(
                name="held",
                type="bool",
                summary="Whether a gripper has hold of the tab.",
                tier="measurable",
            ),
            PublishedState(
                name="held_count",
                type="float",
                summary="How many tab particles are inside the grasp radius.",
                tier="measurable",
            ),
            PublishedState(
                name="held_ever",
                type="bool",
                summary="Whether a gripper has had hold of the tab at any point this episode.",
                tier="measurable",
            ),
            PublishedState(
                name="clear",
                type="float",
                summary="Share of the strip lifted clear of where it settled.",
                tier="measurable",
                range=(0.0, 1.0),
            ),
        ),
        sim=SimRequirements(
            domain="cloth",
            contacts=contacts_with("surface", "robot"),
            contributes_hooks=("bond_force", "handle_pin"),
            notes="The bond field runs as a pre-solve substep hook and the handle pin as a post-solve "
            "one, both as device kernels gated by device-side flags so the substep block stays "
            "capturable. The reference evaluates its bonds in numpy, round-tripping particle "
            "state to the host twice per substep -- 40 synchronises a frame at 20 substeps, and "
            "uncapturable. On device it is neither. "
            "Bonds break on OPENING displacement, not total: a servoed arm straddling the strip "
            "presses particles down and drags them along the box routinely, and tape pressed "
            "onto its surface does not unstick. Only being pulled off it peels.",
        ),
        provenance=(
            "tape_peel's strip: 133 bonds and a settled droop of 0.774 / 0.808 / 0.844 m, matching "
            "the reference implementation."
        ),
    )
)
