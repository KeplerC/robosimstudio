# SPDX-License-Identifier: Apache-2.0
"""What this suite's two things *are*, for the catalogue.

Records here, construction next door -- the split ``surface_prep`` keeps
too. The dimensions are read off the modules that build
with them rather than transcribed beside them: two copies of one number is the
drift this codebase keeps finding, and a catalogue entry that has quietly stopped
describing the thing it names is the worst place to keep the second copy.

Each record names its ``build`` -- the zero-argument factory in ``layout.py``
that returns it placed on this suite's bench -- so a document that types
``box - cable_box`` gets the box where the reach solve put it and
``cable - tunnel_cable`` gets the rod laid through its mouth.
"""

from __future__ import annotations

from dataclasses import replace

from robosimstudio_materials.rod.declare import Rod

from robosimstudio.assetlib.record import AssetRecord
from robosimstudio.assetlib.registry import register_asset
from robosimstudio.assetlib.semantics.features import FeatureSpec
from robosimstudio.assetlib.semantics.published import PublishedState
from robosimstudio.assetlib.semantics.states import StateSpec

from .box import BORE, BOX_DEPTH, BOX_HEIGHT, BOX_LENGTH, ROOF, RUN, WALL, box_frame
from .layout import BOX_SIM, cable_box, tunnel_cable
from .rod import LENGTH, RADIUS, SEGMENTS

__all__ = ["CABLE_BOX", "TUNNEL_CABLE"]

_FRAME = box_frame()

CABLE_BOX = register_asset(
    AssetRecord(
        name="cable_box",
        summary=(
            f"A {BOX_LENGTH * 1000:.0f} x {BOX_DEPTH * 1000:.0f} x {BOX_HEIGHT * 1000:.0f} mm cable-management "
            f"box with a {RUN * 1000:.0f} mm lidded channel of {BORE * 1000:.0f} mm square bore along its "
            "front-top edge, a flared entry at its right end, a rubber mat and a power strip on top."
        ),
        tags=("box", "cable box", "channel", "duct", "raceway"),
        features=(
            FeatureSpec(
                name="bore",
                kind="cavity",
                payload={
                    # The mouth plane sits on the bore axis at the entry face,
                    # and the bore runs +y from there -- which is also the
                    # direction a cable is fed, so `axis` is the insertion
                    # direction the kind asks for rather than a bare orientation.
                    "mouth": [0.0, -RUN / 2, BORE / 2],
                    "axis": [0.0, 1.0, 0.0],
                    "depth": RUN,
                    "half_extents": [BORE / 2, BORE / 2],
                },
                provenance=(
                    "Authored. Stated so the box's defining geometry lives in one place: the "
                    "document's regions are this bore written out by hand -- BDDL has no arithmetic "
                    "-- and tests/test_rgbs_tunnel.py fails when the two disagree."
                ),
            ),
            FeatureSpec(
                name="axis",
                kind="axis",
                payload={"origin": [0.0, 0.0, BORE / 2], "direction": [0.0, 1.0, 0.0]},
                provenance=(
                    "The run, declared the way cable_peg declares its winding axis -- so a measure "
                    "written about an axis reads the same on a post, a spool and a channel."
                ),
            ),
            FeatureSpec(
                name="lid",
                kind="grasp_box",
                payload={
                    "half": [BORE / 2 + WALL, RUN / 2, (BORE + ROOF) / 2],
                    "center_local": [0.0, 0.0, (BORE + ROOF) / 2],
                },
                provenance=(
                    "The lid strip and the bore walls under it, as the box the support hand pinches "
                    "across x to hold the box still: 33 mm wide, the run long, standing from the "
                    "bore floor to the lid's top. Read off box.py rather than retyped."
                ),
            ),
            FeatureSpec(
                name="ledge",
                kind="grasp_box",
                payload={
                    "half": [BORE / 2 + WALL, (_FRAME["y_left"] - _FRAME["y_exit"]) / 2, 0.0025],
                    "center_local": [0.0, (_FRAME["y_left"] + _FRAME["y_exit"]) / 2, -0.0025],
                },
                provenance=(
                    "The open part of the top-front strip past the exit, at the bore floor's level: "
                    "where the connector emerges and is drawn along. Derived by box_frame()."
                ),
            ),
        ),
        # A free body: the support hand has to hold it. The same object the
        # spec carries -- see layout.BOX_SIM for why it has to be on both.
        sim=BOX_SIM,
        provenance="Sized off channel routing.mp4; see this suite's assets/tunnel/PROVENANCE.md.",
        build=cable_box,
    )
)

# ``Rod.from_length`` derives the record and takes no factory, so the factory is
# attached to what it returns -- which is the only field of the declaration that
# is this suite's rather than the rod's.
TUNNEL_CABLE = register_asset(
    replace(
        Rod.from_length(
            name="tunnel_cable",
            length=LENGTH,
            radius=RADIUS,
            summary=(
                f"A {LENGTH:.2f} m black rod cable of {SEGMENTS} capsules with a moulded plug on its "
                "leading end, fed end-first through a cable box's channel."
            ),
            tags=("cable", "wire", "rod"),
            # Tier 0: a chain ships its own coordinate system, so a landmark is a
            # name for a number in [0, 1] and nothing is measured to state one.
            points={
                "end_a": (0.0, "The trailing end. What a hand grips and pushes."),
                "end_b": (1.0, "The fed tip. Whether THIS is past the exit is the task."),
                "midpoint": 0.5,
            },
            # `objects` is the box: the whole episode is this rod against that
            # role, and a rod that did not declare it would be fed through a
            # channel it could not feel.
            touches=("surface", "objects", "hands"),
            states=(
                StateSpec(
                    name="straight",
                    measure="straightness",
                    op=">=",
                    value=0.95,
                    summary="Laid out along its own axis rather than buckled or coiled.",
                    provenance=(
                        "cable_rod's gate, reused unchanged. It reads the same on this rod and it is "
                        "read for the opposite reason: a weave that ends straight wove nothing, and a "
                        "feed that ends straight is a feed that went in without folding."
                    ),
                ),
            ),
            published=(
                # A hold here is a kinematic coupling written into a warp array --
                # not a contact, not a pose -- so a document asking "is the hand on
                # it" has nothing to measure and everything to read.
                PublishedState(
                    name="held",
                    type="bool",
                    summary="The working hand's latch is engaged on a segment.",
                    tier="measurable",
                ),
                PublishedState(
                    name="anchor_held",
                    type="bool",
                    summary="The support hand's latch is engaged on a segment.",
                    tier="measurable",
                ),
            ),
            notes=(
                "Staged on cable_tunnel_960hz@v1: MuJoCo owns the arms and the box, rigid VBD "
                "owns the rod, and the gripper bodies and the box mirror into the VBD side as "
                "mass-scaled proxies, because MuJoCo has no rod joint and VBD should not solve an "
                "arm. The arrangement differs from the weaves' in its contact budget and only there "
                "-- a rod lying in a four-walled bore touches far more shapes at once than a rod on "
                "a spool."
            ),
            provenance=(
                "Authored for thread. The gauge and segment pitch are cable_route's weave rod; the "
                "length is this suite's, set by the run it has to cross and the table its tail "
                "has to land on."
            ),
        ),
        build=tunnel_cable,
    )
)
