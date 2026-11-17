# SPDX-License-Identifier: Apache-2.0
"""The box_decant suite's declarations: what exists, and what may be said of it.

Five records. The carton, the cutter and the packed part are the manipulable
bodies; the seam is the tape -- two strips and the field on them -- declared as
its own record because everything a goal says about cutting resolves through
its published states (peel's tape and polish's mark field, same pattern -- the
field publishes, the document names). The mat is the one fixture, declared
rather than left untyped because the goal names a contact against it twice: the
carton at rest on it, and the knife put back down on it.

Every record names its ``build`` -- the zero-argument factory in ``layout.py``
that returns it placed on this suite's bench -- so a document that types
``carton - shipping_carton`` gets the carton where the reach sweep put it. The
mat is ``decant_mat`` rather than a ``work_mat`` shared with the other
families: three families each stand a mat somewhere different, and a record
that builds one cannot be three placements under one name.
"""

from __future__ import annotations

from robosimstudio.assetlib import AssetRecord, register_asset
from robosimstudio.assetlib.semantics import FeatureSpec, PublishedState, StateSpec
from robosimstudio.specs import SimRequirements, contacts_with

from .carton import SpecShippingCarton, carton_frame
from .knife import SpecBoxCutter, blade_edge_local
from .layout import CONTENTS_HALF, box_cutter, carton_contents, decant_mat, seal_tape, shipping_carton
from .seam import EntitySealTape, SpecSealTape  # noqa: F401 -- registers the entity

_CARTON = SpecShippingCarton(name="carton")
_FRAME = carton_frame(_CARTON)
_KNIFE = SpecBoxCutter(name="knife")
_EDGE_P0, _EDGE_DIR, _EDGE_LEN = blade_edge_local(_KNIFE)
_TAPE = SpecSealTape()


def _mm(v: float) -> str:
    return f"{v * 1e3:.0f}"


SHIPPING_CARTON = register_asset(
    AssetRecord(
        name="shipping_carton",
        summary=(
            f"A {_mm(_CARTON.outer[0])} x {_mm(_CARTON.outer[1])} x {_mm(_CARTON.outer[2])} mm corrugated "
            f"parcel carton, two top flaps meeting across a {_mm(_CARTON.seam_gap)} mm join along its "
            f"{_mm(_FRAME['seam_len'])} mm long axis, the middle {_mm(_TAPE.tape_len)} mm of the join taped."
        ),
        tags=("carton", "box", "parcel", "container"),
        features=(
            FeatureSpec(
                name="body",
                kind="grasp_box",
                payload={
                    "half": [
                        float(v) for v in (0.5 * _CARTON.outer[0], 0.5 * _CARTON.outer[1], 0.5 * _CARTON.outer[2])
                    ],
                    "center_local": [0.0, 0.0, 0.5 * float(_CARTON.outer[2]) - 0.5 * float(_CARTON.wall)],
                },
                provenance=(
                    "The carton's outer box, read off `SpecShippingCarton.outer` rather than "
                    "retyped, so a grip measure answers about the walls and not about the 4 mm "
                    "floor slab the free body is built from."
                ),
            ),
            FeatureSpec(
                name="seam_line",
                kind="axis",
                payload={
                    "origin": [float(v) for v in _FRAME["seam_p0"]],
                    "direction": [float(v) for v in _FRAME["seam_dir"]],
                },
                provenance=(
                    "The line the blade is drawn along, in the carton's OWN frame so it travels "
                    "with a carton that has been shoved -- and turns with one that stands yawed, "
                    "which this one does: the layout lays the carton's local +x along world y. "
                    "Derived by `carton_frame()`, which is also what the entity builds from and "
                    "what the seam field lays its strips on. The `axis` kind carries an origin and "
                    "a direction and nothing else, so the seam's 200 mm LENGTH is not stated here "
                    "-- it is `outer[0]`, and restating it would be the second copy of a number "
                    "that already has an owner."
                ),
            ),
        ),
        # No states and no published states, and that is a MECHANISM rather than
        # a choice. `_published` in robosimstudio_bench/goals/bind.py resolves a published
        # state only out of `snapshot.extras`, and an ObjectSpec lands in the
        # `objects` bin -- so an object cannot publish one at all, and a record
        # that declared one would score every episode zero while reading
        # perfectly correctly. Everything about the carton a goal wants to name
        # is either an atom (`speed`, `contact-force`, `gripped`) or is
        # published by the seam field, which is an extra and can see the base
        # and both flaps anyway.
        sim=SimRequirements(
            domain="rigid",
            contacts=contacts_with("surface", "fixtures", "robot", "objects"),
            notes=(
                "The two top flaps are REVOLUTE LINKS this task's own entity hangs off the "
                "carton's free base -- the pattern vabar/tasks/usbc_insert/entity.py already "
                "ships and steps. They are deliberately not the `articulated` substrate, whose "
                "runtime is declared and not built. The tape's hold is a hinge torque written "
                "into control.joint_f every substep by the seam field, capped by that flap's own "
                "remaining intact fraction. The flaps' free edges stand 12 mm apart so a "
                "fingertip can be put into the join; the tape strips bridge that gap where the "
                "blade goes."
            ),
        ),
        provenance=(
            "Authored for this corpus from sample_videos/box_decanting.mp4: the parcel measured "
            "against the mat's 25 mm grid, its placement against the arms' 620 mm mount spacing, "
            "and the placement then verified reachable by the sweep on layout.py."
        ),
        build=shipping_carton,
    )
)

SEAL_TAPE = register_asset(
    AssetRecord(
        name="seal_tape",
        summary=(
            f"The {_mm(_TAPE.tape_len)} mm strip of yellow tape across the middle of the carton's top join: "
            "two rigid strips, one on each flap, meeting at the centreline, carrying an irreversible "
            "per-cell severance field -- cells part under blade pressure times draw speed, above a "
            "pierce threshold, and never re-seal."
        ),
        tags=("tape", "seal", "adhesive", "field"),
        states=(
            StateSpec(
                name="flaps_shut",
                measure="flap_open",
                op="<=",
                value=0.03,
                summary="Both flaps still down: essentially horizontal.",
                provenance=(
                    "3% of the 110-degree travel, about 3 degrees. Measured on the built carton: "
                    "a flap spawned on its `limit_lower = 0` stop and left alone reads 0.0 degrees, "
                    "and an intact seam sags 1.4 degrees under a 40 N pull -- so this admits a "
                    "taped carton being leaned on and refuses one that is opening. The "
                    "init-state mirror of `flaps_open`."
                ),
            ),
            StateSpec(
                name="flaps_open",
                measure="flap_open",
                op=">=",
                value=0.63,
                summary="Both flaps laid back: past vertical, clear of the opening.",
                provenance=(
                    "0.63 of the 110-degree travel is 69 degrees, the least a flap can lie back "
                    "and still leave a hand a clear descent into the carton. Not 90: two flaps at "
                    "exactly vertical meet each other over the seam."
                ),
            ),
            StateSpec(
                name="far_flap_open",
                measure="flap_far_open",
                op=">=",
                value=0.63,
                summary="The far flap alone laid back past vertical; the near one may still be shut.",
                provenance=(
                    "What the clip's errand ends on: the far flap hooked up and over, the near "
                    "flap left shut with its half of the tape on it. The same 69-degree floor as "
                    "`flaps_open`, on one hinge."
                ),
            ),
            StateSpec(
                name="sealed",
                measure="cut_fraction",
                op="<=",
                value=0.05,
                summary="Essentially untouched: no more than a twentieth of the tape parted.",
                provenance="Designed here; the init-state mirror of `cut`.",
            ),
            StateSpec(
                name="cut",
                measure="cut_fraction",
                op=">=",
                value=0.95,
                summary="Essentially open: at least 95% of the tape's cells parted.",
                provenance=(
                    "0.95 rather than 1.0 for peel's and polish's reason: cells at the tape's two "
                    "ends can sit just inside the sever threshold while the strip is plainly cut, "
                    "and waiting for them measures patience. It is also what refuses a policy that "
                    "cuts a fifth of the tape and then FORCES the flaps: measured, a 20%-cut seam "
                    "yields at 20 N at the flap edge, which a YAM can do."
                ),
            ),
        ),
        published=(
            PublishedState(
                name="cut_fraction",
                type="float",
                tier="perceivable",
                range=(0.0, 1.0),
                summary="Share of the tape's cells parted.",
            ),
            PublishedState(
                name="cut_run",
                type="float",
                tier="perceivable",
                range=(0.0, 1.0),
                summary=(
                    "The longest CONTIGUOUS parted run, as a share of the tape. A strip nicked in "
                    "twenty places and one drawn through once score the same fraction and are not "
                    "the same outcome; only the second opens."
                ),
            ),
            PublishedState(
                name="intact_fraction",
                type="float",
                tier="internal",
                range=(0.0, 1.0),
                summary="What is left of the tape, and the scale factor on the flaps' hold.",
            ),
            PublishedState(
                name="press_force",
                type="float",
                tier="internal",
                unit="N",
                summary="Smoothed blade-against-flaps-and-strips normal force (10-step mean of nonzero).",
            ),
            PublishedState(
                name="blade_slip",
                type="float",
                tier="measurable",
                unit="m",
                summary=(
                    "Arc length travelled in contact and UNDER the pierce gate -- the skate. "
                    "Measured at 128 mm on a full out-and-back draw at 0.69 N against a 2.0 N "
                    "gate, with a cut fraction of exactly 0.000, which is what makes a shy press "
                    "legible."
                ),
            ),
            PublishedState(
                name="blade_on_seam",
                type="bool",
                tier="measurable",
                summary="The blade is touching the tape at all. Separates a shy press from a bad aim.",
            ),
            PublishedState(
                name="near_intact",
                type="float",
                tier="internal",
                range=(0.0, 1.0),
                summary="The near flap's own remaining tape.",
            ),
            PublishedState(
                name="far_intact",
                type="float",
                tier="internal",
                range=(0.0, 1.0),
                summary="The far flap's own remaining tape.",
            ),
            PublishedState(
                name="flaps_free",
                type="float",
                tier="measurable",
                range=(0.0, 1.0),
                summary="Both flaps' tape is spent, so nothing is holding them down.",
            ),
            # The carton's own quantities, published HERE rather than on the
            # carton, because a published state resolves only out of the extras
            # bin and an ObjectSpec is not in it. See shipping_carton.
            PublishedState(
                name="flap_open",
                type="float",
                tier="perceivable",
                range=(0.0, 1.0),
                summary="Mean of the two hinge angles, as a fraction of their travel.",
            ),
            PublishedState(
                name="flap_near_open",
                type="float",
                tier="perceivable",
                range=(0.0, 1.0),
                summary="The near flap alone. One flap open and one shut is its own outcome here.",
            ),
            PublishedState(
                name="flap_far_open",
                type="float",
                tier="perceivable",
                range=(0.0, 1.0),
                summary="The far flap alone: the one the errand hooks open.",
            ),
            PublishedState(
                name="carton_shift",
                type="float",
                tier="measurable",
                unit="m",
                summary=(
                    "How far the carton has slid from where the episode settled it. Measured "
                    "unpinned over one full draw: 0.5 mm at a press under the gate, and 96 mm "
                    "at the very press that opens the tape -- there is no unpinned window."
                ),
            ),
            PublishedState(
                name="carton_yaw",
                type="float",
                tier="measurable",
                unit="rad",
                summary="How far it has turned. A carton that spun has not translated.",
            ),
        ),
        sim=SimRequirements(
            domain="rigid",
            notes=(
                "The severance kernel runs as a post-solve substep hook and the flaps' hold as a "
                "rigid-force one, both device kernels gated by device-side data so the substep "
                "block stays capturable. The press feed is a private ContactProbe over MuJoCo's "
                "own contact table, so the arrangement must run backend 'mujoco' with "
                "use_mujoco_contacts=True. The strips are rigid colliders on the flap links; the "
                "cells are non-colliding decals on the strips."
            ),
        ),
        provenance=(
            "Authored for this corpus. The law is Preston's shape with an offset in the force "
            "term; the offset is what makes the press a threshold rather than a dial. Fitted and "
            "verified by rgbs/scripts/rgbs_decant_bench.py. The strip's size and colour are the "
            "clip's, sampled at sRGB 255/252/174."
        ),
        build=seal_tape,
    )
)

BOX_CUTTER = register_asset(
    AssetRecord(
        name="box_cutter",
        summary=(
            f"A palm-sized safety cutter: a {_mm(2 * _KNIFE.body_half[0])} x {_mm(2 * _KNIFE.body_half[1])} x "
            f"{_mm(2 * _KNIFE.body_half[2])} mm slate body with a charcoal grip block and a red slider, "
            f"{_mm(_EDGE_LEN)} mm of blade out of its nose. Lies flat on the mat."
        ),
        tags=("knife", "cutter", "blade", "tool"),
        features=(
            FeatureSpec(
                name="body",
                kind="grasp_box",
                payload={
                    "half": [float(v) for v in _KNIFE.body_half],
                    "center_local": [0.0, 0.0, 0.0],
                },
                provenance=(
                    "The body box the jaws close across -- 30 mm against a 94 mm span. Read off "
                    "`SpecBoxCutter.body_half`, so the grip measure and the built shape cannot "
                    "disagree."
                ),
            ),
            FeatureSpec(
                name="blade",
                kind="blade",
                payload={
                    "origin": [float(v) for v in _EDGE_P0],
                    "direction": [float(v) for v in _EDGE_DIR],
                    "length": float(_EDGE_LEN),
                },
                provenance=(
                    "The cutting edge in the knife's own frame, from `blade_edge_local()` -- the "
                    "one place it is derived, and the domain the cut law integrates along; its "
                    "far end is the tip the tilted hold leads with. The kind is unregistered and "
                    "legal: the kind set is open, and a blade edge is a segment with a direction "
                    "rather than a rod or an axis."
                ),
            ),
        ),
        states=(
            StateSpec(
                name="held",
                measure="gripped",
                op=">=",
                value=1.0,
                summary="The jaws are stalled at the body's own width, so they are holding it.",
                provenance="The jaw-stall test against the declared 30 mm body. Nothing else here calls a knife held.",
            ),
        ),
        sim=SimRequirements(
            domain="rigid",
            contacts=contacts_with("surface", "fixtures", "robot", "objects"),
            notes=(
                "Box primitives on one free body, never a mesh: MuJoCo hulls a mesh collider, "
                "and a body-plus-blade hulls into a solid wedge with the blade's clearance "
                "filled in. The slider is a non-colliding cylinder. Sharpness is the pierce "
                "gate's, not the geometry's."
            ),
        ),
        provenance=(
            "Authored for this corpus from the safety cutter in sample_videos/box_decanting.mp4: "
            "its size against the mat's grid, its colours sampled off the frames."
        ),
        build=box_cutter,
    )
)

CARTON_CONTENTS = register_asset(
    AssetRecord(
        name="carton_contents",
        summary=(
            f"A {_mm(2 * CONTENTS_HALF[0])} x {_mm(2 * CONTENTS_HALF[1])} x {_mm(2 * CONTENTS_HALF[2])} mm "
            "packed part, the thing the carton is opened for."
        ),
        tags=("part", "contents", "workpiece"),
        features=(
            FeatureSpec(
                name="body",
                kind="grasp_box",
                payload={"half": [float(v) for v in CONTENTS_HALF], "center_local": [0.0, 0.0, 0.0]},
                provenance=(
                    "The procedural box itself: the jaws close across its 60 mm width from above. "
                    "Its HEIGHT is the load-bearing number -- standing on the carton floor it "
                    "leaves its top face 16 mm below the rim, inside a YAM fingertip's 35 mm of "
                    "usable depth, which is what makes the lift a grasp."
                ),
            ),
        ),
        sim=SimRequirements(domain="rigid", contacts=contacts_with("surface", "fixtures", "robot", "objects")),
        provenance="Authored for this corpus. Nothing is asked of it that is not an atom.",
        build=carton_contents,
    )
)

DECANT_MAT = register_asset(
    AssetRecord(
        name="decant_mat",
        summary=(
            "The gridded work mat the carton stands on and the cutter lies on, with the set-down band beside them."
        ),
        tags=("mat", "fixture"),
        sim=SimRequirements(domain="static", contacts=contacts_with("objects", "robot")),
        build=decant_mat,
    )
)
