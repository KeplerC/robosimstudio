# SPDX-License-Identifier: Apache-2.0
"""``thread``'s own assets: a cable box and the cable that goes through its channel.

Imported for their side effects. Holding a record has to be enough to build the
thing it describes, so the entity modules are pulled in here rather than left to
whoever happens to import first -- a declaration whose implementation may or may
not have loaded fails at scene-build time, on a different machine, for a reason
that reads as a missing entity.
"""

from __future__ import annotations

from . import box as _box  # noqa: F401 -- registers EntityCableBox
from . import plug as _plug  # noqa: F401 -- registers the tunnel_cable entity name
from . import rod as _rod  # noqa: F401 -- the rod's dimensions
from .box import (
    BORE,
    BOX_DEPTH,
    BOX_GAP,
    BOX_HEIGHT,
    BOX_LENGTH,
    FLARE,
    FLARE_ANGLE,
    MAT_TOP,
    PARTS,
    ROOF,
    RUN,
    SKINS,
    SLOT,
    WALL,
    EntityCableBox,
    SpecCableBox,
    box_frame,
    flare_box,
    flare_floor_box,
    flare_inner_face,
    flare_roof_box,
    mouth_width,
    shape_labels,
)
from .plug import PLUG_H, PLUG_L, PLUG_W, EntityTunnelCable, fits_bore, head_length
from .records import CABLE_BOX, TUNNEL_CABLE
from .rod import LENGTH, RADIUS, SEGMENTS, RodSpec

__all__ = [
    "BORE",
    "BOX_DEPTH",
    "BOX_GAP",
    "BOX_HEIGHT",
    "BOX_LENGTH",
    "CABLE_BOX",
    "FLARE",
    "FLARE_ANGLE",
    "LENGTH",
    "MAT_TOP",
    "PARTS",
    "PLUG_H",
    "PLUG_L",
    "PLUG_W",
    "RADIUS",
    "ROOF",
    "RUN",
    "SEGMENTS",
    "SKINS",
    "SLOT",
    "TUNNEL_CABLE",
    "WALL",
    "EntityCableBox",
    "EntityTunnelCable",
    "RodSpec",
    "SpecCableBox",
    "box_frame",
    "fits_bore",
    "flare_box",
    "flare_floor_box",
    "flare_inner_face",
    "flare_roof_box",
    "head_length",
    "mouth_width",
    "shape_labels",
]
