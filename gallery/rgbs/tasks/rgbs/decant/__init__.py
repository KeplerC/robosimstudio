# SPDX-License-Identifier: Apache-2.0
"""What this task cuts, and what it cuts it with.

Three declarations and their entities. The carton owns the board and the
hinges; the seam owns the two tape strips, the severance field *and* the torque
that holds the flaps shut, because the thing that knows how much tape is left
is the thing severing it; the cutter owns the blade edge the cut law integrates
along.

Import order is the build order and it matters: ``records`` pulls the three
entity modules in so ``@register_entity`` has run before a scene names one.
"""

from __future__ import annotations

from . import carton, knife, layout, records, seam
from .carton import EntityShippingCarton, SpecShippingCarton, carton_frame
from .knife import EntityBoxCutter, SpecBoxCutter, add_blade, add_body, add_front, blade_edge_local, tip_lead
from .seam import EntitySealTape, SpecSealTape

__all__ = [
    "EntityBoxCutter",
    "EntitySealTape",
    "EntityShippingCarton",
    "SpecBoxCutter",
    "SpecSealTape",
    "SpecShippingCarton",
    "add_blade",
    "add_body",
    "add_front",
    "blade_edge_local",
    "carton",
    "carton_frame",
    "knife",
    "layout",
    "records",
    "seam",
    "tip_lead",
]
