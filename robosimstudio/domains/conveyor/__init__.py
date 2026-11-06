# SPDX-License-Identifier: Apache-2.0
"""The ``conveyor`` domain: a slat belt that carries things by contact.

One bundle per physical kind, as for ``static``: importing this registers the domain and the
entity that builds a belt, so a scene that names a conveyor can build one.
"""

from __future__ import annotations

from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.physics import register_domain

from .entity import ConveyorRef, ConveyorSpec, EntityConveyor, conveyor, slat_path

__all__ = ["ConveyorRef", "ConveyorSpec", "EntityConveyor", "conveyor", "slat_path"]

DOMAIN = register_domain(
    "conveyor",
    state=("bodies", "joints", "shapes"),
    reads="rigid",
    role="fixtures",
    contacts=("objects", "deformables", "robot"),
    summary="A slat conveyor: kinematic slats driven round a loop, carrying what lies on them by contact.",
)

register_entity("conveyor")(EntityConveyor)
