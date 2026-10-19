# SPDX-License-Identifier: Apache-2.0
"""The ``dough`` kind: a plastic solid -- dough, plasticine, clay, butter -- on Newton's implicit MPM.

This file is what ``rss.dough`` builds, imported as ``robosimstudio_materials.dough``.
Run ``examples/12_dough/00_roll_dough.py`` first. The implementation is shared
with the granular and liquid kinds in :mod:`robosimstudio_materials.mpm`; this
module is the domain, the entity that names it, the constructor and the presets.

A plastic solid here is a finite Young's modulus (5e6 Pa), a tensile yield
ratio of 1 so it holds together, and a deviatoric yield stress that decides how
it flows under a press. Measured in a press probe (``probe_c_dough``:
an 80x80x30 mm block, a 100 mm plate pressed 15 mm in, held, released; Newton
1.6.0, RTX PRO 6000): at 1e3 Pa the block spreads to 1.15x its footprint under
the plate, 1.21x after release, and keeps 83% of the compression -- the dough;
3e3 Pa spreads 1.02x and keeps 75% -- the plasticine; 1e4 Pa spreads 1.01x and
keeps 66% -- the clay. The ``Q1`` strain basis blew up on this material (a
198x footprint); the arrangements state ``P0``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.base import Vec3
from robosimstudio.specs.physics import register_domain

from .mpm import (
    BUTTER,
    CLAY,
    DOUGH,
    DOUGH_60HZ,
    PLASTICINE,
    Dough,
    EntityParticleBody,
    ParticleBodySpec,
    ParticleMaterialSpec,
    resolve_material,
)

__all__ = [
    "BUTTER",
    "CLAY",
    "DOUGH",
    "DOUGH_60HZ",
    "PLASTICINE",
    "Dough",
    "EntityDough",
    "ParticleBodySpec",
    "ParticleMaterialSpec",
    "dough",
]


class EntityDough(EntityParticleBody):
    """A plastic solid: :class:`~robosimstudio_materials.mpm.EntityParticleBody` on the ``dough`` domain."""

    domain: ClassVar[str] = "dough"


def dough(
    *,
    name: str = "dough",
    size: Vec3 = (0.08, 0.08, 0.03),
    at: Vec3 = (0.0, 0.0, 0.0),
    material: Any = "dough",
    **kwargs,
) -> ParticleBodySpec:
    """A block of plastic solid, sampled on a lattice, that holds its shape until pressed.

    Args:
        name: Key in handles, probes, measures and particle-set names.
        size: Metres, ``(x, y, z)`` of the block. The default is 12,288
            particles at the default 2.5 mm spacing; ``shape="cylinder"`` makes
            a ball-like puck of it.
        at: Where the **bottom-face centre** goes, in the cell's work-surface
            frame.
        material: ``"dough"``, ``"plasticine"``, ``"clay"``, ``"butter"``, a
            record or a :class:`~robosimstudio_materials.mpm.ParticleMaterialSpec`.
        **kwargs: Any other :class:`~robosimstudio_materials.mpm.ParticleBodySpec`
            field.

    A dough needs the ``mpm`` backend: :data:`~robosimstudio.specs.GENERIC`
    routes ``dough`` to it, and :data:`DOUGH_60HZ` puts a rolling pin on the
    robot's island so the hand can push one.
    """
    return ParticleBodySpec(entity="dough", name=name, size=size, at=at, material=resolve_material(material), **kwargs)


DOMAIN = register_domain(
    "dough",
    state=("particles",),
    reads=("dough", "granular", "cloth"),
    role="deformables",
    contacts=("surface", "fixtures", "objects", "hands", "deformables"),
    summary="A plastic solid: particles on Newton's implicit MPM with a yield stress.",
)

register_entity("dough")(EntityDough)
