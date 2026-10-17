# SPDX-License-Identifier: Apache-2.0
"""The ``liquid`` kind: a pool of water, honey or ketchup on Newton's implicit MPM.

This file is what ``rss.liquid`` builds, imported as ``robosimstudio_materials.liquid``.
Run ``examples/11_liquid/00_pour_honey.py`` first. The implementation is shared
with the granular and dough kinds in :mod:`robosimstudio_materials.mpm`; this
module is the domain, the entity that names it, the constructor and the presets.

Two facts about liquids in Newton 1.6.0 shape everything here, both measured in
the Phase 0 cup probe (``probe_c_water``, RTX PRO 6000):

* **The water preset loses volume.** A 60 mm column in a thick-walled tank at
  4 mm voxels is 34% of its volume after 2 s at rest (69% with the ``P1d``
  strain basis at twice the cost; iterations do not help; 240 Hz gives 52%).
  So the measures of a liquid are **counts** -- ``fill_fraction`` of a bowl --
  and a height or a volume is not a measurement of it. Honey (viscosity 50)
  holds its shape plausibly and is what the first example pours.
* **Thin walls leak under graph capture.** The capturable ``S2`` collider basis
  lets a liquid through a wall thinner than about two voxels (3 mm walls lost
  60% in 2 s at rest); a signed-distance ``margin`` of half a voxel on every
  shape the liquid touches cut that to 5% with ``pic`` and 16% with ``S2``;
  ``pic`` cannot be captured. So a container is named in the spec's
  ``containers`` and gets that margin at build, or its walls are 8 mm or more.
  The projection granular uses is off here: it pushes a liquid into its walls
  and cost 6x.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

from robosimstudio.domains.rigid import body
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.asset import AssetSpec, ObjectSpec
from robosimstudio.specs.base import Vec3
from robosimstudio.specs.physics import register_domain

from .mpm import (
    HONEY,
    KETCHUP,
    LIQUID_60HZ,
    WATER,
    EntityParticleBody,
    Liquid,
    ParticleBodySpec,
    ParticleMaterialSpec,
    resolve_material,
)

__all__ = [
    "HONEY",
    "KETCHUP",
    "LIQUID_60HZ",
    "WATER",
    "EntityLiquid",
    "Liquid",
    "ParticleBodySpec",
    "ParticleMaterialSpec",
    "cup",
    "liquid",
    "mesh_cache_dir",
]


class EntityLiquid(EntityParticleBody):
    """A liquid body: :class:`~robosimstudio_materials.mpm.EntityParticleBody` on the ``liquid`` domain."""

    domain: ClassVar[str] = "liquid"


def liquid(
    *,
    name: str = "liquid",
    size: Vec3 = (0.06, 0.06, 0.06),
    at: Vec3 = (0.0, 0.0, 0.0),
    material: Any = "honey",
    containers: tuple[str, ...] = (),
    **kwargs,
) -> ParticleBodySpec:
    """A block of liquid, sampled on a lattice, that flows when the scene starts.

    Args:
        name: Key in handles, probes, measures and particle-set names.
        size: Metres, ``(x, y, z)`` of the block as built -- a cup's inside, a
            puddle's footprint and depth. The default is 13,824 particles.
        at: Where the **bottom-face centre** goes, in the cell's work-surface
            frame; inside a container, its floor.
        material: ``"honey"``, ``"water"``, ``"ketchup"``, a record or a
            :class:`~robosimstudio_materials.mpm.ParticleMaterialSpec`. Honey
            by default: the water preset loses volume in Newton 1.6 (see the
            module docstring).
        containers: Names of the entities this liquid is held by. Each of their
            collision shapes gets a half-voxel signed-distance margin at build,
            which is what keeps a liquid inside a wall thinner than two voxels.
        **kwargs: Any other :class:`~robosimstudio_materials.mpm.ParticleBodySpec`
            field.

    A liquid needs the ``mpm`` backend with the projection off:
    :data:`LIQUID_60HZ` states it, and puts a cup on the robot's island so the
    hand can tilt one.
    """
    return ParticleBodySpec(
        entity="liquid",
        name=name,
        size=size,
        at=at,
        material=resolve_material(material),
        containers=containers,
        **kwargs,
    )


def mesh_cache_dir() -> Path:
    """Where generated meshes go: ``$ROBOSIMSTUDIO_CACHE/meshes`` or ``~/.cache/robosimstudio/meshes``.

    The same root the procedural textures use
    (:func:`~robosimstudio.core.render.finish.texture_cache_dir`), so one
    variable moves both."""
    root = os.environ.get("ROBOSIMSTUDIO_CACHE")
    base = Path(root) if root else Path.home() / ".cache" / "robosimstudio"
    return base / "meshes"


def cup(
    *,
    name: str = "cup",
    inner: Vec3 = (0.06, 0.06, 0.08),
    wall: float = 0.008,
    at: Vec3 = (0.0, 0.0, 0.0),
    mass: float = 0.3,
    **kwargs,
) -> ObjectSpec:
    """An open-topped box a liquid can be poured from: one free rigid body, a floor and four walls.

    A hollow mesh imported as a collider is sealed by MuJoCo's convexification,
    so the cup is a generated mesh (the five slabs, written once into
    :func:`mesh_cache_dir`) with the ``"shell"`` collider, which replaces the mesh
    with a floor and four wall boxes around the declared cavity -- the one
    container shape the rigid kind can hold as a single body. Name it in the
    liquid's ``containers`` so its shapes get the half-voxel margin; the default
    wall is two voxels, which holds a liquid even without one.

    Args:
        name: The body's name; what ``containers`` and a drag handle refer to.
        inner: The cavity ``(x, y, depth)`` [m].
        wall: Floor and wall thickness [m]. 8 mm is two of the registered 4 mm
            voxels, the thickness below which the capturable collider basis
            leaks (``probe_c_water``).
        at: Where the **outer bottom-face centre** goes, in the work-surface
            frame, so ``at=(x, y, 0)`` stands the cup on the bench.
        mass: Total mass [kg], spread over the five slabs. A hand or the page's
            drag handle lifts the cup and what is in it; 0.3 kg of cup under
            0.18 kg of honey (a 50 mm cube at 1400 kg/m^3) is what the pour
            example lifts.
        **kwargs: Any other :class:`~robosimstudio.specs.ObjectSpec` field.
    """
    import trimesh

    ix, iy, iz = (float(v) for v in inner)
    t = float(wall)
    ox, oy = ix + 2.0 * t, iy + 2.0 * t
    path = mesh_cache_dir() / f"cup_{ix * 1e3:.1f}x{iy * 1e3:.1f}x{iz * 1e3:.1f}_w{t * 1e3:.1f}.obj"
    if not path.is_file():
        slabs = [
            trimesh.creation.box((ox, oy, t)).apply_translation((0.0, 0.0, 0.5 * t)),
            trimesh.creation.box((t, oy, iz)).apply_translation((-0.5 * (ix + t), 0.0, t + 0.5 * iz)),
            trimesh.creation.box((t, oy, iz)).apply_translation((0.5 * (ix + t), 0.0, t + 0.5 * iz)),
            trimesh.creation.box((ix, t, iz)).apply_translation((0.0, -0.5 * (iy + t), t + 0.5 * iz)),
            trimesh.creation.box((ix, t, iz)).apply_translation((0.0, 0.5 * (iy + t), t + 0.5 * iz)),
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        trimesh.util.concatenate(slabs).export(path)
    return body(
        name=name,
        asset=AssetSpec(path=str(path)),
        at=at,
        collider="shell",
        cavity=(-0.5 * ix, -0.5 * iy, t, 0.5 * ix, 0.5 * iy, t + iz),
        mass=float(mass),
        **kwargs,
    )


DOMAIN = register_domain(
    "liquid",
    state=("particles",),
    reads=("liquid", "granular", "cloth"),
    role="deformables",
    contacts=("surface", "fixtures", "objects", "hands", "deformables"),
    summary="A pool of liquid: particles on Newton's implicit MPM, measured by count.",
)

register_entity("liquid")(EntityLiquid)
