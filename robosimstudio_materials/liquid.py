# SPDX-License-Identifier: Apache-2.0
"""The ``liquid`` kind: a pool of water, honey or ketchup on Newton's implicit MPM.

This file is what ``rss.liquid`` builds, imported as ``robosimstudio_materials.liquid``.
Run ``examples/03_materials/12_liquid.py`` first. The implementation is shared
with the granular and dough kinds in :mod:`robosimstudio_materials.mpm`; this
module is the domain, the entity that names it, the constructor and the presets.

Three facts about liquids in Newton 1.6.0 shape everything here, the first two
measured in a cup probe (``probe_c_water``, RTX PRO 6000), the third
in the pour example's first scene, its cup on the page's drag handle (``pour_probe``):

* **The water preset loses volume.** A 60 mm column in a thick-walled tank at
  4 mm voxels is 34% of its volume after 2 s at rest (69% with the ``P1d``
  strain basis at 1.6x the cost; 250 iterations give 41%; 240 Hz gives 52%).
  So the measures of a liquid are **counts** -- ``fill_fraction`` of a bowl --
  and a height or a volume is not a measurement of it. Newton 1.7 keeps the
  loss: a 57 mm column in a 60 mm tank stands at 23% of its height after 2 s on
  1.6.0 and 24% on 1.7.0, 99% of its particles still in the tank (the
  ``water_column`` probe, GPUs 3 and 5). Honey (viscosity 50) holds its shape
  plausibly and is what the first example pours.
* **Thin walls leak under graph capture.** The capturable ``S2`` collider basis
  lets a liquid through a wall thinner than about two voxels (3 mm walls lost
  60% in 2 s at rest); a signed-distance ``margin`` of half a voxel on every
  shape the liquid touches cut that to 5% with ``pic`` and 16% with ``S2``;
  ``pic`` cannot be captured. So a container is named in the spec's
  ``containers`` and gets that margin at build, or its walls are 8 mm or more.
  The projection granular uses is off here: it pushes a liquid into its walls
  and cost 6x.
* **What is poured falls through a floor at 60 Hz.** The band a collider is
  found in is ~1.7 voxels per MPM step, which at 4 mm and 60 Hz is a speed of
  0.36 m/s -- the limit a tool obeys -- and a liquid in free fall passes it
  7 mm below the lip it left. Honey poured from 100 mm into a bowl at 60 Hz
  put 85 of 8,000 particles through the bowl's floor and the bench and 4.0% in
  the bowl; at 120 Hz none went through and 42.5% were in the bowl (240 Hz:
  55.2%, at 1.8x the cost of 120). From 170 mm, 60 Hz put 406 through and
  120 Hz 5, with 65.5% in the bowl. So the liquid table is
  :data:`LIQUID_120HZ`, two MPM steps per control step: the first rate at
  which a pour from a hand's height lands, not one at which nothing tunnels. A
  puddle that is only pushed would be fine at 60.

A fourth, about pushing a puddle rather than pouring one, measured in the wipe
example's scene (``wipe_probe`` and ``session_wipe``, RTX PRO 6000): **what
moves a puddle is a tool held upright.** A 20 mm wiper standing on its edge
and pushed at mid-height by the page's drag handle, whose rotational spring is
soft, pitched over under the puddle's resistance and rode across it, and three
identical runs moved 22%, 24% and 51% of a 12 mm puddle off its patch; neither
a 3 mm voxel, a deeper puddle, the projection, more solver iterations nor a
stiffer yield stress made that repeatable. The same wiper pinched by the hand
pointing down, whose pads span the wiper's whole height, moved 96-98% in three
runs, and 98% of a 6 mm puddle, which is thinner than the collider band. The
grid does not see a thin puddle badly; a tilted face lets it under.

A fifth, about pouring from a hand, measured in the pour example's scene with
the YAM holding the cup (``robot_pour``, ``fixprobe.py`` and
``tests/test_pour_honey``, GPUs 1 and 6, RTX PRO 6000): **how fast honey pours
turns on a few degrees of tilt.** Tipped 123 degrees by the wrist's roll
alone, it crept along the cup's wall and about 56% was in the bowl after 12 s;
at 130, 77% after 6 s; at 140 it left the cup in one mass within a second, 99%
into the bowl. That mass lands from the lip at ~1.3 m/s, past what a 120 Hz
step's collider band catches, and 0 to 8 of the 8,000 particles of its leading
edge went through the bench in seven pours.
"""

from __future__ import annotations

import os
import tempfile
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
    LIQUID_120HZ,
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
    "LIQUID_120HZ",
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

    A liquid needs the ``mpm`` backend with the projection off and, if it is
    to be poured, stepped at 120 Hz: :data:`LIQUID_120HZ` states both, and
    puts a cup on the robot's island so the hand can tilt one.
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
        # Written beside the final name and renamed onto it, so a second
        # process building the same cup reads the whole file or none of it.
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".obj", dir=path.parent)
        os.close(fd)
        try:
            os.chmod(tmp, 0o644)  # mkstemp's 0600 would keep the cache to its writer
            trimesh.util.concatenate(slabs).export(tmp)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
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
    engine="mpm",
    staged_by="LIQUID_120HZ",
)

register_entity("liquid")(EntityLiquid)
