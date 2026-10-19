# SPDX-License-Identifier: Apache-2.0
"""The ``dough`` kind: a plastic solid -- dough, plasticine, clay, butter -- on Newton's implicit MPM.

This file is what ``rss.dough`` builds, imported as ``robosimstudio_materials.dough``.
Run ``examples/12_dough/00_roll_dough.py`` first. The implementation is shared
with the granular and liquid kinds in :mod:`robosimstudio_materials.mpm`; this
module is the domain, the entity that names it, the constructor and the presets,
and :func:`roller`, a rigid tool the YAM can roll a lump flat with.

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

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs import ObjectSpec
from robosimstudio.specs.asset import ShapeSpec
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
    "EntityRoller",
    "ParticleBodySpec",
    "ParticleMaterialSpec",
    "RollerSpec",
    "dough",
    "roller",
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
    routes ``dough`` to it, and :data:`DOUGH_60HZ` puts a free tool -- a
    :func:`roller` -- on the robot's island so the hand can hold one.
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


FINGER_TIP = 0.0165
"""How far the YAM's fingertips reach under its tool centre [m], pointing down (``probe_fingers.py``, GPU 6)."""

SLOT = 0.020
"""The width of the slot each finger closes into, along the roll [m]: the YAM's 12 mm finger blade and 4 mm either side."""

SLOT_WALL = 0.007
"""How high the slot's walls stand over the shoulder [m].

The YAM's finger is a 12 mm blade for its lowest 5 mm only; above that a brace
leaves it toward the finger's front at 46 degrees, and a wall that reached it
would stop the jaws closing (``probe_fingers.py``, GPU 6)."""


@dataclass(frozen=True, kw_only=True)
class RollerSpec(ObjectSpec):
    """A roller on an axle under a slotted handle: :class:`~robosimstudio.specs.ObjectSpec` with the handle's fit.

    The roller is the body the pose places, its axle's centre, lying along its
    frame's y; the handle stands on the axle along the frame's z. ``grip`` is
    the tool centre's height over the axle when the jaws close on the handle.
    ``spins`` puts the handle on a revolute joint about the axle; ``False``
    makes the two one body.
    """

    entity: str = "roller"
    grip: float = 0.05
    spins: bool = True
    axle_friction: float = 0.005
    """The axle's friction torque [N m]: what keeps the handle standing on the drum at rest.

    A handle standing on a free axle is balanced on a point, and anything that
    nudges it tips it over. 0.005 N m holds it without holding the drum: rolled
    three times across the dough by the hand, 160 mm a pass, the drum turned
    4.5-4.8 rad a pass of the 8 a pure roll turns; at 0.02 N m it turned 0.4-1.5,
    and on a free axle 2.9-9.8, overrunning (``probe_roll2.py``, GPU 6)."""


def handle_boxes(grip: float) -> tuple[tuple[Vec3, Vec3], ...]:
    """The handle as ``(centre, half-extents)`` boxes in the roller's frame [m]: a shoulder, a web, two slot walls.

    The web is what the jaws close across (20 mm along y). The shoulder stands
    1.5 mm under the fingertips, so the dough's push up on the roller is carried
    by the fingers' tips rather than by the grip's friction; the walls either
    side of each finger carry the roll's drag the same way. The grip's friction
    held neither: a plain 20 mm plate slid 30 mm up through the YAM's closed jaws
    in one pass over the dough, and the shoulder keeps it within 2 mm over three
    (``probe_roll.py``, ``probe_roll2.py``, GPU 6).
    """
    half_x, half_y = 0.016, 0.018
    shoulder = grip - FINGER_TIP - 0.0015
    top = grip + 0.025
    wall = half_x - SLOT / 2
    return (
        ((0.0, 0.0, shoulder / 2), (half_x, half_y, shoulder / 2)),
        ((0.0, 0.0, (shoulder + top) / 2), (half_x, 0.010, (top - shoulder) / 2)),
        ((SLOT / 2 + wall / 2, 0.0, shoulder + SLOT_WALL / 2), (wall / 2, half_y, SLOT_WALL / 2)),
        ((-SLOT / 2 - wall / 2, 0.0, shoulder + SLOT_WALL / 2), (wall / 2, half_y, SLOT_WALL / 2)),
    )


class EntityRoller(EntityRigid):
    """A :class:`RollerSpec`: the roller on a free joint, the handle on a revolute joint about its axle."""

    spec: RollerSpec
    axle_coord: int = -1
    """The axle's ``joint_q`` index once built; -1 for a welded roller."""
    handle_body: int = -1
    """The handle's body once built; the roller's own for a welded one."""

    def collect_shapes(self, builder) -> None:
        """The drum's shapes and the handle's: the handle's boxes are what the jaws touch.

        :meth:`EntityRigid.collect_shapes` takes the shapes on :attr:`body`, the
        drum, and a spinning handle is a body of its own -- so a contact measure
        between the hand and the roller, and the **contact** tunable, would have
        missed every box the fingers close on.
        """
        bodies = {self.body, self.handle_body}
        self.shapes = np.asarray(
            [s for s in range(builder.shape_count) if builder.shape_body[s] in bodies], dtype=np.int32
        )

    def _add_primitive(self, builder, pose) -> None:
        import warp as wp

        from robosimstudio.scene.build.bodies import set_body_mass

        spec = self.spec
        xform = wp.transform(tuple(float(v) for v in pose[:3]), wp.quat(*[float(v) for v in pose[3:7]]))
        cfg = self._contact_cfg(builder)
        along_y = wp.transform((0.0, 0.0, 0.0), wp.quat(-0.7071068, 0.0, 0.0, 0.7071068))
        roller = builder.add_link(xform=xform, label=spec.name)
        free = builder.add_joint_free(child=roller, label=f"{spec.name}_free_joint")
        self.coord_start = int(builder.joint_q_start[free])
        self.dof_start = int(builder.joint_qd_start[free])
        self.body = roller
        spec.shape.add_to(builder, body=roller, xform=along_y, label=spec.name, cfg=cfg)
        joints = [free]
        handle = builder.add_link(xform=xform, label=f"{spec.name}_handle") if spec.spins else roller
        self.handle_body = handle
        for k, (centre, half) in enumerate(handle_boxes(float(spec.grip))):
            builder.add_shape_box(
                handle,
                xform=wp.transform(centre, wp.quat_identity()),
                hx=half[0],
                hy=half[1],
                hz=half[2],
                cfg=cfg,
                label=f"{spec.name}_handle_{k}",
            )
        if spec.spins:
            joints.append(
                builder.add_joint_revolute(
                    parent=roller,
                    child=handle,
                    axis=(0.0, 1.0, 0.0),
                    friction=float(spec.axle_friction),
                    label=f"{spec.name}_axle",
                )
            )
            self.axle_coord = int(builder.joint_q_start[joints[-1]])
        builder.add_articulation(joints, label=f"{spec.name}_articulation")
        if spec.mass is not None:
            set_body_mass(builder, roller, float(spec.mass))

    def reset(self, pipeline, episode) -> None:
        super().reset(pipeline, episode)
        if self.axle_coord >= 0:
            for state in (pipeline.state_0, pipeline.state_1):
                state.joint_q[self.axle_coord : self.axle_coord + 1].assign(np.zeros(1, dtype=np.float32))


register_entity("roller")(EntityRoller)


def roller(
    *,
    name: str = "roller",
    radius: float = 0.02,
    half_length: float = 0.05,
    at: Vec3 = (0.0, 0.0, 0.0),
    grip: float = 0.05,
    spins: bool = True,
    **kwargs,
) -> RollerSpec:
    """A dough roller a parallel gripper can hold: a roller along y on an axle, and a slotted handle standing on it.

    Args:
        radius, half_length: The roller's, a capsule along the bench's y [m];
            ``half_length`` is its straight part's.
        at: The axle's centre, in the cell's work-surface frame: ``z = radius``
            rests it on the bench.
        grip: The tool centre's height over the axle with the jaws closed on the
            handle [m]; the handle's shoulder stands under the fingertips there.
        spins: The handle on a revolute joint about the axle, so the roller
            turns under a held handle; ``False`` welds the two.
        **kwargs: Any other :class:`RollerSpec` field: ``friction``, ``color``,
            ``axle_friction``.
    """
    return RollerSpec(
        name=name,
        shape=ShapeSpec(kind="capsule", radius=radius, half_height=half_length),
        pose=at,
        grip=grip,
        spins=spins,
        **kwargs,
    )
