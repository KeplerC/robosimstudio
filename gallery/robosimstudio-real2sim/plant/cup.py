"""Glazed ceramic cup; convex wall sectors for rigid contact, closed liquid surface."""

from dataclasses import dataclass
from pathlib import Path

import newton
import numpy as np
import trimesh
import warp as wp

from robosimstudio.core.render import Finish
from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.scene.build.bodies import set_body_mass
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.asset import ObjectSpec, ShapeSpec

ROOT = Path(__file__).resolve().parent
HEIGHT = 0.090
RIM = 0.041
INNER = 0.033
BASE = 0.014


def cup_mesh():
    m = trimesh.creation.revolve(
        np.array(
            [
                (0, 0),
                (0.040, 0),
                (0.040, 0.004),
                (0.041, 0.088),
                (0.041, 0.090),
                (0.033, 0.090),
                (0.032, 0.014),
                (0, 0.014),
            ]
        ),
        sections=96,
    )
    m.fix_normals()
    m.export(ROOT / "assets/cup.obj")
    return m


@dataclass(frozen=True, kw_only=True)
class CupSpec(ObjectSpec):
    entity: str = "watering_cup"


class EntityCup(EntityRigid):
    def _add_primitive(self, builder, pose):
        self.body = builder.add_body(xform=wp.transform(tuple(pose[:3]), wp.quat(*pose[3:])), label=self.name)
        j = builder.joint_count - 1
        self.coord_start = int(builder.joint_q_start[j])
        self.dof_start = int(builder.joint_qd_start[j])
        cfg = builder.default_shape_cfg.copy()
        cfg.density = 2300.0
        cfg.mu = float(self.spec.friction)
        cfg.gap = 0.0005
        cfg.has_particle_collision = False
        cfg.is_visible = False
        cfg.ke = 50000.0
        cfg.kd = 100.0
        builder.add_shape_cylinder(
            body=self.body,
            radius=0.040,
            half_height=0.007,
            xform=wp.transform((0, 0, 0.007), wp.quat_identity()),
            cfg=cfg,
            label=self.name + "_base",
        )
        for k in range(24):
            a, b = 2 * np.pi * np.array([k, k + 1]) / 24
            v = []
            for z, ri, ro in [(0, 0.032, 0.040), (0.090, 0.033, 0.041)]:
                for r, t in [(ri, a), (ro, a), (ro, b), (ri, b)]:
                    v.append([r * np.cos(t), r * np.sin(t), z])
            m = trimesh.convex.convex_hull(np.array(v))
            builder.add_shape_mesh(
                body=self.body,
                mesh=newton.Mesh(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32).ravel()),
                cfg=cfg,
                label=f"{self.name}_wall_{k}",
            )
        m = cup_mesh()
        draw = cfg.copy()
        draw.has_particle_collision = True
        draw.has_shape_collision = False
        draw.is_visible = True
        draw.density = 0.0
        draw.mu = 0.2  # wet interior; the dry grasp surfaces keep 0.8
        builder.add_shape_mesh(
            body=self.body,
            mesh=newton.Mesh(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32).ravel()),
            cfg=draw,
            label=self.name + "_surface",
        )
        set_body_mass(builder, self.body, float(self.spec.mass))


register_entity("watering_cup")(EntityCup)


def cup(at):
    return CupSpec(
        name="cup",
        shape=ShapeSpec(kind="cylinder", radius=RIM, half_height=HEIGHT / 2),
        pose=tuple(at),
        mass=0.12,
        friction=0.8,
        color=(0.94, 0.93, 0.90),
        finish=Finish(roughness=0.18),
        obb_grasp=False,
    )
