"""Elastic plant on the robot's MuJoCo island, coupled to the water MPM.

Each original eight-segment stalk is a chain of capsules on passive,
spring-damped ball joints rooted in the soil. Each original settled blade is
a one-millimetre closed shell on a passive spring joint. Rest targets never
change. Gravity remains active. The cup, fingers and water contact the plant;
plant self-contact is disabled as in the original cloth setup. Only embedded
root segments are filtered against their soil and pot.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import newton
import numpy as np
import warp as wp

from robosimstudio.core.render import Finish
from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.asset import ObjectSpec, ShapeSpec

__all__ = ["EntityPlant", "PlantSpec", "plant_spec"]


@dataclass(frozen=True, kw_only=True)
class PlantSpec(ObjectSpec):
    """A plant rig: ``layout`` is the JSON of :func:`rig_record` (bench-frame geometry and the joint law)."""

    entity: str = "plant_rig"
    layout: str = "{}"
    obb_grasp: bool = False


def _quat_z_to(d: np.ndarray) -> np.ndarray:
    """xyzw turning +z onto unit ``d``."""
    z = np.array([0.0, 0.0, 1.0])
    d = d / np.linalg.norm(d)
    c = float(np.dot(z, d))
    if c > 1.0 - 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if c < -1.0 + 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax = np.cross(z, d)
    ax /= np.linalg.norm(ax)
    h = 0.5 * math.acos(c)
    return np.array([*(ax * math.sin(h)), math.cos(h)])


def _qmat(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _qinv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _qmul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def _tf(p, q):
    return wp.transform(tuple(float(v) for v in p), wp.quat(*(float(v) for v in q)))


def rig_record(layout, law: dict) -> str:
    """The JSON a :class:`PlantSpec` carries: stalk nodes, blade shells (bench frame) and the joint law."""
    return json.dumps(
        {
            "stalks": [
                {"name": s.name, "nodes": s.nodes.tolist(), "radius": s.radius, "leaf": s.leaf} for s in layout.stalks
            ],
            "leaves": [
                {
                    "name": lf.name,
                    "junction": lf.junction.tolist(),
                    "V": lf.shell_V.tolist(),
                    "F": lf.shell_F.tolist(),
                    "area": lf.area,
                    "kind": lf.kind,
                    "UV": lf.shell_UV.tolist(),
                    "texture": lf.tex_c,
                }
                for lf in layout.leaves
            ],
            "law": law,
        }
    )


def plant_spec(name: str, layout, law: dict) -> PlantSpec:
    return PlantSpec(
        name=name,
        shape=ShapeSpec(kind="sphere", radius=0.005),
        pose=(0.0, 0.0, 0.0),
        layout=rig_record(layout, law),
        color=None,
    )


class EntityPlant(EntityRigid):
    """Builds the rig; owns every body and shape it added (the composer routes them by its ``rigid`` domain)."""

    spec: PlantSpec

    def __init__(self, spec, **kw) -> None:
        super().__init__(spec, **kw)
        self.links: dict[str, list[int]] = {}
        self.leaf_body: dict[str, int] = {}
        self.joint_q_ranges: list[tuple[int, int]] = []

    def _add_primitive(self, builder: newton.ModelBuilder, pose) -> None:
        rec = json.loads(self.spec.layout)
        law = rec["law"]
        cfg = builder.default_shape_cfg.copy()
        cfg.collision_group = 1  # ordinary cup, hand and water contacts
        cfg.density = float(law.get("stalk_density", 1000.0))
        cfg.has_particle_collision = True
        cfg.mu = 0.6
        cfg.gap = 0.0002
        frame = np.asarray(self.placed((0, 0, 0, 0, 0, 0, 1)), np.float64)
        Rf = _qmat(frame[3:7])

        def world(x):
            return np.asarray(x, np.float64) @ Rf.T + frame[:3]

        first_body = None
        leaves = {lf["name"]: lf for lf in rec["leaves"]}
        for st in rec["stalks"]:
            nodes = world(st["nodes"])
            r = float(st["radius"])
            joints, bodies = [], []
            parent, parent_pose = -1, None
            for k in range(len(nodes) - 1):
                a, b = nodes[k], nodes[k + 1]
                L = float(np.linalg.norm(b - a))
                q = _quat_z_to(b - a)
                mid = 0.5 * (a + b)
                body = builder.add_link(xform=_tf(mid, q), label=f"{st['name']}_{k}")
                if first_body is None:
                    first_body = body
                builder.add_shape_capsule(
                    body,
                    radius=r,
                    half_height=max(0.5 * L, 1e-4),
                    cfg=cfg,
                    color=(0.57, 0.65, 0.30),
                    label=f"{st['name']}_{k}",
                )
                # the joint at a, oriented as this link
                child_xf = _tf((0.0, 0.0, -0.5 * L), (0, 0, 0, 1))
                if parent < 0:
                    parent_xf = _tf(a, q)
                    k_bend = float(law["k_root"])
                else:
                    pp, pq = parent_pose
                    parent_xf = _tf(_qmat(pq).T @ (a - pp), _qmul(_qinv(pq), q))
                    k_bend = float(law["k_node"])
                j = builder.add_joint_ball(
                    parent=parent,
                    child=body,
                    parent_xform=parent_xf,
                    child_xform=child_xf,
                    armature=float(law["armature"]),
                    damping=float(law["damping"]),
                    custom_attributes={"mujoco:dof_passive_stiffness": k_bend},
                    label=f"{st['name']}_j{k}",
                )
                joints.append(j)
                bodies.append(body)
                parent, parent_pose = body, (mid, q)
            # the blade, at the junction, on its own hinge
            lf = leaves[st["leaf"]]
            J = nodes[-1]
            lb = builder.add_link(xform=_tf(J, (0, 0, 0, 1)), label=f"{lf['name']}")
            V = world(lf["V"]) - J
            mesh = newton.Mesh(
                np.asarray(V, np.float32),
                np.asarray(lf["F"], np.int32).reshape(-1),
                uvs=np.asarray(lf["UV"], np.float32),
                texture=lf["texture"],
            )
            lcfg = cfg.copy()
            lcfg.density = 400.0
            builder.add_shape_mesh(lb, mesh=mesh, cfg=lcfg, label=f"{lf['name']}")
            pp, pq = parent_pose
            jl = builder.add_joint_ball(
                parent=parent,
                child=lb,
                parent_xform=_tf(_qmat(pq).T @ (J - pp), _qinv(pq)),
                child_xform=_tf((0, 0, 0), (0, 0, 0, 1)),
                armature=float(law["armature"]),
                damping=float(law["leaf_damping"]),
                custom_attributes={"mujoco:dof_passive_stiffness": float(law["k_leaf"])},
                label=f"{lf['name']}_hinge",
            )
            joints.append(jl)
            builder.add_articulation(joints, label=f"{st['name']}_articulation")
            from robosimstudio.scene.build.bodies import set_body_mass

            areal = float(law["leaf_areal_density"]) * (1.0 if lf["kind"] == "blade" else 0.7)
            set_body_mass(builder, lb, max(areal * float(lf["area"]), 1e-4))
            self.links[st["name"]] = bodies
            self.leaf_body[lf["name"]] = lb
        # The authored leaves overlap at their silhouettes, as in the original
        # cloth setup with self-contact disabled. External contacts stay on.
        own = list(range(self.range.shapes[0], builder.shape_count))
        for i, a in enumerate(own):
            for b in own[i + 1 :]:
                builder.add_shape_collision_filter_pair(a, b)
        # Petioles are rooted inside the soil: their embedded base is not an
        # obstacle to the soil it grows from. Upper segments remain collidable.
        for a in own:
            label = str(builder.shape_label[a]).split("/")[-1]
            if label.endswith("_0") or label.endswith("_1"):
                for b in range(self.range.shapes[0]):
                    if str(builder.shape_label[b]).split("/")[-1] in ("soil", "pot") or str(
                        builder.shape_label[b]
                    ).split("/")[-1].startswith("soil_core_"):
                        builder.add_shape_collision_filter_pair(a, b)
        self.body = first_body if first_body is not None else -1

    def apply_display(self, builder):
        super().apply_display(builder)
        for s in range(*self.range.shapes):
            src = getattr(builder.shape_source[s], "texture", None)
            self.finishes[s] = Finish(roughness=0.96 if src else 0.55, texture=src)

    def collect_shapes(self, builder) -> None:
        lo, hi = self.range.bodies[0], builder.body_count
        self.shapes = np.asarray([s for s in range(builder.shape_count) if lo <= builder.shape_body[s] < hi], np.int32)

    def compute_grasp_box(self, builder) -> None:
        self.obb = None

    def write_initial_q(self, builder, episode) -> None:
        return

    def reset(self, pipeline, episode) -> None:
        return

    def tunables(self, pipeline=None):
        return ()


register_entity("plant_rig")(EntityPlant)
