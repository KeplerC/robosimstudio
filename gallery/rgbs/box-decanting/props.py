"""Measured rigid props. Carton construction adapted from old decant/carton.py.
The tape cohesion is internal to the carton; there is no object motor or robot attachment.
"""

import math
from dataclasses import dataclass, replace

import newton
import numpy as np
import warp as wp
from seal import TapeLaw

from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.scene.build.bodies import set_body_mass
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.asset import ObjectSpec, ShapeSpec
from robosimstudio_materials.cutting.entity import EntityKnife
from robosimstudio_materials.cutting.spec import KnifeSpec


@dataclass(frozen=True, kw_only=True)
class Carton(ObjectSpec):
    entity: str = "gallery_carton"
    name: str = "carton"
    outer: tuple = (0.154, 0.100, 0.0785)
    wall: float = 0.002
    gap: float = 0.012
    mass: float = 0.085
    friction: float = 0.65
    color: tuple = (0.65, 0.48, 0.28)
    shape: ShapeSpec = None

    def __post_init__(self):
        object.__setattr__(
            self, "shape", ShapeSpec(kind="box", half_extents=(self.outer[0] / 2, self.outer[1] / 2, self.wall / 2))
        )
        super().__post_init__()


@register_entity("gallery_carton")
class CartonEntity(TapeLaw, EntityRigid):
    def _add_primitive(self, builder, pose):
        self.body = builder.add_link(xform=wp.transform(pose[:3], pose[3:]), label=self.spec.name)
        self.free = builder.add_joint_free(child=self.body, label="carton_free")
        self.coord_start = builder.joint_q_start[self.free]
        self.dof_start = builder.joint_qd_start[self.free]
        self.spec.shape.add_to(
            builder, body=self.body, xform=wp.transform_identity(), label="carton_floor", cfg=self._contact_cfg(builder)
        )

    def build(self, builder, episode):
        super().build(builder, episode)
        s = self.spec
        L, W, H = s.outer
        t = s.wall
        cfg = self._contact_cfg(builder)

        def box(body, name, p, h, color):
            return builder.add_shape_box(
                body,
                xform=wp.transform(p, wp.quat_identity()),
                hx=h[0],
                hy=h[1],
                hz=h[2],
                cfg=cfg,
                color=color,
                label=name,
            )

        for sign in [-1, 1]:
            box(self.body, f"end_{sign}", (sign * (L - t) / 2, 0, H / 2), (t / 2, W / 2, (H - t) / 2), s.color)
            box(
                self.body,
                f"side_{sign}",
                (0, sign * (W - t) / 2, H / 2),
                ((L - 2 * t) / 2, t / 2, (H - t) / 2),
                s.color,
            )
        self.flaps = []
        self.hinges = []
        self.dust_flaps = []
        self.dust_hinges = []
        pose = self._placed(self._episode_pose(None))
        base = wp.transform(pose[:3], pose[3:])
        hinge = W / 2 - t
        length = hinge - s.gap / 2
        # Short dust flaps are folded into the carton, with passive creases.
        # They are not fixed supports for either robot hand.
        for sign in [-1, 1]:
            anchor = wp.transform(
                (sign * (L / 2 - t), 0, H - t / 2),
                wp.quat_from_axis_angle(wp.vec3(0, 0, 1), math.pi / 2 if sign > 0 else -math.pi / 2),
            )
            body = builder.add_link(xform=wp.transform_multiply(base, anchor), label=f"dust_flap_{sign}")
            box(body, f"inner_flap_{sign}", (0, 0.013, 0), (W / 2 - t, 0.013, 0.00075), (0.56, 0.41, 0.25))
            j = builder.add_joint_revolute(
                parent=self.body,
                child=body,
                parent_xform=anchor,
                child_xform=wp.transform_identity(),
                axis=(1, 0, 0),
                target_ke=0,
                target_kd=0,
                actuator_mode=newton.JointTargetMode.NONE,
                limit_lower=math.radians(-95),
                limit_upper=0,
                damping=0.0015,
                armature=5e-5,
                limit_ke=1e6,
                limit_kd=1000,
                collision_filter_parent=False,
                label=f"dust_hinge_{sign}",
            )
            builder.joint_q[builder.joint_q_start[j]] = -math.pi / 2
            self.dust_flaps.append(body)
            self.dust_hinges.append(j)
            set_body_mass(builder, body, 0.002)
        for name, sign in [("near", -1), ("far", 1)]:
            anchor = wp.transform(
                (0, sign * hinge, H - t / 2), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.0 if sign < 0 else math.pi)
            )
            body = builder.add_link(xform=wp.transform_multiply(base, anchor), label="flap_" + name)
            box(body, "flap_" + name, (0, length / 2, t / 2), (L / 2, length / 2, t / 2), (0.36, 0.25, 0.14))
            j = builder.add_joint_revolute(
                parent=self.body,
                child=body,
                parent_xform=anchor,
                child_xform=wp.transform_identity(),
                axis=(1, 0, 0),
                target_ke=0,
                target_kd=0,
                actuator_mode=newton.JointTargetMode.NONE,
                limit_lower=0,
                limit_upper=math.radians(95),
                damping=0.0015,
                armature=5e-4,
                limit_ke=1e6,
                limit_kd=1000,
                collision_filter_parent=False,
                label="flap_" + name,
            )
            self.flaps.append(body)
            self.hinges.append(j)
            set_body_mass(builder, body, 0.005)
        self.build_tape(builder)
        builder.add_articulation([self.free, *self.dust_hinges, *self.hinges], label="carton_articulation")
        set_body_mass(builder, self.body, s.mass)

    def bind(self, model):
        super().bind(model)
        self.bind_tape(model)
        self.shapes = np.flatnonzero(
            np.isin(model.shape_body.numpy(), [self.body, *self.dust_flaps, *self.flaps])
        ).astype(np.int32)

    def apply_display(self, builder):
        super().apply_display(builder)
        for i in self.tape_shapes:
            builder.shape_color[i] = (0.85, 0.78, 0.10)
        for i, label in enumerate(builder.shape_label):
            if label.startswith(("tape_yellow_", "tape_slit_")):
                builder.shape_color[i] = (0.85, 0.78, 0.10)
            if label in ("flap_near", "flap_far"):
                builder.shape_color[i] = (0.85, 0.74, 0.55)

    def reset(self, pipeline, episode):
        super().reset(pipeline, episode)
        self.reset_tape()
        for state in (pipeline.state_0, pipeline.state_1):
            if state is None:
                continue
            q = state.joint_q.numpy()
            v = state.joint_qd.numpy()
            for j in self.hinges:
                q[int(self.model.joint_q_start.numpy()[j])] = 0
                v[int(self.model.joint_qd_start.numpy()[j])] = 0
            for j in self.dust_hinges:
                q[int(self.model.joint_q_start.numpy()[j])] = -math.pi / 2
                v[int(self.model.joint_qd_start.numpy()[j])] = 0
            state.joint_q.assign(q)
            state.joint_qd.assign(v)

    def handle(self, model):
        return replace(super().handle(model), bodies=(self.body, *self.dust_flaps, *self.flaps))


@dataclass(frozen=True, kw_only=True)
class SafetyCutter(KnifeSpec):
    entity: str = "gallery_safety_cutter"
    handle_size: tuple = (0.045, 0.014, 0.025)
    blade_length: float = 0.008
    blade_height: float = 0.012
    blade_thickness: float = 0.0006
    handle_color: tuple = (0.035, 0.040, 0.052)
    mass: float = 0.025
    friction: float = 0.8

    @property
    def edge(self):
        return ((0.0225, 0, -0.0125), (1, 0, 0), 0.008)


@register_entity("gallery_safety_cutter")
class SafetyCutterEntity(EntityKnife):
    def build(self, builder, episode):
        super().build(builder, episode)
        # Steel edge on tape, distinct from the rubber/plastic grip.
        builder.shape_material_mu[self.blade_shape] = 0.25

    def _add_primitive(self, builder, pose):
        super()._add_primitive(builder, pose)
        builder.shape_transform[self.blade_shape] = wp.transform((0.0265, 0, -0.0065), wp.quat_identity())
        cfg = self._contact_cfg(builder)
        cfg.density = 100
        self.button = builder.add_shape_cylinder(
            self.body,
            xform=wp.transform((0.007, -0.0077, 0), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), math.pi / 2)),
            radius=0.0055,
            half_height=0.0007,
            cfg=cfg,
            label="cutter_red_slider",
        )
        # Exact visual partition of the existing colliding housing: mass/contact unchanged.
        builder.shape_flags[self.handle_shape] &= ~int(newton.ShapeFlags.VISIBLE)
        visual = builder.ShapeConfig(density=0, has_shape_collision=False)
        self.housing_visuals = []
        for name, x, hx, color in [
            ("foam", -0.0135, 0.009, (0.025, 0.028, 0.032)),
            ("grey", 0.009, 0.0135, (0.42, 0.45, 0.49)),
        ]:
            shape = builder.add_shape_box(
                self.body,
                xform=wp.transform((x, 0, 0), wp.quat_identity()),
                hx=hx,
                hy=0.007,
                hz=0.0125,
                cfg=visual,
                color=color,
                label="cutter_" + name,
            )
            self.housing_visuals.append((shape, color))
        set_body_mass(builder, self.body, self.spec.mass)

    def apply_display(self, builder):
        super().apply_display(builder)
        builder.shape_color[self.button] = (0.55, 0.025, 0.018)
        for shape, color in self.housing_visuals:
            builder.shape_color[shape] = color
