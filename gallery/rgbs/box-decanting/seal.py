"""Twin-local tape cohesion and contact severance, authorized in review round 1.
Adapted from the old decant seam law. No robot attachment or object motor.
"""

from collections import deque

import newton
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from robosimstudio.core.probes.contact import ContactProbe


@wp.kernel(enable_backward=False)
def cohesion(
    q: wp.array(dtype=float),
    qd: wp.array(dtype=float),
    intact: wp.array(dtype=float),
    coords: wp.array(dtype=int),
    dofs: wp.array(dtype=int),
    forces: wp.array(dtype=float),
    n: int,
    kp: float,
    kd: float,
    cap: float,
):
    f = wp.tid()
    remaining = float(0.0)
    for i in range(n):
        remaining += intact[2 * i + f]
    remaining /= float(n)
    torque = remaining * (-kp * q[coords[f]] - kd * qd[dofs[f]])
    forces[dofs[f]] = wp.clamp(torque, -cap * remaining, cap * remaining)


@wp.kernel(enable_backward=False)
def sever(
    q: wp.array(dtype=wp.transform),
    qd: wp.array(dtype=wp.spatial_vector),
    bodies: wp.array(dtype=int),
    points: wp.array(dtype=wp.vec3),
    intact: wp.array(dtype=float),
    press: wp.array(dtype=float),
    knife: int,
    p0: wp.vec3,
    length: float,
    dt: float,
    threshold: float,
    rate: float,
    kerf: float,
):
    i = wp.tid()
    force = press[0]
    if force <= threshold:
        return
    xk = q[knife]
    xf = q[bodies[i]]
    a = wp.transform_point(xk, p0)
    u = wp.transform_vector(xk, wp.vec3(1.0, 0.0, 0.0))
    point = wp.transform_point(xf, points[i])
    along = wp.dot(point - a, u)
    if along < 0.0 or along > length:
        return
    if wp.length(point - a - along * u) > kerf:
        return
    vk = qd[knife]
    vf = qd[bodies[i]]
    v1 = wp.spatial_top(vk) + wp.cross(wp.spatial_bottom(vk), point - wp.transform_get_translation(xk))
    v2 = wp.spatial_top(vf) + wp.cross(wp.spatial_bottom(vf), point - wp.transform_get_translation(xf))
    speed = wp.abs(wp.dot(v1 - v2, u))
    intact[i] = wp.max(0.0, intact[i] - rate * (force - threshold) * speed * dt)


@wp.kernel(enable_backward=False)
def color_cells(intact: wp.array(dtype=float), shapes: wp.array(dtype=int), colors: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    v = intact[i]
    colors[shapes[i]] = v * wp.vec3(0.85, 0.78, 0.10) + (1.0 - v) * wp.vec3(0.075, 0.06, 0.035)


class TapeLaw:
    """Mixin on the carton: both tape halves belong to its passive flap links."""

    tape_length = 0.084
    tape_width = 0.018
    tape_thickness = 0.001
    stations = 28
    pierce_n = 2.0
    cut_rate = 90.0
    kerf_m = 0.007
    cohesion_kp = 120.0
    cohesion_kd = 0.35
    cohesion_cap_nm = 0.20

    def build_tape(self, builder):
        cfg = self._contact_cfg(builder)
        cfg.density = 0.0
        cfg.mu = 0.4
        hinge = self.spec.outer[1] / 2 - self.spec.wall
        z = self.spec.wall + self.tape_thickness / 2
        self.tape_shapes = []
        self.cell_shapes = []
        self.cell_points = []
        self.cell_bodies = []
        # Each half consists of an outer yellow band and a colliding inner cell row.
        for f, body in enumerate(self.flaps):
            shape = builder.add_shape_box(
                body,
                xform=wp.transform((0, hinge - 0.006, z), wp.quat_identity()),
                hx=self.tape_length / 2,
                hy=0.003,
                hz=self.tape_thickness / 2,
                cfg=cfg,
                color=(0.85, 0.78, 0.10),
                label=f"tape_band_{f}",
            )
            self.tape_shapes.append(shape)
        pitch = self.tape_length / self.stations
        for i in range(self.stations):
            x = -self.tape_length / 2 + (i + 0.5) * pitch
            for f, body in enumerate(self.flaps):
                p = (x if f == 0 else -x, hinge - 0.0015, z)
                shape = builder.add_shape_box(
                    body,
                    xform=wp.transform(p, wp.quat_identity()),
                    hx=pitch / 2,
                    hy=0.0015,
                    hz=self.tape_thickness / 2,
                    cfg=cfg,
                    color=(0.85, 0.78, 0.10),
                    label=f"tape_cell_{i}_{f}",
                )
                self.tape_shapes.append(shape)
                # Appearance only: retain this exact 3 mm colliding cell and sample point.
                builder.shape_flags[shape] &= ~int(newton.ShapeFlags.VISIBLE)
                visual = builder.ShapeConfig(density=0, has_shape_collision=False)
                for suffix, cy, hy in [("yellow", hinge - 0.00175, 0.00125), ("slit", hinge - 0.00025, 0.00025)]:
                    painted = builder.add_shape_box(
                        body,
                        xform=wp.transform((p[0], cy, z), wp.quat_identity()),
                        hx=pitch / 2,
                        hy=hy,
                        hz=self.tape_thickness / 2,
                        cfg=visual,
                        color=(0.85, 0.78, 0.10),
                        label=f"tape_{suffix}_{i}_{f}",
                    )
                    if suffix == "slit":
                        self.cell_shapes.append(painted)
                self.cell_points.append((p[0], hinge - 0.0015, z + self.tape_thickness / 2))
                self.cell_bodies.append(body)

        # The cut faces only separate under the 0..95 degree hinges. Their
        # zero-clearance seam is cohesion, not a frictional self-contact pair.
        for a in self.tape_shapes:
            for b in self.tape_shapes:
                if a < b and builder.shape_body[a] != builder.shape_body[b]:
                    builder.add_shape_collision_filter_pair(a, b)

    def bind_tape(self, model):
        device = model.device
        self.intact = wp.ones(2 * self.stations, dtype=float, device=device)
        self.press_device = wp.zeros(1, dtype=float, device=device)
        self.cells_device = wp.array(self.cell_points, dtype=wp.vec3, device=device)
        self.bodies_device = wp.array(self.cell_bodies, dtype=int, device=device)
        self.colors_device = wp.array(self.cell_shapes, dtype=int, device=device)
        self.coords_device = wp.array(
            [int(model.joint_q_start.numpy()[j]) for j in self.hinges], dtype=int, device=device
        )
        self.dofs_device = wp.array(
            [int(model.joint_qd_start.numpy()[j]) for j in self.hinges], dtype=int, device=device
        )
        self.knife_body = list(model.body_label).index("cutter")
        self.blade_shapes = [i for i, s in enumerate(model.shape_label) if s == "cutter/blade"]
        self.force_window = deque(maxlen=5)
        self.press = 0.0
        self.history = []
        self.tape_time = 0.0
        self.tape_control = None

    def on_scene_ready(self, pipeline):
        super().on_scene_ready(pipeline)
        self.tape_control = pipeline.control.joint_f
        self.tape_probe = ContactProbe(pipeline.program.mujoco_solver, self.model, float(pipeline.sim_dt))

    def contribute_hooks(self, hooks):
        super().contribute_hooks(hooks)
        hooks.add("rigid_force", self.tape_hold, name="tape internal cohesion")
        hooks.add("post_solve", self.tape_sever, name="tape contact severance")
        hooks.add("after", self.tape_feed, name="tape measured blade force")

    def tape_hold(self, state, dt):
        if self.tape_control is not None:
            wp.launch(
                cohesion,
                dim=2,
                inputs=[
                    state.joint_q,
                    state.joint_qd,
                    self.intact,
                    self.coords_device,
                    self.dofs_device,
                    self.tape_control,
                    self.stations,
                    self.cohesion_kp,
                    self.cohesion_kd,
                    self.cohesion_cap_nm,
                ],
                device=self.model.device,
            )

    def tape_sever(self, state, dt):
        wp.launch(
            sever,
            dim=2 * self.stations,
            inputs=[
                state.body_q,
                state.body_qd,
                self.bodies_device,
                self.cells_device,
                self.intact,
                self.press_device,
                self.knife_body,
                wp.vec3(0.0225, 0, -0.0125),
                0.008,
                float(dt),
                self.pierce_n,
                self.cut_rate,
                self.kerf_m,
            ],
            device=self.model.device,
        )
        wp.launch(
            color_cells,
            dim=2 * self.stations,
            inputs=[self.intact, self.colors_device, self.model.shape_color],
            device=self.model.device,
        )

    def tape_feed(self, state, dt):
        self.tape_probe.update(step=None)
        force = self.tape_probe.pairwise_force(self.blade_shapes, self.tape_shapes)
        normal = Rotation.from_quat(state.body_q.numpy()[self.body, 3:]).apply([0, 0, 1])
        instantaneous = abs(float(np.dot(force, normal)))
        self.force_window.append(instantaneous)
        self.press = float(np.mean(self.force_window))
        self.press_device.assign(np.array([self.press], dtype=np.float32))
        self.tape_time += float(dt)
        self.history.append((self.tape_time, instantaneous, self.press, self.cut_fraction()))

    def cut_fraction(self):
        rows = self.intact.numpy().reshape(-1, 2)
        return float(np.mean(np.all(rows < 0.05, axis=1)))

    def recolors_live(self):
        return True

    def reset_tape(self):
        self.intact.fill_(1.0)
        self.press_device.zero_()
        self.force_window.clear()
        self.press = 0.0
        self.tape_time = 0.0
        self.history = []
        wp.launch(
            color_cells,
            dim=2 * self.stations,
            inputs=[self.intact, self.colors_device, self.model.shape_color],
            device=self.model.device,
        )
