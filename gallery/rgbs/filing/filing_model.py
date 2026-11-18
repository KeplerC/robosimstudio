"""A free wooden workpiece, a powered spindle, and contact-only ink removal."""

from dataclasses import dataclass

import newton
import numpy as np
import warp as wp

from robosimstudio.core.math import quat_to_mat
from robosimstudio.core.probes.contact import ContactProbe
from robosimstudio.core.render.finish import Finish
from robosimstudio.scene.entity import Entity
from robosimstudio.scene.registry import register_entity

WOOD = (0.88, 0.82, 0.65)
WOOD_FINISH = Finish(roughness=0.8)
BLUE = (0.025, 0.13, 0.65)
DRUM = np.array((0.680, 0.0, 0.896))
RADIUS = 0.0075
HALF = 0.006


@dataclass(frozen=True, kw_only=True)
class FilingSpec:
    entity: str = "filing_apparatus"
    name: str = "apparatus"
    finish: Finish = WOOD_FINISH


@register_entity("filing_apparatus")
class FilingApparatus(Entity):
    def build(self, b, episode):
        cfg = b.default_shape_cfg.copy()
        cfg.mu = 0.6
        cfg.ke, cfg.kd = 60000.0, 500.0
        self.strip_body = list(b.body_label).index("strip")
        self.strip_shapes = [i for i, v in enumerate(b.shape_body) if v == self.strip_body]
        self.spindle_body = b.add_link(xform=wp.transform(DRUM, wp.quat_identity()), label="powered_spindle")
        cfg.density = 1800.0
        # Faceted convex drum provides a multipoint contact manifold.
        n = 32
        a = np.arange(n) * 2 * np.pi / n
        ring = np.column_stack((RADIUS * np.cos(a), RADIUS * np.sin(a)))
        verts = np.vstack((np.column_stack((ring, np.full(n, -HALF))), np.column_stack((ring, np.full(n, HALF)))))
        idx = []
        for i in range(n):
            j = (i + 1) % n
            idx.extend((i, j, n + j, i, n + j, n + i))
        for i in range(1, n - 1):
            idx.extend((0, i + 1, i, n, n + i, n + i + 1))
        self.bit_shape = b.add_shape_mesh(
            self.spindle_body,
            mesh=newton.Mesh(vertices=verts.astype(np.float32), indices=np.array(idx, dtype=np.int32)),
            cfg=cfg,
            color=(0.76, 0.56, 0.45),
            label="sanding_drum",
        )
        j = b.add_joint_revolute(
            parent=-1,
            child=self.spindle_body,
            parent_xform=wp.transform(DRUM, wp.quat_identity()),
            axis=(0.0, 0.0, 1.0),
            target_vel=60.0,
            target_ke=0.0,
            target_kd=0.01,
            effort_limit=0.08,
            armature=0.0002,
            actuator_mode=newton.JointTargetMode.VELOCITY,
            label="powered_tool_motor",
        )
        b.add_articulation([j], label="rotary_sander")
        self.spin_dof = int(b.joint_qd_start[j])
        # Paint cells lie on the colliding wood, 30 microns proud, no separate mechanics.
        self.local = np.array(
            [
                (x, y, 0.00603)
                for x in np.arange(-0.003, 0.0031, 0.001)
                for y in np.arange(-0.007, 0.0071, 0.001)
                if (x / 0.0035) ** 2 + (y / 0.0075) ** 2 <= 1.0
            ]
        )
        decor = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        self.paint = []
        for i, p in enumerate(self.local):
            self.paint.append(
                b.add_shape_box(
                    self.strip_body,
                    xform=wp.transform(p, wp.quat_identity()),
                    hx=0.00051,
                    hy=0.00051,
                    hz=0.00002,
                    cfg=decor,
                    color=BLUE,
                    label=f"ink_{i}",
                )
            )
        self.intensity = np.ones(len(self.local))
        self.force = 0.0
        self.speed = 0.0
        self.samples = []
        self.airborne = 0
        self.both_grips = 0
        self.supported = 0
        self.min_jaw = np.full(4, np.inf)
        self.unexpected = set()
        self.mat_transitions = []
        self.contact_trace = []
        self.elapsed = 0.0

    def bind(self, model):
        super().bind(model)
        self.base_colors = model.shape_color.numpy().copy()
        bodies = list(model.body_label)
        shape_body = model.shape_body.numpy()
        self.jaws = []
        for side in ("left", "right"):
            for jaw in ("left", "right"):
                body = next(i for i, n in enumerate(bodies) if n.endswith(f"{side}_link_{jaw}_finger"))
                self.jaws.append(np.flatnonzero(shape_body == body))
        self.allowed = {*np.concatenate(self.jaws).tolist(), self.bit_shape}

    @property
    def recolors_live(self):
        return True

    def contribute_hooks(self, hooks):
        hooks.add("after", self.measure, name="contact_abrasion", cadence=1)

    def on_scene_ready(self, pipeline):
        self._probe = ContactProbe(pipeline.program.mujoco_solver, self.model, float(pipeline.sim_dt))

    def measure(self, state, dt):
        self._probe.update()
        contact_force = self._probe.pairwise_force(self.strip_shapes, [self.bit_shape])
        self.force = abs(float(contact_force[2]))
        q = state.body_q.numpy()[self.strip_body]
        jaw_forces = np.array(
            [np.linalg.norm(self._probe.pairwise_force(self.strip_shapes, shapes)) for shapes in self.jaws]
        )
        self.elapsed += float(dt)
        rot = quat_to_mat(q[3:])
        lowest = float(q[2] - np.abs(rot[2]) @ np.array((0.0075, 0.040, 0.006)))
        airborne = lowest > 0.7505
        if airborne:
            self.airborne += 1
            self.both_grips += int(np.all(jaw_forces > 0.05))
            self.supported += int(np.all(jaw_forces[:2] > 0.05) or np.all(jaw_forces[2:] > 0.05))
            if self.elapsed <= 92.0:
                self.min_jaw = np.minimum(self.min_jaw, jaw_forces)
        if airborne or q[2] > 0.757:
            aa, bb = self._probe._shape_pairs()
            for a, b, f in zip(aa, bb, self._probe.impulses, strict=True):
                if np.linalg.norm(f) / self._probe.dt < 0.01:
                    continue
                other = int(b) if a in self.strip_shapes else int(a) if b in self.strip_shapes else -1
                if other >= 0 and other not in self.allowed:
                    label = self.model.shape_label[other]
                    if airborne:
                        self.unexpected.add(label)
                    elif label == "table":
                        self.mat_transitions.append((self.elapsed, float(q[2]), lowest))
        self.contact_trace.append(
            (self.elapsed, float(q[2]), self.force, *jaw_forces.tolist(), float(np.mean(self.intensity <= 0.1)))
        )
        self.speed = abs(float(state.joint_qd.numpy()[self.spin_dof]))
        if self.force < 0.1 or self.speed < 10.0:
            return
        q = state.body_q.numpy()[self.strip_body]
        rot = quat_to_mat(q[3:])
        pts = self.local @ rot.T + q[:3]
        radial = np.linalg.norm(pts[:, :2] - DRUM[:2], axis=1)
        # Only the upper marked face in the rotating bottom face's footprint.
        hit = (radial < RADIUS) & (np.abs(pts[:, 2] - (DRUM[2] - HALF)) < 0.0008) & (rot[2, 2] > 0.9)
        sliding = self.speed * radial
        self.intensity[hit] = np.maximum(0.0, self.intensity[hit] - float(dt) * 0.14 * self.force * sliding[hit])
        colors = self.base_colors.copy()
        colors[self.paint] = np.asarray(WOOD) + (np.asarray(BLUE) - np.asarray(WOOD)) * self.intensity[:, None]
        self.model.shape_color.assign(colors)
        self.samples.append((self.force, self.speed, int(hit.sum())))

    def reset(self, pipeline, episode, *, world=0):
        self.intensity[:] = 1.0
        self.force = 0.0
        self.speed = 0.0
        self.samples = []
        self.airborne = 0
        self.both_grips = 0
        self.supported = 0
        self.min_jaw = np.full(4, np.inf)
        self.unexpected = set()
        self.mat_transitions = []
        self.contact_trace = []
        self.elapsed = 0.0
        self.model.shape_color.assign(self.base_colors)

    def read(self):
        return dict(  # noqa: C408
            cleared=float(np.mean(self.intensity <= 0.1)),
            mean_intensity=float(self.intensity.mean()),
            force=self.force,
            speed=self.speed,
            contacts=len(self.samples),
            airborne_steps=self.airborne,
            both_grips_steps=self.both_grips,
            supported_steps=self.supported,
            min_jaw=self.min_jaw.tolist(),
            unexpected_contacts=sorted(self.unexpected),
            mat_transition_contacts=list(self.mat_transitions),
        )
