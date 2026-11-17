"""Appearance and initial orientation only; the shipped brush owns all dynamics."""

import contextlib

import newton
import numpy as np
import warp as wp
from workshop_assets import fibre_mesh, rounded_box, solid_fibre_texture, tex

from robosimstudio.core.render import Finish
from robosimstudio_materials.brush import EntityBrush


@contextlib.contextmanager
def oriented_brush(yaw, underside):
    old_sole, old_build, old_display = EntityBrush._sole_xform, EntityBrush.build, EntityBrush.apply_display

    def sole(self):
        p = old_sole(self)
        t = wp.transform_get_translation(p)
        return wp.transform(wp.vec3(t[0], t[1], 0.75 + underside), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), yaw))

    def build(self, builder, episode):
        old_build(self, builder, episode)
        builder.workshop_brush = self
        visible = int(newton.ShapeFlags.VISIBLE)
        for s in self.handle_shapes:
            builder.shape_flags[int(s)] &= ~visible
        cfg = builder.default_shape_cfg.copy()
        cfg.density = 0
        cfg.has_shape_collision = False
        cfg.has_particle_collision = False
        v, f = rounded_box(self.spec.sole)
        tri = v[f]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        top = (normal[:, 2] > 0) & (np.mean(tri[:, :, 2], axis=1) > self.spec.sole[2] - 0.001)
        v[:, 2] += 0.00002
        self.block_top = builder.add_shape_mesh(
            self.handle_body, mesh=newton.Mesh(v, f[top].flatten()), cfg=cfg, label="brush/block_mark"
        )
        self.fibre_shapes = []
        rng = np.random.default_rng(19)
        for i, bodies in enumerate(self.bristle_bodies):
            offsets = rng.uniform(-1, 1, (36, 2)) * 0.0053
            for seg, body in enumerate(bodies):
                for s in range(*self._spans[2]):
                    if builder.shape_body[s] == int(body):
                        builder.shape_flags[s] &= ~visible
                v, f = fibre_mesh(
                    self.spec.bristle_length / self.spec.bristle_segments,
                    0.00014,
                    offsets,
                    seg,
                    self.spec.bristle_segments,
                )
                self.fibre_shapes.append(
                    builder.add_shape_mesh(
                        int(body), mesh=newton.Mesh(v, f.flatten()), cfg=cfg, label=f"brush/fibres{i}_{seg}"
                    )
                )
        self._spans = (self._spans[0], self._spans[1], (self._spans[2][0], builder.shape_count))

    def display(self, builder):
        old_display(self, builder)
        self.finishes[self.block_top] = Finish(
            roughness=0.63, texture=tex("block_mark_v3.png"), uv_scale=(1 / 0.09, 1 / 0.18), uv_offset=(0.5, 0.5)
        )
        rng = np.random.default_rng(11)
        for s in self.fibre_shapes:
            c = tuple(np.clip(np.array(self.spec.color) * rng.choice([0.75, 0.85, 0.95, 1.05, 1.15]), 0, 1))
            builder.shape_color[s] = c
            self.finishes[s] = Finish(roughness=0.87, texture=solid_fibre_texture(c))

    EntityBrush._sole_xform, EntityBrush.build, EntityBrush.apply_display = sole, build, display
    try:
        yield
    finally:
        EntityBrush._sole_xform, EntityBrush.build, EntityBrush.apply_display = old_sole, old_build, old_display
