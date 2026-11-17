"""Noncolliding dirt decals, hidden only by the brush's measured contact lattice."""

from dataclasses import dataclass

import newton
import numpy as np
import warp as wp
from workshop_assets import ASSETS, tex
from workshop_layout import LAYOUT

import robosimstudio as rss
from robosimstudio.core.render import Finish


@wp.kernel
def update_decals(
    q: wp.array(dtype=wp.transform),
    ids: wp.array(dtype=wp.int32),
    rest: wp.array(dtype=wp.transform),
    cells: wp.array(dtype=wp.int32),
    lattice: wp.array(dtype=wp.int32),
    reset: int,
):
    i = wp.tid()
    pose = rest[i]
    if reset == 0 and lattice[cells[i]] != 0:
        p = wp.transform_get_translation(pose)
        pose = wp.transform(p - wp.vec3(0.0, 0.0, 0.025), wp.transform_get_rotation(pose))
    q[ids[i]] = pose


@dataclass(frozen=True, kw_only=True)
class GrimeSpec:
    entity: str = "workshop_grime"
    name: str = "grime"
    stage: str = "deformables"
    patch: tuple = (0.5, -0.12, 0.7, 0.0)
    color: tuple = (0.45, 0.40, 0.31)


@rss.register_entity("workshop_grime")
class Grime(rss.Entity):
    domain = "static"  # Measurement display, not a handled or simulated material.

    def build(self, builder, episode):
        # The preceding brush build exposes its actual entity, not a duplicated grid definition.
        self.source = builder.workshop_brush
        grid = self.source.scrub_grid()
        self.ij, self.decals, self.poses, self.maps = [], [], [], []
        cfg = newton.ModelBuilder.ShapeConfig(density=0, has_shape_collision=False, has_particle_collision=False)
        strength = np.load(ASSETS / "grime_field.npz")["strength"]
        a, b, c, d = self.spec.patch
        for j in range(grid.ny):
            for i in range(grid.nx):
                x, y = np.asarray(grid.origin) + (np.array([i, j]) + 0.5) * grid.cell
                if not (a <= x < c and b <= y < d):
                    continue
                # The continuous 20-40 mm field defines both shape and fading tint.
                u, v = (x - a) / (c - a), (y - b) / (d - b)
                px = min(strength.shape[1] - 1, int(u * strength.shape[1]))
                py = min(strength.shape[0] - 1, int(v * strength.shape[0]))
                if strength[py, px] < 0.015:
                    continue
                paver = next(
                    (
                        p
                        for name, p in LAYOUT.items()
                        if name.startswith("paver_")
                        and abs(x - 0.6 - p["at"][0]) < p["size"][0] / 2 - grid.cell / 2
                        and abs(y - p["at"][1]) < p["size"][1] / 2 - grid.cell / 2
                    ),
                    None,
                )
                if paver is None:
                    continue
                pose = [x, y, 0.75855, 0, 0, 0, 1]
                body = builder.add_link(
                    xform=wp.transform(pose[:3], pose[3:]), is_kinematic=True, label=f"grime/decal_{j}_{i}"
                )
                joint = builder.add_joint_fixed(
                    parent=-1,
                    child=body,
                    parent_xform=wp.transform(pose[:3], pose[3:]),
                    label=f"grime/display_anchor_{j}_{i}",
                )
                builder.add_articulation([joint], label=f"grime/display_{j}_{i}")
                # UVs are continuous in world metres across cell boundaries.
                half = grid.cell * 0.5
                vertices = np.array(
                    [[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]], np.float32
                )
                uvs = (vertices[:, :2] + np.array([x - a, y - b])) / np.array([c - a, d - b])
                shape = builder.add_shape_mesh(
                    body,
                    mesh=newton.Mesh(vertices, np.array([0, 1, 2, 0, 2, 3]), uvs=uvs),
                    cfg=cfg,
                    color=self.spec.color,
                    label=f"grime_{j}_{i}",
                )
                self.maps.append(shape)
                self.ij.append((j, i))
                self.decals.append(body)
                self.poses.append(pose)
        self.cell_indices = np.array([j * grid.nx + i for j, i in self.ij], dtype=np.int32)
        self.ij = np.asarray(self.ij)
        self.poses = np.asarray(self.poses, np.float32)
        self.decals = np.asarray(self.decals)

    def apply_display(self, builder):
        super().apply_display(builder)
        for shape in self.maps:
            self.finishes[shape] = Finish(roughness=1, texture=tex("grime_blob.png"), ior=1.0)

    def bind(self, model):
        super().bind(model)
        self.ids_device = wp.array(self.decals, dtype=wp.int32, device=model.device)
        self.poses_device = wp.array(self.poses, dtype=wp.transform, device=model.device)
        self.cells_device = wp.array(self.cell_indices, dtype=wp.int32, device=model.device)

    def contribute_hooks(self, hooks):
        hooks.add("after", self.paint, name="grime-from-scrub-lattice", cadence=1)

    def paint(self, state, dt=0):
        # Hook launches one device kernel. Its sole write is q[decal_id].
        wp.launch(
            update_decals,
            dim=len(self.decals),
            inputs=[state.body_q, self.ids_device, self.poses_device, self.cells_device, self.source._grid_device, 0],
            device=state.body_q.device,
        )

    def reset(self, pipeline, episode, *, world=0):
        for state in (pipeline.state_0, pipeline.state_1):
            wp.launch(
                update_decals,
                dim=len(self.decals),
                inputs=[
                    state.body_q,
                    self.ids_device,
                    self.poses_device,
                    self.cells_device,
                    self.source._grid_device,
                    1,
                ],
                device=state.body_q.device,
            )
