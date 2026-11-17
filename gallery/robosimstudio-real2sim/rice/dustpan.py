"""A dustpan as one free rigid body: a floor that ramps up from a thin lip, two side walls and a back wall.

Built the way the cutting kind builds its knife (``EntityKnife``: several shapes on one body, on whatever solver serves
rigid bodies, which under ``rss.GRANULAR_60HZ`` is the arm's MuJoCo island): no file, no convexification of a hollow
mesh. The floor is one convex prism, 2 mm at the lip rising to ``floor`` over ``ramp``, flat to the back, so a pusher
riding the bench can follow it in; the walls are boxes ``wall`` thick (two voxels or more, which the MPM needs). The
back wall is what a hand pinches (jaws closing across it). The body frame: origin at the middle of the lip on the
bench, ``-y`` out of the mouth, ``z`` up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.scene.build.bodies import set_body_mass
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs import ObjectSpec
from robosimstudio.specs.asset import ShapeSpec


@dataclass(frozen=True, kw_only=True)
class DustpanSpec(ObjectSpec):
    entity: str = "splash_dustpan"
    shape: ShapeSpec | None = None
    inner: tuple = (0.18, 0.15)  # the floor's width (x) and depth (y) between the walls [m]
    lip: float = 0.002
    floor: float = 0.010
    ramp: float = 0.035
    wall: float = 0.010  # side walls' thickness [m]
    side_h: float = 0.035
    back: float = 0.020  # back wall's thickness [m]
    back_h: float = 0.055
    mass: float | None = 0.15
    floor_mu: float | None = (
        0.15  # the floor's friction (smooth plastic); the walls keep the builder's, the back wall is gripped
    )

    def __post_init__(self) -> None:
        if self.shape is None:
            w, d = self.inner
            half = (0.5 * w + self.wall, 0.5 * (d + self.back), 0.5 * self.back_h)
            object.__setattr__(self, "shape", ShapeSpec(kind="box", half_extents=half))
        super().__post_init__()

    def floor_mesh(self):
        """The floor prism's (vertices, triangle indices) in the body frame."""
        w, d = self.inner
        prof = [(0.0, 0.0), (d, 0.0), (d, self.floor), (self.ramp, self.floor), (0.0, self.lip)]  # (y, z), convex
        hx = 0.5 * w
        verts = [(x, y, z) for x in (-hx, hx) for (y, z) in prof]
        n = len(prof)
        tris = []
        for i in range(1, n - 1):  # the two end caps (fans), then the sides
            tris.append((0, i + 1, i))
            tris.append((n, n + i, n + i + 1))
        for i in range(n):
            j = (i + 1) % n
            tris.append((i, j, n + j))
            tris.append((i, n + j, n + i))
        v = np.asarray(verts, dtype=np.float32)
        t = np.asarray(tris, dtype=np.int32)
        # outward winding
        c = v.mean(axis=0)
        a, b, e = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
        flip = np.einsum("ij,ij->i", np.cross(b - a, e - a), a - c) < 0
        t[flip] = t[flip][:, ::-1]
        return v, t

    def union_mesh(self):
        """The floor and the walls as one watertight mesh (manifold3d union), body frame."""
        import trimesh

        v, t = self.floor_mesh()
        parts = [trimesh.Trimesh(v, t, process=True)]
        for _name, c, h in self.boxes():
            parts.append(trimesh.creation.box(extents=[2 * x for x in h]).apply_translation(c))
        m = trimesh.boolean.union(parts, engine="manifold")
        return np.asarray(m.vertices, dtype=np.float32), np.asarray(m.faces, dtype=np.int32)

    def boxes(self):
        """The walls as (name, centre, half extents) in the body frame."""
        w, d = self.inner
        hx = 0.5 * w
        return (
            (
                "side_l",
                (-(hx + 0.5 * self.wall), 0.5 * (d + self.back), 0.5 * self.side_h),
                (0.5 * self.wall, 0.5 * (d + self.back), 0.5 * self.side_h),
            ),
            (
                "side_r",
                (hx + 0.5 * self.wall, 0.5 * (d + self.back), 0.5 * self.side_h),
                (0.5 * self.wall, 0.5 * (d + self.back), 0.5 * self.side_h),
            ),
            ("back", (0.0, d + 0.5 * self.back, 0.5 * self.back_h), (hx, 0.5 * self.back, 0.5 * self.back_h)),
        )


class EntityDustpan(EntityRigid):
    spec: DustpanSpec

    def _add_primitive(self, builder, pose) -> None:
        import newton
        import warp as wp

        spec = self.spec
        xform = wp.transform(tuple(float(v) for v in pose[:3]), wp.quat(*[float(v) for v in pose[3:7]]))
        self.body = builder.add_body(xform=xform, label=spec.name)
        joint = builder.joint_count - 1
        self.coord_start = int(builder.joint_q_start[joint])
        self.dof_start = int(builder.joint_qd_start[joint])
        v, t = spec.floor_mesh()
        F = newton.ShapeFlags
        # The pieces: what MuJoCo (the hands, the bench, the pusher) touches and what is drawn; not the grains.
        pieces = []
        pieces.append(
            builder.add_shape_mesh(
                body=self.body,
                xform=wp.transform_identity(),
                mesh=newton.Mesh(v, t.reshape(-1)),
                cfg=self._contact_cfg(builder),
                label=f"{spec.name}/floor",
            )
        )
        for name, c, h in spec.boxes():
            pieces.append(
                builder.add_shape_box(
                    body=self.body,
                    xform=wp.transform(wp.vec3(*c), wp.quat_identity()),
                    hx=h[0],
                    hy=h[1],
                    hz=h[2],
                    cfg=self._contact_cfg(builder),
                    label=f"{spec.name}/{name}",
                )
            )
        for s in pieces:
            builder.shape_flags[s] = int(builder.shape_flags[s]) & ~int(F.COLLIDE_PARTICLES)
        # The grains' collider: one watertight union of the same pieces (abutting pieces merged into one collider
        # mesh leave internal faces at their seams, and the grains leaked through the floor/back-wall seam).
        uv, ut = spec.union_mesh()
        cfg = self._contact_cfg(builder)
        cfg.density = 0.0
        cfg.margin = 0.004
        if spec.floor_mu is not None:
            cfg.mu = float(spec.floor_mu)
        u = builder.add_shape_mesh(
            body=self.body,
            xform=wp.transform_identity(),
            mesh=newton.Mesh(uv, ut.reshape(-1)),
            cfg=cfg,
            label=f"{spec.name}/grains",
        )
        builder.shape_flags[u] = int(F.COLLIDE_PARTICLES)
        if spec.mass is not None:
            set_body_mass(builder, self.body, float(spec.mass))


try:
    register_entity("splash_dustpan")(EntityDustpan)
except Exception:  # already registered in this process
    pass


def dustpan(*, name: str = "dustpan", at=(0.0, 0.0, 0.0), **kwargs) -> DustpanSpec:
    return DustpanSpec(name=name, pose=at, **kwargs)


@register_entity("rice_pusher")
class EntityPusher(EntityRigid):
    """Same 170 x 20 x 40 mm block; the MPM margin belongs only to its particle shape."""

    def _add_primitive(self, builder, pose):
        import newton
        import warp as wp

        super()._add_primitive(builder, pose)
        rigid = builder.shape_count - 1
        builder.shape_flags[rigid] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
        cfg = self._contact_cfg(builder)
        cfg.density = 0.0
        cfg.margin = 0.004
        shape = builder.add_shape_box(
            body=self.body, xform=wp.transform_identity(), hx=0.085, hy=0.010, hz=0.020, cfg=cfg, label="pusher/grains"
        )
        builder.shape_flags[shape] = int(newton.ShapeFlags.COLLIDE_PARTICLES)
