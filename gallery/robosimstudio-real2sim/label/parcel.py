"""Texture compatibility for the parcel, with its original single box collider.

The display mesh has exactly the collider's boundary and is part of the same
body. No per-frame poses are written. The tape adapter only excludes this
non-colliding display mesh from the stock law's box-shape validation.
"""

from pathlib import Path

import newton
import numpy as np
import warp as wp

from robosimstudio.core.render.finish import Finish
from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.scene.registry import register_entity
from robosimstudio_materials.tape import CONTACT_FIELDS, EntityTape

A = Path(__file__).resolve().parent / "assets"


@register_entity("printed_parcel")
class PrintedParcel(EntityRigid):
    def _add_primitive(self, builder, pose):
        super()._add_primitive(builder, pose)
        self.collider = builder.shape_count - 1
        builder.shape_flags[self.collider] &= ~int(newton.ShapeFlags.VISIBLE)
        data = np.load(A / "parcel_mesh.npz")
        mesh = newton.Mesh(
            data["vertices"],
            data["faces"].ravel(),
            uvs=data["uv"],
            texture=str(A / "parcel_atlas_r3.png"),
            compute_inertia=False,
        )
        cfg = self._contact_cfg(builder)
        cfg.density = 0
        cfg.has_shape_collision = False
        cfg.has_particle_collision = False
        self.visual = builder.add_shape_mesh(body=self.body, mesh=mesh, cfg=cfg, label="parcel_textured_surface")

    def apply_display(self, builder):
        super().apply_display(builder)
        self.finishes[self.visual] = Finish(roughness=0.85, texture=str(A / "parcel_atlas_r3.png"))


@register_entity("printed_label")
class PrintedLabel(EntityTape):
    def build(self, builder, episode):
        super().build(builder, episode)
        # Initial conditions only: the freely settled label, with its original
        # in-plane rest lengths, bend rest angles, mass and tab attachments.
        # No state is rewritten during stepping or during the routine.
        rest = np.load(A / "label_rest.npz")
        for i, q in enumerate(rest["q"]):
            builder.particle_q[self._start + i] = wp.vec3(*map(float, q))
        tab = rest["tab"].astype(float)
        builder.body_q[self.tab_body] = wp.transform(wp.vec3(*tab[:3]), wp.quat(*tab[3:]))
        builder.joint_q[self.coord_start : self.coord_start + 7] = tab.tolist()
        self.tab_q0 = tab.copy()

    def _bind_contact(self, pipeline):
        # Stock _bind_contact, with one additional filter: a display-only mesh
        # is not a contact shape. The adhesion arrays and law are unchanged.
        model = pipeline.model
        device = model.device
        bodies = model.shape_body.numpy()
        types = model.shape_type.numpy()
        flags = model.shape_flags.numpy()
        mine = np.flatnonzero((bodies == self.body) & ((flags & int(newton.ShapeFlags.COLLIDE_PARTICLES)) != 0))
        boxes = mine[types[mine] == int(newton.GeoType.BOX)]
        if len(boxes) != len(mine) or not len(boxes):
            raise ValueError("Contact adhesion requires box colliders")
        self._contact = {
            "boxes": wp.array(boxes.astype(np.int32), dtype=wp.int32, device=device),
            "area": wp.array(self.bond_area.astype(np.float32), dtype=float, device=device),
            "normal": wp.array(
                np.tile(self.normal, (len(self.bonds), 1)).astype(np.float32), dtype=wp.vec3, device=device
            ),
            "dmin": wp.zeros(len(self.bonds), dtype=float, device=device),
            "params": wp.array(
                np.array([float(getattr(self.spec.material, f)) for f in CONTACT_FIELDS], dtype=np.float32),
                dtype=float,
                device=device,
            ),
        }
        self._seat_bonds()
