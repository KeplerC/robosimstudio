# SPDX-License-Identifier: Apache-2.0
"""The one door into the vendored cage solver: a block's ``Body``, in the cutter's scaled frame.

Imported only when a block is built, so the scan that registers the kinds never loads ``cagecut``,
its Warp kernels or SciPy. The cutter runs in the block's own frame scaled by ``s`` (one over the
longest side), the size its tolerances are written for. Lengths scale by ``s``, so for the same
motion in metres the density goes in as ``rho / s^2`` and gravity as ``g * s``; forces then scale
by ``s^2``. Sagging under its own weight with its base held, a 100 x 60 x 40 mm block at
E = 5 kPa settles 1.35, 1.37 and 1.32 mm at ``s`` = 10, 5 and 20 (``boxcut.py sag``, GPU 5).
"""

from __future__ import annotations

import numpy as np

from .cagecut.cut.blade import CutError
from .cagecut.geometry.polygon import triangulate_polygon_with_holes
from .cagecut.geometry.snapshot import piece_labels
from .cagecut.pipeline import Body
from .cagecut.scene import Material, Scene, WosParams
from .law import BASE_FACE, BlockFrame, box_cage, box_surface
from .spec import CuttableSpec

__all__ = ["CutError", "make_body", "piece_labels", "triangulate_polygon_with_holes"]

GRAVITY = 9.81
"""[m/s^2]."""


def make_body(spec: CuttableSpec, frame: BlockFrame, *, dt: float) -> Body:
    """The uncut block as a cagecut ``Body``: an eight-corner cage, its base held, its weights solved."""
    s = float(frame.scale)
    size = np.asarray(spec.size, dtype=np.float64) * s
    cage, faces = box_cage(size)
    skin, triangles = box_surface(size, float(spec.mesh_cell) * s)
    material = spec.material
    scene = Scene(
        name=spec.name,
        dim=3,
        config_path=None,
        cage_verts=cage,
        cage_faces=faces,
        mesh_verts=skin,
        mesh_tris=triangles,
        quad_points=None,
        quad_points_path=None,
        cut_specs=[],
        pinned_verts=np.asarray(faces[BASE_FACE], dtype=np.int64),
        material=Material(
            E=float(material.young_modulus),
            nu=float(material.poisson_ratio),
            density=float(material.density) / (s * s),
            rayleigh_alpha=float(spec.damping),
            rayleigh_beta=0.0,
            gravity=np.array([0.0, 0.0, -GRAVITY * s]),
            dt=float(dt),
        ),
        wos=WosParams(num_walks=int(spec.walks)),
        floor_y=None,
    )
    return Body(scene, quad_count=int(spec.quad_points), auto_cut=False, auto_next=False)
