# SPDX-License-Identifier: Apache-2.0
"""Small things for the duck to play with, made in code: a bowl, a ball, a toy goal.

A round prop is drawn as a lathed mesh and collides as boxes -- a ring of wall
slabs and a floor -- which is how the Codex-built kitchen twin made its mortar and
jars: a bowl that collides as its own convex hull is a solid lump nothing falls into.
The meshes are written once into the duck's cache folder.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

import robosimstudio as rss
from assets import cache_dir


def _lathe(profile, sections: int = 64) -> trimesh.Trimesh:
    """A surface of revolution about z from ``(radius, height)`` points [m]."""
    return trimesh.creation.revolve(np.asarray(profile, dtype=np.float64), sections=sections)


def _box(size, at=(0.0, 0.0, 0.0), yaw: float = 0.0) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=size)
    m.apply_transform(trimesh.transformations.rotation_matrix(yaw, (0, 0, 1)))
    m.apply_translation(at)
    return m


def _write(name: str, mesh: trimesh.Trimesh, color) -> str:
    """An OBJ with its colour in a material of its own, which is what ``material_colors`` reads."""
    folder = cache_dir() / "props"
    folder.mkdir(parents=True, exist_ok=True)
    path, mtl = folder / f"{name}.obj", folder / f"{name}.mtl"
    lines = [f"mtllib {mtl.name}", "usemtl paint"]
    lines += ["v {:.6f} {:.6f} {:.6f}".format(*v) for v in mesh.vertices]
    lines += ["f {} {} {}".format(*(f + 1)) for f in mesh.faces]
    path.write_text("\n".join(lines) + "\n")
    mtl.write_text("newmtl paint\nKd {:.4f} {:.4f} {:.4f}\nd 1\n".format(*color))
    return str(path)


def bowl(name: str = "bowl", *, radius: float = 0.09, height: float = 0.045, at=(0.0, 0.0), color=(0.25, 0.55, 0.85)):
    """A wide, low dog bowl standing on the surface at ``at`` (work-surface frame).

    Wide because the duck's beak reaches only about as far as its toes: it can drop into a
    bowl it is standing at the rim of, not one it would have to lean over.
    """
    r, h, t = radius, height, 0.004
    profile = [
        (0, 0),
        (0.55 * r, 0),
        (0.85 * r, 0.25 * h),
        (r, h),
        (r - t, h),
        (0.85 * r - t, 0.25 * h + t),
        (0.55 * r - t, t),
        (0, t),
    ]
    walls = [
        _box((0.010, 2 * math.pi * r / 16 * 1.15, h), (0.88 * r * math.cos(a), 0.88 * r * math.sin(a), h / 2), a)
        for a in np.arange(16) * 2 * math.pi / 16
    ]
    floor = _box((1.1 * r, 1.1 * r, 2 * t), (0, 0, t))
    asset = rss.AssetSpec(
        path=_write(name, _lathe(profile), color),
        material_colors=True,
        collision_paths=tuple(_write(f"{name}_contact_{i}", m, (0.5, 0.5, 0.5)) for i, m in enumerate([floor, *walls])),
    )
    return rss.body(name=name, asset=asset, at=(at[0], at[1], 0.0), mass=0.6, friction=0.8)


def ball(name: str = "ball", *, radius: float = 0.012, at=(0.0, 0.0), color=(0.95, 0.75, 0.1)):
    """A small rubber ball, sized for the beak: 24 mm across, 8 g."""
    return rss.sphere(name=name, radius=radius, at=(at[0], at[1], radius), mass=0.008, friction=1.0, color=color)


def goal(
    name: str = "goal",
    *,
    at=(1.4, 0.0),
    width: float = 0.5,
    height: float = 0.22,
    depth: float = 0.18,
    color=(0.95, 0.95, 0.95),
):
    """A toy football goal facing -x, its mouth centred at ``at``: posts, a crossbar, a back board and a ground bar.

    The ground bar across the mouth is 8 mm high: a ball rolling in at a kick's 0.8 m/s goes over
    it, and one coming back off the back board at a fifth of that cannot climb it again.
    """
    x, y = at
    w, h, d, t = width / 2, height, depth, 0.012
    parts = {
        "post_l": ((t, t, h / 2), (x, y + w, h / 2)),
        "post_r": ((t, t, h / 2), (x, y - w, h / 2)),
        "bar": ((t, w + t, t), (x, y, h + t)),
        "ground": ((0.006, w, 0.004), (x + 0.03, y, 0.004)),
        "back": ((0.004, w, h / 2), (x + d, y, h / 2)),
        "side_l": ((d / 2, 0.004, h / 2), (x + d / 2, y + w, h / 2)),
        "side_r": ((d / 2, 0.004, h / 2), (x + d / 2, y - w, h / 2)),
    }
    frame = ("post_l", "post_r", "bar", "ground")
    return [
        rss.box(name=f"{name}_{k}", half=half, at=pos, static=True, color=color if k in frame else (0.55, 0.75, 0.55))
        for k, (half, pos) in parts.items()
    ]
