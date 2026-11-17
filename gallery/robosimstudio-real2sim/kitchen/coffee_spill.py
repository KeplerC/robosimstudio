"""The October coffee spill, sampled as grounds and individual bean particles.

Assets are generated locally under assets/spill. The pan and pusher are free
rigid bodies held by the stock YAM hands. No force or servo overrides.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


BENCH = (0.6, 0.0, 0.75)
SPILL_AT = (0.0, -0.255)
PUSHER_AT = (0.0, -0.355)
PUSHER_HALF = (0.085, 0.01, 0.02)
PAN_AT = (0.0, -0.165)  # the middle of the dustpan's lip
MESHES = HERE / "assets" / "spill"
BEAN_SPACING = 0.006
BEAN_CUBE = 0.005


def _outline(rx: float, ry: float, seed: int) -> np.ndarray:
    """An irregular closed outline (a spill), 96 points, centred on the origin."""
    rng = np.random.default_rng(seed)
    k = rng.uniform(0, 2 * math.pi, 5)
    a = np.linspace(0, 2 * math.pi, 96, endpoint=False)
    r = (
        1.0
        + 0.04 * np.sin(3 * a + k[0])
        + 0.035 * np.sin(5 * a + k[1])
        + 0.02 * np.sin(7 * a + k[2])
        + 0.012 * np.sin(11 * a + k[3])
    )
    xy = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
    return xy / np.max(np.abs(xy), axis=0) * (rx, ry)


def spill_mesh(rx: float = 0.06, ry: float = 0.05, h: float = 0.018, seed: int = 11) -> str:
    """A low irregular frustum, with a 70% top outline and sloping sides."""
    import trimesh

    path = MESHES / f"grounds_frustum_{rx * 1e3:.0f}x{ry * 1e3:.0f}x{h * 1e3:.0f}_{seed}.obj"
    if path.exists():
        return str(path)
    outline = _outline(rx, ry, seed)
    n = len(outline)
    verts = np.vstack([np.c_[outline, np.zeros(n)], np.c_[0.7 * outline, np.full(n, h)], [[0, 0, 0], [0, 0, h]]])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.extend([(2 * n, j, i), (2 * n + 1, n + i, n + j), (i, j, n + j), (i, n + j, n + i)])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    MESHES.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.obj")
    mesh.export(str(tmp))
    tmp.replace(path)
    return str(path)


def bean_centres(rx: float, ry: float, h: float, seed: int, n_floor: int, n_top: int, corridor=(0.072, 0.088, 0.075)):
    """Bean lattice cells (i, j, k) in the spill's frame: ``n_floor`` on the bench round the patch (outside its outline
    by 3 mm, inside the sweep corridor |x| < cx, -cy0 < y < cy1), ``n_top`` on top of it (k = the layer over ``h``)."""
    from shapely.geometry import Point, Polygon

    poly = Polygon(_outline(rx, ry, seed))
    top_poly = Polygon(0.7 * _outline(rx, ry, seed))
    rng = np.random.default_rng(seed + 100)
    s = BEAN_SPACING
    k_top = int(math.ceil(h / s))
    cx, cy0, cy1 = corridor
    ii = np.arange(-int(cx / s), int(cx / s) + 1)
    jj = np.arange(-int(cy0 / s), int(cy1 / s) + 1)
    cells = [(i, j) for i in ii for j in jj]
    rng.shuffle(cells)
    floor, top = [], []
    for i, j in cells:
        p = Point(i * s, j * s)
        d = poly.exterior.distance(p)
        inside = poly.contains(p)
        if top_poly.contains(p) and top_poly.exterior.distance(p) > 0.004 and len(top) < n_top:
            top.append((i, j, k_top))
        elif not inside and d > 0.004 and len(floor) < n_floor:
            # denser near the patch: keep a far cell with a probability falling off with distance
            if rng.random() < math.exp(-d / 0.03):
                floor.append((i, j, 0))
    return floor + top


def bean_mesh(rx: float = 0.06, ry: float = 0.05, h: float = 0.018, seed: int = 11, n_floor: int = 30, n_top: int = 12):
    """The beans' body: one ``BEAN_CUBE`` cube per bean on a ``BEAN_SPACING`` lattice (the lattice the heap samples on
    this mesh's bound puts exactly one point in each cube). Returns (path, cells)."""
    import trimesh

    cells = bean_centres(rx, ry, h, seed, n_floor, n_top)
    path = MESHES / f"beans_frustum_{rx * 1e3:.0f}x{ry * 1e3:.0f}x{h * 1e3:.0f}_{seed}_{n_floor}_{n_top}.obj"
    if not path.exists():
        s, c = BEAN_SPACING, BEAN_CUBE
        boxes = [
            trimesh.creation.box(extents=(c, c, c)).apply_translation((i * s, j * s, k * s + 0.5 * s))
            for i, j, k in cells
        ]
        mesh = trimesh.util.concatenate(boxes)
        MESHES.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.obj")
        mesh.export(str(tmp))
        tmp.replace(path)
    return str(path), cells


def grounds_material(density: float = 420.0, friction: float = 0.78):
    from robosimstudio_materials.mpm import SAND

    return dataclasses.replace(SAND.spec, density=float(density), friction=float(friction), color=(0.20, 0.12, 0.07))


def beans_material(density: float = 650.0):
    from robosimstudio_materials.mpm import BEANS

    return dataclasses.replace(BEANS.spec, density=float(density), color=(0.38, 0.19, 0.075))


def add_spill(
    scene,
    shift=0.12,
    dx=-0.02,
    spill_r=(0.062, 0.083),
    spill_h=0.0115,
    seed=29,
    beans=(16, 14),
    margin=0.004,
    spacing=0.0025,
    density=420.0,
    friction=0.78,
    bean_density=650.0,
    pan=None,
):
    from dustpan import dustpan

    import robosimstudio as rss
    from robosimstudio import AssetSpec

    sx, sy = SPILL_AT[0] + dx, SPILL_AT[1] + shift
    rx, ry = spill_r
    ol = _outline(rx, ry, seed)
    bc = 0.5 * (
        ol.min(axis=0) + ol.max(axis=0)
    )  # a mesh heap's ``at`` is its bound's bottom centre, not the outline's origin
    scene.add(
        rss.heap(
            name="grounds",
            shape="mesh",
            asset=AssetSpec(path=spill_mesh(rx, ry, spill_h, seed)),
            at=(sx + bc[0], sy + bc[1], 0.0),
            material=grounds_material(density, friction),
            containers=("pusher",),
            container_margin=margin,
            particle_spacing=float(spacing),
        )
    )
    bean_path, cells = bean_mesh(rx, ry, spill_h, seed, *beans)
    lo = np.array([min(c[0] for c in cells), min(c[1] for c in cells)]) * BEAN_SPACING
    hi = np.array([max(c[0] for c in cells), max(c[1] for c in cells)]) * BEAN_SPACING
    mid = 0.5 * (
        lo + hi
    )  # the bound's bottom-centre sits at ``at``: put it where the cells' middle is in the spill frame
    z0 = min(c[2] for c in cells) * BEAN_SPACING + 0.5 * (BEAN_SPACING - BEAN_CUBE)
    scene.add(
        rss.heap(
            name="beans",
            shape="mesh",
            asset=AssetSpec(path=bean_path),
            at=(sx + mid[0], sy + mid[1], z0),
            material=beans_material(bean_density),
            containers=("pusher",),
            container_margin=margin,
            particle_spacing=BEAN_SPACING,
            jitter=0.15,
        )
    )
    px, py = PUSHER_AT[0] + dx, PUSHER_AT[1] + shift
    scene.add(
        rss.box(name="pusher", half=PUSHER_HALF, at=(px, py, PUSHER_HALF[2]), friction=0.3, color=(0.55, 0.38, 0.22))
    )
    ax, ay = PAN_AT[0] + dx, PAN_AT[1] + shift
    scene.add(
        dustpan(
            name="dustpan",
            at=(ax, ay, 0.0),
            color=(0.70, 0.78, 0.76),
            obb_grasp=False,
            **{k: (tuple(v) if isinstance(v, list) else v) for k, v in (pan or {}).items()},
        )
    )
