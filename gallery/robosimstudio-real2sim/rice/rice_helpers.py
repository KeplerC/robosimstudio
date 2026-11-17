"""Scene-local assets and render skin; no replacement rice physics."""

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"


def xy_on_bench(camera, pixels, z=0.0):
    pixels = np.asarray(pixels, float)
    matrix = camera.world_matrix
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(camera.intrinsics).T
    rays = rays @ matrix[:3, :3].T
    origin = matrix[:3, 3]
    p = origin + rays * ((0.75 + z - origin[2]) / rays[:, 2:3])
    return p - (0.6, 0.0, 0.75)


def cup_assets(size=(0.070, 0.070, 0.060)):
    ASSETS.mkdir(exist_ok=True)
    # A lathed rounded ceramic section: 70 mm mouth, 60 mm tall, 2 mm walls.
    profile = np.array(
        [
            [0, -0.030],
            [0.028, -0.030],
            [0.030, -0.0295],
            [0.031, -0.028],
            [0.0314, -0.024],
            [0.0345, 0.026],
            [0.035, 0.028],
            [0.0348, 0.0294],
            [0.034, 0.030],
            [0.033, 0.0296],
            [0.0328, 0.028],
            [0.0292, -0.024],
            [0.028, -0.026],
            [0, -0.026],
        ]
    )
    mesh = trimesh.creation.revolve(profile, sections=96)
    visual = ASSETS / "cup.obj"
    scale = np.array(size) / np.array([0.070, 0.070, 0.060])
    mesh.vertices *= scale
    mesh.export(visual)
    # Authored convex sectors keep the open mouth open without CoACD.
    paths = []
    for j in range(24):
        angles = np.linspace(j * 2 * np.pi / 24, (j + 1) * 2 * np.pi / 24, 4)
        vertices = []
        for z, outer, inner in [(-0.028, 0.031, 0.0285), (0.028, 0.035, 0.0325)]:
            for r in [inner, outer]:
                vertices.extend([[r * np.cos(a), r * np.sin(a), z] for a in angles])
        hull = trimesh.convex.convex_hull(np.array(vertices) * scale)
        path = ASSETS / f"cup_wall_{j:02d}.obj"
        hull.export(path)
        paths.append(str(path))
    floor = trimesh.creation.cylinder(radius=0.030, height=0.004, sections=48)
    floor.apply_translation([0, 0, -0.028])
    floor.vertices *= scale
    path = ASSETS / "cup_base.obj"
    floor.export(path)
    paths.append(str(path))
    return str(visual), tuple(paths)


def spill_asset(camera, outline_px, holes_px, seed=17):
    from shapely.geometry import Point, Polygon

    ASSETS.mkdir(exist_ok=True)
    poly = Polygon(xy_on_bench(camera, outline_px)[:, :2])
    for x, y, r in holes_px:
        xy = xy_on_bench(camera, [[x, y]])[0, :2]
        poly = poly.difference(Point(xy).buffer(r, resolution=12))
    # Triangulation includes interior height samples so the mound has a smooth profile.
    from scipy.spatial import Delaunay

    border = np.array(poly.exterior.coords[:-1])
    points = list(border)
    for ring in poly.interiors:
        points.extend(ring.coords[:-1])
    lo = np.array(poly.bounds[:2])
    hi = np.array(poly.bounds[2:])
    for x in np.arange(lo[0], hi[0], 0.004):
        for y in np.arange(lo[1], hi[1], 0.004):
            if poly.contains(Point(x, y)):
                points.append((x, y))
    points = np.unique(np.array(points), axis=0)
    tri = Delaunay(points)
    faces = np.array([f for f in tri.simplices if poly.covers(Polygon(points[f]))])
    neck = xy_on_bench(camera, [[1110, 530]])[0, :2]
    middle = xy_on_bench(camera, [[1220, 800]])[0, :2]
    distance = np.array([Point(p).distance(poly.boundary) for p in points])
    depth = (
        0.0025
        + 0.016 * np.exp(-np.sum(((points - neck) / (0.030, 0.025)) ** 2, axis=1))
        + 0.0030 * np.exp(-np.sum(((points - middle) / (0.060, 0.050)) ** 2, axis=1))
    )
    depth = 0.0020 + (depth - 0.0020) * np.clip(distance / 0.009, 0, 1)
    verts = np.vstack([np.c_[points, np.full(len(points), 0.0003)], np.c_[points, depth]])
    n = len(points)
    all_faces = [*faces[:, ::-1], *(faces + n)]
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    unique, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    for a, b in unique[counts == 1]:
        all_faces.extend([(a, b, b + n), (a, b + n, a + n)])
    mesh = trimesh.Trimesh(verts, np.array(all_faces), process=True)
    mesh.fix_normals()
    # A sparse halo, modelled volumes sampled by the same heap, not rigid grains.
    scatter = []
    scatter_px = [
        (948, 573),
        (825, 752),
        (855, 968),
        (881, 1027),
        (860, 1068),
        (1041, 1140),
        (1261, 1153),
        (1360, 1161),
        (1533, 681),
        (1470, 1091),
        (1515, 765),
        (1442, 548),
        (1390, 1149),
        (978, 1061),
        (838, 900),
        (1468, 1006),
        (1248, 434),
        (1283, 460),
    ]
    for px in scatter_px:
        p = xy_on_bench(camera, [px])[0, :2]
        if poly.contains(Point(p)):
            continue
        g = trimesh.creation.icosphere(subdivisions=1, radius=1)
        g.vertices *= [0.0020, 0.0018, 0.0018]
        g.apply_translation([*p, 0.0021])
        scatter.append(g)
    mesh = trimesh.util.concatenate([mesh, *scatter])
    mesh.fix_normals()
    # MPM centres the mesh's bounds. Return that centre as the placement.
    lo, hi = mesh.bounds
    at = np.array([*(0.5 * (lo + hi))[:2], lo[2]])
    mesh.apply_translation(-at)
    path = ASSETS / "spill.obj"
    mesh.export(path)
    return str(path), at, (hi - lo), border


def marble_texture(camera):
    """Procedural marble over the whole bench; analytic veins, no photo pixels."""
    ASSETS.mkdir(exist_ok=True)
    path = ASSETS / "marble_final.png"
    if path.exists():
        return str(path)
    rng = np.random.default_rng(49)
    h, w = 4096, 2048
    row, col = np.mgrid[0:h, 0:w]
    world = np.stack([0.3025 + 0.595 * col / w, 0.65 - 1.30 * row / h, np.full((h, w), 0.75005)], axis=-1)
    matrix = camera.world_matrix
    p = (world - matrix[:3, 3]) @ matrix[:3, :3]
    xx = (p[..., 0] / p[..., 2] * camera.intrinsics[0, 0] + 960) / 1920
    yy = (p[..., 1] / p[..., 2] * camera.intrinsics[1, 1] + 640) / 1280
    noise = gaussian_filter(rng.normal(size=(h, w)).astype(np.float32), 25)
    noise /= noise.std()
    fine = gaussian_filter(rng.normal(size=(h, w)).astype(np.float32), 2)
    fine /= fine.std()
    axis = xx - (0.72 * (yy - 0.29) + 0.010 * np.sin(23 * yy) + 0.005 * np.sin(67 * yy))
    modulation = 0.6 + 0.4 * np.sin(36 * yy) ** 2
    vein = (
        (np.exp(-((axis / 0.006) ** 2)) * 0.125 + np.exp(-((axis / 0.018) ** 2)) * 0.045)
        * modulation
        * (0.15 + 0.85 * np.exp(-(((yy - 0.50) / 0.35) ** 2)))
    )
    hair = np.exp(-(((xx - (0.38 + 0.32 * (yy - 0.60) + 0.008 * np.sin(54 * yy))) / 0.001) ** 2)) * 0.009
    level = (
        0.870
        + 0.010 * np.clip(xx, 0, 1)
        + 0.006 * np.clip(yy, 0, 1)
        - 0.018 * (1 - np.clip(yy, 0, 1)) * (0.5 + np.clip(xx, 0, 1))
        + 0.025 * np.clip(yy, 0, 1) * np.clip(xx, 0, 1)
        + 0.0035 * noise
        + 0.0015 * fine
        - vein
        - hair
    )
    rgb = np.stack([level - 0.022, level + 0.008, level + 0.024], axis=-1)
    temporary = path.with_suffix(".tmp.png")
    Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255)).save(temporary)
    temporary.replace(path)
    return str(path)


class RiceSkin:
    """Long-grain mesh bound one-to-one to MPM particles via the public surface callback.

    Orientation is deterministic per particle, not a separately simulated rigid rotation.
    """

    def __init__(self, n):
        rng = np.random.default_rng(816)
        rings = np.linspace(-0.995, 0.995, 9)
        angles_mesh = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        v = np.array(
            [[x, (1 - x * x) ** 0.46 * np.cos(a), (1 - x * x) ** 0.46 * np.sin(a)] for x in rings for a in angles_mesh]
        )
        # A shallow longitudinal furrow, cut into the bran on the upper face.
        v[:, 2] -= 0.18 * np.exp(-((v[:, 1] / 0.22) ** 2)) * np.clip(v[:, 2], 0, 1)
        f = []
        for i in range(8):
            for j in range(12):
                a = i * 12 + j
                b = i * 12 + (j + 1) % 12
                c = b + 12
                d = a + 12
                f.extend([(a, b, c), (a, c, d)])
        for j in range(1, 11):
            f.extend([(0, j + 1, j), (96, 96 + j, 96 + j + 1)])
        f = np.asarray(f)
        self.faces_per_grain = len(f)
        tile = rng.integers(0, 8, n)
        self.tile = tile
        angles = rng.uniform(0, 2 * np.pi, n)
        tilt = rng.normal(0, 0.25, n)
        roll = rng.choice([0.0, np.pi], n) + rng.normal(0, 0.22, n)
        length = rng.uniform(0.0060, 0.0072, n)
        width = rng.uniform(0.0018, 0.00215, n)
        transforms = Rotation.from_euler("ZYX", np.c_[angles, tilt, roll]).as_matrix()
        local = v[None, :, :] * np.stack([length / 2, width / 2, width * 0.42], axis=-1)[:, None, :]
        self.offset = np.einsum("nij,nvj->nvi", transforms, local).astype(np.float32)
        self.floor_z = 0.75005 - self.offset[:, :, 2].min(axis=1)
        self.triangles = (f[None, :, :] + np.arange(n)[:, None, None] * len(v)).reshape(-1, 3).astype(np.int32)
        # Most particle centres rest on the bench plane; the visual grain has finite thickness.
        self.n = n

    def __call__(self, positions):
        centers = np.asarray(positions).copy()
        centers[:, 2] = np.maximum(centers[:, 2], self.floor_z)
        return (centers[:, None, :] + self.offset).reshape(-1, 3), self.triangles


def install_bake(sim, path):
    """Restore a measured particle end state and make Reset use it too."""
    data = np.load(path)
    ref = sim["rice"]
    ent = ref.entity
    pipe = sim.pipeline
    if len(data["positions"]) != len(ref.particles):
        raise ValueError("Rice bake and sampled mesh have different particle counts; regenerate the bake.")
    source = data["anchor_pose"]
    target = np.array([*ref.spec.at, *ref.spec.quat])
    origin = np.array([0.6, 0, 0.75])
    local = (data["positions"] - origin - source[:3]) @ Rotation.from_quat(source[3:]).as_matrix()
    placed = local @ Rotation.from_quat(target[3:]).as_matrix().T + origin + target[:3]
    q = pipe.state_0.particle_q.numpy()
    q[ref.particles] = placed
    for state in (pipe.state_0, pipe.state_1):
        state.particle_q.assign(q)
        state.particle_qd.zero_()
    ent.snapshot_rest(pipe)
    import newton

    pipe.reset_solvers(flags=int(newton.StateFlags.PARTICLE))


def dress_rice(kit, skin, finish):
    """Procedural bran bands on the live mesh; each grain follows one particle."""
    if not hasattr(kit, "stage"):
        return
    from pxr import UsdGeom

    from robosimstudio.core.render import Finish

    palette = finish["grain_palette"]
    # One narrow longitudinal band, with its side varied from grain to grain.
    side = np.where(skin.tile % 2, 1, 4)
    segments = np.repeat(np.tile(np.arange(12), 8), 2)
    segments = np.r_[segments, np.full(20, -1)]
    bands = segments[None, :] == side[:, None]
    group = (skin.tile[:, None] + 8 * bands).reshape(-1)
    for prim in kit.stage.Traverse():
        if str(prim.GetPath()).endswith("/deformables/rice") and prim.IsA(UsdGeom.Mesh):
            for i in range(16):
                subset = UsdGeom.Subset.CreateGeomSubset(
                    UsdGeom.Mesh(prim),
                    f"bran_{i}",
                    UsdGeom.Tokens.face,
                    np.flatnonzero(group == i).tolist(),
                    "materialBind",
                    UsdGeom.Tokens.partition,
                )
                colour = palette[i % 8]
                rgb = np.array([int(colour[j : j + 2], 16) / 255 for j in (1, 3, 5)])
                if i >= 8:
                    rgb = np.clip(rgb * finish["bran_highlight_scale"] + (0.055, 0.045, 0.025), 0, 1)
                kit.library.bind(subset.GetPrim(), Finish(roughness=finish["roughness"]), tuple(rgb))
