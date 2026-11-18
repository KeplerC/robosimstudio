"""Geometry and materials. Only the two canvas prints use cropped reference pixels."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay


class PolygonPath:
    def __init__(self, points):
        self.points = np.asarray(points)

    def contains_points(self, points, radius=0):
        p = np.asarray(points)
        inside = np.zeros(len(p), bool)
        distance = np.full(len(p), np.inf)
        for a, b in zip(self.points, np.roll(self.points, -1, axis=0), strict=True):
            dy = b[1] - a[1]
            if abs(dy) > 1e-12:
                crossing = ((a[1] > p[:, 1]) != (b[1] > p[:, 1])) & (
                    p[:, 0] < (b[0] - a[0]) * (p[:, 1] - a[1]) / dy + a[0]
                )
                inside ^= crossing
            ab = b - a
            t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-10), 0, 1)
            distance = np.minimum(distance, np.linalg.norm(p - a - t[:, None] * ab, axis=1))
        return (inside | (distance <= radius)) if radius >= 0 else (inside & (distance >= -radius))


import trimesh

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
LEAF_NORMALS = {}


def versioned_path(path):
    return str(path)


ORIGIN = np.array([0.6, 0.0, 0.75])
EYE = np.array([-0.50, 0.0, 1.18])
LOOK = np.array([0.6, 0.0, 1.03])
FOV = 38.5
FORWARD = (LOOK - EYE) / np.linalg.norm(LOOK - EYE)
RIGHT = np.cross(FORWARD, [0.0, 0.0, 1.0])
RIGHT /= np.linalg.norm(RIGHT)
UP = np.cross(RIGHT, FORWARD)
FOCAL = 640 / np.tan(np.deg2rad(FOV / 2))


def rgb(h):
    return tuple(int(h.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4))


def unproject(pixel, depth_x=0.0):
    u, v = pixel
    ray = FORWARD + (u - 960) / FOCAL * RIGHT + (640 - v) / FOCAL * UP
    return EYE + ray * ((0.6 + depth_x - EYE[0]) / ray[0]) - ORIGIN


def write_mesh(name, vertices, faces, uvs=None):
    path = ASSETS / f"{name}.obj"
    with path.open("w") as f:
        for v in vertices:
            f.write("v " + " ".join(f"{a:.9f}" for a in v) + "\n")
        if uvs is not None:
            for uv in uvs:
                f.write(f"vt {uv[0]:.8f} {uv[1]:.8f}\n")
        for tri in faces:
            f.write("f " + " ".join(f"{i + 1}/{i + 1}" if uvs is not None else str(i + 1) for i in tri) + "\n")
    return versioned_path(path)


def lathe(name, profile, ribs=0):
    n = 128
    verts = []
    for radius, z in profile:
        for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
            r = radius + (0.0014 * max(0, np.cos(ribs * a)) ** 8 if ribs else 0)
            verts.append([r * np.cos(a), r * np.sin(a), z])
    faces = []
    for j in range(len(profile) - 1):
        for i in range(n):
            a = j * n + i
            b = j * n + (i + 1) % n
            faces.extend([[a, b, b + n], [a, b + n, a + n]])
    return write_mesh(name, verts, faces)


def make_pot():
    base = lathe(
        "pot_base",
        [
            (0, 0),
            (0.047, 0),
            (0.050, 0.004),
            (0.061, 0.126),
            (0.068, 0.128),
            (0.070, 0.132),
            (0.068, 0.136),
            (0.061, 0.136),
            (0.058, 0.131),
            (0.046, 0.007),
            (0, 0.007),
        ],
        12,
    )
    body = trimesh.load(base, force="mesh")
    vertices = np.asarray(body.vertices).copy()
    for k, point in enumerate(vertices):
        r = np.linalg.norm(point[:2])
        if r > 0.043 and point[2] < 0.008:
            angle = np.arctan2(point[1], point[0])
            notch = 0.0
            for a in [np.pi - 0.58, np.pi + 0.58, 0.58, -0.58]:
                delta = np.arctan2(np.sin(angle - a), np.cos(angle - a)) / 0.13
                notch = max(notch, 0.010 * np.sqrt(max(0.0, 1 - delta * delta)))
            vertices[k, 2] = max(point[2], notch)
    body.vertices = vertices
    parts = [body]
    # Raised moulding on the visible face, with the four horizontal braces.
    for z in [0.025, 0.055, 0.085, 0.115]:
        radius = 0.05 + 0.011 * z / 0.126
        bar = trimesh.creation.box(extents=[0.0024, 0.036, 0.002])
        bar.apply_translation([-radius - 0.001, 0, z])
        parts.append(bar)
    for side in [-1, 1]:
        # The ribs follow the nursery pot's taper.
        points = np.array([[-0.049, side * 0.018, 0.003], [-0.062, side * 0.018, 0.125]])
        rail = trimesh.creation.cylinder(radius=0.0014, segment=points, sections=10)
        parts.append(rail)
    mesh = trimesh.util.concatenate(parts)
    return write_mesh("pot", mesh.vertices, mesh.faces)


def make_saucer():
    return lathe(
        "saucer",
        [
            (0, 0),
            (0.069, 0),
            (0.074, 0.003),
            (0.075, 0.007),
            (0.073, 0.010),
            (0.069, 0.010),
            (0.066, 0.005),
            (0, 0.005),
        ],
    )


def make_shade(name):
    return lathe(
        name,
        [
            (0.067, 0.110),
            (0.076, 0.090),
            (0.104, 0.040),
            (0.127, 0.005),
            (0.128, 0),
            (0.125, -0.003),
            (0.123, 0.003),
            (0.101, 0.040),
            (0.074, 0.090),
            (0.065, 0.110),
        ],
    )


def canvas_texture(name, poppies=False):
    return str(ASSETS / ("flower_restored.png" if name.endswith("right") else "teal_yellow_restored.png"))


def soil_texture():
    rng = np.random.default_rng(8)
    a = np.clip(rng.normal(0, 10, (512, 512, 1)) + np.array([52, 35, 23]), 0, 255).astype("uint8")
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    for _ in range(1000):
        x, y = rng.integers(0, 512, 2)
        r = int(rng.integers(1, 5))
        d.ellipse((x - r, y - r, x + r, y + r), fill=(99, 80, 54) if r < 4 else (139, 125, 95))
    path = ASSETS / "soil_surface.png"
    if not path.exists():
        im.save(path)
    return versioned_path(path)


def leaf_point(row, p):
    return unproject(p, row["depth"])


def leaf_texture(row, mean, vt, uvmin, span):
    size = 1024
    rng = np.random.default_rng(row["seed"])
    yy, xx = np.mgrid[:size, :size]
    noise = rng.normal(0, 1.7, (size, size))
    mottling = 3 * np.sin(xx * 0.017 + np.sin(yy * 0.011) * 3) + 2 * np.cos(yy * 0.033 + xx * 0.007)
    base = np.array([int(row["color"][i : i + 2], 16) for i in (1, 3, 5)])
    im = Image.fromarray(
        np.clip(
            base
            + (noise + 2.8 * mottling + 8 * np.sin(xx * 0.025 + 5 * np.sin(yy * 0.009)))[:, :, None]
            * np.array([0.5, 1.1, 0.5]),
            0,
            255,
        ).astype("uint8")
    )
    d = ImageDraw.Draw(im)

    def uv_point(p):
        xyz = leaf_point(row, p) + ORIGIN
        uv = ((xyz - mean) @ vt[:2].T - uvmin) / span
        return (float(uv[0] * size), float((1 - uv[1]) * size))

    junction = np.array(row["junction"], float)
    tip = np.array(row["tip"], float)
    vein = tuple(int(row.get("vein", "#a8b878")[i : i + 2], 16) for i in (1, 3, 5))
    d.line([uv_point(p) for p in [row["outline"][0], junction, tip]], fill=vein, width=15, joint="curve")
    d.line([uv_point(junction), uv_point(row["lobe_b"])], fill=vein, width=11)
    for t, left, right in row["veins"]:
        origin = junction * (1 - t) + tip * t
        for endpoint in (left, right):
            end = np.array(endpoint, float)
            bend = origin * 0.42 + end * 0.58 + np.array([0, -4])
            d.line(
                [uv_point((1 - s) ** 2 * origin + 2 * s * (1 - s) * bend + s * s * end) for s in np.linspace(0, 1, 30)],
                fill=vein,
                width=7,
                joint="curve",
            )
            for s in (0.4, 0.7):
                branch = origin * (1 - s) + end * s
                target = branch + (tip - junction) * 0.07
                d.line([uv_point(branch), uv_point(target)], fill=tuple(int(v * 0.58) for v in vein), width=1)
    path = ASSETS / (row["name"] + ".png")
    im.save(path)
    texture = versioned_path(path)
    # Fine surface relief belongs to the finish; the coarse blade remains cloth.
    height = 0.00020 * np.sin(xx * 0.085 + 2 * np.sin(yy * 0.027)) + 0.00007 * np.sin(
        yy * 0.12 + 3 * np.sin(xx * 0.023)
    )
    dy, dx = np.gradient(height)
    nx = -dx * size / span[0]
    ny = dy * size / span[1]
    normal = np.stack([nx, ny, np.ones_like(nx)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    normal_path = ASSETS / (row["name"] + "_normal.png")
    Image.fromarray(np.uint8(np.clip((normal * 0.5 + 0.5) * 255, 0, 255))).save(normal_path)
    row["normal_texture"] = versioned_path(normal_path)
    LEAF_NORMALS[texture] = row["normal_texture"]
    return texture


def leaf_mesh(row):
    outline = np.array(row["outline"], float)
    from scipy.interpolate import CubicSpline

    closed = np.vstack([outline, outline[0]])
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    outline = CubicSpline(arc, closed, bc_type="periodic")(np.linspace(0, arc[-1], 2 * len(outline), endpoint=False))
    lo, hi = outline.min(axis=0), outline.max(axis=0)
    polygon = PolygonPath(outline)
    samples = np.array([(u, v) for u in np.linspace(lo[0], hi[0], 12) for v in np.linspace(lo[1], hi[1], 15)])
    samples = samples[polygon.contains_points(samples, radius=-5.0)]
    points = np.unique(np.vstack([outline, samples, np.array(row["junction"])[None, :]]), axis=0)
    tri = Delaunay(points).simplices
    checks = [
        points[tri].mean(axis=1),
        (points[tri[:, 0]] + points[tri[:, 1]]) / 2,
        (points[tri[:, 1]] + points[tri[:, 2]]) / 2,
        (points[tri[:, 2]] + points[tri[:, 0]]) / 2,
    ]
    tri = tri[np.all(np.stack([polygon.contains_points(c, radius=0.3) for c in checks]), axis=0)]
    used = np.unique(tri)
    mapping = np.full(len(points), -1)
    mapping[used] = np.arange(len(used))
    points = points[used]
    tri = mapping[tri]
    xyz = np.array([leaf_point(row, p) for p in points])
    uv = (points - lo) / (hi - lo)
    # Flat rest geometry; gravity creates the shallow droop.
    junction = leaf_point(row, row["junction"])
    center = xyz.mean(axis=0)
    path = write_mesh(row["name"] + "_authored", xyz - center, tri)
    geometry_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    loaded = trimesh.load(path, force="mesh")
    local = np.array(loaded.vertices)
    world = (local + center + ORIGIN).astype(np.float32).astype(float)
    mean = world.mean(axis=0)
    _, _, vt = np.linalg.svd(world - mean, full_matrices=False)
    uv = (world - mean) @ vt[:2].T
    uvmin = uv.min(axis=0)
    uv -= uvmin
    span = uv.max(axis=0)
    authored_points = local + center
    authored_uvs = uv.copy()
    texture = leaf_texture(row, mean, vt, uvmin, span)
    held = np.argsort(np.linalg.norm(local + center - junction, axis=1))[:5]
    baked = read_bake().get(row["name"], {})
    if baked.get("authored_geometry_hash") == geometry_hash:
        from scipy.spatial import cKDTree

        xyz = np.asarray(baked["points"])
        origin = xyz.mean(axis=0)
        target = authored_points[cKDTree(authored_uvs).query(np.asarray(baked["authored_uvs"]))[1]]
        target_center = target.mean(axis=0)
        U, _, Vt = np.linalg.svd((xyz - origin).T @ (target - target_center))
        if np.linalg.det(U @ Vt) < 0:
            U[:, -1] *= -1
        rotation = U @ Vt
        xyz = (xyz - origin) @ rotation + target_center
        held_points = (np.asarray(baked["held_points"]) - origin) @ rotation + target_center
        row.update(bake_rotation_row=rotation.tolist(), bake_origin=origin.tolist(), bake_target=target_center.tolist())
        center = xyz.mean(axis=0)
        path = write_mesh(row["name"] + "_settled", xyz - center, baked["triangles"])
        loaded = trimesh.load(path, force="mesh")
        local = np.array(loaded.vertices)
        order = cKDTree(xyz).query(local + center)[1]
        uv = np.asarray(baked["authored_uvs"])[order]
        held = np.unique(cKDTree(local + center).query(held_points)[1])
        junction = np.mean(held_points, axis=0)
    row.update(
        at=center.tolist(),
        size=np.ptp(local, axis=0).tolist(),
        quat=[0, 0, 0, 1],
        asset=path,
        texture=texture,
        uv_scale=(1 / span).tolist(),
        authored_uvs=uv.tolist(),
        authored_geometry_hash=geometry_hash,
        held_indices=held.tolist(),
        particle_count=len(local),
        triangle_count=len(loaded.faces),
        junction_xyz=junction.tolist(),
        baked=bool(baked.get("authored_geometry_hash") == geometry_hash),
    )
    return row


def stem_nodes(row, index):
    root = np.array(
        [
            [0.018, 0.006, 0.13],
            [0, 0.025, 0.13],
            [-0.023, 0.01, 0.13],
            [-0.012, -0.006, 0.13],
            [0.013, -0.026, 0.13],
            [-0.024, -0.026, 0.13],
            [0.032, 0.026, 0.13],
        ][index]
    )
    root[0] *= 0.5
    root[1] = (root[1] + 0.014) * 0.4 - 0.014
    end = np.array(row["junction_xyz"])
    t = np.linspace(0, 1, 9)[:, None]
    mid = root * 0.45 + end * 0.55
    mid[2] += 0.135 if index == 4 else 0.080 if index in (1, 5) else 0.055
    mid[0] += 0.016
    return ((1 - t) ** 2 * root + 2 * (1 - t) * t * mid + t * t * end).tolist()


def make_canvas(name, width, height):
    p = [
        [-0.004, width / 2, -height / 2],
        [-0.004, -width / 2, -height / 2],
        [-0.004, -width / 2, height / 2],
        [-0.004, width / 2, height / 2],
    ]
    return write_mesh(name, p, [[0, 1, 2], [0, 2, 3]], [[0, 0], [1, 0], [1, 1], [0, 1]])


def read_bake():
    p = ASSETS / "plant_settled.json"
    return json.loads(p.read_text()) if p.exists() else {}


def wood_texture():
    return str(ASSETS / "table_veneer.png")


def lights(kit):
    """Visible emissive interior disks, plus a broad warm photographic fill."""
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

    from robosimstudio.core.render import Finish

    for prim in kit.stage.Traverse():
        if prim.GetName() != "DiffuseTexture":
            continue
        asset = str(prim.GetAttribute("inputs:file").Get())
        parent = str(prim.GetPath().GetParentPath())
        shader = UsdShade.Shader.Get(kit.stage, parent + "/PreviewSurface")
        if "/leaf_" in asset:
            shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
            shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.003, 0.003, 0.003))
            normal_path = LEAF_NORMALS.get(prim.GetAttribute("inputs:file").Get().path)
            if normal_path:
                normal = UsdShade.Shader.Define(kit.stage, parent + "/NormalTexture")
                normal.CreateIdAttr("UsdUVTexture")
                normal.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(normal_path))
                normal.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
                normal.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                normal.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
                source = UsdShade.Shader(prim).GetInput("st").GetConnectedSource()
                if source:
                    normal.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(source[0], source[1])
                normal.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                    normal.ConnectableAPI(), "rgb"
                )
        if "/table_veneer" in asset:
            UsdShade.Shader(prim).CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1.12, 0.86, 0.80, 1))
            shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
            shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.12, 0.12, 0.12))
            shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(0.8)
            shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.12)
    for key, pixel in [("left", [483, 46]), ("right", [1132, 48])]:
        center = unproject(pixel, -0.05) + ORIGIN
        lamp_xf = UsdGeom.Xform.Define(kit.stage, "/root/pendant_light_" + key)
        lamp_xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, center - np.array([0, 0, 0.01]))))
        bulb = UsdLux.RectLight.Define(kit.stage, str(lamp_xf.GetPath()) + "/light")
        bulb.GetWidthAttr().Set(0.22)
        bulb.GetHeightAttr().Set(0.22)
        bulb.GetIntensityAttr().Set(15000)
        bulb.GetColorAttr().Set(Gf.Vec3f(1.0, 0.86, 0.63))
        bounce = UsdGeom.Xform.Define(kit.stage, "/root/pendant_wall_" + key)
        bounce.AddTranslateOp().Set(Gf.Vec3d(*map(float, center - np.array([0, 0, 0.025]))))
        bounce.AddRotateYOp().Set(-32)
        warm = UsdLux.DiskLight.Define(kit.stage, str(bounce.GetPath()) + "/light")
        warm.GetRadiusAttr().Set(0.055)
        warm.GetIntensityAttr().Set(18000)
        warm.GetColorAttr().Set(Gf.Vec3f(1.0, 0.72, 0.45))
        UsdLux.ShapingAPI.Apply(warm.GetPrim()).CreateShapingConeAngleAttr().Set(45)
        angle = 0.13
        rot = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        a = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        p = np.vstack(
            [[0, 0, -0.003], np.column_stack([0.126 * np.cos(a), 0.126 * np.sin(a), np.full(len(a), -0.003)])]
        )
        p = p @ rot.T + center
        tri = [[0, 1 + (i + 1) % len(a), 1 + i] for i in range(len(a))]
        name = "glow_" + key
        kit.mesh(name, p, tri, color=(1, 1, 0.94), finish=Finish(roughness=0.9))
        mesh = kit.stage.GetPrimAtPath("/root/scenery/" + name)
        material = UsdShade.Material.Define(kit.stage, "/root/Materials/emission_" + key)
        surface = UsdShade.Shader.Define(kit.stage, "/root/Materials/emission_" + key + "/surface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.9, 0.9, 0.9))
        surface.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(8, 7.8, 7.1))
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material)
    xf = UsdGeom.Xform.Define(kit.stage, "/root/photo_fill")
    xf.AddTranslateOp().Set(Gf.Vec3d(-0.25, 0.65, 1.65))
    xf.AddRotateZOp().Set(-37)
    xf.AddRotateYOp().Set(-55)
    panel = UsdLux.RectLight.Define(kit.stage, "/root/photo_fill/light")
    panel.GetWidthAttr().Set(1.6)
    panel.GetHeightAttr().Set(0.8)
    panel.GetIntensityAttr().Set(1200)
    panel.GetColorAttr().Set(Gf.Vec3f(0.90, 0.93, 1.0))


def make_clock():
    im = Image.new("RGB", (512, 512), (215, 214, 205))
    d = ImageDraw.Draw(im)
    d.ellipse((5, 5, 507, 507), outline=(179, 177, 164), width=16)
    for i in range(60):
        a = 2 * np.pi * i / 60
        r = 202 if i % 5 == 0 else 215
        d.line(
            (256 + r * np.sin(a), 256 - r * np.cos(a), 256 + 228 * np.sin(a), 256 - 228 * np.cos(a)),
            fill=(145, 146, 137),
            width=4 if i % 5 == 0 else 2,
        )
    # Simple geometric numeral six, at the lower edge of the cropped dial.
    d.arc((244, 412, 268, 446), 10, 350, fill=(132, 135, 125), width=4)
    d.line((245, 430, 244, 407, 251, 400, 263, 401), fill=(132, 135, 125), width=4)
    path = ASSETS / "dial_face.png"
    if not path.exists():
        im.save(path)
    a = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    verts = [[0, 0, 0]] + [[0, 0.08 * np.cos(t), 0.08 * np.sin(t)] for t in a]
    uvs = [[0.5, 0.5]] + [[0.5 - 0.5 * np.cos(t), 0.5 + 0.5 * np.sin(t)] for t in a]
    faces = [[0, 1 + (i + 1) % 96, 1 + i] for i in range(96)]
    return write_mesh("clock", verts, faces, uvs), versioned_path(path)


def make_chair():
    # Rounded rail: a bevelled rectangle swept along y, end caps included.
    x, z = 0.0185, 0.0225
    b = 0.005
    profile = [[-x, -z + b], [-x + b, -z], [x - b, -z], [x, -z + b], [x, z - b], [x - b, z], [-x + b, z], [-x, z - b]]
    vertices = [[a, y, c] for y in [-0.095, 0.095] for a, c in profile]
    faces = []
    for i in range(8):
        j = (i + 1) % 8
        faces.extend([[i, j, j + 8], [i, j + 8, i + 8]])
    for i in range(1, 7):
        faces.extend([[0, i + 1, i], [8, 8 + i, 9 + i]])
    return write_mesh("chair", vertices, faces)


def install_authored_uvs(rows):
    """Keep vein coordinates on their vertices, independent of first render time."""
    from robosimstudio.solvers.render import rtx

    if not hasattr(rtx, "_plant_original_planar_uvs"):
        rtx._plant_original_planar_uvs = rtx.planar_uvs

    def stable_uvs(points):
        candidates = [r for r in rows if len(r["authored_uvs"]) == len(points)]
        if not candidates:
            return rtx._plant_original_planar_uvs(points)
        row = min(candidates, key=lambda r: np.linalg.norm(np.mean(points, axis=0) - np.array(r["at"]) - ORIGIN))
        return np.asarray(row["authored_uvs"], dtype=np.float32)

    rtx.planar_uvs = stable_uvs
