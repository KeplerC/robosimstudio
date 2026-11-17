"""Local workshop geometry and procedural maps; no reference pixels are used."""

import contextlib
import math
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"


def tex(name):
    """An immutable content-addressed PNG; live RTX never sees a partial rewrite."""
    import hashlib

    data = (ASSETS / name).read_bytes()
    path = ASSETS / (Path(name).stem + "_" + hashlib.sha256(data).hexdigest()[:12] + Path(name).suffix)
    if not path.exists():
        temporary = path.with_suffix(".part")
        temporary.write_bytes(data)
        temporary.replace(path)
    return str(path)


def solid_fibre_texture(color):
    """Immutable constant albedo; avoids stale USD instance colour on live rebuilds."""
    rgb = tuple(int(round(255 * v)) for v in color)
    name = "fibre_" + "".join(f"{v:02x}" for v in rgb) + ".png"
    path = ASSETS / name
    if not path.exists():
        Image.new("RGB", (4, 4), rgb).save(path)
    return tex(name)


def rounded_box(size, radius=0.002, steps=5):
    """Bevelled cuboid with rounded corners, via a rounded rectangle at four heights."""
    sx, sy, sz = size
    pts = []
    for z, inset in [(0, radius * 0.65), (radius * 0.6, 0), (sz - radius * 0.6, 0), (sz, radius * 0.65)]:
        for cx, cy, start in [
            (sx / 2 - radius, sy / 2 - radius, 0),
            (-sx / 2 + radius, sy / 2 - radius, 90),
            (-sx / 2 + radius, -sy / 2 + radius, 180),
            (sx / 2 - radius, -sy / 2 + radius, 270),
        ]:
            for angle in np.linspace(start, start + 90, steps, endpoint=False):
                t = math.radians(angle)
                rr = radius - inset
                pts.append((cx + rr * math.cos(t), cy + rr * math.sin(t), z))
    ring = 4 * steps
    faces = []
    for layer in range(3):
        for k in range(ring):
            a = layer * ring + k
            b = layer * ring + (k + 1) % ring
            faces.extend([(a, b, b + ring), (a, b + ring, a + ring)])
    for k in range(1, ring - 1):
        faces.extend([(0, k + 1, k), (3 * ring, 3 * ring + k, 3 * ring + k + 1)])
    return np.asarray(pts, np.float32), np.asarray(faces, np.int32)


def grip_mesh(sole_top):
    """A rounded, asymmetric D grip: moulded edges and four finger scallops."""
    from scipy.interpolate import PchipInterpolator

    outer = np.array(
        [
            [0.070, 0],
            [0.064, 0.030],
            [0.051, 0.060],
            [0.035, 0.069],
            [0, 0.071],
            [-0.030, 0.067],
            [-0.057, 0.049],
            [-0.081, 0.016],
            [-0.076, 0],
        ]
    )
    inner = np.array(
        [
            [0.063, 0],
            [0.055, 0.033],
            [0.041, 0.053],
            [0.024, 0.060],
            [0, 0.059],
            [-0.023, 0.058],
            [-0.049, 0.044],
            [-0.047, 0.020],
            [-0.053, 0],
        ]
    )
    t = np.linspace(0, 1, 144)
    knots = np.linspace(0, 1, len(outer))
    out = PchipInterpolator(knots, outer)(t)
    inside = PchipInterpolator(knots, inner)(t)
    inside[:, 1] += (
        0.0015 * np.cos((inside[:, 0] + 0.035) / 0.022 * 2 * np.pi) * np.exp(-((inside[:, 0] / 0.043) ** 10))
    )
    widths = PchipInterpolator(knots, [0.005, 0.006, 0.010, 0.012, 0.012, 0.013, 0.014, 0.020, 0.016])(t)
    v = []
    faces = []
    # Eight vertices around each rounded rectangular cross section.
    for k in range(len(t)):
        delta = out[k] - inside[k]
        depth = np.linalg.norm(delta)
        unit = delta / max(depth, 1e-8)
        bevel = min(0.0027, depth * 0.24, widths[k] * 0.45)
        for x, q in [
            (-widths[k] + bevel, out[k]),
            (widths[k] - bevel, out[k]),
            (widths[k], out[k] - unit * bevel),
            (widths[k], inside[k] + unit * bevel),
            (widths[k] - bevel, inside[k]),
            (-widths[k] + bevel, inside[k]),
            (-widths[k], inside[k] + unit * bevel),
            (-widths[k], out[k] - unit * bevel),
        ]:
            v.append((x, q[0], q[1] + sole_top))
    for k in range(len(t) - 1):
        for j in range(8):
            a = k * 8 + j
            b = k * 8 + (j + 1) % 8
            faces.extend([(a, b, b + 8), (a, b + 8, a + 8)])
    for j in range(1, 7):
        faces.extend([(0, j + 1, j), ((len(t) - 1) * 8, (len(t) - 1) * 8 + j, (len(t) - 1) * 8 + j + 1)])
    return np.asarray(v, np.float32), np.asarray(faces, np.int32)[:, ::-1]


def fibre_mesh(length, radius, offsets, segment, segments):
    """Fine fibre skins attached to one actual simulated capsule."""
    vertices = []
    faces = []
    for k, (ox, oy) in enumerate(offsets):
        a = len(vertices)
        for end in (0, 1):
            t = (segment + end) / segments
            spread = 0.20 + 0.80 * t
            z = (end - 0.5) * length
            for j in range(4):
                phi = j * np.pi / 2
                vertices.append(
                    (
                        ox * spread + radius * np.cos(phi) + 0.0004 * np.sin(k * 2.4 + t * 5) * t,
                        oy * spread + radius * np.sin(phi) + 0.0004 * np.cos(k * 1.7 + t * 4) * t,
                        z,
                    )
                )
        for j in range(4):
            b = a + j
            c = a + (j + 1) % 4
            faces.extend([(b, c, c + 4), (b, c + 4, b + 4)])
    return np.asarray(vertices, np.float32), np.asarray(faces, np.int32)


def workshop_light(kit):
    """Directional warm sunlight and open-sky fill on the path-tracer stage."""
    if not hasattr(kit, "stage"):
        return
    import hashlib

    from pxr import Gf, Usd, UsdGeom, UsdLux

    # OVRTX survives a page rebuild. Distinct paths prevent stale compiled
    # materials/textures being reused for a new revision's USD stage.
    revision = hashlib.sha1(
        Path(__file__).read_bytes() + Path(__file__).with_name("scene.py").read_bytes()
    ).hexdigest()[:12]
    if kit.stage.GetPrimAtPath("/root/Materials"):
        editor = Usd.NamespaceEditor(kit.stage)
        editor.MovePrimAtPath("/root/Materials", "/root/WorkshopMaterials_" + revision)
        if not editor.ApplyEdits():
            raise RuntimeError("Could not namespace workshop materials")
    dome = UsdLux.DomeLight.Define(kit.stage, "/root/workshop_sky")
    dome.GetIntensityAttr().Set(3000)
    dome.GetColorAttr().Set(Gf.Vec3f(0.95, 0.96, 1))
    sun = UsdLux.DistantLight.Define(kit.stage, "/root/workshop_sun")
    sun.GetIntensityAttr().Set(1800)
    sun.GetAngleAttr().Set(0.6)
    sun.GetColorAttr().Set(Gf.Vec3f(1, 0.94, 0.84))
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(40, -40, 5))

    from pxr import Sdf, UsdShade

    for prim in list(kit.stage.Traverse()):
        if prim.GetName() != "DiffuseTexture":
            continue
        file = UsdShade.Shader(prim).GetInput("file").Get()
        filename = Path(file.path).name if file else ""
        shader = UsdShade.Shader(prim)
        if filename.startswith("grime_blob_"):
            shader.CreateOutput("a", Sdf.ValueTypeNames.Float)
            parent = str(prim.GetPath().GetParentPath())
            surface = UsdShade.Shader.Get(kit.stage, parent + "/PreviewSurface")
            surface.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(shader.ConnectableAPI(), "a")
            surface.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0)
        if filename.startswith("wood_straight_"):
            shader.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0.78, 0.86, 0.95, 1))
        if filename.startswith("band_scan_"):
            shader.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0.78, 0.80, 0.84, 1))
        if not (filename.startswith("plaster_scan_") or filename.startswith("band_scan_")):
            continue
        parent = str(prim.GetPath().GetParentPath())
        normal = UsdShade.Shader.Define(kit.stage, parent + "/NormalTexture")
        normal.CreateIdAttr("UsdUVTexture")
        normal.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex("plaster_normal.png"))
        normal.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        normal.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
        normal.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        normal.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        normal.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        st = UsdShade.Shader.Get(kit.stage, parent + "/UvTransform").GetOutput("result")
        normal.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st)
        normal.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        UsdShade.Shader.Get(kit.stage, parent + "/PreviewSurface").CreateInput(
            "normal", Sdf.ValueTypeNames.Normal3f
        ).ConnectToSource(normal.ConnectableAPI(), "rgb")


def paver_mesh(size, seed):
    """Worn, shallow exposed-aggregate relief over an exactly planar contact slab."""
    sx, sy, sz = size
    nx = max(3, int(sx / 0.003))
    ny = max(3, int(sy / 0.003))
    rng = np.random.default_rng(seed)
    x = np.linspace(-sx / 2 + 0.001, sx / 2 - 0.001, nx)
    y = np.linspace(-sy / 2 + 0.001, sy / 2 - 0.001, ny)
    xx, yy = np.meshgrid(x, y)
    noise = rng.normal(0, 0.00020, (ny, nx))
    z = sz / 2 + np.clip(noise, -0.00045, 0.00045)
    z[[0, -1], :] -= 0.0005
    z[:, [0, -1]] -= 0.0005
    xx[:, [0, -1]] += rng.uniform(-0.0005, 0.0005, (ny, 2))
    yy[[0, -1], :] += rng.uniform(-0.0005, 0.0005, (2, nx))
    v = np.stack([xx, yy, z], -1).reshape(-1, 3).tolist()
    f = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            f.extend([(a, a + 1, a + nx + 1), (a, a + nx + 1, a + nx)])
    border = (
        list(range(nx))
        + [j * nx + nx - 1 for j in range(1, ny)]
        + list(range(ny * nx - 2, (ny - 1) * nx - 1, -1))
        + [j * nx for j in range(ny - 2, 0, -1)]
    )
    start = len(v)
    for i in border:
        p = v[i].copy()
        p[0] = np.clip(p[0] * 1.012, -sx / 2, sx / 2)
        p[1] = np.clip(p[1] * 1.012, -sy / 2, sy / 2)
        p[2] = -sz / 2
        v.append(p)
    for k, a in enumerate(border):
        b = border[(k + 1) % len(border)]
        c = start + k
        d = start + (k + 1) % len(border)
        f.extend([(a, c, d), (a, d, b)])
    return np.asarray(v, np.float32), np.asarray(f, np.int32)


@contextlib.contextmanager
def paver_edges():
    import newton

    from robosimstudio.domains.static.entity import EntityStatic

    old = EntityStatic._add_box

    def add(self, builder):
        old(self, builder)
        if not self.spec.name.startswith("paver_"):
            return
        builder.shape_flags[-1] &= ~int(newton.ShapeFlags.VISIBLE)
        body, xform = self._shape_frame()
        size = np.asarray(self.spec.shape.half_extents) * 2
        seed = sum(ord(c) for c in self.spec.name)
        v, f = paver_mesh(size, seed)
        cfg = builder.default_shape_cfg.copy()
        cfg.density = 0
        cfg.has_shape_collision = False
        cfg.has_particle_collision = False
        builder.add_shape_mesh(
            body, xform=xform, mesh=newton.Mesh(v, f.flatten()), cfg=cfg, label=self.spec.name + "/worn_surface"
        )

    EntityStatic._add_box = add
    try:
        yield
    finally:
        EntityStatic._add_box = old
