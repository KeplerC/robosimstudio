"""Local procedural mesh constructors; all artifacts stay beside this helper."""

import math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)


def rgb(h):
    return tuple(int(h[i : i + 2], 16) / 255 for i in (1, 3, 5))


def lathe(profile, n=96):
    # profile follows the outside up, then the inside down (radius, z).
    p = np.asarray(profile, float)
    t = np.arange(n) * 2 * np.pi / n
    v = np.stack(
        [
            np.outer(p[:, 0], np.cos(t)),
            np.outer(p[:, 0], np.sin(t)),
            np.broadcast_to(p[:, 1, None], (len(p), n)),
        ],
        -1,
    ).reshape(-1, 3)
    f = []
    for j in range(len(p) - 1):
        for i in range(n):
            a = j * n + i
            b = j * n + (i + 1) % n
            c = b + n
            d = a + n
            f.extend([(a, b, c), (a, c, d)])
    return trimesh.Trimesh(v, f, process=False)


def box(size, at=(0, 0, 0), bevel=0):
    if not bevel:
        m = trimesh.creation.box(extents=size)
    else:
        h = np.array(size) / 2
        b = min(bevel, min(h) * 0.8)
        vv = []
        for axis in range(3):
            for sign in (-1, 1):
                for a in (-1, 1):
                    for c in (-1, 1):
                        p = h.copy() - b
                        p[axis] = h[axis]
                        signs = np.ones(3)
                        signs[axis] = sign
                        signs[(axis + 1) % 3] = a
                        signs[(axis + 2) % 3] = c
                        vv.append(p * signs)
        m = trimesh.convex.convex_hull(np.array(vv))
    m.apply_translation(at)
    return m


def ellipsoid(size, at=(0, 0, 0), subdivisions=2):
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1)
    m.apply_scale(np.array(size) / 2)
    m.apply_translation(at)
    return m


def tube(points, radius, n=12):
    p = np.asarray(points, float)
    r = np.broadcast_to(radius, (len(p),))
    v = []
    for j, point in enumerate(p):
        tangent = p[min(j + 1, len(p) - 1)] - p[max(0, j - 1)]
        tangent /= np.linalg.norm(tangent)
        helper = np.array([0, 0, 1.0]) if abs(tangent[2]) < 0.9 else np.array([1.0, 0, 0])
        u = np.cross(tangent, helper)
        u /= np.linalg.norm(u)
        w = np.cross(tangent, u)
        for a in np.arange(n) * 2 * np.pi / n:
            v.append(point + r[j] * (np.cos(a) * u + np.sin(a) * w))
    f = []
    for j in range(len(p) - 1):
        for i in range(n):
            a = j * n + i
            b = j * n + (i + 1) % n
            f.extend([(a, b, b + n), (a, b + n, a + n)])
    f.extend([(0, i + 1, i) for i in range(1, n - 1)])
    a = (len(p) - 1) * n
    f.extend([(a, a + i, a + i + 1) for i in range(1, n - 1)])
    return trimesh.Trimesh(v, f, process=False)


def ring(radius, z, wire=0.0015, axis="z", center=(0, 0, 0)):
    a = np.linspace(0, 2 * np.pi, 97)
    p = np.stack([radius * np.cos(a), radius * np.sin(a), np.full_like(a, z)], 1)
    if axis == "x":
        p = p[:, [2, 0, 1]]
    if axis == "y":
        p = p[:, [0, 2, 1]]
    return tube(p + np.asarray(center), wire, 8)


def rotate(mesh, degrees, axis=(0, 0, 1)):
    m = mesh.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(degrees), axis))
    return m


def write_obj(name, parts):
    # Explicit material sections retain all colours on one rigid body.
    path = ASSETS / f"{name}.obj"
    mtl = ASSETS / f"{name}.mtl"
    lines = [f"mtllib {mtl.name}"]
    materials = []
    offset = 1
    for k, (mesh, color) in enumerate(parts):
        lines.extend([f"o part_{k}", f"usemtl mat_{k}"])
        lines.extend("v {:.7f} {:.7f} {:.7f}".format(*tuple(v)) for v in mesh.vertices)
        lines.extend("f {} {} {}".format(*(np.array(f) + offset)) for f in mesh.faces)
        offset += len(mesh.vertices)
        materials.extend(
            [
                f"newmtl mat_{k}",
                "Kd {:.6f} {:.6f} {:.6f}".format(*rgb(color)),
                "Ka 0 0 0",
                "d 1",
            ]
        )
    path.write_text("\n".join(lines) + "\n")
    mtl.write_text("\n".join(materials) + "\n")
    return str(path)


def asset(name, parts, colliders):
    from robosimstudio import AssetSpec

    path = write_obj(name, parts)
    proxies = tuple(write_obj(f"{name}_contact_{i}", [(m, "#888888")]) for i, m in enumerate(colliders))
    return AssetSpec(path=path, material_colors=True, collision_paths=proxies)


def concrete():
    path = ASSETS / "concrete_v8.png"
    if path.exists():
        return str(path)
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(82)
    a = np.zeros((1024, 1024))
    for sigma, amp in [(1, 2), (4, 4), (18, 7), (65, 10)]:
        n = gaussian_filter(rng.normal(size=a.shape), sigma, mode="wrap")
        a += amp * n / max(n.std(), 1e-8)
    a += rng.normal(0, 1.3, a.shape)
    col = np.array([158, 163, 160]) + a[:, :, None] * 0.47
    Image.fromarray(np.clip(col, 0, 255).astype("uint8")).save(path)
    return str(path)


def plate(diameter=0.26, height=0.018):
    r = diameter / 2
    h = height
    return lathe(
        [
            (0, 0),
            (0.65 * r, 0),
            (0.72 * r, 0.002),
            (0.90 * r, 0.009),
            (0.985 * r, h - 0.003),
            (r, h - 0.001),
            (0.998 * r, h),
            (0.98 * r, h + 0.0005),
            (0.87 * r, 0.011),
            (0.69 * r, 0.006),
            (0, 0.006),
        ]
    )


def jar(radius, height, style, fill, color, seed=1):
    rng = np.random.default_rng(seed)
    r = radius
    h = height
    glass = "#edf3eb"
    metal = "#b4b9af"
    wood = "#98643f"
    parts = []
    profile = [
        (0, 0),
        (r * 0.90, 0),
        (r, 0.004),
        (r, h * 0.81),
        (r * 0.92, h * 0.94),
        (r * 0.92, h),
        (r * 0.87, h),
        (r * 0.87, h * 0.94),
        (r - 0.001, h * 0.80),
        (r - 0.001, 0.003),
        (0, 0.003),
    ]
    parts.append((lathe(profile), glass))
    if style == "wood":
        parts.extend(
            [
                (
                    lathe(
                        [
                            (0, h - 0.002),
                            (r * 0.94, h - 0.002),
                            (r * 1.075, h + 0.003),
                            (r * 1.09, h + 0.008),
                            (r * 1.05, h + 0.014),
                            (0, h + 0.014),
                        ]
                    ),
                    wood,
                ),
                (ring(r * 0.95, h - 0.004, 0.0015), "#e0dacc"),
            ]
        )
        # Arch handle with an actual opening, rounded rectangular loop.
        p = [
            (0, -0.028, h + 0.015),
            (0, -0.028, h + 0.036),
            (0, -0.022, h + 0.041),
            (0, 0.022, h + 0.041),
            (0, 0.028, h + 0.036),
            (0, 0.028, h + 0.015),
        ]
        parts.append((tube(p, 0.006, 8), wood))
    elif style == "clip":
        parts.extend(
            [
                (
                    lathe(
                        [
                            (0, h),
                            (r * 0.94, h),
                            (r, h + 0.003),
                            (r, h + 0.007),
                            (r * 0.9, h + 0.010),
                            (0, h + 0.010),
                        ]
                    ),
                    glass,
                ),
                (ring(r * 0.96, h - 0.003), metal),
                (ring(r * 1.02, h + 0.003), metal),
                (ring(r * 0.96, h - 0.013), metal),
                (ring(r * 0.96, h - 0.023), metal),
            ]
        )
        for s in (-1, 1):
            parts.append(
                (
                    tube(
                        [
                            (0, s * r * 0.98, h + 0.004),
                            (-0.009, s * r * 1.12, h),
                            (-0.011, s * r * 1.14, h - 0.051),
                            (-0.006, s * r * 1.06, h - 0.057),
                            (0, s * r, h - 0.043),
                        ],
                        0.0017,
                    ),
                    metal,
                )
            )
    else:
        for z in [h - 0.003, h + 0.001, h + 0.005]:
            parts.append((ring(r * 0.93, z, 0.002), glass))
    if fill == "oats":
        parts.append(
            (
                lathe(
                    [
                        (0, 0.004),
                        (r - 0.004, 0.004),
                        (r - 0.004, h * 0.83),
                        (0, h * 0.83),
                    ]
                ),
                "#d4cbb6",
            )
        )
        flakes = [[], [], []]
        colors = ["#d8d2bd", "#c4b99e", "#e5ddc8"]
        for _iz, z in enumerate(np.arange(0.006, h * 0.83, 0.003)):
            for angle in np.arange(0, 2 * np.pi, 0.085):
                a = angle + rng.uniform(-0.04, 0.04)
                rr = r - 0.004 + rng.uniform(-0.0005, 0.0005)
                m = ellipsoid((0.0012, 0.0035, 0.0024), subdivisions=1)
                m = rotate(m, a * 180 / np.pi)
                m.apply_translation((rr * np.cos(a), rr * np.sin(a), z + rng.uniform(-0.001, 0.001)))
                flakes[int(rng.integers(3))].append(m)
        for meshes, col in zip(flakes, colors, strict=False):
            parts.append((trimesh.util.concatenate(meshes), col))
    elif fill == "spaghetti":
        strands = []
        for _ in range(190):
            a = rng.uniform(0, 2 * np.pi)
            rho = np.sqrt(rng.random()) * (r - 0.009)
            x, y = rho * np.cos(a), rho * np.sin(a)
            top = h * 0.82 + rng.uniform(-0.006, 0.006)
            strands.append(
                tube(
                    [(x * 0.72, y * 0.72, 0.004), (x, y, top)],
                    rng.uniform(0.0008, 0.0012),
                    6,
                )
            )
        parts.append((trimesh.util.concatenate(strands), "#ead8b0"))
    elif fill in ("coffee", "nuts", "biscuits"):
        groups = [[], [], []]
        if fill == "biscuits":
            for level in range(3):
                m = ellipsoid((0.040, 0.040, 0.014), (0, 0, 0.011 + level * 0.013), 2)
                groups[level].append(m)
            colors = ["#956038", "#b17b49", "#8b572e"]
        else:
            coffee = fill == "coffee"
            step = 0.0065 if coffee else 0.014
            height = 0.030 if coffee else h * 0.73
            dz = 0.0038 if coffee else 0.012
            colors = ["#3c2115", "#58301d", "#70442a"] if coffee else ["#ab9063", "#c6ac7c", "#90724c"]
            for _layer, z in enumerate(np.arange(0.006, height, dz)):
                for x in np.arange(-r + step, r - step, step):
                    for y in np.arange(-r + step, r - step, step):
                        xx = x + rng.uniform(-0.27, 0.27) * step
                        yy = y + rng.uniform(-0.27, 0.27) * step
                        if xx * xx + yy * yy > (r - 0.005) ** 2:
                            continue
                        sz = np.array((0.008, 0.005, 0.004)) if coffee else np.array((0.019, 0.012, 0.011))
                        m = ellipsoid(sz * rng.uniform(0.8, 1.1), subdivisions=1 if coffee else 2)
                        if not coffee:
                            vv = m.vertices.copy()
                            vv *= rng.uniform(0.79, 1.19, (len(vv), 1))
                            m.vertices = vv
                        m = rotate(m, rng.uniform(0, 360))
                        m = rotate(m, rng.uniform(-45, 45), axis=(0, 1, 0))
                        m.apply_translation((xx, yy, z + rng.uniform(-0.3, 0.3) * dz))
                        groups[int(rng.integers(3))].append(m)
        for meshes, col in zip(groups, colors, strict=False):
            if meshes:
                parts.append((trimesh.util.concatenate(meshes), col))
    return parts


def teapot():
    p = []
    black = "#15190f"
    p.append(
        (
            lathe(
                [
                    (0, 0),
                    (0.047, 0),
                    (0.071, 0.008),
                    (0.086, 0.028),
                    (0.092, 0.062),
                    (0.091, 0.084),
                    (0.080, 0.105),
                    (0.065, 0.115),
                    (0.049, 0.117),
                    (0.045, 0.109),
                    (0, 0.109),
                ]
            ),
            black,
        )
    )
    p.append(
        (
            lathe(
                [
                    (0, 0.113),
                    (0.041, 0.113),
                    (0.051, 0.117),
                    (0.053, 0.121),
                    (0.045, 0.127),
                    (0, 0.130),
                ]
            ),
            black,
        )
    )
    p.append((tube([(0, 0, 0.129), (0, 0, 0.146)], 0.006), black))
    p.append((ellipsoid((0.018, 0.018, 0.019), (0, 0, 0.151)), black))
    a = np.linspace(0, np.pi, 80)
    p.append(
        (
            tube(
                np.stack([np.zeros_like(a), 0.075 * np.cos(a), 0.103 + 0.129 * np.sin(a)], 1),
                0.0028,
            ),
            black,
        )
    )
    p.append(
        (
            tube(
                [
                    (0, 0.073, 0.072),
                    (0, 0.093, 0.078),
                    (0, 0.118, 0.101),
                    (0, 0.143, 0.115),
                ],
                np.array([0.021, 0.020, 0.015, 0.009]),
                24,
            ),
            black,
        )
    )
    p.append((lathe([(0, 0), (0.008, 0), (0.009, 0.002), (0, 0.002)]), "#111510"))
    return p


def fork(length=0.16):
    black = "#303022"
    meshes = [
        tube([(0, 0, 0.002), (length * 0.76, 0, 0.002)], 0.0027, 8),
        box((0.013, 0.024, 0.003), (length * 0.79, 0, 0.002), 0.001),
    ]
    for y in [-0.010, -0.0033, 0.0033, 0.010]:
        meshes.append(tube([(length * 0.80, y, 0.002), (length, y, 0.005)], 0.0014, 6))
    return [(trimesh.util.concatenate(meshes), black)]


# Core extension seam: detailed imported visuals with primitive compound contacts.
# This is scene-local; no toolkit code or global solver behaviour is replaced.
from dataclasses import dataclass

from robosimstudio import AssetSpec, ObjectSpec, register_entity
from robosimstudio.domains.rigid.entity import EntityRigid


@dataclass(frozen=True, kw_only=True)
class KitchenRigidSpec(ObjectSpec):
    entity: str = "kitchen_rigid_v2"
    contact_boxes: tuple = ()
    visual_size: tuple = ()
    full_contact: bool = False
    contact_meshes: tuple = ()


@register_entity("kitchen_rigid_v2")
class KitchenRigid(EntityRigid):
    def build(self, builder, episode):
        import newton
        import warp as wp

        from robosimstudio.scene.build.bodies import preserve_body_inertia

        start = builder.shape_count
        super().build(builder, episode)
        if not self.spec.full_contact:
            for i in range(start, builder.shape_count):
                builder.shape_flags[i] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES)
        cfg = builder.default_shape_cfg.copy()
        cfg.density = 0.0
        cfg.mu = float(self.spec.friction or 0.7)
        cfg.ke = 6e4
        cfg.kd = 500.0
        cfg.gap = 0.0002
        for i, (half, pos, q) in enumerate(self.spec.contact_boxes):
            with preserve_body_inertia(builder, self.body):
                s = builder.add_shape_box(
                    self.body,
                    hx=half[0],
                    hy=half[1],
                    hz=half[2],
                    xform=wp.transform(pos, wp.quat(*q)),
                    cfg=cfg,
                    label=f"{self.spec.name}_support_{i}",
                )
            builder.shape_flags[s] &= ~int(newton.ShapeFlags.VISIBLE)

        for i, path in enumerate(self.spec.contact_meshes):
            mesh = trimesh.load(path, force="mesh", process=False)
            with preserve_body_inertia(builder, self.body):
                shape = builder.add_shape_mesh(
                    body=self.body,
                    xform=wp.transform_identity(),
                    mesh=newton.Mesh(
                        np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32).reshape(-1)
                    ),
                    cfg=cfg,
                    label=f"{self.spec.name}_rim_{i}",
                )
            builder.shape_flags[shape] &= ~int(newton.ShapeFlags.VISIBLE)


def rigid(name, parts, colliders, size, extra_meshes=(), **kwargs):
    from scipy.spatial.transform import Rotation

    verts = np.vstack([m.vertices for m, c in parts])
    factor = np.asarray(size) / (verts.max(0) - verts.min(0))
    for m, _c in parts:
        m.apply_scale(factor)
    for m in colliders:
        m.apply_scale(factor)
    mesh_paths = []
    for i, mesh in enumerate(extra_meshes):
        mesh.apply_scale(factor)
        mesh_paths.append(write_obj(f"{name}_rim_{i}", [(mesh, "#888888")]))
    specs = []
    for m in colliders:
        # Input support meshes are boxes. OBB recovers their exact placement.
        b = m.bounding_box_oriented
        xf = b.primitive.transform
        specs.append(
            (
                tuple(b.primitive.extents / 2),
                tuple(xf[:3, 3]),
                tuple(Rotation.from_matrix(xf[:3, :3]).as_quat()),
            )
        )
    return KitchenRigidSpec(
        name=name,
        asset=AssetSpec(path=write_obj(name, parts), material_colors=True),
        contact_boxes=tuple(specs),
        visual_size=tuple(size),
        full_contact=name not in ("mortar", "mortar_saucer", "utensil_jar", "whisk"),
        contact_meshes=tuple(mesh_paths),
        **kwargs,
    )


from robosimstudio.core.render.scenery import register_scenery


@register_scenery("kitchen_daylight")
def daylight(kit):
    if not hasattr(kit, "stage"):
        return
    from pxr import Gf, UsdGeom, UsdLux
    from scipy.spatial.transform import Rotation

    from robosimstudio.solvers.render.rtx import LAB_PANEL_PER_KEY

    xf = UsdGeom.Xform.Define(kit.stage, "/root/KitchenWindow")
    xf.AddTranslateOp().Set(Gf.Vec3d(-0.05, 0.85, 1.65))
    direction = np.array([0.65, -0.85, -0.7])
    direction /= np.linalg.norm(direction)
    rotation = trimesh.geometry.align_vectors([0, 0, -1], direction)[:3, :3]
    q = Rotation.from_matrix(rotation).as_quat()
    xf.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
    light = UsdLux.RectLight.Define(kit.stage, "/root/KitchenWindow/light")
    light.GetWidthAttr().Set(0.9)
    light.GetHeightAttr().Set(1.2)
    light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.99, 0.98))
    light.GetIntensityAttr().Set(LAB_PANEL_PER_KEY * 0.9)


def bevel_tile(width, height, depth=0.004, inset=0.0035):
    """A broad ceramic bevel independent of its much smaller thickness."""
    verts = []
    for x, shrink in [(-depth / 2, inset), (depth / 2, 0)]:
        for y, z in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
            verts.append((x, y * (width / 2 - shrink), z * (height / 2 - shrink)))
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7)]
    for i in range(4):
        j = (i + 1) % 4
        faces.extend([(i, j, j + 4), (i, j + 4, i + 4)])
    return trimesh.Trimesh(verts, np.asarray(faces)[:, ::-1], process=False)


def clip_mesh(mesh, axis, bound, keep_less=True):
    """Clip visual tiles to a wall extent; no physics or image manipulation."""
    verts = []
    faces = []
    for tri in mesh.triangles:
        poly = list(tri)
        new = []
        for a, b in zip(poly, poly[1:] + poly[:1], strict=False):
            ia = (a[axis] <= bound) if keep_less else (a[axis] >= bound)
            ib = (b[axis] <= bound) if keep_less else (b[axis] >= bound)
            if ia:
                new.append(a)
            if ia != ib:
                t = (bound - a[axis]) / (b[axis] - a[axis])
                new.append(a + t * (b - a))
        if len(new) > 2:
            base = len(verts)
            verts.extend(new)
            faces.extend([(base, base + i, base + i + 1) for i in range(1, len(new) - 1)])
    return trimesh.Trimesh(verts, faces, process=False)


def kitchen_look(kit):
    """Scene-local native glass via the core scenery hook. No renderer patches.

    OmniGlass and its parameter names are from the installed OVRTX examples and
    Base/OmniGlass.mdl. A hollow jar needs more than the default three refractions.
    """
    daylight(kit)
    if not hasattr(kit, "stage"):
        return
    from pxr import Gf, Sdf, UsdLux, UsdShade
    from scipy.spatial.transform import Rotation

    from robosimstudio.solvers.render.rtx import LAB_PANEL_PER_KEY

    window = kit.stage.GetPrimAtPath("/root/KitchenWindow")
    window.GetAttribute("xformOp:translate").Set(Gf.Vec3d(-0.60, 0.40, 1.65))
    direction = np.array([1.5, -0.4, -0.6])
    direction /= np.linalg.norm(direction)
    q = Rotation.from_matrix(trimesh.geometry.align_vectors([0, 0, -1], direction)[:3, :3]).as_quat()
    window.GetAttribute("xformOp:orient").Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
    light = UsdLux.RectLight(kit.stage.GetPrimAtPath("/root/KitchenWindow/light"))
    light.GetWidthAttr().Set(1.6)
    light.GetHeightAttr().Set(1.2)
    light.GetIntensityAttr().Set(LAB_PANEL_PER_KEY * 0.9)
    # Kitchen fill and ceiling light replace the lab lighting preset.
    from pxr import UsdGeom

    from robosimstudio.solvers.render.rtx import LAB_DOME_PER_AMBIENT

    dome = UsdLux.DomeLight.Define(kit.stage, "/root/KitchenFill/light")
    dome.GetColorAttr().Set(Gf.Vec3f(0.98, 0.99, 1.0))
    dome.GetIntensityAttr().Set(LAB_DOME_PER_AMBIENT * 1.6)
    xf = UsdGeom.Xform.Define(kit.stage, "/root/KitchenCeiling")
    xf.AddTranslateOp().Set(Gf.Vec3d(0.6, 0, 2.45))
    ceiling = UsdLux.RectLight.Define(kit.stage, "/root/KitchenCeiling/light")
    ceiling.GetWidthAttr().Set(1.4)
    ceiling.GetHeightAttr().Set(0.9)
    ceiling.GetColorAttr().Set(Gf.Vec3f(0.97, 0.99, 1.0))
    ceiling.GetIntensityAttr().Set(LAB_PANEL_PER_KEY * 0.65)
    # Bake uniform default-material colours. With mixed MDL / PreviewSurface
    # materials this renderer can lose the shared displayColor reader on reload.
    # The values come from this LIVE USD stage, not a separate layout copy.
    from robosimstudio.core.render.finish import Finish

    for prim in list(kit.stage.Traverse()):
        attr = prim.GetAttribute("primvars:displayColor")
        if not attr:
            continue
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            continue
        surf = UsdShade.Shader(kit.stage.GetPrimAtPath(str(material.GetPath()) + "/PreviewSurface"))
        if not surf or not surf.GetInput("roughness"):
            continue
        rough = surf.GetInput("roughness").Get()
        if rough is None or abs(float(rough) - 0.6) > 0.00001:
            continue
        times = attr.GetTimeSamples()
        value = attr.Get(times[0]) if times else attr.Get()
        if value is None or not len(value):
            continue
        linear = np.asarray(value[0], float)
        display = np.where(
            linear <= 0.0031308,
            linear * 12.92,
            1.055 * np.power(linear, 1 / 2.4) - 0.055,
        )
        kit.library.bind(prim, Finish(roughness=0.6), tuple(display))
    # The core creates the viewport product AFTER this hook, then copies it for
    # the named camera. Pre-authoring its settings survives that DefinePrim.
    product = kit.stage.DefinePrim(
        "/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0",
        "RenderProduct",
    )
    product.CreateAttribute("omni:rtx:rtpt:maxSpecularAndTransmissionBounces", Sdf.ValueTypeNames.UInt).Set(16)
    product.CreateAttribute(
        "omni:rtx:indirectDiffuse:maxSpecularAndTransmissionBounces",
        Sdf.ValueTypeNames.Int,
    ).Set(8)
    for prim in list(kit.stage.Traverse()):
        if prim.GetTypeName() == "RenderProduct":
            prim.CreateAttribute(
                "omni:rtx:rtpt:maxSpecularAndTransmissionBounces",
                Sdf.ValueTypeNames.UInt,
            ).Set(16)
            prim.CreateAttribute(
                "omni:rtx:indirectDiffuse:maxSpecularAndTransmissionBounces",
                Sdf.ValueTypeNames.Int,
            ).Set(8)
        if prim.GetName() != "PreviewSurface":
            continue
        preview = UsdShade.Shader(prim)
        opacity = preview.GetInput("opacity")
        if not opacity or float(opacity.Get() or 1) > 0.01:
            continue
        material = UsdShade.Material(prim.GetParent())
        shader = UsdShade.Shader.Define(kit.stage, str(material.GetPath()) + "/NativeGlass")
        shader.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
        shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
        shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
        shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.47)
        shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.015)
        shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.01)
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
        for output in ["surface", "volume", "displacement"]:
            material.CreateOutput("mdl:" + output, Sdf.ValueTypeNames.Token).ConnectToSource(
                shader.ConnectableAPI(), "out"
            )


def clip_plane(mesh, normal, bound):
    normal = np.asarray(normal, float)
    vertices = []
    faces = []
    for tri in mesh.triangles:
        poly = list(tri)
        new = []
        for a, b in zip(poly, poly[1:] + poly[:1], strict=False):
            da = a @ normal - bound
            db = b @ normal - bound
            if da <= 0:
                new.append(a)
            if (da <= 0) != (db <= 0):
                new.append(a + (b - a) * da / (da - db))
        if len(new) > 2:
            base = len(vertices)
            vertices.extend(new)
            faces.extend([(base, base + i, base + i + 1) for i in range(1, len(new) - 1)])
    return trimesh.Trimesh(vertices, faces, process=False)


def tile_edge(mesh):
    """The cut tile ends leave a 66 mm diagonal zigzag against white plaster."""
    pieces = []
    period = 0.066
    zmin, zmax = mesh.bounds[:, 2]
    for k in range(-2, 22):
        lo = 0.027 + k * period
        hi = lo + period
        if hi < zmin or lo > zmax:
            continue
        m = clip_mesh(clip_mesh(mesh, 2, max(0, lo), False), 2, hi)
        m = clip_plane(m, (0, 1, -1), 0.27 - lo) if k % 2 == 0 else clip_plane(m, (0, 1, 1), 0.336 + lo)
        if len(m.faces):
            pieces.append(m)
    return trimesh.util.concatenate(pieces) if pieces else trimesh.Trimesh()


def blade(length, width, thickness):
    """A flat wooden blade with semicircular ends."""
    r = width / 2
    a = np.linspace(-np.pi / 2, np.pi / 2, 17)
    xy = np.vstack(
        [
            np.c_[(length / 2 - r) + r * np.cos(a), r * np.sin(a)],
            np.c_[-(length / 2 - r) - r * np.cos(a), -r * np.sin(a)],
        ]
    )
    n = len(xy)
    vertices = np.vstack([np.c_[xy, np.zeros(n)], np.c_[xy, np.full(n, thickness)]])
    faces = []
    for i in range(1, n - 1):
        faces.extend([(0, i + 1, i), (n, n + i, n + i + 1)])
    for i in range(n):
        j = (i + 1) % n
        faces.extend([(i, j, n + j), (i, n + j, n + i)])
    return trimesh.Trimesh(vertices, faces, process=False)
