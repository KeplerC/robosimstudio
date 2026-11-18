"""Local procedural geometry and textures; no pixels are sampled from the reference."""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)


def obj(name, vertices, faces, uv=None):
    path = ASSETS / (name + ".obj")
    with path.open("w") as f:
        for p in vertices:
            f.write("v {:.9f} {:.9f} {:.9f}\n".format(*tuple(p)))
        if uv is not None:
            for p in uv:
                f.write("vt {:.9f} {:.9f}\n".format(*tuple(p)))
        for tri in faces:
            if uv is None:
                f.write("f " + " ".join(str(int(i) + 1) for i in tri) + "\n")
            else:
                f.write("f " + " ".join(f"{int(i) + 1}/{int(i) + 1}" for i in tri) + "\n")
    return str(path)


def lathe(profile, sides=96):
    v = []
    f = []
    for r, z in profile:
        for a in np.linspace(0, 2 * np.pi, sides, endpoint=False):
            v.append((r * np.cos(a), r * np.sin(a), z))
    for k in range(len(profile) - 1):
        for j in range(sides):
            a = k * sides + j
            b = k * sides + (j + 1) % sides
            c = b + sides
            d = a + sides
            f.extend(((a, b, c), (a, c, d)))
    return np.array(v), np.array(f)


def combine(meshes):
    vs = []
    fs = []
    off = 0
    for v, f in meshes:
        vs.extend(v)
        fs.extend(np.asarray(f) + off)
        off += len(v)
    return np.array(vs), np.array(fs)


def tube(points, radius, sides=6):
    pts = np.asarray(points)
    vs = []
    fs = []
    for i, p in enumerate(pts):
        tangent = pts[min(i + 1, len(pts) - 1)] - pts[max(0, i - 1)]
        tangent = tangent / np.linalg.norm(tangent)
        a = np.cross(tangent, (0, 0, 1))
        if np.linalg.norm(a) < 0.01:
            a = np.cross(tangent, (0, 1, 0))
        a /= np.linalg.norm(a)
        b = np.cross(tangent, a)
        for t in np.linspace(0, 2 * np.pi, sides, endpoint=False):
            vs.append(p + radius * (np.cos(t) * a + np.sin(t) * b))
    for i in range(len(pts) - 1):
        for j in range(sides):
            a = i * sides + j
            b = i * sides + (j + 1) % sides
            fs.extend(((a, b, b + sides), (a, b + sides, a + sides)))
    return np.array(vs), np.array(fs)


def cachepot():
    profile = [
        (0, 0.0),
        (0.058, 0.0),
        (0.064, 0.004),
        (0.067, 0.018),
        (0.068, 0.122),
        (0.067, 0.130),
        (0.062, 0.132),
        (0.060, 0.126),
        (0.060, 0.014),
        (0, 0.014),
    ]
    parts = [lathe(profile)]
    # Raised hexagonal ceramic ribs, individual rings on the cylinder.
    cols = 28
    2 * np.pi * 0.068 / cols
    for row in range(9):
        z = 0.015 + row * 0.01275
        for col in range(cols):
            centre = (col + 0.5 * (row % 2)) * 2 * np.pi / cols
            pts = []
            for a in np.linspace(np.pi / 6, 2 * np.pi + np.pi / 6, 7):
                theta = centre + 0.0073 * np.cos(a) / 0.068
                pts.append((0.0685 * np.cos(theta), 0.0685 * np.sin(theta), z + 0.0085 * np.sin(a)))
            parts.append(tube(pts, 0.00115, 6))
    v, f = combine(parts)
    v[:, 2] *= 0.124 / 0.132
    return obj("cachepot", v, f)


def leaf_mesh(length, width, spathe=False):
    verts = []
    faces = []
    held = []
    for i, t in enumerate(np.linspace(0, 1, 11)):
        # Broad basal lobes, a real notch, and a tapered tip. Non-degenerate 10x10 quad topology.
        hw = 0.5 * width * (0.62 + 0.76 * np.sin(np.pi * t) ** 0.68) * (1 - 0.985 * t) ** 0.65
        for j, u in enumerate(np.linspace(-1, 1, 11)):
            x = length * (t + 0.13 * (1 - t) ** 5 * (1 - abs(u)) - 0.13)
            y = hw * u
            z = length * (
                0.065 * u * u * np.sin(np.pi * t) ** 0.5 - 0.075 * t * t + 0.012 * np.sin(3 * np.pi * t + u) * abs(u)
            )
            if spathe:
                z = length * (
                    0.20 * u * u * np.sin(np.pi * t) ** 0.5 - 0.025 * t * t + 0.025 * np.sin(4 * np.pi * t + u) * abs(u)
                )
            else:
                z += length * 0.035 * np.sin(4 * np.pi * t + u * 0.7) * abs(u) ** 1.5
            verts.append((x, y, z))
            if i <= 2 and abs(u) <= 0.41:
                held.append(i * 11 + j)
    for i in range(10):
        for j in range(10):
            a = i * 11 + j
            faces.extend(((a, a + 11, a + 12), (a, a + 12, a + 1)))
    verts = [*verts[:110], verts[115]]
    faces = [tri for tri in faces if max(tri) < 110]
    faces += [(99 + j, 110, 100 + j) for j in range(10)]
    return np.array(verts), np.array(faces), held


def blade_texture(name, vertices, color, spathe=False):
    # Renderer gives cloth a PCA planar projection. Paint veins in exactly that projection.
    p = np.asarray(vertices)
    _, _, vt = np.linalg.svd(p - p.mean(0), full_matrices=False)
    uv = (p - p.mean(0)) @ vt[:2].T
    uv -= uv.min(0)
    span = np.maximum(uv.max(0), 1e-6)
    tex = np.zeros((384, 384, 3), dtype=float)
    rgb = np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)])
    rng = np.random.default_rng(23)
    _yy, _xx = np.mgrid[:384, :384]
    from scipy.interpolate import griddata

    ts = np.r_[np.repeat(np.linspace(0, 1, 11)[:10], 11), 1.0]
    us = np.r_[np.tile(np.linspace(-1, 1, 11), 10), 0.0]
    qx, qy = np.meshgrid(np.linspace(0, span[0], 384), np.linspace(span[1], 0, 384))
    t = griddata(uv, ts, (qx, qy), method="linear", fill_value=0.5)
    u = griddata(uv, us, (qx, qy), method="linear", fill_value=0.0)
    mott = 0.74 + 0.20 * np.sin(np.pi * t) + 0.13 * np.abs(u) + 0.045 * np.sin(32 * t + u * 3)
    if spathe:
        mott = 0.64 + 0.25 * np.abs(u) + 0.18 * t + 0.10 * np.sin(20 * t + u * 5)
        if name == "spathe_04":
            mott -= 0.38 * np.exp(-(((t - 0.28) / 0.24) ** 2) - ((u + 0.38) / 0.28) ** 2)
            mott -= 0.24 * np.exp(-(((t - 0.22) / 0.18) ** 2) - ((u - 0.40) / 0.3) ** 2)
    mott += 0.025 * rng.normal(size=(384, 384))
    tex[:] = rgb * mott[:, :, None]
    im = Image.fromarray(np.uint8(np.clip(tex, 0, 255)))
    d = ImageDraw.Draw(im)
    vein = tuple(np.uint8(np.clip(rgb * (0.67 if spathe else 1.25) + ([3, 2, 0] if spathe else [6, 12, 1]), 0, 255)))
    if name == "leaf_01":
        vein = tuple(np.uint8(np.minimum(rgb * np.array([1.65, 1.4, 1.35]) + [9, 14, 3], 255)))

    def pixel(idx):
        u, v = uv[min(idx, len(uv) - 1)] / span
        return (int(u * 383), int((1 - v) * 383))

    d.line([pixel(i * 11 + 5) for i in range(11)], fill=vein, width=5 if not spathe else 4)
    for i in (2, 4, 6, 8):
        for side in (0, 10):
            d.line(
                [
                    pixel(i * 11 + 5),
                    pixel(min(i + 1, 10) * 11 + (2 if side == 0 else 8)),
                    pixel(min(i + 2, 10) * 11 + side),
                ],
                fill=vein,
                width=2,
            )
    im = im.filter(ImageFilter.GaussianBlur(0.5))
    digest = hashlib.sha1(im.tobytes()).hexdigest()[:10]
    path = ASSETS / (name + "_v3_" + digest + ".png")
    if not path.exists():
        tmp = path.with_suffix(".tmp.png")
        im.save(tmp)
        tmp.replace(path)
    return str(path), tuple(1 / span)




def soil():
    path = ASSETS / "soil_dark_v2.png"
    if not path.exists():
        rng = np.random.default_rng(51)
        a = rng.uniform(0.5, 1.4, (512, 512, 1)) * np.array([57, 42, 24])
        im = Image.fromarray(np.uint8(np.clip(a, 0, 255)))
        d = ImageDraw.Draw(im)
        for _ in range(650):
            x, y = rng.integers(0, 512, 2)
            r = int(rng.integers(1, 6))
            c = (79, 63, 32) if rng.random() < 0.85 else (116, 107, 75)
            d.ellipse((x - r, y - r, x + r, y + r), fill=c)
        im.save(path)
    return str(path)




def tinted_plaster(source):
    path = ASSETS / "plaster_blue_v2.png"
    if not path.exists():
        a = np.asarray(Image.open(source).resize((1024, 1024))).astype(float)
        light = a.mean(2) / max(a.mean(), 1)
        rng = np.random.default_rng(17)
        a = light[:, :, None] * np.array([86, 112, 116]) + rng.normal(0, 2, (1024, 1024, 1))
        Image.fromarray(np.uint8(np.clip(a, 0, 255))).save(path)
    return str(path)


def worn_board(source):
    pale = "plywood" in source
    path = ASSETS / ("pale_board_v3.png" if pale else "table_board_v3.png")
    if not path.exists():
        from scipy.ndimage import gaussian_filter

        rng = np.random.default_rng(75)
        a = np.asarray(Image.open(source).resize((1024, 1024))).astype(float)
        a = a * 0.65 + np.array([88, 80, 65] if pale else [40, 31, 19])
        wear = gaussian_filter(rng.random((1024, 1024)), 22)
        wear = (wear - wear.min()) / (wear.max() - wear.min())
        a = a * (0.75 + 0.4 * wear[:, :, None])
        im = Image.fromarray(np.uint8(np.clip(a, 0, 255)))
        d = ImageDraw.Draw(im)
        for _ in range(45):
            x, y = rng.integers(0, 1024, 2)
            length = int(rng.integers(4, 100))
            d.line(
                (x, y, x + length, y + int(rng.integers(-2, 3))),
                fill=(171, 152, 124) if pale else (130, 107, 77),
                width=1,
            )
        im.save(path)
    return str(path)








def plaster_mesh(size):
    from scipy.ndimage import gaussian_filter

    nx, ny = 180, 65
    yy, zz = np.meshgrid(np.linspace(-size[1] / 2, size[1] / 2, nx), np.linspace(-size[2] / 2, size[2] / 2, ny))
    rng = np.random.default_rng(99)
    noise = gaussian_filter(rng.normal(size=(ny, nx)), 0.8)
    noise /= max(noise.std(), 1e-6)
    xx = -size[0] / 2 + 0.0010 * noise
    v = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    f = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            k = i * nx + j
            f.extend(((k, k + nx + 1, k + 1), (k, k + nx, k + nx + 1)))
    return obj("rough_plaster", v, np.asarray(f))


def beveled_board(size, bevel=0.004):
    """Closed convex wooden board with a four-millimetre worn edge."""
    import itertools

    import trimesh

    half = np.asarray(size) / 2
    vertices = []
    for signs in itertools.product((-1, 1), repeat=3):
        for axis in range(3):
            point = half - bevel
            point[axis] = half[axis]
            vertices.append(point * np.asarray(signs))
    mesh = trimesh.convex.convex_hull(np.asarray(vertices))
    return obj("beveled_lid_board", mesh.vertices, mesh.faces)
