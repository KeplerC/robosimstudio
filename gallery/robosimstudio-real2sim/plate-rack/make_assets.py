"""Author the plate: a ceramic side plate as a surface of revolution, and its colliders.

    python make_assets.py

Everything under ``assets/plate/`` is written here, from the numbers below;
nothing is downloaded. The shelf and the drying racks are primitives and are
built in ``scene.py``.

* ``plate.obj`` + ``texture.png`` -- the drawn plate: a closed profile (foot
  ring, flat well, a 30 degree flange, a round lip) revolved about its axis in
  96 meridians. Watertight, so its mass properties are the solid's own.
* ``hull_NNN.obj`` -- the collider: one convex piece for the well and foot, and
  one per (32 sectors x 3 flange bands). Each is the hull of the *drawn mesh's
  own vertices* in its cell, and the cells' edges fall on meridians and profile
  vertices, so every drawn facet lies inside one piece -- exactly, by
  construction, not by sampling. Each piece is then pushed out by ``SKIN``.
* ``plate.json`` -- the design numbers and the mass properties, which
  ``scene.py`` and ``demo.py`` read instead of measuring the mesh.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets" / "plate"

# --------------------------------------------------------------- the design
R = 0.090
"""Outer radius [m]: a 180 mm side plate."""
FOOT = (0.046, 0.054, 0.004)
"""Foot ring: inner and outer radius, and how far it stands below the well's underside [m]."""
WELL_TOP = 0.0085
"""Height of the well's top (eating) surface over the foot [m]: a 4.5 mm well."""
FLANGE_START = 0.058
"""Radius where the flange leaves the well [m]."""
FLANGE_ANGLE = math.radians(30.0)
"""The flange's rise [rad]. Chosen, not found: a pinch follows the flange, so a
plate held upright tilts the wrist by exactly this, and 30-35 degrees is where
this arm reaches highest."""
THICKNESS = 0.0038
"""Wall thickness across the flange [m]."""
LIP_END = 0.0875
"""Radius where the flange's straight run ends and the lip rounds over [m]."""
MERIDIANS = 96
SECTORS = 32
BANDS = (0.068, 0.078)
"""Collider cuts: radii between the flange's start and the rim [m]."""
SKIN = 0.0003
"""How far each collider piece is pushed out past the drawn plate [m]: past what
any contact presses into it -- 0.1-0.15 mm under the fingertips' flat pads,
under 0.1 on the shelf -- so drawn surfaces never meet, and no further, since it
is also how far a drawn plate stands off everything it touches. It was 1 mm
while the fingertips collided as their own domed hulls, which pressed in
0.4-0.9 mm, and the plates visibly hovered in the jaws and over the rails."""
DENSITY = 2400.0
"""Glazed stoneware [kg/m^3]."""
RIM_COLOR = (38, 64, 140)
"""The glaze band near the rim, sRGB."""


def profile() -> tuple[list, list]:
    """The top (eating side) and bottom curves in (r, z), centre to rim, and the lip arc joining them."""
    t_v = THICKNESS / math.cos(FLANGE_ANGLE)  # the wall's vertical thickness on the flange
    tan = math.tan(FLANGE_ANGLE)
    well_under = WELL_TOP - 0.0045

    def top_z(r):
        return WELL_TOP + max(0.0, r - FLANGE_START) * tan

    def bot_z(r):
        return max(well_under, top_z(r) - t_v)

    # top: flat well, a short blend into the flange, the flange to the lip
    top = [(0.0, WELL_TOP)]
    for r in np.linspace(0.010, FLANGE_START - 0.004, 5):
        top.append((float(r), WELL_TOP))
    for s in np.linspace(0.0, 1.0, 6)[1:]:  # quadratic blend over 8 mm
        r0, r1 = FLANGE_START - 0.004, FLANGE_START + 0.004
        r = r0 + s * (r1 - r0)
        z = WELL_TOP + (s * s) * (r1 - FLANGE_START) * tan
        top.append((float(r), float(z)))
    for r in np.linspace(FLANGE_START + 0.004, LIP_END, 10)[1:]:
        top.append((float(r), float(top_z(r))))
    # bottom: under the well, the foot ring, under the flange
    fi, fo, fh = FOOT
    bot = [(0.0, well_under)]
    for r in np.linspace(0.010, fi - 0.002, 4):
        bot.append((float(r), well_under))
    bot += [(fi - 0.001, well_under - 0.0015), (fi, 0.0015 * 0.0 + 0.0), (fo, 0.0), (fo + 0.001, well_under - 0.0015),
            (fo + 0.002, well_under)]
    for r in np.linspace(fo + 0.004, LIP_END, 12):
        bot.append((float(r), float(bot_z(r))))
    # a vertex exactly at every collider cut on both curves, so no facet straddles two cells
    top, bot = _with_cuts(top), _with_cuts(bot)
    # the lip: a half round from the bottom curve's end to the top curve's end
    (rb, zb), (rt, zt) = bot[-1], top[-1]
    cr, cz = 0.5 * (rb + rt), 0.5 * (zb + zt)
    rad = 0.5 * math.hypot(rt - rb, zt - zb)
    start = math.atan2(zb - cz, rb - cr)
    lip = []
    for s in np.linspace(0.0, 1.0, 9)[1:-1]:
        a = start + s * math.pi  # counter-clockwise from the bottom point, over the outside, to the top point
        lip.append((cr + rad * math.cos(a), cz + rad * math.sin(a)))
    # the lip's outermost point must be the plate's radius: scale the arc's reach
    reach = max(p[0] for p in lip) - cr
    if reach > 0:
        lip = [(cr + (p[0] - cr) * ((R - cr) / reach), p[1]) for p in lip]
    # the profile loop runs top curve outward, then the lip from its top end
    # over the outside to its bottom end, then the bottom curve inward
    return top, bot, lip[::-1]


def _with_cuts(curve: list) -> list:
    """``curve`` (increasing r) with a vertex inserted at each collider cut radius it spans."""
    out = list(curve)
    for cut in (FLANGE_START, *BANDS):
        rs = [p[0] for p in out]
        if any(abs(r - cut) < 1e-9 for r in rs):
            continue
        for k in range(len(out) - 1):
            (r0, z0), (r1, z1) = out[k], out[k + 1]
            if r0 < cut < r1:
                z = z0 + (z1 - z0) * (cut - r0) / (r1 - r0)
                out.insert(k + 1, (cut, z))
                break
    return out


def revolve(top, bot, lip):
    """A closed triangle mesh: the profile loop (top outward, lip, bottom inward) revolved."""
    loop = list(top) + list(lip) + list(reversed(bot))  # top pole ... rim ... bottom pole
    n = MERIDIANS
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    verts, rz = [], []
    ring_index = []
    for k, (r, z) in enumerate(loop):
        if r < 1e-9:  # a pole: one vertex
            ring_index.append([len(verts)] * n)
            verts.append((0.0, 0.0, z))
            rz.append((0.0, z))
            continue
        idx = []
        for th in theta:
            idx.append(len(verts))
            verts.append((r * math.cos(th), r * math.sin(th), z))
            rz.append((r, z))
        ring_index.append(idx)
    faces = []
    for k in range(len(loop) - 1):
        a, b = ring_index[k], ring_index[k + 1]
        for j in range(n):
            j2 = (j + 1) % n
            quad = (a[j], a[j2], b[j2], b[j])
            tris = [(quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])]
            for t in tris:
                if len({t[0], t[1], t[2]}) == 3:
                    faces.append(t)
    mesh = trimesh.Trimesh(np.asarray(verts), np.asarray(faces), process=False)
    mesh.fix_normals()
    return mesh, np.asarray(rz), loop


def texture(path: Path, size: int = 512) -> None:
    """White glaze with a coloured band near the rim and a hairline at the well's edge."""
    from PIL import Image

    yy, xx = np.mgrid[0:size, 0:size]
    rho = np.hypot((xx + 0.5) / size - 0.5, (yy + 0.5) / size - 0.5) / 0.49  # 1.0 at the rim
    img = np.empty((size, size, 3), dtype=np.float64)
    img[:] = (244, 243, 238)
    band = (rho > 0.905) & (rho < 0.955)
    img[band] = RIM_COLOR
    line = (rho > 0.655) & (rho < 0.668)
    img[line] = RIM_COLOR
    Image.fromarray(img.astype(np.uint8)).save(path)


def uvs(verts: np.ndarray) -> np.ndarray:
    return np.c_[0.5 + 0.49 * verts[:, 0] / R, 0.5 + 0.49 * verts[:, 1] / R]


def pieces(mesh: trimesh.Trimesh, rz: np.ndarray) -> list[np.ndarray]:
    """Vertex sets whose hulls cover the drawn plate exactly: the well, then sector x band cells."""
    v = np.asarray(mesh.vertices)
    r = rz[:, 0]
    theta = np.mod(np.arctan2(v[:, 1], v[:, 0]), 2.0 * np.pi)
    eps = 1e-9
    out = [v[r <= FLANGE_START + eps]]
    cuts = [FLANGE_START, *BANDS, R + 1.0]
    width = 2.0 * np.pi / SECTORS
    for i in range(SECTORS):
        a0, a1 = i * width, (i + 1) * width
        d = np.mod(theta - a0 + 1e-7, 2.0 * np.pi)
        in_arc = (d <= (a1 - a0) + 2e-7) | (r < eps)
        for b0, b1 in zip(cuts[:-1], cuts[1:]):
            sel = in_arc & (r >= b0 - eps) & (r <= b1 + eps)
            if sel.sum() >= 4:
                out.append(v[sel])
    return out


def offset_hull(points: np.ndarray, distance: float) -> trimesh.Trimesh:
    """The hull of ``points`` grown by at least ``distance`` everywhere (Minkowski with a circumscribing icosahedron)."""
    ico = trimesh.creation.icosahedron()
    inradius = float(np.min(np.abs(np.einsum("ij,ij->i", ico.face_normals, ico.vertices[ico.faces[:, 0]]))))
    corners = ico.vertices * (distance / inradius)
    hull = trimesh.convex.convex_hull((points[:, None, :] + corners[None, :, :]).reshape(-1, 3))
    if not hull.is_volume:
        raise RuntimeError("collider piece is not a closed volume")
    return hull


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*"):
        old.unlink()
    top, bot, lip = profile()
    mesh, rz, loop = revolve(top, bot, lip)
    if not mesh.is_watertight or mesh.volume <= 0.0:
        raise RuntimeError("the revolved plate is not a closed solid")
    uv = uvs(np.asarray(mesh.vertices))
    lines = [f"v {x:.7f} {y:.7f} {z:.7f}" for x, y, z in mesh.vertices]
    lines += [f"vt {u:.7f} {w:.7f}" for u, w in uv]
    lines += [f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in mesh.faces]
    (OUT / "plate.obj").write_text("\n".join(lines) + "\n")
    texture(OUT / "texture.png")

    hulls = [offset_hull(p, SKIN) for p in pieces(mesh, rz)]
    for i, h in enumerate(hulls):
        h.export(OUT / f"hull_{i:03d}.obj")

    # every drawn point inside the collider, checked independently of how it was built
    check, _ = trimesh.sample.sample_surface_even(mesh, 300000, seed=7)
    check = np.vstack([check, mesh.vertices])
    inside = np.full(len(check), np.inf)
    for h in hulls:
        nrm = h.face_normals
        off = np.einsum("ij,ij->i", nrm, h.vertices[h.faces[:, 0]])
        inside = np.minimum(inside, (check @ nrm.T - off).max(axis=1))
    if inside.max() > -0.5 * SKIN:
        raise RuntimeError(f"a drawn point is only {-inside.max() * 1000:.3f} mm inside the collider")

    mass = DENSITY * float(mesh.volume)
    meta = {
        "R": R,
        "depth": float(mesh.bounds[1][2]),
        "flange_start": FLANGE_START,
        "flange_angle": FLANGE_ANGLE,
        "thickness": THICKNESS,
        "lip_end": LIP_END,
        "well_top": WELL_TOP,
        "foot": list(FOOT),
        "skin": SKIN,
        "density": DENSITY,
        "mass": mass,
        "volume": float(mesh.volume),
        "com": [float(c) for c in mesh.center_mass],
        "inertia_unit_density": np.asarray(mesh.moment_inertia, dtype=np.float64).tolist(),
        "bounds": np.asarray(mesh.bounds, dtype=np.float64).tolist(),
        "hulls": len(hulls),
        "drawn_inside_collider_min_mm": round(float(-inside.max()) * 1000, 3),
    }
    (OUT / "plate.json").write_text(json.dumps(meta, indent=1))
    print(
        f"[plate] {2 * R * 1000:.0f} x {meta['depth'] * 1000:.1f} mm, {mass * 1000:.0f} g, {len(mesh.faces)} triangles, "
        f"watertight; {len(hulls)} collider pieces, every drawn point >= {meta['drawn_inside_collider_min_mm']} mm inside"
    )


if __name__ == "__main__":
    main()
