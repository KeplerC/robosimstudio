# SPDX-License-Identifier: Apache-2.0
"""A camera solved from points whose place in the world and in the picture are both known.

Moving a camera by eye until a still lines up takes a twin many saves, and each
save changes the pose, the field of view and the roll at once. A clip already
holds the answer wherever a point of known position shows: the corners of the
bench, a robot's base, a fixture's edge, the crossings of a gridded mat. Given
six or more of them, :func:`fit_camera` solves the pinhole -- the focal length,
and the pose -- by least squares on the reprojection error, and says how far
each point lands from where it was clicked, so a point clicked wrong shows as
the one with the large error.

The answer is a :class:`~robosimstudio.core.cameras.CameraSpec`'s keywords
(``eye``, ``look_at``, ``up``, ``fx``/``fy``/``cx``/``cy``), in the world frame,
as every camera is, and it goes into ``scene.py`` as written.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["fit_camera", "overlay", "parse_point"]


def parse_point(text: str) -> tuple[tuple[float, float, float], tuple[float, float]]:
    """``"0.6,0.3,0.75=812,455"`` -> ``((0.6, 0.3, 0.75), (812.0, 455.0))``: a world point and its pixel."""
    world, sep, pixel = str(text).replace(" ", "").partition("=")
    if not sep:
        raise ValueError(f"a point is x,y,z=u,v (world metres = pixel), got {text!r}")
    xyz = [float(v) for v in world.split(",")]
    uv = [float(v) for v in pixel.split(",")]
    if len(xyz) != 3 or len(uv) != 2:
        raise ValueError(f"a point is x,y,z=u,v (world metres = pixel), got {text!r}")
    return (xyz[0], xyz[1], xyz[2]), (uv[0], uv[1])


def _rodrigues(r: np.ndarray) -> np.ndarray:
    """The rotation matrix of an axis-angle vector."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(np.asarray(r, dtype=np.float64)).as_matrix()


def _project(params: np.ndarray, world: np.ndarray, cx: float, cy: float, free_center: bool) -> np.ndarray:
    f = math.exp(params[0])
    rot = _rodrigues(params[1:4])  # world -> camera
    t = params[4:7]
    if free_center:
        cx, cy = params[7], params[8]
    cam = world @ rot.T + t
    z = np.where(np.abs(cam[:, 2]) < 1e-9, 1e-9, cam[:, 2])
    return np.stack([f * cam[:, 0] / z + cx, f * cam[:, 1] / z + cy], axis=1)


def _sphere(n: int) -> list[tuple[float, float, float]]:
    """``n`` directions spread evenly over the sphere (a Fibonacci lattice)."""
    out = []
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - z * z))
        a = i * math.pi * (3.0 - math.sqrt(5.0))
        out.append((r * math.cos(a), r * math.sin(a), z))
    return out


def _look_rotation(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World -> camera rotation of a camera at ``eye`` looking at ``target``, in CameraSpec's basis."""
    from robosimstudio.core.math import look_at_matrix

    return look_at_matrix(eye, target)[:3, :3].T


def _rotvec(rot: np.ndarray) -> np.ndarray:
    """The axis-angle vector of a rotation matrix (the inverse of :func:`_rodrigues`)."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(rot).as_rotvec()


def fit_camera(
    world: Any,
    pixels: Any,
    width: int,
    height: int,
    *,
    free_center: bool = False,
    fov_y: float | None = None,
) -> dict[str, Any]:
    """The pinhole that best puts ``world`` points (N x 3, metres) on ``pixels`` (N x 2), and how well.

    The focal length is solved unless ``fov_y`` fixes it; the principal point is
    the image centre unless ``free_center`` (which wants ten or more points
    spread over the picture). Returns the CameraSpec keywords, the RMS and the
    per-point reprojection error in pixels.
    """
    from scipy.optimize import least_squares

    world = np.asarray(world, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if len(world) != len(pixels):
        raise ValueError("as many pixels as world points")
    need = 10 if free_center else (4 if fov_y else 6)
    if len(world) < need:
        raise ValueError(f"{len(world)} points; this fit needs at least {need}")
    cx0, cy0 = 0.5 * width, 0.5 * height
    fixed_f = None if fov_y is None else 0.5 * height / math.tan(math.radians(fov_y) * 0.5)
    centre = world.mean(axis=0)
    spread = max(float(np.linalg.norm(world - centre, axis=1).max()), 0.05)

    def residual(p: np.ndarray) -> np.ndarray:
        if fixed_f is not None:
            p = p.copy()
            p[0] = math.log(fixed_f)
        return (_project(p, world, cx0, cy0, free_center) - pixels).ravel()

    best, behind = None, None
    # Nothing here is convex: start from cameras all round the points, at several distances and fields of
    # view, each looking at their centre, and keep the best fit that has the points in front of it.
    directions = [np.array(d, dtype=np.float64) for d in _sphere(26)]
    for fov in [fov_y] if fov_y else (30.0, 50.0, 75.0):
        f = 0.5 * height / math.tan(math.radians(fov) * 0.5)
        for direction in directions:
            for distance in (2.5, 5.0):
                eye = centre + direction * spread * distance
                rot = _look_rotation(eye, centre)
                x0 = np.concatenate([[math.log(f)], _rotvec(rot), -rot @ eye] + ([[cx0, cy0]] if free_center else []))
                sol = least_squares(residual, x0, method="lm" if len(world) * 2 > len(x0) else "trf")
                depth = (world @ _rodrigues(sol.x[1:4]).T + sol.x[4:7])[:, 2]
                cost = float(np.mean(sol.fun**2))
                if (depth <= 0).mean() > 0.5:
                    # seen from behind: a fit there means one axis of the points is flipped, or pairs are crossed
                    behind = cost if behind is None else min(behind, cost)
                    continue
                if best is None or cost < best[0]:
                    best = (cost, sol.x)
    # a set on one plane fits as well from behind as from in front: mirrored only when the front fit is poor
    mirrored = behind is not None and (best is None or (best[0] > 1.0 and behind < 0.1 * best[0]))
    if mirrored:
        best = None
    if best is None:
        if mirrored:
            raise ValueError(
                "these points fit only from behind the camera, as a mirror image: one world axis is flipped "
                "(+y is to the robot's left when it faces +x), or left and right pixels are swapped"
            )
        raise ValueError("no pose puts these points in front of a camera; check the pairs")
    params = best[1]
    f = fixed_f if fixed_f is not None else math.exp(params[0])
    params = params.copy()
    params[0] = math.log(f)
    rot = _rodrigues(params[1:4])
    t = params[4:7]
    cx, cy = (float(params[7]), float(params[8])) if free_center else (cx0, cy0)
    proj = _project(params, world, cx0, cy0, free_center)
    err = np.linalg.norm(proj - pixels, axis=1)
    depth = (world @ rot.T + t)[:, 2]
    eye = -rot.T @ t
    forward, down = rot[2], rot[1]  # camera +z and +y in the world
    spec = {
        "width": int(width),
        "height": int(height),
        "fx": round(f, 2),
        "fy": round(f, 2),
        "cx": round(cx, 2),
        "cy": round(cy, 2),
        "eye": tuple(round(float(v), 4) for v in eye),
        "look_at": tuple(round(float(v), 4) for v in eye + forward),
        "up": tuple(round(float(v), 4) for v in -down),
    }
    return {
        "spec": spec,
        "fov_y": round(math.degrees(2.0 * math.atan(0.5 * height / f)), 2),
        "rms_px": round(float(math.sqrt(np.mean(err**2))), 2),
        "errors_px": [round(float(e), 2) for e in err],
        "behind": int((depth <= 0).sum()),
        "projected": proj.tolist(),
    }


def overlay(still: str | Path, pixels: Any, projected: Any, out: str | Path, errors: Any = ()) -> Path:
    """The still with each clicked pixel (yellow ring) and where the fitted camera puts its point (cyan dot)."""
    from PIL import Image, ImageDraw

    with Image.open(still) as opened:
        img = opened.convert("RGB")
    draw = ImageDraw.Draw(img)
    r = max(4, img.width // 240)
    errors = list(errors) or [None] * len(pixels)
    for i, ((u, v), (pu, pv), e) in enumerate(zip(pixels, projected, errors, strict=False)):
        draw.ellipse((u - 2 * r, v - 2 * r, u + 2 * r, v + 2 * r), outline=(255, 220, 0), width=max(2, r // 2))
        draw.line((u, v, pu, pv), fill=(255, 80, 80), width=max(1, r // 3))
        draw.ellipse((pu - r, pv - r, pu + r, pv + r), fill=(0, 230, 255))
        label = f"{i}" if e is None else f"{i}: {e:.1f}px"
        draw.text((u + 2 * r + 2, v - 2 * r), label, fill=(255, 220, 0))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out
