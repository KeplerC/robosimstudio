"""Signed distances to the scene's own primitives -- the shelf, the racks -- read off a built model.

Both are drawn exactly as they collide (boxes and capsules), so a distance to one
is a distance to what a viewer sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import newton
import numpy as np

from robosimstudio.core.math import quat_to_mat


@dataclass
class Box:
    label: str
    T: np.ndarray
    half: np.ndarray

    def signed(self, P: np.ndarray) -> np.ndarray:
        """Signed distance [m] from each point to the box's surface: negative inside."""
        Q = np.abs((P - self.T[:3, 3]) @ self.T[:3, :3]) - self.half
        outside = np.linalg.norm(np.maximum(Q, 0.0), axis=1)
        inside = np.minimum(np.max(Q, axis=1), 0.0)
        return outside + inside


@dataclass
class Capsule:
    label: str
    a: np.ndarray
    b: np.ndarray
    r: float

    def signed(self, P: np.ndarray) -> np.ndarray:
        ab = self.b - self.a
        t = np.clip(((P - self.a) @ ab) / max(float(ab @ ab), 1e-18), 0.0, 1.0)
        return np.linalg.norm(P - (self.a + t[:, None] * ab), axis=1) - self.r


def _T(row) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(np.asarray(row[3:7], dtype=np.float64))
    T[:3, 3] = row[:3]
    return T


def statics(model, prefixes=("shelf_", "rack_")) -> list:
    """The world-fixed boxes and capsules whose spec names start with one of ``prefixes``."""
    labels = [str(x).rsplit("/", 1)[-1] for x in model.shape_label]
    kinds = model.shape_type.numpy()
    tf = model.shape_transform.numpy()
    scale = model.shape_scale.numpy()
    body = model.shape_body.numpy()
    out = []
    for s, lab in enumerate(labels):
        if not lab.startswith(prefixes) or body[s] != -1:
            continue
        T = _T(tf[s])
        if kinds[s] == int(newton.GeoType.BOX):
            out.append(Box(lab, T, np.asarray(scale[s][:3], dtype=np.float64)))
        elif kinds[s] == int(newton.GeoType.CAPSULE):
            r, h = float(scale[s][0]), float(scale[s][1])
            out.append(Capsule(lab, T[:3, 3] - h * T[:3, 2], T[:3, 3] + h * T[:3, 2], r))
    return out


def nearest(P: np.ndarray, prims: list, near: float = 0.02) -> tuple[float, str]:
    """The smallest signed distance from any of ``P`` to any primitive, and which one [m]."""
    best, which = np.inf, ""
    lo, hi = P.min(axis=0) - near, P.max(axis=0) + near
    for p in prims:
        if isinstance(p, Capsule):
            plo = np.minimum(p.a, p.b) - p.r
            phi = np.maximum(p.a, p.b) + p.r
        else:
            ext = np.abs(p.T[:3, :3]) @ p.half
            plo, phi = p.T[:3, 3] - ext, p.T[:3, 3] + ext
        if np.any(phi < lo) or np.any(plo > hi):
            continue
        d = float(p.signed(P).min())
        if d < best:
            best, which = d, p.label
    return best, which
