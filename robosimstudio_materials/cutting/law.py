# SPDX-License-Identifier: Apache-2.0
"""The geometry and the edge law between a knife and a block, in plain numpy.

Nothing here touches Warp or the cage solver, so every rule the cutting kind applies can be read
and tested on a CPU. Four things:

* the block's frames: the world, the block's own (metres from its lowest corner, ``z`` up) and the
  cutter's (the block's own scaled so its longest side is 1, the size the vendored solver's
  tolerances are written for);
* the box the block starts as: an eight-vertex cage and a welded triangle skin;
* the cut a knife starts: the plane its blade lies in, the line its edge sweeps along, and the
  turn that keeps that line off every face plane it crosses;
* the edge law: how hard the block pushes back on the edge, and how far the cut runs.

**The edge law is this kind's own; the paper's solver has no contact.** The block meets the
edge at the cut's front, the line where it is still whole. With the edge ``delta`` metres past
the front, ``g`` metres short of the block's far side and ``L`` metres of it inside the block,
the block pushes back along the cut's sweep with::

    F = push * E * L * delta * (g + delta) / g

a stiffness of ``push * E`` per metre of edge for a shallow press, growing without bound as the
edge squeezes the last of the block against the board. The cut runs when ``F`` reaches the
blade's threshold, ``toughness / sharpness * L``: the front moves to where the push equals the
threshold, and the knife feels the threshold while it cuts. A sharp knife in tofu sees a
threshold indentation of ``toughness / (sharpness * E) = 0.13 mm``; a blade a thousand times
duller never reaches it before the arm stalls, and the block does not part.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from robosimstudio.core.math import quat_to_mat

__all__ = [
    "BlockFrame",
    "EdgeContact",
    "blade_normal",
    "box_cage",
    "box_surface",
    "cut_line",
    "edge_contact",
    "edge_push",
]


# =============================================================================
# 1. Frames
# =============================================================================


@dataclass(frozen=True)
class BlockFrame:
    """Where a block stands, and the cutter's scaled copy of its frame.

    Attributes:
        origin: The block's lowest corner in world [m].
        rotation: Block axes in world, as columns.
        size: The block's extents [m].
        scale: Cutter units per metre: one over the longest side.
    """

    origin: np.ndarray
    rotation: np.ndarray
    size: np.ndarray
    scale: float

    @classmethod
    def from_base(cls, base_pose, size) -> BlockFrame:
        """From the world pose ``(x, y, z, qx, qy, qz, qw)`` of the middle of the block's base."""
        pose = np.asarray(base_pose, dtype=np.float64).reshape(-1)
        rotation = quat_to_mat(pose[3:7])
        size = np.asarray(size, dtype=np.float64)
        origin = pose[:3] - rotation @ np.array([0.5 * size[0], 0.5 * size[1], 0.0])
        return cls(origin=origin, rotation=rotation, size=size, scale=1.0 / float(size.max()))

    def to_block(self, points) -> np.ndarray:
        """World points [m] in the block's own frame [m]."""
        return (np.asarray(points, dtype=np.float64) - self.origin) @ self.rotation

    def to_cutter(self, points) -> np.ndarray:
        return self.to_block(points) * self.scale

    def from_cutter(self, points) -> np.ndarray:
        """Cutter points back in world [m]."""
        return (np.asarray(points, dtype=np.float64) / self.scale) @ self.rotation.T + self.origin

    def vector_to_block(self, vectors) -> np.ndarray:
        return np.asarray(vectors, dtype=np.float64) @ self.rotation

    def vector_to_world(self, vectors) -> np.ndarray:
        return np.asarray(vectors, dtype=np.float64) @ self.rotation.T


# =============================================================================
# 2. The box
# =============================================================================

BOX_FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (0, 4, 7, 3), (1, 2, 6, 5))
"""The cage's six quads, counter-clockwise round their outward normals: base, top, -y, +y, -x, +x."""

BASE_FACE = 0
"""Index of the base in :data:`BOX_FACES`: the face the board holds."""


def box_cage(size) -> tuple[np.ndarray, list[np.ndarray]]:
    """The eight corners of a ``[0, size]`` box and its six faces."""
    sx, sy, sz = (float(v) for v in size)
    verts = np.array(
        [[0, 0, 0], [sx, 0, 0], [sx, sy, 0], [0, sy, 0], [0, 0, sz], [sx, 0, sz], [sx, sy, sz], [0, sy, sz]], float
    )
    return verts, [np.array(f, dtype=np.int64) for f in BOX_FACES]


def box_surface(size, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """A welded triangle skin over a ``[0, size]`` box, about ``cell`` per triangle leg."""
    size = np.asarray(size, dtype=np.float64)
    n = [max(1, round(float(s) / float(cell))) for s in size]
    verts: list[np.ndarray] = []
    tris: list[tuple[int, int, int]] = []

    def face(origin, u, v, nu, nv) -> None:
        base = len(verts)
        for j in range(nv + 1):
            for i in range(nu + 1):
                verts.append(origin + u * (i / nu) + v * (j / nv))
        for j in range(nv):
            for i in range(nu):
                a = base + j * (nu + 1) + i
                tris.append((a, a + 1, a + nu + 2))
                tris.append((a, a + nu + 2, a + nu + 1))

    o = np.zeros(3)
    x, y, z = np.diag(size)
    face(o, y, x, n[1], n[0])
    face(z, x, y, n[0], n[1])
    face(o, x, z, n[0], n[2])
    face(y, z, x, n[2], n[0])
    face(o, z, y, n[2], n[1])
    face(x, y, z, n[1], n[2])
    points = np.asarray(verts)
    key = np.round(points / (float(cell) * 1.0e-3)).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return points[first], inverse.reshape(-1)[np.asarray(tris)].astype(np.int64)


# =============================================================================
# 3. The cut a knife starts
# =============================================================================

MIN_TILT = math.radians(2.0)
"""The least angle between the cut's line and any face plane it crosses.

The vendored topology hosts a cut tip where the line meets a face, and refuses a face parallel to
the line; its own loader turns a line within 2 degrees of one away (``scene.py``). A knife held
level over a block is exactly that case, so the line is turned within the blade's plane until it
clears every crossed face by this much."""


def blade_normal(edge0, edge1, edgeward) -> np.ndarray:
    """The unit normal of the plane a blade lies in: square to its edge and to the way it cuts."""
    line = np.asarray(edge1, float) - np.asarray(edge0, float)
    line /= np.linalg.norm(line)
    b = np.asarray(edgeward, float)
    normal = np.cross(line, b - np.dot(b, line) * line)
    return normal / np.linalg.norm(normal)


def cut_line(edge0, edge1, edgeward, face_normals, crossed) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """The cut a knife starts: ``(line, sweep, normal, turn)``, unit vectors and the turn [rad].

    The plane is the blade's: it holds the edge and the way the blade cuts (``edgeward``). Its line
    is the edge's direction, turned about the plane's normal by the least multiple of half a degree
    that keeps it :data:`MIN_TILT` off every face plane the cut crosses; ``sweep`` is the in-plane
    direction square to the line, on the edgeward side.

    Args:
        edge0, edge1: The edge's ends.
        edgeward: Spine to edge, in the same frame.
        face_normals: ``(F, 3)`` unit normals of the faces to clear.
        crossed: ``(F,)`` which of them the plane crosses.
    """
    edge0, edge1 = np.asarray(edge0, float), np.asarray(edge1, float)
    line = edge1 - edge0
    line /= np.linalg.norm(line)
    b = np.asarray(edgeward, float)
    b = b - np.dot(b, line) * line
    b /= np.linalg.norm(b)
    normal = np.cross(line, b)
    normal /= np.linalg.norm(normal)
    normals = np.asarray(face_normals, float).reshape(-1, 3)[np.asarray(crossed, bool)]
    limit = math.sin(MIN_TILT)
    turn = 0.0
    for k in range(181):
        angle = math.radians(0.5 * ((k + 1) // 2) * (1 if k % 2 else -1))
        turned = math.cos(angle) * line + math.sin(angle) * np.cross(normal, line)
        if not len(normals) or float(np.min(np.abs(normals @ turned))) >= limit:
            line, turn = turned, angle
            break
    sweep = np.cross(normal, line)
    if np.dot(sweep, b) < 0.0:
        sweep = -sweep
    return line, sweep, normal, turn


# =============================================================================
# 4. The edge in the block, and the law
# =============================================================================


@dataclass(frozen=True)
class EdgeContact:
    """The part of a knife's edge over the block's footprint, in the block's frame [m].

    Attributes:
        length: How much of the edge lies over the footprint [m].
        points: The samples that do, ``(k, 3)``.
        centroid: Their middle.
        inside: How much of it lies between the block's base and top [m]: the edge law's ``L``, which
            a knife tipped with only its lowest few millimetres in the block has little of.
        inside_centroid: The middle of that part, where the push acts: on a tipped knife, low on the
            edge, not at the middle of all of it over the footprint. :attr:`centroid` with none inside.
    """

    length: float
    points: np.ndarray
    centroid: np.ndarray
    inside: float = 0.0
    inside_centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))


def _clip(edge0: np.ndarray, step: np.ndarray, lo: float, hi: float, axis: int, top: float) -> tuple[float, float]:
    """The part ``[lo, hi]`` of the segment's parameter that lies in ``[0, top]`` along ``axis``."""
    if abs(step[axis]) < 1.0e-15:
        return (lo, hi) if 0.0 <= edge0[axis] <= top else (1.0, 0.0)
    a = (0.0 - edge0[axis]) / step[axis]
    b = (top - edge0[axis]) / step[axis]
    return max(lo, min(a, b)), min(hi, max(a, b))


def edge_contact(edge0, edge1, size, samples: int = 17) -> EdgeContact:
    """The part of the segment ``edge0``-``edge1`` (block frame) over the block's ``x``-``y`` footprint.

    The segment is clipped to the footprint exactly, and ``samples`` points are spread over what is left;
    :attr:`EdgeContact.inside` clips it to the block's height too.
    """
    edge0, edge1 = np.asarray(edge0, float), np.asarray(edge1, float)
    step = edge1 - edge0
    lo, hi = 0.0, 1.0
    for axis in (0, 1):
        lo, hi = _clip(edge0, step, lo, hi, axis, float(size[axis]))
    if hi <= lo:
        middle = 0.5 * (edge0 + edge1)
        return EdgeContact(length=0.0, points=np.zeros((0, 3)), centroid=middle, inside_centroid=middle)
    s = np.linspace(lo, hi, int(samples))
    points = edge0[None, :] + s[:, None] * step[None, :]
    below, above = _clip(edge0, step, lo, hi, 2, float(size[2]))
    norm = float(np.linalg.norm(step))
    centroid = points.mean(axis=0)
    return EdgeContact(
        length=norm * (hi - lo),
        points=points,
        centroid=centroid,
        inside=norm * max(above - below, 0.0),
        inside_centroid=edge0 + 0.5 * (below + above) * step if above > below else centroid,
    )


def edge_push(
    *, depth: float, gap: float, length: float, young: float, push: float, threshold: float, floor: float
) -> tuple[float, float]:
    """The block's push on the edge [N] and how far behind the edge the front ends [m].

    Args:
        depth: How far the edge is past the front, along the sweep [m]; nothing pushes at or below 0.
        gap: How far the edge is short of the block's far side [m].
        length: Edge inside the block [m].
        young: The block's E [Pa].
        push: The law's stiffness as a multiple of ``E`` per metre of edge.
        threshold: The blade's threshold per metre of edge [N/m]: toughness over sharpness.
        floor: Within this of the far side, the rest parts [m].

    Returns:
        ``(force, lag)``: the push, and the front's new distance behind the edge, ``depth`` when the
        threshold is not reached (the front stays) and ``0`` when the edge is within ``floor`` of
        the far side (the block parts through).
    """
    if depth <= 0.0 or length <= 0.0:
        return 0.0, max(depth, 0.0)
    if gap <= floor:
        return float(threshold * length), 0.0
    k = float(push) * float(young) * float(length)
    force = k * depth * (gap + depth) / gap
    limit = float(threshold) * float(length)
    if force <= limit:
        return force, depth
    a = limit / k
    lag = 0.5 * (math.sqrt(gap * gap + 4.0 * gap * a) - gap)
    return limit, min(lag, depth)
