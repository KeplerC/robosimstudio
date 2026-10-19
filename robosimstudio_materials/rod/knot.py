# SPDX-License-Identifier: Apache-2.0
"""Knots: a rod laid in an overhand knot, and the measures that say it is still one.

A knot is a rod whose centreline crosses itself, held apart where it crosses by
nothing but its own contact. On SuperDex that is ``RodSpec.self_contact``: the
rod's skin is tested against itself, so a strand pulled across another is pushed
back instead of passing through it. Without it a pulled knot unties through
itself, which is the control every knot measure here is built to catch.

:func:`overhand` lays the nodes of a loose overhand knot flat on the bench, two
straight tails out of it; the rest of the rod is the cable you already know.

Three measures, all on the ordered centreline:

* :func:`self_clearance` -- the closest any two segments not next to each other
  come, axis to axis, less the rod's diameter. Never below zero by more than the
  contact's own give, or the rod went through itself.
* :func:`crossings` -- how many times the centreline crosses itself seen from
  above. An overhand knot shows three; a rod that untied through itself and was
  pulled straight shows none.
* :func:`knot_length` -- how much rod lies between its first crossing and its
  last: the length tied up in the knot, which falls as the knot is pulled tight.
"""

from __future__ import annotations

import math

import numpy as np

from robosimstudio.assetlib.semantics.atoms import ArgSpec, AtomSpec, register_atom
from robosimstudio.assetlib.semantics.handlers import handler
from robosimstudio.core.probes.clearance import segment_segment

__all__ = ["crossings", "knot_length", "overhand", "self_clearance"]


def overhand(
    *,
    at=(0.0, 0.0),
    size: float = 0.05,
    radius: float = 0.005,
    tail: float = 0.15,
    spacing: float = 0.008,
    gap: float = 0.2 * math.pi,
) -> np.ndarray:
    """The nodes of a loose overhand knot lying on the bench, tails along y, in the work-surface frame.

    An open trefoil: the knot's three lobes, each strand passing over, under and
    over again, lifted at the over-crossings by enough that no two strands touch
    as built (3.8 mm apart for a 12 mm rope tied 100 mm across), and two tails
    bending out of it to -y and +y along the bench. Resampled to ``spacing`` along its length, so
    every element is the same length.

    Args:
        at: The knot's centre on the bench [m].
        size: Across the knot's lobes [m], roughly.
        radius: The rod's radius [m]: the lowest node rests at it, and an
            over-strand's axis is four of it above the under-strand's.
        tail: Each straight tail's length [m].
        spacing: The element length [m]. Keep it under the rod's diameter, so
            the skin is closed where a strand bends round another.

    Returns:
        ``[n, 3]`` nodes, the first at the end of the -y tail.
    """
    # A trefoil, (sin t + 2 sin 2t, cos t - 2 cos 2t, -sin 3t), with the tip of one lobe cut away
    # (t = pi/3 +- gap, between the crossings at 0.09 pi and 0.58 pi), turned so that lobe points
    # along +x. Each end is then on a side of the lobe, and each tail bends out of it sideways, one
    # to -y and one to +y, as the two ends of an overhand knot leave it.
    t = np.linspace(math.pi / 3 + float(gap), math.pi / 3 + 2.0 * math.pi - float(gap), 480)
    x = np.sin(t) + 2.0 * np.sin(2.0 * t)
    y = np.cos(t) - 2.0 * np.cos(2.0 * t)
    z = -np.sin(3.0 * t)
    tip = math.atan2(
        math.cos(math.pi / 3) - 2.0 * math.cos(2 * math.pi / 3), math.sin(math.pi / 3) + 2 * math.sin(2 * math.pi / 3)
    )
    c, s = math.cos(-tip), math.sin(-tip)
    x, y = c * x - s * y, s * x + c * y
    scale = size / 6.0
    lift = 2.0 * radius  # the over-strand's axis four radii above the under-strand's at a crossing
    knot = np.stack([x * scale, y * scale, radius + lift * (z + 1.0)], axis=1)
    bend = max(2.5 * radius, 0.25 * size)  # the tails' turn [m], wider than the rod

    def tail_from(end: np.ndarray, heading: np.ndarray, side: float) -> np.ndarray:
        """From ``end`` along ``heading``, round to ``side`` y, and straight on: ``[m, 3]``, the end first."""
        d = heading / np.linalg.norm(heading)
        knee = end + bend * d
        out = knee + np.array([0.0, side * bend, 0.0])
        u = np.linspace(0.0, 1.0, 24)[:, None]
        arc = (1 - u) ** 2 * end + 2 * u * (1 - u) * knee + u**2 * out
        far = out + np.array([0.0, side * float(tail), 0.0])
        arc[:, 2] = np.linspace(end[2], radius, len(arc))
        far[2] = radius
        return np.vstack([arc, far])

    lower_first = knot[0, 1] < knot[-1, 1]
    head = tail_from(knot[0], knot[0] - knot[1], -1.0 if lower_first else 1.0)
    foot = tail_from(knot[-1], knot[-1] - knot[-2], 1.0 if lower_first else -1.0)
    path = np.vstack([head[::-1], knot[1:-1], foot])
    if not lower_first:
        path = path[::-1]  # the first node is the end of the -y tail
    path[:, :2] += np.asarray(at, dtype=np.float64) - 0.5 * (path[0, :2] + path[-1, :2])
    # Resample at equal arc length.
    legs = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(legs)])
    count = max(2, int(round(arc[-1] / float(spacing))) + 1)
    s_new = np.linspace(0.0, arc[-1], count)
    return np.stack([np.interp(s_new, arc, path[:, k]) for k in range(3)], axis=1)


def _segment_distances(points: np.ndarray, skip: int, *, closed: bool = False) -> np.ndarray:
    """Axis-to-axis distance between every pair of segments at least ``skip`` apart along the rod, ``[pairs]``.

    The clearance probe's closest points of two segments, a zero-length leg measured as the point it is.
    ``closed``: ``points`` go once round a loop, its closing segment last, and two segments are as
    far apart as the shorter way round.
    """
    a0, a1 = points[:-1], points[1:]
    i, j = np.triu_indices(len(a0), k=max(1, int(skip)))
    if closed:
        keep = len(a0) - (j - i) >= int(skip)
        i, j = i[keep], j[keep]
    return segment_segment(a0, a1, a0, a1)[i, j]


def self_clearance(positions: np.ndarray, radius: float, *, skip: int | None = None, closed: bool = False) -> float:
    """The closest two non-neighbouring segments come, axis to axis, less the rod's diameter [m].

    Positive is a gap between the two surfaces; zero is touching; negative is one
    strand inside another, by that much. Segments fewer than ``skip`` apart along
    the rod are neighbours and not compared: by default those within a diameter
    and a half of arc of each other, which a bend cannot bring into contact.

    Args:
        positions: ``[n, 3]`` nodes in order [m].
        radius: The rod's radius [m].
        skip: Segments this many apart or more are compared; ``None`` derives it.
        closed: A loop (``RodSpec.closed``, a band): its closing segment is measured too, and
            neighbours are counted the shorter way round, across the seam. A repeated first node
            at the end (a band's seam particle) is the same loop.
    """
    points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if closed:
        if len(points) > 1 and np.array_equal(points[0], points[-1]):
            points = points[:-1]
        points = np.vstack([points, points[:1]])  # the closing segment last
    if len(points) < 4:
        return math.inf
    if skip is None:
        spacing = float(np.median(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        skip = max(2, int(math.ceil(3.0 * radius / max(spacing, 1e-9))) + 1)
    pairs = _segment_distances(points, skip, closed=closed)
    if not len(pairs):
        return math.inf
    return float(pairs.min() - 2.0 * radius)


def _crossing_pairs(points: np.ndarray) -> list[tuple[int, int]]:
    """Every pair of non-adjacent segments whose projections onto the bench cross, ``(i, j)`` with ``i < j``."""
    p = points[:, :2]
    a, b = p[:-1], p[1:]
    n = len(a)
    found = []
    for i in range(n - 2):
        j = np.arange(i + 2, n)
        d1 = b[i] - a[i]
        d2 = b[j] - a[j]
        r = a[j] - a[i]
        denom = d1[0] * d2[:, 1] - d1[1] * d2[:, 0]
        ok = np.abs(denom) > 1e-18
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (r[:, 0] * d2[:, 1] - r[:, 1] * d2[:, 0]) / denom
            u = (r[:, 0] * d1[1] - r[:, 1] * d1[0]) / denom
        hit = ok & (s >= 0.0) & (s < 1.0) & (u >= 0.0) & (u < 1.0)
        found.extend((i, int(k)) for k in j[hit])
    return found


def crossings(positions: np.ndarray) -> int:
    """How many times the centreline crosses itself, seen from above.

    The crossings of its projection onto the bench, so not an invariant of the
    knot -- a loop laid over itself shows one -- but an overhand knot cannot show
    fewer than three, and a rod pulled straight shows none.
    """
    points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    return len(_crossing_pairs(points)) if len(points) >= 4 else 0


def knot_length(positions: np.ndarray) -> float:
    """The rod between its first crossing and its last, along it [m]: what the knot ties up.

    0 when it crosses nowhere. Pulled tight, the knot gives rod back to its tails
    and this falls.
    """
    points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    pairs = _crossing_pairs(points) if len(points) >= 4 else []
    if not pairs:
        return 0.0
    segments = [k for pair in pairs for k in pair]
    lo, hi = min(segments), max(segments)
    return float(np.linalg.norm(np.diff(points[lo : hi + 2], axis=0), axis=1).sum())


register_atom(
    AtomSpec(
        name="crossings",
        summary="How many times a 1-D body crosses itself seen from above: three for an overhand knot, none pulled straight.",
        fn=crossings,
        args=(ArgSpec(name="body", substrates=("rod",)),),
        directions=("check", "observe"),
        returns=("crossings", "knot_length"),
        provenance="Designed here for knots; a projection's crossings, so not a knot invariant, which the summary says.",
    )
)


@handler(
    "crossings", substrates=("rod",), summary="The centreline's crossings seen from above, and the rod between them."
)
def _crossings(positions: np.ndarray) -> dict:
    return {"crossings": crossings(positions), "knot_length": knot_length(positions)}
