# SPDX-License-Identifier: Apache-2.0
"""Bands: a rod closed into a loop, of rubber.

A band is a :class:`~robosimstudio_materials.rod.spec.RodSpec` with
``closed=True`` whose nodes go once round a circle: a Cosserat rod with no ends,
which SuperDex closes with an element from the last node back to the first (its
tubular skin with it). Only that engine builds one; the capsule chain refuses a
closed rod by name. :func:`band` lays one flat on the bench and
:class:`BandMaterial` says what it is made of; :data:`RUBBER` and :data:`O_RING`
are the two shipped.

A band's stiffness is its cord's: ``EA`` for the stretch that makes it grip what
it is wound round, ``EI`` for how round it stays when nothing does. Both come
from the modulus and the cord radius (``EntityRodSuperdex.host_actor``).

:func:`loop_winding` is the measure a band is for: how many times a closed loop
goes round an axis -- 1 round a peg it encircles, 0 beside one, and an integer
whatever shape the loop is pulled into.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robosimstudio.assetlib.semantics.atoms import ArgSpec, AtomSpec, register_atom
from robosimstudio.assetlib.semantics.handlers import handler
from robosimstudio.specs import Vec3

from .spec import RodSpec

__all__ = ["O_RING", "RUBBER", "BandMaterial", "band", "encircles", "loop_winding"]


@dataclass(frozen=True, kw_only=True)
class BandMaterial:
    """What a band is made of, as SuperDex's rod reads it.

    Attributes:
        cord_radius: Radius of the cord's section [m]. The rod is round; a flat
            band of the same area is drawn and touched as this round cord.
        young_modulus: Elastic modulus [Pa]: ``EA``, ``EI`` and ``GJ`` with the radius.
        density: [kg/m^3].
        friction: Coulomb coefficient against what it touches. The rod reads
            friction as ``sqrt(RodSpec.mu_cable)``, so :func:`band` passes its square.
        poisson_ratio: Near 0.5 for a rubber: only ``G = E / 2(1 + v)`` uses it.
        stiffness_damping: Damping proportional to stiffness [s].
        friction_falloff: Slip speed [m/s] below which friction grows with slip.
            1 mm/s, as the SuperDex cable: at the engine's 10 mm/s nothing is static.
    """

    cord_radius: float
    young_modulus: float
    density: float
    friction: float
    poisson_ratio: float = 0.49
    stiffness_damping: float = 1.0e-3
    friction_falloff: float = 1.0e-3


RUBBER = BandMaterial(cord_radius=0.002, young_modulus=1.5e6, density=1100.0, friction=0.9)
"""A natural-rubber band, 4 mm thick, at 1.5 MPa: stretched a fifth over its rest length it pulls with
``E * pi * r^2 / 5`` = 3.8 N. Stretch is what it is for."""

O_RING = BandMaterial(cord_radius=0.0025, young_modulus=8.0e6, density=1250.0, friction=0.6)
"""A nitrile O-ring, 5 mm cord (NBR 70 Shore A, about 8 MPa at small strain): four times the rubber's
modulus on a fatter cord, so it stays round where a band flops, and is hard to stretch."""


def band(
    *,
    name: str = "band",
    diameter: float = 0.06,
    at: Vec3 = (0.0, 0.0, 0.0),
    material: BandMaterial = RUBBER,
    segments: int = 32,
    normal: Vec3 = (0.0, 0.0, 1.0),
    **kwargs,
) -> RodSpec:
    """A band round a circle: flat on the bench, or standing, hung on a peg.

    Args:
        diameter: Of the cord's centreline at rest [m]; the circumference, pi
            times it, is the length the band does not stretch at.
        at: The circle's centre, in the work-surface frame. Lying flat (the
            default ``normal``), ``z`` is raised by the rod's radius (a
            ``radius`` passed in ``kwargs``, else the material's cord), so
            ``z = 0`` lies on the bench.
        material: :data:`RUBBER`, :data:`O_RING` or one of your own.
        segments: Nodes round the loop, one element each. Keep the element (pi
            times the diameter over this) above the cord's diameter.
        normal: The circle's axis: ``(0, 0, 1)`` lies flat; ``(1, 0, 0)`` stands
            in the y-z plane, as a band hung on a peg pointing along x hangs.
        kwargs: Any other :class:`RodSpec` field but ``nodes`` and ``closed``, which the band sets.

    Needs the rod kind on SuperDex (``backend("superdex")``): a capsule chain has two ends.
    """
    if segments < 3:
        raise ValueError(f"band {name!r}: a loop needs at least 3 segments, got {segments}")
    stated = sorted({"nodes", "closed"} & set(kwargs))
    if stated:
        raise TypeError(
            f"band {name!r}: a band is closed round its diameter, so it takes no {' or '.join(stated)}; "
            "a loop of your own shape is RodSpec(nodes=..., closed=True)"
        )
    axis = np.asarray(normal, dtype=np.float64)
    axis /= float(np.linalg.norm(axis))
    u = np.cross(axis, [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [0.0, 1.0, 0.0])
    u /= float(np.linalg.norm(u))
    v = np.cross(axis, u)
    cord = float(kwargs.get("radius", material.cord_radius))  # the radius the rod will have
    centre = np.asarray(at, dtype=np.float64) + (np.array([0.0, 0.0, cord]) if abs(axis[2]) >= 0.9 else 0.0)
    angles = np.linspace(0.0, 2.0 * math.pi, int(segments), endpoint=False)
    radius = 0.5 * float(diameter)
    nodes = tuple(tuple(float(c) for c in centre + radius * (math.cos(a) * u + math.sin(a) * v)) for a in angles)
    fields = {
        "radius": material.cord_radius,
        "young_modulus": material.young_modulus,
        "density": material.density,
        "poisson_ratio": material.poisson_ratio,
        "stiffness_damping": material.stiffness_damping,
        "friction_falloff": material.friction_falloff,
        "mu_cable": material.friction**2,
        **kwargs,
    }
    return RodSpec(name=name, nodes=nodes, closed=True, **fields)


def _axis(origin, direction) -> tuple[np.ndarray, np.ndarray]:
    """``(point, unit direction)`` of an axis. ``(x, y)`` names a vertical line, and only that."""
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    d = d / float(np.linalg.norm(d))
    o = np.asarray(origin, dtype=np.float64).reshape(-1)
    if o.shape == (2,):
        if not np.allclose(d[:2], 0.0, atol=1.0e-9):
            raise ValueError(
                f"an (x, y) origin names a vertical axis; along {tuple(float(c) for c in d)} it needs (x, y, z), "
                f"got {origin}"
            )
        return np.array([o[0], o[1], 0.0]), d
    if o.shape != (3,):
        raise ValueError(f"an axis's origin is (x, y, z), or (x, y) for a vertical one, got {origin}")
    return o, d


def _subtended(points: np.ndarray, o: np.ndarray, d: np.ndarray) -> np.ndarray:
    """The signed angle [rad] each segment of a closed loop subtends about an axis, the closing segment last."""
    u = np.cross(d, [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= float(np.linalg.norm(u))
    v = np.cross(d, u)
    rel = points - o
    angles = np.arctan2(rel @ v, rel @ u)
    delta = np.diff(np.concatenate([angles, angles[:1]]))
    return np.arctan2(np.sin(delta), np.cos(delta))


def loop_winding(positions: np.ndarray, origin, direction=(0.0, 0.0, 1.0)) -> float:
    """How many times a closed loop goes round an axis: signed, near an integer.

    The angle each segment subtends about the axis, summed round the whole loop
    -- the closing segment from the last point back to the first included -- in
    turns. A loop round the axis reads +-1 however it is pulled out of round; one
    beside it reads 0; twice round, 2. Unlike
    :func:`~robosimstudio_materials.rod.measures.winding_turns`, which counts only
    segments near the axis because a cable's ends leave it, a loop has no ends,
    so every segment counts and the answer is a whole number.

    Args:
        positions: ``[n, 3]`` points in loop order. A repeated first point at the
            end (a band's seam particle) is a zero-length segment and changes nothing.
        origin: A point on the axis [m]; ``(x, y)`` is enough for a vertical one,
            and any other direction refuses it.
        direction: The axis's direction; vertical by default. Only its line
            matters, and its sign only the sign of the answer.
    """
    o, d = _axis(origin, direction)
    points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3:
        return 0.0
    return float(_subtended(points, o, d).sum() / (2.0 * math.pi))


def encircles(positions: np.ndarray, origin, direction, length: float | None = None) -> bool:
    """Whether a closed loop is round a peg: once round its axis, and on the peg rather than past its end.

    :func:`loop_winding` counts turns round the axis's whole line, so a band
    hanging in front of a peg's tip, the line through the peg passing through
    the loop, would count. ``length`` closes that: at least half of the turn
    must be made between ``origin`` and ``origin + length * direction``, each
    segment placed by its height along the axis, not by how near the axis it
    is. A band lying slack round a post reads ``True`` however wide it is; one
    held over the post's top, or slid more than half off its end, ``False``.
    ``None`` takes the line.

    Args:
        positions: ``[n, 3]`` points in loop order [m].
        origin: The peg's root on its axis [m]. ``(x, y)`` is enough for a
            vertical axis with no ``length``, as for :func:`loop_winding`; a
            length is counted from the root, so it needs ``(x, y, z)``.
        direction: Along the peg, root to tip.
        length: The peg's length [m], or ``None``.
    """
    root = np.asarray(origin, dtype=np.float64).reshape(-1)
    if length is not None and root.shape != (3,):
        raise ValueError(
            f"encircles: a peg's length is counted from its root, so origin needs its height, got {origin}"
        )
    if abs(loop_winding(positions, origin, direction)) < 0.5:
        return False
    if length is None:
        return True
    o, d = _axis(root, direction)
    points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    along = (points - o) @ d
    middle = 0.5 * (along + np.roll(along, -1))  # each segment's height, the closing one last
    on = (middle >= 0.0) & (middle <= float(length))
    return bool(abs(_subtended(points, o, d)[on].sum()) >= math.pi)


register_atom(
    AtomSpec(
        name="encircles",
        summary="Whether a closed 1-D body -- a band, an O-ring -- goes once round an axis: a peg it is on.",
        fn=encircles,
        args=(
            ArgSpec(name="body", substrates=("rod",)),
            ArgSpec(name="about", kinds=("axis", "rod"), reads="feature"),
        ),
        directions=("check", "observe"),
        returns=("encircles", "turns"),
        unit="turns",
        provenance=(
            "Designed here for the closed SuperDex rod (RodSpec.closed): a loop's winding about a line is an "
            "integer, so the check is |turns| >= 0.5 with no calibration band; a rod feature (a peg's two ends) "
            "or an axis that states a length is held to the peg's span: half the turn must be made between its "
            "root and its tip, by height along it, or a band hanging in front of its tip would also count."
        ),
    )
)


@handler(
    "encircles",
    kinds=("", "axis"),
    substrates=("rod", ""),
    summary="The loop's turns about the axis, every segment counted and the loop closed.",
)
def _encircles_axis(positions: np.ndarray, axis: dict) -> dict:
    """A loop's points against an ``axis`` feature in world: ``origin``, ``direction`` and an optional ``length``."""
    origin, direction = axis["origin"], axis.get("direction", (0.0, 0.0, 1.0))
    length = axis.get("length")
    return {
        "turns": loop_winding(positions, origin, direction),
        "encircles": encircles(positions, origin, direction, None if length is None else float(length)),
    }


@handler(
    "encircles",
    kinds=("", "rod"),
    substrates=("rod", ""),
    summary="Round a peg declared as a rod feature: its two ends, root first, so the band must be on it.",
)
def _encircles_rod(positions: np.ndarray, rod: dict) -> dict:
    """A loop's points against a ``rod`` feature in world: ``axis`` is the peg's two ends."""
    ends = np.asarray(rod["axis"], dtype=np.float64).reshape(2, 3)
    span = ends[1] - ends[0]
    return {
        "turns": loop_winding(positions, ends[0], span),
        "encircles": encircles(positions, ends[0], span, float(np.linalg.norm(span))),
    }
