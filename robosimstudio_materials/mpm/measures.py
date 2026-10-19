# SPDX-License-Identifier: Apache-2.0
"""Measures over a particle body: how much of it is where, how high it stands, how far it spread.

Five atoms on the ``granular`` layout, which the three particle kinds all read
as (``liquid`` and ``dough`` fall back to it, and it falls back to ``cloth``, so
``containment``, ``height`` and ``speed`` apply to a heap unchanged). Every one
of them is a **count**: the share of particles in a box, the particle mass in a
box, the share lying outside a box on the surface, a quantile of heights, a
footprint ratio. That is a choice forced by the backend and worth stating once:
Newton 1.6's water-like preset loses volume at tabletop scale (``probe_c_water``:
a 60 mm column in a thick tank is 34% of its volume after 2 s at rest), so a
height or a volume of a liquid is a number about the solver, and the number of
particles inside the bowl is a number about the pour.

The ``granular`` layout is minted here, before any kind declares a chain ending
in it: the domain scan imports ``dough`` before ``granular`` (alphabetically),
and a fallback must exist before a chain names it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosimstudio.assetlib.semantics.atoms import ArgSpec, AtomCalibration, AtomSpec, register_atom
from robosimstudio.assetlib.semantics.handlers import handler
from robosimstudio.assetlib.vocabulary.regions import containment, region_bounds
from robosimstudio.specs.registries import register_substrate

__all__ = [
    "HEIGHT_QUANTILE",
    "LAYOUT",
    "fill_fraction",
    "heap_height",
    "spill_fraction",
    "spread",
]

LAYOUT = "granular"
"""The state layout every particle kind reads as; a cloud of unordered points, like ``cloth``."""

register_substrate(LAYOUT, ("cloth",), origin="robosimstudio_materials.mpm")

HEIGHT_QUANTILE = 0.95
"""Which quantile of the particle heights is the body's height. Not the maximum:
a single grain thrown up by a tool, or a liquid particle the solver has let go,
is always somewhere above the pile."""


# ---------------------------------------------------------------- functions
def fill_fraction(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """What share of ``points`` lies inside the world box ``[lo, hi]``: :func:`containment`, under the name a pour uses."""
    return containment(points, lo, hi)


def spill_fraction(
    points: np.ndarray, lo: np.ndarray, hi: np.ndarray, *, floor: float | None = None, band: float | None = None
) -> float:
    """What share of ``points`` lies outside the box's footprint, at any height unless a band is given.

    Outside in x or y, whatever its z: what has left the patch it was placed
    on, whether it lies flat or stands in the ridge a wiper piles it into (the
    wipe example's 6 mm puddle, pushed 180 mm by the hand, ends as a 10 mm
    ridge, and a reading that wanted it within 5 mm of the bench counted 77%
    of a 98% push; ``session_wipe``, RTX PRO 6000, Newton 1.6.0). With
    ``band`` and ``floor``, only what lies within ``band`` of ``floor`` in z
    counts: the material that missed the tray and is on the bench, as against
    what is still falling -- a particle above the band is not a spill yet. One
    is given with the other.
    """
    array = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if not len(array):
        return 0.0
    if (floor is None) != (band is None):
        raise TypeError("spill_fraction: floor and band go together; give both or neither")
    outside = (array[:, 0] < lo[0]) | (array[:, 0] > hi[0]) | (array[:, 1] < lo[1]) | (array[:, 1] > hi[1])
    if band is not None:
        outside &= array[:, 2] <= float(floor) + float(band)
    return float(outside.mean())


def heap_height(points: np.ndarray, floor: float, *, quantile: float = HEIGHT_QUANTILE) -> float:
    """How high a body stands over ``floor`` [m]: a high quantile of its particles' z minus the floor."""
    array = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if not len(array):
        return 0.0
    return float(np.quantile(array[:, 2], quantile) - float(floor))


def spread(points: np.ndarray, baseline: np.ndarray) -> float:
    """Bounding-box footprint area over the settled one: 1 as settled, more once pressed or poured out.

    The axis-aligned box round every particle, so a trail, a ridge or one
    stray grain widens it as much as a true spread does; a reading of it says
    how far the body reaches, not how much of it moved. A count
    (:func:`fill_fraction`, :func:`spill_fraction`) says that.
    """
    now = np.atleast_2d(np.asarray(points, dtype=np.float64))
    then = np.atleast_2d(np.asarray(baseline, dtype=np.float64))
    if not len(now) or not len(then):
        return 0.0
    area_then = float(np.ptp(then[:, 0]) * np.ptp(then[:, 1]))
    if area_then <= 1.0e-9:
        return 0.0
    return float(np.ptp(now[:, 0]) * np.ptp(now[:, 1])) / area_then


# ------------------------------------------------------------------ catalogue
register_atom(
    AtomSpec(
        name="fill_fraction",
        summary="What share of a particle body lies inside a region.",
        args=(
            ArgSpec(name="body", substrates=(LAYOUT,)),
            ArgSpec(name="region", kinds=("region",), reads="region"),
        ),
        directions=("check", "observe"),
        returns=("fraction",),
        cost="cheap",
        provenance=(
            "containment, under the name a scoop or a pour is scored by. A count of particles rather "
            "than a height or a volume, because Newton 1.6's water-like preset loses volume at tabletop "
            "scale (probe_c_water) and a count is unaffected."
        ),
        calibration=AtomCalibration(
            positives="A heap built inside the region reads 1.0 after its settle.",
            negatives="A heap beside the region reads 0.0; one half pushed over the rim reads about 0.5.",
            gate=None,
            bench="the particle kinds' medium tests",
            notes="No gate: a task states its own share.",
        ),
    )
)


@handler(
    "fill_fraction",
    kinds=("", "region"),
    substrates=(LAYOUT, ""),
    summary="The share of the body's particles inside the box.",
)
def _fill_fraction(points: np.ndarray, region: dict, *, origin=(0.0, 0.0, 0.0)) -> dict[str, Any]:
    lo, hi = region_bounds(region, origin)
    return {"fraction": fill_fraction(points, lo, hi)}


register_atom(
    AtomSpec(
        name="mass_in_region",
        summary="How much of a particle body lies inside a region, by mass [kg].",
        args=(
            ArgSpec(name="body", substrates=(LAYOUT,), reads="handle"),
            ArgSpec(name="region", kinds=("region",), reads="region"),
        ),
        directions=("check", "observe"),
        returns=("mass", "fraction"),
        unit="kg",
        cost="cheap",
        reads_scene=True,
        provenance=(
            "fill_fraction times the body's mass: every particle of one body weighs its lattice cell's "
            "volume times the material's density, so the count is the mass."
        ),
    )
)


@handler(
    "mass_in_region",
    kinds=("", "region"),
    substrates=(LAYOUT, ""),
    summary="Particles inside the box, times the particle mass.",
)
def _mass_in_region(snapshot, body: str, region: dict) -> dict[str, Any]:
    entity, probe = _entity_and_probe(snapshot, body)
    lo, hi = region_bounds(region, (0.0, 0.0, 0.0))
    points = probe.points(body)
    fraction = fill_fraction(points, lo, hi)
    mass = fraction * len(points) * float(getattr(entity, "particle_mass", 0.0))
    return {"mass": mass, "fraction": fraction, f"{body}/mass_in_region_kg": mass}


register_atom(
    AtomSpec(
        name="spill_fraction",
        summary="What share of a particle body lies outside a region's footprint, down on the surface.",
        args=(
            ArgSpec(name="body", substrates=(LAYOUT,), reads="handle"),
            ArgSpec(name="region", kinds=("region",), reads="region"),
        ),
        directions=("check", "observe"),
        returns=("fraction",),
        cost="cheap",
        reads_scene=True,
        provenance=(
            "Designed here. Outside the region in x or y and within two particle spacings of the surface "
            "the body was placed on: what missed the tray and is lying on the bench, as against what is "
            "still in the air or in the heap. The function behind it counts at any height unless given "
            "the band, which is how a wipe reads the ridge its wiper piles up."
        ),
        calibration=AtomCalibration(
            positives="A puddle poured on the bench beside its region reads close to 1.0.",
            negatives="A heap resting inside its region reads 0.0.",
            gate=None,
            bench="the particle kinds' medium tests",
        ),
    )
)


@handler(
    "spill_fraction",
    kinds=("", "region"),
    substrates=(LAYOUT, ""),
    summary="Outside the box's footprint and within two spacings of the body's surface.",
)
def _spill_fraction(snapshot, body: str, region: dict) -> dict[str, Any]:
    entity, probe = _entity_and_probe(snapshot, body)
    lo, hi = region_bounds(region, (0.0, 0.0, 0.0))
    spacing = float(getattr(getattr(entity, "spec", None), "particle_spacing", 0.0025))
    fraction = spill_fraction(probe.points(body), lo, hi, floor=_floor_of(entity), band=2.0 * spacing)
    return {"fraction": fraction, f"{body}/spill_fraction": fraction}


register_atom(
    AtomSpec(
        name="heap_height",
        summary="How high a particle body stands over the surface it was placed on [m].",
        args=(ArgSpec(name="body", substrates=(LAYOUT,), reads="handle"),),
        directions=("check", "observe"),
        returns=("height",),
        unit="m",
        cost="cheap",
        reads_scene=True,
        provenance=(
            "Designed here: the 95th percentile of the particles' z minus the work surface. For the water "
            "preset this is a height of particles, not of water -- the preset loses volume in Newton 1.6 "
            "(probe_c_water: a 63 mm column reads 17-21 mm after 2 s at rest) -- so a liquid's fill is "
            "fill_fraction, and this is for a heap or a lump."
        ),
        calibration=AtomCalibration(
            positives="A 30 mm block of rice reads 0.028-0.030 m after its settle.",
            negatives="The same rice pushed flat across the bench reads under 0.01 m.",
            gate=None,
            bench="the particle kinds' medium tests",
        ),
    )
)


@handler("heap_height", substrates=(LAYOUT,), summary="The 95th percentile of the particles' z over the surface.")
def _heap_height(snapshot, body: str) -> dict[str, Any]:
    entity, probe = _entity_and_probe(snapshot, body)
    height = heap_height(probe.points(body), _floor_of(entity))
    return {"height": height, f"{body}/heap_height_m": height}


register_atom(
    AtomSpec(
        name="spread",
        summary="How far a particle body has spread: its footprint over the footprint it settled into.",
        args=(ArgSpec(name="body", substrates=(LAYOUT,), reads="handle"),),
        directions=("check", "observe"),
        returns=("spread",),
        cost="cheap",
        reads_scene=True,
        provenance=(
            "The probe's footprint ratio against its settled baseline, as cloth_coverage reads a sheet. "
            "1.0 is as settled; a pressed dough reads above 1 and stays there (probe_c_dough: 1.15 at the "
            "peak of a 15 mm press, 1.21 after release); a heap scooped into a tray reads whatever the "
            "tray allows."
        ),
        calibration=AtomCalibration(
            positives="A dough block pressed 15 mm by a 100 mm plate reads 1.15 under the plate and 1.21 released.",
            negatives="A block left alone reads 1.0; one tipped onto its side reads under 1.",
            gate=None,
            bench="probe_c_dough, Newton 1.6.0, RTX PRO 6000",
        ),
    )
)


@handler("spread", substrates=(LAYOUT,), summary="Footprint area over the settled footprint area.")
def _spread(snapshot, body: str) -> dict[str, Any]:
    _entity, probe = _entity_and_probe(snapshot, body)
    if probe.baseline is None:
        raise RuntimeError(f"{body!r} has no settled baseline; spread is measured against the settle")
    value = spread(probe.points(body), probe.baseline[probe.indices(body)])
    return {"spread": value, f"{body}/spread": value}


# -------------------------------------------------------------------- helpers
def _entity_and_probe(snapshot, body: str) -> tuple[Any, Any]:
    probe = snapshot.particles.get(body)
    if probe is None:
        raise KeyError(f"{body!r} is a particle body, but the scene carries no particle probe for it")
    entities = getattr(snapshot, "entities", None) or {}
    entity = entities.get(body) if hasattr(entities, "get") else None
    if entity is None:
        entity = (getattr(snapshot, "extras", None) or {}).get(body)
    return entity, probe


def _floor_of(entity: Any) -> float:
    """The world z of the surface the body stands on, read off the entity; 0 for one that cannot say."""
    floor = getattr(entity, "floor_z", None)
    return float(floor) if floor is not None else 0.0
