# SPDX-License-Identifier: Apache-2.0
"""The ``water`` kind: water that keeps its volume, on Genesis's SPH, poured by the robot's hand.

This file is what ``rss.water`` builds, imported as ``robosimstudio_materials.water``.
Run ``examples/03_materials/14_water.py`` first. The liquid kind (``liquid.py``)
pours honey on Newton's implicit MPM because that solver's water loses volume at
tabletop scale (a 57 mm column stands at 24% of its height after 2 s on Newton
1.7.0); this kind is water on a solver that holds it, Genesis's weakly compressible
SPH, which runs as a host engine beside the Newton islands
(:mod:`robosimstudio.engines.genesis.sph`). It needs the ``[genesis]`` extra, which
lives in a virtual environment of its own (``pyproject.toml``).

What a pour measured, in the bridge's probes and in the example's scene with the
YAM holding the cup (``p3_pour.py``, ``p5_nowhere.py``, ``p14_leak.py``,
``p15_cupcol.py``, ``tests/test_pour_water.py``; 4 mm particles, RTX PRO 6000,
genesis-world 1.3.3), and what each fixed here:

* **The water keeps its volume.** 1,728 particles poured into a 60 mm square glass
  stand 29.1 mm at their top particle, where their count at one spacing cubed each
  predicts 30.7 mm: the top particle's centre sits half a spacing under the
  surface. At rest in the 50 mm cup their centres span 46.7 x 46.4 x 39.9 mm, which
  with half a spacing of skin each way is 112 cm^3 for the 110.6 cm^3 their count
  fills. So :func:`fill_level` reads a height, and the count and the height agree.
* **The walls hold it.** Standing 10 s in the cup (8 mm walls) and in the glass
  (6 mm walls, a particle and a half), the water kept every particle inside, none
  under the bench. What leaves a container is poured or splashed over its rim.
* **The stiffness is a speed limit.** Weakly compressible SPH is stable only while a
  pressure wave crosses less than a particle a substep. At Genesis's default
  stiffness (5e4 Pa) 4 mm water needed 40 substeps a control step: at 30, 1,713 of
  1,728 particles went through the cup and the bench, and at 20 the resting water fell
  through the bench (``p1_rest.py``). At 1e4 Pa it holds in 20 (1,723 of 1,728 into
  the glass), and 5e3 Pa lost 618 at 16 (``p3_pour.py``). The viscosity
  has its own limit (Genesis's, ``0.01 * (2 * spacing)**2 / mu``), which is why
  :data:`WATER` states ``mu`` 5e-4: at 1e-3 it allows 0.64 ms, and a 20-substep step
  is 0.83 ms. So has the cohesion: :data:`OIL`'s ``gamma`` of 0.03 kept none of 1,728
  particles in the cup at 20 substeps and all of them at 35 (``oil_rest.py``). An
  island whose substeps are longer than its liquid holds by any of the three is
  refused by name, with the substeps it needs
  (:func:`~robosimstudio.engines.genesis.sph.longest_substep`).
* **The particle spacing is the solver's.** Genesis's SPH has one particle size per
  scene; 4 mm costs 14 ms a control step for 1,728 particles at 20 substeps. 3 mm
  rests in the cup at 40 substeps (1.001 of its weight) and costs 32 ms for 4,352
  particles; its scripted pour put all of them on the bench beside the glass, which
  was not followed up (``p3_pour.py``).

The measures are :func:`fill_level` (the share of the water in a container's inside
and the height it stands there) and the shared particle measures through the
``granular`` layout this kind reads as (``fill_fraction``, ``spill_fraction``,
``heap_height``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

import newton
import numpy as np

from robosimstudio.assetlib.record import AssetRecord, SpecFactory
from robosimstudio.assetlib.semantics.atoms import ArgSpec, AtomCalibration, AtomSpec, register_atom
from robosimstudio.assetlib.semantics.handlers import handler
from robosimstudio.assetlib.vocabulary.regions import region_bounds
from robosimstudio.core.math import quat_to_mat
from robosimstudio.core.probes.particle import ParticleProbe
from robosimstudio.domains.capabilities import ParticleMixin, host_bridge
from robosimstudio.scene.entity import Entity
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.arrangement import GENERIC, backend
from robosimstudio.specs.base import Quat, Vec3
from robosimstudio.specs.params import REBUILD, ui
from robosimstudio.specs.physics import HostActor, SimRequirements, contacts_with, register_domain

from .mpm import measures as _granular  # noqa: F401 -- mints the granular layout this kind falls back to

__all__ = [
    "OIL",
    "WATER",
    "WATER_60HZ",
    "EntityWater",
    "EntityWaterGenesis",
    "Water",
    "WaterMaterialSpec",
    "WaterSpec",
    "fill_level",
    "water",
]


# =============================================================================
# 1. The material
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class WaterMaterialSpec:
    """One liquid, in the terms Genesis's ``SPH.Liquid`` takes.

    Every field is ``REBUILD``: the engine copies the material into its particles when
    it builds, so nothing here is live on a running page.

    Attributes:
        density: Rest density [kg/m^3]. A particle weighs ``density * spacing**3``,
            the volume settled water fills (:mod:`robosimstudio.engines.genesis.sph`).
        stiffness: The Tait equation's stiffness [Pa]: how hard the water resists
            compression, and how fast a pressure wave crosses it -- which is what
            bounds the substep (the module docstring).
        exponent: The Tait equation's exponent.
        viscosity: Genesis's ``mu``, a kernel viscosity; its own stability bound is
            ``0.01 * (2 * spacing)**2 / mu`` per substep.
        surface_tension: Genesis's ``gamma``, a cohesion between neighbours. Past
            water's 0.01 it bounds the substep too (:data:`OIL`).
        color: Display colour.
    """

    density: float = field(default=1000.0, metadata=ui(tier=REBUILD, unit="kg/m^3", range=(500.0, 2000.0)))
    stiffness: float = field(default=1.0e4, metadata=ui(tier=REBUILD, unit="Pa", log=True, range=(1.0e3, 1.0e5)))
    exponent: float = field(default=7.0, metadata=ui(tier=REBUILD, range=(1.0, 7.0)))
    viscosity: float = field(default=5.0e-4, metadata=ui(tier=REBUILD, log=True, range=(1.0e-5, 1.0e-2)))
    surface_tension: float = field(default=0.01, metadata=ui(tier=REBUILD, range=(0.0, 0.1)))
    color: Vec3 = (0.36, 0.62, 0.92)

    via_model: ClassVar[tuple[str, ...]] = ("color",)
    """Drawn, not simulated: the colour reaches the page and never the engine."""

    def __post_init__(self) -> None:
        for key in ("density", "stiffness", "exponent"):
            if not float(getattr(self, key)) > 0.0:
                raise ValueError(f"{key} must be > 0, got {getattr(self, key)}")
        for key in ("viscosity", "surface_tension"):
            if float(getattr(self, key)) < 0.0:
                raise ValueError(f"{key} must be >= 0, got {getattr(self, key)}")


WATER = WaterMaterialSpec()
"""Water: 1000 kg/m^3, 1e4 Pa, ``mu`` 5e-4 -- the setting a 20-substep pour holds (the module docstring)."""

OIL = WaterMaterialSpec(density=900.0, viscosity=1.0e-3, surface_tension=0.03, color=(0.86, 0.74, 0.30))
"""A thin oil: lighter, twice water's ``mu``, three times its cohesion, and 35 substeps a 60 Hz control step at 4 mm.

Its cohesion sets them (:func:`~robosimstudio.engines.genesis.sph.longest_substep`): at
rest in the pour example's cup its 1,728 particles all left it at 20 substeps, 353 did
at 27, and at 35 all stayed and the cup read 0.976 N of their 0.976 N weight, at 37 ms
a control step against water's 26 there (``oil_rest.py``). :data:`WATER_60HZ` states 20
and refuses it by name; ``WATER_60HZ.with_backend("water", backend("genesis",
substeps=35, precision="32"))`` runs it."""

_MATERIALS = {"water": WATER, "oil": OIL}


def _material(material: Any) -> WaterMaterialSpec:
    if isinstance(material, WaterMaterialSpec):
        return material
    if isinstance(material, str) and material in _MATERIALS:
        return _MATERIALS[material]
    raise TypeError(f"material must be a WaterMaterialSpec or one of {sorted(_MATERIALS)}, got {material!r}")


# =============================================================================
# 2. The spec and the entity
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class WaterSpec:
    """A block of water, sampled on a lattice, that flows when the scene starts.

    Attributes:
        entity: The registered class that builds it, ``"water"``.
        name: Key in handles, probes, measures and particle-set names.
        size: ``(x, y, z)`` [m] of the block as built: a cup's inside, its depth.
        at: Where the **bottom-face centre** goes, in the cell's work-surface frame.
        quat: Orientation, xyzw, about the bottom-face centre.
        material: A :class:`WaterMaterialSpec`, or a preset's name (``"water"``, ``"oil"``), resolved here.
        particle_spacing: The lattice spacing [m], which is also Genesis's particle
            size: every liquid in one island states the same.
        settle_frames: Control steps run before the rest state is recorded.
        color: Display RGB, or ``None`` for the material's.
        render: ``"points"`` draws each particle as a sphere of half the spacing;
            ``"surface"`` also offers the path tracer the isosurface Newton wraps
            round the particles each frame
            (:class:`~robosimstudio.core.render.particles.ParticleSurfacer`, anisotropic
            for a liquid's flat top), as the MPM kinds' ``render="surface"`` does.
            Over the cup of water at rest it is 3,406 vertices, 3.3 ms a frame after a
            2.6 s first call (``p13_surface.py``, RTX PRO 6000). No path-traced frame of
            it has been made: the Genesis environment holds no ``ovrtx``.
        sim: What this body needs of physics, or ``None`` for the kind's default.
    """

    RENDERS: ClassVar[tuple[str, ...]] = ("points", "surface")

    sim: SimRequirements | None = None
    stage: ClassVar[str] = "deformables"
    entity: str = "water"
    name: str = "water"
    size: Vec3 = (0.048, 0.048, 0.048)
    at: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = (0.0, 0.0, 0.0, 1.0)
    material: WaterMaterialSpec = field(default_factory=lambda: WATER)
    particle_spacing: float = field(default=0.004, metadata=ui(tier=REBUILD, unit="m", range=(0.002, 0.01)))
    settle_frames: int = 30
    color: Vec3 | None = None
    render: str = "points"

    def __post_init__(self) -> None:
        object.__setattr__(self, "material", _material(self.material))
        if self.render not in self.RENDERS:
            raise ValueError(f"water {self.name!r}: render must be one of {self.RENDERS}, got {self.render!r}")
        if any(not float(v) > 0.0 for v in self.size):
            raise ValueError(f"water {self.name!r}: size must be positive, got {self.size}")
        if not float(self.particle_spacing) > 0.0:
            raise ValueError(f"water {self.name!r}: particle_spacing must be > 0 m, got {self.particle_spacing}")
        if int(self.settle_frames) < 0:
            raise ValueError(f"water {self.name!r}: settle_frames must be >= 0")

    @property
    def counts(self) -> tuple[int, int, int]:
        """Lattice points along each axis."""
        h = float(self.particle_spacing)
        return tuple(max(1, int(round(float(s) / h))) for s in self.size)  # type: ignore[return-value]

    @property
    def particle_count(self) -> int:
        nx, ny, nz = self.counts
        return nx * ny * nz

    @property
    def mass(self) -> float:
        """Total mass [kg]: the particle count times ``density * spacing**3``."""
        return float(self.particle_count) * float(self.particle_spacing) ** 3 * float(self.material.density)


def lattice(spec: WaterSpec) -> np.ndarray:
    """The particles in the spec's own frame ``[n, 3]``: one per cell at the spacing, bottom-face centre at 0.

    Regular, without jitter, because Genesis's SPH rests a liquid sampled on its own
    regular lattice: in a cup this one's top particle settles at 40.65 and 40.99 mm
    in two runs and Genesis's own sampling at 40.75 mm (``p5_nowhere.py``).
    """
    h = float(spec.particle_spacing)
    nx, ny, nz = spec.counts
    axes = [(np.arange(n) + 0.5) * h for n in (nx, ny, nz)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    grid[:, 0] -= 0.5 * nx * h
    grid[:, 1] -= 0.5 * ny * h
    return grid


class EntityWater(ParticleMixin, Entity):
    """A water body: Newton holds its particles, Genesis moves them."""

    spec: WaterSpec
    domain: ClassVar[str] = "water"
    hosted: ClassVar[bool] = False
    """Whether Genesis integrates this water, as :class:`EntityWaterGenesis` says. Only it builds one: on a Newton
    solver the inactive particles would never move, and under GENERIC the kind falls to VBD, whose settle died in an
    illegal memory access. Decided by the class the compiler picks for the arrangement, as for a bag."""

    def __init__(self, spec: WaterSpec, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.particles: np.ndarray = np.zeros(0, dtype=np.int32)
        self.sets: dict[str, np.ndarray] = {}
        self.probe: ParticleProbe | None = None
        self.rest: dict[str, np.ndarray] = {}
        self.material: WaterMaterialSpec = spec.material
        self.bridge: Any = None
        """The Genesis bridge stepping this water, found at scene-ready."""

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        """The lattice as Newton particles, flagged inactive: no Newton solver integrates them."""
        del episode
        spec = self.spec
        if not self.hosted:
            # Refused rather than built: no Newton solver moves the inactive particles.
            raise NotImplementedError(
                f"water {spec.name!r}: water is Genesis's SPH liquid; route the 'water' kind to backend('genesis'), "
                "as rss.WATER_60HZ does (it needs a GPU and the [genesis] extra)"
            )
        placed = self.placed((*spec.at, *spec.quat))
        rotation = quat_to_mat(np.asarray(placed[3:7], dtype=np.float64))
        world = lattice(spec) @ rotation.T + np.asarray(placed[:3], dtype=np.float64)
        n = len(world)
        h = float(spec.particle_spacing)
        builder.add_particles(
            pos=[tuple(float(c) for c in row) for row in world],
            vel=[(0.0, 0.0, 0.0)] * n,
            mass=[h**3 * float(self.material.density)] * n,
            radius=[0.5 * h] * n,
            flags=[0] * n,
        )

    def host_actor(self) -> HostActor:
        """Genesis's ``sph`` actor, with the material it reads."""
        material = self.material
        return HostActor(
            kind="sph",
            params={
                "spacing": float(self.spec.particle_spacing),
                "density": float(material.density),
                "stiffness": float(material.stiffness),
                "exponent": float(material.exponent),
                "viscosity": float(material.viscosity),
                "surface_tension": float(material.surface_tension),
            },
        )

    def handle(self, model: newton.Model) -> Any:
        """The particles as a point body, drawn at half the spacing, and their surface when ``render="surface"``."""
        handle = super().handle(model)
        handle.point_radius = 0.5 * float(self.spec.particle_spacing)
        if self.spec.render == "surface":
            from robosimstudio.core.render.particles import ParticleSurfacer

            handle.surface = ParticleSurfacer(handle.point_radius, device=model.device, anisotropic=True)
        return handle

    def tunables(self, pipeline: Any = None):
        """None: the engine copied the material when it built the particles, so nothing here is live."""
        del pipeline
        return ()

    def on_scene_ready(self, pipeline) -> None:
        self.bridge = host_bridge(pipeline, self.name)
        self.join_settle(pipeline, (int(self.spec.settle_frames), 0))

    def settle(self, pipeline) -> None:
        """Let the lattice come to rest. Plain steps: the engine ticks inside them."""
        for _ in range(int(self.spec.settle_frames)):
            pipeline.step()
        pipeline.sim_time = 0.0
        pipeline.step_count = 0

    def update_probe(self, state: newton.State, step: int | None = None, *, readback=None) -> ParticleProbe:
        """Read this step's particle positions, once per control step."""
        if self.probe is None:
            raise RuntimeError(f"water {self.spec.name!r} is not bound yet")
        if readback is not None and getattr(readback, "particle_q", None) is not None and len(readback.particle_q):
            self.probe.update_arrays(readback.particle_q, readback.particle_qd, step)
        else:
            self.probe.update(state, step)
        return self.probe

    def substrate_points(self, snapshot) -> np.ndarray | None:
        """This body's particles, ``[n, 3]``, from the snapshot's probe for it."""
        probe = getattr(snapshot, "particles", {}).get(self.name) if snapshot is not None else None
        if probe is None:
            return None
        return np.asarray(probe.points(self.name), dtype=np.float64)

    @property
    def particle_mass(self) -> float:
        """Mass of one particle [kg]."""
        return float(self.spec.particle_spacing) ** 3 * float(self.material.density)

    @property
    def floor_z(self) -> float:
        """The work surface this body was placed on, in world z [m]."""
        return float(self.frame[2]) + float(self.spec.at[2])


def water(
    *,
    name: str = "water",
    size: Vec3 = (0.048, 0.048, 0.048),
    at: Vec3 = (0.0, 0.0, 0.0),
    material: Any = "water",
    **kwargs,
) -> WaterSpec:
    """A block of water that flows when the scene starts.

    Needs Genesis: :data:`WATER_60HZ` routes ``water`` to it, and a build without the
    ``[genesis]`` extra raises and names it.

    Args:
        name: Key in handles and particle-set names.
        size: ``(x, y, z)`` [m] of the block: a cup's inside and its depth. The default
            is 12 particles a side at 4 mm, 1,728 particles, 110.6 g.
        at: The **bottom-face centre**, in the cell's work-surface frame; inside a cup,
            its floor.
        material: ``"water"``, ``"oil"`` or a :class:`WaterMaterialSpec`.
        **kwargs: Any other :class:`WaterSpec` field.
    """
    return WaterSpec(name=name, size=size, at=at, material=material, **kwargs)


TOUCHES = ("surface", "fixtures", "objects")
"""What water contacts: the bench, a glass's walls, the cup. Not the hands: the fingers hold the cup from outside,
and a mirror of each finger mesh would be a signed-distance field the water never meets."""


class Water:
    """Water as a task document's type: ``register_asset(Water.from_spec(rss.water(...), name="cup_of_water"))``."""

    @classmethod
    def from_spec(cls, spec: WaterSpec, *, name: str, summary: str = "") -> AssetRecord:
        """The water a constructor returned, as the :class:`AssetRecord` a document's ``(:objects ...)`` names.

        Its ``at`` is where the water starts; the document's object name replaces its ``name``.
        """
        if not isinstance(spec, WaterSpec):
            raise TypeError(f"Water.from_spec takes a WaterSpec, got {type(spec).__name__}")
        return AssetRecord(
            name=name,
            summary=summary or f"water, {spec.particle_count} particles, {1e3 * spec.mass:.0f} g",
            tags=("water",),
            sim=SimRequirements(domain="water", contacts=contacts_with(*TOUCHES)),
            build=SpecFactory(spec),
        )


# =============================================================================
# 3. The measure
# =============================================================================


def fill_level(points: np.ndarray, lo: np.ndarray, hi: np.ndarray, spacing: float) -> dict[str, float]:
    """How full a container is: the share of the water inside its box, and how high it stands there.

    Args:
        points: The water's particles ``[n, 3]``, world frame.
        lo, hi: The container's inside as a world box; ``lo[2]`` is its floor.
        spacing: The particle spacing [m].

    Returns:
        ``fraction``: the share of the particles inside the box. ``height`` [m]: the
        95th percentile of their heights over the floor, plus half a spacing, which is
        where the surface stands over its top particles' centres. ``expected`` [m]: the
        height their count fills at one spacing cubed each, over the box's floor area;
        water that kept its volume reads ``height`` close to it. The percentile, which
        a splash on top cannot lift, reads a little under the top of a deep column:
        24.0 mm where 1,352 particles standing in the glass fill 24.0 (``p14_leak.py``),
        29.4 to 29.7 mm where the poured 1,728 fill 30.5 to 30.7 (``p12_harvest.py``).
    """
    array = np.atleast_2d(np.asarray(points, dtype=np.float64))
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    inside = np.all((array >= lo) & (array <= hi), axis=1) if len(array) else np.zeros(0, bool)
    count = int(inside.sum())
    area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
    if not count:
        return {"fraction": 0.0, "height": 0.0, "expected": 0.0}
    top = float(np.quantile(array[inside, 2], 0.95)) + 0.5 * float(spacing)
    return {
        "fraction": count / len(array),
        "height": max(0.0, top - float(lo[2])),
        "expected": count * float(spacing) ** 3 / area if area > 0.0 else 0.0,
    }


register_atom(
    AtomSpec(
        name="fill_level",
        summary="How full a container is: the share of a liquid inside a region, and the height it stands there.",
        args=(
            ArgSpec(name="body", substrates=("water",), reads="handle"),
            ArgSpec(name="region", kinds=("region",), reads="region"),
        ),
        directions=("check", "observe"),
        returns=("height", "fraction"),
        unit="m",
        cost="cheap",
        reads_scene=True,
        provenance=(
            "Designed here, on a liquid that keeps its volume (Genesis's SPH): the count in the region and the "
            "95th percentile of their heights plus half a spacing, over the region's floor."
        ),
        calibration=AtomCalibration(
            positives=(
                "1,728 particles of 4 mm water poured into a 60 mm square glass by the YAM, through its arm panel, "
                "read 0.992 to 0.999 and 29.4 to 29.7 mm in three drives, against 30.5 to 30.7 mm their count "
                "fills (p12_harvest.py, GPU 1, RTX PRO 6000).",
            ),
            negatives=(
                "The same gesture with the wrist turned 58 degrees keeps all 1,728 in the cup and reads 0.0; "
                "carried 100 mm to the side it reads 0.0006, one particle, with 99.2% on the bench (p12_harvest.py).",
            ),
            gate=None,
            bench="examples/03_materials/14_water driven through its right arm panel, tests/test_pour_water.py",
            notes="No gate: a task states its own line. The height is the measure a fill line reads.",
        ),
    )
)


@handler("fill_level", kinds=("", "region"), substrates=("water",), summary="Share in the region, height there [m].")
def _fill_level(snapshot, body: str, region: dict) -> dict[str, float]:
    probe = snapshot.particles.get(body)
    if probe is None:
        raise KeyError(f"{body!r} is a water body, but the scene carries no particle probe for it")
    entities = getattr(snapshot, "entities", None) or {}
    entity = entities.get(body) if hasattr(entities, "get") else None
    spacing = float(getattr(getattr(entity, "spec", None), "particle_spacing", 0.004))
    lo, hi = region_bounds(region, (0.0, 0.0, 0.0))
    out = fill_level(probe.points(body), lo, hi, spacing)
    return {**out, f"{body}/fill_level_m": out["height"]}


# =============================================================================
# 4. Registration
# =============================================================================


WATER_60HZ = replace(
    GENERIC,
    backends=(
        ("articulated", backend("mujoco")),
        # A cup is a body of the arm's island, so a hand can hold it (as LIQUID_120HZ puts it).
        ("rigid", backend("mujoco")),
        # Twenty SPH substeps at float32 (the module docstring, and the bridge's for the precision).
        ("water", backend("genesis", substeps=20, precision="32")),
        ("*", backend("vbd", rigid_body_contact_buffer_size=256)),
    ),
)
"""GENERIC with ``water`` on Genesis's SPH and the free rigid bodies on the arm's MuJoCo island, 60 Hz control."""

DOMAIN = register_domain(
    "water",
    state=("particles",),
    reads=("water", "granular", "cloth"),
    role="deformables",
    contacts=TOUCHES,
    summary="Water that keeps its volume: particles on Genesis's SPH, a host engine beside Newton.",
    engine="genesis",
    staged_by="WATER_60HZ",
)

register_entity("water")(EntityWater)


class EntityWaterGenesis(EntityWater):
    """:class:`EntityWater` for Genesis, which integrates the water itself: Newton must leave it alone."""

    hosted = True


register_entity("water", engine="genesis")(EntityWaterGenesis)
