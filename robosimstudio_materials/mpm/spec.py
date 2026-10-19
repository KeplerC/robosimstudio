# SPDX-License-Identifier: Apache-2.0
"""What a particle body *is*, as data: the matter and the lump of it.

Two frozen specs shared by the three particle kinds (``granular``, ``liquid``,
``dough``), which differ in their presets and their domain and in nothing a
builder reads. :class:`ParticleMaterialSpec` is Newton's implicit-MPM material
field for field, in SI and under Newton's own names, so a number read in the
solver's source is the number here. :class:`ParticleBodySpec` is a region of
space filled with that material: a box, a cylinder or a mesh's interior,
sampled on a lattice.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

from robosimstudio.core.render.finish import Finish
from robosimstudio.specs.asset import AssetSpec
from robosimstudio.specs.base import Quat, Vec3
from robosimstudio.specs.params import REBUILD, RECAPTURE, ui
from robosimstudio.specs.physics import SimRequirements

__all__ = [
    "MPM_KEYS",
    "NEWTON_MPM_STRESS_SCALE",
    "ParticleBodySpec",
    "ParticleMaterialSpec",
]

NEWTON_MPM_STRESS_SCALE = 0.5
"""What a stress-valued preset is multiplied by before it reaches the solver.

0.5 for Newton 1.7. Its implicit MPM fixed the stress normalisation (newton
#4277: the transpose of the symmetric-part operator in
``rheology_solver_kernels.py`` lost a stray factor of one half), so a given
stress now pushes the grid twice as hard, and the 1.7 changelog says to halve
``young_modulus``, ``yield_pressure``, ``yield_stress`` and ``viscosity`` for a
scene tuned on 1.6. The presets below were measured on Newton 1.6.0, where the
scale was 1.0; this keeps each reading as the pascals it names and behaving as
it did there."""

MPM_KEYS: tuple[str, ...] = (
    "young_modulus",
    "poisson_ratio",
    "damping",
    "hardening",
    "friction",
    "yield_pressure",
    "tensile_yield_ratio",
    "yield_stress",
    "hardening_rate",
    "softening_rate",
    "dilatancy",
    "viscosity",
)
"""The per-particle material attributes Newton's implicit MPM registers, in the
order ``SolverImplicitMPM.register_custom_attributes`` declares them. Each is
``model.mpm.<name>``, one float per particle."""

_STRESS_KEYS = ("young_modulus", "yield_pressure", "yield_stress", "viscosity")
"""The attributes in pascals (and pascal-seconds), which :data:`NEWTON_MPM_STRESS_SCALE` applies to."""


@dataclass(frozen=True, kw_only=True)
class ParticleMaterialSpec:
    """One particle material: Newton's implicit-MPM fields, plus a density.

    The defaults are Newton's own registration defaults -- a dry, rigid-grained,
    cohesionless granular -- with a density of 1500 kg/m^3. A liquid is this with
    ``friction=0`` and ``tensile_yield_ratio=1``; a viscous liquid adds
    ``viscosity``; a plastic solid lowers ``young_modulus`` to something finite
    and sets a ``yield_stress``. The presets in
    :mod:`robosimstudio_materials.mpm.presets` are those four shapes, named.

    Every field but :attr:`density` is ``RECAPTURE``: it is a per-particle array
    on the model (``model.mpm.<name>``) the solver re-reads when told the model's
    properties changed, and the notification is host work a captured graph does
    not replay, so a write lands on the next capture. Density becomes particle
    mass at build and cannot move afterwards.

    Attributes:
        density: Bulk density [kg/m^3]. A particle's mass is its lattice cell's
            volume times this (:attr:`ParticleBodySpec.particle_volume`).
        young_modulus: Elastic modulus [Pa]. 1e15 is Newton's "rigid grains":
            a granular's stiffness is the grid's, not the grains'. 5e6 reads as
            a soft solid that can be pressed (dough, clay).
        poisson_ratio: Poisson ratio of the elastic part.
        damping: Elastic damping relaxation time [s]. Zero for grains; 1e-3
            stops a soft solid ringing after a press.
        hardening: Hardening factor of the plastic flow (snow uses 10).
        friction: Internal friction coefficient. 0.68 is Newton's rice, 0 a
            liquid.
        yield_pressure: Pressure above which the material compacts [Pa].
            1e12-1e15 is incompressible for a tabletop.
        tensile_yield_ratio: Cohesion as a fraction of the yield pressure: 0
            for dry grains, which fly apart freely; 1 for anything that holds
            together under tension (a liquid, a dough).
        yield_stress: Deviatoric yield stress [Pa]. 0 flows freely; a finite
            value is a Bingham threshold -- 3e2 for ketchup, 1e3 for dough.
        hardening_rate: Rate of plastic hardening.
        softening_rate: Rate of plastic softening.
        dilatancy: Dilatancy of the plastic flow.
        viscosity: Viscosity of the plastic flow [Pa*s]. 50 is honey-like in
            Newton's own example.
    """

    density: float = field(
        default=1500.0,
        metadata=ui(tier=REBUILD, unit="kg/m^3", log=True, range=(100.0, 3000.0), group="mass"),
    )
    young_modulus: float = field(
        default=1.0e15,
        metadata=ui(tier=RECAPTURE, unit="Pa", log=True, range=(1.0e4, 1.0e15), group="elasticity"),
    )
    poisson_ratio: float = field(
        default=0.3,
        metadata=ui(tier=RECAPTURE, range=(0.0, 0.49), group="elasticity"),
    )
    damping: float = field(
        default=0.0,
        metadata=ui(tier=RECAPTURE, unit="s", range=(0.0, 0.1), group="elasticity"),
    )
    hardening: float = field(
        default=0.0,
        metadata=ui(tier=RECAPTURE, range=(0.0, 20.0), group="plasticity"),
    )
    friction: float = field(
        default=0.5,
        metadata=ui(tier=RECAPTURE, range=(0.0, 1.5), step=0.01, group="flow"),
    )
    yield_pressure: float = field(
        default=1.0e15,
        metadata=ui(tier=RECAPTURE, unit="Pa", log=True, range=(1.0e2, 1.0e15), group="plasticity"),
    )
    tensile_yield_ratio: float = field(
        default=0.0,
        metadata=ui(tier=RECAPTURE, range=(0.0, 1.0), step=0.01, group="flow"),
    )
    yield_stress: float = field(
        default=0.0,
        metadata=ui(tier=RECAPTURE, unit="Pa", range=(0.0, 2.0e4), group="flow"),
    )
    hardening_rate: float = field(
        default=1.0,
        metadata=ui(tier=RECAPTURE, range=(0.0, 10.0), group="plasticity"),
    )
    softening_rate: float = field(
        default=1.0,
        metadata=ui(tier=RECAPTURE, range=(0.0, 10.0), group="plasticity"),
    )
    dilatancy: float = field(
        default=0.0,
        metadata=ui(tier=RECAPTURE, range=(0.0, 1.0), group="plasticity"),
    )
    viscosity: float = field(
        default=0.0,
        metadata=ui(tier=RECAPTURE, unit="Pa*s", range=(0.0, 500.0), group="flow"),
    )
    color: Vec3 = (0.86, 0.80, 0.62)
    """Display colour, for a body that does not state its own."""

    MPM_KEYS: ClassVar[tuple[str, ...]] = MPM_KEYS

    def __post_init__(self) -> None:
        if not self.density > 0.0:
            raise ValueError(f"density must be > 0 kg/m^3, got {self.density}")
        if not 0.0 <= self.poisson_ratio < 0.5:
            raise ValueError(f"poisson_ratio must be in [0, 0.5), got {self.poisson_ratio}")
        for key in ("young_modulus", "yield_pressure"):
            if not getattr(self, key) > 0.0:
                raise ValueError(f"{key} must be > 0 Pa, got {getattr(self, key)}")
        for key in (
            "damping",
            "friction",
            "yield_stress",
            "viscosity",
            "hardening",
            "hardening_rate",
            "softening_rate",
        ):
            if getattr(self, key) < 0.0:
                raise ValueError(f"{key} must be >= 0, got {getattr(self, key)}")
        if not 0.0 <= self.tensile_yield_ratio <= 1.0:
            raise ValueError(f"tensile_yield_ratio must be in [0, 1], got {self.tensile_yield_ratio}")

    def custom_attributes(self) -> dict[str, float]:
        """The material as the builder takes it: ``{"mpm:<name>": value}`` per key, stress scaled."""
        out: dict[str, float] = {}
        for key in MPM_KEYS:
            value = float(getattr(self, key))
            if key in _STRESS_KEYS:
                value *= NEWTON_MPM_STRESS_SCALE
            out[f"mpm:{key}"] = value
        return out

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Every described field, density first then :data:`MPM_KEYS`."""
        return tuple(f.name for f in fields(cls) if f.name != "color")


@dataclass(frozen=True, kw_only=True)
class ParticleBodySpec:
    """A region of space filled with one particle material.

    Attributes:
        name: Key in handles, probes, measures and particle-set names.
        entity: Which registered class builds it -- ``"heap"``, ``"liquid"`` or
            ``"dough"``, which is what decides the kind (and so the backend) when
            :attr:`sim` says nothing. The constructors set it.
        shape: ``"box"``, ``"cylinder"`` or ``"mesh"``.
        size: Extents [m]. A box's ``(x, y, z)``; a cylinder's ``(diameter,
            diameter, height)`` about its own z; ignored for a mesh, whose
            :class:`~robosimstudio.specs.AssetSpec` carries its scale.
        asset: The mesh whose interior is filled, for ``shape="mesh"``.
            Resolved like every other asset path; its triangles must enclose
            a volume (``trimesh.contains`` decides).
        at: Where the **bottom-face centre** goes [m], in the cell's
            work-surface frame, as ``rss.sponge`` places a sponge. For a mesh,
            the bottom of its bound.
        quat: Orientation, xyzw, about the bottom-face centre.
        material: The matter, a :class:`ParticleMaterialSpec`.
        particle_spacing: Lattice spacing [m]. 2.5 mm is 1.6 samples per axis
            of the backend's registered 4 mm voxel, which is what the
            probes ran (their 29k-particle rice bed); Newton's own examples put
            2-3 per axis. Fewer than 1.5 per voxel leaves cells with no sample
            and the surface tears; more than 4 costs memory for no change in
            motion, since the step is the colliders' cost, not the particles'.
        jitter: Random offset per particle as a fraction of the spacing, so the
            lattice does not alias with the grid. Seeded from :attr:`name`.
        particle_sets: ``(name, selector)`` pairs over the rest positions, in
            the work-surface frame -- the selectors in
            :mod:`robosimstudio.domains.selection`.
        settle_frames: Control steps run before the rest state is recorded.
        settle_substeps: Substeps per settle frame, or 0 for the run's own. The
            run's own is the right choice here: the MPM steps once per control
            step either way, and a finer settle only runs the arm's solver at
            a coarser rate than it was tuned for.
        render: ``"points"`` draws each particle as a sphere of its radius,
            half its lattice cell's cube root; ``"surface"`` also offers a
            reconstructed surface mesh to a renderer that can take one per
            frame (the path tracer), through ``DeformableHandle.surface``.
        surface_voxel: Voxel of the surface reconstruction [m], or ``None``
            for 0.45 of the solver's.
        containers: Names of rigid or static entities this body is poured
            into or rests in. Each of their collision shapes gets a signed
            distance ``margin`` of :attr:`container_margin`, which is what
            keeps a liquid inside a thin wall under graph capture: measured in
            the cup probe (``probe_c_water``), a 3 mm wall at 4 mm
            voxels lost 60% of the water through its walls in 2 s at rest and
            5% with a 1.5 mm margin. A tool that pushes a heap along the
            bench names itself here too, at one voxel: the grid cannot resolve
            the sliver between a tool's underside and the bench, so the bottom
            voxel of grains lags the tool and passes under it. A 160 mm pusher
            standing on the bench, swept 290 mm through a 16,000-particle heap
            of rice at 0.15 m/s (``probe_sweep``, GPU 3, RTX PRO 6000, Newton
            1.6.0), left 40% of the rice behind itself with no margin and 10%
            with a 4 mm one. Under the particle arrangements the rigid
            contacts are MuJoCo's, which zeroes an authored geom margin under
            its native CCD and says so once at build, so the hand and the
            bench still meet the tool's true faces; Newton's own collision
            pipeline would offset them by the margin.
        container_margin: The margin [m], or ``None`` for 2 mm, half the
            registered voxel.
        color: Display RGB in [0, 1], or ``None`` for the material's.
        opacity: Display alpha in (0, 1].
        finish: Finish under the path-traced backend, or ``None`` for the
            finish keyed to the colour.
        sim: What this body needs of physics, or ``None`` for the kind's
            default.
    """

    sim: SimRequirements | None = None
    stage: ClassVar[str] = "deformables"
    entity: str = "heap"
    name: str = "heap"
    shape: str = "box"
    size: Vec3 = (0.10, 0.10, 0.03)
    asset: AssetSpec | None = None
    at: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = (0.0, 0.0, 0.0, 1.0)
    material: ParticleMaterialSpec = field(default_factory=ParticleMaterialSpec)
    particle_spacing: float = field(
        default=0.0025, metadata=ui(tier=REBUILD, unit="m", log=True, range=(0.001, 0.01), group="sampling")
    )
    jitter: float = field(default=0.2, metadata=ui(tier=REBUILD, range=(0.0, 0.5), group="sampling"))
    particle_sets: tuple[tuple[str, Callable[[Any], Any]], ...] = ()
    settle_frames: int = 30
    settle_substeps: int = 0
    render: str = "points"
    surface_voxel: float | None = None
    containers: tuple[str, ...] = ()
    container_margin: float | None = None
    color: Vec3 | None = None
    opacity: float = 1.0
    finish: Finish | None = None

    SHAPES: ClassVar[tuple[str, ...]] = ("box", "cylinder", "mesh")
    RENDERS: ClassVar[tuple[str, ...]] = ("points", "surface")

    def __post_init__(self) -> None:
        where = f"particle body {self.name!r}"
        if self.shape not in self.SHAPES:
            raise ValueError(f"{where}: shape must be one of {self.SHAPES}, got {self.shape!r}")
        if self.render not in self.RENDERS:
            raise ValueError(f"{where}: render must be one of {self.RENDERS}, got {self.render!r}")
        if self.shape == "mesh" and self.asset is None:
            raise ValueError(f"{where}: shape 'mesh' needs an asset to fill")
        if self.shape != "mesh" and any(not float(v) > 0.0 for v in self.size):
            raise ValueError(f"{where}: size must be positive, got {self.size}")
        if not float(self.particle_spacing) > 0.0:
            raise ValueError(f"{where}: particle_spacing must be > 0 m, got {self.particle_spacing}")
        if not 0.0 <= float(self.jitter) <= 0.5:
            raise ValueError(f"{where}: jitter is a fraction of the spacing in [0, 0.5], got {self.jitter}")
        if not 0.0 < self.opacity <= 1.0:
            raise ValueError(f"{where}: opacity must be in (0, 1], got {self.opacity}")
        if int(self.settle_frames) < 0 or int(self.settle_substeps) < 0:
            raise ValueError(f"{where}: settle_frames and settle_substeps must be >= 0")
        if self.container_margin is not None and not float(self.container_margin) > 0.0:
            raise ValueError(f"{where}: container_margin must be > 0 m, got {self.container_margin}")

    @property
    def particle_count(self) -> int | None:
        """How many particles a box or cylinder allocates, or ``None`` for a mesh.

        A mesh keeps the lattice points ``trimesh.contains`` puts inside it,
        which only the fill knows: the built entity's ``len(particles)``.
        :attr:`size` is ignored for a mesh, so a count off it would be a guess.
        """
        if self.shape == "mesh":
            return None
        h = float(self.particle_spacing)
        nx, ny, nz = (max(1, int(round(float(s) / h))) for s in self.size)
        if self.shape == "cylinder":
            # The lattice points inside the inscribed ellipse, to within a ring.
            return int(round(nx * ny * nz * 3.141592653589793 / 4.0))
        return nx * ny * nz

    @property
    def particle_volume(self) -> float:
        """The volume one particle of a box or cylinder stands for [m^3]: its lattice cell's.

        The lattice puts one sample per cell of ``size / round(size / spacing)``
        so that it ends on the faces, and a cell is a spacing cubed only where
        the spacing divides the size: a 6 mm puddle at 2.5 mm is two layers of
        3 mm. A particle's mass is this volume times the density, and its
        radius is half its cube root, which is what Newton's implicit MPM
        reads back as the particle's volume (``8 r^3``), so the body weighs
        what its material does over the region it fills. A mesh's cell is its
        bound's, known once it is filled; this is the spacing cubed for one.
        """
        h = float(self.particle_spacing)
        if self.shape == "mesh":
            return h**3
        volume = 1.0
        for extent in self.size:
            volume *= float(extent) / max(1, int(round(float(extent) / h)))
        return volume

    @property
    def mass(self) -> float | None:
        """Total mass [kg]: the particle count times the particle mass, or ``None`` for a mesh, as its count is."""
        count = self.particle_count
        if count is None:
            return None
        return float(count) * self.particle_volume * float(self.material.density)
