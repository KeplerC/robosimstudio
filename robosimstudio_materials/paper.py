# SPDX-License-Identifier: Apache-2.0
"""Paper in one file: a sheet whose folds stay, from its spec to its registration.

This file is what ``rss.paper`` builds, imported as ``robosimstudio_materials.paper``. It is cloth
(``cloth.py``) plus one thing: a bending hinge that yields. Run ``examples/03_materials/05_paper.py``
first.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import newton
import numpy as np
import warp as wp

from robosimstudio.assetlib.materials import CalibrationSpec, MaterialRecord, register_material, register_material_kind
from robosimstudio.assetlib.semantics.atoms import ArgSpec, AtomCalibration, AtomSpec, register_atom
from robosimstudio.assetlib.semantics.handlers import handler
from robosimstudio.core.pipeline import ScenePipeline
from robosimstudio.domains.selection import resolve_selector
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.base import Vec3
from robosimstudio.specs.params import LIVE, ui
from robosimstudio.specs.physics import register_domain
from robosimstudio_materials.cloth import ClothMaterialSpec, ClothSpec, EntityCloth, crease_of

__all__ = [
    "CARDBOARD",
    "PAPER",
    "PLASTIC_FIELDS",
    "EntityPaper",
    "PaperMaterialSpec",
    "PaperSpec",
    "crease_angle",
    "dihedral_angles",
    "paper",
    "papers",
    "sheet_edge_rows",
]


# =============================================================================
# 1. What this file is, and how it is found
# =============================================================================
# Importing this file registers paper. The library finds it by scanning ``robosimstudio_materials``,
# like cloth. Paper is the one kind here that imports another: a sheet of paper IS a cloth with a
# plastic hinge, so its spec, entity and measures extend ``cloth.py``'s rather than restate them.
#
# The hinge model. VBD bends a sheet with a discrete-shell hinge per interior edge: an energy
# ``k * (theta - rest)^2 / 2`` in the dihedral angle ``theta`` between the two triangles, where
# ``rest`` is ``model.edge_rest_angle``. Cloth keeps ``rest`` at the flat sheet's angle, so every
# fold springs back. Paper moves it: after every solve a kernel reads the current angle of each
# hinge of this sheet and, where it has gone further than :attr:`PaperMaterialSpec.yield_angle`
# past the rest angle, carries the rest angle toward it. That is the perfectly plastic hinge of
# the folding literature, after the bending yield of Narain, Pfaff and O'Brien ("Folding and
# Crumpling Adaptive Sheets", ACM Trans. Graph. 32(4), SIGGRAPH 2013, doi:10.1145/2461912.2462010):
# below the yield the sheet is elastic, above it the excess bend is kept. Their yield acts on each
# face's bending strain against a yield curvature [1/m], and they also soften a face's bending
# stiffness as its damage grows; here the yield is an angle [rad] on each hinge's rest angle, and
# nothing softens. Their rule keeps all of the excess (``plastic_rate`` 1) and never relaxes; a
# ``plastic_rate`` below 1 and ``plastic_recovery`` are this file's. VBD reads ``edge_rest_angle``
# from the model on every launch (``solver_vbd.py`` passes ``self.model.edge_rest_angle`` to its
# bending kernels; verified at 1.6.0 by writing the array on a built scene and watching the sheet
# move), so the write takes effect on the next substep, inside a captured graph too.


# =============================================================================
# 2. The spec: what a sheet of paper is, as data
# =============================================================================
# Exported as ``rss.PaperMaterialSpec`` and ``rss.PaperSpec``.


@dataclass(frozen=True, kw_only=True)
class PaperMaterialSpec(ClothMaterialSpec):
    """Cloth constitutive parameters plus a plastic hinge.

    Every :class:`~robosimstudio_materials.cloth.ClothMaterialSpec` field applies. The three
    below are the crease. All three are LIVE: they live in an array the entity owns and its kernel
    reads every substep, so a slider moves them without a recapture.

    Attributes:
        yield_angle: How far a hinge bends past its rest angle before the bend is kept [rad].
            ``0`` switches the plasticity off: the sheet is an ordinary elastic cloth. (Not the
            limit of a vanishing yield, which would be a hinge with no stiffness at all.)
        plastic_rate: What fraction of the excess bend is kept per substep, in ``(0, 1]``. ``1``
            is the perfectly plastic hinge: the rest angle is always exactly ``yield_angle``
            behind the bend. Lower values let a hinge resist a fast bend before it gives.
        plastic_recovery: How fast a crease relaxes back toward flat [1/s]. ``0`` is a crease
            that stays; a cardboard flap that slowly straightens wants a small value.
    """

    yield_angle: float = field(
        default=0.0,
        metadata=ui(
            tier=LIVE,
            unit="rad",
            range=(0.0, 1.5),
            group="plastic",
            doc="Bend past the rest angle by more than this and the excess is kept. 0 is elastic cloth.",
        ),
    )
    plastic_rate: float = field(
        default=1.0,
        metadata=ui(tier=LIVE, range=(0.0, 1.0), group="plastic", doc="Share of the excess bend kept per substep."),
    )
    plastic_recovery: float = field(
        default=0.0,
        metadata=ui(tier=LIVE, unit="1/s", range=(0.0, 5.0), group="plastic", doc="How fast a crease flattens again."),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.yield_angle < 0.0:
            raise ValueError(f"paper yield_angle cannot be negative, got {self.yield_angle}")
        if not 0.0 <= self.plastic_rate <= 1.0:
            raise ValueError(f"paper plastic_rate must be in [0, 1], got {self.plastic_rate}")
        if self.plastic_recovery < 0.0:
            raise ValueError(f"paper plastic_recovery cannot be negative, got {self.plastic_recovery}")


PLASTIC_FIELDS: dict[str, int] = {"yield_angle": 0, "plastic_rate": 1, "plastic_recovery": 2}
"""The three plastic fields and their slots in the entity's parameter array, which the kernel reads."""


# =============================================================================
# 3. Materials: paper and cardboard, and what each was measured against
# =============================================================================
# Section 8 registers each with its calibration.


PAPER = PaperMaterialSpec(
    shell="membrane",
    # 80 g/m^2 office paper, 0.1 mm thick. The thickness is a record; a membrane's contact
    # thickness is its particle radius, half of it, the membrane convention.
    areal_density=0.08,
    thickness=1.0e-4,
    particle_radius=5.0e-4,
    # In-plane, the fabrics' values rather than something stiffer. Paper hardly stretches, but
    # VBD's bending converges against the in-plane stiffness: on the drape bench at 5 iterations,
    # the droop that edge_ke 20 gives at 1e4 needs edge_ke 200 at 1e5 (5.0 cm either way), so a
    # tenfold in-plane stiffness bought nothing but a tenfold bending number.
    tri_ke=1.0e4,
    tri_ka=1.0e4,
    tri_kd=1.0e-2,
    # The drape. Calibrated on the drape bench at a 10.5 mm cell against copy paper: section 8.
    edge_ke=30.0,
    edge_kd=1.0e-4,
    zero_rest_angle=True,
    friction=0.3,
    contact_ke=1.0e4,
    contact_kd=1.0e-2,
    # The crease: a hinge bent 20 degrees past flat starts to keep the bend, and keeps all of it
    # at once (rate 1). Measured in tests/test_paper_crease.py: a strip folded over by a pin,
    # pressed flat and released stays folded at half its length with these, and with yield_angle 0
    # the same strip springs open flat. Rate 1 does not bake transients in: the drape bench's
    # strip, which swings from level to hanging and back on release, keeps 0.009 rad at most.
    yield_angle=0.35,
    plastic_rate=1.0,
    plastic_recovery=0.0,
)
"""Copy paper, 80 g/m^2, as a membrane whose hinges crease."""

CARDBOARD = replace(
    PAPER,
    # 500 g/m^2, about a 0.5 mm folding carton board. Shipped as a stiff membrane rather than a
    # slab: a slab's crease is in its core, which the hinge model does not reach, so a slab of this
    # thickness would fold and spring back.
    areal_density=0.5,
    thickness=5.0e-4,
    particle_radius=2.5e-4,
    # Measured on the drape bench at a 10 mm cell: section 8. Past about 500 the strip's droop is
    # set by how far VBD converges, not by this number, so the flap example raises the iterations.
    edge_ke=500.0,
    edge_kd=1.0e-3,
    # Board creases where it is scored or bent hard: a smaller yield than paper, so a flap pushed
    # over keeps the hinge, and the stiffer skin carries the rest of the flap flat.
    yield_angle=0.25,
)
"""Folding carton board, 500 g/m^2, as a stiff membrane whose hinges crease."""

papers = SimpleNamespace(PAPER=PAPER, CARDBOARD=CARDBOARD)
"""The two above under one name, which is what ``rss.papers`` resolves to, beside ``rss.materials``."""


@dataclass(frozen=True, kw_only=True)
class PaperSpec(ClothSpec):
    """A sheet of paper: a :class:`~robosimstudio_materials.cloth.ClothSpec` whose folds stay.

    Every cloth field applies, and the defaults are an A4 sheet of :data:`PAPER`. ``rss.paper``
    returns one.

    Attributes:
        pinned: The name of a particle set, from :attr:`~robosimstudio_materials.cloth.ClothSpec.particle_sets`,
            that is fixed where it is built: the bound edge of a box flap. Zero mass, which is
            Newton's kinematic particle, so the solver leaves it exactly in place. ``""`` pins nothing.
    """

    entity: str = "paper"
    name: str = "paper"
    size: tuple[float, float] = (0.21, 0.297)
    cells: tuple[int, int] = (20, 28)
    material: PaperMaterialSpec = field(default_factory=lambda: PAPER)
    color: Vec3 = (0.96, 0.96, 0.93)
    pinned: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.material, PaperMaterialSpec):
            raise ValueError(f"paper {self.name!r} needs a PaperMaterialSpec, got {type(self.material).__name__}")
        if self.pinned and self.pinned not in dict(self.particle_sets):
            raise ValueError(f"paper {self.name!r} pins {self.pinned!r}, which is not one of its particle sets")


# =============================================================================
# 4. The kernel: the hinge angle, in VBD's own convention, and the yield
# =============================================================================


@wp.func
def hinge_angle(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3) -> wp.vec2:
    """The dihedral angle of one hinge, and whether it could be read.

    The four points are an edge row's: ``x0``, ``x1`` the vertices opposite the edge, ``x2`` to
    ``x3`` the edge. The normals, the edge direction and the ``atan2`` are written as Newton's VBD
    bending kernel writes them (``particle_vbd_kernels.py``,
    ``evaluate_dihedral_angle_based_bending_force_hessian``), so the angle this returns is the one
    whose difference from ``edge_rest_angle`` the solver penalises, sign included. The builder's
    ``add_edge`` computes the rest angle the same way. Returns ``(angle, 1)``, or ``(0, 0)`` for a
    hinge too degenerate to have one, which is the case the solver also skips.
    """
    n1 = wp.cross(x2 - x0, x3 - x0)
    n2 = wp.cross(x3 - x1, x2 - x1)
    e = x3 - x2
    n1_norm = wp.length(n1)
    n2_norm = wp.length(n2)
    e_norm = wp.length(e)
    if n1_norm < 1.0e-6 or n2_norm < 1.0e-6 or e_norm < 1.0e-6:
        return wp.vec2(0.0, 0.0)
    n1_hat = n1 / n1_norm
    n2_hat = n2 / n2_norm
    e_hat = e / e_norm
    sin_theta = wp.dot(wp.cross(n1_hat, n2_hat), e_hat)
    cos_theta = wp.dot(n1_hat, n2_hat)
    return wp.vec2(wp.atan2(sin_theta, cos_theta), 1.0)


@wp.kernel
def _crease_kernel(
    rows: wp.array(dtype=wp.int32),
    edge_indices: wp.array2d(dtype=wp.int32),
    params: wp.array(dtype=float),
    dt: float,
    particle_q: wp.array(dtype=wp.vec3),
    edge_rest_angle: wp.array(dtype=float),
):
    """Carry each hinge's rest angle toward its bend, past the yield. One thread per hinge.

    Launched unconditionally every substep, so the block stays capturable; ``params[0] <= 0`` is
    the off switch and returns before any read. The excess is wrapped into ``(-pi, pi]`` before
    the yield test: a sheet folded flat onto itself sits at a dihedral of about ``+-pi``, where
    contact noise flips the sign, and an unwrapped difference there would read as a bend the other
    way and write a mirrored crease.
    """
    i = wp.tid()
    yield_angle = params[0]
    if yield_angle <= 0.0:
        return
    e = rows[i]
    i0 = edge_indices[e, 0]
    i1 = edge_indices[e, 1]
    if i0 < 0 or i1 < 0:
        return
    angle = hinge_angle(particle_q[i0], particle_q[i1], particle_q[edge_indices[e, 2]], particle_q[edge_indices[e, 3]])
    if angle[1] == 0.0:
        return
    rest = edge_rest_angle[e]
    excess = angle[0] - rest
    if excess > wp.pi:
        excess = excess - 2.0 * wp.pi
    elif excess < -wp.pi:
        excess = excess + 2.0 * wp.pi
    if wp.abs(excess) > yield_angle:
        rest = rest + params[1] * (excess - wp.sign(excess) * yield_angle)
    rest = rest - params[2] * dt * rest
    edge_rest_angle[e] = rest


@wp.kernel
def _angles_kernel(
    rows: wp.array(dtype=wp.int32),
    edge_indices: wp.array2d(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    angles: wp.array(dtype=float),
):
    """Read the hinge angle of each row, for a test or a readout. A boundary edge reads 0."""
    i = wp.tid()
    e = rows[i]
    i0 = edge_indices[e, 0]
    i1 = edge_indices[e, 1]
    if i0 < 0 or i1 < 0:
        angles[i] = 0.0
        return
    angle = hinge_angle(particle_q[i0], particle_q[i1], particle_q[edge_indices[e, 2]], particle_q[edge_indices[e, 3]])
    angles[i] = angle[0]


def dihedral_angles(particle_q: Any, edge_indices: Any, rows: Any = None) -> np.ndarray:
    """The hinge angle of each edge row, read through :func:`hinge_angle`.

    Args:
        particle_q: Positions, a warp array of ``vec3`` on any device.
        edge_indices: Edge rows ``[n, 4]`` as ``(opposite0, opposite1, v0, v1)``, a warp array.
        rows: Which rows to read, or ``None`` for all of them.
    """
    device = particle_q.device
    if rows is None:
        rows = np.arange(edge_indices.shape[0], dtype=np.int32)
    rows_wp = wp.array(np.asarray(rows, dtype=np.int32), dtype=wp.int32, device=device)
    out = wp.zeros(len(rows_wp), dtype=wp.float32, device=device)
    wp.launch(
        _angles_kernel, dim=len(rows_wp), inputs=[rows_wp, edge_indices, particle_q], outputs=[out], device=device
    )
    return out.numpy().astype(np.float64)


def sheet_edge_rows(model: Any, particles: tuple[int, int], *, num_worlds: int = 1, stride: int = 0) -> np.ndarray:
    """The edge rows belonging to one sheet, in every world.

    ``particles`` is the template's range; a replicated model holds a copy of it in every world at
    the particle ``stride``. Column 2 of an edge row is a vertex on the edge, so it is always a
    real particle, and an edge never spans two entities.
    """
    edges = getattr(model, "edge_indices", None)
    if edges is None or not model.edge_count:
        return np.zeros(0, dtype=np.int32)
    on_edge = edges.numpy().reshape(-1, 4)[:, 2]
    lo, hi = particles
    mine = np.zeros(len(on_edge), dtype=bool)
    for world in range(max(1, int(num_worlds))):
        mine |= (on_edge >= lo + world * stride) & (on_edge < hi + world * stride)
    return np.flatnonzero(mine).astype(np.int32)


# =============================================================================
# 5. The entity: cloth, plus the hinge kernel and the crease's reset
# =============================================================================


class EntityPaper(EntityCloth):
    """One sheet of paper: a cloth whose hinges yield.

    Everything about building, settling, holding and reading the sheet is :class:`EntityCloth`'s.
    This adds a ``post_solve`` kernel over the sheet's hinges (section 4), the three plastic
    parameters in a device array the kernel reads, and the crease's place in a reset: the rest
    angles are snapshotted with the settled particles and restored with them, because a crease is
    episode state exactly as a fold is.
    """

    spec: PaperSpec
    domain = "paper"
    world_aware = True
    """The kernel runs over this sheet's hinges in every world at once, so a replicated build is fine."""

    def __init__(self, spec: PaperSpec, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.rows: np.ndarray = np.zeros(0, dtype=np.int32)
        """This sheet's edge rows in the stepping model, every world, world 0 first."""
        self.rest_angles: np.ndarray | None = None
        """The rest angle of each of :attr:`rows` as the scene became ready: what a reset restores."""
        self.pinned: np.ndarray = np.zeros(0, dtype=np.int32)
        """Global indices of the particles :attr:`PaperSpec.pinned` fixed, every world."""
        self._rows: Any = None
        self._params: Any = None
        self._rest: Any = None

    # -------------------------------------------------------------- phase 2
    def bind(self, model: newton.Model) -> None:
        super().bind(model)
        stepping = self.stepping_model if self.stepping_model is not None else model
        count = int(getattr(self.worlds, "num_worlds", 1) or 1)
        stride = int(getattr(self.worlds, "particles_per_world", 0) or 0)
        self.rows = sheet_edge_rows(stepping, self.range.particles, num_worlds=count, stride=stride)
        device = stepping.device
        self._rows = wp.array(self.rows, dtype=wp.int32, device=device)
        self._params = wp.zeros(3, dtype=wp.float32, device=device)
        self.write_plastic(self.spec.material)
        # The array the kernel writes. The pipeline model's, unless on_scene_ready finds the
        # sheet's solver reading a copy, in which case it retargets to that copy.
        self._rest = stepping.edge_rest_angle
        if self.spec.pinned:
            rest = model.particle_q.numpy().astype(np.float64)[self.particles] - np.asarray(self.frame[:3])
            local = resolve_selector(dict(self.spec.particle_sets)[self.spec.pinned], rest, self.particles)
            if not len(local):
                raise ValueError(f"paper {self.spec.name!r}: its pinned set {self.spec.pinned!r} selects no particle")
            base = np.asarray(local, dtype=np.int64)
            self.pinned = np.concatenate([base + world * stride for world in range(count)]).astype(np.int32)

    def apply_model_materials(self, model: newton.Model) -> None:
        """Cloth's contact scalars, and no mass on the pinned particles, before a solver copies either."""
        super().apply_model_materials(model)
        if len(self.pinned):
            for array in (model.particle_mass, model.particle_inv_mass):
                values = array.numpy()
                values[self.pinned] = 0.0
                array.assign(values)

    def write_plastic(self, material: PaperMaterialSpec) -> None:
        """Put a material's three plastic fields where the kernel reads them. LIVE: no recapture."""
        self._params.assign(
            np.array([material.yield_angle, material.plastic_rate, material.plastic_recovery], dtype=np.float32)
        )

    def tunables(self, pipeline=None):
        """Cloth's material tunable, with the three plastic fields routed to the kernel's array.

        ``LiveMaterial.write_field`` answers ``False`` for a name it does not know, which a panel
        shows as "nothing to write to"; the plastic fields are known here instead.
        """
        base = super().tunables(pipeline)
        if not base:
            return base
        (material,) = base
        inner = material.write

        def write(name: str, value: Any) -> bool:
            slot = PLASTIC_FIELDS.get(name)
            if slot is None:
                return inner(name, value)
            values = self._params.numpy()
            values[slot] = float(value)
            self._params.assign(values)
            return True

        return (replace(material, write=write),)

    # -------------------------------------------------------------- phase 3
    def contribute_hooks(self, hooks) -> None:
        hooks.add("post_solve", self._crease, name=f"crease:{self.spec.name}")

    def _crease(self, state: newton.State, dt: float) -> None:
        model = self.stepping_model if self.stepping_model is not None else self.model
        wp.launch(
            _crease_kernel,
            dim=len(self.rows),
            inputs=[self._rows, model.edge_indices, self._params, float(dt), state.particle_q],
            outputs=[self._rest],
            device=model.device,
        )

    # -------------------------------------------------------------- phase 4
    def on_scene_ready(self, pipeline: ScenePipeline) -> None:
        """Point the kernel at the array the solver reads, then settle and snapshot like cloth."""
        self._rest = self.solver_rest_angles(pipeline)
        super().on_scene_ready(pipeline)

    def solver_rest_angles(self, pipeline: Any) -> Any:
        """The ``edge_rest_angle`` array this sheet's solver bends against.

        Not always the pipeline model's. Measured at Newton 1.6.0: the one-way coupler hands VBD
        the model itself, so the pointers agree; Newton's coupled solver (``GENERIC``'s ``proxy``)
        hands it a ``ModelView`` holding its own copy of the edge arrays, and VBD reads the copy on
        every launch. A crease written to the model's array would then bend nothing, so the kernel
        and the reset address whichever array the solver holds. The copy keeps the model's rows
        when the sheet's island owns every particle, which is what a VBD island does; a compacted
        copy is refused rather than written to by the wrong rows.
        """
        model = self.stepping_model if self.stepping_model is not None else self.model
        mine = model.edge_rest_angle
        program = getattr(pipeline, "program", None)
        solver_for = getattr(program, "solver_for", None)
        try:
            solver = solver_for(self.spec.name) if callable(solver_for) else None
        except (KeyError, LookupError):
            solver = None
        theirs = getattr(getattr(solver, "model", None), "edge_rest_angle", None)
        if theirs is None or theirs.ptr == mine.ptr:
            return mine
        if tuple(theirs.shape) != tuple(mine.shape):
            raise RuntimeError(
                f"paper {self.spec.name!r}: its solver reads a compacted edge_rest_angle of shape {theirs.shape}, "
                f"not the model's {mine.shape}; the crease kernel addresses hinges by the model's rows"
            )
        return theirs

    def snapshot_rest(self, pipeline: Any) -> None:
        """The settled particles, and the rest angles that go with them."""
        super().snapshot_rest(pipeline)
        if len(self.rows):
            self.rest_angles = self._rest.numpy()[self.rows].copy()

    # -------------------------------------------------------------- phase 5
    def reset(self, pipeline: ScenePipeline, episode, *, world: int = 0) -> None:
        """Flatten the creases of the last episode along with its folds."""
        super().reset(pipeline, episode, world=world)
        if self.rest_angles is not None and len(self.rows):
            values = self._rest.numpy()
            values[self.rows] = self.rest_angles
            self._rest.assign(values)

    # ---------------------------------------------------------------- runtime
    def crease_rows(self, world: int = 0) -> np.ndarray:
        """This sheet's edge rows in one world."""
        count = int(getattr(self.worlds, "num_worlds", 1) or 1)
        per_world = len(self.rows) // max(1, count)
        return self.rows[world * per_world : (world + 1) * per_world]

    def crease_angles(self, world: int = 0) -> np.ndarray:
        """The rest angle of each hinge of this sheet, now, in one world [rad]."""
        rows = self.crease_rows(world)
        return self._rest.numpy().astype(np.float64)[rows] if len(rows) else np.zeros(0)

    def hinge_midpoints(self, world: int = 0) -> np.ndarray:
        """Where each hinge of this sheet lay when it settled, in the work-surface frame ``[n, 3]``."""
        rows = self.crease_rows(world)
        if not len(rows) or not self.rest:
            return np.zeros((0, 3))
        model = self.stepping_model if self.stepping_model is not None else self.model
        edges = model.edge_indices.numpy().reshape(-1, 4)[rows]
        q = np.asarray(self.rest["particle_q"], dtype=np.float64)
        return 0.5 * (q[edges[:, 2]] + q[edges[:, 3]]) - np.asarray(self.frame[:3])

    def hinge_directions(self, world: int = 0) -> np.ndarray:
        """Each hinge's edge vector as the sheet settled ``[n, 3]``, in the work-surface frame's axes."""
        rows = self.crease_rows(world)
        if not len(rows) or not self.rest:
            return np.zeros((0, 3))
        model = self.stepping_model if self.stepping_model is not None else self.model
        edges = model.edge_indices.numpy().reshape(-1, 4)[rows]
        q = np.asarray(self.rest["particle_q"], dtype=np.float64)
        return q[edges[:, 3]] - q[edges[:, 2]]

    def cell_size(self) -> float:
        """The sheet's hinge spacing [m]: the median rest length of its edges.

        The median, not the mean: a grid's diagonals are 1.41 cells long and a third of its edges,
        so the mean of a 10 mm grid reads 11.2 mm where the median reads the cell.
        """
        rows = self.crease_rows(0)
        if not len(rows):
            return 0.0
        model = self.stepping_model if self.stepping_model is not None else self.model
        return float(np.median(model.edge_rest_length.numpy()[rows]))


# =============================================================================
# 6. The constructor: ``paper``
# =============================================================================


def paper(
    *,
    name: str = "paper",
    size: tuple[float, float] = (0.21, 0.297),
    cells: tuple[int, int] = (20, 28),
    at: Vec3 = (0.0, 0.0, 0.0),
    material: Any = None,
    **kwargs,
) -> PaperSpec:
    """A rectangular sheet of paper, generated as a grid. ``rss.sheet`` for a sheet that creases.

    Args:
        name: Key in handles and particle-set names.
        size: Metres, ``(x, y)``. The default is A4.
        cells: Quads along each axis. The default is a 10.5 mm cell, the one :data:`PAPER` was
            calibrated at; a hinge's rigidity scales with the cell, so another resolution is
            another sheet.
        at: Where the sheet's origin **corner** goes, in the cell's work-surface frame.
        material: A :class:`PaperMaterialSpec` such as one of ``rss.papers``; ``None`` is :data:`PAPER`.
        **kwargs: Any other :class:`PaperSpec` field: ``pinned`` with a particle set fixes an edge.
    """
    extra = {} if material is None else {"material": material}
    return PaperSpec(name=name, size=size, cells=cells, pose=at, **extra, **kwargs)


# =============================================================================
# 7. Measures: the crease
# =============================================================================
# Every cloth atom applies to paper through the substrate chain ``("paper", "cloth")``: fold_extent,
# fold_score, held, coverage. This one is paper's own.


register_atom(
    AtomSpec(
        name="crease_angle",
        summary="How far a sheet stays turned through a crease line [rad], from the hinge angles it has kept.",
        args=(
            ArgSpec(name="paper", substrates=("paper",), reads="handle"),
            ArgSpec(name="panel", kinds=("panel",), reads="feature", optional=True),
        ),
        directions=("check", "observe"),
        returns=("angle", "peak", "edges"),
        unit="rad",
        cost="cheap",
        reads_scene=True,
        provenance=(
            "Designed here. Reads the sheet's own rest angles, which the hinge kernel writes, so it "
            "measures what the sheet will keep rather than how it happens to be bent now."
        ),
        calibration=AtomCalibration(
            positives=(
                "A 0.2 x 0.05 m strip of PAPER folded over by a pin, pressed flat and released reads "
                "3.0-3.2 rad through its fold line two seconds after release, its sharpest hinge at "
                "1.4-1.8 rad, and lies at 0.51 of its length (tests/test_paper_crease.py, yield 0.35, "
                "rate 1, 10 mm cells)."
            ),
            negatives=(
                "The same strip with yield_angle 0 reads exactly 0, as does the sheet before the fold: "
                "a flat sheet settling on a bench writes no crease. The strip carried over and laid "
                "down but not pressed, a loop 30 mm high sharing its turn over three hinge columns, "
                "reads 1.5 and springs open to its full length when released."
            ),
            gate=2.2,
            bench="a strip of PAPER folded over by a pin, pressed and released, in tests/test_paper_crease.py",
            notes=(
                "With a line, the angle is the sheet's turning angle through it: the kept angles of the "
                "hinges that run along the line within one cell of it, summed across the line and "
                "averaged along it, so a sharp fold reads close to pi less the yield however many hinges "
                "share it. The gate sits between the laid loop's 1.5, which does not hold, and the "
                "pressed fold's 3.0, which does. Without a line it is the mean kept angle over every "
                "hinge of the sheet, a crumple measure, with no gate of its own."
            ),
        ),
    )
)


def _creases(snapshot, paper: str, panel: dict[str, Any] | None) -> dict[str, Any]:
    entity = snapshot.entities.get(paper)
    if entity is None or not len(entity.rows):
        return {"angle": 0.0, "peak": 0.0, "edges": 0.0, f"{paper}/crease_angle": 0.0}
    kept = np.abs(entity.crease_angles())
    if panel is None:
        mean = float(kept.mean())
        return {"angle": mean, "peak": float(kept.max()), "edges": float(len(kept)), f"{paper}/crease_angle": mean}
    axis, origin = crease_of(panel)
    cell = entity.cell_size()
    mid = entity.hinge_midpoints()
    direction = entity.hinge_directions()
    # The hinges a fold about this line bends are the ones that run along it: an edge across
    # the line turns with the sheet and keeps no angle of its own.
    along = np.abs(direction[:, axis]) <= 0.5 * np.linalg.norm(direction, axis=1)
    near = along & (np.abs(mid[:, axis] - origin) <= cell + 1.0e-6)
    if not near.any():
        return {"angle": 0.0, "peak": 0.0, "edges": 0.0, f"{paper}/crease_angle": 0.0}
    # The axis the line runs along is the one its hinges run along: ``1 - axis`` for a sheet lying
    # on the bench, ``x`` for a flap standing up whose line is stated in height (``axis`` 2).
    other = int(np.argmax(np.abs(direction[near]).mean(axis=0)))
    span = float(mid[near][:, other].max() - mid[near][:, other].min()) + cell
    rows = max(1.0, span / cell) if cell > 0.0 else 1.0
    angle = float(kept[near].sum() / rows)
    return {"angle": angle, "peak": float(kept[near].max()), "edges": float(near.sum()), f"{paper}/crease_angle": angle}


@handler(
    "crease_angle",
    kinds=("", "panel"),
    substrates=("paper", ""),
    summary="The sheet's turning angle through the panel's crease line, from the hinges kept along it.",
)
def crease_angle(snapshot, paper: str, panel: dict[str, Any] | None = None) -> dict[str, Any]:
    """The crease the panel declares, read on the hinges along it; the sheet's mean kept angle without one."""
    return _creases(snapshot, paper, panel)


@handler("crease_angle", substrates=("paper",), summary="The mean kept hinge angle over the whole sheet.")
def _crease_anywhere(snapshot, paper: str) -> dict[str, Any]:
    return _creases(snapshot, paper, None)


# =============================================================================
# 8. Registration
# =============================================================================


register_material_kind("paper", PaperMaterialSpec)

_DRAPE = "the cloth drape bench (ASTM D1388, clamped strip)"

register_material(
    MaterialRecord(
        name="paper",
        spec=PAPER,
        notes="Copy paper, 80 g/m^2, as a membrane whose hinges crease. An A4 sheet to fold.",
        calibration=CalibrationSpec(
            fabric="copy paper (80 gsm)",
            bench=_DRAPE,
            knob="edge_ke",
            cell=0.0105,
            bending_length_cm=5.96,
            target_cm=(5.0, 7.0),
            sweep=((20.0, 5.33), (30.0, 5.96), (40.0, 6.41), (50.0, 6.75), (60.0, 7.01)),
            notes=(
                "Measured at a 10.5 mm cell, the A4 example's, under GENERIC's VBD (16 substeps, "
                "5 iterations): a 12 x 6 cell strip 300 mm up in a void, its first three columns "
                "held at zero mass, a 105 mm overhang, the droop read at the tip row's middle once "
                "four half-second readings agreed to 0.01 (20-30 s of settling: the strip swings "
                "for a long time at edge_kd 1e-4). With the hinge kernel on at yield 0.35 the same "
                "strip reads 5.95 cm and its largest kept angle is 0.009 rad: the swing does not "
                "crease it. At tri_ke 1e5 the same droop needs edge_ke 200; see the material."
            ),
        ),
    )
)

register_material(
    MaterialRecord(
        name="cardboard",
        spec=CARDBOARD,
        notes="Folding carton board, 500 g/m^2, as a stiff membrane whose hinges crease. A box flap.",
        calibration=CalibrationSpec(
            fabric="folding carton board",
            bench=_DRAPE,
            knob="edge_ke",
            cell=0.010,
            bending_length_cm=13.6,
            target_cm=None,
            sweep=((200.0, 8.24), (500.0, 8.54), (1000.0, 8.8), (2000.0, 8.93)),
            notes=(
                "Not a row of the reference table. The sweep is at GENERIC's 5 VBD iterations, where "
                "the droop saturates near 19% of a 100 mm overhang whatever the stiffness: the solve, "
                "not the board, sets it. At 20 iterations the same strip at 500 droops 5.0% (13.6 cm; "
                "2000 gives 4.5%, 14.1 cm), which is the setting the box-flap example stages. "
                "Measured on paper's bench at a 10 mm cell, tri_ke 1e4."
            ),
        ),
    )
)


DOMAIN = register_domain(
    "paper",
    state=("particles",),
    reads=("paper", "cloth"),
    role="deformables",
    contacts=("surface", "fixtures", "objects", "hands", "deformables"),
    summary="A membrane of particles whose bending hinges yield: paper and card.",
    engine="vbd",
    staged_by="GENERIC",
)

register_entity("paper")(EntityPaper)


# Import-only; running it directly would register paper twice.
