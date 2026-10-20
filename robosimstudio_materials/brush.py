# SPDX-License-Identifier: Apache-2.0
"""The ``brush`` kind: a handle over a field of short stiff bristles, each a capsule chain on rod joints.

This file is what ``rss.brush`` builds, imported as ``robosimstudio_materials.brush``.
Run ``examples/03_materials/07_brush.py`` first.

A brush is a sole of ``rows`` strips under a grip, and ``rows x columns``
bristles rooted in the strips' undersides, pointing down: each bristle a few
overlapping capsules chained by Newton's rod joints -- built by the rod kind's
own :func:`~robosimstudio_materials.rod.add_capsule_chain`, solved by the same
rigid VBD -- and welded at its root to its strip. The kind reads as
``("brush", "rod")``. What makes it a brush rather than a bundle of cables is
the grip the field hangs from and the one thing a brush is for, which this kind
measures: where its tips have been while down on the surface (:func:`scrubbed`).

**Two grips, because a hand can hold only one of them.** Left alone, the brush
builds its own: a capsule lying on the sole, on VBD with the bristles, which
then hold the grip up as a field does. The YAM cannot pinch that grip: a body
on VBD that stands firm, squeezed by fingers on MuJoCo, is a stiff wall behind
a one-step-lagged proxy. Under ``GENERIC`` as shipped, the hand closing with
its tool centre 10 mm over the grip's axis leaves a 12 x 12 field's state
non-finite; closing 2.5 mm higher, the state stays finite but the jaws shut on
nothing and the grip falls off the bench. A 2 x 2 field's grip slips out of the
jaws, in some runs as they close and in others as the hand rises
(``grasp_own.py``). An earlier draft tried thirty-odd coupling variants on the
12 x 12 field, all with sticky contact matching and rigid contact history on
(proxy mass scale 1 to 1000, relaxed and Aitken feedback, lagged mode, ramped
closing, contact stiffness 1e3 to 1e4 and damping to 5e3, softer fields), and
in every one the jaws were thrown open or the state blew up (``diag_grasp``).
Named instead (``rss.brush(handle="handle")``), any body already in the scene
is the grip, and an ``rss.box`` under :data:`BRUSH_60HZ` is on the arm's
solver, where the same hand holds it at 22.3 mm. Read every tick, the handle
slips at most 0.12 mm in the fingers while the hand presses 6 mm, lifts 20 mm
and strokes 150 mm; it peaks at 0.17 mm in the ticks after the stroke stops and
eases to 0.14 mm over the next 40 (``brush_probe.py``).

**What a named handle costs.** The welds from it to the strips drag the field
along one way: Newton 1.6's VBD returns *contact* forces to the other solver
and never joint forces (``SolverVBD.coupling_harvest_proxy_wrenches``). So a
handle on MuJoCo is built gravity-compensated -- left without it, the handle
of ``03_materials/07_brush`` falls 28.9 mm through its own bristles to the bench
within 1 s (``nogravcomp.py``) -- it stays where a hand lets go of it, and the
arm feels neither the press nor the bristles' drag: a press goes as deep as the
hand goes. Pressed 6 mm, 400 nylon bristles bow up to 4.8 mm with their tips
sliding up to 1.4 mm, none goes into the bench or up past the sole, and 1 s
after the hand lifts the brush 20 mm off the bench every tip is within 0.036 mm
of its rest place on its strip; soft ones bow 9 mm, their tips slide 11 mm, and
they come back within 0.18 mm (``press.py``). Every measurement here is on an
RTX PRO 6000, GPU 3, Newton 1.6.0, graph-captured, beside the YAM on
``bench_075``, two runs each, on :data:`~robosimstudio.specs.GENERIC` and
:data:`BRUSH_60HZ` as they ship (no sticky contact matching, no rigid contact
history) unless it says otherwise.

**Contact.** Every shape of one brush shares a collision group of its own
(Newton's negative groups collide with everything but their own number), so no
bristle meets another or the sole; bristle against bench, fixture and object
stays. The bristles and the sole do not meet the robot either -- the hand meets
the brush at the grip -- and a named handle does not meet its own bristles:
with the scene's 100 mm contact gap each would be a contact pair to every
capsule within reach. A 5 x 5 field on its own grip whose bristles and sole met
the robot overflowed VBD's 256-row per-body contact buffer in every substep it
was stepped, 630 to 1091 rows, the hand 80 to 100 mm over the bench, and not
once with them kept apart (``overflow.py``).

**Budget.** Every capsule is a body the coupled VBD iterates, and VBD solves a
body's joints serially in one thread, so the bristles are welded to strips, a
row each, rather than to one sole: without the robot, 144 bristles on one strip
(a 1 x 144 field) cost 34.0 ms a control step, and 12 x 12 on twelve strips
12.0 ms. Beside the YAM, 4 segments, VBD at 5 iterations and 1 substep, on its
own grip under ``GENERIC``: 5 x 5 is 14.0 ms, 8 x 8 16.0, 12 x 12 18.1,
16 x 16 20.7, 20 x 20 22.5 and 22 x 22 23.6; on a named handle under
:data:`BRUSH_60HZ`, 20 x 20 is 22.4, 22 x 22 23.4 and 24 x 24 24.6
(``cost.py``). 20 x 20 is the default, 2.5 ms under 25 ms on its own grip and
2.6 ms on a named handle. 22 x 22 fits too, by 1.4 and 1.6 ms; it is not the
default because every press, slip and scrub number here, in the examples and
in their tests was measured on 400 bristles. At 20 x 20 on its own grip, 10
iterations cost 37.8 ms, 20 cost 68.5 and 2 substeps 38.5, so none of them is
the default; at 5 iterations and 1 substep the grip stands within 0.09 mm of
where it was built and every tip within 1.2 mm.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

import newton
import numpy as np
import warp as wp

from robosimstudio.assetlib.materials import MaterialRecord, get_material, register_material, register_material_kind
from robosimstudio.assetlib.record import AssetRecord, SpecFactory
from robosimstudio.assetlib.semantics.atoms import ArgSpec, AtomCalibration, AtomSpec, register_atom
from robosimstudio.assetlib.semantics.handlers import handler
from robosimstudio.assetlib.semantics.states import StateSpec
from robosimstudio.assetlib.sim import SimRequirements, contacts_with
from robosimstudio.assetlib.vocabulary.regions import region_bounds
from robosimstudio.core.render.palette import DEFORMABLE
from robosimstudio.scene.entity import Entity
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.base import Vec3
from robosimstudio.specs.params import REBUILD, RECAPTURE, ui
from robosimstudio.specs.physics import GENERIC, register_domain

# The rod kind minted the ``rod`` layout this one reads as, so it is imported
# before the chain below names it; the builder helper is the one a cable is
# built with.
from robosimstudio_materials.rod import add_capsule_chain, rod_slot_dofs

__all__ = [
    "BRUSH_60HZ",
    "INERTIA_FLOOR",
    "NYLON",
    "PROXY_MASS_SCALE",
    "SCRUB_BAND",
    "SCRUB_CELL",
    "SCRUB_REACH",
    "SOFT",
    "Brush",
    "BrushMaterialSpec",
    "BrushSpec",
    "EntityBrush",
    "ScrubGrid",
    "brush",
    "presets",
    "resolve_material",
    "scrubbed",
]


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class BrushMaterialSpec:
    """What a bristle is made of: its chain's stiffness, its matter, its grip on the bench.

    The stiffness pairs are a view of ``joint_target_ke``/``kd`` over the
    bristle joints' slots, as a cable's drive gains are (shear takes the stretch
    pair, twist the bend pair). RECAPTURE: the field solves inside the coupled
    solver's view and the write reaches it through ``notify_model_changed``,
    host work a captured graph does not replay.

    Attributes:
        bend_ke: Per-joint bend stiffness [N*m/rad]: whether the field holds a
            handle up and how far a press bows it.
        bend_kd: Bend damping [N*m*s/rad].
        stretch_ke: Per-joint stretch stiffness [N/m].
        stretch_kd: Stretch damping [N*s/m].
        friction: The bristles' friction coefficient; Newton combines it with
            the surface's as ``sqrt(mu_a * mu_b)``.
        density: Bristle matter [kg/m^3]. Built into the capsule masses, so a
            fresh build only.
    """

    bend_ke: float = field(
        default=0.01, metadata=ui(tier=RECAPTURE, unit="N*m", range=(1e-5, 1.0), log=True, group="bend")
    )
    bend_kd: float = field(
        default=2.0e-4, metadata=ui(tier=RECAPTURE, unit="N*m*s", range=(1e-7, 1e-1), log=True, group="bend")
    )
    stretch_ke: float = field(
        default=1.0e4, metadata=ui(tier=RECAPTURE, unit="N/m", range=(1e2, 1e6), log=True, group="stretch")
    )
    stretch_kd: float = field(
        default=0.1, metadata=ui(tier=RECAPTURE, unit="N*s/m", range=(1e-3, 1e2), log=True, group="stretch")
    )
    friction: float = field(default=0.6, metadata=ui(tier=RECAPTURE, range=(0.0, 3.0), group="contact"))
    density: float = field(default=1150.0, metadata=ui(tier=REBUILD, unit="kg/m^3", range=(100.0, 5000.0)))

    def drive_gains(self) -> dict[str, float]:
        """The eight numbers a bristle's joints are built with, in the rod kind's spelling."""
        return {
            "stretch_ke": float(self.stretch_ke),
            "stretch_kd": float(self.stretch_kd),
            "shear_ke": float(self.stretch_ke),
            "shear_kd": float(self.stretch_kd),
            "bend_ke": float(self.bend_ke),
            "bend_kd": float(self.bend_kd),
            "twist_ke": float(self.bend_ke),
            "twist_kd": float(self.bend_kd),
        }


register_material_kind("brush", BrushMaterialSpec)

NYLON = register_material(
    MaterialRecord(
        name="nylon_bristle",
        spec=BrushMaterialSpec(),
        notes=(
            "A dish brush's nylon bristle: 400 of them hold the brush's own grip up within 0.09 mm, bow up to "
            "4.8 mm under a 6 mm press and spring back within 0.036 mm (cost.py, press.py)."
        ),
    )
)
SOFT = register_material(
    MaterialRecord(
        name="soft_bristle",
        spec=BrushMaterialSpec(bend_ke=1.0e-3, bend_kd=2.0e-5, friction=0.4, density=1300.0),
        notes=(
            "A paint brush's bristle, ten times easier to bend than nylon: a field of 144 cannot hold the "
            "brush's own grip up and settles 12.6 mm under it; under a named handle 400 bow 9 mm in a 6 mm "
            "press (cost.py, press.py)."
        ),
    )
)

presets: dict[str, MaterialRecord] = {"nylon": NYLON, "soft": SOFT}
"""The bristle materials by their short names, as ``rss.brush(material=...)`` takes them."""


def resolve_material(material: Any) -> BrushMaterialSpec:
    """A bristle material as a spec, from a spec, a :class:`MaterialRecord`, a preset name or a registered one."""
    if isinstance(material, BrushMaterialSpec):
        return material
    if isinstance(material, MaterialRecord):
        material = material.spec
    elif isinstance(material, str):
        record = presets.get(material) or get_material(material)
        material = record.spec
    else:
        raise TypeError(
            f"material must be a BrushMaterialSpec, a MaterialRecord or a name, got {type(material).__name__}"
        )
    if not isinstance(material, BrushMaterialSpec):
        raise TypeError(f"material is a {type(material).__name__}, not a brush material (nylon, soft)")
    return material


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class BrushSpec:
    """A brush: the sole, the bristle field, the grip and where it stands.

    Sizes are in metres, ``(x, y, z)`` for the sole. The bristles stand on a
    ``rows x columns`` lattice under the sole (rows along y, columns along x),
    inset half a pitch from its edge, pointing down. :attr:`at` is where the
    **bristle tips' footprint centre** stands, in the cell's work-surface
    frame, so ``at=(x, y, 0)`` stands the brush on its bristles on the bench;
    the sole's underside is one bristle length and one bristle radius above it.
    """

    stage: ClassVar[str] = "deformables"
    """Built in the deformables pass, after the objects, so a named handle already stands."""
    name: str = "brush"
    entity: str = "brush"
    sole: Vec3 = (0.08, 0.05, 0.008)
    """The plate the bristles are rooted in [m], cut into one strip per row."""
    rows: int = 20
    """Bristles along y, and sole strips."""
    columns: int = 20
    """Bristles along x, on each strip."""
    bristle_length: float = 0.02
    bristle_radius: float = 0.00075
    bristle_segments: int = 4
    """Capsules per bristle, at least 2. Their spacing (``length / segments``) must stay above ``2 * radius``."""
    flare: float = 0.15
    """How far the outermost bristles lean out from the sole's middle at rest [rad]; the middle ones stand straight.

    A dish brush's outer tufts flare, and this is that shape, not a press law:
    400 nylon bristles pressed 4 and 6 mm through the hand's IK bow 3.7 and
    4.7 mm at this flare and 3.7 and 4.6 mm at none (``press_ik``, GPU 3)."""
    material: BrushMaterialSpec = field(default_factory=BrushMaterialSpec)
    at: Vec3 = (0.0, 0.0, 0.0)
    color: tuple[float, float, float] = DEFORMABLE
    """The bristles' colour; the sole and the brush's own grip take :attr:`handle_color`."""
    handle: str = ""
    """The name of a body already in the scene to root the field in, or empty for the brush's own grip.

    The named body is welded to the strips where it stands, so it is placed on
    the sole by whoever adds it; the brush builds no grip then."""
    handle_length: float = 0.08
    """The brush's own grip, end to end along x [m]: a capsule lying on the sole."""
    handle_radius: float = 0.011
    """The brush's own grip's radius [m]: 22 mm across the jaws."""
    handle_color: tuple[float, float, float] = (0.15, 0.15, 0.17)
    handle_density: float = 700.0
    """The sole's and the own grip's matter [kg/m^3]: a plastic, 42 g at the default sizes."""
    handle_friction: float = 1.0
    contact_ke: float = 1.0e5
    """Each bristle capsule's contact stiffness [N/m], ten times a cable's.

    At a cable's 1e4, a 4 mm press through the hand's IK sank 400 nylon tips
    0.93 mm into the bench; at 3e4 their rounded ends went 0.45 mm in; at 1e5
    they stay on it, the lowest tip's axis end 0.78 mm over the top (``press_ik``,
    GPU 3), at the same cost a control step."""
    contact_kd: float = 0.1
    contact_gap: float = 0.002
    """Contact detection gap round each bristle capsule [m]."""
    hand_contact: bool = False
    """Whether the bristles and the sole collide with the robot. The grip always does."""
    sim: SimRequirements | None = None

    def __post_init__(self) -> None:
        name = self.name
        if len(self.sole) != 3 or any(float(v) <= 0.0 for v in self.sole):
            raise ValueError(f"brush {name!r}: the sole needs three positive extents, got {self.sole}")
        if int(self.rows) < 1 or int(self.columns) < 1:
            raise ValueError(f"brush {name!r}: rows and columns are at least 1, got {self.rows} x {self.columns}")
        if int(self.bristle_segments) < 2:
            raise ValueError(
                f"brush {name!r}: a bristle is a chain of at least 2 segments, got {self.bristle_segments}"
            )
        if float(self.bristle_radius) <= 0.0 or float(self.bristle_length) <= 0.0:
            raise ValueError(f"brush {name!r}: bristle length and radius are positive")
        if not 0.0 <= float(self.flare) < 0.5 * math.pi:
            raise ValueError(f"brush {name!r}: flare is an angle from the vertical in [0, pi/2), got {self.flare}")
        spacing = float(self.bristle_length) / int(self.bristle_segments)
        if spacing <= 2.0 * float(self.bristle_radius):
            raise ValueError(
                f"brush {name!r}: segment spacing {spacing * 1e3:.2f} mm is not above twice the radius "
                f"({2e3 * float(self.bristle_radius):.2f} mm); fewer segments or thinner bristles"
            )
        if not self.handle:
            radius, length = float(self.handle_radius), float(self.handle_length)
            if radius <= 0.0 or length < 2.0 * radius:
                raise ValueError(
                    f"brush {name!r}: the grip is a capsule, so its length {length} is at least twice its radius {radius}"
                )
            if float(self.sole[0]) < length or float(self.sole[1]) < 2.0 * radius:
                raise ValueError(
                    f"brush {name!r}: the sole {tuple(self.sole[:2])} must be at least the grip's footprint "
                    f"({length}, {2.0 * radius})"
                )
        if not isinstance(self.material, BrushMaterialSpec):
            raise TypeError(f"brush {name!r}: material is a {type(self.material).__name__}, not a BrushMaterialSpec")

    @property
    def bristle_count(self) -> int:
        return int(self.rows) * int(self.columns)

    @property
    def body_count(self) -> int:
        """The bodies this brush builds: every capsule, a strip per row, and its own grip unless one is named."""
        return self.bristle_count * int(self.bristle_segments) + int(self.rows) + (0 if self.handle else 1)

    @property
    def pitch(self) -> tuple[float, float]:
        """Bristle spacing [m], ``(along x, along y)``."""
        return float(self.sole[0]) / int(self.columns), float(self.sole[1]) / int(self.rows)

    def leans(self) -> np.ndarray:
        """``(count,)`` each bristle's lean from the vertical [rad]: :attr:`flare` scaled by how far out its root is."""
        roots = self.roots()
        reach = np.linalg.norm(roots / (0.5 * np.asarray(self.sole[:2], dtype=np.float64)), axis=1)
        return float(self.flare) * np.clip(reach / max(float(reach.max()), 1e-12), 0.0, 1.0)

    def roots(self) -> np.ndarray:
        """``(count, 2)`` bristle roots under the sole, in its frame [m], row-major (a row is one y)."""
        px, py = self.pitch
        xs = (np.arange(int(self.columns)) + 0.5) * px - 0.5 * float(self.sole[0])
        ys = (np.arange(int(self.rows)) + 0.5) * py - 0.5 * float(self.sole[1])
        gx, gy = np.meshgrid(xs, ys)
        return np.stack([gx.ravel(), gy.ravel()], axis=1)

    @property
    def sole_height(self) -> float:
        """The sole's underside over the surface the tips stand on, as built [m]: the straight middle bristles' reach.

        The leaning ones stop short of the surface by ``length * (1 - cos(lean))``,
        ``20 mm x (1 - cos 0.15)``, 0.22 mm, for the outermost at the default
        flare, and reach it under the brush's weight."""
        return float(self.bristle_length) + float(self.bristle_radius)

    @property
    def sole_top(self) -> float:
        """The sole's top over the surface, where a named handle stands [m]."""
        return self.sole_height + float(self.sole[2])

    @property
    def grip_height(self) -> float:
        """The brush's own grip's axis over the surface, as built [m]."""
        return self.sole_top + float(self.handle_radius)


# ---------------------------------------------------------------------------
# The scrub measure
# ---------------------------------------------------------------------------

SCRUB_CELL = 0.005
"""The coverage lattice's cell [m]: a tip that was down inside a 5 mm square has scrubbed it."""

SCRUB_BAND = 0.002
"""How far over the surface a tip's rounded end may be and still count as down [m]."""

SCRUB_REACH = 0.30
"""Half the side of the coverage lattice [m], centred where the brush was built: a patch
further than this from the brush's start is outside what it records."""


@dataclass
class ScrubGrid:
    """Where the tips have been while down: a lattice of :data:`SCRUB_CELL` squares over the surface.

    One per world. The entity's kernel marks the device copy every substep; this
    host object is how it is read, and how the measure is tested without a scene.

    Attributes:
        origin: The lattice's lower corner ``(x, y)`` in world [m].
        cell: Cell side [m].
        nx, ny: Cells along x and y.
        cells: ``(ny, nx)`` of 0 or 1.
    """

    origin: tuple[float, float]
    cell: float
    nx: int
    ny: int
    cells: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int32))

    def __post_init__(self) -> None:
        if not self.cells.size:
            self.cells = np.zeros((int(self.ny), int(self.nx)), dtype=np.int32)

    @classmethod
    def around(cls, centre_xy, *, reach: float = SCRUB_REACH, cell: float = SCRUB_CELL) -> ScrubGrid:
        """A lattice ``2 * reach`` a side centred on ``centre_xy``."""
        n = int(math.ceil(2.0 * float(reach) / float(cell)))
        x0 = float(centre_xy[0]) - 0.5 * n * float(cell)
        y0 = float(centre_xy[1]) - 0.5 * n * float(cell)
        return cls(origin=(x0, y0), cell=float(cell), nx=n, ny=n)

    def mark(self, xy) -> None:
        """Mark the cells under these ``(N, 2)`` world points; points off the lattice are ignored."""
        pts = np.atleast_2d(np.asarray(xy, dtype=np.float64))[:, :2]
        if not len(pts):
            return
        ix = np.floor((pts[:, 0] - self.origin[0]) / self.cell).astype(np.int64)
        iy = np.floor((pts[:, 1] - self.origin[1]) / self.cell).astype(np.int64)
        keep = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        self.cells[iy[keep], ix[keep]] = 1

    def clear(self) -> None:
        self.cells[:] = 0

    def _patch(self, lo, hi) -> tuple[slice, slice]:
        """The rows and columns whose cell **centres** lie inside the patch ``[lo, hi]`` (x, y)."""
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        cx = self.origin[0] + (np.arange(self.nx) + 0.5) * self.cell
        cy = self.origin[1] + (np.arange(self.ny) + 0.5) * self.cell
        xs = np.flatnonzero((cx >= lo[0]) & (cx <= hi[0]))
        ys = np.flatnonzero((cy >= lo[1]) & (cy <= hi[1]))
        if not len(xs) or not len(ys):
            return slice(0, 0), slice(0, 0)
        return slice(int(ys[0]), int(ys[-1]) + 1), slice(int(xs[0]), int(xs[-1]) + 1)

    def patch_cells(self, lo, hi) -> int:
        """How many cells the patch ``[lo, hi]`` covers."""
        rows, cols = self._patch(lo, hi)
        return int(self.cells[rows, cols].size)

    def fraction(self, lo, hi) -> float:
        """The share of the patch's cells a tip has scrubbed; 0 for a patch off the lattice."""
        rows, cols = self._patch(lo, hi)
        block = self.cells[rows, cols]
        return float(block.mean()) if block.size else 0.0


def scrubbed(tips_down, lo, hi, *, cell: float = SCRUB_CELL) -> float:
    """The share of the patch ``[lo, hi]`` (x, y, world [m]) whose ``cell`` squares a down tip has visited.

    ``tips_down`` is every ``(x, y)`` a bristle tip occupied while down on the
    surface over the whole scrub so far -- a history, which is what coverage
    is. Cells are anchored at ``lo``, so a 100 mm patch at 5 mm is 20 x 20 of
    them, and a point on a cell's far edge belongs to the next cell.
    """
    lo = np.asarray(lo, dtype=np.float64)[:2]
    hi = np.asarray(hi, dtype=np.float64)[:2]
    n = np.maximum(np.round((hi - lo) / float(cell)).astype(np.int64), 0)
    if not n.all():
        return 0.0
    grid = ScrubGrid(origin=(float(lo[0]), float(lo[1])), cell=float(cell), nx=int(n[0]), ny=int(n[1]))
    grid.mark(tips_down)
    return float(grid.cells.mean())


@wp.kernel
def _mark_tips(
    body_q: wp.array(dtype=wp.transform),
    tips: wp.array(dtype=wp.int32),
    half: float,
    ceiling: float,
    x0: float,
    y0: float,
    cell: float,
    nx: int,
    ny: int,
    per_world: int,
    grid: wp.array(dtype=wp.int32),
):
    """Mark the lattice cell under every tip whose axis end is at or under ``ceiling``.

    A tip is the far ``+z`` end of its bristle's last capsule: ``add_rod``
    points each capsule's local ``+z`` at the next node, and a bristle is built
    root to tip. One launch covers every world, ``per_world`` tips to a world
    and one lattice slab of ``nx * ny`` cells each.
    """
    t = wp.tid()
    p = wp.transform_point(body_q[tips[t]], wp.vec3(0.0, 0.0, half))
    if p[2] > ceiling:
        return
    ix = int(wp.floor((p[0] - x0) / cell))
    iy = int(wp.floor((p[1] - y0) / cell))
    if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
        return
    w = t // per_world
    grid[w * nx * ny + iy * nx + ix] = 1


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

INERTIA_FLOOR = 1.5e-10
"""The least principal moment a bristle capsule is built with [kg*m^2].

Newton 1.6's ``finalize`` treats a principal moment under 1e-10 as degenerate
and adds 1e-6 to all three, and a bristle capsule is far under it: 5 mm of
0.75 mm-radius nylon is 12.2 mg, with moments of 3.9e-11 across and 3.3e-12
along, as ``add_rod`` builds it. Left to that correction its moment across
grows 26,000 times and the capsule all but stops turning: a sideways push at
its end meets 12.2 mg, the whole capsule, where its own moments give 4.1 mg. At
this floor the end meets 8.1 mg. Each is ``1 / (1/m + r^2/I)`` with ``r`` the
2.5 mm from the capsule's centre to its end (``probe_brush_masses``, GPU 5)."""


def _floor_inertia(builder: newton.ModelBuilder, bodies) -> None:
    """Raise each body's principal moments to :data:`INERTIA_FLOOR` in the builder, keeping its axes."""
    for b in bodies:
        tensor = np.asarray(builder.body_inertia[b], dtype=np.float64).reshape(3, 3)
        values, axes = np.linalg.eigh(0.5 * (tensor + tensor.T))
        floored = axes @ np.diag(np.maximum(values, INERTIA_FLOOR)) @ axes.T
        builder.body_inertia[b] = wp.mat33(*floored.ravel().tolist())
        builder.body_inv_inertia[b] = wp.mat33(*np.linalg.inv(floored).ravel().tolist())


def _free_coord(builder: newton.ModelBuilder, body: int) -> int:
    """The first coordinate of the free joint ``body`` hangs on, or -1 for a body on none (a fixed handle)."""
    free = int(newton.JointType.FREE)
    for j in range(builder.joint_count):
        if int(builder.joint_child[j]) == int(body) and int(builder.joint_type[j]) == free:
            return int(builder.joint_q_start[j])
    return -1


def _z_axes(quats) -> np.ndarray:
    """``(N, 3)`` local +z axes in world of ``(N, 4)`` xyzw quaternions, normalising as ``quat_to_mat`` does.

    The third column of :func:`~robosimstudio.core.math.quat_to_mat` over the
    whole array at once, with its identity for a quaternion of zero length.
    """
    q = np.asarray(quats, dtype=np.float64).reshape(-1, 4)
    x, y, z, w = q.T
    n = np.einsum("ij,ij->i", q, q)
    s = np.divide(2.0, n, out=np.zeros_like(n), where=n >= 1.0e-12)
    return np.stack([s * (x * z + y * w), s * (y * z - x * w), 1.0 - s * (x * x + y * y)], axis=1)


def _carried(poses: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """``(N, 7)`` poses moved rigidly with a body that went from pose ``start`` to pose ``end``."""
    from robosimstudio.core.math import mat_to_quat, quat_multiply, quat_to_mat

    poses, start, end = (np.asarray(v, dtype=np.float64) for v in (poses, start, end))
    turn = quat_to_mat(end[3:7]) @ quat_to_mat(start[3:7]).T
    q = mat_to_quat(turn)
    # The turn's quaternion times each pose's, as a matrix on the right-hand factor: the
    # product is linear in it, so the library's product with each basis quaternion is a column.
    left = np.stack([quat_multiply(q, basis) for basis in np.eye(4)], axis=1)
    out = np.empty_like(poses)
    out[:, :3] = (poses[:, :3] - start[:3]) @ turn.T + end[:3]
    out[:, 3:7] = poses[:, 3:7] @ left.T
    return out


_GROUPS = itertools.count(2)
"""Collision groups handed to brushes, one each, negated: ``-2``, ``-3``, ... (``-1`` is Newton's
"collide with everything", ``0`` nothing, ``1`` what every other shape carries)."""


class EntityBrush(Entity):
    """The sole strips, the bristles and their welds, a grip unless one is named, and the coverage lattice."""

    spec: BrushSpec
    domain: ClassVar[str] = "brush"
    world_aware: ClassVar[bool] = True
    """The coverage kernel runs on the stepping model over every world's tips, each world
    keeps a lattice slab of its own, and a reset restores one world's bodies."""

    def __init__(self, spec: BrushSpec, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.handle_body: int = -1
        """The grip: the brush's own, or the named handle's body."""
        self.coord_start: int = -1
        """The own grip's free-joint coordinates; -1 for a named handle, which its own entity restores."""
        self._handle_coord: int = -1
        """A named handle's free-joint coordinates, where its reset writes the episode's pose; -1 otherwise."""
        self.bodies: np.ndarray = np.zeros(0, dtype=np.int32)
        self.shapes: np.ndarray = np.zeros(0, dtype=np.int32)
        self.joints: np.ndarray = np.zeros(0, dtype=np.int32)
        self.strip_bodies: np.ndarray = np.zeros(0, dtype=np.int32)
        """One sole strip per row of bristles, welded under the grip."""
        self.strip_coords: list[int] = []
        """Under a named handle, each strip's free-joint coordinates (:meth:`build`); empty under its own grip."""
        self.bristle_bodies: np.ndarray = np.zeros((0, 0), dtype=np.int32)
        """``(count, segments)`` capsule bodies, root to tip, template-indexed."""
        self.handle_shapes: np.ndarray = np.zeros(0, dtype=np.int32)
        """The own grip's capsule and the strips' boxes: the plastic, as against the bristles."""
        self.half_height: float = 0.0
        """Half of one capsule's chord [m]; the tip is this far along the last capsule's ``+z``."""
        self.group: int = -next(_GROUPS)
        self._spans: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = ((0, 0), (0, 0), (0, 0))
        self._rest_body_q: np.ndarray | None = None
        self._rest_joint_q: np.ndarray | None = None
        self._grid_device: wp.array | None = None
        self._tips_device: wp.array | None = None
        self._grid: ScrubGrid | None = None

    # -------------------------------------------------------------- phase 1
    @property
    def _num_worlds(self) -> int:
        return int(getattr(self.worlds, "num_worlds", 1) or 1)

    @property
    def tips(self) -> np.ndarray:
        """The last capsule of each bristle, template-indexed."""
        return self.bristle_bodies[:, -1] if self.bristle_bodies.size else np.zeros(0, dtype=np.int32)

    @property
    def floor_z(self) -> float:
        """The work surface the tips stand on, in world z [m]."""
        return float(self.frame[2]) + float(self.spec.at[2])

    def _sole_xform(self) -> wp.transform:
        """Where the sole's underside centre goes, in world: :meth:`BrushSpec.sole_height` over :attr:`BrushSpec.at`."""
        spec = self.spec
        pose = self.placed((*spec.at, 0.0, 0.0, 0.0, 1.0))
        return wp.transform(
            wp.vec3(float(pose[0]), float(pose[1]), float(pose[2]) + spec.sole_height),
            wp.quat(*[float(v) for v in pose[3:7]]),
        )

    def _plastic_cfg(self, builder: newton.ModelBuilder) -> Any:
        """The own grip and the strips: the scene's own contact law, as a rigid box gets, in this brush's group."""
        cfg = builder.default_shape_cfg.copy()
        cfg.density = float(self.spec.handle_density)
        cfg.mu = float(self.spec.handle_friction)
        cfg.collision_group = self.group
        return cfg

    def _bristle_cfg(self) -> Any:
        """The bristle capsules' contact settings, in this brush's group."""
        spec = self.spec
        return newton.ModelBuilder.ShapeConfig(
            density=float(spec.material.density),
            ke=float(spec.contact_ke),
            kd=float(spec.contact_kd),
            mu=float(spec.material.friction),
            margin=0.0,
            gap=float(spec.contact_gap),
            collision_group=self.group,
        )

    def _named_body(self, builder: newton.ModelBuilder, name: str) -> int:
        """The body labelled ``name`` (its last path part), built before this brush."""
        hits = [b for b, label in enumerate(builder.body_label) if str(label).split("/")[-1] == name]
        if len(hits) != 1:
            raise KeyError(
                f"brush {self.spec.name!r}: handle {name!r} names {len(hits)} bodies; add the handle to the scene "
                "under a name nothing else uses"
            )
        return int(hits[0])

    def _robot_shapes(self, builder: newton.ModelBuilder) -> list[int]:
        """Every colliding shape of the scene's robot, built before this brush; empty with no robot."""
        if self.embodiment is None or not hasattr(self.embodiment, "builder_shapes"):
            return []
        collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
        return [int(s) for s in self.embodiment.builder_shapes(builder) if int(builder.shape_flags[s]) & collide]

    def _grip(self, builder: newton.ModelBuilder, sole: wp.transform) -> tuple[int, wp.transform, list[int], list[int]]:
        """The body the strips weld to, its pose, the joints its articulation starts with, and the shapes it added."""
        spec = self.spec
        if spec.handle:
            body = self._named_body(builder, spec.handle)
            pose = builder.body_q[body]
            self.handle_body, self.coord_start = body, -1
            self._handle_coord = _free_coord(builder, body)
            # On another solver its support cannot come back from the bristles
            # (see the module docstring), so there it holds itself up.
            from robosimstudio.core.builder import set_body_attribute

            set_body_attribute(builder, "mujoco:gravcomp", 1.0, [body])
            return (
                body,
                wp.transform(wp.vec3(*[float(v) for v in pose[:3]]), wp.quat(*[float(v) for v in pose[3:7]])),
                [],
                [],
            )
        body = builder.add_link(xform=sole, label=spec.name)
        free = builder.add_joint_free(child=body, label=f"{spec.name}_free_joint")
        self.handle_body, self.coord_start = int(body), int(builder.joint_q_start[free])
        radius = float(spec.handle_radius)
        # A capsule's axis is its local z; turned a quarter about y, it lies along x.
        along_x = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * math.pi)
        shape = builder.add_shape_capsule(
            body,
            xform=wp.transform(wp.vec3(0.0, 0.0, float(spec.sole[2]) + radius), along_x),
            radius=radius,
            half_height=0.5 * float(spec.handle_length) - radius,
            cfg=self._plastic_cfg(builder),
            label=f"{spec.name}/grip",
        )
        return int(body), sole, [free], [int(shape)]

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        """The grip, one sole strip per row welded under it, and a row of bristles welded under each strip.

        The strips are what keep the field affordable (module docstring): per
        row, no body carries more than ``columns + 1`` joints. The grip, its
        strips and their welds are one articulation, built first so its joints
        run contiguously as Newton requires; each bristle is its own chain.

        A named handle is already in an articulation of its own, which Newton
        1.7 refuses to see joined to another through a joint. There each strip
        is rooted to the world by a free joint, its own articulation, and the
        weld from the handle stays outside every articulation -- the way 1.7
        roots a cable and leaves its extra welds out. VBD solves a free joint
        as no constraint, so the welds hold the strips to the handle as before.
        """
        del episode
        spec = self.spec
        from newton.solvers import SolverVBD

        # The per-joint VBD attributes must exist before add_rod writes any.
        SolverVBD.register_custom_attributes(builder)
        body_start, joint_start, shape_start = builder.body_count, builder.joint_count, builder.shape_count

        sole = self._sole_xform()
        self.strip_coords = []
        grip, grip_xform, articulation, own = self._grip(builder, sole)
        handle_shapes = list(own)
        sx, sy, sz = (float(v) for v in spec.sole)
        roots = spec.roots().reshape(int(spec.rows), int(spec.columns), 2)
        plastic = self._plastic_cfg(builder)
        strips: list[tuple[int, wp.transform]] = []
        for r in range(int(spec.rows)):
            strip_xform = wp.transform_multiply(
                sole, wp.transform(wp.vec3(0.0, float(roots[r, 0, 1]), 0.0), wp.quat_identity())
            )
            strip = builder.add_link(xform=strip_xform, label=f"{spec.name}/strip{r:02d}")
            strips.append((int(strip), strip_xform))
            if spec.handle:
                root = builder.add_joint_free(child=strip, label=f"{spec.name}/strip{r:02d}_free")
                builder.add_articulation([root], label=f"{spec.name}/strip{r:02d}_articulation")
                self.strip_coords.append(int(builder.joint_q_start[root]))
            handle_shapes.append(
                builder.add_shape_box(
                    strip,
                    xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * sz), wp.quat_identity()),
                    hx=0.5 * sx,
                    hy=0.5 * sy / int(spec.rows),
                    hz=0.5 * sz,
                    cfg=plastic,
                    label=f"{spec.name}/sole{r:02d}",
                )
            )
            weld = builder.add_joint_fixed(
                parent=grip,
                child=strip,
                parent_xform=wp.transform_multiply(wp.transform_inverse(grip_xform), strip_xform),
                child_xform=wp.transform_identity(),
                label=f"{spec.name}/strip{r:02d}_weld",
            )
            if not spec.handle:
                articulation.append(weld)
        if articulation:
            builder.add_articulation(articulation, label=f"{spec.name}_articulation")

        # The hand meets the brush at the grip and nowhere else (BrushSpec.hand_contact),
        # and a named handle never meets its own bristles (module docstring).
        robot = [] if spec.hand_contact else self._robot_shapes(builder)
        named = (
            [int(t) for t in range(builder.shape_count) if int(builder.shape_body[t]) == grip] if spec.handle else []
        )
        for s in handle_shapes[len(own) :]:
            for t in robot:
                builder.add_shape_collision_filter_pair(s, t)

        bristle_cfg = self._bristle_cfg()
        gains = spec.material.drive_gains()
        down = wp.transform_vector(sole, wp.vec3(0.0, 0.0, -1.0))
        segments = int(spec.bristle_segments)
        leans = spec.leans()
        chains: list[list[int]] = []
        for r, (strip, strip_xform) in enumerate(strips):
            for c in range(int(spec.columns)):
                k = r * int(spec.columns) + c
                x, y = float(roots[r, c, 0]), float(roots[r, c, 1])
                out = math.hypot(x, y)
                tilt = float(leans[k])
                lean = (
                    wp.vec3(math.sin(tilt) * x / out, math.sin(tilt) * y / out, -math.cos(tilt)) if out > 0.0 else None
                )
                rod = newton.Rod.create_straight(
                    start=wp.transform_point(strip_xform, wp.vec3(x, 0.0, 0.0)),
                    direction=down if lean is None else wp.transform_vector(sole, lean),
                    length=float(spec.bristle_length),
                    segment_count=segments,
                    radius=float(spec.bristle_radius),
                )
                first, first_shape = builder.body_count, builder.shape_count
                add_capsule_chain(
                    builder, rod, cfg=bristle_cfg, color=spec.color, gains=gains, label=f"{spec.name}/bristle{k:03d}"
                )
                chains.append(list(range(first, builder.body_count)))
                _floor_inertia(builder, range(first, builder.body_count))
                for s in range(first_shape, builder.shape_count):
                    for t in (*robot, *named):
                        builder.add_shape_collision_filter_pair(s, t)
                # Welded where it was built: the weld's parent frame is the root
                # capsule's build pose in the strip's frame.
                root = builder.body_q[first]
                root_xform = wp.transform(
                    wp.vec3(*[float(v) for v in root[:3]]), wp.quat(*[float(v) for v in root[3:7]])
                )
                builder.add_joint_fixed(
                    parent=strip,
                    child=first,
                    parent_xform=wp.transform_multiply(wp.transform_inverse(strip_xform), root_xform),
                    child_xform=wp.transform_identity(),
                    label=f"{spec.name}/root{k:03d}",
                )
        self.strip_bodies = np.asarray([strip for strip, _ in strips], dtype=np.int32)
        self.bristle_bodies = np.asarray(chains, dtype=np.int32).reshape(-1, segments)
        self.handle_shapes = np.asarray(handle_shapes, dtype=np.int32)
        self.half_height = 0.5 * float(spec.bristle_length) / segments
        self._spans = (
            (body_start, builder.body_count),
            (joint_start, builder.joint_count),
            (shape_start, builder.shape_count),
        )

    def apply_display(self, builder: newton.ModelBuilder) -> None:
        """The grip and the sole in the handle's colour, the bristles in theirs."""
        from robosimstudio.core.render.finish import finish_for

        bristle = tuple(float(v) for v in self.spec.color)
        plastic = tuple(float(v) for v in self.spec.handle_color)
        handle = {int(s) for s in self.handle_shapes}
        for s in range(*self.range.shapes):
            color, role = (plastic, "objects") if s in handle else (bristle, "deformables")
            builder.shape_color[s] = color
            self.finishes[s] = finish_for(color, role)

    def owned_shapes(self) -> np.ndarray:
        """Nothing world-fixed: every shape sits on a strip, a capsule or the own grip."""
        return np.zeros(0, dtype=np.int32)

    # -------------------------------------------------------------- phase 2
    def bind(self, model: newton.Model) -> None:
        super().bind(model)
        (b0, b1), (j0, j1), (s0, s1) = self._spans
        self.bodies = np.arange(b0, b1, dtype=np.int32)
        self.joints = np.arange(j0, j1, dtype=np.int32)
        self.shapes = np.arange(s0, s1, dtype=np.int32)

    @property
    def bristle_shapes(self) -> np.ndarray:
        """Every bristle capsule's shape: this brush's shapes but the plastic ones."""
        return np.setdiff1d(self.shapes, self.handle_shapes).astype(np.int32)

    # -------------------------------------------------------------- phase 3
    def contribute_hooks(self, hooks) -> None:
        """Mark the lattice under every down tip, after every solve."""
        hooks.add("post_solve", self._mark, name=f"{self.name}-scrub", cadence=1)

    def _ensure_device(self, device) -> None:
        if self._tips_device is not None and self._tips_device.device == device:
            return
        tips = self.tips
        if self.worlds is not None:
            tips = self.worlds.spread_bodies(tips).reshape(-1)
        self._tips_device = wp.array(np.asarray(tips, dtype=np.int32), dtype=wp.int32, device=device)
        grid = self._lattice()
        self._grid_device = wp.zeros(self._num_worlds * grid.nx * grid.ny, dtype=wp.int32, device=device)

    def _lattice(self) -> ScrubGrid:
        """The host lattice, centred on the brush's build footprint; made once."""
        if self._grid is None:
            centre = self.placed((*self.spec.at, 0.0, 0.0, 0.0, 1.0))[:2]
            self._grid = ScrubGrid.around(centre)
        return self._grid

    def _mark(self, state, dt) -> None:
        del dt
        if not self.bristle_bodies.size:
            return
        self._ensure_device(state.body_q.device)
        grid = self._lattice()
        n = int(len(self.tips))
        wp.launch(
            _mark_tips,
            dim=n * self._num_worlds,
            inputs=[
                state.body_q,
                self._tips_device,
                float(self.half_height),
                float(self.floor_z + self.spec.bristle_radius + SCRUB_BAND),
                float(grid.origin[0]),
                float(grid.origin[1]),
                float(grid.cell),
                int(grid.nx),
                int(grid.ny),
                n,
                self._grid_device,
            ],
            device=state.body_q.device,
        )

    # -------------------------------------------------------------- phase 4
    def on_scene_ready(self, pipeline) -> None:
        """Remember every world's build pose, which a reset restores, and size the lattice before any capture."""
        self._rest_body_q = pipeline.state_0.body_q.numpy().copy()
        self._rest_joint_q = pipeline.state_0.joint_q.numpy().copy()
        self._ensure_device(pipeline.state_0.body_q.device)

    def _global(self, indices, world: int) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if self.worlds is None:
            return indices
        return np.asarray([self.worlds.body(int(b), world) for b in indices], dtype=np.int64)

    def reset(self, pipeline, episode, *, world: int = 0) -> None:
        """This world's strips, bristles and own grip back on their build pose, at rest, and its lattice cleared.

        A named handle is its own entity's to restore, and an episode may put it
        somewhere other than where it was built. Its reset runs first (it was
        built first) and writes that pose into the joint coordinates, so the
        strips and bristles go back to their build pose on the handle, carried
        to where the handle is now. The handle's body is put there too: the
        kinematics pass after the entities places the strips' welds from it,
        and would otherwise place them from where the last episode left it.
        """
        del episode
        if self._rest_body_q is None:
            return
        rows = self._global(self.bodies, world)
        rest = self._rest_body_q[rows]
        handle: tuple[int, np.ndarray] | None = None
        if self._handle_coord >= 0:
            row = int(self._global([self.handle_body], world)[0])
            c = self._handle_coord if self.worlds is None else int(self.worlds.coord(self._handle_coord, world))
            placed = pipeline.state_0.joint_q.numpy()[c : c + 7]
            rest = _carried(rest, self._rest_body_q[row], placed)
            handle = (row, placed)
        starts = [self.coord_start] if self.coord_start >= 0 else list(self.strip_coords)
        coords = np.concatenate([np.arange(c, c + 7, dtype=np.int64) for c in starts]) if starts else None
        if coords is not None and self.worlds is not None:
            coords = np.asarray([self.worlds.coord(int(c), world) for c in coords], dtype=np.int64)
        joint_rest = None
        if coords is not None and self._rest_joint_q is not None:
            joint_rest = self._rest_joint_q[coords]
            if handle is not None and self.coord_start < 0:
                # A strip's free joint holds its body's pose, so it is carried with the handle as the body is:
                # left at the build pose, the kinematics pass would put the strips back where they were built.
                index = {int(b): i for i, b in enumerate(self.bodies)}
                joint_rest = rest[[index[int(b)] for b in self.strip_bodies]].reshape(-1)
        for state in (pipeline.state_0, pipeline.state_1):
            body_q = state.body_q.numpy()
            body_q[rows] = rest
            if handle is not None:
                body_q[handle[0]] = handle[1]
            state.body_q.assign(body_q)
            body_qd = state.body_qd.numpy()
            body_qd[rows] = 0.0
            state.body_qd.assign(body_qd)
            if joint_rest is not None:
                joint_q = state.joint_q.numpy()
                joint_q[coords] = joint_rest
                state.joint_q.assign(joint_q)
        if world == 0:
            # A teleport leaves the coupled solver holding last episode's contact
            # anchors and warm starts. ``flags=0`` clears those and nothing else:
            # by default the solvers also put every joint coordinate back to the
            # model's, which undid each entity reset before this one, the
            # handle's episode pose among them.
            reset_solvers = getattr(pipeline, "reset_solvers", None)
            if reset_solvers is not None:
                reset_solvers(flags=0)
        self.clear_scrub(world)

    # ---------------------------------------------------------------- reading
    def scrub_grid(self, world: int = 0) -> ScrubGrid:
        """This world's coverage lattice, read back from the device: its own slab and no other world's."""
        grid = self._lattice()
        out = ScrubGrid(origin=grid.origin, cell=grid.cell, nx=grid.nx, ny=grid.ny)
        if self._grid_device is not None:
            size = grid.nx * grid.ny
            flat = self._grid_device[world * size : (world + 1) * size].numpy()
            out.cells = np.asarray(flat, dtype=np.int32).reshape(grid.ny, grid.nx)
        return out

    def clear_scrub(self, world: int | None = None) -> None:
        """Forget where the tips have been, in one world or all, zeroing that slab on the device."""
        if self._grid_device is None:
            return
        if world is None:
            self._grid_device.zero_()
            return
        grid = self._lattice()
        size = grid.nx * grid.ny
        self._grid_device[world * size : (world + 1) * size].zero_()

    def scrubbed(self, lo, hi, *, world: int = 0) -> float:
        """The share of the patch ``[lo, hi]`` (world x, y [m]) this world's tips have scrubbed."""
        return self.scrub_grid(world).fraction(lo, hi)

    def _template_body_q(self, body_q, world: int) -> np.ndarray:
        """``body_q`` as one world sees it, template-indexed: a per-world view passes through."""
        values = np.asarray(body_q)
        if self.worlds is None or len(values) == self.worlds.bodies_per_world:
            return values
        return self.worlds.body_view(values, world)

    def tip_positions(self, body_q, *, world: int = 0) -> np.ndarray:
        """``(count, 3)`` tip points in world [m], from a read-back ``body_q`` (whole or one world's)."""
        poses = self._template_body_q(body_q, world)[self.tips]
        return poses[:, :3] + _z_axes(poses[:, 3:7]) * float(self.half_height)

    def read(self, form: str, snapshot):
        """The tips as this brush's ``points``: where it touches the world."""
        if form != "points" or not self.bristle_bodies.size:
            return None
        context = getattr(snapshot, "ctx", snapshot)
        return self.tip_positions(snapshot.body_q, world=int(getattr(context, "world", 0)))

    def handle(self, model):
        """The grip first, then every body the brush built, so a click lands on any of them."""
        del model
        if not len(self.bodies):
            return None
        from robosimstudio.scene.handles import EntityHandle

        bodies = tuple(dict.fromkeys((int(self.handle_body), *(int(b) for b in self.bodies))))
        return EntityHandle(
            name=self.name, body=int(self.handle_body), bodies=bodies, coord_start=self.coord_start, shapes=self.shapes
        )

    def published_states(self, snapshot) -> dict[str, float]:
        context = getattr(snapshot, "ctx", snapshot)
        grid = self.scrub_grid(int(getattr(context, "world", 0)))
        return {"n_bristles": float(self.spec.bristle_count), "scrubbed_cells": float(grid.cells.sum())}

    def tunables(self, pipeline=None):
        """The bristles' stiffness over their joints, and their contact material over their capsules."""
        from robosimstudio.core.tuning import ArrayWriter, ShapeWriter, Tunable
        from robosimstudio.specs import ContactMaterialSpec

        if self.model is None:
            return ()
        out = []
        # The bristles' rod joints, a slot each; the welds and an own grip's free joint are not among them.
        stretch, shear, bend, twist = rod_slot_dofs(self.model, self.joints)
        capsules = self.bristle_shapes
        if len(bend):

            class _BristleWriter(ArrayWriter):
                flags = 1 << 1
                """``ModelFlags.JOINT_DOF_PROPERTIES``; Newton 1.6's VBD reseeds the rod slots' penalties on it."""

                ROWS: ClassVar[dict[str, tuple[str, np.ndarray]]] = {
                    "bend_ke": ("joint_target_ke", np.concatenate([bend, twist])),
                    "bend_kd": ("joint_target_kd", np.concatenate([bend, twist])),
                    "stretch_ke": ("joint_target_ke", np.concatenate([stretch, shear])),
                    "stretch_kd": ("joint_target_kd", np.concatenate([stretch, shear])),
                }

                def read(self) -> dict[str, float]:
                    return {name: self._read(attr, rows) for name, (attr, rows) in self.ROWS.items()}

                def __call__(self, field_name: str, value) -> bool:
                    row = self.ROWS.get(field_name)
                    if row is None:
                        return False
                    return self.landed(self._write(row[0], row[1], float(value)))

            writer = _BristleWriter(self.model, pipeline)
            friction = ShapeWriter(self.model, capsules, pipeline)
            built = self.spec.material

            def write(field_name: str, value) -> bool:
                if field_name == "friction":
                    return friction("mu", value)
                return writer(field_name, value)

            out.append(
                Tunable(
                    name="material",
                    spec=replace(built, friction=float(friction.read().get("mu", built.friction)), **writer.read()),
                    write=write,
                    summary=(
                        f"Bend and stretch over {len(bend)} bristle joints, and the bristles' friction. Bend is "
                        "whether the field holds the grip up and how far a press bows it; density is built in."
                    ),
                )
            )
        if len(capsules):
            shapes = ShapeWriter(self.model, capsules, pipeline)
            out.append(
                Tunable(
                    name="contact",
                    spec=ContactMaterialSpec(**shapes.read()),
                    write=shapes,
                    summary=f"What the {len(capsules)} bristle capsules feel on contact.",
                )
            )
        return tuple(out)


# ---------------------------------------------------------------------------
# Constructor, arrangement, record, registration
# ---------------------------------------------------------------------------


def brush(
    *,
    name: str = "brush",
    at: Vec3 = (0.0, 0.0, 0.0),
    material: Any = "nylon",
    rows: int = 20,
    columns: int = 20,
    handle: str = "",
    **kwargs,
) -> BrushSpec:
    """A brush standing on its bristles: a sole of strips, a field of short stiff capsule chains under it, and a grip.

    Args:
        name: Key in handles, measures and the panel.
        at: Where the bristle tips' footprint centre stands, in the cell's
            work-surface frame: ``at=(x, y, 0)`` stands it on the bench.
        material: ``"nylon"`` (a dish brush), ``"soft"`` (a paint brush), a
            registered material's name, its record, or a :class:`BrushMaterialSpec`.
        rows: Bristles along y.
        columns: Bristles along x.
        handle: A body already in the scene to root the field in -- the one a
            hand closes on, standing on the sole -- or empty for the brush's own
            grip, which a hand cannot pinch (module docstring).
        **kwargs: Any other :class:`BrushSpec` field: ``sole``,
            ``bristle_length``, ``bristle_radius``, ``bristle_segments``, ``color``.

    :data:`~robosimstudio.specs.GENERIC` puts the bristles on ``vbd`` beside
    the robot; :data:`BRUSH_60HZ` also puts a named handle on the arm's solver.
    """
    return BrushSpec(
        name=name, at=at, material=resolve_material(material), rows=rows, columns=columns, handle=handle, **kwargs
    )


PROXY_MASS_SCALE = 1000.0
"""How much heavier the bristles' solver sees a named handle than the arm's solver does.

The strips are welded to the handle's mirror inside VBD, and that mirror is a
body VBD moves: with the handle moved 12 mm down over half a second by writing
its free-joint coordinates and velocity, at its own 53 g the strips went
0.48 mm down with it, the field pushing the mirror back up instead of the
mirror pressing the field, and at this scale 11.5 mm (``jointq.py``, GPU 3).
The fingers meet nothing on VBD, so nothing else feels the scale."""

BRUSH_60HZ = GENERIC.with_backend("rigid", GENERIC.backend_for("articulated")).with_(
    coupling=GENERIC.coupling.with_(mass_scale=PROXY_MASS_SCALE)
)
"""``GENERIC`` with free rigid bodies on the arm's solver, so a brush's named handle is one a hand can hold,
and its mirror on the bristles' solver :data:`PROXY_MASS_SCALE` times heavier.

The bristles stay on ``vbd`` (GENERIC's ``"*"``), at its 5 iterations and 1
substep, with no sticky contact matching and no rigid contact history. The
module docstring's press numbers and its named-handle budget were measured on
this arrangement as it ships, and its own-grip budget on ``GENERIC``, whose VBD
entry this is."""

TOUCHES = ("surface", "fixtures", "objects", "hands")
"""What a brush contacts unless it says otherwise: the bench, a fixture, a thing it scrubs, the fingers."""


class Brush:
    """Constructors for a brush's asset record, from the spec that places one."""

    @staticmethod
    def from_spec(
        spec: BrushSpec,
        *,
        name: str,
        summary: str = "",
        tags: tuple[str, ...] = ("brush",),
        touches: tuple[str, ...] = TOUCHES,
        states: tuple[StateSpec, ...] = (),
        notes: str = "",
    ) -> AssetRecord:
        """The brush ``rss.brush`` returned, as the :class:`AssetRecord` a task document places by type.

        Args:
            spec: What :func:`brush` returned. Its ``at`` is where it starts, and
                a named ``handle`` is the document object of that name.
            name: What a task document calls the type.
            touches: Roles this brush contacts, passed to :func:`contacts_with`.

        Raises:
            TypeError: For a spec that is not a :class:`BrushSpec`.
        """
        if not isinstance(spec, BrushSpec):
            raise TypeError(f"Brush.from_spec takes a BrushSpec, got {type(spec).__name__}")
        return AssetRecord(
            name=name,
            summary=summary
            or f"a brush of {spec.rows} x {spec.columns} bristles, {spec.bristle_length * 1e3:.0f} mm long",
            tags=tuple(tags),
            states=tuple(states),
            sim=SimRequirements(domain="brush", contacts=contacts_with(*touches), notes=notes),
            build=SpecFactory(spec),
        )


DOMAIN = register_domain(
    "brush",
    state=("bodies", "joints", "shapes"),
    reads=("brush", "rod"),
    role="deformables",
    contacts=("surface", "fixtures", "objects", "hands", "deformables"),
    summary="A brush: sole strips and a field of short capsule-chain bristles on VBD, under a grip.",
    engine="vbd",
    staged_by="BRUSH_60HZ",
)

register_entity("brush")(EntityBrush)


# ------------------------------------------------------------------ catalogue
register_atom(
    AtomSpec(
        name="scrubbed",
        summary="What share of a patch a brush's tips have swept while down on the surface.",
        args=(
            ArgSpec(name="body", substrates=("brush",), reads="handle"),
            ArgSpec(name="region", kinds=("region",), reads="region"),
        ),
        directions=("check", "observe"),
        returns=("fraction",),
        cost="cheap",
        reads_scene=True,
        provenance=(
            "Designed here. A lattice of 5 mm cells over the surface; a cell is scrubbed once any bristle tip's "
            "rounded end was over it within 2 mm of the surface. A history over the episode, kept on the device "
            "by the entity's post-solve kernel and cleared by a reset, so a stroke counts whatever the brush "
            "does afterwards."
        ),
        calibration=AtomCalibration(
            positives=(
                "Three strokes of the examples' brush across their 100 mm square read 1.0 (tests/test_scrub_bench); "
                "one stroke reads 0.85 (press.py)."
            ),
            negatives="The same brush pressed on the bench beside the square reads 0.0 (tests/test_scrub_bench).",
            gate=None,
            bench="the brush kind's medium tests",
            notes="No gate: a task states its own share.",
        ),
    )
)


@handler(
    "scrubbed",
    kinds=("", "region"),
    substrates=("brush", ""),
    summary="The share of the region's 5 mm cells a down tip has visited.",
)
def _scrubbed(snapshot, body: str, region: dict) -> dict[str, Any]:
    entities = getattr(snapshot, "entities", None) or {}
    entity = entities.get(body) if hasattr(entities, "get") else None
    if entity is None:
        entity = (getattr(snapshot, "extras", None) or {}).get(body)
    if entity is None or not hasattr(entity, "scrub_grid"):
        raise KeyError(f"{body!r} is not a brush the scene carries an entity for")
    lo, hi = region_bounds(region, (0.0, 0.0, 0.0))
    context = getattr(snapshot, "ctx", snapshot)
    fraction = entity.scrubbed(lo, hi, world=int(getattr(context, "world", 0)))
    return {"fraction": fraction, f"{body}/scrubbed": fraction}
