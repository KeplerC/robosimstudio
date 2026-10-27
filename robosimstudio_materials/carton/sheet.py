# SPDX-License-Identifier: Apache-2.0
"""The sheet model: a carton as one VBD membrane whose hinges yield, with its own crease along every scored line.

It is :mod:`~robosimstudio_materials.paper` laid out from a die-line instead of a grid. Every panel
is meshed on one grid and the panels are merged where they meet, so each crease is a row of mesh
edges. Two things change from paper, both per hinge: a hinge on a crease line bends with the
board's ``crease_edge_ke`` and keeps its bend past the board's ``yield_angle``, while a hinge inside
a panel bends with the sheet's ``edge_ke`` and keeps a bend only past the sheet's own, larger yield.
The kernel is paper's, with the yield read per hinge.

A flap that was ``folded`` before the carton was built starts with its crease already kept: its
rest angle is set, as the scene becomes ready, to the angle it is built at less the board's yield,
so it springs open by that much when nothing holds it.

**Folding carton board only.** At corrugated board's stiffness VBD does not converge at robot rates:
the stiffest panel that held an erected box up was about a hundredth of C-flute's, and a hand lifting
a flap caved the side wall in instead (the package docstring has the numbers). Corrugated board is
the panels model's.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import warp as wp

from robosimstudio_materials.paper import EntityPaper, PaperMaterialSpec, PaperSpec, dihedral_angles, hinge_angle

from .board import FOLDING_CARTON, BoardSpec
from .dieline import STYLES, Dieline, mesh

__all__ = ["CartonSheetSpec", "EntityCartonSheet"]


@dataclass(frozen=True, kw_only=True)
class CartonSheetSpec(PaperSpec):
    """A carton on the sheet model. ``rss.carton`` returns one for a ``"sheet"`` board.

    The fields are :class:`~robosimstudio_materials.carton.panels.CartonSpec`'s, on a paper sheet:
    ``material`` is the board's ``sheet`` unless stated, and ``fixed`` pins the glued panels (a blank's
    base, an erected box's body) where they are built, as a jig would hold them.
    """

    entity: str = "carton_sheet"
    name: str = "carton"
    mesh: str = "given"
    style: str = "tray"
    size: tuple[float, float, float] = (0.12, 0.10, 0.05)
    board: BoardSpec = field(default_factory=lambda: FOLDING_CARTON)
    material: PaperMaterialSpec = field(default_factory=lambda: FOLDING_CARTON.sheet)
    contents: float = 0.0
    fixed: bool = False
    cell: float = 0.01
    seam: float = 0.006
    dieline_options: tuple[tuple[str, Any], ...] = ()
    color: tuple[float, float, float] = (0.70, 0.52, 0.32)
    settle_frames: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.style not in STYLES:
            raise ValueError(f"carton {self.name!r}: style {self.style!r} is not one of {sorted(STYLES)}")
        if self.board.model != "sheet":
            raise ValueError(f"carton {self.name!r}: board {self.board.name!r} is {self.board.model}, not a sheet")
        if self.contents < 0.0:
            raise ValueError(f"carton {self.name!r}: contents cannot be negative, got {self.contents}")

    def dieline(self) -> Dieline:
        options = dict(self.dieline_options)
        if self.style == "rsc":
            options.setdefault("seam", self.seam)
            # Over the minor flaps by two particle radii and a little, so the layers start apart.
            options.setdefault("rise", 2.0 * self.material.particle_radius + 5.0e-4)
        return STYLES[self.style](self.size, cell=self.cell, **options)


@wp.kernel
def _crease_kernel(
    rows: wp.array(dtype=wp.int32),
    yields: wp.array(dtype=float),
    edge_indices: wp.array2d(dtype=wp.int32),
    params: wp.array(dtype=float),
    dt: float,
    particle_q: wp.array(dtype=wp.vec3),
    edge_rest_angle: wp.array(dtype=float),
):
    """Paper's crease kernel with the yield read per hinge, from ``yields``, instead of per sheet.

    ``params`` is paper's array: ``params[0] <= 0`` switches every hinge off, and ``params[1]`` and
    ``params[2]`` are the rate and the recovery.
    """
    i = wp.tid()
    if params[0] <= 0.0:
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
    y = yields[i]
    if wp.abs(excess) > y:
        rest = rest + params[1] * (excess - wp.sign(excess) * y)
    rest = rest - params[2] * dt * rest
    edge_rest_angle[e] = rest


class EntityCartonSheet(EntityPaper):
    """A paper sheet meshed from a die-line, with a stiffness and a yield of its own on every crease."""

    spec: CartonSheetSpec
    domain = "paper"
    world_aware = False
    """The per-hinge arrays are world 0's; a replicated build would crease every world by world 0's yields."""

    def __init__(self, spec: CartonSheetSpec, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.layout: dict = {}
        """The mesh's panels, crease rows and free-edge rows by panel name, as local vertex indices."""
        self.is_crease = np.zeros(0, dtype=bool)
        """Which of :attr:`rows` lie on a crease line."""
        self._yields: Any = None
        self._pipeline: Any = None
        self._built = np.zeros(0)

    # -------------------------------------------------------------- phase 1
    def surface(self) -> tuple[np.ndarray, np.ndarray]:
        vertices, triangles, self.layout = mesh(self.spec.dieline(), self.spec.cell)
        return vertices, triangles

    # -------------------------------------------------------------- phase 2
    def _local(self, indices) -> np.ndarray:
        return np.asarray(indices, dtype=np.int64) + int(self.range.particles[0])

    def bind(self, model) -> None:
        super().bind(model)
        stepping = self.stepping_model if self.stepping_model is not None else model
        edges = stepping.edge_indices.numpy().reshape(-1, 4)[self.rows]
        lo = int(self.range.particles[0])
        keys = [(min(a, b) - lo, max(a, b) - lo) for a, b in zip(edges[:, 2], edges[:, 3], strict=True)]
        creases = {(min(a, b), max(a, b)) for a, b in self.layout["creases"]}
        self.is_crease = np.array([k in creases for k in keys], dtype=bool)
        board = self.spec.board
        bending = stepping.edge_bending_properties.numpy()
        bending[self.rows[self.is_crease], 0] = board.crease_edge_ke
        stepping.edge_bending_properties.assign(bending)
        yields = np.where(self.is_crease, board.yield_angle, self.spec.material.yield_angle).astype(np.float32)
        self._yields = wp.array(yields, dtype=wp.float32, device=stepping.device)
        if self.spec.fixed:
            dieline = self.spec.dieline()
            glued = sorted({v for p in dieline.glued for v in self.layout["panels"][p.name]})
            self.pinned = self._local(glued).astype(np.int32)

    def apply_model_materials(self, model) -> None:
        """Paper's, then the contents' mass spread over the bottom panel's free particles.

        Not over a pinned one: paper made it kinematic by giving it no mass, and mass given back frees it.
        A carton held by its bottom carries its contents on the jig.
        """
        super().apply_model_materials(model)
        if self.spec.contents <= 0.0:
            return
        bottom = self._local(self.layout["panels"].get("bottom", self.layout["panels"].get("base", [])))
        bottom = np.setdiff1d(bottom, self.pinned)
        if not len(bottom):
            return
        mass = model.particle_mass.numpy()
        mass[bottom] += self.spec.contents / max(1, len(bottom))
        model.particle_mass.assign(mass)
        model.particle_inv_mass.assign(np.where(mass > 0.0, 1.0 / np.maximum(mass, 1.0e-12), 0.0).astype(np.float32))

    # -------------------------------------------------------------- phase 3
    def _crease(self, state, dt) -> None:
        model = self.stepping_model if self.stepping_model is not None else self.model
        wp.launch(
            _crease_kernel,
            dim=len(self.rows),
            inputs=[self._rows, self._yields, model.edge_indices, self._params, float(dt), state.particle_q],
            outputs=[self._rest],
            device=model.device,
        )

    # -------------------------------------------------------------- phase 4
    def on_scene_ready(self, pipeline) -> None:
        """Paper's, then the folded flaps' creases set to their built bend less the yield."""
        super().on_scene_ready(pipeline)
        self._pipeline = pipeline
        # The rest angles as built, which the builder took off the mesh: what :meth:`kept_angle` counts from.
        self._built = self._rest.numpy()[self.rows].astype(np.float64)
        folded = [p.name for p in self.spec.dieline().creased if p.folded]
        if not folded:
            return
        rows = self._hinge_rows(folded)
        model = self.stepping_model if self.stepping_model is not None else self.model
        built = dihedral_angles(pipeline.state_0.particle_q, model.edge_indices, self.rows[rows])
        rest = self._rest.numpy()
        give = min(self.spec.board.yield_angle, 0.99 * float(np.abs(built).min()))
        rest[self.rows[rows]] = built - np.sign(built) * give
        self._rest.assign(rest)
        if self.rest_angles is not None:
            self.rest_angles = rest[self.rows].copy()

    def _hinge_rows(self, names) -> np.ndarray:
        """Which of :attr:`rows` run along the crease of the panels ``names``."""
        model = self.stepping_model if self.stepping_model is not None else self.model
        edges = model.edge_indices.numpy().reshape(-1, 4)[self.rows]
        lo = int(self.range.particles[0])
        line = set()
        for name in names:
            line |= set(self.layout["hinge"][name])
        return np.flatnonzero(
            [(a - lo) in line and (b - lo) in line for a, b in zip(edges[:, 2], edges[:, 3], strict=True)]
        )

    # ---------------------------------------------------------------- runtime
    def fold_angle(self, name: str) -> float:
        """How far panel ``name`` has turned on its crease now [rad]: 0 as built, growing the way it opens."""
        panel = self.spec.dieline().panel(name)
        q = self._pipeline.state_0.particle_q.numpy().astype(np.float64)
        d = q[self._local(self.layout["free"][name])].mean(0) - q[self._local(self.layout["hinge"][name])].mean(0)
        up = d[2] if panel.opens == "up" else -d[2]
        return float(np.arctan2(up, float(np.dot(d, np.asarray(panel.v, dtype=np.float64)))))

    def kept_angle(self, name: str) -> float:
        """The bend panel ``name``'s crease keeps now [rad]: how far the rest angles along its crease row have moved
        from the angles the carton was built at. A flat panel's starts at 0; a ``folded`` flap's starts at its yield."""
        rows = self._hinge_rows([name])
        if not len(rows):
            return 0.0
        return float(np.abs(self._rest.numpy()[self.rows[rows]] - self._built[rows]).mean())

    def tunables(self, pipeline=None):
        """Paper's material tunable, with the two fields this carton holds per hinge routed to the panels.

        ``yield_angle`` is the panels' yield: it goes to the hinges inside panels, and the creases keep
        the board's (0 still switches plasticity off, as on paper). ``edge_ke`` is the panels' bending: the
        write goes to every hinge of the sheet, then the creases get the board's ``crease_edge_ke`` back.
        """
        from robosimstudio.scene.tuning import LiveMaterial, locate_cloth

        base = super().tunables(pipeline)
        if not base:
            return base
        (material,) = base
        inner = material.write
        target = locate_cloth(self.model, self)
        creases = LiveMaterial(self.model, replace(target, edges=self.rows[self.is_crease]), pipeline)

        def write(name: str, value: Any) -> bool:
            done = inner(name, value)
            if name == "yield_angle":
                yields = self._yields.numpy()
                yields[~self.is_crease] = float(value)
                self._yields.assign(yields)
            elif name == "edge_ke" and done:
                creases.write_field("edge_ke", self.spec.board.crease_edge_ke)
            return done

        return (replace(material, write=write),)
