# SPDX-License-Identifier: Apache-2.0
"""A carton's die-line: its panels, which crease joins which, and which flaps are taped.

Both models read the same die-line. The panels model (``panels.py``) makes the glued panels one rigid
body and every creased panel a link on a hinge; the sheet model (``sheet.py``) meshes all the panels as
one triangle sheet and puts every crease on a row of mesh edges, which is what Narain, Pfaff and
O'Brien (SIGGRAPH 2013) ask of a fold line, so the bend is not spread over the triangles beside it.

A style is laid out in the carton's own frame: z up, the bottom (or the blank) at z = 0, centred in x
and y, metres. Two ship:

* ``rsc``, a regular slotted container, erected and closed: the bottom and four walls glued, the two
  minor flaps (on the end walls, x = +-L/2) lying flat over the opening, the two major flaps (on the
  side walls, y = +-W/2) over them, meeting at y = 0 across a seam, taped.
* ``tray``, a flat blank: a base and four walls in the plane, each wall creased to the base.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Dieline", "Panel", "mesh", "rsc", "tray"]


@dataclass(frozen=True)
class Panel:
    """One flat rectangle of board.

    Attributes:
        name: Key in the die-line, and the suffix of the link and joint labels a panels carton builds.
        origin: A corner [m]. For a creased panel, one end of its crease.
        u: Unit axis along the crease (or along the panel's first side, for a glued one).
        a: Length along ``u`` [m].
        v: Unit axis from the crease to the free edge.
        b: Length along ``v`` [m], the flap's depth.
        joint: ``"glued"``, part of the carton's body; ``"crease"``, hinged to the body on its ``v = 0`` edge.
        opens: For a creased panel, which way its angle grows from 0 as built: ``"up"`` (a box flap
            opening) or ``"down"`` (a blank's wall folding over the edge of whatever it lies on).
        rise: How far a creased panel lies above its crease when closed [m]: a major flap lies over the
            minor flaps, not through them. The sheet lifts every row past the first by this; the panels
            model raises the hinge.
        folded: Whether the panel was folded over on its crease before the carton was built: an erected
            box's flaps were folded shut from standing. Its crease already keeps a bend, so it rests
            the board's yield open of how it is built and springs open when nothing holds it.
    """

    name: str
    origin: tuple[float, float, float]
    u: tuple[float, float, float]
    a: float
    v: tuple[float, float, float]
    b: float
    joint: str = "glued"
    opens: str = "up"
    rise: float = 0.0
    folded: bool = False

    def __post_init__(self) -> None:
        if self.joint not in ("glued", "crease"):
            raise ValueError(f"panel {self.name!r}: joint must be 'glued' or 'crease', got {self.joint!r}")
        if self.opens not in ("up", "down"):
            raise ValueError(f"panel {self.name!r}: opens must be 'up' or 'down', got {self.opens!r}")

    @property
    def hinge(self) -> np.ndarray:
        """The middle of the crease [m]."""
        return np.asarray(self.origin, dtype=np.float64) + 0.5 * self.a * np.asarray(self.u, dtype=np.float64)


@dataclass(frozen=True)
class Dieline:
    """A style laid out at a size: its panels and which creased flaps are taped shut."""

    style: str
    size: tuple[float, float, float]
    panels: tuple[Panel, ...]
    taped: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict)

    def panel(self, name: str) -> Panel:
        for p in self.panels:
            if p.name == name:
                return p
        raise KeyError(f"{self.style} has no panel {name!r}; it has {[p.name for p in self.panels]}")

    @property
    def creased(self) -> tuple[Panel, ...]:
        return tuple(p for p in self.panels if p.joint == "crease")

    @property
    def glued(self) -> tuple[Panel, ...]:
        return tuple(p for p in self.panels if p.joint == "glued")


def _snap(length: float, cell: float) -> float:
    """``length`` rounded to whole cells, at least one: a flap's crease row must land on the wall's grid."""
    return max(cell, round(length / cell) * cell)


def _floor(length: float, cell: float) -> float:
    """``length`` rounded down to whole cells, at least one."""
    return max(cell, math.floor(length / cell + 1.0e-9) * cell)


def _on_grid(style: str, size: tuple[float, ...], cell: float) -> None:
    """Refuse a size that is not a whole number of cells: a flap's crease row would miss its wall's vertices.

    The sheet is then not one sheet: a flap shares a vertex or two with its wall instead of a row of hinges,
    and hangs there. The panels model would not mind, but both models build from one die-line.
    """
    off = [v for v in size if abs(v / cell - round(v / cell)) > 1.0e-6]
    if off:
        near = tuple(round(round(v / cell) * cell, 6) for v in size)
        raise ValueError(f"{style} size {tuple(size)} is not whole {cell * 1e3:g} mm cells; {near} is")


def rsc(
    size: tuple[float, float, float],
    *,
    cell: float = 0.01,
    seam: float = 0.006,
    slot: float = 0.005,
    rise: float = 0.0045,
) -> Dieline:
    """A regular slotted container ``size = (L, W, H)``, erected, closed and taped across its seam.

    Args:
        cell: The sheet model's mesh cell [m]. The size must be whole cells and the slot and the minor
            flaps' depth are snapped to it, so every flap's crease row lies on its wall's top-edge
            vertices; the panels model uses the same numbers, so the two models build the same box.
            The minor flaps are ``W / 2`` deep, rounded down, and stop a cell short of each other.
        seam: Gap between the major flaps' free edges [m]. A real RSC's is a few millimetres; a closed
            YAM jaw is about 9 mm thick, so a hand reaching through the seam needs its jaws open.
        slot: Inset of every flap at each end [m], the die's slots, so no two flaps share a vertex.
        rise: How far the major flaps lie over the minor ones [m]; more than the board's thickness.
    """
    L, W, H = (float(v) for v in size)
    _on_grid("rsc", (L, W, H), cell)
    if not 0.0 < seam < W:
        raise ValueError(f"rsc seam must be in (0, W), got {seam}")
    x0, x1, y0, y1 = -L / 2, L / 2, -W / 2, W / 2
    s = _snap(slot, cell)
    minor = _floor(min(W / 2, (L - cell) / 2), cell)
    major = W / 2 - seam / 2
    panels = (
        Panel("bottom", (x0, y0, 0.0), (1, 0, 0), L, (0, 1, 0), W),
        Panel("side_neg", (x0, y0, 0.0), (1, 0, 0), L, (0, 0, 1), H),
        Panel("side_pos", (x0, y1, 0.0), (1, 0, 0), L, (0, 0, 1), H),
        Panel("end_neg", (x0, y0, 0.0), (0, 1, 0), W, (0, 0, 1), H),
        Panel("end_pos", (x1, y0, 0.0), (0, 1, 0), W, (0, 0, 1), H),
        Panel("minor_neg", (x0, y0 + s, H), (0, 1, 0), W - 2 * s, (1, 0, 0), minor, joint="crease", folded=True),
        Panel("minor_pos", (x1, y0 + s, H), (0, 1, 0), W - 2 * s, (-1, 0, 0), minor, joint="crease", folded=True),
        Panel(
            "major_neg", (x0 + s, y0, H), (1, 0, 0), L - 2 * s, (0, 1, 0), major, joint="crease", rise=rise, folded=True
        ),
        Panel(
            "major_pos",
            (x0 + s, y1, H),
            (1, 0, 0),
            L - 2 * s,
            (0, -1, 0),
            major,
            joint="crease",
            rise=rise,
            folded=True,
        ),
    )
    return Dieline("rsc", (L, W, H), panels, taped=("major_neg", "major_pos"), meta={"seam": seam, "slot": s})


def tray(size: tuple[float, float, float], *, cell: float = 0.01, slot: float = 0.01) -> Dieline:
    """A flat tray blank ``size = (x, y, wall)``: the base ``x`` by ``y`` and a wall ``wall`` wide on each side.

    The walls are inset by ``slot`` at the corners, so they fold without meeting, and each opens
    ``"down"``: lying on a block the base's size, a wall pushed over the block's edge folds round it.
    """
    bx, by, h = (float(v) for v in size)
    _on_grid("tray", (bx, by, h), cell)
    x0, x1, y0, y1 = -bx / 2, bx / 2, -by / 2, by / 2
    s = _snap(slot, cell)
    panels = (
        Panel("base", (x0, y0, 0.0), (1, 0, 0), bx, (0, 1, 0), by),
        Panel("wall_yneg", (x0 + s, y0, 0.0), (1, 0, 0), bx - 2 * s, (0, -1, 0), h, joint="crease", opens="down"),
        Panel("wall_ypos", (x0 + s, y1, 0.0), (1, 0, 0), bx - 2 * s, (0, 1, 0), h, joint="crease", opens="down"),
        Panel("wall_xneg", (x0, y0 + s, 0.0), (0, 1, 0), by - 2 * s, (-1, 0, 0), h, joint="crease", opens="down"),
        Panel("wall_xpos", (x1, y0 + s, 0.0), (0, 1, 0), by - 2 * s, (1, 0, 0), h, joint="crease", opens="down"),
    )
    return Dieline("tray", (bx, by, h), panels, meta={"slot": s})


STYLES = {"rsc": rsc, "tray": tray}
"""Each style's layout function, by the name ``rss.carton(style=...)`` takes."""


# =============================================================================
# The sheet model's mesh
# =============================================================================


def mesh(dieline: Dieline, cell: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """The die-line as one triangle sheet, panels merged where they share an edge.

    Returns ``(vertices [n, 3], triangles [m, 3], info)``. ``info`` holds, by panel name, its vertex
    indices (``panels``), its crease row and its free-edge row (``hinge`` and ``free``, creased panels
    only), and ``creases``: every edge whose two triangles lie on different panels, as vertex pairs.
    A grid's diagonal alternates cell by cell, so the sheet has no preferred shear direction.
    """
    key: dict[tuple, int] = {}
    verts: list[np.ndarray] = []
    tris: list[tuple[int, int, int]] = []
    owner: list[str] = []
    info: dict = {"panels": {}, "hinge": {}, "free": {}}

    def vid(p: np.ndarray) -> int:
        k = tuple(np.round(p, 6))
        if k not in key:
            key[k] = len(verts)
            verts.append(p)
        return key[k]

    for panel in dieline.panels:
        nu = max(1, int(round(panel.a / cell)))
        nv = max(1, int(round(panel.b / cell)))
        o, u, v = (np.asarray(x, dtype=np.float64) for x in (panel.origin, panel.u, panel.v))
        up = np.array([0.0, 0.0, panel.rise])
        ids = np.zeros((nu + 1, nv + 1), dtype=np.int64)
        for i in range(nu + 1):
            for j in range(nv + 1):
                ids[i, j] = vid(o + u * panel.a * i / nu + v * panel.b * j / nv + (up if j > 0 else 0.0))
        for i in range(nu):
            for j in range(nv):
                a, b, c, d = ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]
                pair = ((a, b, c), (a, c, d)) if (i + j) % 2 == 0 else ((a, b, d), (b, c, d))
                for t in pair:
                    tris.append(tuple(int(x) for x in t))
                    owner.append(panel.name)
        info["panels"][panel.name] = sorted(set(ids.ravel().tolist()))
        if panel.joint == "crease":
            info["hinge"][panel.name] = ids[:, 0].tolist()
            info["free"][panel.name] = ids[:, -1].tolist()

    sides: dict[tuple[int, int], list[str]] = {}
    for t, name in zip(tris, owner, strict=True):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            sides.setdefault((min(a, b), max(a, b)), []).append(name)
    creases = []
    for (a, b), names in sides.items():
        if len(names) > 2:
            raise ValueError(f"{dieline.style}: edge {(a, b)} is shared by {names}, which no sheet can be")
        if len(names) == 2 and names[0] != names[1]:
            creases.append((a, b))
    info["creases"] = creases
    return np.asarray(verts), np.asarray(tris, dtype=np.int32), info
