# SPDX-License-Identifier: Apache-2.0
"""Cartons: a box or a blank laid out from a die-line, whose creases keep the folds a hand gives them.

This package is what ``rss.carton`` builds. A carton is board cut and scored to a die-line
(:mod:`.dieline`): flat panels joined along creases. What a hand does to one -- cut its tape and
lift a flap, fold a blank's wall over an edge -- happens at the creases, and a crease is plastic: past
a small bend it keeps what it is given. Both models here put that yield on each crease, after the
perfectly plastic hinge :mod:`robosimstudio_materials.paper` already ships (Narain, Pfaff and
O'Brien, SIGGRAPH 2013); they differ in what lies between the creases.

**Two models, chosen by the board** (:mod:`.board`):

* ``"panels"`` (:mod:`.panels`), for corrugated board: rigid plates on revolute hinges whose spring
  reference yields, on MuJoCo with the robot. :data:`C_FLUTE` is one. Stage :data:`CARTON_60HZ`.
* ``"sheet"`` (:mod:`.sheet`), for folding carton board: a VBD membrane meshed from the die-line,
  every crease a row of mesh edges with a stiffness and a yield of its own. :data:`FOLDING_CARTON`
  is one. It runs under any arrangement that steps paper.

Why two. Single-wall C-flute bends about a hundred times stiffer than the board ``rss.papers.CARDBOARD``
simulates (about 13 and 6 N m along and across its flutes, a sandwich estimate, against 0.012). Three
probes had the YAM open a taped 200 x 150 x 80 mm C-flute box and fold a tray blank's wall over a
block's edge, on 2026-10-27, RTX PRO 6000:

* As a VBD membrane the stiffest panel that still held the box up was about D 0.12 N m (``edge_ke``
  5,000 at 20 iterations; 15,000 sagged and 50,000 went to NaN). Lifting a flap caved the side wall
  in by 69 mm and the crease kept 20 degrees. The same model folded folding carton board well: a
  tray wall pressed to 82 degrees kept 73, at 0.36x real time. Hence the sheet model, for that board.
* As rigid panels on plastic hinges, on MuJoCo, the box opened: cut, its flaps sprang to 14 degrees
  (the yield), the YAM hooked one with a finger of its open jaws and swung it to 140, and it kept 127;
  the box, with 1 kg in it, moved 0.2 mm. A C-flute tray wall pressed to 86 degrees kept 73. At 5.3 to
  6.8 ms per 60 Hz control step, 2.5 to 3.2x real time. Hence the panels model, for corrugated.
* On Genesis's IPC with libuipc's plastic shell bending the fold kept best (88 held, 85 kept; elastic
  creases sprang back to 12), at 0.003 to 0.017x real time: a reference, not a robot simulator. It is
  not here.

**The crease numbers are estimates.** Neither board's crease stiffness or yield is fitted to a real
box; :mod:`.board` says what each number is. Two consequences a task meets: a hook that pulls an empty
box drags it (84 g of C-flute slid 26 mm and the flap slipped off at 27 degrees), so open a box with
something in it or a second hand on it; and the panels model's plates never dent or buckle, so a
squeeze or a stack load reads nothing.

**Reading one.** ``sim["box"].entity`` is the carton. On panels, :meth:`EntityCarton.angle` is a
hinge's angle and :meth:`EntityCarton.references` what each keeps; :meth:`EntityCarton.cut_tape`
cuts the tape. On a sheet, :meth:`EntityCartonSheet.fold_angle` is a panel's turn on its crease and
:meth:`EntityCartonSheet.kept_angle` what its crease keeps. Every angle is 0 as built and grows the way
its panel opens: a box flap upward, a blank's wall down over the edge it lies on. What a crease keeps is
counted the same way, so a flap folded shut before the carton was built keeps its yield from the start.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.arrangement import GENERIC
from robosimstudio.specs.base import Vec3
from robosimstudio.specs.physics import register_domain

from .board import C_FLUTE, FOLDING_CARTON, BoardSpec
from .dieline import STYLES, Dieline, Panel
from .panels import CartonSpec, EntityCarton
from .sheet import CartonSheetSpec, EntityCartonSheet

__all__ = [
    "CARTON_60HZ",
    "C_FLUTE",
    "FOLDING_CARTON",
    "BoardSpec",
    "CartonSheetSpec",
    "CartonSpec",
    "Dieline",
    "EntityCarton",
    "EntityCartonSheet",
    "Panel",
    "carton",
    "cartons",
]


CARTON_60HZ = GENERIC.with_backend("carton", GENERIC.backend_for("articulated"))
"""``GENERIC`` with a panels carton on the arm's MuJoCo, one island with the robot, so a finger pushes a
flap with MuJoCo's own contact and the hinges are MuJoCo joints the yield kernel can reach.

Everything else is GENERIC's, VBD for ``"*"`` included, so a sheet carton or a cloth beside the box
still steps. The open-box measurements in the package docstring were taken on this arrangement's
equivalent (every body on one MuJoCo island at GENERIC's 60 Hz and 16 substeps)."""

cartons = SimpleNamespace(C_FLUTE=C_FLUTE, FOLDING_CARTON=FOLDING_CARTON)
"""The boards that ship, which is what ``rss.cartons`` resolves to."""

SIZES = {"rsc": (0.20, 0.15, 0.08), "tray": (0.12, 0.10, 0.05)}
"""Each style's default size [m]: an RSC a down-pointing YAM hand can reach every flap of, and a tray blank."""


def carton(
    *,
    name: str = "carton",
    style: str = "rsc",
    size: tuple[float, float, float] | None = None,
    at: Vec3 = (0.0, 0.0, 0.0),
    board: BoardSpec | None = None,
    contents: float = 0.0,
    taped: bool = True,
    fixed: bool = False,
    cell: float = 0.01,
    **kwargs: Any,
) -> CartonSpec | CartonSheetSpec:
    """A carton of ``style`` in ``board``, its bottom's middle at ``at``.

    Args:
        name: Key in handles.
        style: ``"rsc"`` (an erected, closed and taped box) or ``"tray"`` (a flat blank).
        size: ``(L, W, H)`` for an RSC, ``(x, y, wall)`` for a tray [m]; ``None`` is the style's :data:`SIZES`.
        at: Where the middle of the carton's underside goes, in the cell's work-surface frame [m].
        board: One of ``rss.cartons``; ``None`` is :data:`C_FLUTE`. Its ``model`` picks the spec.
        contents: Mass inside [kg], on the bottom.
        taped: Whether an RSC's major flaps start taped. The sheet model has no tape.
        fixed: Hold the glued panels where they are built: welded on panels, pinned on a sheet.
        cell: The die-line's grid [m]: the sheet's mesh cell, and the snap both models share.
        **kwargs: Any other field of the spec the board picks: ``seam``, ``color``, ``quat``.
    """
    board = C_FLUTE if board is None else board
    if style not in STYLES:
        raise ValueError(f"carton {name!r}: style {style!r} is not one of {sorted(STYLES)}")
    size = SIZES[style] if size is None else tuple(float(v) for v in size)
    common = {
        "name": name,
        "style": style,
        "size": size,
        "board": board,
        "contents": contents,
        "fixed": fixed,
        "cell": cell,
    }
    if board.model == "panels":
        return CartonSpec(pose=tuple(float(v) for v in at), taped=taped, **common, **kwargs)
    if taped and style == "rsc":
        raise ValueError(f"carton {name!r}: the sheet model has no tape; build a {board.name} RSC with taped=False")
    # A membrane's particles are spheres of its particle radius: lifted by one, it rests where ``at`` says.
    r = board.sheet.particle_radius
    return CartonSheetSpec(
        pose=(float(at[0]), float(at[1]), float(at[2]) + r), material=board.sheet, **common, **kwargs
    )


DOMAIN = register_domain(
    "carton",
    state=("bodies", "joints", "shapes"),
    reads=("carton", "rigid"),
    role="objects",
    contacts=("surface", "fixtures", "objects", "robot", "deformables"),
    summary="Rigid board panels on crease hinges that yield: a corrugated box or blank.",
    engine="mujoco",
    staged_by="CARTON_60HZ",
)

register_entity("carton")(EntityCarton)
register_entity("carton_sheet")(EntityCartonSheet)
