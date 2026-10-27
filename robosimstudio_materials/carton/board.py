# SPDX-License-Identifier: Apache-2.0
"""A cardboard grade, and the two that ship: C-flute corrugated board and folding carton board.

A board says which model it is simulated with, because the two grades need different ones (the
package docstring has the measurements): corrugated board is far too stiff for a VBD membrane at
robot rates and is built as rigid ``"panels"`` on plastic crease hinges, while folding carton board
is a ``"sheet"`` whose hinges yield, :mod:`~robosimstudio_materials.paper`'s material with a crease of
its own along every scored line.

**The crease numbers are estimates, not measurements.** Nothing here has been fitted to a real box
yet; each field says where its number came from.

**Fitting a panels board to a real box** takes two numbers, because its crease is perfectly plastic:
pushed open, a flap's moment rises along the spring until the bend passes the yield and then stays
at ``crease_stiffness * width * yield_angle``, and let go it springs back by exactly ``yield_angle``.
So push a real flap of crease length ``w`` [m] through 90 degrees with one finger at ``d`` [m] from
the crease, read the force ``F`` [N] it levels off at, let go, and read how far it springs back, ``s``
[rad]: ``yield_angle = s`` and ``crease_stiffness = F * d / (s * w)``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from robosimstudio_materials.paper import CARDBOARD, PaperMaterialSpec

__all__ = ["C_FLUTE", "FOLDING_CARTON", "BoardSpec"]


@dataclass(frozen=True, kw_only=True)
class BoardSpec:
    """One grade of board, as both models read it.

    Attributes:
        name: A label for a reader.
        model: ``"panels"`` (rigid panels on plastic hinges, on MuJoCo) or ``"sheet"`` (a VBD
            membrane whose hinges yield).
        thickness: The board's caliper [m]. The panels model's plates are this thick.
        areal_density: Mass per area [kg/m^2].
        friction: Coulomb friction of the board's surface.
        crease_stiffness: Panels: a crease hinge's spring per metre of crease [N m/rad per m].
        crease_damping: Panels: its damping per metre of crease [N m s/rad per m].
        yield_angle: How far a crease bends past its rest angle before the bend is kept [rad], in
            both models. A box flap folded shut from standing rests this far open of shut, so a
            cut flap springs up to it.
        tape_stiffness: Panels: what a strip of tape across the seam adds to a taped flap's hinge,
            per metre of crease [N m/rad per m], holding it shut until :meth:`cut_tape`.
        armature: Panels: the hinge's reflected inertia [kg m^2]. Small; it keeps a light flap's
            hinge well conditioned in MuJoCo.
        sheet: Sheet: the membrane between the creases, a :class:`PaperMaterialSpec`. Its
            ``yield_angle`` is the panels' own, larger than a crease's.
        crease_edge_ke: Sheet: the bending stiffness of a hinge on a crease line, the ``edge_ke`` that
            replaces the sheet's on those hinges.
    """

    name: str
    model: str
    thickness: float
    areal_density: float
    friction: float = 0.5
    crease_stiffness: float = 0.0
    crease_damping: float = 0.0
    yield_angle: float = 0.25
    tape_stiffness: float = 0.0
    armature: float = 1.0e-5
    sheet: PaperMaterialSpec | None = None
    crease_edge_ke: float = 0.0

    def __post_init__(self) -> None:
        if self.model not in ("panels", "sheet"):
            raise ValueError(f"board {self.name!r}: model must be 'panels' or 'sheet', got {self.model!r}")
        if self.model == "sheet" and self.sheet is None:
            raise ValueError(f"board {self.name!r} is a sheet and names no sheet material")
        if self.thickness <= 0.0 or self.areal_density <= 0.0:
            raise ValueError(f"board {self.name!r}: thickness and areal_density must be positive")
        if self.yield_angle <= 0.0:
            raise ValueError(f"board {self.name!r}: yield_angle must be positive, got {self.yield_angle}")


C_FLUTE = BoardSpec(
    name="C-flute corrugated, single wall",
    model="panels",
    # About 4 mm: C-flute's caliper is 3.6-4.0 mm.
    thickness=0.004,
    # Two 200 g/m^2 liners and a 150 g/m^2 medium taken up about 1.45 times by its flutes, about
    # 0.62 kg/m^2. A 200 x 150 x 80 mm box of it weighs 84 g.
    areal_density=0.63,
    friction=0.5,
    # Estimates. 0.3 N m/rad for a 190 mm flap holds a flap closed against its own weight with a
    # few degrees to spare and lets the YAM's hook open it at the 2-3 N its fingers push with; the
    # yield of 0.25 rad is what a cut flap springs to (14 degrees). Neither is fitted to a real box:
    # the YAM pushing a real flap through 90 degrees and letting go would fit both.
    crease_stiffness=0.3 / 0.19,
    crease_damping=0.004 / 0.19,
    yield_angle=0.25,
    # Ten times the crease: a taped flap stays shut under the hook's pull until the tape is cut.
    tape_stiffness=3.0 / 0.19,
)
"""Single-wall C-flute corrugated board, 4 mm: a shipping box, as rigid panels on plastic crease hinges."""

FOLDING_CARTON = BoardSpec(
    name="folding carton board",
    model="sheet",
    thickness=5.0e-4,
    areal_density=0.5,
    friction=0.5,
    # A scored crease bends easily and keeps its bend early; the panel between creases is
    # :data:`~robosimstudio_materials.paper.CARDBOARD`, which only keeps a bend past 0.6 rad. The
    # crease's edge_ke is a tenth of the panel's. Estimates, measured only as far as a tray wall the
    # YAM pressed over a block's edge keeping 73 degrees of an 82-degree fold.
    yield_angle=0.15,
    crease_edge_ke=50.0,
    sheet=replace(CARDBOARD, zero_rest_angle=False, friction=0.5, yield_angle=0.6),
)
"""Folding carton board, 0.5 mm: a cereal box or a tray, as a VBD sheet whose creases yield."""
