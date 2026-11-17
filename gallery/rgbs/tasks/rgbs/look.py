# SPDX-License-Identifier: Apache-2.0
"""How the ``rgbs`` bench looks: the real cell's surfaces, for the path tracer.

The five reference clips in ``sample_videos/`` show one bench: interlocking
grey foam mats with a pale 25 mm grid printed on them (300 mm tiles), an oak ledge
and a slatted oak wall behind the arms, soft office light from above, and a
YAM whose shells are white and whose joints are black. Nothing about that is
physics, so none of it is a spec's business -- and it is all a person notices
first. This module states it once, in the three forms the render side takes:

* **Finishes** (:class:`~robosimstudio.core.render.finish.Finish`) the variant
  builders put on their specs beside the colour -- kraft board, stainless,
  black print, white foam, clear film.
* **Textures**, procedural and registered by name, so the grid, the kraft and
  the oak are forty lines of numpy each rather than a PNG nobody can diff.
* **Scenery** (:func:`~robosimstudio.core.render.scenery.register_scenery`):
  the mat laid over the bench top and the ledge and wall behind the arm
  mounts -- geometry the camera sees and nothing collides with. Placed against
  the cell the task stands in, so it follows the bench rather than a number.

Plus the one :data:`RGBS_LOOK` every variant states as its ``render``. It is
the house look with the ``lab`` lighting and this scenery *added*, because a
task that states any :class:`~robosimstudio.core.render.spec.RenderSpec`
replaces the house look wholesale and an all-``None`` one would silently drop
the tiled tracer's fill light and background.

**Only the path-traced backend sees any of this.** Colours are shared by every
backend and are set on the specs; finishes, scenery and the lighting preset are
RTX's alone. So a tiled observation of ``rgbs/seam`` is exactly what it was
before this module existed, and the re-render of the same recording is the one
that looks like the clip.

**Framing.** The calibrated camera hangs over the arms and looks out across the
bench, so what it sees beyond the work is the bench's ``+x`` edge -- and that is
where the ledge and the slat wall stand. The arms are behind and below the lens,
reaching into the bottom of the frame.

Reference values, sampled off the clips (sRGB): mat 180/180/179 with lines at
about 215/220/222; oak ledge 242/201/161; slat gaps 63/62/64; carton 133/100/60;
tape 255/252/174; sponge 248/251/251; end blocks 31/32/37; arm shells 243/243/246.

**The recipe.** Collect cheap, photograph once::

    python rgbs/scripts/rgbs_rgbs_reference.py --videos ../sample_videos          # stills + insets
    python scripts/look_snapshot.py --task rgbs/seam --renderer rtx \
        --cameras overhead,close,top \
        --reference outputs/rgbs_reference/box_decanting/inset_3.png   # tune here
    python scripts/bench_run.py --task rgbs/seam --solve routine --record \
        --renderer rtx --max-steps 3000 --out outputs/rgbs_look --name seam
    python scripts/render_offline.py --run outputs/rgbs_look/seam --renderer rtx \
        --quality final --width 1280 --height 960 --every 5 --channels rgb

``strip``, ``panel`` and ``handle`` are driven by ``--solve graph --graph gt``;
``thread`` and ``seam`` by ``--solve routine``; ``wipe`` has no policy yet and
records under ``idle``. A graph run records one row per *policy* step, so its
re-render wants ``--every 1`` and a low ``--fps``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from robosimstudio.core.render.finish import BRUSHED_METAL, GLOSS, MATTE, SATIN, Finish, register_texture
from robosimstudio.core.render.look import STUDIO
from robosimstudio.core.render.room import LAB_GRID_MAT, LAB_MAT_GREY, LAB_OAK, LAB_OAK_COLOR
from robosimstudio.core.render.room import lab as _lab_room
from robosimstudio.core.render.scenery import register_scenery
from robosimstudio.core.render.spec import RenderSpec

__all__ = [
    "BLACK",
    "BLACK_PLASTIC",
    "CLEAR_FILM",
    "CUTTER_GREY",
    "DREMEL_GREY",
    "DRUM_PINK",
    "DUCT_CREAM",
    "FILM_CLEAR",
    "GRID_MAT",
    "INK_BLUE",
    "KRAFT",
    "KRAFT_BROWN",
    "MAT_GREY",
    "OAK",
    "OAK_COLOR",
    "PANEL_BLACK",
    "PANEL_GLOSS",
    "RGBS_LOOK",
    "SCENERY_NAME",
    "SPONGE_WHITE",
    "STEEL",
    "TAPE",
    "WHITE_FOAM",
    "grid_finish_at",
    "rgbs_lab",
]

# ------------------------------------------------------------------ colours
#
# Linear RGB in [0, 1], as ``shape_color`` wants. None of these sits within the
# palette test's 2e-3 of a palette entry; they are the clip's materials, not
# the corpus's roles, and they are named here so a builder never writes the
# floats by hand.

MAT_GREY = LAB_MAT_GREY
"""The foam mat between its grid lines. Carried by the grid texture; stated for
the untextured fallback. Owned by :mod:`robosimstudio.core.render.room` since
the lab became a room any suite may stand in; re-exported here because this
suite's own objects are painted against it."""

KRAFT_BROWN = (0.235, 0.127, 0.045)
"""Single-wall corrugated board, the shaded face. The kraft texture is built on it."""

OAK_COLOR = LAB_OAK_COLOR
"""The ledge and the slats, lit. Owned by the library room, re-exported here."""

DUCT_CREAM = (0.687, 0.552, 0.352)
"""A PVC cable duct."""

BLACK = (0.018, 0.018, 0.021)
"""Black PLA prints and the bar's end blocks."""

CUTTER_GREY = (0.30, 0.31, 0.33)
"""A box cutter's body."""

DREMEL_GREY = (0.062, 0.060, 0.060)
"""The rotary tool's housing."""

DRUM_PINK = (0.55, 0.33, 0.26)
"""A used sanding drum."""

PANEL_BLACK = (0.030, 0.030, 0.035)
"""The film's substrate: a glossy black block."""

FILM_CLEAR = (0.92, 0.94, 0.96)
"""Protective film, which is clear: the colour is the tint of its highlights."""

SPONGE_WHITE = (0.93, 0.94, 0.94)
"""Open-cell foam."""

INK_BLUE = (0.060, 0.090, 0.300)
"""A marker streak on stainless."""

# ------------------------------------------------------------------ finishes

GRID_MAT = LAB_GRID_MAT
"""The gridded foam mat: one 30 cm tile repeated, so the lines land on world
multiples of 25 mm on any box that carries it. The library's, so a fixture
standing on the mat and the mat itself cannot drift apart."""

KRAFT = Finish(roughness=0.8, texture="rgbs_kraft", uv_scale=(4.0, 4.0))
OAK = LAB_OAK
BLACK_PLASTIC = SATIN
STEEL = replace(BRUSHED_METAL, roughness=0.3, metallic=0.95)
WHITE_FOAM = replace(MATTE, roughness=0.95)
PANEL_GLOSS = replace(GLOSS, roughness=0.2)
CLEAR_FILM = Finish(roughness=0.15, opacity=0.35, ior=1.5, roughness_texture="rgbs_crinkle", uv_scale=(12.0, 12.0))
"""A 60 um film: mostly clear, creased where it was handled. The crinkle map
modulates roughness rather than colour, which is what a crease is."""
TAPE = Finish(roughness=0.3, opacity=0.62)
"""BOPP packing tape at the opacity ``peel_tape`` already states."""


def grid_finish_at(xy_surface) -> Finish:
    """The mat's grid on a fixture standing at ``xy_surface`` [m, surface frame].

    A box prototype's UVs start at its own centre, so a plate laid on the mat
    would carry a grid that starts at *its* corner and disagrees with the mat's
    by whatever the plate's position is modulo 25 mm. Shifting the texture by
    the plate's world centre, in repeats, puts the two grids on top of each
    other, and the plate reads as part of the mat it stands on.
    """
    from .staging import CELL

    cx, cy = float(CELL.center[0]) + float(xy_surface[0]), float(CELL.center[1]) + float(xy_surface[1])
    su, sv = GRID_MAT.uv_scale
    return replace(GRID_MAT, uv_offset=(round((cx * su) % 1.0, 4), round((cy * sv) % 1.0, 4)))


# ------------------------------------------------------------------ textures


def _srgb(linear: tuple[float, float, float]) -> np.ndarray:
    c = np.asarray(linear, dtype=np.float64)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def _noise(shape: tuple[int, int], seed: int, octaves: int = 4) -> np.ndarray:
    """Value noise in [0, 1], a few octaves, for mottling and grain."""
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, dtype=np.float64)
    amplitude, total = 1.0, 0.0
    for o in range(octaves):
        cells = 4 * 2**o
        coarse = rng.random((cells + 1, cells + 1))
        ys = np.linspace(0, cells, shape[0], endpoint=False)
        xs = np.linspace(0, cells, shape[1], endpoint=False)
        y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
        fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
        fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
        a = coarse[y0][:, x0]
        b = coarse[y0][:, x0 + 1]
        c = coarse[y0 + 1][:, x0]
        d = coarse[y0 + 1][:, x0 + 1]
        out += amplitude * ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy)
        total += amplitude
        amplitude *= 0.5
    return out / total


def _kraft(size: int = 512) -> np.ndarray:
    """Corrugated board: kraft brown, mottled, with faint flute lines."""
    base = _srgb(KRAFT_BROWN)
    mottle = (_noise((size, size), seed=23, octaves=4) - 0.5) * 0.10
    fibre = (np.random.default_rng(5).random((size, size)) - 0.5) * 0.05
    flutes = 0.02 * np.sin(np.arange(size) / size * 2 * np.pi * 40)[:, None]
    img = base[None, None, :] * (1.0 + mottle + fibre + flutes)[..., None]
    return np.clip(img, 0.0, 1.0)


def _crinkle(size: int = 512) -> np.ndarray:
    """Roughness map for a handled film: clear between creases, dull along them."""
    n = _noise((size, size), seed=41, octaves=5)
    ridges = 1.0 - np.abs(2.0 * n - 1.0)
    creases = np.clip((ridges - 0.82) / 0.18, 0.0, 1.0)
    return np.clip(0.12 + 0.6 * creases, 0.0, 1.0)


register_texture("rgbs_kraft", _kraft, version=1)
register_texture("rgbs_crinkle", _crinkle, version=1)

# ------------------------------------------------------------------ scenery

SCENERY_NAME = "rgbs_lab"


@register_scenery(SCENERY_NAME)
def rgbs_lab(kit) -> None:
    """The mat over the bench, the oak ledge at its ``+x`` edge, the slat wall,
    standing in a pale room.

    Every box and every constant this used to draw by hand now lives in
    :func:`robosimstudio.core.render.room.lab`, unchanged. It moved because this
    is the only backdrop in the repository that was measured against footage of
    the real cell, which makes it the one worth *offering* -- a suite that wants
    a photographic lab rather than an industrial shed should be able to say so
    without copying forty lines out of here. The name stays because the nine
    documents that state it are a record, and a record is not renamed to tidy an
    implementation.
    """
    _lab_room(kit)


# ------------------------------------------------------------------ the look

RGBS_LOOK: RenderSpec = replace(STUDIO, environment="lab", scenery=SCENERY_NAME, quality="preview")
"""What every ``rgbs`` variant renders under. The house look, plus the lab
lighting and the backdrop the path tracer alone honours."""
