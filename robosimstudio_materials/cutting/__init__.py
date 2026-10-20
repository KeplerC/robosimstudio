# SPDX-License-Identifier: Apache-2.0
"""The ``cutting`` kind: a block a knife parts, on the walk-on-spheres cage cutter.

This package is what ``rss.cuttable`` and ``rss.knife`` build, imported as
``robosimstudio_materials.cutting``. Run ``examples/03_materials/10_cutting.py`` first.

**The solver is vendored, not written here.** ``cagecut/`` is the code of "Fast Cutting of Cage Based
Deformation with Walk on Spheres" (Saeed et al., SIGGRAPH 2026; MIT, its licence and provenance in
``cagecut/``): a block is an elastic body (St. Venant-Kirchhoff, backward Euler) whose degrees of
freedom are the corners of a cage, its interior and its drawn skin carried by harmonic coordinates
that walks on spheres estimate on the GPU, and a blade sweeping a plane splits the cage and updates
only the walks the new cut surface touches. Its own demo, the starfish, steps in 3.3 ms and cuts in 8
to 26 ms a step over 8,158 skin points (``demo_star.py``, GPU 5). Here a block starts as an
eight-corner box cage with 1,024 walks over 785 interior points and a 994-point skin: building it
solves the walks in 0.6 s (1.2 to 1.7 s the first time in a process), the cage steps in 3.2 ms, and a cut step costs 7.6 ms (median of 51
in each of three drives, 15 ms at most warm; ``measure.py``, GPU 5).

**What this kind adds to it**, in ``law.py`` and ``entity.py``:

* *The board is a fixture.* The paper's solver has no contact, so the block's base face is held
  where it stands, and every vertex a cut puts on it is held too. Under its own weight a 100 x 60
  x 40 mm block settles 0.15 mm (tofu), 0.35 mm (cake) and 0.01 mm (foam) (``presets.py``).
* *The knife is a rigid body.* :func:`knife` builds a handle and a blade on one free body, on the
  arm's solver (:data:`CUTTING_50HZ`), so the pads hold it by contact like any box. Its edge, the
  ``blade`` feature, is a segment; whether it parts the block is a threshold, not geometry.
* *The cut starts in the blade's plane.* When the edge first goes into the block, the cutter is
  given the plane the blade lies in and the edge's line, turned within that plane by the least
  multiple of half a degree that clears every face it crosses by 2 degrees: the topology cannot
  host a tip on a face parallel to the line, so a knife held level gets a line 2 degrees off level
  and the cut runs up to 2.1 mm ahead of the edge at one end of a 60 mm block. The line is unbounded:
  a knife spanning only part of the block parts all of it. The plane stays the cut's until it goes
  right through; an edge more than 5 mm off it cuts nothing, and a new cut starts only once the
  edge has been out over the block again.
* *The edge law, this kind's own* (``law.py``): past the cut's front the block pushes back on the
  edge, ``E`` per metre of edge for a shallow press and without bound as the edge squeezes the last
  of the block against the board; the front runs when the push reaches the blade's threshold,
  ``toughness / sharpness`` per metre of edge in the block, and the last 2.5 mm part when the edge
  meets the board. A sharp knife through 60 mm of tofu, cake and foam feels 0.18, 0.12 and 15 N.
* *Two-way.* The push goes onto the knife as a wrench in every substep of the next control step
  (``engines.feedback``: an EMA of 0.35 on a rising push, capped at 5 m/s a step), so the arm feels
  the cut and a dull blade stalls it; its reaction goes onto the cage's front, or onto the face the
  edge presses before a cut starts. The push is the law's, read off the undented block, not the
  cage's elastic energy gradient.

**Drawing.** The block's eight particles are its corners, written each control step; its handle is a
particle body whose ``surface`` is the cutter's own skin and cut faces, which the studio and the
path tracer both draw. The two pieces stay held where they stand and touch, as a real block on a
board does, so a finished cut shows as the blade standing through the block and as ``pieces``. The
corners do not determine the cut, so each step's drawing is kept under the corners it wrote: the
page's **Export video** replays the block as each step drew it, and a replay in another process draws
none (``EntityCuttable.surface_mesh``).

**Measures.** ``pieces`` counts the cut cage's material pieces and ``cut_fraction`` reads how far the
last cut ran; ``tests/test_cut_tofu`` drives the example's arm panel and reads both.

**Without a GPU.** The cutter is Warp code and runs on Warp's CPU device, one thread, beside the
knife on MuJoCo's C library, so this kind builds and cuts on a CPU (``tests/test_cpu_materials``):
a control step in which the edge advances costs 1 to 2 s there against 7.6 ms on a GPU, and one in
which it does not, 9 to 13 ms. The example's cut through its page averaged 273 ms a control step and
parted the tofu in two.
"""

from __future__ import annotations

from robosimstudio.assetlib.materials import CalibrationSpec, MaterialRecord, register_material, register_material_kind
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.physics import MUJOCO_500HZ, register_domain

from .declare import Cuttable
from .entity import EntityCuttable, EntityKnife
from .measures import TOUCHES
from .spec import (
    CAKE,
    FOAM,
    TOFU,
    CuttableMaterialSpec,
    CuttableSpec,
    KnifeSpec,
    cuttable,
    cuttables,
    knife,
)

__all__ = [
    "CAKE",
    "CUTTING_50HZ",
    "FOAM",
    "TOFU",
    "Cuttable",
    "CuttableMaterialSpec",
    "CuttableSpec",
    "EntityCuttable",
    "EntityKnife",
    "KnifeSpec",
    "cuttable",
    "cuttables",
    "knife",
]

CUTTING_50HZ = MUJOCO_500HZ
"""``MUJOCO_500HZ``: the arm, the knife and the board on one MuJoCo solver at 50 Hz over 500 Hz.

The block needs no solver of Newton's: its cage steps once a control step after the substeps. The
knife has to be on the arm's solver, so the pads hold its handle by friction."""

register_material_kind("cutting", CuttableMaterialSpec)

for _name, _spec, _what in (
    ("tofu", TOFU, "firm tofu"),
    ("sponge_cake", CAKE, "sponge cake"),
    ("pu_foam", FOAM, "flexible polyurethane foam, 30 kg/m^3"),
):
    register_material(
        MaterialRecord(
            name=_name,
            spec=_spec,
            notes=f"A block of {_what} a knife parts; its constants are sourced in spec.py.",
            calibration=CalibrationSpec(
                fabric=_what,
                bench="sourced, not measured here: see the comments on the preset in robosimstudio_materials/cutting/spec.py",
                knob="toughness",
            ),
        )
    )

DOMAIN = register_domain(
    "cutting",
    state=(),
    reads="cutting",
    role="deformables",
    contacts=TOUCHES,
    summary="A block a knife parts: a walk-on-spheres cage, its base held by the board, cut where the edge pushes past a threshold.",
)

register_entity("cuttable")(EntityCuttable)
register_entity("knife")(EntityKnife)
