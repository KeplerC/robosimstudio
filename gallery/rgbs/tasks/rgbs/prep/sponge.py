# SPDX-License-Identifier: Apache-2.0
"""The abrasive: a kitchen sponge, a block of open-cell foam simulated by SuperDex.

**Foam rather than a pad, because the geometry is the task.** A rigid pad
pressed onto a round bar touches it along a line; foam wraps it and touches an
arc, and how much of an arc is a function of how hard you press. Measured on
the 22 mm bar this suite calibrated its removal law on
(``rgbs/scripts/rgbs_prep_bench.py``), the contact spans 43 deg of the
circumference at first touch, 87 deg at 3 mm of overdrive and 126 deg at 8 mm.

**Its own record, under its own name.** The redundancy rule here is that a
second registration under a taken name is accepted when it says the *same
thing*; this sponge is a different size and lattice, so it says a different
thing and takes a different name. A task may not import a sibling's
declaration either way.

**Sized off the clip, diced for the readback.** 50 x 75 x 30 mm is the block
in the overhead inset -- three squares by two -- with its long axis along the
bar. The lattice was chosen from a measured sweep on the earlier 70 x 45 mm
block, and the finding carries: force and wrap angle are converged from
3.75 mm nodes to 6 mm nodes (1.82, 1.87, 1.91 N at 4 mm), and only the cost of
reading the contact patch back moves (5.6, 2.3, 1.8 ms), because SuperDex hands
per-sample contacts over as Python objects. ``(9, 13, 6)`` is ~5.5 mm nodes,
980 particles, and still resolves the field's 2.5 mm cells: contact samples
are quadrature points on surface triangles, about three per triangle, so they
run ~2 mm apart even at 5 mm nodes.
"""

from __future__ import annotations

import numpy as np

from robosimstudio.assetlib import AssetRecord, register_asset
from robosimstudio.assetlib.sim import SimRequirements, contacts_with

from ..look import SPONGE_WHITE, WHITE_FOAM
from .layout import LAYOUT, Layout

__all__ = ["CELLS", "PREP_SPONGE", "SIZE", "SPONGE_AT", "prep_sponge", "wipe_sponge"]

SIZE = LAYOUT.sponge_size
"""Extents [m]: 75 mm ALONG the bar, 50 mm across, 30 mm thick -- the layout's."""

CELLS = LAYOUT.sponge_cells
"""Lattice divisions -- 980 particles at ~5.5 mm spacing. See the module docstring."""

SURFACE_CLEARANCE = 0.001
"""Built a millimetre above whatever it stands on and settled out of it."""

SPONGE_AT = LAYOUT.sponge_at
"""Where the sponge starts [m, surface frame]: its bottom-face centre on the mat,
past the near block, on the far side of the bar's line.

The layout's arithmetic: 70 mm of clear mat from the block's outer face, 35 mm
beyond the bar's axis. Where the *hand* can reach it is a separate fact, and a
good one: swept with the TCP as the grasp point, the work arm solves a top-down
grasp to under a tenth of a millimetre everywhere on this mat, and the
100 mm standoff over this point solves to 0.0 mm. That was not true of every
candidate -- the standoff over world (0.685, -0.14), where an earlier layout put
the block, missed by 34.9 mm and the plan gate refused the pick.

No stand. An earlier version put it on one, on the strength of a reach map that
turned out to be measuring the wrong thing -- the same arm picks a 30 mm rigid
cube off this bench and lifts it 80 mm, so the hand reaches the bench perfectly
well. The stand only gave the sponge somewhere to be knocked off."""


_SPONGE_CONTACTS = SimRequirements(
    domain="soft",
    # `objects` and `fixtures` are the ones that matter and are easy to leave
    # off: they put the BAR and the BLOCKS into the sponge's mirror set, and
    # without them the foam is co-simulated against a world containing the
    # bench and nothing else -- every contact query returns table samples,
    # every press force reads zero, and nothing raises.
    contacts=contacts_with("surface", "hands", "fixtures", "objects"),
)


def _lower_face(rest: np.ndarray) -> np.ndarray:
    """The wiping face: the lowest layer of nodes as the sponge sits at rest."""
    return rest[:, 2] < rest[:, 2].min() + 0.6 * (SIZE[2] / CELLS[2])


def _waist(rest: np.ndarray) -> np.ndarray:
    """The mid-height band the jaws close on, and what a grasp falls back to pinning."""
    low, high = rest[:, 2].min(), rest[:, 2].max()
    mid, band = 0.5 * (low + high), 0.30 * (high - low)
    return np.abs(rest[:, 2] - mid) <= band


def prep_sponge(*, at: tuple[float, float, float]):
    """The sponge as this task stands it up, at ``at`` in the surface frame.

    ``settle_frames`` is the shipped default: the block is built a millimetre
    high and drops onto the mat, and the drape it lands in is the start state.

    The domain is imported here rather than at module scope, which is what
    every other consumer of it in this tree does: ``robosimstudio.domains.soft``
    is a domain *pack*, shipped from the repo's top-level ``domains/`` and
    installed into ``robosimstudio/_domain_packs``, so there is no
    ``robosimstudio/domains/soft`` on disk for an import sorter to resolve and
    the module gets filed as third-party beside numpy.
    """
    from robosimstudio.domains.soft import SOFT_FOAM, SoftSpec

    x, y, z = (float(v) for v in at)
    return SoftSpec(
        name="sponge",
        size=SIZE,
        cells=CELLS,
        pose=(x, y, z + SURFACE_CLEARANCE),
        material=SOFT_FOAM,
        particle_sets=(("face", _lower_face), ("waist", _waist)),
        settle_frames=30,
        color=SPONGE_WHITE,
        finish=WHITE_FOAM,
        sim=_SPONGE_CONTACTS,
    )


def wipe_sponge(layout: Layout = LAYOUT):
    """The sponge where ``wipe`` starts it: on the mat, past the near block.

    The zero-argument form of :func:`prep_sponge`, and what the record below
    names as its ``build`` -- a document types ``sponge - prep_sponge`` and gets
    this back. The parameterised factory stays for a bench that wants the same
    foam somewhere else.
    """
    return prep_sponge(at=layout.sponge_at)


# The record last, because it names the factory that places the sponge.
PREP_SPONGE = register_asset(
    AssetRecord(
        name="prep_sponge",
        summary=(
            "A 75 x 50 x 30 mm kitchen sponge, simulated as a SuperDex FEM body: the abrasive a "
            "surface-prep wipe is made with, which conforms around a round bar instead of touching "
            "it on a line."
        ),
        tags=("sponge", "foam", "abrasive", "deformable"),
        features=(),
        sim=_SPONGE_CONTACTS,
        build=wipe_sponge,
    )
)
