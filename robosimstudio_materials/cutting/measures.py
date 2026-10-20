# SPDX-License-Identifier: Apache-2.0
"""The cutting kind's measures: how many pieces the block is in, and how far the last cut has run.

Both read the block's own cutter. ``pieces`` is the number of material components of the cage
(``cagecut.geometry.snapshot.piece_labels``): two once a cut has gone right through, and a slit that
stops short of the far side leaves one. ``cut_fraction`` is the front's run through the block along
the cut's sweep, from where the edge met it to the far side.
"""

from __future__ import annotations

from robosimstudio.assetlib.semantics.atoms import ArgSpec, AtomCalibration, AtomSpec, register_atom
from robosimstudio.assetlib.semantics.handlers import handler

__all__ = ["TOUCHES"]

TOUCHES: tuple[str, ...] = ()
"""What a block contacts in Newton: nothing. It meets the knife through the edge law alone."""


def _block(snapshot, name: str):
    entities = getattr(snapshot, "entities", None) or {}
    entity = entities.get(name) if hasattr(entities, "get") else None
    if entity is None or not hasattr(entity, "pieces"):
        raise KeyError(f"{name!r} is not a cuttable block the scene carries an entity for")
    return entity


_BENCH = "examples/03_materials/10_cutting driven through its right arm panel, tests/test_cut_tofu.py"

register_atom(
    AtomSpec(
        name="pieces",
        summary="How many pieces a cuttable block is in: the material components of its cut cage.",
        args=(ArgSpec(name="block", substrates=("cutting",), reads="handle"),),
        directions=("check", "observe"),
        returns=("count",),
        cost="cheap",
        reads_scene=True,
        provenance=(
            "The vendored walk-on-spheres cutter's own piece labels (cagecut.geometry.snapshot.piece_labels): "
            "connected components of the cage, a cavity shell counted with the solid round it."
        ),
        calibration=AtomCalibration(
            positives=("examples/03_materials/10_cutting, the knife taken and pushed down through the tofu: 2.",),
            negatives=(
                "The same drive with the hand held 15 mm higher, the edge 2.1 mm over the board: 1. A blade a "
                "thousand times duller pushed the same way: 1.",
            ),
            gate=None,
            bench=_BENCH,
            notes="A count; a goal states the pieces it wants.",
        ),
    )
)

register_atom(
    AtomSpec(
        name="cut_fraction",
        summary="How far the last cut has run through a cuttable block, from where the edge met it to its far side.",
        args=(ArgSpec(name="block", substrates=("cutting",), reads="handle"),),
        directions=("check", "observe"),
        returns=("fraction", "depth"),
        cost="cheap",
        reads_scene=True,
        provenance="The cutter's sweep time at the front, from first contact to the cage's farthest corner.",
        calibration=AtomCalibration(
            positives=("examples/03_materials/10_cutting cut through: 1.0.",),
            negatives=("The knife held over the block: 0.",),
            gate=None,
            bench=_BENCH,
            notes="Depth in metres from where the edge met the block.",
        ),
    )
)


@handler("pieces", substrates=("cutting",), summary="The block's piece count.")
def _pieces(snapshot, block: str) -> dict[str, float]:
    count = float(_block(snapshot, block).pieces())
    return {"count": count, f"{block}/pieces": count}


@handler("cut_fraction", substrates=("cutting",), summary="The last cut's run through the block, and its depth [m].")
def _cut_fraction(snapshot, block: str) -> dict[str, float]:
    entity = _block(snapshot, block)
    fraction = entity.cut_fraction()
    return {"fraction": fraction, "depth": entity.cut_depth(), f"{block}/cut_fraction": fraction}
