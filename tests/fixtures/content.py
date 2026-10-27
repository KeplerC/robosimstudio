# SPDX-License-Identifier: Apache-2.0
"""Whether any content registered an asset record here, for a test that has to say "if there is any".

The library declares no :class:`~robosimstudio.assetlib.AssetRecord` of its own.
The records are content's: a task collection declares each beside its meshes,
and they are loaded when a root holds a collection. A release may ship the
library with no collection at all, and then the catalog is not small, it is
empty -- ``list_assets()`` is ``[]``, ``find_asset`` answers ``None`` for every
name, and ``get_asset`` raises a ``KeyError`` that lists nothing.

Most tests about an asset belong to the collection that declares it and are
withheld with it. A few are about the catalog the library keeps on both sides of
that line -- that it validates what it is given, that one reader survives every
record -- and those have to state both answers rather than assume a corpus. This
is the one question they ask.

Asked of the registry rather than of the filesystem, so a collection on a root
outside the checkout counts and a directory nothing loaded does not.

A sweep over the corpus does not skip on a tree that has none: :func:`corpus`
hands it the registry's names, or, with nothing registered, one synthetic record
registered for the block -- so the catalog's readers (``problems``,
``asset_to_dict``, the interface parser) run on every tree, not only on one that
ships a collection.
"""

from __future__ import annotations

import contextlib
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path

from robosimstudio.assetlib import AssetRecord, SimRequirements, list_assets
from robosimstudio.assetlib.semantics import FeatureSpec, StateSpec
from robosimstudio.assetlib.semantics.published import PublishedState

__all__ = ["HAS_CONTENT_ASSETS", "corpus", "needs_content_assets", "synthetic_record"]

HAS_CONTENT_ASSETS = bool(list_assets())
"""Whether the asset registry holds anything in this checkout."""

needs_content_assets = unittest.skipUnless(HAS_CONTENT_ASSETS, "no content registered an asset record in this tree")
"""Decorator for a test or a class that sweeps the corpus or reaches for one asset by name."""


SYNTHETIC = "probe_towel"
"""The synthetic record's name: a fixture name no collection ships."""


def synthetic_record(home: Path | None = None) -> AssetRecord:
    """A well-formed record with one of every part the catalog form carries.

    A cloth with a ``set`` feature, a state about that feature with its
    provenance, a published quantity, tags, and a mesh file declared beside
    ``home`` -- so a sweep over it reads the same branches a sweep over a real
    collection does.
    """
    return AssetRecord(
        name=SYNTHETIC,
        summary="A towel the catalog sweeps read when no content declared one.",
        tags=("towel",),
        features=(FeatureSpec(name="hem", kind="set", tier="recipe", payload={"select": {"op": "half", "axis": "y"}}),),
        states=(
            StateSpec(
                name="folded",
                measure="fold_extent",
                op="<=",
                value=0.6,
                of="hem",
                summary="folded in half",
                provenance="a synthetic record: the threshold is the fixture's",
            ),
        ),
        published=(PublishedState(name="sag", type="float", summary="how far the middle hangs", unit="m"),),
        sim=SimRequirements(domain="cloth"),
        files={"mesh": "towel.obj"},
        home=home,
        provenance="tests/fixtures/content.py",
    )


@contextlib.contextmanager
def corpus() -> Iterator[list[str]]:
    """The asset names a sweep walks: the registry's, or one synthetic record registered for the block.

    The synthetic record goes through :func:`register_asset`, so its declared
    file is resolved beside its home the way a collection's is, and the
    registry is put back on the way out.
    """
    if HAS_CONTENT_ASSETS:
        yield list_assets()
        return
    from robosimstudio.assetlib.registry import ASSETS, register_asset
    from tests.fixtures.registries import restored

    with tempfile.TemporaryDirectory() as home, restored(ASSETS):
        (Path(home) / "towel.obj").write_text("")
        register_asset(synthetic_record(Path(home)))
        yield list_assets()
