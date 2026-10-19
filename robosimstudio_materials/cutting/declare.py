# SPDX-License-Identifier: Apache-2.0
"""A cuttable block and its knife as asset records, for a task document to place by type."""

from __future__ import annotations

from robosimstudio.assetlib.record import AssetRecord, SpecFactory
from robosimstudio.assetlib.sim import SimRequirements, contacts_with

from .measures import TOUCHES
from .spec import CuttableSpec, KnifeSpec

__all__ = ["Cuttable"]


class Cuttable:
    """Constructors for the asset records of a block and of the knife that cuts it."""

    @staticmethod
    def from_spec(
        spec: CuttableSpec, *, name: str, summary: str = "", tags: tuple[str, ...] = ("food", "cuttable")
    ) -> AssetRecord:
        """The block ``rss.cuttable`` returned, as an :class:`AssetRecord`. Its knife must be placed too.

        Raises:
            TypeError: For a spec that is not a :class:`CuttableSpec`.
        """
        if not isinstance(spec, CuttableSpec):
            raise TypeError(f"Cuttable.from_spec takes a CuttableSpec, got {type(spec).__name__}")
        x, y, z = (1e3 * float(v) for v in spec.size)
        return AssetRecord(
            name=name,
            summary=summary or f"a {x:.0f} x {y:.0f} x {z:.0f} mm block a knife can cut",
            tags=tuple(tags),
            sim=SimRequirements(domain="cutting", contacts=contacts_with(*TOUCHES)),
            build=SpecFactory(spec),
        )

    @staticmethod
    def knife(spec: KnifeSpec, *, name: str, summary: str = "") -> AssetRecord:
        """The knife ``rss.knife`` returned, as a rigid body record carrying its ``blade`` feature."""
        from robosimstudio.domains.rigid import Body

        if not isinstance(spec, KnifeSpec):
            raise TypeError(f"Cuttable.knife takes a KnifeSpec, got {type(spec).__name__}")
        return Body.from_spec(
            spec,
            name=name,
            summary=summary or f"a knife with a {1e3 * spec.blade_length:.0f} mm blade",
            tags=("knife", "tool"),
            features=(spec.blade_feature(),),
        )
