# SPDX-License-Identifier: Apache-2.0
"""Declaring a particle body for a task document: the spec you already built, as an asset record.

A document names a *type* and places one: ``(:objects rice - rice_heap)``. This
turns the spec a constructor returned into the :class:`AssetRecord` that type
resolves to, the way :class:`~robosimstudio.domains.rigid.declare.Body` does for
a cube::

    register_asset(Heap.from_spec(rss.heap(name="rice", size=(0.12, 0.12, 0.03)), name="rice_heap"))

The record's domain is read off the spec's entity -- a ``heap`` is granular, a
``liquid`` liquid, a ``dough`` dough -- and each named constructor refuses a
spec of another kind, so a document that says ``Liquid`` gets a liquid. Nothing
here registers; the task does, under its own name.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from robosimstudio.assetlib.record import AssetRecord, SpecFactory
from robosimstudio.assetlib.semantics.states import StateSpec
from robosimstudio.assetlib.sim import SimRequirements, contacts_with

from .spec import ParticleBodySpec

__all__ = ["Dough", "Heap", "Liquid", "ParticleBody"]

TOUCHES = ("surface", "fixtures", "objects", "hands")
"""What a particle body contacts unless it says otherwise: the bench, a tray, a tool, the fingers."""


class ParticleBody:
    """Constructors for a particle body's asset record, from the spec that places one."""

    DOMAIN: ClassVar[str] = ""
    """The domain a subclass pins, or empty to take the spec's entity's."""

    @classmethod
    def from_spec(
        cls,
        spec: ParticleBodySpec,
        *,
        name: str,
        summary: str = "",
        tags: Sequence[str] = (),
        touches: Sequence[str] = TOUCHES,
        states: Sequence[StateSpec] = (),
        notes: str = "",
    ) -> AssetRecord:
        """A particle body that is a constructor's spec, as an :class:`AssetRecord`.

        Args:
            spec: What ``rss.heap``, ``rss.liquid`` or ``rss.dough`` returned.
                Its ``at`` is where the body starts; the document's object name
                replaces its ``name``.
            name: What a task document calls the type.
            touches: Roles this body contacts, passed to :func:`contacts_with`.

        Raises:
            TypeError: For a spec that is not a :class:`ParticleBodySpec`, or
                whose entity is of another kind than this constructor's.
        """
        if not isinstance(spec, ParticleBodySpec):
            raise TypeError(f"{cls.__name__}.from_spec takes a ParticleBodySpec, got {type(spec).__name__}")
        from robosimstudio.scene.registry import get_entity

        domain = str(getattr(get_entity(spec.entity), "domain", ""))
        if cls.DOMAIN and domain != cls.DOMAIN:
            raise TypeError(
                f"{cls.__name__}.from_spec: {spec.name!r} is built by {spec.entity!r}, a {domain} body, "
                f"not a {cls.DOMAIN} one"
            )
        # A mesh is counted by its fill, at build, so its summary states no count.
        count = spec.particle_count
        body = f"a {domain} body of {spec.shape} shape" + ("" if count is None else f", {count} particles")
        return AssetRecord(
            name=name,
            summary=summary or body,
            tags=tuple(tags) or (domain,),
            states=tuple(states),
            sim=SimRequirements(domain=domain, contacts=contacts_with(*touches), notes=notes),
            build=SpecFactory(spec),
        )


class Heap(ParticleBody):
    """A granular body's record: ``Heap.from_spec(rss.heap(...), name=...)``."""

    DOMAIN = "granular"


class Liquid(ParticleBody):
    """A liquid body's record: ``Liquid.from_spec(rss.liquid(...), name=...)``."""

    DOMAIN = "liquid"


class Dough(ParticleBody):
    """A plastic solid's record: ``Dough.from_spec(rss.dough(...), name=...)``."""

    DOMAIN = "dough"
