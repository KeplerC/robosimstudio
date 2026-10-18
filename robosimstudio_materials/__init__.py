# SPDX-License-Identifier: Apache-2.0
"""The kinds of deformable thing the library ships: cloth, soft bodies, the rope, paper, bags, the brush and the particle materials.

A **kind** is one physical kind of thing, carried as one bundle: the spec that
says what one is, the entity that builds it, the measures that read it, its
materials and, where a foreign engine serves it, the variant that runs there.
Each kind here is one module or one package -- ``cloth.py``, ``soft.py``,
``paper.py``, ``bag.py``, ``brush.py``, ``granular.py``, ``liquid.py`` and ``dough.py``
are read top to bottom, ``rod/`` is a package -- and the base kinds stand alone:
none of ``cloth``, ``soft`` and ``rod`` imports another. ``paper``, ``bag`` and
``brush`` are built on another kind and say so: a sheet of paper is a cloth whose
bending hinges yield and a bag is a cloth sewn into a pocket, so each extends
``cloth.py``, and a brush's bristles are the rope's capsule chains on a handle, so
it builds them with ``rod``'s own builder; a tree that ships paper or the bag ships
cloth, and one that ships the brush ships the rope. The
three particle kinds share one implementation in the ``mpm/`` package, which is
not itself a kind: it registers no domain and no entity.

The kinds here: ``cloth`` (``rss.sheet``), ``soft`` (``rss.sponge``), ``rod``
(``rss.rod``), ``paper`` (``rss.paper``), ``bag`` (``rss.bag``, a SuperDex shell
that collides with itself), ``brush`` (``rss.brush``), and the three particle kinds on
Newton's implicit MPM -- ``granular`` (``rss.heap``), ``liquid`` (``rss.liquid``)
and ``dough`` (``rss.dough``). The kinds the library itself is made of --
``rigid``, ``static``, ``articulated`` -- stay in :mod:`robosimstudio.domains`,
because the composer, the robots and the cells are built from them.

This package stands **beside** the library and imports only :mod:`robosimstudio`
-- never the studio, the benchmark or the agents -- and the library never
imports it: it names ``robosimstudio_materials`` as a string in two places only,
the registries' scan list and the flat-name table (``robosimstudio._PACKS``).
That is what lets a tree leave this package out and still build a rigid scene,
with ``rss.sheet`` saying what is missing rather than failing three layers down.

A kind is found by scanning, not by import: the ``DOMAINS`` registry and the
registries beside it import every module in this package on the first lookup
(:meth:`robosimstudio.registry.Registry.load`), and a kind registers itself as
it is imported. Adding a shipped kind is adding a module here; a kind of your
own registers the same way from a package on ``ROBOSIMSTUDIO_DOMAIN_ROOTS``, as
``examples/04_extending`` shows.
"""

from __future__ import annotations

__all__: list[str] = []
