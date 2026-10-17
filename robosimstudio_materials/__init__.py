# SPDX-License-Identifier: Apache-2.0
"""The kinds of deformable thing the library ships: cloth, soft bodies, the rope, and the particle materials.

A **kind** is one physical kind of thing, carried as one bundle: the spec that
says what one is, the entity that builds it, the measures that read it, its
materials and, where a foreign engine serves it, the variant that runs there.
Each kind here is one module or one package -- ``cloth.py`` and ``soft.py`` are
read top to bottom, ``rod/`` is a package -- and each stands alone: no kind
imports another.

The kinds here: ``cloth`` (``rss.sheet``), ``soft`` (``rss.sponge``), ``rod``
(``rss.rod``), and the three particle kinds on Newton's implicit MPM --
``granular`` (``rss.heap``), ``liquid`` (``rss.liquid``) and ``dough``
(``rss.dough``) -- which share one implementation in the ``mpm/`` package
(not itself a kind: it registers no domain and no entity). The kinds the
library itself is made of -- ``rigid``, ``static``, ``articulated`` -- stay in
:mod:`robosimstudio.domains`, because the composer, the robots and the cells
are built from them.

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
