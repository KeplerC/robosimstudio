# SPDX-License-Identifier: Apache-2.0
"""What the three particle kinds share: one spec, one entity, one set of measures, one solver island.

``granular`` (``rss.heap``), ``liquid`` (``rss.liquid``) and ``dough``
(``rss.dough``) are the same thing to Newton's implicit MPM -- particles on a
background grid with a material per particle -- and differ in their presets,
their domain and the projection their arrangement states. So the implementation
is here once and each kind is a short module: a domain, an entity subclass
naming it, a constructor and its presets. This package is not a kind: it
registers no domain and no entity, and the kinds import it (and not each other).

The modules, in reading order:

* ``spec`` -- :class:`ParticleMaterialSpec` (Newton's twelve ``mpm:*`` fields
  and a density) and :class:`ParticleBodySpec` (a box, a cylinder or a mesh,
  filled on a lattice).
* ``presets`` -- the named materials, one material kind ``"granular"`` for all
  three: rice, sand, beans, flour; water, honey, ketchup; dough, plasticine,
  clay, butter.
* ``entity`` -- :class:`EntityParticleBody`: registers the MPM's custom
  attributes, samples and adds the particles, settles, resets the MPM's history
  with the positions, offers its material as a tunable and its particles as a
  point handle with an optional reconstructed surface.
* ``measures`` -- ``fill_fraction``, ``mass_in_region``, ``spill_fraction``,
  ``heap_height`` and ``spread``, on the ``granular`` layout.
* ``physics`` -- :data:`GRANULAR_60HZ`, :data:`LIQUID_120HZ`, :data:`DOUGH_60HZ`:
  ``GENERIC`` with the rigid props on the robot's island and the projection,
  the iterations and the rate stated per kind.
* ``declare`` -- :class:`Heap`, :class:`Liquid`, :class:`Dough` asset records
  for a task document.
"""

from __future__ import annotations

from . import measures as _measures  # noqa: F401 -- registers the atoms and the granular layout
from . import presets as _presets  # noqa: F401 -- registers the material kind and its records
from .declare import Dough, Heap, Liquid, ParticleBody
from .entity import EntityParticleBody, island_solver, resolve_material, sample_positions
from .physics import DOUGH_60HZ, GRANULAR_60HZ, LIQUID_120HZ, with_particles_on
from .presets import BEANS, BUTTER, CLAY, DOUGH, FLOUR, HONEY, KETCHUP, PLASTICINE, RICE, SAND, WATER, presets
from .spec import MPM_KEYS, NEWTON_MPM_STRESS_SCALE, ParticleBodySpec, ParticleMaterialSpec

__all__ = [
    "BEANS",
    "BUTTER",
    "CLAY",
    "DOUGH",
    "DOUGH_60HZ",
    "FLOUR",
    "GRANULAR_60HZ",
    "HONEY",
    "KETCHUP",
    "LIQUID_120HZ",
    "MPM_KEYS",
    "NEWTON_MPM_STRESS_SCALE",
    "PLASTICINE",
    "RICE",
    "SAND",
    "WATER",
    "Dough",
    "EntityParticleBody",
    "Heap",
    "Liquid",
    "ParticleBody",
    "ParticleBodySpec",
    "ParticleMaterialSpec",
    "island_solver",
    "presets",
    "resolve_material",
    "sample_positions",
    "with_particles_on",
]
