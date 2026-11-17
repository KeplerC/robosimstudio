# SPDX-License-Identifier: Apache-2.0
"""What this task calls the library's rod, and the one rod it feeds.

The names are the task's and the mechanism is the library's, which is the split
``vabar/tasks/cable_route/rod.py`` argues for and this follows: ``tunnel_cable``
is what *this* suite's documents ask for, and binding it here is what keeps the
library from holding content vocabulary. ``register_entity`` returns the class
unchanged and ``ENTITIES`` compares registrations by (module basename, qualname),
so two tasks binding the same library class under different names is accepted.

The dimensions are this suite's own, and they are chosen against the bore and
the bench rather than inherited:

* **10 mm gauge in a 17 mm bore.** The weaves' own radius, reused because it is
  the one this stack has driven a rod at; the bore is sized to the connector
  rather than to the cable (``plug.py``).
* **0.44 m, and the bench is what caps it.** The rod spawns straight behind the
  entry, level with the bore, and its tail falls to the table during the
  settle. The entry is the box's right end at y -0.20 and the table's edge is
  at -0.65, so 0.41 m of tail is what fits with 50 mm to spare. What the length
  buys is the run: ``layout.py::slack`` needs it to cover the run, the exit
  clearance and a tail a hand can still hold, and 0.44 is what lets the lidded
  channel be 250 mm rather than the conduit's 200.
* **38 capsules.** 11.6 mm of pitch, which must stay above ``2 * radius`` or
  the capsules interpenetrate at rest, and which stays on the weaves' 11.9 mm
  closely enough that the bend-stiffness parity derivation carries over -- it
  scales by the pitch anyway (``layout.py::_bend``).
"""

from __future__ import annotations

from robosimstudio_materials.rod import (
    CableLatchSpec,
    EntityCable,
    RodSpec,
)

__all__ = ["LENGTH", "RADIUS", "SEGMENTS", "CableLatchSpec", "EntityCable", "RodSpec"]

LENGTH = 0.44
RADIUS = 0.005
SEGMENTS = 38
"""The fed rod [m, m, count]. Pitch is ``LENGTH / SEGMENTS`` and must exceed
``2 * RADIUS``; ``tests/test_rgbs_tunnel.py`` is what says so, not a comment."""

# ``tunnel_cable`` is registered by ``plug.py``, not here. The entity this suite
# builds is ``EntityTunnelCable`` -- the library rod with a moulded plug welded
# to its leading capsule -- and the name has to resolve to the thing with the
# connector on it, or the document builds a bare rod and every plug measurement
# in the routine reads a body that does not exist.
