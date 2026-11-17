# SPDX-License-Identifier: Apache-2.0
"""The privileged tools this suite's documents name.

Three registrars, one per family that has one, re-exported from
``solutions/`` -- see that package's docstring for why the indirection exists
(a document may not name a dotted module in its own task).

Imported lazily inside each function rather than at module scope, and that is
not tidiness: resolving ``peel``'s two tools should not import ``polish``'s,
which pulls in ``warp`` and builds a grip-force controller. A document that asks
for one family's tools gets one family's imports.
"""

from __future__ import annotations

__all__ = [
    "register_polish_tools",
    "register_surface_prep_tools",
    "register_tape_peel_tools",
]


def register_tape_peel_tools(registry, env, bus) -> None:
    """``sim.get_tab`` and ``sim.release_anchor``, for the four peel variants."""
    from .routine_peel import register_tape_peel_tools as _register

    _register(registry, env, bus)


def register_polish_tools(registry, env, bus) -> None:
    """The strip-carry and press tools, for ``strip``."""
    from .routine_polish import register_polish_tools as _register

    _register(registry, env, bus)


def register_surface_prep_tools(registry, env, bus) -> None:
    """The bar-hold and wipe tools, for ``wipe``."""
    from .routine_prep import register_surface_prep_tools as _register

    _register(registry, env, bus)
