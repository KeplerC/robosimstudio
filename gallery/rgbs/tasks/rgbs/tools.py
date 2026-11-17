# SPDX-License-Identifier: Apache-2.0
"""The privileged tools this suite's documents name.

Two registrars, one per family that has one, each a re-export of its routine
module's: a document names ``"tools:X"``, and a dot in that module part would
make it an absolute import rather than this task's own module.

Imported lazily inside each function rather than at module scope, so that
resolving ``wipe``'s tools does not import ``polish``'s, which pulls in
``warp`` and builds a grip-force controller. A document that asks for one
family's tools gets one family's imports.
"""

from __future__ import annotations

__all__ = [
    "register_polish_tools",
    "register_surface_prep_tools",
]


def register_polish_tools(registry, env, bus) -> None:
    """The strip-carry and press tools, for ``strip``."""
    from .routine_polish import register_polish_tools as _register

    _register(registry, env, bus)


def register_surface_prep_tools(registry, env, bus) -> None:
    """The bar-hold and wipe tools, for ``wipe``."""
    from .routine_prep import register_surface_prep_tools as _register

    _register(registry, env, bus)
