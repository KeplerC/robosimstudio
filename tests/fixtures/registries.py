# SPDX-License-Identifier: Apache-2.0
"""Register something for one test and take it back, whatever the test does.

Ten modules registered a probe domain, coupler or texture and popped it from the
registry's private ``_items`` afterwards -- by hand, sometimes outside a
``finally``, so a test that failed half-way left its probe registered for every
module the same process ran after it. :func:`restored` puts the registry back
to exactly what it held, names and origins, on the way out.

It loads the registry first. A registry fills itself lazily, on the first
lookup, by importing the modules that register into it; a snapshot taken before
that would put back the empty registry afterwards while it still believed
itself loaded, and every later lookup in the process would miss. A populator
added inside the block (``add_discover``) is taken back with the rest. Watchers
(``registry.watch``) are not told of the restore: a test that watches a registry
undoes its own watch.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

__all__ = ["registered", "restored"]


@contextlib.contextmanager
def restored(*registries: Any) -> Iterator[None]:
    """Snapshot each registry's entries and put them back on exit, removing anything added."""
    for registry in registries:
        registry.load()
    saved = [
        (registry, dict(registry._items), dict(registry._origins), list(registry._discover), registry._loaded)
        for registry in registries
    ]
    try:
        yield
    finally:
        for registry, items, origins, discover, loaded in saved:
            registry._items.clear()
            registry._items.update(items)
            registry._origins.clear()
            registry._origins.update(origins)
            registry._discover[:] = discover
            registry._loaded = loaded


@contextlib.contextmanager
def registered(registry: Any, name: str, value: Any, *, origin: str = "test") -> Iterator[Any]:
    """``registry.register(name, value)`` for the duration of the block; yields what was registered."""
    with restored(registry):
        yield registry.register(name, value, origin=origin)
