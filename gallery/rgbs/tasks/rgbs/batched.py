# SPDX-License-Identifier: Apache-2.0
"""One device read per instant, however many worlds ask for it.

Every field in this suite -- the taint on the bar, the marks on the strip, the
tape across the carton's seam, the bonds under the film -- is one Warp array
holding a block per world, and every one of them is read by
``published_states``, which a wave calls **once per world per control step**.

A slice of a Warp array is not a partial transfer. ``field.numpy()[a:b]`` copies
the whole array to the host and then throws away the part the caller did not
want, so a per-world read is ``W`` copies of the same bytes and ``W`` stream
syncs every control step -- and the syncs are the expensive half, because they
serialise against the wave that is trying to step. At three worlds that is
already three times what it should be; at twelve it is the run.

So the read is keyed on the *instant* rather than on the world. Every world of a
wave is stepped on one clock and sees one ``(step, sim_time)``, which is exactly
the grain a fleet-wide array changes at: the first world to ask pays the copy
and the rest are handed the same rows.

**That key is right here and wrong one layer up**, which is worth stating
because the two caches sit in the same methods. A *published state* is per world
and must be keyed on the scoring context, which a wave builds one of per world;
``EntityFilm`` was keyed on ``(step, sim_time)`` and served world 0's peel to
every other world for it. The array underneath is not per world, and is.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["HostField", "instant"]


class HostField:
    """A device array brought to the host once per instant.

    Args:
        read: Callable returning the whole array on the host --
            ``some_warp_array.numpy``, bound late so the array can be
            reallocated at bind without this being rebuilt.

    A key of ``None`` always re-reads, which is what a tool asking outside the
    scoring path wants: it has no instant to name and no other caller to share
    with.
    """

    __slots__ = ("_key", "_read", "_rows")

    def __init__(self, read) -> None:
        self._read = read
        self._key: Any = object()
        self._rows: np.ndarray | None = None

    def rows(self, at: Any = None) -> np.ndarray:
        """The whole array on the host, copied only if ``at`` is new."""
        if at is None or self._rows is None or at != self._key:
            self._rows = self._read()
            self._key = at
        return self._rows

    def invalidate(self) -> None:
        """Forget the copy. A reset writes the device array behind our back."""
        self._rows = None
        self._key = object()


def instant(context: Any) -> tuple[int, float] | None:
    """The scoring instant a context names, or ``None`` when it names none.

    ``(step, sim_time)`` and not the context object: the object is per world and
    the array these key is not. See the module docstring.
    """
    step = getattr(context, "step", None)
    if step is None:
        return None
    return int(step), float(getattr(context, "sim_time", 0.0))
