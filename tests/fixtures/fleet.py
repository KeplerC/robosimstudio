# SPDX-License-Identifier: Apache-2.0
"""The fleet side of a batched policy, without a fleet.

:class:`GivesUpOnceStarted` is for the one race every "a late answer" test has:
the asker must give up *while* its request is being served. With a real
``Event.wait(0.1)`` that holds only when the serving thread gets there inside
the 100 ms, which a loaded host does not promise, and a test that lost the race
failed on ``started.is_set()`` rather than on anything it was about. Handed to
the request as its ``done`` event, the asker's timed wait times out exactly
once the request is running, and never before.
"""

from __future__ import annotations

import threading

__all__ = ["GivesUpOnceStarted"]


class GivesUpOnceStarted(threading.Event):
    """A ``done`` event whose *timed* wait returns ``False`` only once ``started`` is set.

    An untimed ``wait()`` is the real one: that is the asker waiting for an
    answer it knows is coming, which is not the race this removes.
    """

    def __init__(self, started: threading.Event, *, patience: float = 5.0) -> None:
        super().__init__()
        self.started = started
        self.patience = patience

    def wait(self, timeout: float | None = None) -> bool:
        if timeout is None:
            return super().wait()
        if not self.started.wait(self.patience):
            raise AssertionError(f"the request was never served in {self.patience} s")
        return super().wait(0)
