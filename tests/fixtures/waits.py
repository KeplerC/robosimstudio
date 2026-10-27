# SPDX-License-Identifier: Apache-2.0
"""Waiting on another thread without a spin that can hang the tier, and time that a test owns.

``while bus._queue.empty(): pass`` waits forever if the worker dies before it
posts, and a fast tier that hangs is worse than one that fails: nobody reads
its output. :func:`wait_until` gives up after a deadline and says what it was
waiting for.

A test that sleeps 50 ms and then asserts that something did *not* happen
passes whenever the thread under test was starved -- which is when it matters.
Code that reads ``time.monotonic`` can be given a :class:`FakeClock` instead, so
the test moves time and nothing is left to the scheduler.
"""

from __future__ import annotations

import contextlib
import importlib
import time
from collections.abc import Callable, Iterator
from unittest import mock

__all__ = ["FakeClock", "fake_clock", "wait_until"]


def wait_until(
    predicate: Callable[[], object], timeout: float = 5.0, interval: float = 0.001, what: str = "the condition"
) -> object:
    """Poll ``predicate`` until it returns something truthy and return that; fail after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        got = predicate()
        if got:
            return got
        if time.monotonic() >= deadline:
            raise AssertionError(f"waited {timeout} s for {what}, and it never came")
        time.sleep(interval)


class FakeClock:
    """A ``monotonic``/``time``/``sleep`` that move only when told to; ``sleep`` advances it."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)


class _FakeTime:
    """A module's ``time``, with ``monotonic``, ``time``, ``perf_counter`` and ``sleep`` on the fake clock.

    Everything else is the real ``time`` module's, so ``time.strftime`` still works.
    """

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock

    def __getattr__(self, name: str):
        if name in ("monotonic", "time", "perf_counter"):
            return self._clock
        if name == "sleep":
            return self._clock.sleep
        return getattr(time, name)


@contextlib.contextmanager
def fake_clock(*modules: str, start: float = 1000.0) -> Iterator[FakeClock]:
    """Give each named module one shared :class:`FakeClock` in place of its ``time``.

    ``fake_clock("robosimstudio_policy_agent.fleet.barrier")`` replaces that
    module's own binding of ``time`` -- what its ``import time`` made -- so its
    ``time.monotonic()`` and ``time.sleep()`` read and move the fake clock, while
    the ``time`` module itself, and every other module and thread, keep real time.
    A dotted name that is not a module (``"pkg.mod.monotonic"``, for a module
    that did ``from time import monotonic``) is patched with the clock itself,
    or with its ``sleep`` when the name ends in ``sleep``.
    """
    clock = FakeClock(start)
    with contextlib.ExitStack() as stack:
        for target in modules:
            try:
                module = importlib.import_module(target)
            except ImportError:
                module = None
            if module is not None:
                if not hasattr(module, "time"):
                    raise AttributeError(f"{target} has no `time` of its own to replace; name its function instead")
                stack.enter_context(mock.patch.object(module, "time", _FakeTime(clock)))
            else:
                stack.enter_context(mock.patch(target, clock.sleep if target.endswith("sleep") else clock))
        yield clock
