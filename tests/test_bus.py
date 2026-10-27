# SPDX-License-Identifier: Apache-2.0
"""The sim bus: a worker's request is serviced, withdrawn or released, and says which.

When an episode ends, ``SolutionGraph`` closes the bus, and ``SimBus.close``
releases every in-flight worker with an error. The graph runtime turns that into
a node failure and routes it to ``on_error`` -- so a perfectly healthy episode
leaves a node error in its trace. ``tool_hanging``'s ``place_sg.hang`` does this
on runs that score 1.00 and terminate as successes, because the evaluator
declares the hang done while the node is still driving the withdrawal.

That is unavoidable at the runtime level: the bus cannot finish a request whose
simulator has stopped. What is avoidable is being unable to *tell* it apart from
a real fault, which is what :class:`BusClosed` is for -- a timeout means
something was too slow, a close means the episode simply finished. Anything
scanning traces for real failures needs the distinction, and matching on a
message string is not a way to have it.

A request its caller gave up on is withdrawn, never run behind its back, and an
interrupt raised on the owner's thread is the owner's, not the worker's.
"""

from __future__ import annotations

import threading
import unittest
from unittest import mock

from robosimstudio.core import bus as bus_module
from robosimstudio.core.bus import BusClosed, BusTimeout, SimBus
from tests.fixtures.waits import wait_until

TIER = "fast"
"""No GPU: the bus is plain threading."""


class TestClosingIsNotTimingOut(unittest.TestCase):
    def test_a_call_after_close_says_the_episode_ended(self) -> None:
        bus = SimBus()
        bus.close()
        with self.assertRaises(BusClosed) as caught:
            bus.call(lambda: None)
        self.assertIn("episode ended", str(caught.exception))

    def test_waiting_for_a_step_after_close_says_the_same(self) -> None:
        bus = SimBus()
        bus.close()
        with self.assertRaises(BusClosed):
            bus.wait_for_step(after=0)

    def test_an_in_flight_request_is_released_as_closed(self) -> None:
        """The case that actually happens: the episode ends mid-node."""
        bus = SimBus(timeout=5.0)
        bus.owner_thread = -1  # never this thread, so the call really queues
        caught: list[BaseException] = []

        def worker() -> None:
            try:
                bus.call(lambda: None)
            except BaseException as exc:
                caught.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        wait_until(lambda: not bus._queue.empty(), what="the worker's request on the queue")
        bus.close()
        thread.join(timeout=5.0)
        self.assertEqual(len(caught), 1)
        self.assertIsInstance(caught[0], BusClosed)

    def test_a_real_timeout_is_not_a_close(self) -> None:
        """The other direction, which is what makes the type worth having."""
        bus = SimBus(timeout=0.01)
        bus.owner_thread = -1
        with self.assertRaises(BusTimeout) as caught:
            bus.call(lambda: None)
        self.assertNotIsInstance(caught.exception, BusClosed)
        self.assertIn("not serviced", str(caught.exception))

    def test_existing_handlers_still_catch_it(self) -> None:
        """A subclass rather than a sibling, so no caller has to change."""
        bus = SimBus()
        bus.close()
        with self.assertRaises(BusTimeout):
            bus.call(lambda: None)


class CallWithin(unittest.TestCase):
    def test_a_call_its_caller_gave_up_on_never_runs(self) -> None:
        bus, ran = SimBus(timeout=30.0), []
        bus.owner_thread = -1  # an owner that is not this thread, and is not pumping
        with self.assertRaises(BusTimeout):
            bus.call_starting_within(0.1, ran.append, "late")
        bus.owner_thread = None
        bus.pump()
        self.assertEqual(ran, [], "withdrawn when its caller stopped waiting, not run behind its back")

    def test_a_call_started_in_time_runs_to_its_end_and_answers(self) -> None:
        """The caller's window closes while the call is running, by construction rather than by a race.

        The request's ``done`` is one whose timed wait -- the caller's window --
        ends only once a pump has started the call, and the call itself returns
        only once its caller has stopped timing it. Nothing here depends on how
        fast either thread is scheduled.
        """
        running, untimed = threading.Event(), threading.Event()

        class _WindowClosesWhileRunning(threading.Event):
            def wait(self, timeout: float | None = None) -> bool:
                if timeout is None:
                    untimed.set()
                    return super().wait()
                running.wait(5.0)
                return self.is_set()

        class _Request(bus_module._Request):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.done = _WindowClosesWhileRunning()

        def slow() -> str:
            running.set()
            wait_until(untimed.is_set, what="the caller to stop timing the call")
            return "done"

        bus = SimBus(timeout=30.0)
        bus.owner_thread = -1
        result: list[str] = []
        with mock.patch.object(bus_module, "_Request", _Request):
            worker = threading.Thread(target=lambda: result.append(bus.call_starting_within(0.1, slow)))
            worker.start()
            wait_until(lambda: not bus._queue.empty(), what="the worker's request on the queue")
            bus.owner_thread = None
            bus.pump()  # taken in time, and runs past the window
            worker.join(5)
        self.assertEqual(result, ["done"], "never a timeout for something that ran")


class TestABusRequestThatTimedOutNeverRuns(unittest.TestCase):
    def test_a_late_pump_skips_it(self) -> None:
        from robosimstudio.core.bus import BusTimeout, SimBus

        bus = SimBus(timeout=0.05)
        bus.owner_thread = -1  # never this thread, so the call really queues
        ran: list[int] = []
        with self.assertRaises(BusTimeout):
            bus.call(lambda: ran.append(1))
        self.assertEqual(bus.pump(), 0, "a withdrawn request is not serviced")
        self.assertEqual(ran, [], "the caller was told it timed out, so it must not run behind its back")

    def test_an_interrupt_on_the_owner_propagates_and_releases_the_worker(self) -> None:
        from robosimstudio.core.bus import BusClosed, SimBus

        bus = SimBus(timeout=5.0)
        bus.owner_thread = -1
        caught: list[BaseException] = []

        def interrupt() -> None:
            raise KeyboardInterrupt

        def worker() -> None:
            try:
                bus.call(interrupt)
            except BaseException as exc:
                caught.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        wait_until(lambda: not bus._queue.empty(), what="the worker's request on the queue")
        with self.assertRaises(KeyboardInterrupt):
            bus.pump()
        thread.join(2.0)
        self.assertEqual(len(caught), 1)
        self.assertIsInstance(caught[0], BusClosed, "the worker must not be handed the owner's interrupt")
        self.assertTrue(bus.closed)


if __name__ == "__main__":
    unittest.main()
