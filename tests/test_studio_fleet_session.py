# SPDX-License-Identifier: Apache-2.0
"""The studio's fleet page: each world's start pose from the fleet, its verdicts from the runner, its rate.

The page takes each world's start pose and each world's verdict the way a
batched run does -- the settled pose off the fleet, the score through the
runner's own ``score_world`` -- so watching a wave shows what the same wave
would be reported with. The step rate it reads out is counted over a second of
the page's own clock.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from tests.fixtures.waits import fake_clock

TIER = "fast"
"""No GPU: a stand-in fleet hands out settled poses."""


class _Snapshot:
    def body(self, world):
        # A pose that differs from the fleet's settled one, so a session that
        # read the snapshot itself would be caught.
        return np.full((2, 7), -1.0, dtype=np.float32)


class _Fleet:
    def __init__(self, num_worlds):
        self.settled = [np.full((2, 7), float(w), dtype=np.float32) for w in range(num_worlds)]
        self.contexts = []

    def reset_wave(self, episodes):
        self.episodes = list(episodes)

    def snapshot(self):
        return _Snapshot()

    def settled_body_q(self, world):
        return self.settled[world]

    def context(self, world, snapshot, *, settled_body_q):
        self.contexts.append((world, settled_body_q))
        return (world, settled_body_q)


class _Evaluator:
    def reset(self, context):
        self.context = context


class _Policy:
    def reset_wave(self, slots):
        self.slots = slots


class TestFleetSessionSettled(unittest.TestCase):
    def test_reset_wave_reads_the_fleets_settled_pose(self):
        from robosimstudio_studio import fleet_session

        session = object.__new__(fleet_session.FleetSession)
        session.episode0 = 0
        session.num_worlds = 3
        session.task = object()
        session.fleet = _Fleet(3)
        session.policy = _Policy()
        session._evaluator = _Evaluator
        with mock.patch.object(fleet_session, "plan_waves", return_value=[[0, 1]]):
            session.reset_wave()

        self.assertEqual(len(session.slots), 3)
        for slot in session.slots:
            self.assertIs(slot.settled_body_q, session.fleet.settled[slot.world])
            self.assertIs(slot.evaluator.context[1], session.fleet.settled[slot.world])


class _Slot:
    def __init__(self) -> None:
        self.finished: dict | None = None

    def finish(self, **verdict) -> None:
        self.finished = verdict


def _scoring(step_count: int = 40):
    from robosimstudio_studio import fleet_session

    session = object.__new__(fleet_session.FleetSession)
    session.slots = [_Slot(), _Slot(), _Slot()]
    session.fleet, session.step_count, session.limit = object(), step_count, 500
    session.policy = mock.Mock()
    session.policy.termination_requests.return_value = {1: "the graph says done"}
    session.refreshed = 0
    session._refresh_gui = lambda: setattr(session, "refreshed", session.refreshed + 1)
    return session


class TestTheVerdicts(unittest.TestCase):
    def test_a_world_the_runner_calls_done_is_finished_with_its_verdict(self) -> None:
        from robosimstudio_studio import fleet_session

        session = _scoring()
        asked = []

        def score_world(slot, fleet, snapshot, *, step, limit, request):
            asked.append((session.slots.index(slot), step, limit, request))
            done = slot is session.slots[1]
            return done, {"success": True, "reward": 1.0, "close_kind": "goal"}

        with mock.patch.object(fleet_session, "score_world", score_world):
            session._score(snapshot=None, live=np.array([True, True, False]))
        self.assertEqual([(0, 40, 500, ""), (1, 40, 500, "the graph says done")], asked, "a parked world is not scored")
        self.assertIsNone(session.slots[0].finished)
        self.assertEqual(
            {"success": True, "reward": 1.0, "close_kind": "goal", "steps": 40, "wall_s": 0.0},
            {k: v for k, v in session.slots[1].finished.items() if k != "info"},
        )
        self.assertEqual(1, session.refreshed)


class TestTheRate(unittest.TestCase):
    def _session(self, clock):
        from robosimstudio_studio import fleet_session

        session = object.__new__(fleet_session.FleetSession)
        session._rate_window, session.rate = (clock(), 0), 0.0
        return session

    def test_it_is_the_steps_in_the_second_just_past(self) -> None:
        with fake_clock("robosimstudio_studio.fleet_session") as clock:
            session = self._session(clock)
            for _ in range(20):
                clock.advance(1.0 / 16.0)  # exact in binary: the sixteenth tick lands on the second
                session._tick_rate()
        # 16 ticks reach the second; the rate is read then, and the next window holds the four after it
        self.assertEqual(16.0, session.rate)
        self.assertEqual(4, session._rate_window[1])

    def test_a_window_short_of_a_second_leaves_the_rate_alone(self) -> None:
        with fake_clock("robosimstudio_studio.fleet_session") as clock:
            session = self._session(clock)
            session.rate = 7.0
            for _ in range(9):
                clock.advance(0.1)
                session._tick_rate()
        self.assertEqual(7.0, session.rate)
        self.assertEqual(9, session._rate_window[1])


if __name__ == "__main__":
    unittest.main()
