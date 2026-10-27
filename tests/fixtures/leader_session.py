# SPDX-License-Identifier: Apache-2.0
"""Stand-ins for a studio session and a leader thread, for the feedback tests.

Shared by ``test_leader_force_feedback_servos.py`` (the servo writes, the
preflight and one leader thread's tick) and
``test_leader_force_feedback_session.py`` (the two-clock exchange and the whole
session's way in and out). Nothing here opens a serial port, builds a scene or
starts a thread: :class:`Loop` is the real loop with its thread never started,
and the rest is just enough of an env and a session for the code under test to
ask its questions of -- the studio's own fakes, from :mod:`tests.fixtures.studio_fakes`,
so no two fixtures hold two classes under one name.
"""

from __future__ import annotations

from robosimstudio.robots.teleop import force_feedback as ffb
from tests.fixtures.leader_feedback import instant
from tests.fixtures.studio_fakes import FakeArmPanel, FakeEnv, FakeSession, FakeSpace

__all__ = ["FakeArmPanel", "FakeEnv", "FakeSession", "FakeSpace", "InstantWaits", "Loop"]


class InstantWaits:
    """Mixed into a test case: torque-off settles and command pacing cost nothing."""

    def setUp(self):
        super().setUp()
        waits = instant()
        waits.__enter__()
        self.addCleanup(waits.__exit__, None, None, None)


class Loop(ffb.LeaderArmLoop):
    """A loop whose thread is never started; ``tick`` is driven directly."""

    def __init__(self, servos, **kwargs):
        options = {
            "rate_hz": 100.0,
            "stale_after_s": 0.25,
            # 1.0 keeps the fixtures on the floor: with fewer than two
            # publishes there is no measured cadence to scale, which is the
            # state every test here drives the loop in.
            "stale_factor": 1.0,
            "neutral_dwell_s": 0.5,
            "feedback": True,
            "notify": lambda message: None,
            **kwargs,
        }
        super().__init__("left", servos, servos.config, **options)
