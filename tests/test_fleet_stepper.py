# SPDX-License-Identifier: Apache-2.0
"""The fleet stepper's ``StepService``: how a motion is installed, judged and finished."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from robosimstudio_policy_agent.fleet import stepper

TIER = "fast"
"""A stub fleet and scripted snapshots: no GPU, no scene."""


class ReachVerdictCadenceTest(unittest.TestCase):
    def test_judged_once_per_stall_window(self) -> None:
        service = object.__new__(stepper.StepService)
        service.arm_names = ["arm"]
        service.cadence = 1
        service.step = 0
        service.fleet = SimpleNamespace(
            scene=SimpleNamespace(embodiment=SimpleNamespace(arms={"arm": SimpleNamespace(ee_body=0)}))
        )
        service._offsets = lambda name: ()
        service._release_arm = lambda world, name: None
        judged: list[int] = []
        service._beyond_reach = lambda world, name, program, target, snapshot: judged.append(program.stalled) or False
        program = stepper._Program(
            "servo",
            goal=(np.array([1.0, 0.0, 0.0]), "arm", None),
            deadline=1000,
            tolerance=0.001,
            stall_steps=12,
        )
        snapshot = SimpleNamespace(body=lambda world: None)
        with mock.patch.object(stepper, "tcp_pose", lambda *a: (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))):
            for tick in range(60):
                service.step = tick
                done, _error, _result = service._finished(0, 0, program, snapshot)
                self.assertFalse(done)
        # Stalled from the second readback: windows close at 12, 24, 36, 48.
        self.assertEqual(judged, [12, 24, 36, 48])


if __name__ == "__main__":
    unittest.main()
