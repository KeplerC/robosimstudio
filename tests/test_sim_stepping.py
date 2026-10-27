# SPDX-License-Identifier: Apache-2.0
"""Every door that steps a ``Sim`` runs its hooks: pre-step before the physics, post-step after it.

``Sim.run`` and ``rss.draw`` both step a scene, and a hook that only one of them
ran would hold a robot's targets in one and not the other.
"""

from __future__ import annotations

import importlib
import types
import unittest
from unittest import mock

TIER = "fast"
"""CPU only: a ``Sim`` with a stand-in pipeline and a stand-in page, no scene build."""


class TestEveryDoorRunsThePostStepHooks(unittest.TestCase):
    def _sim(self):
        from robosimstudio.sim.sim import Sim

        sim = Sim.__new__(Sim)
        sim.pipeline = types.SimpleNamespace(step=lambda: None)
        sim.step_count = 0
        sim.pre_step_hooks, sim.last_pre_step_hooks = [], []
        sim.post_step_hooks, sim.last_post_step_hooks = [], []
        sim._warned_nonfinite = True  # nothing to read the finite flags from
        return sim

    def test_run(self) -> None:
        sim = self._sim()
        seen: list[int] = []
        sim.post_step_hooks.append(lambda s: seen.append(s.step_count))
        sim.run(3)
        self.assertEqual(seen, [1, 2, 3])

    def test_a_pre_step_hook_runs_before_the_physics(self) -> None:
        """``./ctl physics`` holds the robot's targets here: after whatever drove the robot, before the step."""
        sim = self._sim()
        order: list[str] = []
        sim.pipeline = types.SimpleNamespace(step=lambda: order.append("physics"))
        sim.pre_step_hooks.append(lambda s: order.append(f"pre {s.step_count}"))
        sim.post_step_hooks.append(lambda s: order.append(f"post {s.step_count}"))
        sim.run(2)
        self.assertEqual(order, ["pre 0", "physics", "post 1", "pre 1", "physics", "post 2"])

    def test_draw(self) -> None:
        draw = importlib.import_module("robosimstudio.sim.draw")  # the package exports the function by that name

        sim = self._sim()
        seen: list[int] = []
        sim.post_step_hooks.append(lambda s: seen.append(s.step_count))
        page = mock.MagicMock()
        page.__enter__.return_value = page
        page.is_running.return_value = True
        with mock.patch.object(draw, "viewer", return_value=page):
            draw.draw(sim, steps=2)
        self.assertEqual(seen, [1, 2])
        self.assertEqual(sim.step_count, 2)


if __name__ == "__main__":
    unittest.main()
