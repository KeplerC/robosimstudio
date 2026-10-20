# SPDX-License-Identifier: Apache-2.0
"""The scored tape example: the same peel on the task's page, and the task says the goal is met.

``examples/06_tasks/08_tape_scored`` is ``03_materials/08_tape`` as a task document with a goal,
``(>= (peeled_fraction label) 0.8)``. The task's hands start at the YAM's own home, pointing ahead,
so a person first turns the right gizmo to point down; this writes that turn into the arm panel's
cursor, takes the right hand over the tab, then drives the peel ``tests/test_peel_a_label`` drives:
down onto the tab, **Close**, up 30 mm, then seven times back 20 mm and up 10 mm. ``evaluate`` reads
the atom and says success.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick
from tests.test_peel_a_label import peel

TIER = "medium"
"""One GPU: the task built once and driven through its page, about 38 ms a control step."""

PORT = 9793
TOOL_DOWN = (0.0, 1.0, 0.0, 0.0)
"""The YAM's tool pointing down, jaws closing along world y: the turn a person gives the right gizmo first."""


class TestTheTaskGoalTurnsTrue(unittest.TestCase):
    """The right hand turned down, the label peeled as in 00: ``evaluate`` reports the goal met."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.env, kwargs = _run("06_tasks/08_tape_scored.py")
        sim = getattr(cls.env, "sim", cls.env)
        label = sim["label"].entity
        cls.session = session = open_session(cls.env, PORT, kwargs.get("page"))
        tick(session, 30)
        cls.before = label.peeled()
        session.arms["right"].cursor_quat = np.asarray(TOOL_DOWN, dtype=np.float64)
        above = label.tab_pose(session.env.state)[:3] + np.array([0.0, 0.0, 0.05])
        cls.above = go(session, "right", above)
        cls.got = peel(session, label)
        _, _, cls.info = cls.env.evaluate()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_document_builds_the_label_stuck_down(self) -> None:
        self.assertEqual(0.0, self.before, "nothing has let go before the hand comes")

    def test_the_hand_got_where_it_was_dragged(self) -> None:
        self.assertLess(self.above, 0.008, f"over the tab, the hand ended {self.above * 1e3:.1f} mm short")
        self.assertLess(self.got["miss"], 0.008, f"the hand ended {self.got['miss'] * 1e3:.1f} mm short")

    def test_the_goal_is_met(self) -> None:
        self.assertTrue(self.info["success"], f"metrics: {self.info['metrics']}")
        self.assertGreaterEqual(float(self.info["metrics"]["fraction"]), 0.8)
        self.assertAlmostEqual(self.got["peeled"][-1], float(self.info["metrics"]["fraction"]))


if __name__ == "__main__":
    unittest.main()
