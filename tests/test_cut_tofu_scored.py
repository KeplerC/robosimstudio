# SPDX-License-Identifier: Apache-2.0
"""The scored cutting example: the same cut on the task's page, and the task says the goal is met.

``examples/06_tasks/09_cutting_scored`` is ``03_materials/10_cutting`` as a task document with a goal,
``(>= (pieces tofu) 2)``. Its embodiment starts the right hand over the knife's handle as
``03_materials/10_cutting`` does, so this drives the cut ``tests/test_cut_tofu`` drives: down onto the
handle, **Close**, up, across and down to the board. ``evaluate`` reads the atom and says success.
"""

from __future__ import annotations

import unittest

from tests.fixtures.examples import _run
from tests.fixtures.grasp import open_session, tick
from tests.test_cut_tofu import cut

TIER = "medium"
"""One GPU: the task built once and driven through its page, about 11 ms a control step."""

PORT = 9843


class TestTheTaskGoalTurnsTrue(unittest.TestCase):
    """The knife taken and pushed through the tofu as in 00: ``evaluate`` reports the goal met."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.env, kwargs = _run("06_tasks/09_cutting_scored.py")
        sim = getattr(cls.env, "sim", cls.env)
        tofu, knife = sim["tofu"].entity, sim["knife"]
        cls.session = session = open_session(cls.env, PORT, kwargs.get("page"))
        tick(session, 30)
        cls.before = tofu.pieces()
        _, _, cls.unmet = cls.env.evaluate()
        cls.got = cut(session, tofu, knife)
        _, _, cls.info = cls.env.evaluate()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_document_builds_the_tofu_whole(self) -> None:
        self.assertEqual(1, self.before)
        self.assertFalse(self.unmet["success"], f"metrics: {self.unmet['metrics']}")

    def test_the_hand_got_where_it_was_dragged(self) -> None:
        self.assertLess(self.got["miss"], 0.008, f"the hand ended {self.got['miss'] * 1e3:.1f} mm short")

    def test_the_goal_is_met(self) -> None:
        self.assertTrue(self.info["success"], f"metrics: {self.info['metrics']}")
        self.assertEqual(2.0, float(self.info["metrics"]["count"]))
        self.assertEqual(2, self.got["pieces"])


if __name__ == "__main__":
    unittest.main()
