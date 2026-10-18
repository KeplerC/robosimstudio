# SPDX-License-Identifier: Apache-2.0
"""The scored bag example: the same gesture on the task's page, and the task says the goal is met.

``examples/16_bags/01_open_a_bag_scored`` is ``00_open_a_bag`` as a task document with a goal,
``(>= (mouth_opening bag) 0.03)``. The task's hands start at the YAM's own home, pointing ahead,
so a person first turns each gizmo to point down; this writes that turn into the arm panels'
cursors, then drives them through :mod:`tests.fixtures.grasp` as ``tests/test_open_a_bag``
does: the left hand closed onto the closed end, the right hand down onto the lip, **Close**,
up and back. ``evaluate`` reads the atom and says success.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""One GPU: the task built once and driven through its page for about 8 s of scene."""

PORT = 9763
Z = 0.728
"""``bench_0728``'s work surface [m]."""
TOOL_DOWN = (0.0, 1.0, 0.0, 0.0)
"""The YAM's tool pointing down, jaws closing along world y: the turn a person gives each gizmo first."""


class TestTheTaskGoalTurnsTrue(unittest.TestCase):
    """Both hands turned down, the bag opened as in 00: ``evaluate`` reports the goal met."""

    @classmethod
    def setUpClass(cls) -> None:
        from robosimstudio_materials.bag import layer_gap, mouth_opening

        cls.env, kwargs = _run("16_bags/01_open_a_bag_scored.py")
        sim = getattr(cls.env, "sim", cls.env)
        bag = sim["bag"]
        points = lambda: np.asarray(bag.particle_q, dtype=np.float64)  # noqa: E731
        cls.self_contact = bool(bag.entity.spec.self_contact)
        cls.session = session = open_session(cls.env, PORT, kwargs.get("page"))
        tick(session, 30)
        for arm in ("left", "right"):
            session.arms[arm].cursor_quat = np.asarray(TOOL_DOWN, dtype=np.float64)
        lip = points()[bag.entity.top[int(bag.entity.spec.cells[0]) + 1 :].ravel()]
        order = np.argsort(lip[:, 2])
        x, y = float(np.median(lip[:, 0])), float(np.interp(Z + 0.02, lip[order, 2], lip[order, 1]))
        session.arms["left"].grip_request = 1.0
        miss = [go(session, "left", (0.6, 0.08, Z + 0.10)), go(session, "left", (0.6, 0.08, Z + 0.019), speed=0.06)]
        miss += [go(session, "right", (x, y, Z + 0.10)), go(session, "right", (x, y, Z + 0.02), speed=0.06)]
        session.arms["right"].grip_request = 1.0
        tick(session, 60)
        miss += [
            go(session, "right", (x, y, Z + 0.12), speed=0.06),
            go(session, "right", (x, y + 0.05, Z + 0.12), speed=0.04),
        ]
        tick(session, 60)
        cls.miss = max(miss)
        cls.mouth = mouth_opening(points(), bag.entity)
        cls.gap = layer_gap(points(), bag.entity)
        _, _, cls.info = cls.env.evaluate()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_document_builds_the_bag_that_touches_itself(self) -> None:
        self.assertTrue(self.self_contact, "the record carries the spec's self-contact")

    def test_the_hands_got_where_they_were_dragged(self) -> None:
        self.assertLess(self.miss, 0.008, f"a hand ended {self.miss * 1e3:.1f} mm short")

    def test_the_goal_is_met(self) -> None:
        # 59.9 mm and 63.6 cm^2, the layers 3.63 mm apart at worst (p12_scored.py, GPU 3).
        self.assertTrue(self.info["success"], f"metrics: {self.info['metrics']}")
        self.assertGreaterEqual(float(self.info["metrics"]["height"]), 0.03)
        self.assertGreater(self.gap, 0.0015, f"least gap {self.gap * 1e3:.2f} mm")


if __name__ == "__main__":
    unittest.main()
