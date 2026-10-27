# SPDX-License-Identifier: Apache-2.0
"""``EntityEmbodiment``'s builder-side reading of a hand: which fingers one equality ties together."""

from __future__ import annotations

import unittest

TIER = "fast"
"""No GPU: a stand-in builder's mimic rows."""


class TestAHandsLinkageIsFoundEitherWayRound(unittest.TestCase):
    """``ActuationSpec`` damping, friction and armature reach every finger an equality carries."""

    @staticmethod
    def _builder(rows, enabled=None):
        from types import SimpleNamespace

        # Five one-DOF joints; mimic rows as (joint0, joint1), joint0 following joint1.
        return SimpleNamespace(
            constraint_mimic_joint0=[a for a, _ in rows],
            constraint_mimic_joint1=[b for _, b in rows],
            constraint_mimic_enabled=list(enabled) if enabled is not None else [True] * len(rows),
            joint_qd_start=[0, 1, 2, 3, 4],
            joint_armature=[0.0] * 5,
        )

    def test_the_driven_finger_may_be_either_side_of_the_row(self) -> None:
        from robosimstudio.robots.entity import EntityEmbodiment

        # The arena YAM's equality: the driven finger (1) is joint0, following the carried one (2).
        geared = EntityEmbodiment._geared_dofs(self._builder([(1, 2)]))
        self.assertEqual((1, 2), geared[1])
        self.assertEqual((1, 2), geared[2])
        self.assertNotIn(0, geared)

    def test_a_chain_of_rows_is_one_linkage_and_a_disabled_row_ties_nothing(self) -> None:
        from robosimstudio.robots.entity import EntityEmbodiment

        geared = EntityEmbodiment._geared_dofs(self._builder([(2, 1), (3, 2), (4, 0)], enabled=(True, True, False)))
        self.assertEqual((1, 2, 3), geared[3])
        self.assertNotIn(4, geared)


if __name__ == "__main__":
    unittest.main()
