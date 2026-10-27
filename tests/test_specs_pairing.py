# SPDX-License-Identifier: Apache-2.0
"""``TaskSpec.with_pairing``: re-pairing a task with a scene or a robot keeps what it does not name."""

import unittest

from robosimstudio.specs import TaskSpec

TIER = "fast"
"""Specs only: nothing is built."""


class TestWithPairingKeepsAPinnedGripper(unittest.TestCase):
    def test_an_empty_gripper_means_the_tasks_own(self):
        pinned = TaskSpec(name="t", scene="cellA", embodiment="robotX").with_pairing(gripper="hand_g")
        moved = pinned.with_pairing(scene="cellB")
        self.assertEqual((moved.scene, moved.embodiment, moved.gripper), ("cellB", "robotX", "hand_g"))
        self.assertEqual(pinned.with_pairing().gripper, "hand_g")
        self.assertIn(("cellB", "robotX", "hand_g"), moved.binding_keys)


if __name__ == "__main__":
    unittest.main()
