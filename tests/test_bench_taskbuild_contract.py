# SPDX-License-Identifier: Apache-2.0
"""The task folder contract: an import that climbs past a task's root reaches out of it."""

import unittest

TIER = "fast"
"""Pure path arithmetic: no scene, no GPU."""


class TestReachesOut(unittest.TestCase):
    def test_climbing_past_the_root_is_outside_at_any_depth(self):
        from robosimstudio_bench.taskbuild.contract import reaches_out

        self.assertFalse(reaches_out(("t", "assets"), 2, "x"))
        self.assertTrue(reaches_out(("t", "assets"), 3, "x"))
        self.assertTrue(reaches_out(("t", "assets"), 4, "x"))
        self.assertTrue(reaches_out(("t", "assets"), 6, "x"))
        self.assertFalse(reaches_out(("t", "assets"), 4, "_shared.y"))


if __name__ == "__main__":
    unittest.main()
