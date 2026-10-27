# SPDX-License-Identifier: Apache-2.0
"""``robosimstudio.arms``: which arm a slot names, and a scene that has none."""

from __future__ import annotations

import types
import unittest
from unittest import mock

TIER = "fast"
"""No GPU: a stand-in sim."""


class TestArmNames(unittest.TestCase):
    def test_arm_name_on_a_scene_without_a_task(self) -> None:
        from robosimstudio import arms

        sim = types.SimpleNamespace(request=types.SimpleNamespace(full_name="bare"))
        with mock.patch.object(arms, "arm_order", return_value=()):
            with self.assertRaisesRegex(RuntimeError, "'bare' has no arms"):
                arms.arm_name(sim, 0)


if __name__ == "__main__":
    unittest.main()
