# SPDX-License-Identifier: Apache-2.0
"""Where a scene parks the objects an episode does not use: one lattice, the task's."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

TIER = "fast"
"""No GPU: specs and a stand-in scene."""


class TestOneParkLattice(unittest.TestCase):
    def test_a_hand_assembled_scene_parks_on_the_task_lattice(self) -> None:
        from robosimstudio.scene.parking import scene_park_poses
        from robosimstudio.specs import park_lattice
        from robosimstudio.specs import task as task_module

        names = [f"o{i}" for i in range(11)]
        scene = SimpleNamespace(request=SimpleNamespace(), objects=dict.fromkeys(names))
        # A row length other than the shipped one: a hard-coded copy would not follow it.
        row = task_module._PARK_ROW
        task_module._PARK_ROW = 5
        try:
            poses = scene_park_poses(scene)
            expected = {name: park_lattice(index) for index, name in enumerate(names)}
        finally:
            task_module._PARK_ROW = row
        self.assertAlmostEqual(task_module.PARK_PITCH, expected["o5"][1], msg="the sixth object starts row two")
        for name in names:
            np.testing.assert_allclose(poses[name], expected[name])


if __name__ == "__main__":
    unittest.main()
