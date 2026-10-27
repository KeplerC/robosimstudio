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


class TestAParkedShapesContactsAreDropped(unittest.TestCase):
    """``restrict_contacts``: a reader of the restricted view never sees a parked shape's row."""

    PARKED, ITEM, TABLE, FINGER = 4, 0, 1, 2

    def _view(self):
        from tests.fixtures.snapshots import contact_view

        return contact_view(
            [
                (self.TABLE, self.ITEM, (0.0, 0.0, 0.2)),
                (self.PARKED, self.TABLE, (0.0, 0.0, 9.0)),  # parked shape first
                (self.FINGER, self.PARKED, (9.0, 0.0, 0.0)),  # parked shape second
                (self.FINGER, self.ITEM, (0.0, 0.1, 0.0)),
            ],
            depths=[-0.001, -0.002, -0.003, -0.004],
        )

    def test_rows_naming_a_parked_shape_in_either_column_go(self) -> None:
        from robosimstudio.scene.parking import restrict_contacts

        kept = restrict_contacts(self._view(), [self.PARKED])
        self.assertEqual([[self.TABLE, self.ITEM], [self.FINGER, self.ITEM]], kept.geoms.tolist())
        np.testing.assert_allclose([[0.0, 0.0, 0.2], [0.0, 0.1, 0.0]], kept.impulses)
        np.testing.assert_allclose([-0.001, -0.004], kept.depths, err_msg="each row keeps its own depth")
        np.testing.assert_array_equal([0.0, 0.0, 0.0], kept.pairwise_impulse([self.TABLE], [self.PARKED]))

    def test_the_kept_rows_answer_as_they_did(self) -> None:
        from robosimstudio.scene.parking import restrict_contacts

        view = self._view()
        kept = restrict_contacts(view, [self.PARKED])
        np.testing.assert_allclose(
            view.pairwise_force([self.ITEM], [self.TABLE, self.FINGER]),
            kept.pairwise_force([self.ITEM], [self.TABLE, self.FINGER]),
        )
        self.assertEqual(view.dt, kept.dt)

    def test_nothing_to_drop_is_the_same_view_back(self) -> None:
        """The ordinary path costs nothing: no copy when no parked shape, or none of them, is in the set."""
        from robosimstudio.scene.parking import restrict_contacts
        from tests.fixtures.snapshots import contact_view

        view = self._view()
        self.assertIs(view, restrict_contacts(view, []))
        self.assertIs(view, restrict_contacts(view, [7, 8]))
        empty = contact_view([])
        self.assertIs(empty, restrict_contacts(empty, [self.PARKED]))
        self.assertIsNone(restrict_contacts(None, [self.PARKED]))


if __name__ == "__main__":
    unittest.main()
