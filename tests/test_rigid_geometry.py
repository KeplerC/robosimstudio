# SPDX-License-Identifier: Apache-2.0
"""Mating geometry: whether a rod threads a loop, whether it sits inside one, and where a feature is.

:func:`threaded` asks whether a rod's axis *segment* crosses a loop's disc with
room for the rod's own thickness; :func:`encloses` asks whether the closest
point of a finite rod lies inside the loop's opening, plane or no plane;
:func:`feature_geometry` hands a placed payload back with one representative
point. The answers here are closed forms on hand-placed loops and rods, plus
the properties a geometric verdict must have: invariance under a rigid motion
of the whole set-up, indifference to which end of a rod is named first, and
refusals of a degenerate rod or loop that say why.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from robosimstudio.core.math import quat_from_axis_angle, quat_to_mat
from robosimstudio.domains.rigid.measures.geometry import encloses, feature_geometry, threaded

TIER = "fast"
"""Pure numpy on dictionaries: no model, no GPU."""

RING = {"center": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0], "inner_r": 0.01}
"""A 10 mm ring at the origin, facing up."""


def _rod(p0, p1, radius: float = 0.002) -> dict:
    return {"axis": [list(p0), list(p1)], "radius": radius}


class TestThreaded(unittest.TestCase):
    def test_a_rod_down_the_middle_is_threaded_with_the_room_left_reported(self) -> None:
        got = threaded(RING, _rod((0, 0, -0.1), (0, 0, 0.1)))
        self.assertTrue(got["threaded"])
        self.assertTrue(got["crossings"])
        self.assertAlmostEqual(0.0, got["offset"], places=12)
        self.assertAlmostEqual(0.008, got["clearance"], places=12)
        np.testing.assert_allclose(got["at"], [0.0, 0.0, 0.0], atol=1e-12)

    def test_crossing_the_plane_past_the_rods_end_is_not_threaded(self) -> None:
        """Hanging past the end of a rail is not hanging, though the line through it would be dead centre."""
        got = threaded(RING, _rod((0, 0, 0.05), (0, 0, 0.1)))
        self.assertFalse(got["threaded"])
        self.assertFalse(got["crossings"])
        self.assertAlmostEqual(0.0, got["offset"], places=12, msg="the near miss is still measured")

    def test_a_rod_parallel_to_the_loop_never_crosses_it(self) -> None:
        got = threaded(RING, _rod((-0.1, 0, 0.05), (0.1, 0, 0.05)))
        self.assertEqual((False, False, None), (got["threaded"], got["crossings"], got["at"]))
        self.assertEqual(math.inf, got["offset"])
        self.assertAlmostEqual(0.008, got["clearance"], places=12, msg="the rod would still fit, if it crossed")

    def test_a_rod_too_fat_for_the_loop_reads_negative_clearance(self) -> None:
        got = threaded(RING, _rod((0, 0, -0.1), (0, 0, 0.1), radius=0.012))
        self.assertTrue(got["crossings"], "it does cross the plane")
        self.assertFalse(got["threaded"], "but cannot fit through")
        self.assertAlmostEqual(-0.002, got["clearance"], places=12)

    def test_a_tilted_rod_is_read_where_its_axis_line_meets_the_plane(self) -> None:
        """A rod leaning at 45 degrees through (0.004, 0, 0): offset 4 mm, clearance what is left of the ring."""
        got = threaded(RING, _rod((0.004 - 0.05, 0, -0.05), (0.004 + 0.05, 0, 0.05), radius=0.001))
        np.testing.assert_allclose(got["at"], [0.004, 0.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(0.004, got["offset"], places=12)
        self.assertAlmostEqual(0.01 - 0.001 - 0.004, got["clearance"], places=12)
        self.assertTrue(got["threaded"])

    def test_which_end_of_the_rod_is_named_first_does_not_matter(self) -> None:
        a, b = (0.003, 0.002, -0.04), (-0.001, 0.001, 0.06)
        forward, backward = threaded(RING, _rod(a, b)), threaded(RING, _rod(b, a))
        for key in ("threaded", "crossings"):
            self.assertEqual(forward[key], backward[key], key)
        for key in ("offset", "clearance"):
            self.assertAlmostEqual(forward[key], backward[key], places=12, msg=key)

    def test_moving_the_whole_set_up_rigidly_changes_no_verdict(self) -> None:
        rotation = quat_to_mat(quat_from_axis_angle((0.3, -0.5, 0.8), 1.1))
        shift = np.array([0.4, -0.2, 0.75])

        def moved(point):
            return (rotation @ np.asarray(point, dtype=float) + shift).tolist()

        ring = {
            "center": moved(RING["center"]),
            "normal": (rotation @ np.array(RING["normal"])).tolist(),
            "inner_r": 0.01,
        }
        for p0, p1 in (((0.004, 0, -0.05), (0.004, 0.001, 0.05)), ((0.02, 0, -0.05), (0.02, 0, 0.05))):
            with self.subTest(through=p0):
                here, there = threaded(RING, _rod(p0, p1)), threaded(ring, _rod(moved(p0), moved(p1)))
                self.assertEqual(here["threaded"], there["threaded"])
                self.assertAlmostEqual(here["offset"], there["offset"], places=12)
                self.assertAlmostEqual(here["clearance"], there["clearance"], places=12)

    def test_the_normals_length_is_not_part_of_the_question(self) -> None:
        long_normal = {**RING, "normal": [0.0, 0.0, 7.5]}
        rod = _rod((0.003, 0, -0.05), (0.003, 0, 0.05))
        self.assertEqual(threaded(RING, rod), threaded(long_normal, rod))

    def test_a_degenerate_rod_or_loop_is_refused_by_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "rod axis has zero length"):
            threaded(RING, _rod((0.1, 0.2, 0.3), (0.1, 0.2, 0.3)))
        with self.assertRaisesRegex(ValueError, "loop normal is degenerate"):
            threaded({**RING, "normal": [0.0, 0.0, 0.0]}, _rod((0, 0, -1), (0, 0, 1)))


class TestEncloses(unittest.TestCase):
    def test_a_hook_tilted_off_the_plane_still_encloses(self) -> None:
        """The segment ends 3 mm short of the ring's plane: threaded says no, encloses says it is inside."""
        rod = _rod((0.002, 0, 0.003), (0.002, 0, 0.05))
        self.assertFalse(threaded(RING, rod)["crossings"])
        got = encloses(RING, rod)
        self.assertTrue(got["encloses"])
        self.assertAlmostEqual(float(np.hypot(0.002, 0.003)), got["offset"], places=12)
        np.testing.assert_allclose(got["at"], [0.002, 0.0, 0.003], atol=1e-12)

    def test_the_closest_point_is_clamped_to_the_rods_end(self) -> None:
        got = encloses(RING, _rod((0.03, 0, 0), (0.08, 0, 0)))
        self.assertAlmostEqual(0.03, got["offset"], places=12)
        self.assertAlmostEqual(0.01 - 0.03, got["clearance"], places=12)
        self.assertFalse(got["encloses"])

    def test_a_tolerance_widens_the_opening_by_exactly_itself(self) -> None:
        rod = _rod((0.012, -0.05, 0), (0.012, 0.05, 0))
        plain = encloses(RING, rod)
        widened = encloses({**RING, "enclosure_tolerance": 0.003}, rod)
        self.assertFalse(plain["encloses"])
        self.assertTrue(widened["encloses"])
        self.assertAlmostEqual(0.003, widened["clearance"] - plain["clearance"], places=12)
        self.assertEqual(plain["offset"], widened["offset"])

    def test_a_rod_of_no_length_is_refused_by_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "rod axis has zero length"):
            encloses(RING, _rod((0, 0, 0), (0, 0, 0)))


class TestFeatureGeometry(unittest.TestCase):
    def test_a_loop_is_at_its_centre_and_says_it_is_in_the_world(self) -> None:
        got = feature_geometry({"center": np.array([0.1, 0.2, 0.3]), "normal": (0, 0, 1), "inner_r": 0.01})
        self.assertEqual([0.1, 0.2, 0.3], got["at"])
        self.assertEqual("world", got["frame"])
        self.assertEqual(0.01, got["inner_r"], "a scalar is left as it is")
        self.assertEqual([0.0, 0.0, 1.0], got["normal"], "a direction comes back as a plain list")

    def test_a_rod_with_no_centre_is_at_its_axis_midpoint(self) -> None:
        got = feature_geometry({"axis": [[0.88, 0.1, 1.0], [0.819, 0.1, 1.2]], "radius": 0.003})
        np.testing.assert_allclose(got["at"], [0.8495, 0.1, 1.1], atol=1e-9)

    def test_a_centre_wins_over_a_mouth_and_a_name_is_not_a_coordinate(self) -> None:
        got = feature_geometry({"mouth": [1.0, 1.0, 1.0], "center": [0.0, 0.0, 2.0], "label": "bore"})
        self.assertEqual([0.0, 0.0, 2.0], got["at"])
        self.assertEqual("bore", got["label"])

    def test_a_payload_with_no_point_has_no_position(self) -> None:
        self.assertIsNone(feature_geometry({"radius": 0.01, "label": "x"})["at"])


if __name__ == "__main__":
    unittest.main()
