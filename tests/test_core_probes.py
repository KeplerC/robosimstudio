# SPDX-License-Identifier: Apache-2.0
"""The host-side probes: particle coverage, segment and capsule clearance, and the fleet contact probe.

An unfolding task is scored on coverage, so the metric is the task; clearance is
segment-to-segment distance in bulk; the fleet contact probe reads ``nacon``
under the stream lock and has an empty view before its first read.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import newton
import numpy as np

from robosimstudio.core.probes.clearance import ClearanceProbe, _segment_segment
from robosimstudio.core.probes.contact import FleetContactProbe
from robosimstudio.core.probes.particle import ParticleProbe
from robosimstudio.core.streams import stream_lock
from tests.fixtures.arrays import host

TIER = "fast"
"""No GPU: probes over bare arrays and stand-in models."""


class TestCoverage(unittest.TestCase):
    """Occupancy coverage, and the bounding box it exists to replace.

    An unfolding task is *scored* on this, so the metric is the task. A measure
    a policy can satisfy without unfolding anything is not a hard task with a
    lenient gate, it is a different and much easier task wearing the name.
    """

    CELL = 0.02

    def _probe(self, points, baseline):
        """A probe over bare arrays, with no model behind it."""
        probe = ParticleProbe.__new__(ParticleProbe)
        probe.positions = np.asarray(points, dtype=np.float64)
        probe.baseline = np.asarray(baseline, dtype=np.float64)
        probe.indices = lambda name=None: np.arange(len(probe.positions))
        probe.points = lambda name=None: probe.positions
        return probe

    @staticmethod
    def _flat(side: float = 0.3, cells: int = 30):
        grid = np.stack(
            np.meshgrid(np.linspace(0, side, cells + 1), np.linspace(0, side, cells + 1), indexing="xy"), axis=-1
        ).reshape(-1, 2)
        return np.c_[grid, np.zeros(len(grid))]

    def test_flat_sheet_is_fully_covered(self):
        flat = self._flat()
        self.assertAlmostEqual(self._probe(flat, flat).coverage(cell=self.CELL), 1.0, places=6)

    def test_a_ball_covers_almost_nothing(self):
        flat = self._flat()
        rng = np.random.default_rng(0)
        ball = np.c_[
            rng.uniform(0.12, 0.18, len(flat)), rng.uniform(0.12, 0.18, len(flat)), rng.uniform(0.0, 0.06, len(flat))
        ]
        self.assertLess(self._probe(ball, flat).coverage(cell=self.CELL), 0.1)

    def test_flicking_the_corners_out_does_not_count_as_unfolding(self):
        """The case that rules the bounding box out, and the reason this exists.

        A garment lying in a ball with its four corners pulled to the extremes
        fills its bounding rectangle exactly as well as one spread flat, so a
        rectangle scores it 1.00. It has not been unfolded; it has been given
        four whiskers. A policy rewarded on the rectangle finds this, because it
        is enormously cheaper than actually flattening a garment.
        """
        flat = self._flat()
        rng = np.random.default_rng(0)
        gamed = np.c_[
            rng.uniform(0.12, 0.18, len(flat)), rng.uniform(0.12, 0.18, len(flat)), rng.uniform(0.0, 0.06, len(flat))
        ]
        gamed[:4, :2] = [(0.0, 0.0), (0.3, 0.0), (0.0, 0.3), (0.3, 0.3)]
        probe = self._probe(gamed, flat)
        lo, hi = gamed[:, :2].min(axis=0), gamed[:, :2].max(axis=0)
        box = float((hi[0] - lo[0]) * (hi[1] - lo[1]) / 0.09)
        self.assertAlmostEqual(box, 1.0, places=6, msg="the bounding box should be fooled -- that is the point")
        self.assertLess(probe.coverage(cell=self.CELL), 0.1, "occupancy must not be")

    def test_folding_in_half_halves_the_coverage(self):
        """A known answer: the same cloth over half the table."""
        flat = self._flat()
        folded = flat.copy()
        far = folded[:, 1] > 0.15
        folded[far, 1] = 0.3 - folded[far, 1]
        folded[far, 2] = 0.005
        self.assertAlmostEqual(self._probe(folded, flat).coverage(cell=self.CELL), 0.5, places=2)

    def test_empty_and_degenerate_are_zero_not_an_error(self):
        """A metric read before anything is built must not raise."""
        empty = np.zeros((0, 3))
        self.assertEqual(self._probe(empty, empty).covered_area(cell=self.CELL), 0.0)
        self.assertEqual(self._probe(empty, empty).coverage(cell=self.CELL), 0.0)

    def test_a_bad_cell_is_rejected(self):
        flat = self._flat()
        with self.assertRaises(ValueError):
            self._probe(flat, flat).covered_area(cell=0.0)


class TestSegmentDistance(unittest.TestCase):
    """The kernel under :class:`ClearanceProbe`: segment to segment, in bulk."""

    @staticmethod
    def _one(p1, q1, p2, q2) -> float:
        return float(_segment_segment(np.array([p1]), np.array([q1]), np.array([p2]), np.array([q2]))[0, 0])

    def test_parallel_segments_are_their_offset_apart(self):
        d = self._one((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.5, 0.0), (1.0, 0.5, 0.0))
        self.assertAlmostEqual(d, 0.5, places=9)

    def test_crossing_segments_touch(self):
        d = self._one((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))
        self.assertAlmostEqual(d, 0.0, places=9)

    def test_skew_segments_take_the_common_normal(self):
        """Crossing in plan, one unit apart in height."""
        d = self._one((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 1.0), (0.0, 1.0, 1.0))
        self.assertAlmostEqual(d, 1.0, places=9)

    def test_the_closest_points_can_be_endpoints(self):
        """Collinear and disjoint: the answer is the gap between the ends."""
        d = self._one((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (3.0, 0.0, 0.0), (4.0, 0.0, 0.0))
        self.assertAlmostEqual(d, 2.0, places=9)

    def test_degenerate_segments_are_points(self):
        """A sphere is a zero-length segment, which is why it needs no case."""
        d = self._one((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 3.0, 4.0), (0.0, 3.0, 4.0))
        self.assertAlmostEqual(d, 5.0, places=9)

    def test_it_is_symmetric(self):
        a = self._one((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.3, -0.4, 0.7), (0.9, 0.2, 0.1))
        b = self._one((0.3, -0.4, 0.7), (0.9, 0.2, 0.1), (0.0, 0.0, 0.0), (1.0, 1.0, 0.0))
        self.assertAlmostEqual(a, b, places=9)


class TestClearanceProbe(unittest.TestCase):
    """Two capsules on two bodies, and what the model's filtering does to them."""

    RADIUS = 0.05
    HALF = 0.20

    def _model(self, *, filtered=(), collide=(True, True)):
        """Two z-aligned capsules, each at its own body's origin."""
        capsule = int(newton.GeoType.CAPSULE)
        flags = [int(newton.ShapeFlags.COLLIDE_SHAPES) if on else 0 for on in collide]
        return SimpleNamespace(
            shape_count=2,
            shape_body=host([0, 1], np.int32),
            shape_flags=host(flags, np.int32),
            shape_type=host([capsule, capsule], np.int32),
            shape_scale=host([[self.RADIUS, self.HALF, 0.0]] * 2),
            # Identity local transforms: the shapes sit at their bodies.
            shape_transform=host([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 2),
            shape_collision_radius=host([self.HALF + self.RADIUS] * 2),
            shape_collision_filter_pairs=set(filtered),
        )

    @staticmethod
    def _poses(*positions):
        return np.array([[*p, 0.0, 0.0, 0.0, 1.0] for p in positions])

    def test_side_by_side_capsules_report_the_gap_between_their_surfaces(self):
        probe = ClearanceProbe(self._model())
        # Axes 0.3 apart, two 0.05 radii between them.
        gap, a, b = probe.closest([0], [1], self._poses((0.0, 0.0, 0.0), (0.3, 0.0, 0.0)))
        self.assertAlmostEqual(gap, 0.3 - 2 * self.RADIUS, places=9)
        self.assertEqual((a, b), (0, 1))

    def test_overlapping_capsules_report_a_negative_gap(self):
        probe = ClearanceProbe(self._model())
        gap = probe.between([0], [1], self._poses((0.0, 0.0, 0.0), (0.06, 0.0, 0.0)))
        self.assertAlmostEqual(gap, 0.06 - 2 * self.RADIUS, places=9)
        self.assertLess(gap, 0.0)

    def test_end_to_end_capsules_measure_along_their_axes(self):
        """The half-heights count: the answer is not centre-to-centre."""
        probe = ClearanceProbe(self._model())
        gap = probe.between([0], [1], self._poses((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        self.assertAlmostEqual(gap, 1.0 - 2 * self.HALF - 2 * self.RADIUS, places=9)

    def test_a_filtered_pair_is_infinitely_far_apart(self):
        """Not a collision however close it comes -- a gripper's pads meet by design."""
        probe = ClearanceProbe(self._model(filtered=[(0, 1)]))
        gap, a, b = probe.closest([0], [1], self._poses((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        self.assertEqual(gap, float("inf"))
        self.assertEqual((a, b), (-1, -1))
        self.assertEqual(probe.live_pairs([0], [1]), 0)

    def test_a_non_colliding_shape_is_not_measured(self):
        probe = ClearanceProbe(self._model(collide=(True, False)))
        self.assertEqual(probe.between([0], [1], self._poses((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))), float("inf"))
        self.assertEqual(probe.live_pairs([0], [1]), 0)

    def test_live_pairs_counts_what_the_broad_phase_would_test(self):
        self.assertEqual(ClearanceProbe(self._model()).live_pairs([0], [1]), 1)

    def test_a_box_is_bounded_rather_than_missed(self):
        """A box becomes a capsule around its longest axis -- never optimistic."""
        model = self._model()
        model.shape_type = host([int(newton.GeoType.BOX), int(newton.GeoType.BOX)], np.int32)
        model.shape_scale = host([[0.01, 0.02, 0.10]] * 2)
        probe = ClearanceProbe(model)
        # Long axes side by side across x: the true surface gap is 0.5 - 2*0.01,
        # and the bound thickens each by hypot(0.01, 0.02) instead.
        gap = probe.between([0], [1], self._poses((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)))
        self.assertLess(gap, 0.5 - 2 * 0.01, "the bound must never claim more room than there is")
        self.assertAlmostEqual(gap, 0.5 - 2 * float(np.hypot(0.01, 0.02)), places=9)


class TestSegmentAgainstPoint(unittest.TestCase):
    def _d(self, p1, q1, p2, q2):
        return float(
            _segment_segment(
                np.array([p1], float), np.array([q1], float), np.array([p2], float), np.array([q2], float)
            )[0, 0]
        )

    def test_point_second_measures_to_nearest_point_on_segment(self):
        self.assertAlmostEqual(self._d((0, 0, 0), (1, 0, 0), (0.5, 1, 0), (0.5, 1, 0)), 1.0)

    def test_both_orders_agree(self):
        self.assertAlmostEqual(self._d((0.5, 1, 0), (0.5, 1, 0), (0, 0, 0), (1, 0, 0)), 1.0)

    def test_point_past_the_end_clamps(self):
        self.assertAlmostEqual(self._d((0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 0, 0)), 1.0)


class _LockCheckedArr:
    def __init__(self, value, seen):
        self.value, self.seen = np.asarray(value), seen

    def numpy(self):
        self.seen.append(stream_lock._is_owned())
        return self.value


class TestFleetContactProbe(unittest.TestCase):
    def _probe(self, nacon):
        probe = FleetContactProbe.__new__(FleetContactProbe)
        seen = []
        probe.solver = SimpleNamespace(mjw_data=SimpleNamespace(nacon=_LockCheckedArr([nacon], seen)), mjw_model=None)
        probe._ids = np.zeros(4, dtype=np.int32)
        probe.geoms = np.zeros((0, 2), dtype=np.int32)
        probe.impulses = np.zeros((0, 3))
        probe.depths = np.zeros(0)
        probe.worldids = np.zeros(0, dtype=np.int32)
        probe.geom_to_shape = np.zeros((1, 1), dtype=np.int32)
        probe.dt = 0.01
        probe.step = None
        probe._views = {}
        return probe, seen

    def test_a_view_before_the_first_read_is_empty(self):
        k = 3
        solver = SimpleNamespace(
            mjc_geom_to_newton_shape=host(np.arange(k, dtype=np.int32).reshape(1, k)),
            mjw_data=SimpleNamespace(contact=SimpleNamespace(geom=np.zeros((4, 2)))),
        )
        probe = FleetContactProbe(
            solver,
            SimpleNamespace(device="cpu"),
            0.01,
            worlds=SimpleNamespace(shapes_per_world=k, num_worlds=1),
        )
        view = probe.view(0)
        self.assertEqual(len(view.depths), 0)
        self.assertEqual(len(view.penetrations()), 0)

    def test_nacon_read_holds_the_stream_lock(self):
        probe, seen = self._probe(0)
        probe.update_all(1)
        self.assertEqual(seen, [True])
        self.assertEqual(len(probe.view(0).depths), 0)


class TestCoverageNaNBaseline(unittest.TestCase):
    def test_nan_baseline_reads_nan(self):
        probe = ParticleProbe.__new__(ParticleProbe)
        probe.positions = np.array([[0.0, 0, 0], [0.05, 0, 0]])
        probe.baseline = np.array([[np.nan, np.nan, 0.0], [np.nan, 0, 0]])
        probe.indices = lambda name=None: np.arange(2)
        probe.points = lambda name=None: probe.positions
        self.assertTrue(np.isnan(probe.coverage(cell=0.01)))

    def test_nan_rows_in_baseline_are_not_a_cell(self):
        probe = ParticleProbe.__new__(ParticleProbe)
        probe.positions = np.array([[0.0, 0, 0], [0.05, 0, 0], [0.05, 0, 0]])
        probe.baseline = np.array([[0.0, 0, 0], [0.05, 0, 0], [np.nan, 0, 0]])
        probe.indices = lambda name=None: np.arange(3)
        probe.points = lambda name=None: probe.positions
        self.assertAlmostEqual(probe.coverage(cell=0.01), 1.0)


if __name__ == "__main__":
    unittest.main()
