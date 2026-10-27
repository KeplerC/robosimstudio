# SPDX-License-Identifier: Apache-2.0
"""The renderer-independent half of ``core.render``: the live view's budget and the channel colourisers.

A live view renders behind a measured share of the wall clock, not a frame rate
alone; a depth or label image is turned into something displayable without
losing what it exists to show.
"""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio.core.render.live import RenderBudget
from robosimstudio.core.render.video import colorize_depth, colorize_labels

TIER = "fast"
"""No GPU: arithmetic over timestamps and numpy images."""


class TestRenderBudget(unittest.TestCase):
    """The gate a live view renders behind.

    The bug this guards: a live view paced by a target frame rate alone. Ask for
    15 fps of a backend that takes 300 ms and the loop spends all of its wall
    clock rendering, the step rate collapses, and the picture is of a scene that
    has stopped moving. The cap has to be measured, and it has to be a share of
    the clock rather than a frame count.
    """

    def test_a_cheap_frame_runs_at_the_target_rate(self):
        """When the cost is affordable, the frame rate is the only limit."""
        budget = RenderBudget(target_fps=10.0, max_load=0.5)
        budget.record(started=0.0, cost=0.004)
        budget.record(started=0.1, cost=0.004)
        self.assertAlmostEqual(budget.interval, 0.1, places=6)
        self.assertFalse(budget.due(0.15))
        self.assertTrue(budget.due(0.2))

    def test_an_expensive_frame_self_limits_to_its_share(self):
        """The load cap stretches the interval past what the target asked for.

        38 ms a frame at a 30% cap is one frame every 127 ms -- 7.9 fps, not the
        15 that was requested, and that is the whole mechanism.
        """
        budget = RenderBudget(target_fps=15.0, max_load=0.3)
        for i in range(3):
            budget.record(started=i * 0.2, cost=0.038)
        self.assertAlmostEqual(budget.interval, 0.038 / 0.3, places=6)
        self.assertAlmostEqual(1.0 / budget.interval, 7.9, places=1)

    def test_the_first_measurement_is_replaced_rather_than_blended(self):
        """A backend's warm-up frame must not throttle the view for a minute.

        A tiled sensor compiles its megakernel on the first launch, so frame one
        can cost a hundred times frame two. Folded into an EMA that would leave
        the interval seconds long well after the view became cheap.
        """
        budget = RenderBudget(target_fps=60.0, max_load=0.3)
        budget.record(started=0.0, cost=2.0)
        budget.record(started=3.0, cost=0.004)
        self.assertAlmostEqual(budget.cost, 0.004, places=6)

    def test_the_cap_waits_for_a_cost_it_can_trust(self):
        """While warming up, the frame-rate floor paces things, not the cap.

        Otherwise the one expensive warm-up frame sets an interval that has to
        elapse before the cheap frame that would correct it is allowed to run.
        """
        budget = RenderBudget(target_fps=20.0, max_load=0.3)
        budget.record(started=0.0, cost=2.0)
        self.assertAlmostEqual(budget.interval, 0.05, places=6)

    def test_the_first_frame_is_always_due(self):
        """Nothing measured yet is not a reason to withhold the first frame."""
        self.assertTrue(RenderBudget(target_fps=1.0).due(0.0))

    def test_nothing_is_reported_until_a_window_closes(self):
        """A rate over a fraction of a second is noise, so it is not published."""
        budget = RenderBudget(target_fps=10.0, max_load=0.5, window=1.0)
        for i in range(5):
            budget.record(started=i * 0.1, cost=0.02)
        self.assertEqual(budget.fps, 0.0)
        self.assertEqual(budget.frames, 5, "frames are still counted while the window fills")

    def test_load_and_fps_are_measured_over_a_closed_window(self):
        """The readout reports what the loop paid, not what was configured.

        A settled window is exact because its start is the previous window's
        closing *time* rather than a frame: 10 frames of 20 ms in the second
        second is 10 fps at 20% load, which is the number to compare against the
        cap. Only the first window is a frame long, since it has no predecessor
        to start from.
        """
        budget = RenderBudget(target_fps=10.0, max_load=0.5, window=1.0)
        for i in range(21):
            budget.record(started=i * 0.1, cost=0.02)
        self.assertAlmostEqual(budget.fps, 10.0, places=6)
        self.assertAlmostEqual(budget.load, 0.2, places=6)

    def test_a_zero_target_leaves_the_cap_in_charge(self):
        """'As often as my share allows' is a legitimate setting."""
        budget = RenderBudget(target_fps=0.0, max_load=0.25)
        for i in range(3):
            budget.record(started=i, cost=0.05)
        self.assertAlmostEqual(budget.interval, 0.2, places=6)

    def test_a_degenerate_cap_does_not_divide_by_zero(self):
        """A cap of zero is clamped rather than raising inside the loop."""
        budget = RenderBudget(target_fps=10.0, max_load=0.0)
        for i in range(3):
            budget.record(started=i, cost=0.01)
        self.assertTrue(np.isfinite(budget.interval))

    def test_reset_forgets_the_measurement(self):
        """Another backend costs what it costs; carrying a number over lies."""
        budget = RenderBudget(target_fps=10.0, max_load=0.3)
        for i in range(3):
            budget.record(started=i, cost=0.5)
        budget.reset()
        self.assertEqual(budget.cost, 0.0)
        self.assertEqual(budget.frames, 0)
        self.assertTrue(budget.due(0.0))


class TestColorizeChannels(unittest.TestCase):
    """Turning a non-rgb channel into something displayable."""

    def test_depth_misses_stay_black(self):
        """0 means a miss everywhere in this stack, including in a preview."""
        depth = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
        image = colorize_depth(depth)
        self.assertEqual(image.shape, (2, 2, 3))
        np.testing.assert_array_equal(image[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(image[1, 1], [0, 0, 0])
        # Near is bright, so the closer hit outranks the further one.
        self.assertGreater(int(image[0, 1, 0]), int(image[1, 0, 0]))

    def test_labels_are_coloured_and_the_background_is_black(self):
        labels = np.array([[0, 1], [2, 3]], dtype=np.int32)
        image = colorize_labels(labels)
        self.assertEqual(image.shape, (2, 2, 3))
        np.testing.assert_array_equal(image[0, 0], [0, 0, 0])
        self.assertGreater(int(image[0, 1].sum()), 0)

    def test_adjacent_ids_are_far_apart(self):
        """Two objects lying against each other must not be two shades of one colour.

        A linear hue ramp gives consecutive ids nearly the same colour, which is
        the one thing a segmentation view exists to distinguish.
        """
        colors = colorize_labels(np.arange(1, 9, dtype=np.int32)).reshape(-1, 3).astype(np.int32)
        gaps = np.abs(np.diff(colors, axis=0)).max(axis=1)
        self.assertGreater(int(gaps.min()), 60)

    def test_the_same_id_always_gets_the_same_colour(self):
        """A label's colour cannot depend on what else is in frame."""
        alone = colorize_labels(np.array([7], dtype=np.int32))
        crowded = colorize_labels(np.array([3, 7, 11], dtype=np.int32))
        np.testing.assert_array_equal(alone[0], crowded[1])


class TestThePaletteIsNewtons(unittest.TestCase):
    def test_build_order_is_newtons_palette(self) -> None:
        """The copy in ``palette`` is only safe while it still equals the original."""
        import newton

        from robosimstudio.core.render.palette import BUILD_ORDER

        self.assertEqual(tuple(map(tuple, newton.ModelBuilder._SHAPE_COLOR_PALETTE)), BUILD_ORDER)


if __name__ == "__main__":
    unittest.main()
