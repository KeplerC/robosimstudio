# SPDX-License-Identifier: Apache-2.0
"""The heavy elastica the cloth drape bench inverts: a clamped strip's droop, and the bending length read back from it.

``elastica_droop`` solves the inextensible cantilever under its own weight by
shooting; ``bending_length_from_droop`` inverts its tabulated curve. What is
held: the small-deflection limit is the beam formula's ``beta / 8``; a rigid
strip does not droop; droop rises with ``beta`` and never reaches the strip's
length; a bending length goes to a droop and comes back; and a droop off the
curve reads NaN rather than a clamped value. The reference fabrics a reading
resembles are the ones whose band covers it.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from robosimstudio_materials.cloth import (
    REFERENCE_FABRICS,
    bending_length_from_droop,
    droop_curve,
    elastica_droop,
    resembles,
)

TIER = "fast"
"""An RK4 shooting solve in numpy, about two seconds for the whole table: no GPU."""


class TestTheDroop(unittest.TestCase):
    def test_a_light_strip_droops_the_beam_formulas_eighth(self) -> None:
        """``w L^4 / 8 B`` over ``L`` is ``beta / 8``: the elastica's small-deflection limit."""
        betas = np.array([1.0e-3, 1.0e-2])
        np.testing.assert_allclose(elastica_droop(betas, steps=100, shots=60), betas / 8.0, rtol=1.0e-4)

    def test_a_rigid_strip_does_not_droop_and_a_scalar_comes_back_a_scalar(self) -> None:
        self.assertEqual(0.0, elastica_droop(0.0, steps=10, shots=10))
        self.assertIsInstance(elastica_droop(0.5, steps=50, shots=40), float)
        self.assertEqual((2,), elastica_droop(np.array([0.5, 1.0]), steps=50, shots=40).shape)

    def test_droop_rises_with_beta_and_stays_short_of_the_length(self) -> None:
        betas, droops = droop_curve()
        self.assertTrue(np.all(np.diff(droops) > 0.0))
        self.assertGreater(float(droops[0]), 0.0)
        self.assertLess(float(droops[-1]), 1.0)
        self.assertGreater(float(droops[-1]), 0.9, "at beta 100 the strip hangs nearly straight down")
        self.assertEqual(len(betas), len(droops))


class TestTheInversion(unittest.TestCase):
    def test_a_bending_length_round_trips_through_its_droop(self) -> None:
        cases = ((0.1, 0.025), (0.2, 0.06), (0.15, 0.12))
        droops = elastica_droop(np.array([(length / bending) ** 3 for length, bending in cases]))
        for (length, bending), droop in zip(cases, droops, strict=True):
            with self.subTest(length=length, bending=bending):
                self.assertAlmostEqual(bending, bending_length_from_droop(droop, length), delta=1.0e-4 * bending)

    def test_a_droop_off_the_curve_reads_nan_rather_than_a_clamped_value(self) -> None:
        _, droops = droop_curve()
        for droop in (0.0, float(droops[0]), float(droops[-1]), 0.99):
            with self.subTest(droop=droop):
                self.assertTrue(math.isnan(bending_length_from_droop(droop, 0.1)))


class TestWhatAReadingResembles(unittest.TestCase):
    def test_the_fabrics_whose_band_covers_it_and_none_for_no_reading(self) -> None:
        self.assertEqual((), resembles(float("nan")))
        for fabric in REFERENCE_FABRICS:
            with self.subTest(fabric=fabric.name):
                middle = 0.5 * sum(fabric.bending_cm)
                self.assertIn(fabric.name, resembles(middle))
                self.assertNotIn(fabric.name, resembles(fabric.bending_cm[1] * 3.0 + 10.0))


if __name__ == "__main__":
    unittest.main()
