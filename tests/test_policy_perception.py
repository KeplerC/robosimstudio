# SPDX-License-Identifier: Apache-2.0
"""The palette segmenter: what perception falls back to when SAM3's weights are not there.

It reads RGB and nothing else, and promises only the synthetic palette the
scenes are painted in: an orange thing, a blue thing, a white cable. Each is
asked of a picture drawn here, with the answer known by construction -- where
the patch is, and which of two white things is the thin one.
"""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

TIER = "fast"
"""Synthetic pictures through OpenCV on the CPU: no GPU, no weights, no scene."""

HAS_CV2 = importlib.util.find_spec("cv2") is not None


def _picture(background=(128, 128, 128)) -> np.ndarray:
    picture = np.empty((120, 160, 3), dtype=np.uint8)
    picture[...] = background
    return picture


def _segment(picture: np.ndarray, query: str, max_results: int = 0) -> dict:
    from robosimstudio_policy_agent.perception_tools import _palette_segment

    return _palette_segment(picture, query, max_results)


@unittest.skipUnless(HAS_CV2, "opencv is not installed")
class TestThePaletteFindsWhatItWasPaintedWith(unittest.TestCase):
    def test_an_orange_patch_is_found_where_it_is_and_the_grey_bench_is_not(self):
        picture = _picture()
        picture[30:50, 40:70] = (255, 140, 0)
        found = _segment(picture, "the orange plug")
        self.assertEqual(found["backend"], "palette")
        self.assertEqual(found["boxes"], [[40.0, 30.0, 70.0, 50.0]])
        self.assertEqual(int(found["masks"][0].sum()), 20 * 30)

    def test_a_blue_bin_is_found_and_a_blue_grey_bench_is_not(self):
        picture = _picture(background=(118, 124, 134))
        picture[60:100, 20:60] = (30, 60, 220)
        found = _segment(picture, "blue bin")
        self.assertEqual(found["boxes"], [[20.0, 60.0, 60.0, 100.0]])

    def test_a_thin_white_cable_outranks_a_white_block(self):
        picture = _picture(background=(60, 60, 60))
        picture[20:40, 100:120] = (250, 250, 250)  # a block
        picture[80:83, 10:110] = (250, 250, 250)  # the cable
        found = _segment(picture, "white cable")
        self.assertEqual(found["boxes"][0], [10.0, 80.0, 110.0, 83.0])
        self.assertGreater(found["scores"][0], 0.95)
        self.assertLess(found["scores"][1], 0.75)
        self.assertEqual(len(_segment(picture, "white cable", max_results=1)["masks"]), 1)

    def test_a_speck_is_not_a_thing(self):
        picture = _picture()
        picture[10:14, 10:14] = (255, 140, 0)
        self.assertEqual(_segment(picture, "orange")["masks"], [])

    def test_a_prompt_outside_the_palette_is_refused_by_name(self):
        with self.assertRaisesRegex(RuntimeError, "does not support 'a red mug'"):
            _segment(_picture(), "a red mug")


def _reference_opening(binary: np.ndarray) -> np.ndarray:
    """The opening by a 2x2 square, by its definition: the union of every 2x2 block wholly inside the mask."""
    inside = binary > 0
    fits = inside[:-1, :-1] & inside[1:, :-1] & inside[:-1, 1:] & inside[1:, 1:]
    out = np.zeros_like(inside)
    for dy in (0, 1):
        for dx in (0, 1):
            out[dy : dy + fits.shape[0], dx : dx + fits.shape[1]] |= fits
    return out


@unittest.skipUnless(HAS_CV2, "opencv is not installed")
class TestTheOpeningIsTheSameAtEveryEdge(unittest.TestCase):
    """The off-image pixels are background: a strip one pixel wide is removed wherever it lies."""

    def test_a_one_pixel_strip_is_removed_in_column_zero_as_in_column_nine(self):
        for column in (0, 9):
            with self.subTest(column=column):
                picture = _picture()
                picture[10:90, column] = (255, 140, 0)
                self.assertEqual(_segment(picture, "orange")["masks"], [])

    def test_random_masks_open_to_the_reference(self):
        from robosimstudio_policy_agent.perception_tools import _opened

        rng = np.random.default_rng(0)
        for trial in range(50):
            binary = (rng.random((24, 32)) < 0.6).astype(np.uint8) * 255
            with self.subTest(trial=trial):
                np.testing.assert_array_equal(_opened(binary) > 0, _reference_opening(binary))


if __name__ == "__main__":
    unittest.main()
