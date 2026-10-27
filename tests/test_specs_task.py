# SPDX-License-Identifier: Apache-2.0
"""The task-side specs: an arrangement's rates."""

from __future__ import annotations

import unittest

TIER = "fast"
"""No GPU: frozen dataclasses."""


class TestAnArrangementStatesItsRates(unittest.TestCase):
    def test_substeps_must_be_a_positive_whole_number(self) -> None:
        from robosimstudio.specs import ArrangementSpec

        for bad in (0, -2, 3.5):
            with self.assertRaises(ValueError):
                ArrangementSpec(substeps=bad)
        with self.assertRaises(ValueError):
            ArrangementSpec(control_hz=0.0)


if __name__ == "__main__":
    unittest.main()
