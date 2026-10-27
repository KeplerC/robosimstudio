# SPDX-License-Identifier: Apache-2.0
"""``Gripper``: one 0 (open) .. 1 (closed) command over a hand's finger joints, and the one place metres become it.

Over Warp arrays on the CPU and hand-built joint vectors. What is held: the
refusals its constructor and :meth:`~Gripper.calibrate` name; the command
expanded into each finger joint's target, mirrored or geared; the width read
back off the joints, and the closed fraction from it; and the width-command law
that :meth:`~Gripper.command_for_width` inverts -- an identity both before a
measured fit is installed and after.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from tests.fixtures.arrays import host

TIER = "fast"
"""Warp arrays on its CPU device and a host joint vector: no GPU."""

CPU = SimpleNamespace(device="cpu")
"""The model, as far as a gripper reads one: where its arrays live."""


def _gripper(**kwargs):
    from robosimstudio.solvers.motion.gripper import Gripper

    coords = kwargs.pop("coords", (7, 8))
    return Gripper(CPU, coords, **kwargs)


class TestWhatItRefuses(unittest.TestCase):
    def test_a_hand_it_cannot_drive_is_refused_by_name(self) -> None:
        cases = {
            "no finger joint": ({"coords": ()}, "at least one finger joint"),
            "mirrored with three joints": ({"coords": (1, 2, 3)}, "exactly two mirrored finger joints, got 3"),
            "an open value short": ({"coords": (1, 2), "open_q": (0.1,)}, "open_q has 1 values for 2"),
            "a closed value short": (
                {"coords": (1, 2), "open_q": (0.1, 0.2), "closed_q": (0.0,)},
                "closed_q has 1 values for 2",
            ),
            "one width coordinate": ({"width_coords": (3,)}, "two coordinates whose difference is the gap, got 1"),
        }
        for what, (kwargs, message) in cases.items():
            with self.subTest(what), self.assertRaisesRegex(ValueError, message):
                _gripper(**kwargs)

    def test_a_fit_with_no_slope_cannot_be_inverted(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero slope"):
            _gripper().calibrate(gap_per_command=0.0, gap_at_zero=0.08)


class TestTheCommandDrivesTheJoints(unittest.TestCase):
    def _targets(self, gripper, command: float) -> np.ndarray:
        import warp as wp

        targets = wp.array(np.full(10, 5.0, dtype=np.float32), dtype=wp.float32, device="cpu")
        gripper.command = command
        gripper.apply(targets)
        return targets.numpy()

    def test_mirrored_fingers_interpolate_from_plus_and_minus_the_open_width_to_shut(self) -> None:
        gripper = _gripper(open_width=0.04)
        written = self._targets(gripper, 0.25)
        np.testing.assert_allclose(written[[7, 8]], [0.03, -0.03], atol=1e-7)
        np.testing.assert_array_equal(np.full(8, 5.0, dtype=np.float32), np.delete(written, [7, 8]))

    def test_a_geared_hand_goes_from_its_own_open_to_its_own_closed(self) -> None:
        gripper = _gripper(coords=(4,), open_q=(0.0,), closed_q=(0.785,), span=0.085)
        self.assertEqual("geared", gripper.kind)
        self.assertAlmostEqual(0.785 * 0.6, float(self._targets(gripper, 0.6)[4]), places=6)

    def test_the_command_is_held_to_open_and_shut(self) -> None:
        gripper = _gripper()
        gripper.command = 1.7
        self.assertEqual(1.0, gripper.command)
        gripper.command = -0.2
        self.assertEqual(0.0, gripper.command)


class TestTheWidthItReadsBack(unittest.TestCase):
    def test_each_kind_of_hand_reads_its_gap_and_its_closed_fraction(self) -> None:
        q = np.zeros(10)
        q[[7, 8]] = (0.015, -0.015)
        q[[2, 3]] = (0.02, -0.01)
        q[4] = 0.785 * 0.4
        state = SimpleNamespace(joint_q=host(q))
        mirrored = _gripper(open_width=0.04)
        paired = _gripper(coords=(2,), open_q=(0.04,), closed_q=(0.0,), width_coords=(2, 3))
        geared = _gripper(coords=(4,), open_q=(0.0,), closed_q=(0.785,), span=0.085)
        for name, gripper, gap, kind in (
            ("mirrored", mirrored, 0.03, "prismatic"),
            ("one driven, a pair measured", paired, 0.03, "prismatic"),
            ("geared", geared, 0.085 * 0.6, "geared"),
        ):
            with self.subTest(name):
                self.assertEqual(kind, gripper.kind)
                self.assertAlmostEqual(gap, gripper.width(state), places=7)
                self.assertAlmostEqual(1.0 - gap / gripper.span, gripper.closed_fraction(state), places=6)


class TestTheWidthLaw(unittest.TestCase):
    """``width_for_command`` and ``command_for_width`` are one law, read both ways."""

    def test_width_to_command_to_width_is_the_identity_until_calibrated(self) -> None:
        gripper = _gripper(open_width=0.04)
        self.assertFalse(gripper.calibrated)
        for gap in np.linspace(0.0, gripper.span, 9):
            self.assertAlmostEqual(gap, gripper.width_for_command(gripper.command_for_width(gap)), places=12)
        self.assertEqual(0.0, gripper.command_for_width(1.0), "wider than open is open")
        self.assertEqual(1.0, gripper.command_for_width(-0.01), "and narrower than shut is shut")

    def test_and_after_a_measured_fit_is_installed(self) -> None:
        gripper = _gripper(open_width=0.04)
        gripper.calibrate(gap_per_command=-0.07, gap_at_zero=0.075)
        self.assertTrue(gripper.calibrated)
        for command in np.linspace(0.0, 1.0, 9):
            gap = gripper.width_for_command(command)
            self.assertAlmostEqual(0.075 - 0.07 * command, gap, places=12)
            self.assertAlmostEqual(command, gripper.command_for_width(gap), places=12)


if __name__ == "__main__":
    unittest.main()
