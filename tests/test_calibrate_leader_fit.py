# SPDX-License-Identifier: Apache-2.0
"""The leader calibration tool's arithmetic: the sweep's fit, the re-zero, the parser and the preflight.

``scripts/calibrate_leader_arms.py`` turns a sweep of encoder readings into a
calibration. The fit has to read the sign, the gearing and the offset off the
data rather than assume them; a re-zeroed offset has to round-trip through
``to_radians``. The tool on its page, over a built scene, is
``tests.test_calibrate_leader_arms``'s.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

TIER = "fast"
"""Loads the script by path and calls its pure functions: no scene, no page, no GPU."""


ROOT = Path(__file__).resolve().parents[1]


TRUE_SIGN, TRUE_TPR, TRUE_OFFSET = -1.0, 600.0, 2000.0
"""The leader this file pretends to be: every encoder counting backwards, geared
away from the direct-drive seed, centred off 2048. Each is a value the sweep has
to *measure* rather than assume."""


def _tool():
    """``scripts/calibrate_leader_arms.py`` as a module, loaded by file path."""
    spec = importlib.util.spec_from_file_location(
        "_rss_calibrate_leader_arms", ROOT / "scripts" / "calibrate_leader_arms.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ticks(angle: float, offset: float = TRUE_OFFSET) -> float:
    """What the pretend leader's encoder reads at a joint angle: the inverse of ``to_radians``."""
    return offset + TRUE_SIGN * TRUE_TPR * angle


class TestTheArithmetic(unittest.TestCase):
    """The two pure functions, without a scene."""

    def setUp(self) -> None:
        self.cal = _tool()

    def test_the_fit_reads_the_sign_off_the_data(self) -> None:
        angles = [math.radians(d) for d in self.cal.SWEEP_DEG]
        tpr, sign, offset, residual = self.cal._fit_joint(angles, [_ticks(a) for a in angles])
        self.assertAlmostEqual(tpr, TRUE_TPR, places=6)
        self.assertEqual(sign, TRUE_SIGN)
        self.assertAlmostEqual(offset, TRUE_OFFSET, places=6)
        self.assertLess(residual, 1e-6)

    def test_a_zeroed_offset_round_trips_through_to_radians(self) -> None:
        from robosimstudio.robots.teleop.lerobot_controller import JointCalibration

        for sign in (1.0, -1.0):
            with self.subTest(sign=sign):
                home = 1.047
                ticks = 2500.0
                offset = self.cal.zeroed_offset(ticks, sign=sign, ticks_per_radian=TRUE_TPR, angle=home)
                joint = JointCalibration(name="j", ticks_per_radian=TRUE_TPR, sign=sign, offset_ticks=offset)
                self.assertAlmostEqual(joint.to_radians(ticks), home, places=9)

    def test_the_parser_takes_leader_or_its_old_spelling(self) -> None:
        parser = self.cal.create_parser()
        self.assertEqual(parser.parse_args(["--leader", "right:/dev/x"]).leader, "right:/dev/x")
        self.assertEqual(parser.parse_args(["--arms", "right:/dev/x"]).leader, "right:/dev/x")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--leader", "right:/dev/x", "--zero-only", "--gripper-only"])

    def test_the_preflight_refuses_in_one_sentence(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self.cal.preflight({"right": "/dev/ch340-NOPE"}, Path("nope.json"), "full")
        message = str(caught.exception)
        self.assertTrue("lerobot-feetech" in message or "no device at /dev/ch340-NOPE" in message, message)
        self.assertNotIn("\n", message)


if __name__ == "__main__":
    unittest.main()
