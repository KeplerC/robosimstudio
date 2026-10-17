# SPDX-License-Identifier: Apache-2.0
"""``twist_turns`` on frames built by hand: a uniform turn reads 1.0, a bend with no roll reads 0.

The measure the twist example's readout rests on, checked without a scene: a
chain of capsule frames whose axes all point one way and whose roll about
that axis advances evenly to one turn; the same chain bent round a half
circle with its frames parallel-transported (no roll); and the two put
together, since the measure's claim is that it reads the roll with the bend
taken out. A frame's local ``+Z`` is the capsule's axis, as ``add_rod``
builds them.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from robosimstudio.core.math import quat_from_axis_angle, quat_multiply
from robosimstudio_materials.rod.measures import twist_turns

TIER = "fast"


def _uniform_twist(n: int, turns: float) -> np.ndarray:
    """``n`` frames along ``+Z``, each rolled a further share of ``turns`` about it."""
    return np.array([quat_from_axis_angle((0.0, 0.0, 1.0), 2.0 * math.pi * turns * k / (n - 1)) for k in range(n)])


def _hook(n: int, sweep: float) -> np.ndarray:
    """``n`` frames whose axes sweep ``sweep`` radians about world ``+X``, carried with no roll."""
    return np.array([quat_from_axis_angle((1.0, 0.0, 0.0), sweep * k / (n - 1)) for k in range(n)])


class TestTwistTurnsOnSyntheticFrames(unittest.TestCase):
    def test_a_uniform_turn_over_the_chain_reads_one(self) -> None:
        """From four frames up: over two, one turn is the 2*pi jump the last test shows is invisible."""
        for n in (4, 8, 24):
            self.assertAlmostEqual(twist_turns(_uniform_twist(n, 1.0)), 1.0, places=6, msg=f"{n} frames")

    def test_the_sign_follows_the_hand(self) -> None:
        self.assertAlmostEqual(twist_turns(_uniform_twist(12, -0.5)), -0.5, places=6)

    def test_a_middle_third_reads_its_share(self) -> None:
        """What the example's readout sees: 8 of 23 joints of a one-turn twist over 24 frames."""
        frames = _uniform_twist(24, 1.0)
        self.assertAlmostEqual(twist_turns(frames[8:17]), 8.0 / 23.0, places=6)

    def test_a_bent_but_unrolled_hook_reads_zero(self) -> None:
        for sweep in (0.5, math.pi, 1.5 * math.pi):
            self.assertAlmostEqual(twist_turns(_hook(16, sweep)), 0.0, places=6, msg=f"sweep {sweep}")

    def test_a_twist_survives_a_bend(self) -> None:
        """Roll each frame about its own axis after the bend: the bend is taken out and the roll remains."""
        hook = _hook(16, math.pi)
        rolled = np.array(
            [
                quat_multiply(q, quat_from_axis_angle((0.0, 0.0, 1.0), 2.0 * math.pi * 0.75 * k / 15))
                for k, q in enumerate(hook)
            ]
        )
        self.assertAlmostEqual(twist_turns(rolled), 0.75, places=5)

    def test_one_frame_or_none_reads_zero(self) -> None:
        self.assertEqual(twist_turns(np.zeros((0, 4))), 0.0)
        self.assertEqual(twist_turns(_uniform_twist(2, 1.0)[:1]), 0.0)

    def test_a_whole_turn_at_one_joint_is_invisible(self) -> None:
        """Why ``EntityRod.roll`` rate-limits: a relative measure cannot see a 2*pi jump between neighbours."""
        frames = np.array(
            [quat_from_axis_angle((0.0, 0.0, 1.0), 0.0), quat_from_axis_angle((0.0, 0.0, 1.0), 2.0 * math.pi)]
        )
        self.assertAlmostEqual(twist_turns(frames), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
