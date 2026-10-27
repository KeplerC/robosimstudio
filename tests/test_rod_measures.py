# SPDX-License-Identifier: Apache-2.0
"""The rod's shape measures on hand-built geometry: twist about the centreline, winding about an axis, straightness.

``twist_turns`` on frames built by hand: a uniform turn reads 1.0, a bend with
no roll reads 0. The measure the twist example's readout rests on, checked
without a scene: a
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

from robosimstudio.core.math import quat_from_axis_angle, quat_multiply, quat_to_mat
from robosimstudio_materials.rod.measures import curl_retained, straightness, twist_turns, winding_turns
from tests.fixtures.grids import chain, helix

TIER = "fast"
"""Pure numpy on hand-built frames and curves: no GPU."""


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


def _helix(origin, u, v, d, *, turns: float = 2.0, radius: float = 0.03, n: int = 80) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi * turns, n)
    return (
        np.asarray(origin)
        + np.outer(radius * np.cos(t), u)
        + np.outer(radius * np.sin(t), v)
        + np.outer(np.linspace(-0.02, 0.02, n), d)
    )


class TestWindingIsAboutTheDeclaredAxis(unittest.TestCase):
    """``rod/measures.py``: a horizontal axis (a hook's) counts its turns."""

    def test_a_horizontal_axis(self) -> None:
        from robosimstudio_materials.rod import measures

        origin = np.array([0.3, -0.1, 0.2])
        wrap = _helix(origin, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
        got = measures._winding_about_axis(wrap, {"origin": origin, "direction": (1.0, 0.0, 0.0)})
        self.assertAlmostEqual(got["turns"], 2.0, delta=0.05)
        along = np.array([0.05, 0.0, 0.0])
        ends = np.stack([origin - along, origin + along])
        self.assertAlmostEqual(measures._winding_about_rod(wrap, {"axis": ends})["turns"], 2.0, delta=0.05)

    def test_a_vertical_axis_reads_as_before(self) -> None:
        from robosimstudio_materials.rod import measures

        origin = np.array([0.1, 0.2, 0.05])
        wrap = _helix(origin, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        before = measures.winding_turns(wrap, origin[:2])
        self.assertEqual(
            measures._winding_about_axis(wrap, {"origin": origin, "direction": (0, 0, 1)})["turns"], before[0]
        )


def _rigid(seed: int):
    """A random rotation and translation: ``(R [3, 3], t [3], quat_xyzw)``."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    angle = float(rng.uniform(0.0, np.pi))
    quat = quat_from_axis_angle(axis, angle)
    return quat_to_mat(quat), rng.uniform(-1.0, 1.0, 3), quat


class TestStraightness(unittest.TestCase):
    """End to end over path length: 1 straight, ``2/pi`` for a half circle, and blind to where the chain lies."""

    def test_a_straight_chain_is_one_and_retains_no_curl(self) -> None:
        line = chain(segments=12)
        self.assertAlmostEqual(1.0, straightness(line), places=12)
        self.assertAlmostEqual(0.0, curl_retained(line), places=12)

    def test_a_half_circle_is_two_over_pi(self) -> None:
        t = np.linspace(0.0, np.pi, 4001)
        arc = np.stack([0.3 * np.cos(t), 0.3 * np.sin(t), np.zeros_like(t)], axis=1)
        self.assertAlmostEqual(2.0 / np.pi, straightness(arc), places=6)

    def test_moving_or_reversing_the_chain_changes_nothing(self) -> None:
        bent = helix(turns=0.6, radius=0.05, segments=30, rise=0.04)
        rotation, shift, _ = _rigid(1)
        want = straightness(bent)
        self.assertAlmostEqual(want, straightness(bent @ rotation.T + shift), places=12)
        self.assertAlmostEqual(want, straightness(bent[::-1]), places=12)

    def test_a_chain_with_no_length_is_called_straight(self) -> None:
        self.assertEqual(1.0, straightness(np.zeros((1, 3))))
        self.assertEqual(1.0, straightness(np.tile([0.2, 0.1, 0.0], (5, 1))))


class TestWindingTurns(unittest.TestCase):
    def test_a_helix_of_k_turns_reads_k_either_way_round(self) -> None:
        for turns in (0.5, 1.0, 3.0, -2.0):
            with self.subTest(turns=turns):
                wound = helix(turns=turns, radius=0.04, segments=200, centre=(0.3, -0.1), rise=0.05)
                got, engaged = winding_turns(wound, (0.3, -0.1))
                self.assertAlmostEqual(abs(turns), got, places=9)
                self.assertEqual(199, engaged)

    def test_a_figure_eight_cancels_and_a_hank_beside_the_axis_reads_nothing(self) -> None:
        there = helix(turns=1.0, radius=0.04, segments=100)
        back = helix(turns=-1.0, radius=0.04, segments=100)
        self.assertAlmostEqual(0.0, winding_turns(np.concatenate([there, back]), (0.0, 0.0))[0], places=9)
        beside = helix(turns=2.0, radius=0.04, segments=100, centre=(0.5, 0.0))
        self.assertEqual((0.0, 0), winding_turns(beside, (0.0, 0.0)))

    def test_turning_or_moving_chain_and_axis_together_changes_nothing(self) -> None:
        wound = helix(turns=1.75, radius=0.05, segments=150, centre=(0.1, 0.2))
        want = winding_turns(wound, (0.1, 0.2))
        yaw = quat_to_mat(quat_from_axis_angle((0.0, 0.0, 1.0), 0.9))
        shift = np.array([-0.4, 0.3, 0.2])
        axis = yaw @ np.array([0.1, 0.2, 0.0]) + shift
        got = winding_turns(wound @ yaw.T + shift, axis[:2])
        self.assertAlmostEqual(want[0], got[0], places=9)
        self.assertEqual(want[1], got[1])


class TestTwistIsAPropertyOfTheChain(unittest.TestCase):
    def test_turning_the_whole_chain_changes_nothing_and_reversing_it_flips_the_sign(self) -> None:
        hook = _hook(16, math.pi)
        rolled = np.array(
            [
                quat_multiply(q, quat_from_axis_angle((0.0, 0.0, 1.0), 2.0 * math.pi * 0.3 * k / 15))
                for k, q in enumerate(hook)
            ]
        )
        want = twist_turns(rolled)
        _, _, turn = _rigid(2)
        self.assertAlmostEqual(want, twist_turns(np.array([quat_multiply(turn, q) for q in rolled])), places=9)
        self.assertAlmostEqual(-want, twist_turns(rolled[::-1]), places=9)


if __name__ == "__main__":
    unittest.main()
