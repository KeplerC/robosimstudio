# SPDX-License-Identifier: Apache-2.0
"""A knot, as geometry: the overhand knot's nodes and the three measures that say it is still one.

No engine and no GPU. The rope pulled tight by the robot, and the control that
pulls one through itself, are ``tests/test_knot_pull``.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

TIER = "fast"
"""Pure numpy."""

RADIUS = 0.006


def _knot() -> np.ndarray:
    from robosimstudio_materials.rod.knot import overhand

    return overhand(at=(0.03, 0.0), size=0.10, radius=RADIUS, tail=0.12, spacing=0.008)


class TestTheOverhandKnot(unittest.TestCase):
    def test_it_crosses_itself_three_times_and_lies_on_the_bench(self) -> None:
        from robosimstudio_materials.rod.knot import crossings

        nodes = _knot()
        self.assertEqual(3, crossings(nodes))
        self.assertAlmostEqual(RADIUS, float(nodes[:, 2].min()), places=9)

    def test_no_two_strands_touch_as_built(self) -> None:
        """Built clear, so the first step pushes nothing apart: 3.8 mm between the closest strands."""
        from robosimstudio_materials.rod.knot import self_clearance

        self.assertGreater(self_clearance(_knot(), RADIUS), 0.003)

    def test_its_tails_run_along_y_and_its_ends_are_where_the_hands_go(self) -> None:
        nodes = _knot()
        np.testing.assert_allclose(nodes[0, :2], (0.03, -0.1647), atol=5e-4)
        np.testing.assert_allclose(nodes[-1, :2], (0.03, 0.1647), atol=5e-4)
        for tail in (nodes[:10], nodes[-10:]):
            self.assertLess(float(np.ptp(tail[:, 0])), 1e-9, "a tail is straight along y")

    def test_its_elements_are_equal_to_within_a_chord(self) -> None:
        """Resampled at equal arc length, so a leg across a bend is a chord: 0.4 mm short of 8 at most."""
        legs = np.linalg.norm(np.diff(_knot(), axis=0), axis=1)
        self.assertLess(float(np.ptp(legs)), 0.05 * float(legs.max()))


class TestTheKnotMeasures(unittest.TestCase):
    def test_a_straight_rope_crosses_nothing_and_ties_up_nothing(self) -> None:
        from robosimstudio_materials.rod.knot import crossings, knot_length, self_clearance

        line = np.stack([np.zeros(40), np.linspace(-0.2, 0.2, 40), np.full(40, RADIUS)], axis=1)
        self.assertEqual(0, crossings(line))
        self.assertEqual(0.0, knot_length(line))
        self.assertTrue(math.isinf(self_clearance(line, RADIUS)) or self_clearance(line, RADIUS) > 0.0)

    def test_a_strand_through_another_reads_a_diameter_in(self) -> None:
        """Two runs of one rope crossing at the same height: one axis on the other, 2r inside."""
        from robosimstudio_materials.rod.knot import self_clearance

        a = np.stack([np.linspace(-0.1, 0.1, 26), np.zeros(26), np.full(26, RADIUS)], axis=1)
        turn = np.array(
            [[0.1 + 0.03 * math.sin(t), 0.03 - 0.03 * math.cos(t), RADIUS] for t in np.linspace(0, math.pi, 12)]
        )
        b = np.stack([np.full(26, 0.0), np.linspace(0.06, -0.1, 26), np.full(26, RADIUS)], axis=1)
        rope = np.vstack([a, turn[1:], np.array([[0.05, 0.06, RADIUS]]), b])
        self.assertAlmostEqual(-2 * RADIUS, self_clearance(rope, RADIUS), delta=1e-6)

    def test_lifting_the_crossing_strand_clear_reads_the_gap(self) -> None:
        from robosimstudio_materials.rod.knot import self_clearance

        a = np.stack([np.linspace(-0.1, 0.1, 26), np.zeros(26), np.full(26, RADIUS)], axis=1)
        far = np.array([[0.15, 0.05, RADIUS]])
        b = np.stack([np.zeros(26), np.linspace(0.1, -0.1, 26), np.full(26, RADIUS + 0.015)], axis=1)
        self.assertAlmostEqual(0.015 - 2 * RADIUS, self_clearance(np.vstack([a, far, b]), RADIUS), delta=1e-6)

    def test_a_zero_length_leg_is_measured_from_its_point(self) -> None:
        """Two coincident nodes make a leg that is a point: 12 mm over the middle of a run, it is 12 mm from it."""
        from robosimstudio_materials.rod.knot import _segment_distances

        rope = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.012, 0.0], [0.05, 0.012, 0.0], [0.05, 0.012, 0.0]])
        self.assertAlmostEqual(0.012, float(_segment_distances(rope, 3)[0]), places=12)  # the first leg and the point

    def test_a_loop_is_measured_round_its_seam(self) -> None:
        """A band's closing segment meets its first at node 0: neighbours there, not a strand through another."""
        from robosimstudio_materials.rod.band import band
        from robosimstudio_materials.rod.knot import self_clearance

        spec = band(diameter=0.06, segments=32)
        ring, cord = np.asarray(spec.nodes), spec.radius
        seam = np.vstack([ring, ring[:1]])  # as a band's particles read, node 0 drawn twice
        self.assertAlmostEqual(-2 * cord, self_clearance(seam, cord), places=12, msg="read open, its two ends touch")
        across = 2 * 0.03 * math.sin(math.pi / 16) - 2 * cord  # three segments apart: two legs of chord between
        for points in (seam, ring, np.roll(ring, 7, axis=0)):
            self.assertAlmostEqual(across, self_clearance(points, cord, closed=True), places=12)

    def test_the_rope_in_the_knot_is_between_its_first_and_last_crossing(self) -> None:
        from robosimstudio_materials.rod.knot import knot_length

        nodes = _knot()
        whole = float(np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum())
        self.assertTrue(0.25 < knot_length(nodes) < whole - 0.2, "the tails are not in it")

    def test_crossings_is_an_atom_on_the_rod_substrate(self) -> None:
        import robosimstudio_materials.rod  # noqa: F401 -- registers it
        from robosimstudio.assetlib.semantics.atoms import get_atom

        atom = get_atom("crossings")
        self.assertEqual(("crossings", "knot_length"), tuple(atom.returns))


class TestTheRopesSelfContact(unittest.TestCase):
    """What SuperDex is told: a declared ``sim`` first, as a bag and a cloth read it, else the spec's field."""

    def test_a_declared_sim_turns_it_on_and_off_whatever_the_spec_says(self) -> None:
        from robosimstudio.specs.physics import SimRequirements
        from robosimstudio_materials.rod import RodSpec
        from robosimstudio_materials.rod.superdex import EntityRodSuperdex

        for field in (True, False):
            with self.subTest(spec=field):
                rope = EntityRodSuperdex(RodSpec(name="rope", nodes=tuple(map(tuple, _knot())), self_contact=field))
                self.assertIs(field, rope.host_actor().params["self_contact"])
            for declared in (True, False):
                with self.subTest(spec=field, declared=declared):
                    sim = SimRequirements(domain="rod", self_contact=declared)
                    spec = RodSpec(name="rope", nodes=tuple(map(tuple, _knot())), self_contact=field, sim=sim)
                    self.assertIs(declared, EntityRodSuperdex(spec).host_actor().params["self_contact"])


if __name__ == "__main__":
    unittest.main()
