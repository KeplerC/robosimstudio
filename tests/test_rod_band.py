# SPDX-License-Identifier: Apache-2.0
"""A closed rod, as data: the band constructor, its materials, and the measures a loop answers.

No engine and no GPU: the spec's closing rules, the circle :func:`band` lays,
and the winding a loop scores about an axis. The SuperDex build of a closed rod
and the robot's drive are ``tests/test_band_on_pegs``.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

TIER = "fast"
"""Pure numpy and dataclasses."""


def _circle(n: int, radius: float = 0.03, centre=(0.0, 0.0, 0.0)) -> np.ndarray:
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.stack([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a), np.full(n, centre[2])], axis=1)


class TestAClosedRodSpec(unittest.TestCase):
    def test_a_loop_has_one_element_per_node_and_its_closing_leg_in_its_length(self) -> None:
        from robosimstudio_materials.rod import RodSpec

        nodes = tuple(map(tuple, _circle(32)))
        spec = RodSpec(name="band", nodes=nodes, closed=True)
        self.assertEqual(32, spec.segments)
        self.assertAlmostEqual(32 * 2 * 0.03 * math.sin(math.pi / 32), spec.length, places=9)
        self.assertEqual(31, RodSpec(name="cord", nodes=nodes).segments, "open, the same nodes are 31 legs")

    def test_a_loop_refuses_what_only_an_end_can_have(self) -> None:
        from robosimstudio_materials.rod import RodSpec

        nodes = tuple(map(tuple, _circle(8)))
        for extra in ({"anchors": ("start",)}, {"joins": (("start", "plug", 0.01),)}, {"stress_free": "straight"}):
            with self.subTest(**{k: str(v) for k, v in extra.items()}):
                with self.assertRaises(ValueError):
                    RodSpec(name="band", nodes=nodes, closed=True, **extra)
        with self.assertRaises(ValueError):
            RodSpec(name="band", closed=True)  # a loop needs its nodes
        with self.assertRaises(ValueError):
            RodSpec(name="band", nodes=nodes[:2], closed=True)
        with self.assertRaises(ValueError):
            RodSpec(name="band", nodes=(*nodes, nodes[0]), closed=True)  # the first node repeated: a zero leg


class TestTheBandConstructor(unittest.TestCase):
    def test_a_flat_band_lies_on_the_bench_round_its_centre(self) -> None:
        from robosimstudio_materials.rod.band import RUBBER, band

        spec = band(diameter=0.06, at=(0.1, -0.2, 0.0), segments=24)
        nodes = np.asarray(spec.nodes)
        self.assertTrue(spec.closed)
        self.assertEqual(24, len(nodes))
        np.testing.assert_allclose(np.linalg.norm(nodes[:, :2] - (0.1, -0.2), axis=1), 0.03, atol=1e-12)
        np.testing.assert_allclose(nodes[:, 2], RUBBER.cord_radius, atol=1e-12)
        self.assertEqual(RUBBER.cord_radius, spec.radius)

    def test_a_standing_band_stands_in_the_plane_normal_to_its_axis(self) -> None:
        from robosimstudio_materials.rod.band import band

        nodes = np.asarray(band(diameter=0.08, at=(0.0, 0.0, 0.2), normal=(1.0, 0.0, 0.0)).nodes)
        np.testing.assert_allclose(nodes[:, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(nodes[:, 1:] - (0.0, 0.2), axis=1), 0.04, atol=1e-12)

    def test_the_material_reaches_the_spec_the_way_the_rod_reads_it(self) -> None:
        """The rod reads friction as sqrt(mu_cable), so the band passes the square."""
        from robosimstudio_materials.rod.band import O_RING, band

        spec = band(material=O_RING)
        self.assertAlmostEqual(O_RING.friction, spec.mu_cable**0.5)
        self.assertEqual(O_RING.young_modulus, spec.young_modulus)
        self.assertEqual(O_RING.friction_falloff, spec.friction_falloff)

    def test_the_o_ring_is_the_stiffer_and_the_rubber_the_stretchier(self) -> None:
        from robosimstudio_materials.rod.band import O_RING, RUBBER

        def axial(m) -> float:
            return m.young_modulus * math.pi * m.cord_radius**2

        def flexural(m) -> float:
            return m.young_modulus * math.pi * m.cord_radius**4 / 4

        self.assertGreater(axial(O_RING), 5 * axial(RUBBER))
        self.assertGreater(flexural(O_RING), 5 * flexural(RUBBER))


class TestTheLoopMeasures(unittest.TestCase):
    def test_once_round_reads_one_beside_reads_zero_twice_reads_two(self) -> None:
        from robosimstudio_materials.rod.band import loop_winding

        loop = _circle(40)
        self.assertAlmostEqual(1.0, loop_winding(loop, (0.0, 0.0)), places=9)
        self.assertAlmostEqual(-1.0, loop_winding(loop[::-1], (0.0, 0.0)), places=9)
        self.assertAlmostEqual(0.0, loop_winding(loop, (0.1, 0.0)), places=9)
        self.assertAlmostEqual(2.0, loop_winding(np.concatenate([loop, loop]), (0.0, 0.0)), places=9)

    def test_a_seam_copy_of_the_first_point_changes_nothing(self) -> None:
        from robosimstudio_materials.rod.band import loop_winding

        loop = _circle(16)
        self.assertAlmostEqual(loop_winding(loop, (0.0, 0.0)), loop_winding(np.vstack([loop, loop[:1]]), (0.0, 0.0)))

    def test_the_winding_does_not_care_how_far_the_loop_is_pulled(self) -> None:
        """Stretched round two pegs 125 mm apart, the band still reads one turn about each."""
        from robosimstudio_materials.rod.band import loop_winding

        top = _circle(20, 0.008, (0.0, 0.0, 0.0))[:11]  # half round each peg, joined by two straight runs
        bottom = _circle(20, 0.008, (0.0, -0.125, 0.0))[10:]
        loop = np.vstack([top, bottom])
        loop = loop[:, [2, 0, 1]]  # the pegs' axes along x
        for z in (0.0, -0.125):
            self.assertAlmostEqual(1.0, abs(loop_winding(loop, (0.0, 0.0, z), (1.0, 0.0, 0.0))), places=6)

    def test_on_a_peg_means_round_it_and_touching_it_within_its_length(self) -> None:
        from robosimstudio_materials.rod.band import encircles

        hung = _circle(32, 0.04, (0.0, -0.032, 0.0))[:, [2, 0, 1]] + (0.06, 0.0, 0.0)  # its top on the peg, 60 mm out
        peg = (np.array([0.10, 0.0, 0.0]), (-1.0, 0.0, 0.0))
        self.assertTrue(encircles(hung, *peg, 0.08), "hung 40 mm from the root of an 80 mm peg")
        self.assertFalse(encircles(hung, *peg, 0.03), "in front of the tip of a 30 mm peg, round only its line")
        self.assertTrue(encircles(hung, *peg), "with no length, the line is enough")
        self.assertFalse(encircles(hung + np.array([0.0, 0.2, 0.0]), *peg, 0.08), "beside the peg")

    def test_a_peg_declared_as_a_rod_feature_is_held_to_its_two_ends(self) -> None:
        """The goal's form: a ``rod`` feature's two ends in world, as the task's posts declare them."""
        from robosimstudio_materials.rod.band import _encircles_rod

        hung = _circle(32, 0.04, (0.0, -0.032, 0.0))[:, [2, 0, 1]] + np.array([0.06, 0.0, 0.0])
        on = _encircles_rod(hung, {"axis": [[0.02, 0.0, 0.0], [0.10, 0.0, 0.0]], "radius": 0.006})
        short = _encircles_rod(hung, {"axis": [[0.07, 0.0, 0.0], [0.10, 0.0, 0.0]], "radius": 0.006})
        self.assertTrue(on["encircles"])
        self.assertAlmostEqual(1.0, abs(on["turns"]), places=6)
        self.assertFalse(short["encircles"], "in front of a 30 mm peg's tip")

    def test_encircles_is_an_atom_on_the_rod_substrate(self) -> None:
        import robosimstudio_materials.rod  # noqa: F401 -- registers it
        from robosimstudio.assetlib.semantics.atoms import get_atom

        atom = get_atom("encircles")
        self.assertEqual(("encircles", "turns"), tuple(atom.returns))
        self.assertEqual(("rod",), tuple(atom.args[0].substrates))


if __name__ == "__main__":
    unittest.main()
