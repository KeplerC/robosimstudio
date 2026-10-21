# SPDX-License-Identifier: Apache-2.0
"""The tape's contact adhesion law against ipc-toolkit itself, on the CPU.

``robosimstudio_materials/tape.py`` ports ipc-toolkit's adhesion model (Fang, Li et al., IEEE TVCG 2024;
ipc-toolkit v1.6.0, ``src/ipc/adhesion/adhesion.cpp``) as warp functions. The numbers below are
ipc-toolkit's own, from its Python wheel ``ipctk`` 1.6.0 in a scratch environment
(``gen_ref.py``): the normal adhesion force on a point over a triangle, ``-grad`` of
``NormalAdhesionPotential(peak, reach, strength, 1)`` over collisions built with the contact
thickness ``dmin``, and ``tangential_adhesion_f1``. The wheel is not a dependency; the table is.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "fast"
"""No GPU: the two warp functions launched on the CPU device."""

NORMAL = (
    # dmin [m], peak [m], reach [m], strength [Pa], centre distance [m], ipctk's pull [Pa]
    (0.0005, 0.0005, 0.001, 4000, 0.00052, 56.576),
    (0.0005, 0.0005, 0.001, 4000, 0.00071666667, 1007.5802),
    (0.0005, 0.0005, 0.001, 4000, 0.00091333333, 2845.5949),
    (0.0005, 0.0005, 0.001, 4000, 0.00111, 3615.5808),
    (0.0005, 0.0005, 0.001, 4000, 0.0013066667, 2268.8844),
    (0.0005, 0.0005, 0.001, 4000, 0.0015033333, 0.0),
    (0.0005, 0.0005, 0.001, 4000, 0.0017, 0.0),
    (0.0005, 0.0002, 0.001, 30000, 0.00061833333, 14612.112),
    (0.0005, 0.0002, 0.001, 30000, 0.000815, 31470.941),
    (0.0005, 0.0002, 0.001, 30000, 0.0010116667, 30215.262),
    (0.0005, 0.0002, 0.001, 30000, 0.0012083333, 23242.681),
    (0.0005, 0.0002, 0.001, 30000, 0.001405, 9441.8395),
    (0.0005, 0.0002, 0.001, 30000, 0.0016016667, 0.0),
    (0.0, 0.001, 0.002, 500, 4e-05, 0.032),
    (0.0, 0.001, 0.002, 500, 0.00043333333, 40.685185),
    (0.0, 0.001, 0.002, 500, 0.00082666667, 282.46281),
    (0.0, 0.001, 0.002, 500, 0.00122, 510.692),
    (0.0, 0.001, 0.002, 500, 0.0016133333, 375.6796),
    (0.0, 0.001, 0.002, 500, 0.0020066667, 0.0),
)

MOLLIFIER = (
    # eps [m], slide y [m], ipctk's tangential_adhesion_f1
    (0.001, 0.0, 0.0),
    (0.001, 0.0004, 0.64),
    (0.001, 0.0008, 0.96),
    (0.001, 0.0012, 0.96),
    (0.001, 0.0016, 0.64),
    (0.001, 0.0024, 0.0),
    (2.5e-05, 5e-06, 0.36),
    (2.5e-05, 1.5e-05, 0.84),
    (2.5e-05, 2.5e-05, 1.0),
    (2.5e-05, 4.5e-05, 0.36),
    (2.5e-05, 5.5e-05, 0.0),
)


class TestThePortMatchesIpcToolkit(unittest.TestCase):
    def test_the_normal_pull_is_ipc_toolkits_to_float32(self) -> None:
        from robosimstudio_materials.tape import adhesion_curve

        rows = np.asarray(NORMAL, dtype=np.float64)
        for params in sorted({tuple(r[:4]) for r in rows}):
            sel = np.all(rows[:, :4] == params, axis=1)
            dmin, peak, reach, strength = params
            pull, _ = adhesion_curve(rows[sel, 4], dmin=dmin, peak=peak, reach=reach, strength=strength)
            # The port agrees with ipctk to 2e-7 of the strength at every point (float32 on the device).
            np.testing.assert_allclose(pull, rows[sel, 5], atol=1e-5 * strength, err_msg=f"params {params}")

    def test_the_pull_peaks_at_the_strength_where_the_presets_put_it(self) -> None:
        from robosimstudio_materials.tape import adhesion_curve

        # A contact thickness equal to the peak lift, as the presets have it: largest at the peak, zero
        # at contact and from the reach on.
        dist = np.linspace(0.0005, 0.0016, 1101)
        pull, _ = adhesion_curve(dist, dmin=0.0005, peak=0.0005, reach=0.001, strength=4000.0)
        self.assertAlmostEqual(float(dist[np.argmax(pull)]), 0.001, delta=2e-6)
        self.assertAlmostEqual(float(pull.max()), 4000.0, delta=1.0)
        self.assertEqual(0.0, float(pull[0]))
        self.assertTrue(np.all(pull[dist > 0.0015 + 1e-6] == 0.0))

    def test_the_tangential_mollifier_is_ipc_toolkits(self) -> None:
        from robosimstudio_materials.tape import adhesion_curve

        rows = np.asarray(MOLLIFIER, dtype=np.float64)
        for eps in sorted(set(rows[:, 0])):
            sel = rows[:, 0] == eps
            _, mollified = adhesion_curve(rows[sel, 1], dmin=0.0005, peak=0.0005, reach=0.001, strength=1.0, eps=eps)
            np.testing.assert_allclose(mollified, rows[sel, 2], atol=1e-6, err_msg=f"eps {eps}")

    def test_a_bond_starts_at_the_pulls_slope(self) -> None:
        from robosimstudio_materials.tape import TapeMaterialSpec, adhesion_curve, contact_stiffness

        material = TapeMaterialSpec(adhesion="contact", strength=4000.0, peak_opening=0.0005, reach=0.001)
        lift = 1.0e-6
        pull, _ = adhesion_curve(np.array([0.0005 + lift]), dmin=0.0005, peak=0.0005, reach=0.001, strength=4000.0)
        self.assertAlmostEqual(
            contact_stiffness(material, 0.0005), float(pull[0]) / lift, delta=0.01 * float(pull[0]) / lift
        )


class TestTheSpecs(unittest.TestCase):
    def test_contact_adhesion_needs_a_peak_inside_the_reach(self) -> None:
        from robosimstudio_materials.tape import TapeMaterialSpec

        with self.assertRaises(ValueError):
            TapeMaterialSpec(adhesion="contact", strength=1.0, peak_opening=0.002, reach=0.001)
        with self.assertRaises(ValueError):
            TapeMaterialSpec(adhesion="glue")

    def test_a_loose_tape_needs_contact_adhesion(self) -> None:
        import robosimstudio as rss

        with self.assertRaises(ValueError):
            rss.tape(name="t", on="box", material=rss.tapes.PAPER_LABEL, stuck=False)


if __name__ == "__main__":
    unittest.main()
