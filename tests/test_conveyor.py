# SPDX-License-Identifier: Apache-2.0
"""``robosimstudio/domains/conveyor``: the slat path and the spec, with nothing built.

Whether a belt carries a box and a mailer is ``tests/test_conveyor_carries.py``. Here: a slat's
pose is continuous all the way round its loop, its velocity is the derivative of that pose,
the top run moves at the belt's speed and nothing else does, and a spec that cannot be built is
refused by name.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

TIER = "fast"
"""Numpy and the spec classes only."""

L, D, DROP, K = 1.4, 0.03, 0.04, 8.0


def _path(s):
    from robosimstudio.domains.conveyor import slat_path

    return slat_path(s, L, D, DROP, K)


class TestTheSlatPath(unittest.TestCase):
    def test_the_loop_is_continuous_and_closes(self) -> None:
        from robosimstudio.domains.conveyor import ConveyorSpec

        period = ConveyorSpec(length=L, dive=D, drop=DROP, return_ratio=K).period
        s = np.linspace(0.0, period, 20001)
        xz = np.array([_path(v)[:2] for v in s])
        step = np.linalg.norm(np.diff(xz, axis=0), axis=1)
        # The fastest part of the loop (the return, K times the belt) moves K * ds per ds.
        self.assertLess(step.max(), K * (s[1] - s[0]) * 1.01, "the path jumps")
        np.testing.assert_allclose(xz[0], xz[-1], atol=1e-9)

    def test_the_velocity_is_the_derivative_of_the_pose(self) -> None:
        h = 1e-6
        for s in (0.3, L + 0.5 * D, L + D + 0.05, L + D + (L + 2 * D) / K + 0.5 * D):
            with self.subTest(s=s):
                x0, z0, dx, dz = _path(s)
                x1, z1, _, _ = _path(s + h)
                self.assertAlmostEqual(dx, (x1 - x0) / h, places=4)
                self.assertAlmostEqual(dz, (z1 - z0) / h, places=4)

    def test_only_the_top_run_is_at_the_surface(self) -> None:
        for s in np.linspace(0.0, 4.0, 4001):
            x, z, dx, _ = _path(s)
            if z > -1e-12:
                self.assertTrue(-1e-9 <= x <= L + 1e-9, (s, x))
                self.assertAlmostEqual(1.0, dx, msg="a slat at the surface moves at the belt's speed")
            else:
                self.assertLessEqual(z, 0.0)
                self.assertGreaterEqual(z, -DROP - 1e-12)

    def test_the_return_runs_under_the_top_at_its_ratio(self) -> None:
        _, z, dx, dz = _path(L + D + 0.01)
        self.assertAlmostEqual(-DROP, z)
        self.assertAlmostEqual(-K, dx)
        self.assertEqual(0.0, dz)


class TestTheSpec(unittest.TestCase):
    def test_the_constructor_and_the_flat_names(self) -> None:
        import robosimstudio as rss
        from robosimstudio.domains.conveyor import ConveyorSpec

        spec = rss.conveyor(name="infeed", at=(0.1, 0.0, 0.06), yaw=math.pi / 2, speed=0.15)
        self.assertIsInstance(spec, ConveyorSpec)
        self.assertIs(rss.ConveyorSpec, ConveyorSpec)
        self.assertEqual("conveyor", spec.entity)
        self.assertEqual("conveyor", spec.sim.domain)
        self.assertEqual((0.1, 0.0, 0.06), spec.at)

    def test_a_spec_that_cannot_be_built_is_refused(self) -> None:
        from robosimstudio.domains.conveyor import ConveyorSpec

        for bad, word in (
            ({"length": 0.0}, "length"),
            ({"pitch": -0.1}, "pitch"),
            ({"gap": 0.08, "pitch": 0.08}, "gap"),
            ({"drop": 0.01, "thickness": 0.012}, "drop"),
        ):
            with self.subTest(**bad), self.assertRaisesRegex(ValueError, word):
                ConveyorSpec(**bad)

    def test_the_domain_is_registered_as_a_fixture(self) -> None:
        import robosimstudio.domains.conveyor  # noqa: F401
        from robosimstudio.specs.registries import get_domain

        info = get_domain("conveyor")
        self.assertEqual("fixtures", info.role)
        self.assertIn("deformables", info.contacts)


if __name__ == "__main__":
    unittest.main()
