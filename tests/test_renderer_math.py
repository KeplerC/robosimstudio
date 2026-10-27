# SPDX-License-Identifier: Apache-2.0
"""The renderers' pure arithmetic: the USD camera pose the RTX tracer is handed, and the gl renderer's sphere budget.

``rtx._usd_camera_matrix`` turns this package's camera-to-world pose (``+y``
down, ``+z`` forward, column vectors) into USD's (``+y`` up, ``-z`` forward,
row vectors): two axis flips and a transpose, each a one-liner that is wrong
in a way that looks right. Held as what it must preserve: a proper rotation,
the same world point for the same camera point, and the same forward.

``gl.sphere_strides`` thins a heap of grains to a sphere budget. Held over two
thousand random heaps: anything that fits is drawn whole, nothing exceeds the
budget by more than one sphere a batch, and the few loose particles beside a
heap stay whole.
"""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio.core.math import quat_from_axis_angle, quat_to_mat

TIER = "fast"
"""4x4 matrices and integer arithmetic: no GPU and no renderer."""


def _poses(count: int, seed: int) -> list[np.ndarray]:
    """Random camera-to-world poses in this package's convention, ``[4, 4]``."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        pose = np.eye(4)
        pose[:3, :3] = quat_to_mat(quat_from_axis_angle(rng.normal(size=3), rng.uniform(0.0, np.pi)))
        pose[:3, 3] = rng.uniform(-2.0, 2.0, 3)
        out.append(pose)
    return out


class TestTheUsdCameraPose(unittest.TestCase):
    def test_its_basis_is_a_proper_rotation_and_its_translation_the_last_row(self) -> None:
        from robosimstudio.solvers.render.rtx import _usd_camera_matrix

        for pose in _poses(20, 0):
            usd = _usd_camera_matrix(pose)
            np.testing.assert_allclose(usd[:3, :3] @ usd[:3, :3].T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(
                1.0, float(np.linalg.det(usd[:3, :3])), places=12, msg="two flips keep it right-handed"
            )
            np.testing.assert_array_equal(pose[:3, 3], usd[3, :3])
            np.testing.assert_array_equal([0.0, 0.0, 0.0, 1.0], usd[:, 3])

    def test_a_point_in_front_of_the_camera_lands_where_it_did(self) -> None:
        """A camera point ``(x, y, z)`` here is ``(x, -y, -z)`` to USD; through each pose it is one world point."""
        from robosimstudio.solvers.render.rtx import _usd_camera_matrix

        rng = np.random.default_rng(1)
        for pose in _poses(20, 2):
            usd = _usd_camera_matrix(pose)
            x, y = rng.uniform(-1.0, 1.0, 2)
            z = rng.uniform(0.1, 3.0)
            world = pose @ np.array([x, y, z, 1.0])
            np.testing.assert_allclose(np.array([x, -y, -z, 1.0]) @ usd, world, atol=1e-12)

    def test_the_cameras_forward_is_usds_minus_z_and_its_down_is_usds_minus_y(self) -> None:
        from robosimstudio.solvers.render.rtx import _usd_camera_matrix

        for pose in _poses(10, 3):
            usd = _usd_camera_matrix(pose)
            np.testing.assert_allclose(-usd[2, :3], pose[:3, 2], atol=1e-12)
            np.testing.assert_allclose(-usd[1, :3], pose[:3, 1], atol=1e-12)


def _heaps(count: int, seed: int) -> list[list[int]]:
    """Random scenes of particle bodies: a few heaps, some loose particles, sometimes many small bodies."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        kind = rng.integers(3)
        if kind == 0:
            sizes = rng.integers(1, 300_000, rng.integers(1, 6))
        elif kind == 1:
            sizes = np.concatenate([rng.integers(1, 999, rng.integers(1, 30)), rng.integers(1000, 400_000, 2)])
        else:
            sizes = rng.integers(1, 999, rng.integers(50, 200))
        out.append([int(n) for n in sizes])
    return out


class TestTheSphereBudget(unittest.TestCase):
    def test_what_fits_is_drawn_whole(self) -> None:
        from robosimstudio.solvers.render.gl import sphere_strides

        self.assertEqual([1, 1, 1], sphere_strides([10, 20_000, 29_970], budget=50_000))
        self.assertEqual([], sphere_strides([], budget=50_000))

    def test_any_scene_is_drawn_within_the_budget_and_one_sphere_a_batch(self) -> None:
        from robosimstudio.solvers.render.gl import sphere_strides

        budget = 50_000
        for sizes in _heaps(2000, 4):
            strides = sphere_strides(sizes, budget=budget)
            self.assertEqual(len(sizes), len(strides))
            self.assertTrue(all(k >= 1 for k in strides), (sizes, strides))
            drawn = sum(-(-n // k) for n, k in zip(sizes, strides, strict=True))
            self.assertLessEqual(drawn, budget + len(sizes), (sizes, strides))

    def test_loose_particles_beside_a_heap_stay_whole_unless_they_fill_half_the_budget(self) -> None:
        from robosimstudio.solvers.render.gl import SMALL_BATCH, sphere_strides

        for sizes in _heaps(2000, 5):
            strides = sphere_strides(sizes, budget=50_000)
            if sum(sizes) <= 50_000:
                continue
            small = sum(n for n in sizes if n < SMALL_BATCH)
            if small > 25_000:
                self.assertEqual(1, len(set(strides)), "every batch thinned alike")
            else:
                self.assertTrue(all(k == 1 for n, k in zip(sizes, strides, strict=True) if n < SMALL_BATCH))


if __name__ == "__main__":
    unittest.main()
