# SPDX-License-Identifier: Apache-2.0
"""``CameraSpec``: its intrinsics, and the copies it makes of itself.

A field of view and the focal length it implies give one K; a rescaled camera
keeps its field of view; ``scaled`` and ``with_pose`` keep every field of a
subclass and its type.
"""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from robosimstudio.core.cameras import CameraSpec

TIER = "fast"
"""No GPU: a frozen dataclass and numpy."""


class TestCameraSpec(unittest.TestCase):
    """Intrinsics, and the aperture form the tiled backend renders through."""

    def test_fov_and_explicit_intrinsics_agree(self):
        """A field of view and the focal length it implies give the same K."""
        by_fov = CameraSpec(name="c", width=640, height=480, fov_y=45.0)
        f = by_fov.intrinsics[1, 1]
        by_k = CameraSpec(name="c", width=640, height=480, fx=f, fy=f)
        np.testing.assert_allclose(by_fov.intrinsics, by_k.intrinsics, atol=1e-9)

    def test_scaled_preserves_field_of_view(self):
        """Rescaling a camera scales K and leaves the field of view alone."""
        spec = CameraSpec(name="c", width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        half = spec.scaled(320, 240)
        self.assertAlmostEqual(spec.fov_y_from_intrinsics, half.fov_y_from_intrinsics, places=9)
        self.assertAlmostEqual(half.intrinsics[0, 0], 250.0, places=9)

    def test_unknown_channel_raises(self):
        """A camera asking for a channel no backend defines is an error."""
        with self.assertRaises(ValueError):
            CameraSpec(name="c", channels=("rgb", "normals"))


class TestCameraCopiesKeepEveryField(unittest.TestCase):
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class _Tagged(CameraSpec):
        tag: int = 0

    def test_scaled_keeps_fields_and_type(self):
        cam = self._Tagged(width=64, height=48, fx=50.0, cx=30.0, tag=7)
        small = cam.scaled(32, 24)
        self.assertIs(type(small), self._Tagged)
        self.assertEqual(small.tag, 7)
        np.testing.assert_allclose(small.intrinsics, [[25.0, 0, 15.0], [0, 25.0, 12.0], [0, 0, 1]])

    def test_with_pose_keeps_fields_and_drops_xform(self):
        cam = self._Tagged(xform=(0, 0, 1, 0, 0, 0, 1), tag=3)
        moved = cam.with_pose((1, 2, 3), (0, 0, 0))
        self.assertIs(type(moved), self._Tagged)
        self.assertEqual(moved.tag, 3)
        self.assertIsNone(moved.xform)
        self.assertEqual(moved.eye, (1.0, 2.0, 3.0))


if __name__ == "__main__":
    unittest.main()
