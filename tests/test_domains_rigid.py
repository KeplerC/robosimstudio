# SPDX-License-Identifier: Apache-2.0
"""A rigid object's reset: its declared pose, an episode's override, and only its own coordinates."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

TIER = "fast"
"""No GPU: Warp arrays on the CPU and a stand-in pipeline."""


class TestAnEpisodeThatDoesNotNameTheObject(unittest.TestCase):
    def setUp(self) -> None:
        from robosimstudio.domains.rigid.entity import EntityRigid
        from robosimstudio.specs import ObjectSpec

        wp.init()
        self.entity = EntityRigid(ObjectSpec(name="a", pose=(0.1, 0.2, 0.03)))
        self.entity.coord_start = 2
        self.pipeline = SimpleNamespace(
            state_0=SimpleNamespace(joint_q=wp.zeros(9, dtype=wp.float32, device="cpu")),
            state_1=SimpleNamespace(joint_q=wp.zeros(9, dtype=wp.float32, device="cpu")),
        )

    def _start(self, episode) -> np.ndarray:
        self.entity.reset(self.pipeline, episode)
        return self.pipeline.state_0.joint_q.numpy()[2:5].copy()

    def test_it_returns_to_its_declared_pose(self) -> None:
        expected = self._start(None)
        moved = self._start({"object_poses": {"a": (0.5, 0.5, 0.03, 0.0)}})
        self.assertFalse(np.allclose(moved, expected))
        np.testing.assert_allclose(self._start({"object_poses": {}}), expected, atol=1e-6)
        np.testing.assert_allclose(self.entity.declared_q[:3], (0.1, 0.2, 0.03))

    def test_only_its_own_coordinates_are_written(self) -> None:
        states = (_SliceOnly(9), _SliceOnly(9))
        pipeline = SimpleNamespace(
            state_0=SimpleNamespace(joint_q=states[0]), state_1=SimpleNamespace(joint_q=states[1])
        )
        self.entity.reset(pipeline, {"object_poses": {"a": (0.5, 0.5, 0.03, 0.0)}})
        self.entity.reset(self.pipeline, {"object_poses": {"a": (0.5, 0.5, 0.03, 0.0)}})
        expected = self.pipeline.state_0.joint_q.numpy()[2:9]
        for joint_q in states:
            self.assertEqual(len(joint_q.assigns), 1)
            key, values = joint_q.assigns[0]
            self.assertEqual(key, slice(2, 9))
            self.assertEqual(values.dtype, np.float32)
            self.assertEqual(values.shape, (7,))
            np.testing.assert_allclose(values, expected, atol=1e-6)


class _SliceOnly:
    """A ``joint_q`` that refuses a whole-vector read and records slice assigns."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.assigns: list[tuple[slice, np.ndarray]] = []

    def numpy(self):
        raise AssertionError("copied the whole joint_q to the host")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, key):
        owner = self

        class _View:
            def assign(self, values) -> None:
                owner.assigns.append((key, np.array(values, copy=True)))

        return _View()


if __name__ == "__main__":
    unittest.main()
