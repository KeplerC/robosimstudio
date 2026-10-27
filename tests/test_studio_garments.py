# SPDX-License-Identifier: Apache-2.0
"""The studio viewer's per-frame costs: no GLB kept for a deforming garment, a pushed static batch skipped."""

from __future__ import annotations

import types
import unittest

import numpy as np

from robosimstudio.core.render.finish import GLOSS
from robosimstudio_studio import finish_gltf
from robosimstudio_studio.garments import ViewerStudio

TIER = "fast"
"""No GPU, no browser: a viewer built without its constructor."""


class TestDeformingGlbIsNotKept(unittest.TestCase):
    def test_an_uncached_encode_leaves_the_cache_as_it_was(self) -> None:
        finish_gltf.clear_cache()
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        for frame in range(3):
            moved = points + 0.01 * frame
            blob = finish_gltf.glb_for(moved, faces, (0.5, 0.5, 0.5), GLOSS, cache=False)
            if blob is None:
                self.skipTest("no trimesh: nothing is encoded")
        self.assertEqual(len(finish_gltf._CACHE), 0)


class TestStaticPbrBatch(unittest.TestCase):
    def _viewer(self) -> ViewerStudio:
        viewer = ViewerStudio.__new__(ViewerStudio)
        viewer._qualify = lambda name: name
        viewer._static_skip_ok = True
        viewer._static_batches = {"/b"}
        viewer._scene_handles = {}
        viewer._pbr_handles = {"/b": {"/b/m0": object()}}
        viewer._finishes = {0: GLOSS}
        viewer._host_shape_color = None
        return viewer

    def test_a_pushed_pbr_batch_is_skipped(self) -> None:
        viewer = self._viewer()
        drawn = []
        viewer._log_pbr_instances = lambda *a: drawn.append(a) or True
        viewer.log_instances("/b", "/mesh", None, None, None, None)
        self.assertEqual(drawn, [])

    def test_colours_are_read_back_once_per_repack(self) -> None:
        viewer = self._viewer()
        reads = []

        class _Colors:
            def numpy(self):
                reads.append(1)
                return np.ones((2, 3), dtype=np.float32)

        viewer.model = types.SimpleNamespace(shape_color=_Colors())
        viewer._meshes = {"/mesh": object()}
        viewer._plane_meshes = set()
        viewer._live_shapes = frozenset()
        viewer._batch_shapes = lambda q: [0, 1]
        viewer._pbr_groups = {}
        viewer._glb = lambda data, color, finish: None  # stop after the colours
        viewer._retire_pbr_nodes = lambda *a: None
        xforms = np.zeros((2, 7), dtype=np.float32)
        xforms[:, 6] = 1.0
        for _ in range(3):
            viewer._log_pbr_instances("/b", "/mesh", xforms, None)
        self.assertEqual(len(reads), 1)
        viewer.mark_shape_colors_dirty()
        viewer._log_pbr_instances("/b", "/mesh", xforms, None)
        self.assertEqual(len(reads), 2)


if __name__ == "__main__":
    unittest.main()
