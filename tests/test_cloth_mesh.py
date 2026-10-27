# SPDX-License-Identifier: Apache-2.0
"""``robosimstudio_materials/cloth.py``: a membrane mesh is simplified and scaled once on its way to the builder.

No GPU: a recording builder.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

TIER = "fast"
"""No GPU. A stand-in builder."""


class _Builder:
    """Records what ``add_cloth_mesh`` was handed."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def add_cloth_mesh(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestTheMembraneMeshIsSimplified(unittest.TestCase):
    def test_simplify_faces_and_scale_reach_the_builder(self) -> None:
        import trimesh

        import robosimstudio as rss
        from robosimstudio_materials.cloth import ClothMaterialSpec, ClothSpec, EntityCloth

        sphere = trimesh.creation.icosphere(subdivisions=2)  # 320 faces
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sheet.obj"
            sphere.export(path)
            spec = ClothSpec(
                name="shirt",
                mesh="mesh",
                asset=rss.AssetSpec(path=str(path), scale=2.0),
                material=ClothMaterialSpec(shell="membrane"),
                simplify_faces=40,
            )
            entity = EntityCloth(spec)
            builder = _Builder()
            # ``fast_simplification`` is optional: stand in a coarser sphere for the decimated mesh.
            coarse = trimesh.creation.icosphere(subdivisions=1)  # 80 faces
            with mock.patch.object(trimesh.Trimesh, "simplify_quadric_decimation", return_value=coarse):
                entity.build(builder, None)
                want_vertices, want_faces = entity.surface()

        (call,) = builder.calls
        self.assertLess(len(call["indices"]) // 3, len(sphere.faces), "simplify_faces held on a membrane")
        self.assertEqual(len(want_faces) * 3, len(call["indices"]))
        self.assertEqual(len(want_vertices), entity.surface_count)
        # Scaled once: by ``surface``, not again by the builder.
        self.assertEqual(1.0, call["scale"])
        radius = max(float(np.linalg.norm([v[0], v[1], v[2]])) for v in call["vertices"])
        self.assertAlmostEqual(2.0, radius, delta=0.05)


if __name__ == "__main__":
    unittest.main()
