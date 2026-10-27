# SPDX-License-Identifier: Apache-2.0
"""``robosimstudio_materials/cloth.py``: a sheet zeroes only its own creases, in every world it is copied into.

No GPU: a model made of CPU warp arrays.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

TIER = "fast"
"""No GPU. A stand-in model of CPU warp arrays."""


class TestZeroRestAngleIsPerSheet(unittest.TestCase):
    def test_one_sheet_does_not_flatten_anothers_creases(self) -> None:
        import warp as wp

        from robosimstudio_materials.cloth import ClothMaterialSpec, ClothSpec, EntityCloth

        wp.init()
        # Two sheets of four particles each; one edge per sheet, column 2 on the sheet.
        edges = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
        model = SimpleNamespace(
            edge_indices=wp.array(edges.reshape(-1), dtype=wp.int32, device="cpu"),
            edge_rest_angle=wp.array(np.array([0.5, 0.7], dtype=np.float32), dtype=wp.float32, device="cpu"),
        )
        flat = EntityCloth(ClothSpec(name="flat", material=ClothMaterialSpec(zero_rest_angle=True)))
        flat.range.particles = (0, 4)
        creased = EntityCloth(ClothSpec(name="creased", material=ClothMaterialSpec(zero_rest_angle=False)))
        creased.range.particles = (4, 8)

        flat.apply_material(model)
        creased.apply_material(model)

        np.testing.assert_allclose([0.0, 0.7], model.edge_rest_angle.numpy())

    def test_every_world_of_a_replicated_sheet_is_flattened(self) -> None:
        import warp as wp

        from robosimstudio_materials.cloth import ClothMaterialSpec, ClothSpec, EntityCloth

        wp.init()
        # Two worlds of eight particles: in each, a flat sheet on 0..3 and a creased one on 4..7.
        edges = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], dtype=np.int32)
        model = SimpleNamespace(
            edge_indices=wp.array(edges.reshape(-1), dtype=wp.int32, device="cpu"),
            edge_rest_angle=wp.array(np.array([0.5, 0.7, 0.5, 0.7], dtype=np.float32), dtype=wp.float32, device="cpu"),
        )
        worlds = SimpleNamespace(num_worlds=2, particles_per_world=8)
        flat = EntityCloth(ClothSpec(name="flat", material=ClothMaterialSpec(zero_rest_angle=True)))
        flat.range.particles = (0, 4)
        flat.worlds = worlds
        creased = EntityCloth(ClothSpec(name="creased", material=ClothMaterialSpec(zero_rest_angle=False)))
        creased.range.particles = (4, 8)
        creased.worlds = worlds

        flat.apply_material(model)
        creased.apply_material(model)

        np.testing.assert_allclose([0.0, 0.7, 0.0, 0.7], model.edge_rest_angle.numpy())


if __name__ == "__main__":
    unittest.main()
