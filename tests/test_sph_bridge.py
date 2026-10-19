# SPDX-License-Identifier: Apache-2.0
"""Genesis's SPH bridge on its own: where spilled water rests.

``tests/test_pour_water.py`` is the bridge in the example's scene; this is what
that scene never does. Water on a void cell has nothing under it but the ground
plane, which the bridge does not build as a body.

Needs the ``[genesis]`` extra, which lives in a virtual environment of its own
(``pyproject.toml``); it skips without it, as ``tests/test_genesis.py`` does.
"""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

TIER = "medium"
"""One GPU, the [genesis] extra: a 64-particle scene."""


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


HAS_GENESIS = _importable("genesis")


@unittest.skipUnless(HAS_GENESIS, "the [genesis] extra is not installed")
class TestSpilledWaterRestsOnTheGround(unittest.TestCase):
    """A block of water dropped 30 mm onto a void cell's ground plane lies on it, not 218 mm under it."""

    def test_the_water_lies_on_the_plane(self) -> None:
        import robosimstudio as rss

        scene = rss.Scene(cell="void", physics=rss.WATER_60HZ)
        water = scene.add(rss.water(name="water", size=(0.016, 0.016, 0.016), at=(0.0, 0.0, 0.03), settle_frames=0))
        sim = scene.build()
        sim.run(90).sync()
        z = np.asarray(water.particle_q, dtype=np.float64)[:, 2]
        self.assertTrue(sim.state_is_finite())
        self.assertGreaterEqual(float(z.min()), 0.0, f"water stands {1e3 * z.min():.1f} mm, under the ground")
        self.assertLess(float(z.max()), 0.02, f"water stands {1e3 * z.max():.1f} mm, still falling")


if __name__ == "__main__":
    unittest.main()
