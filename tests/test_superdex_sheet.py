# SPDX-License-Identifier: Apache-2.0
"""A cloth on SuperDex is a shell in a frame at its own centroid, and every path through the bridge agrees.

The bridge's shell builder used to build a sheet in world coordinates at an identity root, which
the float32 engine solves worst at bench height (a bag there never converged; see
``robosimstudio_materials.bag``, section 1), and handed it the engine's contact (mu 0.5) whatever
the cloth said. It now builds every sheet about its centroid with the offset in the root, reads
the nodes back through the root, and states the cloth's own friction. What moved with the root is
checked here on the engine, at two worlds, each its own scene: the read-back against what Newton
holds, a pin (the engine takes its target in the world), a placement, and a reset.
``tests/test_superdex_shell.py`` checks the builder against a stand-in engine.
"""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

TIER = "medium"
"""One small SuperDex scene, built once at two worlds: a sheet on a bench, no robot."""

HAS_SUPERDEX = importlib.util.find_spec("superdex") is not None
SHIFT = np.array([0.05, -0.03, 0.02])
"""How far the sheet is placed from where it lay [m]."""


@unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed; nothing builds the sheet")
class TestASheetOnSuperdex(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss
        from robosimstudio.domains.capabilities import host_bridge

        physics = rss.replace(
            rss.GENERIC, backends=(("cloth", rss.backend("superdex", substeps=2)), *rss.GENERIC.backends)
        )
        scene = rss.Scene(cell="bench_0728", physics=physics, name="sheet_superdex")
        scene.add(rss.sheet(name="towel", size=(0.2, 0.2), cells=(10, 10), at=(-0.1, -0.1, 0.02), settle_frames=10))
        cls.sim = scene.build(num_worlds=2)
        cls.bridge = host_bridge(cls.sim.pipeline, "towel")
        cls.bridge.restore_rest()
        cls.rest = {e["world"]: cls._read(e) for e in cls.bridge.actors}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def setUp(self) -> None:
        self.bridge.restore_rest()

    @staticmethod
    def _read(entry: dict) -> np.ndarray:
        return np.asarray(entry["read"](), dtype=np.float64).reshape(-1, 3)

    def _entry(self, world: int) -> dict:
        return next(e for e in self.bridge.actors if e["entity"] == "towel" and e["world"] == world)

    def test_each_worlds_sheet_is_rooted_at_its_centroid_and_reads_back_where_newton_holds_it(self) -> None:
        self.assertEqual(2, len(self.bridge.scenes), "one scene per world")
        self.sim.run(3).sync()
        held = self.sim.pipeline.state_0.particle_q.numpy().astype(np.float64)
        for world in (0, 1):
            with self.subTest(world=world):
                entry = self._entry(world)
                root = np.asarray(list(entry["actor"].get_root_transform().translation))
                self.assertGreater(np.linalg.norm(root), 0.5, "the root is the sheet's centroid on the bench")
                np.testing.assert_allclose(root, np.asarray(entry["prev"], np.float64).mean(axis=0), atol=0.03)
                start, count = int(entry["start"]), int(entry["count"])
                np.testing.assert_allclose(self._read(entry), held[start : start + count], atol=1.0e-5)

    def test_the_cloths_own_friction_reaches_the_engine(self) -> None:
        import robosimstudio as rss

        for world in (0, 1):
            contact = self._entry(world)["actor"].get_contact_params()
            self.assertAlmostEqual(rss.ClothMaterialSpec().friction, contact.coulomb_friction_coefficient, places=6)

    def test_a_pin_holds_a_node_at_its_world_target(self) -> None:
        """The engine takes a node constraint's target in the world, whatever the actor's root."""
        entry = self._entry(1)
        target = self._read(entry)[0] + np.array([0.0, 0.0, 0.03])
        pin = self.bridge.pin_nodes("towel", np.array([int(entry["start"])]), target[None, :], world=1)
        try:
            self.sim.run(60).sync()
            self.assertLess(np.linalg.norm(self._read(entry)[0] - target), 5.0e-4)
        finally:
            self.bridge.unpin(pin)

    def test_a_placed_sheet_lands_where_placed_and_a_reset_brings_it_back(self) -> None:
        entry = self._entry(1)
        target = self._read(entry) + SHIFT
        self.assertTrue(self.bridge.place_actor("towel", 1, target))
        np.testing.assert_allclose(self._read(entry), target, atol=1.0e-5)
        np.testing.assert_allclose(np.asarray(entry["prev"], np.float64), target, atol=1.0e-5)
        self.sim.run(5).sync()
        self.bridge.restore_rest()
        np.testing.assert_allclose(self._read(entry), self.rest[1], atol=1.0e-6)
        self.assertTrue(self.sim.state_is_finite())


if __name__ == "__main__":
    unittest.main()
