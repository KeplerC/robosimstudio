# SPDX-License-Identifier: Apache-2.0
"""``SuperdexBridge.place_actor`` puts an actor's nodes where it is told, whatever its root.

Every actor the bridge's own builders make sits at an identity root, so its local
nodes are world nodes. A bag (``robosimstudio_materials.bag``) is built about its
own centroid, with the offset in its root transform, because the engine is
float32 and a bag solved in bench-height coordinates never converged. A
placement written straight into its local nodes read back a whole centroid away,
0.61 m along x and 0.73 m up, and the next step started from there. Here a bag
and a sponge (an identity root) in one scene are each moved and read back.
"""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

TIER = "medium"
"""One small SuperDex scene, built once: a bag and a sponge on a bench, no robot."""

HAS_SUPERDEX = importlib.util.find_spec("superdex") is not None
SHIFT = np.array([0.05, -0.03, 0.02])
"""How far each actor is placed from where it lay [m]."""


def _entry(bridge, entity: str) -> dict:
    return next(e for e in bridge.actors if e["entity"] == entity)


@unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed; nothing builds the bag")
class TestPlacingAnActorReadsItBackWherePlaced(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        scene = rss.Scene(cell="bench_0728", physics=rss.BAG_60HZ, name="place_actors")
        scene.add(rss.bag(name="bag", settle_frames=10))
        scene.add(rss.sponge(name="sponge", at=(0.0, 0.25, 0.0), settle_frames=10))
        cls.sim = scene.build()
        cls.bridge = cls.sim["bag"].entity.bridge

    def _place(self, entity: str) -> tuple[np.ndarray, dict]:
        entry = _entry(self.bridge, entity)
        target = np.asarray(entry["read"](), dtype=np.float64).reshape(-1, 3) + SHIFT
        self.assertTrue(self.bridge.place_actor(entity, 0, target))
        return target, entry

    def test_a_bag_built_about_its_centroid_lands_where_it_is_placed(self) -> None:
        root = np.asarray(list(_entry(self.bridge, "bag")["actor"].get_root_transform().translation))
        self.assertGreater(np.linalg.norm(root), 0.5, "the bag's root is its centroid, not identity")
        target, entry = self._place("bag")
        read = np.asarray(entry["read"](), dtype=np.float64).reshape(-1, 3)
        np.testing.assert_allclose(read, target, atol=1.0e-5)
        np.testing.assert_allclose(np.asarray(entry["prev"], dtype=np.float64), target, atol=1.0e-5)
        self.sim.run(1).sync()
        moved = np.asarray(self.sim["bag"].particle_q, dtype=np.float64).mean(axis=0) - target.mean(axis=0)
        self.assertLess(float(np.abs(moved).max()), 0.005, f"one step later the bag is {moved} m from its placement")

    def test_an_actor_at_an_identity_root_places_as_before(self) -> None:
        target, entry = self._place("sponge")
        np.testing.assert_allclose(np.asarray(entry["read"](), dtype=np.float64).reshape(-1, 3), target, atol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
