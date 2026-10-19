# SPDX-License-Identifier: Apache-2.0
"""``SuperdexBridge.place_actor`` puts an actor's nodes where it is told, whatever its root.

A soft body sits at an identity root, so its local nodes are world nodes. A shell
-- a sheet or a bag (``robosimstudio_materials.bag``) -- is built about its own
centroid, with the offset in its root transform, because the engine is float32
and a bag solved in bench-height coordinates never converged. A placement
written straight into a bag's local nodes read back a whole centroid away,
0.61 m along x and 0.73 m up, and the next step started from there. Here a bag
and a sponge (an identity root) in one scene are each moved and read back, and a
band, a rod the engine sets no nodes of, is refused by name.
"""

from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

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


@unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed; nothing builds the band")
class TestARodIsRefusedByName(unittest.TestCase):
    """The engine takes node positions on a soft body or a shell only: asked to place a rod, it raised
    from inside ("CDisplacementSlice ... required"), a band or an open cable alike. Refused by name,
    as a body on a rigid core is, and the band is where it was."""

    def test_placing_a_band_is_refused_and_moves_nothing(self) -> None:
        import robosimstudio as rss
        from robosimstudio_materials.rod.band import band

        engine = rss.backend("superdex", substeps=2, feedback="off")
        physics = rss.replace(rss.GENERIC, backends=(("rod", engine), *rss.GENERIC.backends))
        scene = rss.Scene(cell="bench_075", physics=physics, name="place_band")
        loop = scene.add(band(name="band", diameter=0.06, segments=16))
        scene.build()
        bridge = loop.entity.bridge
        entry = _entry(bridge, "band")
        before = np.asarray(entry["read"](), dtype=np.float64).reshape(-1, 3)
        self.assertEqual(17, len(before), "16 nodes and the seam")
        for rows in (before, before[:-1]):  # with the seam row and without it
            with self.subTest(rows=len(rows)), self.assertRaisesRegex(NotImplementedError, "'band' is a rod"):
                bridge.place_actor("band", 0, rows + SHIFT)
        np.testing.assert_array_equal(before, np.asarray(entry["read"](), dtype=np.float64).reshape(-1, 3))


@unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed; nothing builds the bag")
class TestTheBagsContactFalloffIsSaidOnce(unittest.TestCase):
    """The bag's falloff reaches the engine through ``contact_params``; with none stated, the engine's own stands."""

    def test_a_bag_that_states_no_falloff_keeps_the_engines(self) -> None:
        import robosimstudio as rss
        from robosimstudio_materials.bag import BagSpec, build_bag_actor

        scene = rss.Scene(cell="bench_0728", physics=rss.BAG_60HZ, name="bag_falloff")
        scene.add(rss.bag(name="bag", settle_frames=0))
        bridge = scene.build()["bag"].entity.bridge
        record = next(r for r in bridge.recipe["actors"] if r.get("kind") == "bag")
        experimental = bridge.physics.experimental
        built = []
        real = experimental.create_shell_actor
        with mock.patch.object(experimental, "create_shell_actor", lambda s, p: built.append(p) or real(s, p)):
            for falloff in (None, 2.0e-3):
                params = {k: v for k, v in record["params"].items() if k != "friction_falloff"}
                if falloff is not None:
                    params["friction_falloff"] = falloff
                build_bag_actor(bridge, {**record, "name": f"bag_{falloff}", "params": params})
        engine = float(bridge.physics.ContactParams().friction_falloff_vel)
        self.assertAlmostEqual(engine, built[0].contact.friction_falloff_vel, places=6)
        self.assertAlmostEqual(BagSpec().friction_falloff, engine, places=6, msg="the spec's default is the engine's")
        self.assertAlmostEqual(2.0e-3, built[1].contact.friction_falloff_vel, places=6)


if __name__ == "__main__":
    unittest.main()
