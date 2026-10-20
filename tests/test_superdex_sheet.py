# SPDX-License-Identifier: Apache-2.0
"""A cloth on SuperDex is a shell in a frame at its own centroid, and every path through the bridge agrees.

The bridge's shell builder used to build a sheet in world coordinates at an identity root, which
the float32 engine solves worst at bench height (a bag there never converged; see
``robosimstudio_materials.bag``, section 1), and handed it the engine's contact (mu 0.5) whatever
the cloth said. It now builds every sheet about its centroid with the offset in the root, reads
the nodes back through the root, and states the cloth's own friction. What moved with the root is
checked here on the engine, at two worlds, each its own scene: the read-back against what Newton
holds, a pin (the engine takes its target in the world), a placement, and a reset. And a weld
stated on a sheet or a bag, which the bridge's weld pass builds for a shell as for a soft body.
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


@unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed; nothing builds the sheet")
class TestASheetOverAStaticBox(unittest.TestCase):
    """A towel on SuperDex dropped over a static 10 cm box, no robot, the rest of GENERIC.

    The scene did neither of two things. It did not build: a static box was then a body with no
    joint, and a model with none failed at its first reset (``tests/test_a_model_without_joints.py``),
    then twice more where the episode reset reads joint coordinates. And with an arm beside it,
    where it built, the box was no mirror on SuperDex -- a body on VBD, an island of particles too,
    got no channel -- so the towel lay through it on the bench, its node over the box at 0.751 m;
    0.855 m now, on the box's 0.849 m top (``c5a_drape.py``), and 0.854 m with no arm
    (``c5b_drape.py``). The box was not held to its pose then either: VBD integrated the jointless
    body, which slid 9 mm under the towel in those two seconds. A static box is now welded to the
    world and kinematic (``tests/test_a_static_box_stays_put.py``), so it stays put and the model
    has a joint; the mirror channel is what this class still checks.
    """

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss
        from robosimstudio.domains.capabilities import host_bridge

        physics = rss.replace(
            rss.GENERIC, backends=(("cloth", rss.backend("superdex", substeps=2)), *rss.GENERIC.backends)
        )
        scene = rss.Scene(cell="bench_075", physics=physics, name="sheet_over_a_box")
        scene.add(rss.sheet(name="towel", size=(0.3, 0.3), cells=(20, 20), at=(-0.15, -0.15, 0.12), settle_frames=0))
        cls.box = scene.add(rss.box(name="box", half=(0.05, 0.05, 0.05), at=(0.0, 0.0, 0.05), static=True))
        cls.sim = scene.build()
        cls.bridge = host_bridge(cls.sim.pipeline, "towel")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def test_the_box_is_a_mirror_and_the_towel_lies_on_it(self) -> None:
        labels = list(self.sim.model.shape_label)
        self.assertIn("box", [labels[m.shape].rsplit("/", 1)[-1] for m in self.bridge.mirrors])
        self.sim.run(120).sync()
        q = np.asarray(self.bridge.actors[0]["read"](), dtype=np.float64).reshape(-1, 3)
        box = np.asarray(self.box.position, dtype=np.float64)
        over = q[np.argmin(np.linalg.norm(q[:, :2] - box[:2], axis=1))]
        self.assertGreater(over[2], box[2] + 0.05, "the towel lies on the box's top, not through it")

    def test_it_resets_and_steps_again(self) -> None:
        for _ in range(2):
            self.sim.reset()
            self.sim.run(5).sync()
        self.assertTrue(self.sim.state_is_finite())


class _Arr:
    def __init__(self, a) -> None:
        self.a = np.asarray(a)

    def numpy(self):
        return self.a


def _grid(origin, n: int = 5, size: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """A flat ``n`` x ``n`` sheet, its corner at ``origin``: vertices and triangles."""
    xs = np.linspace(0.0, size, n + 1)
    vertices = np.array([(x, y, 0.0) for x in xs for y in xs]) + np.asarray(origin)
    tris = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            tris += [(a, a + n + 1, a + 1), (a + n + 1, a + n + 2, a + 1)]
    return vertices, np.asarray(tris)


@unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed; nothing builds the sheet")
class TestAWeldStatedOnASheetHolds(unittest.TestCase):
    """The bridge's weld pass joins a sheet's or a bag's stated nodes as it does a soft body's.

    So both builders declare ``welds``. On the engine alone: two 0.1 m sheets at bench height, the
    second's corner on a node of the first, whose x = 0.6 m edge is pinned. The second hangs off the
    first only by a weld stated in the first's record, kind ``shell`` or ``bag``: measured 0.000 mm
    apart after a second, the welded corner sagging 83 mm with the first sheet's free edge, where
    unwelded it falls 4.9 m (``c1_shell_weld.py``).
    """

    def _hang(self, kind: str, weld: bool) -> tuple[float, float, int]:
        from types import SimpleNamespace

        import robosimstudio_materials.bag  # noqa: F401 -- registers the bag's builder
        from robosimstudio.engines.superdex.bridge import SuperdexBridge

        va, ta = _grid((0.6, 0.0, 0.8))
        vb, tb = _grid((0.7, 0.0, 0.8))
        joined = int(np.argmin(np.linalg.norm(va - vb[0], axis=1)))
        empty = _Arr(np.zeros((0, 7)))
        model = SimpleNamespace(
            particle_q=_Arr(np.vstack([va, vb]).astype(np.float32)),
            tri_indices=_Arr(np.vstack([ta, tb + len(va)])),
            tri_count=len(ta) + len(tb),
            tet_count=0,
            body_count=0,
            shape_count=0,
            **{f"shape_{key}": empty for key in ("type", "scale", "transform", "body", "material_mu")},
        )
        sheet = {"thickness": 0.004, "areal_density": 0.3, "young": 1.0e5, "poisson": 0.3, "self_contact": False}
        stated = {**sheet, "welds": [{"to": "b", "nodes": [joined], "contact": False}]} if weld else sheet
        record = {"kind": kind, "particle_start": 0, "particle_count": len(va), "params": stated}
        recipe = {
            "island": "weld",
            "gravity": [0.0, 0.0, -9.81],
            "actors": [
                {**record, "name": "a", "entity": "a"},
                {"name": "b", "entity": "b", "kind": "shell", "particle_start": len(va), "particle_count": len(vb)},
            ],
            "mirrors": [],
        }
        recipe["actors"][1]["params"] = sheet  # the hung sheet states no weld of its own
        bridge = SuperdexBridge(model, recipe)
        edge = np.flatnonzero(va[:, 0] < 0.6 + 1.0e-9)
        bridge.pin_nodes("a", edge, va[edge], stiffness=1.0e4, damping=10.0)
        for _ in range(240):
            bridge.scene.step(1.0 / 240.0)
        a, b = (np.asarray(e["read"](), dtype=np.float64).reshape(-1, 3) for e in bridge.actors)
        return float(np.linalg.norm(a[joined] - b[0])), float(vb[0, 2] - b[0, 2]), len(bridge.welds)

    def test_a_sheet_and_a_bag_each_hold_what_they_weld(self) -> None:
        for kind in ("shell", "bag"):
            with self.subTest(kind=kind):
                gap, drop, welds = self._hang(kind, weld=True)
                self.assertEqual(1, welds)
                self.assertLess(gap, 1.0e-4, "the welded pair stays together")
                self.assertLess(drop, 0.2, "the second sheet hangs off the first")
        _, drop, welds = self._hang("shell", weld=False)
        self.assertEqual((0, True), (welds, drop > 1.0), "unwelded, the second sheet falls")


if __name__ == "__main__":
    unittest.main()
