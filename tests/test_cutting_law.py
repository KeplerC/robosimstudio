# SPDX-License-Identifier: Apache-2.0
"""The cutting kind's rules, read on a CPU: the frames, the box, the cut's line and the edge law.

``robosimstudio_materials/cutting/law.py`` holds every rule the kind applies between a knife and a
block in plain numpy; the cage solver it drives is exercised by ``tests/test_cut_tofu``. These pin
the arithmetic: a frame that round-trips, a cage whose faces point out, a level knife's line turned
2 degrees off the faces it crosses, an edge clipped to the block's footprint, and a push that grows
with the press, caps at the blade's threshold while it cuts, and lets the block part when the edge
meets the board.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from tests.fixtures.packs import has_pack

TIER = "fast"
"""No GPU: numpy, and a builder on the CPU that is never finalized."""


@unittest.skipUnless(has_pack("cutting"), "the cutting kind is not installed")
class TestTheLaw(unittest.TestCase):
    def setUp(self) -> None:
        from robosimstudio_materials.cutting import law

        self.law = law

    def test_a_block_frame_round_trips(self) -> None:
        frame = self.law.BlockFrame.from_base((0.6, -0.16, 0.762, 0.0, 0.0, 0.3826834, 0.9238795), (0.06, 0.10, 0.04))
        points = np.array([[0.61, -0.15, 0.78], [0.55, -0.2, 0.8]])
        back = frame.from_cutter(frame.to_cutter(points))
        np.testing.assert_allclose(points, back, atol=1e-12)
        self.assertAlmostEqual(10.0, frame.scale)
        base_middle = frame.to_block([0.6, -0.16, 0.762])
        np.testing.assert_allclose([0.03, 0.05, 0.0], base_middle, atol=1e-12)

    def test_the_cage_faces_point_out_and_the_skin_is_welded(self) -> None:
        verts, faces = self.law.box_cage((0.6, 1.0, 0.4))
        volume = 0.0
        for face in faces:
            p = verts[face]
            for i in range(1, len(face) - 1):
                volume += float(np.dot(p[0], np.cross(p[i], p[i + 1]))) / 6.0
        self.assertAlmostEqual(0.24, volume)
        skin, tris = self.law.box_surface((0.6, 1.0, 0.4), 0.05)
        self.assertEqual(len(skin), len(np.unique(np.round(skin, 9), axis=0)), "no duplicate skin vertices")
        self.assertEqual(int(tris.max()) + 1, len(skin))

    def test_a_level_knife_s_line_is_turned_two_degrees(self) -> None:
        normals = np.array([[0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0]], float)
        crossed = np.array([True, True, False, False, True, True])
        line, sweep, normal, turn = self.law.cut_line([0, 0.05, 0.05], [0.12, 0.05, 0.05], [0, 0, -1], normals, crossed)
        self.assertAlmostEqual(2.0, abs(math.degrees(turn)))
        self.assertGreaterEqual(float(np.min(np.abs(normals[crossed] @ line))), math.sin(math.radians(2.0)) - 1e-12)
        np.testing.assert_allclose([0, 1, 0], np.abs(normal), atol=1e-12)
        self.assertGreater(float(sweep @ [0, 0, -1]), 0.99)
        tilted = self.law.cut_line([0, 0.05, 0.05], [0.12, 0.05, 0.06], [0, 0, -1], normals, crossed)
        self.assertEqual(0.0, tilted[3], "a knife tilted past 2 degrees keeps its own line")

    def test_the_edge_is_clipped_to_the_footprint(self) -> None:
        contact = self.law.edge_contact([-0.045, 0.05, 0.03], [0.075, 0.05, 0.03], (0.06, 0.10, 0.04))
        self.assertAlmostEqual(0.06, contact.length)
        np.testing.assert_allclose([0.03, 0.05, 0.03], contact.centroid, atol=1e-12)
        self.assertEqual(0.0, self.law.edge_contact([0, 0.2, 0], [0.1, 0.2, 0], (0.06, 0.1, 0.04)).length)
        self.assertAlmostEqual(0.06, contact.inside, msg="a level edge under the top is in the block all along")
        np.testing.assert_allclose(contact.centroid, contact.inside_centroid, atol=1e-12)

    def test_the_law_s_length_is_the_edge_inside_the_block(self) -> None:
        """A knife tipped 14 degrees, its lowest point 7 mm under the 40 mm top: 29 mm of edge is in the block,
        and the push acts at their middle."""
        tilt = math.radians(14.0)
        low = np.array([0.06, 0.05, 0.033])
        high = low - 0.12 * np.array([math.cos(tilt), 0.0, -math.sin(tilt)])
        contact = self.law.edge_contact(high, low, (0.06, 0.10, 0.04))
        self.assertAlmostEqual(0.06 / math.cos(tilt), contact.length, msg="over the footprint")
        self.assertAlmostEqual(0.007 / math.sin(tilt), contact.inside, msg="under the top")
        # The push acts at the middle of what is in the block, 16 mm down the edge from the footprint's.
        np.testing.assert_allclose([0.06 - 0.0035 / math.tan(tilt), 0.05, 0.0365], contact.inside_centroid, atol=1e-12)
        self.assertAlmostEqual(0.03, contact.centroid[0])
        up = np.array([0.0, 0.0, 0.02])
        over = self.law.edge_contact(high + up, low + up, (0.06, 0.10, 0.04))
        self.assertGreater(over.length, 0.0)
        self.assertEqual(0.0, over.inside, "an edge over the top is not in the block")
        np.testing.assert_allclose(over.centroid, over.inside_centroid, atol=1e-12)

    def test_the_push_grows_caps_at_the_threshold_and_lets_go_at_the_floor(self) -> None:
        push = self.law.edge_push
        common = {"length": 0.06, "young": 23.5e3, "push": 1.0, "threshold": 3.0, "floor": 0.0025}
        self.assertEqual((0.0, 0.0), push(depth=-0.001, gap=0.03, **common))
        force, lag = push(depth=0.0001, gap=0.03, **common)
        self.assertAlmostEqual(23.5e3 * 0.06 * 0.0001 * 0.0301 / 0.03, force)
        self.assertEqual(0.0001, lag, "under the threshold the front stays")
        force, lag = push(depth=0.002, gap=0.03, **common)
        self.assertAlmostEqual(3.0 * 0.06, force)
        self.assertAlmostEqual(
            3.0 / 23.5e3, lag, delta=1e-6, msg="a deep block's threshold indentation is toughness / E"
        )
        self.assertEqual((0.18, 0.0), tuple(round(v, 12) for v in push(depth=0.001, gap=0.002, **common)))
        dull = push(depth=0.012, gap=0.02, **{**common, "threshold": 3000.0})
        self.assertEqual(0.012, dull[1], "a blade 1000 times duller cuts nothing at 12 mm")
        self.assertAlmostEqual(23.5e3 * 0.06 * 0.012 * 0.032 / 0.02, dull[0])


@unittest.skipUnless(has_pack("cutting"), "the cutting kind is not installed")
class TestTheKnifeAndTheBlockAsBuilt(unittest.TestCase):
    """What the two entities take from their specs, on a CPU builder that is never finalized."""

    def _knife(self, **kwargs):
        import newton

        import robosimstudio as rss
        from robosimstudio_materials.cutting.entity import EntityKnife

        builder = newton.ModelBuilder()
        knife = EntityKnife(rss.knife(name="knife", **kwargs))
        knife._add_primitive(builder, (0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 1.0))
        return builder, knife

    def test_the_knife_weighs_its_densities_unless_it_states_a_mass(self) -> None:
        builder, knife = self._knife()
        self.assertAlmostEqual(0.096, float(builder.body_mass[knife.body]), places=3, msg="25 g handle, 71 g blade")
        self.assertGreaterEqual(knife.coord_start, 0, "a free body")
        builder, knife = self._knife(mass=0.2)
        self.assertAlmostEqual(0.2, float(builder.body_mass[knife.body]), places=6)

    def test_a_static_knife_is_welded_where_it_stands(self) -> None:
        import newton

        builder, knife = self._knife(static=True)
        self.assertEqual(-1, knife.coord_start, "no free joint's coordinates to reset")
        self.assertNotIn(int(newton.JointType.FREE), [int(t) for t in builder.joint_type])
        self.assertEqual(2, sum(int(b) == knife.body for b in builder.shape_body), "the handle and the blade")
        # A bare link was no weld: VBD integrated it and MuJoCo refused it (EntityRigid._add_welded_link).
        welds = [j for j in range(builder.joint_count) if int(builder.joint_child[j]) == knife.body]
        self.assertEqual(1, len(welds), "one joint holds it")
        self.assertEqual(
            (int(newton.JointType.FIXED), -1),
            (int(builder.joint_type[welds[0]]), int(builder.joint_parent[welds[0]])),
            "fixed to the world",
        )
        self.assertEqual(int(newton.BodyFlags.KINEMATIC), int(builder.body_flags[knife.body]), "and nothing pushes it")

    def test_the_block_finds_its_knife_by_name_and_says_where_a_missing_one_goes(self) -> None:
        import robosimstudio as rss
        from robosimstudio_materials.cutting.entity import EntityCuttable

        builder, knife = self._knife()
        block = EntityCuttable(rss.cuttable(name="tofu", knife=rss.knife(name="knife")))
        block.build(builder, None)
        self.assertEqual(knife.body, block.knife_body)
        lone = EntityCuttable(rss.cuttable(name="tofu", knife=rss.knife(name="cleaver")))
        with self.assertRaisesRegex(KeyError, "its knife 'cleaver'.* before the block"):
            lone.build(builder, None)

    def test_the_block_is_drawn_in_its_material_s_colour_unless_it_states_one(self) -> None:
        """Where the path tracer and the studio look: the spec's ``color``, else the entity's material's."""
        from types import SimpleNamespace

        import robosimstudio as rss
        from robosimstudio.sim.sim import Sim
        from robosimstudio_materials.cutting.entity import EntityCuttable
        from robosimstudio_studio.garments import garment_visuals

        cake, knife = rss.cuttables.CAKE, rss.knife(name="knife")
        for color, want in ((None, cake.color), ((0.2, 0.3, 0.4), (0.2, 0.3, 0.4))):
            with self.subTest(color=color):
                block = EntityCuttable(rss.cuttable(name="block", knife=knife, material=cake, color=color))
                handle = SimpleNamespace(particles=np.arange(8), triangles=np.zeros((0, 3)))
                scene = SimpleNamespace(
                    handles=SimpleNamespace(deformables={"block": handle}), by_name={"block": block}
                )
                self.assertEqual({"block": want}, Sim._deformable_colors(SimpleNamespace(scene=scene)), "path tracer")
                (visual,) = garment_visuals(scene)
                self.assertEqual(want, visual.color, "studio")

    def test_a_cut_face_is_triangulated_once_a_snapshot_and_drawn_where_the_cage_is(self) -> None:
        """The cutter replaces its snapshot when the cut moves; while one holds, a step regathers the cage."""
        from types import SimpleNamespace
        from unittest import mock

        import robosimstudio as rss
        from robosimstudio_materials.cutting import solver
        from robosimstudio_materials.cutting.entity import EntityCuttable
        from robosimstudio_materials.cutting.law import BlockFrame, box_cage

        spec = rss.cuttable(name="tofu", knife=rss.knife(name="knife"))
        block = EntityCuttable(spec)
        block.block = BlockFrame.from_base((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), spec.size)
        cage, _ = box_cage(np.asarray(spec.size) * block.block.scale)

        def snapshot():
            """One live cut face, the cage's base square."""
            plane = {"face_u": [np.array([1.0, 0.0, 0.0])], "face_v": [np.array([0.0, 1.0, 0.0])]}
            return SimpleNamespace(
                num_faces=1, face_alive=[True], face_loops=lambda f: [np.arange(4)], verts=cage, **plane
            )

        x = cage.copy()
        block.body = SimpleNamespace(
            sim=SimpleNamespace(x=x),
            corrected_mesh_weights=lambda: np.eye(8),
            mesh_tris=np.array([[4, 5, 6]]),
            cutter=SimpleNamespace(snapshot=snapshot()),
            cut_face_ids={0},
        )
        triangulate = solver.triangulate_polygon_with_holes
        with mock.patch.object(solver, "triangulate_polygon_with_holes", wraps=triangulate) as tri:
            block._refresh_surface()
            x[:4] += 0.01
            block._refresh_surface()
            self.assertEqual(1, tri.call_count, "the snapshot held: its triangles are kept")
            np.testing.assert_allclose(block.block.from_cutter(x[:4]), block._surface[0][8:], atol=1e-6)
            self.assertEqual((3, 3), block._surface[1].shape, "the skin's triangle and the face's two")
            block.body.cutter.snapshot = snapshot()
            block._refresh_surface()
            self.assertEqual(2, tri.call_count, "a new snapshot is triangulated again")

    def test_every_live_field_of_the_block_has_a_writer(self) -> None:
        """A LIVE field promises a write lands on the next step, which needs a tunable to carry it."""
        from dataclasses import fields

        import robosimstudio as rss
        from robosimstudio.specs.params import LIVE, param_ui
        from robosimstudio_materials.cutting.entity import EntityCuttable
        from robosimstudio_materials.cutting.spec import CuttableMaterialSpec

        tiers = {
            f.name: getattr(param_ui(CuttableMaterialSpec, f.name), "tier", "") for f in fields(CuttableMaterialSpec)
        }
        live = {name for name, tier in tiers.items() if tier == LIVE}
        block = EntityCuttable(rss.cuttable(name="tofu", knife=rss.knife(name="knife")))
        offered = {f.name for t in block.tunables() for f in fields(t.spec)}
        self.assertEqual(set(), live - offered, "LIVE on the spec, and nothing on the block writes it")


if __name__ == "__main__":
    unittest.main()
