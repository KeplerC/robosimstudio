# SPDX-License-Identifier: Apache-2.0
"""The ``thread`` kind's geometry and declarations, and the settings its physics table states.

A cap holds on a neck because two meshes interleave, so the meshes are held to
what a thread is: closed surfaces, a ridge that climbs one pitch a turn, room
between the flanks and none for a straight pull. The parts say where they mate,
a hand says where its roll-axis grip point is, and the table's settings reach a
builder: a finger's contact priority and a part's impedance.
"""

from __future__ import annotations

import math
import unittest

import newton
import numpy as np

import robosimstudio as rss
from robosimstudio.core.builder import set_shape_attribute
from robosimstudio.robots.registry import get_gripper
from robosimstudio_materials.thread import (
    THREAD_60HZ,
    ThreadedSpec,
    ThreadSpec,
    bottle,
    bulb,
    cap,
    cap_mesh,
    neck_mesh,
    screwed,
    socket,
)

TIER = "fast"
"""No GPU: meshes as arrays, specs, and a ``newton.ModelBuilder`` that is never finalized."""


def _closed(faces: np.ndarray) -> bool:
    """Every edge is shared by exactly two triangles, once each way."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    count = int(faces.max()) + 1
    forward = edges[:, 0].astype(np.int64) * count + edges[:, 1]
    backward = edges[:, 1].astype(np.int64) * count + edges[:, 0]
    return len(np.unique(forward)) == len(forward) and set(forward.tolist()) == set(backward.tolist())


def _volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _inside_cap(points: np.ndarray, thread: ThreadSpec, *, turned: float = 0.0, raised: float = 0.0) -> np.ndarray:
    """Which points lie inside a cap's ridge, for a cap ``turned`` [turns] and ``raised`` [m] on the neck's frame."""
    radius = np.hypot(points[:, 0], points[:, 1])
    angle = np.arctan2(points[:, 1], points[:, 0]) - 2.0 * math.pi * turned
    z = points[:, 2] - raised
    start = thread.start - 0.5 * thread.pitch
    inside = np.zeros(len(points), dtype=bool)
    for k in range(-2, 5):
        wound = np.mod(angle, 2.0 * math.pi) + 2.0 * math.pi * k
        on = (wound >= 0.0) & (wound <= 2.0 * math.pi * thread.turns)
        centre = start + thread.pitch * wound / (2.0 * math.pi)
        # The ridge's half-width at this radius: its crest is at the smaller radius.
        depth = np.clip((thread.inner_radius - radius) / thread.ridge_height, 0.0, 1.0)
        half = thread.ridge_half_base - depth * (thread.ridge_half_base - thread.ridge_half_crest)
        inside |= (
            on
            & (radius < thread.inner_radius)
            & (radius > thread.inner_radius - thread.ridge_height)
            & (np.abs(z - centre) < half)
        )
    return inside


def _neck_ridge(thread: ThreadSpec) -> np.ndarray:
    """The neck's vertices that stand proud of its core: its ridge."""
    vertices, _ = neck_mesh(thread)
    radius = np.hypot(vertices[:, 0], vertices[:, 1])
    return vertices[(radius > thread.core_radius + 1e-5) & (radius < thread.crest_radius + 1e-6)]


class TestTheMeshes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.thread = ThreadSpec()
        cls.neck, cls.cap = neck_mesh(cls.thread), cap_mesh(cls.thread)

    def test_both_are_closed_and_face_outward(self) -> None:
        for name, (vertices, faces) in (("neck", self.neck), ("cap", self.cap)):
            with self.subTest(part=name):
                self.assertTrue(_closed(faces), "an open mesh has no inside for a distance field to sign")
                self.assertGreater(_volume(vertices, faces), 1.0e-6)
                self.assertTrue(np.isfinite(vertices).all())

    def test_the_sizes_are_the_spec_s(self) -> None:
        t = self.thread
        neck, cap_ = self.neck[0], self.cap[0]
        self.assertAlmostEqual(t.crest_radius, float(np.hypot(neck[:, 0], neck[:, 1]).max()), delta=0.0015)
        self.assertAlmostEqual(t.grip_radius, float(np.hypot(cap_[:, 0], cap_[:, 1]).max()), delta=1e-6)
        self.assertAlmostEqual(0.0, float(neck[:, 2].max()), delta=1e-9, msg="the rim is the neck's z = 0")
        self.assertAlmostEqual(-t.finish_length, float(neck[:, 2].min()), delta=1e-9)
        self.assertAlmostEqual(t.top, float(cap_[:, 2].max()), delta=1e-9)
        self.assertAlmostEqual(-t.cap_depth, float(cap_[:, 2].min()), delta=1e-9)
        self.assertAlmostEqual(0.0276, 2.0 * t.crest_radius, delta=1e-6)
        self.assertAlmostEqual(0.0317, 2.0 * t.grip_radius, delta=1e-6)

    def test_the_ridge_climbs_one_pitch_a_turn(self) -> None:
        """The crest's height against its angle, over the turns the ridge is wound: a line of slope pitch / 2 pi."""
        t = self.thread
        vertices = self.neck[0]
        radius = np.hypot(vertices[:, 0], vertices[:, 1])
        crest = vertices[np.abs(radius - t.crest_radius) < 1e-6]  # not the collar, which is wider
        angle = np.mod(np.arctan2(crest[:, 1], crest[:, 0]), 2.0 * math.pi)
        # Unwind: the turn a crest point is on is the one that puts it nearest the helix.
        turn = np.round((crest[:, 2] - t.start - t.pitch * angle / (2.0 * math.pi)) / t.pitch)
        wound = np.round(angle + 2.0 * math.pi * turn, 6)
        # One point a facet: the middle of the crest's flat, which is a few mesh rows wide.
        where = np.unique(wound)
        middle = np.array([crest[wound == w, 2].mean() for w in where])
        slope, start = np.polyfit(where, middle, 1)
        self.assertAlmostEqual(t.pitch, 2.0 * math.pi * slope, delta=0.002 * t.pitch)
        self.assertAlmostEqual(t.start, start, delta=1e-4)
        self.assertGreater(float(wound.max() - wound.min()), 2.0 * math.pi * (t.turns - 0.2), "wound its two turns")

    def test_screwed_home_the_ridges_interleave_with_the_stated_play(self) -> None:
        """No point of the neck's ridge is inside the cap's, seated or anywhere along the helix; a straight pull meets it."""
        t = self.thread
        ridge = _neck_ridge(t)
        self.assertGreater(len(ridge), 1000)
        self.assertAlmostEqual(0.525e-3, t.axial_play, delta=1e-6)
        half = 0.5 * t.axial_play
        for turned in (0.0, 0.25, 0.5, 1.0, 1.5, 1.9):
            with self.subTest(turns_out=turned):
                on_helix = t.pitch * turned
                self.assertEqual(0, int(_inside_cap(ridge, t, turned=turned, raised=on_helix).sum()))
                # Inside the play, either way, nothing meets; past it, the flanks do.
                for off in (-0.8 * half, 0.8 * half):
                    self.assertEqual(0, int(_inside_cap(ridge, t, turned=turned, raised=on_helix + off).sum()))
        for pulled in (half + 0.15e-3, 1.0e-3, 0.5 * t.pitch):
            with self.subTest(pulled_mm=round(pulled * 1e3, 2)):
                self.assertGreater(int(_inside_cap(ridge, t, raised=pulled).sum()), 100, "a straight pull is stopped")

    def test_after_its_turns_the_cap_lifts_clear(self) -> None:
        t = self.thread
        ridge = _neck_ridge(t)
        out = t.turns + 0.1
        for lifted in (0.0, 0.002, 0.01):
            self.assertEqual(0, int(_inside_cap(ridge, t, turned=out, raised=t.pitch * out + lifted).sum()))

    def test_a_coarser_mesh_is_still_a_thread(self) -> None:
        coarse = ThreadSpec(segments=90, row=0.3e-3)
        for _, faces in (neck_mesh(coarse), cap_mesh(coarse)):
            self.assertTrue(_closed(faces))
        self.assertLess(len(neck_mesh(coarse)[1]), len(self.neck[1]) // 2)

    def test_a_thread_that_cannot_mesh_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not fit one pitch"):
            ThreadSpec(pitch=2.0e-3)
        with self.assertRaisesRegex(ValueError, "radial clearance"):
            ThreadSpec(radial_clearance=2.0e-3)
        with self.assertRaisesRegex(ValueError, "narrower at its crest"):
            ThreadSpec(ridge_half_crest=1.0e-3)


class TestTheParts(unittest.TestCase):
    def test_each_constructor_builds_its_part_stiff_and_hard_to_slide(self) -> None:
        for make, part in ((bottle, "bottle"), (cap, "cap"), (bulb, "bulb"), (socket, "socket")):
            with self.subTest(part=part):
                spec = make()
                self.assertIsInstance(spec, ThreadedSpec)
                self.assertEqual((part, "threaded"), (spec.part, spec.entity))
                self.assertEqual((0.998, 0.3, 1.0e7), (spec.impedance, spec.friction, spec.ke))
                self.assertEqual("cylinder", spec.shape.kind, "a stated primitive: the entity builds it")
        self.assertTrue(socket().static, "a lamp holder is screwed to something")
        self.assertFalse(bottle().static or cap().static or bulb().static)
        self.assertTrue(bottle(static=True).static)
        self.assertIsNone(cap(impedance=None).impedance, "a part may be built as Newton builds one")

    def test_the_flat_names_are_the_kind_s(self) -> None:
        for name in ("THREAD_60HZ", "ThreadSpec", "ThreadedSpec", "bottle", "bulb", "cap", "socket"):
            self.assertIs(getattr(rss, name), getattr(__import__("robosimstudio_materials.thread").thread, name))

    def test_the_mating_points(self) -> None:
        t = ThreadSpec()
        self.assertAlmostEqual(0.249, bottle().rim, delta=1e-9)
        self.assertAlmostEqual(0.154, bottle(body=(0.03, 0.085)).rim, delta=1e-9)
        self.assertAlmostEqual(t.top, socket().rim, delta=1e-12)
        self.assertEqual(0.0, cap().rim)
        self.assertAlmostEqual(-0.005, cap().grasp_height, delta=1e-9, msg="the middle of the ribbed band")
        self.assertAlmostEqual(0.0301, bulb().grasp_height, delta=1e-9, msg="on the glass, over the base")

    def test_screwed_places_the_mate_on_the_helix(self) -> None:
        flask = bottle(at=(0.1, -0.2, 0.0))
        home = screwed(flask)
        np.testing.assert_allclose(home["at"], (0.1, -0.2, 0.249 + 1.0e-4), atol=1e-12)
        self.assertEqual((0.0, 0.0, 0.0, 1.0), home["quat"])
        out = screwed(flask, turns_out=0.25, lift=0.0)
        self.assertAlmostEqual(0.249 + 0.75e-3, out["at"][2], delta=1e-12)
        np.testing.assert_allclose(out["quat"], (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)), atol=1e-12)
        lamp = rss.bulb(**screwed(socket(at=(0.0, 0.0, 0.05)), turns_out=2.0, lift=0.0))
        self.assertAlmostEqual(0.05 + 1.5e-3 + 6.0e-3, lamp.pose[2], delta=1e-12)
        with self.assertRaisesRegex(ValueError, "a cap goes on a bottle"):
            screwed(cap())
        with self.assertRaisesRegex(ValueError, "is turned"):
            screwed(bottle(quat=(0.0, 0.0, 0.38268343, 0.92387953)))

    def test_what_is_not_a_part_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "part must be one of"):
            ThreadedSpec(name="x", part="nut")
        with self.assertRaisesRegex(ValueError, "takes no asset"):
            ThreadedSpec(name="x", asset=rss.AssetSpec(path="x.obj"))
        with self.assertRaisesRegex(ValueError, "multiple of 8"):
            cap(sdf_resolution=100)
        with self.assertRaisesRegex(ValueError, r"inside \(0, 1\)"):
            cap(impedance=1.0)


class TestTheTable(unittest.TestCase):
    def test_it_states_the_three_settings_and_the_room_they_need(self) -> None:
        mujoco = THREAD_60HZ.backend_for("rigid")
        self.assertEqual("mujoco", mujoco.solver)
        self.assertIs(False, mujoco.get("use_mujoco_contacts"), "Newton's pipeline finds the contacts")
        self.assertEqual((2048, 8192), (mujoco.get("nconmax"), mujoco.get("njmax")))
        collision = THREAD_60HZ.collision
        self.assertEqual(1, collision.finger_priority)
        self.assertEqual(2.0e-3, collision.rigid_gap)
        self.assertEqual(2048, collision.rigid_contact_max)
        self.assertAlmostEqual(1200.0, 1.0 / THREAD_60HZ.sim_dt)

    def test_no_other_table_changed(self) -> None:
        """Opt in: a scene that stages nothing, or the hand-lift table, mixes contact parameters as it did."""
        for table in (rss.GENERIC, rss.MUJOCO_500HZ, rss.CPU, rss.ArrangementSpec()):
            self.assertEqual(0, table.collision.finger_priority)
        self.assertIsNone(rss.box().impedance)
        with self.assertRaisesRegex(ValueError, "whole number"):
            rss.CollisionSpec(finger_priority=0.5)


class TestTheSettingsReachABuilder(unittest.TestCase):
    def test_a_shape_attribute_is_written_on_the_named_shapes_only(self) -> None:
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        body = builder.add_body()
        shapes = [builder.add_shape_sphere(body, radius=0.01) for _ in range(3)]
        self.assertEqual(2, set_shape_attribute(builder, "mujoco:geom_priority", 1, shapes[:2]))
        values = builder.custom_attributes["mujoco:geom_priority"].values
        self.assertEqual({shapes[0]: 1, shapes[1]: 1}, {k: v for k, v in values.items() if v})
        self.assertEqual(1, set_shape_attribute(builder, "mujoco:geom_solimp", (0.998, 0.998, 0.001, 0.5, 2.0), [2]))
        self.assertEqual({shapes[0]: 1, shapes[1]: 1}, {k: v for k, v in values.items() if v}, "others left alone")

    def test_without_the_solver_s_attributes_nothing_is_written(self) -> None:
        builder = newton.ModelBuilder()
        shape = builder.add_shape_sphere(builder.add_body(), radius=0.01)
        self.assertEqual(0, set_shape_attribute(builder, "mujoco:geom_priority", 1, [shape]))


class TestTheRollAxisGripPoint(unittest.TestCase):
    """A hand names a second grip point, and says where the tool centre goes to put it on a part."""

    def setUp(self) -> None:
        self.hand = get_gripper("yam_linear_arena")

    def test_the_yam_names_one_on_its_wrist_axis(self) -> None:
        ((name, point),) = self.hand.grasp_points
        self.assertEqual("roll_axis", name)
        self.assertEqual((0.0, 0.0), tuple(point[:2]), "on the wrist's own z")
        self.assertAlmostEqual(0.0957, point[2])

    def test_tool_down_the_tool_centre_goes_beside_and_below_it(self) -> None:
        """Tool down, jaws along y: the wrist is 95.7 mm over the point, and the pads 44 mm off the axis, 130 mm down."""
        down = (0.0, 1.0, 0.0, 0.0)
        tcp = np.asarray(self.hand.tcp_for("roll_axis", (0.3, 0.1, 0.5), down))
        np.testing.assert_allclose(np.linalg.norm(tcp[:2] - (0.3, 0.1)), 0.044, atol=1e-9)
        self.assertAlmostEqual(0.5 - (0.130 - 0.0957), tcp[2], delta=1e-9)

    def test_turning_about_the_tool_axis_swings_the_tool_centre_round_the_point(self) -> None:
        down = np.array([0.0, 1.0, 0.0, 0.0])
        point = np.array([0.3, 0.1, 0.5])
        seen = []
        for angle in (0.0, 0.5 * math.pi, math.pi):
            about_z = np.array([0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)])
            x1, y1, z1, w1 = about_z
            x2, y2, z2, w2 = down
            quat = (
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            )
            seen.append(np.asarray(self.hand.tcp_for("roll_axis", point, quat)) - point)
        np.testing.assert_allclose(seen[0][:2], -seen[2][:2], atol=1e-9, err_msg="half a turn: the other side")
        self.assertAlmostEqual(0.0, float(seen[0][:2] @ seen[1][:2]), delta=1e-9, msg="a quarter turn: square to it")
        for offset in seen:
            self.assertAlmostEqual(seen[0][2], offset[2], delta=1e-9, msg="and no higher or lower")

    def test_a_point_the_hand_does_not_name_is_refused_by_name(self) -> None:
        with self.assertRaisesRegex(KeyError, "has no grasp point 'palm'.*roll_axis"):
            self.hand.tcp_for("palm", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        with self.assertRaisesRegex(KeyError, "it has \\[\\]"):
            get_gripper("franka_hand_robosuite").tcp_for("roll_axis", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
