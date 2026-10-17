# SPDX-License-Identifier: Apache-2.0
"""The suction hand without a GPU: its spec, its pairing, the seal's geometry, and the policy tool's arguments.

:mod:`tests.test_suction_pick` shows the cup picking a plate up; this holds the
parts that are arithmetic. The hand is registered as an attachable hand with no
fingers and a span of 0; ``yam_bimanual_suction`` puts it on the right arm only,
with the left arm exactly ``yam_bimanual``'s; the gripper axis refuses the cup
for the whole two-root robot with the reason. :func:`face_on_box` reads gap,
tilt and cover off synthetic poses, :meth:`Suction.seal` turns them into the
sentences the page shows, and ``sim.vacuum`` appears only on a robot with a cup,
is tagged privileged, and maps its arguments onto :meth:`Suction.hold` and
:meth:`Suction.release`.
"""

from __future__ import annotations

import math
import types
import unittest
import xml.etree.ElementTree as ET

import numpy as np

TIER = "fast"
"""No GPU: specs, numpy and a CPU model. Gates every commit."""


def _down(x: float = 0.0, y: float = 0.0, z: float = 0.0, tilt_deg: float = 0.0) -> np.ndarray:
    """A face frame at ``(x, y, z)`` pointing down (+z to world -z), tilted about world x by ``tilt_deg``."""
    half = math.radians(180.0 + tilt_deg) / 2
    return np.array([x, y, z, math.sin(half), 0.0, 0.0, math.cos(half)])


PLATE_HALF = (0.075, 0.075, 0.003)
PLATE = np.array([0.0, 0.0, 0.003, 0.0, 0.0, 0.0, 1.0])
"""A 150 x 150 x 6 mm plate lying on z = 0: its top face at z = 6 mm."""


class TestTheHand(unittest.TestCase):
    def test_it_is_an_attachable_hand_with_no_fingers(self) -> None:
        from robosimstudio.robots.registry import attachable_grippers, get_gripper

        hand = get_gripper("suction")
        self.assertIn("suction", attachable_grippers())
        self.assertEqual((), hand.finger_names)
        self.assertEqual("", hand.finger_token, "the token would sweep an arm's own fingers in as the cup's")
        self.assertEqual("suction_pad", hand.suction_pad)
        self.assertEqual(hand.suction_pad, hand.tcp_body, "the tool centre is the face")
        self.assertEqual(0.0, hand.effective_span, "a cup closes across nothing")
        self.assertEqual("", hand.mount_body, "the cup bolts to whatever wrist the arm names")
        self.assertEqual("suction_pad", hand.for_arm("right").suction_pad)

    def test_the_face_is_the_pad_bodys_origin(self) -> None:
        """The convention the seal reads: the face geom ends at the pad's origin and the cup reaches along +z."""
        from robosimstudio.paths import resolve_asset
        from robosimstudio.robots.registry import get_gripper

        hand = get_gripper("suction")
        root = ET.parse(resolve_asset(hand.asset)).getroot()
        pad = next(b for b in root.iter("body") if b.get("name") == "suction_pad")
        face = next(g for g in pad.iter("geom") if g.get("name") == "suction_face")
        radius, half_height = (float(v) for v in face.get("size").split())
        z = float(face.get("pos").split()[2])
        self.assertAlmostEqual(0.0, z + half_height, places=9, msg="the face does not end at the pad's origin")
        self.assertAlmostEqual(hand.pad_radius, radius)
        self.assertAlmostEqual(0.060, float(pad.get("pos").split()[2]), msg="60 mm from flange to face")


class TestThePairing(unittest.TestCase):
    def test_the_cup_is_on_the_right_arm_and_the_left_is_untouched(self) -> None:
        from robosimstudio.robots.registry import get_embodiment, hands_of

        plain, cup = get_embodiment("yam_bimanual"), get_embodiment("yam_bimanual_suction")
        self.assertEqual(plain.arms[0], cup.arms[0], "the left arm changed")
        right = cup.arms[1]
        self.assertEqual(("right", "suction"), (right.name, right.gripper))
        self.assertEqual(tuple(f"right_joint{i}" for i in range(1, 7)), right.joint_names)
        self.assertIsNotNone(right.gripper_mount)
        self.assertEqual(6, len(right.home_q))
        self.assertEqual(["yam_linear_arena", "suction"], hands_of(cup))

    def test_the_gripper_axis_refuses_a_cup_for_both_arms_and_says_how(self) -> None:
        from robosimstudio.robots.registry import _admits_gripper, get_embodiment

        ok, why = _admits_gripper(get_embodiment("yam_bimanual"), "suction")
        self.assertFalse(ok)
        self.assertIn("ArmSpec.gripper", why)
        self.assertEqual((True, ""), _admits_gripper(get_embodiment("panda"), "suction"))


class TestTheSealGeometry(unittest.TestCase):
    def test_a_level_cup_one_millimetre_over_the_plate(self) -> None:
        from robosimstudio.core.suction import face_on_box

        gap, tilt, covered = face_on_box(_down(z=0.007), 0.015, PLATE, PLATE_HALF)
        self.assertAlmostEqual(0.001, gap, places=9)
        self.assertAlmostEqual(0.0, tilt, places=6)
        self.assertTrue(covered)

    def test_pressed_in_reads_negative(self) -> None:
        from robosimstudio.core.suction import face_on_box

        gap, _, _ = face_on_box(_down(z=0.0055), 0.015, PLATE, PLATE_HALF)
        self.assertAlmostEqual(-0.0005, gap, places=9)

    def test_tilt_is_the_angle_between_the_normals(self) -> None:
        from robosimstudio.core.suction import face_on_box

        _, tilt, _ = face_on_box(_down(z=0.007, tilt_deg=20.0), 0.015, PLATE, PLATE_HALF)
        self.assertAlmostEqual(20.0, math.degrees(tilt), places=4)

    def test_over_the_edge_is_not_covered(self) -> None:
        from robosimstudio.core.suction import face_on_box

        _, _, covered = face_on_box(_down(x=0.07, z=0.007), 0.015, PLATE, PLATE_HALF)
        self.assertFalse(covered)

    def test_a_cup_over_the_plate_pointing_up_faces_its_sides_not_its_underside(self) -> None:
        """The underside is the face the cup's normal matches, but through the plate: a right angle, not a seal."""
        from robosimstudio.core.suction import face_on_box

        _, tilt, _ = face_on_box(np.array([0.0, 0.0, 0.007, 0.0, 0.0, 0.0, 1.0]), 0.015, PLATE, PLATE_HALF)
        self.assertAlmostEqual(90.0, math.degrees(tilt), places=4)

    def test_a_yawed_plate_on_its_side(self) -> None:
        """A plate yawed 30 degrees and standing on an edge: the cup is checked against the face it points at."""
        from robosimstudio.core.math import quat_multiply
        from robosimstudio.core.suction import face_on_box

        yaw = np.array([0.0, 0.0, math.sin(math.radians(15.0)), math.cos(math.radians(15.0))])
        roll = np.array([math.sin(math.radians(45.0)), 0.0, 0.0, math.cos(math.radians(45.0))])
        standing = np.concatenate([[0.0, 0.0, 0.075], quat_multiply(yaw, roll)])
        gap, tilt, covered = face_on_box(_down(z=0.152), 0.015, standing, PLATE_HALF)
        self.assertAlmostEqual(0.002, gap, places=6)
        self.assertAlmostEqual(0.0, tilt, places=5)
        self.assertFalse(covered, "a 6 mm edge cannot carry a 30 mm cup")


class TestTheSealSays(unittest.TestCase):
    """:meth:`Suction.seal` over a CPU model: a pad body and a plate body, posed by hand."""

    @classmethod
    def setUpClass(cls) -> None:
        import newton
        import warp as wp

        from robosimstudio.core.suction import Suction

        builder = newton.ModelBuilder()
        pad = builder.add_body(xform=wp.transform_identity())
        builder.add_shape_cylinder(pad, radius=0.015, half_height=0.003)
        plate = builder.add_body(xform=wp.transform_identity())
        builder.add_shape_box(plate, hx=PLATE_HALF[0], hy=PLATE_HALF[1], hz=PLATE_HALF[2])
        ball = builder.add_body(xform=wp.transform_identity())
        builder.add_shape_sphere(ball, radius=0.02)
        block = builder.add_body(xform=wp.transform_identity())
        builder.add_shape_box(block, hx=0.02, hy=0.02, hz=0.02)
        cls.model = builder.finalize(device="cpu")
        cls.pad, cls.plate, cls.ball, cls.block = pad, plate, ball, block
        cls.cup = Suction(
            cls.model, arm="right", pad_body=pad, pad_radius=0.015, rows={plate: 0}, owners={plate: "plate"}
        )

    def poses(self, face: np.ndarray) -> np.ndarray:
        body_q = np.zeros((4, 7))
        body_q[:, 6] = 1.0
        body_q[self.pad], body_q[self.plate] = face, PLATE
        body_q[self.ball, :3] = (1.0, 0.0, 0.02)
        body_q[self.block, :3] = (-1.0, 0.0, 0.02)
        return body_q

    def test_on_the_plate_it_seals(self) -> None:
        report = self.cup.seal(self.plate, self.poses(_down(z=0.0065)))
        self.assertTrue(report.sealed, report.why)
        self.assertAlmostEqual(0.0005, report.gap_m, places=6)

    def test_each_refusal_says_why(self) -> None:
        cases = {
            "degrees off the face": (self.plate, _down(z=0.007, tilt_deg=20.0)),
            "mm off the part": (self.plate, _down(z=0.010)),
            "over the part's edge": (self.plate, _down(x=0.07, z=0.0065)),
            # Under the plate, pointing down: its top is behind the plate, so only the sides face the cup.
            "90.0 degrees": (self.plate, _down(z=-0.006)),
            # 15 mm into a 40 mm block: past the 10 mm a soft contact could explain.
            "inside the part": (self.block, _down(x=-1.0, z=0.025)),
        }
        for words, (body, face) in cases.items():
            with self.subTest(words):
                report = self.cup.seal(body, self.poses(face))
                self.assertFalse(report.sealed)
                self.assertIn(words, report.why)

    def test_a_ball_offers_no_face(self) -> None:
        report = self.cup.seal(self.ball, self.poses(_down(z=0.0065)))
        self.assertFalse(report.sealed)
        self.assertIn("no flat face", report.why)

    def test_under_finds_the_plate(self) -> None:
        report = self.cup.under(self.poses(_down(z=0.0065)))
        self.assertTrue(report.sealed)
        self.assertEqual(self.plate, report.body)

    def test_a_hold_before_the_scene_is_ready_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            self.cup.hold(self.plate)

    def test_the_break_force_is_the_vacuum_times_the_face(self) -> None:
        from robosimstudio.core.suction import BREAK_PRESSURE

        self.assertAlmostEqual(BREAK_PRESSURE * math.pi * 0.015**2, self.cup.break_force, places=3)


class _Registry:
    def __init__(self) -> None:
        self.calls: dict[str, dict] = {}

    def register_callable(self, name, fn, **kwargs):
        if name in self.calls:
            raise AssertionError(f"duplicate registration of {name!r}")
        self.calls[name] = {"fn": fn, **kwargs}
        return fn


class _Bus:
    def call(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


class _Cup:
    """What the tool touches of a :class:`~robosimstudio.core.suction.Suction`."""

    def __init__(self) -> None:
        from robosimstudio.core.suction import SealReport

        self.calls: list[tuple] = []
        self.held, self.tier = -1, ""
        self.report = SealReport(True, 0.0004, math.radians(1.0), 7)

    def hold(self, body=None, *, tier="auto"):
        self.calls.append(("hold", body, tier))
        if self.report.sealed:
            self.held, self.tier = 7, "weld"
        return self.report

    def release(self):
        self.calls.append(("release",))
        self.held, self.tier = -1, ""


def _env(cups: dict) -> types.SimpleNamespace:
    plate = types.SimpleNamespace(body=7)
    cloth = types.SimpleNamespace(particles=[0, 1])
    scene = types.SimpleNamespace(
        embodiment=types.SimpleNamespace(suctions=cups), by_name={"plate": plate, "towel": cloth}
    )
    return types.SimpleNamespace(scene=scene)


class TestTheVacuumTool(unittest.TestCase):
    def setUp(self) -> None:
        from robosimstudio_policy_agent.sim_tools import register_vacuum

        self.cup = _Cup()
        self.registry = _Registry()
        self.tool = register_vacuum(self.registry, _env({"right": self.cup}), _Bus())

    def test_only_a_robot_with_a_cup_gets_it_and_it_is_privileged(self) -> None:
        from robosimstudio_policy_agent.sim_tools import register_vacuum

        self.assertEqual(["sim.vacuum"], sorted(self.registry.calls))
        self.assertIn("privileged", self.registry.calls["sim.vacuum"]["tags"])
        bare = _Registry()
        self.assertIsNone(register_vacuum(bare, _env({}), _Bus()))
        self.assertEqual({}, bare.calls)

    def test_on_seals_on_whatever_is_under_the_first_cup(self) -> None:
        out = self.tool.vacuum()
        self.assertEqual(("hold", None, "auto"), self.cup.calls[-1])
        self.assertEqual(
            {"on", "sealed", "held", "gap_m", "tilt_deg", "tier", "arm", "why"}, set(out), "the documented keys"
        )
        self.assertEqual((True, "plate", "right", "weld"), (out["sealed"], out["held"], out["arm"], out["tier"]))
        self.assertAlmostEqual(1.0, out["tilt_deg"])

    def test_a_named_part_and_a_tier_reach_the_hold(self) -> None:
        self.tool.vacuum(object_name="plate", tier="wrench")
        self.assertEqual(("hold", 7, "wrench"), self.cup.calls[-1])

    def test_off_lets_go(self) -> None:
        out = self.tool.vacuum(on=False)
        self.assertEqual(("release",), self.cup.calls[-1])
        self.assertFalse(out["on"] or out["sealed"])

    def test_a_refused_seal_says_why(self) -> None:
        from robosimstudio.core.suction import SealReport

        self.cup.report = SealReport(False, 0.0091, 0.0, 7, "the face is 9.1 mm off the part")
        out = self.tool.vacuum()
        self.assertFalse(out["sealed"])
        self.assertEqual("", out["held"])
        self.assertIn("9.1 mm", out["why"])

    def test_bad_arguments_are_refused(self) -> None:
        from robosimstudio_policy_agent.refusal import ToolRefusal

        for kwargs in ({"arm": "left"}, {"object_name": "towel"}, {"object_name": "nothing"}, {"tier": "glue"}):
            with self.subTest(**kwargs), self.assertRaises(ToolRefusal):
                self.tool.vacuum(**kwargs)


if __name__ == "__main__":
    unittest.main()
