# SPDX-License-Identifier: Apache-2.0
"""The suction hand without a GPU: its spec, its pairing, the seal's geometry, and the policy tool's arguments.

:mod:`tests.test_suction_pick` shows the cup picking a plate up; this holds the
parts that are arithmetic. The hand is registered as an attachable hand with no
fingers and a span of 0; ``yam_bimanual_suction`` puts it on the right arm only,
with the left arm exactly ``yam_bimanual``'s; the gripper axis refuses the cup
for the whole two-root robot with the reason. :func:`face_on_box` reads gap,
tilt and cover off synthetic poses, :meth:`Suction.seal` turns them into the
sentences the page shows (its **Seal** readout looking without overwriting
:attr:`Suction.last`), a hold asked for again keeps the part it has, and
``sim.vacuum`` appears only on a robot with a cup, is tagged privileged, and
maps its arguments onto :meth:`Suction.hold` and :meth:`Suction.release`. Only
a robot carrying a cup is kept to one world. The weld holds only the parts on
the pad's own MuJoCo island, the wrench lets go at one stretch whatever the
part weighs, and ``sim.suction`` asked for an arm without a cup names the arms
with one.
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

    def test_a_hand_only_some_arms_carry_is_refused_for_the_whole_robot(self) -> None:
        """The binding puts its hand on every arm, so one of the robot's own is admitted only if every arm has it."""
        from robosimstudio.robots.registry import _admits_gripper, get_embodiment

        for gripper in ("suction", "yam_linear_arena"):
            with self.subTest(gripper):
                ok, why = _admits_gripper(get_embodiment("yam_bimanual_suction"), gripper)
                self.assertFalse(ok)
                self.assertIn("ArmSpec.gripper", why)
        self.assertEqual((True, ""), _admits_gripper(get_embodiment("yam_bimanual"), "yam_linear_arena"))

    def _built_spec(self, gripper: str):
        """The robot spec the compiler would build ``yam_bimanual_suction`` from under a binding naming ``gripper``.

        The compiler is stopped where it builds the robot.
        """
        from unittest import mock

        from robosimstudio.scene import compiler
        from robosimstudio.scene.request import BuildRequest
        from robosimstudio.specs import Binding

        class _Stop(Exception):
            pass

        specs = []

        def robot(spec, **kwargs):
            specs.append(spec)
            raise _Stop

        real = compiler._resolve_entity
        request = BuildRequest(
            embodiment="yam_bimanual_suction",
            binding=Binding(embodiment="yam_bimanual_suction", gripper=gripper),
        )
        with mock.patch.object(
            compiler, "_resolve_entity", lambda path: robot if path == compiler._EMBODIMENT_ENTITY else real(path)
        ):
            with self.assertRaises(_Stop):
                compiler._build_base(request, None)
        (spec,) = specs
        return spec

    def test_a_hand_the_binding_names_does_not_inherit_the_cups_mount(self) -> None:
        """A binding's hand replaces the cup, and the mount the right arm states for the cup goes with it.

        The left arm already carries the named hand and is left exactly as it is.
        """
        from robosimstudio.robots.registry import get_embodiment

        spec = self._built_spec("yam_linear_arena")
        self.assertEqual("yam_linear_arena", spec.gripper)
        self.assertEqual(get_embodiment("yam_bimanual_suction").arms[0], spec.arms[0])
        self.assertEqual(("", None), (spec.arms[1].gripper, spec.arms[1].gripper_mount))

    def test_an_arm_that_already_carries_the_named_hand_keeps_its_mount(self) -> None:
        """Naming the cup the right arm carries leaves that arm as it is, so the cup is not bolted at the flange."""
        from robosimstudio.robots.embodiments.yam_bimanual_suction import SUCTION_MOUNT

        spec = self._built_spec("suction")
        self.assertEqual("suction", spec.gripper)
        self.assertEqual(
            [("", None), ("suction", SUCTION_MOUNT)], [(arm.gripper, arm.gripper_mount) for arm in spec.arms]
        )


TWIN_ARMS = """<mujoco model="twin_arms">
  <worldbody>
    <body name="base">
      <geom type="box" size="0.05 0.05 0.05" mass="2"/>
      <body name="left_link1" pos="0 0.2 0.1">
        <joint name="left_joint1" type="hinge" axis="0 0 1" range="-3 3"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.5"/>
        <body name="left_link2" pos="0.2 0 0">
          <joint name="left_joint2" type="hinge" axis="0 1 0" range="-3 3"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.02" mass="0.3"/>
        </body>
      </body>
      <body name="right_link1" pos="0 -0.2 0.1">
        <joint name="right_joint1" type="hinge" axis="0 0 1" range="-3 3"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.5"/>
        <body name="right_link2" pos="0.2 0 0">
          <joint name="right_joint2" type="hinge" axis="0 1 0" range="-3 3"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.02" mass="0.3"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="left_joint1" kp="100"/>
    <position joint="left_joint2" kp="100"/>
    <position joint="right_joint1" kp="100"/>
    <position joint="right_joint2" kp="100"/>
  </actuator>
</mujoco>
"""
"""Two two-joint arms on one base, one articulation: the shape on which both arms can take a welded hand."""


class TestTwoCupsOnOneRobot(unittest.TestCase):
    """A cup on each arm of :data:`TWIN_ARMS`: the two imports carry the same labels, and each arm finds its own."""

    def test_each_arm_seals_with_its_own_pad(self) -> None:
        import tempfile
        from pathlib import Path

        import newton
        import warp as wp

        from robosimstudio.robots.entity import EntityEmbodiment
        from robosimstudio.specs import ArmSpec, EmbodimentSpec
        from robosimstudio.specs.asset import AssetSpec

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "twin_arms.xml"
            path.write_text(TWIN_ARMS)
            arms = tuple(
                ArmSpec(
                    name=side,
                    joint_names=(f"{side}_joint1", f"{side}_joint2"),
                    ee_body_name=f"{side}_link2",
                    gripper="suction",
                )
                for side in ("left", "right")
            )
            spec = EmbodimentSpec(name="twin_cups", asset=AssetSpec(path=str(path)), arms=arms)
            robot = EntityEmbodiment(spec)
            builder = newton.ModelBuilder()
            robot.build_into(builder, None)
        plate = builder.add_body(xform=wp.transform((0.3, 0.0, 0.0), wp.quat_identity()), label="plate")
        builder.add_shape_box(plate, hx=PLATE_HALF[0], hy=PLATE_HALF[1], hz=PLATE_HALF[2])
        part = types.SimpleNamespace(name="plate", range=types.SimpleNamespace(bodies=(plate, plate + 1)))
        robot.after_scene_built(builder, [robot, part])
        model = builder.finalize(device="cpu")
        robot.bind(model)
        labels = model.body_label
        parent_of = dict(zip(model.joint_child.numpy(), model.joint_parent.numpy(), strict=True))
        pads = {arm: cup.pad_body for arm, cup in robot.suctions.items()}
        self.assertEqual(["left", "right"], sorted(pads))
        self.assertEqual(2, len(set(pads.values())), "both arms took one pad")
        for arm, pad in pads.items():
            with self.subTest(arm):
                self.assertTrue(labels[pad].endswith("/suction_pad"))
                self.assertEqual(pad, robot.arms[arm].ee_body, "the tool centre is this arm's face")
                body = pad
                while not labels[body].endswith("_link2"):
                    body = int(parent_of[body])
                self.assertTrue(labels[body].endswith(f"{arm}_link2"), f"{arm}'s pad hangs off {labels[body]}")
                self.assertEqual([plate], sorted(robot.suctions[arm].rows))


class TestOnlyACupKeepsARobotToOneWorld(unittest.TestCase):
    """The compiler refuses to replicate an entity that puts kernels in the substep.

    A cup's wrench is such a kernel and addresses world 0 alone; a robot with
    jaws has none and replicates, as every robot did before cups existed.
    """

    def test_jaws_replicate_and_a_cup_is_refused(self) -> None:
        from robosimstudio.robots.entity import EntityEmbodiment
        from robosimstudio.robots.registry import get_embodiment
        from robosimstudio.scene.compiler import _adds_substep_kernels

        self.assertFalse(_adds_substep_kernels(EntityEmbodiment(get_embodiment("yam_bimanual"))))
        self.assertTrue(_adds_substep_kernels(EntityEmbodiment(get_embodiment("yam_bimanual_suction"))))


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

    def test_a_pose_gone_non_finite_reads_nan_rather_than_raising(self) -> None:
        """A blown-up state still gives the page's readout a number to print, as a finite one does."""
        from robosimstudio.core.suction import face_on_box

        gap, tilt, covered = face_on_box(np.full(7, np.nan), 0.015, PLATE, PLATE_HALF)
        self.assertTrue(math.isnan(gap) and math.isnan(tilt))
        self.assertFalse(covered)


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

    def test_the_seal_readout_leaves_last_to_the_hold(self) -> None:
        """The page's **Seal** readout re-tests the cup every refresh; ``last`` stays what the hold found."""
        from robosimstudio.core.suction import Suction
        from robosimstudio_studio.panels import seal_text

        cup = Suction(self.model, arm="right", pad_body=self.pad, pad_radius=0.015, rows={self.plate: 0})
        now = [self.poses(_down(z=0.0065))]
        # A CPU model has no MuJoCo solver, so the hold is the wrench tier's.
        cup.pipeline = types.SimpleNamespace(
            state_0=types.SimpleNamespace(body_q=types.SimpleNamespace(numpy=lambda: now[0]))
        )
        held = cup.hold()
        self.assertTrue(held.sealed and cup.holding, held.why)
        # The part rides 1 mm lower under the cup than where it sealed.
        now[0] = self.poses(_down(z=0.0075))
        self.assertTrue(seal_text(cup).startswith("sealed, gap 1.5 mm"), seal_text(cup))
        self.assertIs(held, cup.last, "the readout's look replaced the hold's report")
        cup.release()
        # Not holding, the readout looks under the cup; the refusal of an armed box is a hold too.
        now[0] = self.poses(_down(z=0.010))
        refused = cup.hold()
        now[0] = self.poses(_down(z=0.0065))
        self.assertTrue(seal_text(cup).startswith("not sealed, gap 0.5 mm"), seal_text(cup))
        self.assertIs(refused, cup.last)
        # A test the cup is asked for still records.
        self.assertIs(cup.seal(self.plate), cup.last)

    def _cup(self, now: list):
        """A cup over the plate and the block whose pipeline reads ``now[0]``; on a CPU model, the wrench tier."""
        from robosimstudio.core.suction import Suction

        cup = Suction(self.model, arm="right", pad_body=self.pad, pad_radius=0.015, rows={self.plate: 0, self.block: 1})
        cup.pipeline = types.SimpleNamespace(
            state_0=types.SimpleNamespace(body_q=types.SimpleNamespace(numpy=lambda: now[0]))
        )
        return cup

    def test_asking_again_for_the_held_part_keeps_it(self) -> None:
        """A wrench hold rides millimetres under the face, past the seal's gap; asked again, the cup keeps it."""
        now = [self.poses(_down(z=0.0065))]
        cup = self._cup(now)
        self.assertTrue(cup.hold(self.plate).sealed)
        offset = cup._offset_dev.numpy().copy()
        # The plate rides 5.5 mm under the face, as a 2 kg plate does on the wrench tier.
        now[0] = self.poses(_down(z=0.0115))
        for ask in ({"body": self.plate}, {}, {"body": self.plate, "tier": "wrench"}):
            again = cup.hold(**ask)
            self.assertTrue(again.sealed, again.why)
            self.assertEqual((self.plate, "wrench"), (cup.held, cup.tier))
            self.assertAlmostEqual(0.0055, again.gap_m, places=6)
            self.assertIs(again, cup.last)
        np.testing.assert_array_equal(offset, cup._offset_dev.numpy(), "the hold was re-made where the plate is now")

    def test_a_weld_hold_forgets_an_earlier_break(self) -> None:
        """``broke`` is since the last hold, so a weld hold after a wrench seal broke reads False."""
        import warp as wp

        now = [self.poses(_down(z=0.0065))]
        cup = self._cup(now)
        rows = types.SimpleNamespace(
            equality_constraint_enabled=wp.zeros(2, dtype=wp.bool, device="cpu"),
            equality_constraint_relpose=wp.zeros(2, dtype=wp.transform, device="cpu"),
        )
        # What the weld tier touches of a SolverMuJoCo: its rows and the notify that copies them.
        cup.mujoco = types.SimpleNamespace(
            model=types.SimpleNamespace(mujoco=rows), notify_model_changed=lambda flags: None
        )
        cup.weldable = frozenset({self.plate})
        cup._broke_dev.fill_(1)
        self.assertTrue(cup.broke, "the stand-in for a wrench seal that broke")
        report = cup.hold(self.plate, tier="weld")
        self.assertTrue(report.sealed and cup.tier == "weld", report.why)
        self.assertTrue(bool(rows.equality_constraint_enabled.numpy()[0]))
        self.assertFalse(cup.broke, "a weld hold still reports a wrench seal broken before it")

    def test_an_armed_box_looks_at_the_cup_once_a_step(self) -> None:
        """Ticked and 3.5 mm over the plate: the drain's hold is the step's one look, and the readout shows it."""
        from robosimstudio_studio.panels import PanelArm, seal_text

        looks = [0]

        def body_q():
            looks[0] += 1
            return self.poses(_down(z=0.0095))

        cup = self._cup([None])
        cup.pipeline.state_0.body_q.numpy = body_q
        panel = types.SimpleNamespace(suction=cup, vacuum=False, vacuum_request=True, _held_last=False, _looked=None)
        PanelArm._drain_vacuum(panel)
        text = seal_text(cup, panel._looked)
        self.assertEqual(1, looks[0], "the readout copied the poses and tested the cup again")
        self.assertTrue(text.startswith("not sealed, gap 3.5 mm"), text)
        self.assertFalse(cup.holding)

    def test_a_held_cup_is_asked_once_a_drain(self) -> None:
        """On the wrench tier whether the cup holds is a read off the device; the drain makes one."""
        from robosimstudio_studio.panels import PanelArm

        cup = self._cup([self.poses(_down(z=0.0065))])
        self.assertTrue(cup.hold(self.plate).sealed)
        reads, device = [0], cup._held_dev

        class _Counted:
            def numpy(self):
                reads[0] += 1
                return device.numpy()

            def assign(self, values):
                device.assign(values)

        cup._held_dev = _Counted()
        panel = types.SimpleNamespace(suction=cup, vacuum=True, vacuum_request=None, _held_last=True, _looked=None)
        PanelArm._drain_vacuum(panel)
        self.assertEqual(1, reads[0])
        self.assertTrue(panel.vacuum and cup.holding)

    def test_a_refused_part_leaves_the_held_one_held(self) -> None:
        now = [self.poses(_down(z=0.0065))]
        cup = self._cup(now)
        self.assertTrue(cup.hold(self.plate).sealed)
        refused = cup.hold(self.block)
        self.assertFalse(refused.sealed)
        self.assertEqual(self.plate, cup.held, "a seal that failed on the block let the plate go")


class TestAPartOfSeveralShapes(unittest.TestCase):
    """A plate with a small boss 50 mm off its middle: two boxes, and the cup is over the plate's."""

    def test_the_face_the_cup_is_over_seals_though_another_is_nearer(self) -> None:
        import newton
        import warp as wp

        from robosimstudio.core.suction import Suction

        builder = newton.ModelBuilder()
        pad = builder.add_body(xform=wp.transform_identity())
        builder.add_shape_cylinder(pad, radius=0.015, half_height=0.003)
        plate = builder.add_body(xform=wp.transform_identity())
        builder.add_shape_box(plate, hx=PLATE_HALF[0], hy=PLATE_HALF[1], hz=PLATE_HALF[2])
        # 0.5 mm proud of the plate's top, 20 mm square, 50 mm along x: nowhere near the cup's 15 mm radius.
        builder.add_shape_box(
            plate, xform=wp.transform((0.05, 0.0, 0.0005), wp.quat_identity()), hx=0.01, hy=0.01, hz=0.003
        )
        model = builder.finalize(device="cpu")
        cup = Suction(model, arm="right", pad_body=pad, pad_radius=0.015, rows={plate: 0})
        body_q = np.zeros((2, 7))
        body_q[:, 6] = 1.0
        body_q[pad], body_q[plate] = _down(z=0.007), PLATE
        # The boss's top is 0.5 mm from the face and the plate's 1 mm; only the plate's is under the cup.
        report = cup.seal(plate, body_q)
        self.assertTrue(report.sealed, report.why)
        self.assertAlmostEqual(0.001, report.gap_m, places=6)
        # Off both, the refusal is still the nearest face's.
        body_q[pad] = _down(z=0.012)
        report = cup.seal(plate, body_q)
        self.assertFalse(report.sealed)
        self.assertAlmostEqual(0.0055, report.gap_m, places=6)


class TestTheWrenchBreaksAtOneStretch(unittest.TestCase):
    """The wrench tier lets go at ``break_force / WRENCH_STIFFNESS`` of stretch (12.4 mm), whatever the part weighs.

    The kernel on a CPU model at 960 Hz, where a 5 g part's spring is softened to 1152 N/m.
    """

    def _breaks(self, mass: float, stretch: float) -> bool:
        import newton
        import warp as wp

        from robosimstudio.core.suction import Suction

        builder = newton.ModelBuilder()
        pad = builder.add_body(xform=wp.transform_identity(), mass=1.0)
        part = builder.add_body(xform=wp.transform((0.0, 0.0, -stretch), wp.quat_identity()), mass=mass)
        model = builder.finalize(device="cpu")
        cup = Suction(model, arm="right", pad_body=pad, pad_radius=0.015)
        cup._held_dev.assign(np.array([part], dtype=np.int32))
        state = model.state()
        cup.apply(state, 1.0 / 960.0)
        return cup.broke

    def test_a_light_part_and_a_heavy_one_let_go_at_the_same_stretch(self) -> None:
        from robosimstudio.core.suction import BREAK_PRESSURE, WRENCH_STIFFNESS

        limit = BREAK_PRESSURE * math.pi * 0.015**2 / WRENCH_STIFFNESS
        self.assertAlmostEqual(0.0124, limit, places=4)
        for mass in (0.005, 2.0):
            with self.subTest(mass=mass):
                self.assertFalse(self._breaks(mass, limit - 0.001))
                self.assertTrue(self._breaks(mass, limit + 0.001))


class _Program:
    """What :meth:`Suction.attach` asks of a :class:`~robosimstudio.scene.physics.compose.PhysicsProgram`.

    ``islands`` is ``(name, solver, {entity: bodies})`` in stepping order; :attr:`mujoco_solver`
    is the real program's rule, the first MuJoCo island's solver.
    """

    def __init__(self, *islands: tuple[str, str, dict[str, tuple[int, ...]]]) -> None:
        self.islands = [
            types.SimpleNamespace(
                name=name,
                backend=types.SimpleNamespace(solver=solver),
                entities=tuple(owned),
                bodies=np.array(sorted(b for bodies in owned.values() for b in bodies), dtype=np.int32),
            )
            for name, solver, owned in islands
        ]
        self._solvers = {
            name: types.SimpleNamespace(model=types.SimpleNamespace(mujoco=object()), island=name)
            for name, _, _ in islands
        }

    def island_of(self, entity: str):
        for island in self.islands:
            if entity in island.entities:
                return island
        raise KeyError(entity)

    def solver_of_island(self, name: str):
        return self._solvers[name]

    @property
    def mujoco_solver(self):
        return next((self._solvers[i.name] for i in self.islands if i.backend.solver == "mujoco"), None)


class TestTheWeldIsThePadsOwnSolver(unittest.TestCase):
    """A weld row holds a part only inside the solver that steps the pad, which need not be the first MuJoCo one."""

    def _attached(self, program: _Program):
        from robosimstudio.core.suction import Suction

        model = types.SimpleNamespace(device="cpu")
        cup = Suction(
            model,
            arm="right",
            pad_body=1,
            pad_radius=0.015,
            rows={3: 0, 4: 1},
            owners={3: "plate_a", 4: "plate_b"},
        )
        cup.attach(types.SimpleNamespace(program=program))
        return cup

    def test_a_part_on_another_mujoco_island_is_held_by_the_wrench(self) -> None:
        cup = self._attached(
            _Program(
                ("mujoco", "mujoco", {"plate_a": (3,)}), ("mujoco_1", "mujoco", {"robot": (0, 1), "plate_b": (4,)})
            )
        )
        self.assertEqual(frozenset({4}), cup.weldable)
        self.assertEqual("mujoco_1", cup.mujoco.island, "the rows were looked for in another solver")
        self.assertFalse(cup.weld_available(3))

    def test_a_pad_on_no_mujoco_island_welds_nothing(self) -> None:
        cup = self._attached(
            _Program(
                ("featherstone", "featherstone", {"robot": (0, 1)}),
                ("mujoco", "mujoco", {"plate_a": (3,), "plate_b": (4,)}),
            )
        )
        self.assertEqual((None, frozenset()), (cup.mujoco, cup.weldable))

    def test_one_island_for_both_welds_every_part_with_a_row(self) -> None:
        cup = self._attached(_Program(("mujoco", "mujoco", {"robot": (0, 1), "plate_a": (3,), "plate_b": (4,)})))
        self.assertEqual(frozenset({3, 4}), cup.weldable)


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
        self.owners = {7: "plate", 9: "roller"}
        """What the robot builds it with: each part it can seal on, by the entity that built it."""

    def hold(self, body=None, *, tier="auto"):
        self.calls.append(("hold", body, tier))
        if self.report.sealed:
            self.held, self.tier = self.report.body, "weld"
        return self.report

    def release(self):
        self.calls.append(("release",))
        self.held, self.tier = -1, ""


def _env(cups: dict) -> types.SimpleNamespace:
    plate = types.SimpleNamespace(body=7)
    cloth = types.SimpleNamespace(particles=[0, 1])
    roller = types.SimpleNamespace(body=8, handle_body=9)  # the drum is the body; the cup is on the handle's web
    scene = types.SimpleNamespace(
        embodiment=types.SimpleNamespace(suctions=cups), by_name={"plate": plate, "towel": cloth, "roller": roller}
    )
    return types.SimpleNamespace(scene=scene)


class TestSimSuction(unittest.TestCase):
    def test_an_arm_without_a_cup_is_refused_with_the_cup_arms(self) -> None:
        from robosimstudio.sim.sim import Sim

        cup = object()
        sim = Sim.__new__(Sim)
        sim.scene = types.SimpleNamespace(embodiment=types.SimpleNamespace(suctions={"right": cup}))
        self.assertIs(cup, sim.suction("right"))
        self.assertIs(cup, sim.suction())
        with self.assertRaises(KeyError) as caught:
            sim.suction("left")
        self.assertIn("no suction hand on arm 'left'; the cups are on ['right']", str(caught.exception))


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

    def test_a_part_off_its_entitys_body_is_named(self) -> None:
        """Sealed on a roller's handle, the cup held "": the scene was asked whose ``body`` it is, the drum's."""
        from robosimstudio.core.suction import SealReport

        self.cup.report = SealReport(True, 0.0004, 0.0, 9)
        out = self.tool.vacuum()
        self.assertEqual((True, "roller"), (out["sealed"], out["held"]))

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
