# SPDX-License-Identifier: Apache-2.0
"""The Sharpa Wave hand and the UR5e arms it ships on, held to their files.

What is held: the hand's 22 joints and the gains the spec restates are the
vendored MJCF's own; the postures the spec names fit inside every joint's
range; the thumb-index pinch meets where the spec says it does, on both hands,
by forward kinematics on the vendored files; and the composed robots name the
labels their arms and hands are indexed by, with a position servo on every
joint, so Newton drives them from ``joint_target_q``.

MuJoCo on the CPU, no scene, no GPU.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

TIER = "fast"
"""Forward kinematics on the CPU; no scene is built."""

REPO = Path(__file__).resolve().parents[1]
HANDS = REPO / "assets" / "embodiments" / "sharpa_wave"
ROBOTS = REPO / "assets" / "embodiments" / "ur5e_sharpa"


def _model(path: Path):
    import mujoco

    return mujoco.MjModel.from_xml_path(str(path))


def _names(model, kind: str) -> list[str]:
    import mujoco

    obj = {
        "joint": mujoco.mjtObj.mjOBJ_JOINT,
        "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
        "body": mujoco.mjtObj.mjOBJ_BODY,
    }[kind]
    count = {"joint": model.njnt, "actuator": model.nu, "body": model.nbody}[kind]
    return [mujoco.mj_id2name(model, obj, i) for i in range(count)]


def _elastomers(model, side: str, q) -> tuple[np.ndarray, np.ndarray]:
    """The thumb and index fingertip elastomer centres at ``q``, in the palm frame."""
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)

    def centre(finger: str) -> np.ndarray:
        return data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_{finger}_elastomer")].copy()

    return centre("thumb"), centre("index")


class TestTheSpecIsTheFiles(unittest.TestCase):
    def test_the_joints_and_their_gains_are_the_files_own(self) -> None:
        from robosimstudio.robots.grippers.sharpa_wave import FILE_ACTUATION, GAIN, JOINTS, KP, KV, SHARPA_WAVE

        for side in ("right", "left"):
            model = _model(HANDS / f"{side}_sharpa_wave.xml")
            self.assertEqual(_names(model, "joint"), [f"{side}_{j}" for j in JOINTS], side)
            self.assertEqual(_names(model, "actuator"), [f"{side}_{j}_ctrl" for j in JOINTS], side)
            np.testing.assert_allclose(model.actuator_gainprm[:, 0], KP, rtol=1e-6, err_msg=f"{side} kp")
            np.testing.assert_allclose(
                -model.actuator_biasprm[:, 2], KV, atol=2e-3, err_msg=f"{side} kv"
            )  # the left resolves 1e-3 off
        hand = SHARPA_WAVE.for_arm("right")
        self.assertEqual(hand.finger_names, tuple(f"right_{j}" for j in JOINTS))
        self.assertEqual(FILE_ACTUATION.target_ke_per_dof, KP)
        self.assertEqual(FILE_ACTUATION.target_kd_per_dof, KV)
        np.testing.assert_allclose(hand.actuation.target_ke_per_dof, GAIN * np.asarray(KP))

    def test_every_posture_fits_inside_the_joint_ranges(self) -> None:
        from robosimstudio.robots.grippers.sharpa_wave import SHARPA_WAVE

        model = _model(HANDS / "right_sharpa_wave.xml")
        lower, upper = model.jnt_range[:, 0], model.jnt_range[:, 1]
        for name, values in {
            **SHARPA_WAVE.postures,
            "open": SHARPA_WAVE.finger_open_q,
            "closed": SHARPA_WAVE.finger_closed_q,
        }.items():
            q = np.asarray(values)
            self.assertEqual(len(q), 22, name)
            self.assertTrue(np.all(q >= lower - 1e-9) and np.all(q <= upper + 1e-9), f"{name}: {q - upper}")

    def test_the_pinch_meets_at_the_tool_centre_on_both_hands(self) -> None:
        from robosimstudio.robots.grippers.sharpa_wave import FLAT, PINCH, READY, SHARPA_WAVE, SHARPA_WAVE_LEFT

        for side, spec in (("right", SHARPA_WAVE), ("left", SHARPA_WAVE_LEFT)):
            model = _model(HANDS / f"{side}_sharpa_wave.xml")
            thumb, index = _elastomers(model, side, FLAT)
            self.assertAlmostEqual(float(np.linalg.norm(thumb - index)), 0.124, delta=1e-3, msg=f"{side} flat")
            thumb, index = _elastomers(model, side, READY)
            self.assertAlmostEqual(float(np.linalg.norm(thumb - index)), spec.span, delta=1e-3, msg=f"{side} span")
            thumb, index = _elastomers(model, side, PINCH)
            # The elastomer centres meet at their own thickness apart.
            self.assertLess(float(np.linalg.norm(thumb - index)), 0.013, side)
            # The tool centre is the middle of the open pinch.
            thumb, index = _elastomers(model, side, READY)
            np.testing.assert_allclose((thumb + index) / 2.0, spec.tcp_offset, atol=1e-3, err_msg=f"{side} tcp")
            # Ready, the thumb stands off the index finger along the closing axis
            # and not along the approach: the thing goes between them.
            thumb, index = _elastomers(model, side, READY)
            gap = thumb - index
            self.assertAlmostEqual(float(np.dot(gap, spec.close_axis)), spec.span, delta=1e-3, msg=f"{side} close axis")
            self.assertLess(abs(float(np.dot(gap, spec.approach_axis))), 2e-3, f"{side} approach")

    def test_the_fingertip_colliders_all_carry_the_soft_elastomer(self) -> None:
        """Upstream's left file put the soft solref on four visual geoms; here every collider has it."""
        import mujoco

        for side in ("right", "left"):
            model = _model(HANDS / f"{side}_sharpa_wave.xml")
            for finger in ("thumb", "index", "middle", "ring", "pinky"):
                geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_{finger}_elastomer")
                self.assertGreater(geom, -1, f"{side} {finger}")
                self.assertEqual(int(model.geom_contype[geom]), 1, f"{side} {finger} collides")
                np.testing.assert_allclose(model.geom_solref[geom], (0.06, 0.9), err_msg=f"{side} {finger} solref")


class TestTheComposedRobots(unittest.TestCase):
    def test_the_files_name_what_the_embodiments_index(self) -> None:
        from robosimstudio.robots.registry import get_embodiment, resolve_gripper

        for name in ("ur5e_sharpa", "ur5e_sharpa_bimanual"):
            spec = get_embodiment(name)
            model = _model(REPO / "assets" / spec.asset.path)
            joints, bodies = set(_names(model, "joint")), set(_names(model, "body"))
            for arm in spec.arms:
                hand = resolve_gripper(spec, arm)
                self.assertTrue(set(arm.joint_names) <= joints, (name, arm.name, set(arm.joint_names) - joints))
                self.assertTrue(set(hand.finger_names) <= joints, (name, arm.name, set(hand.finger_names) - joints))
                self.assertIn(arm.ee_body_name, bodies, (name, arm.name))
                self.assertIn(arm.base_body_name, bodies, (name, arm.name))
                self.assertIn(hand.tcp_body, bodies, (name, arm.name))
                for side in hand.finger_bodies:
                    for body in side:
                        self.assertIn(body, bodies, (name, arm.name, body))
            # Every joint has its own position servo: Newton drives a <position>
            # from joint_target_q and reads a <general> as direct control.
            self.assertEqual(model.nu, model.njnt, name)
            text = (REPO / "assets" / spec.asset.path).read_text()
            self.assertEqual(text.count("<position "), model.nu, name)
            self.assertNotIn("<general ", text.split("<actuator>")[1], name)

    def test_the_home_holds_the_pinch_points_over_the_middle_of_the_bench(self) -> None:
        """Both pinch points 0.2 m either side of the centre line and 0.1 m up, approaching 35 degrees from vertical."""
        import mujoco

        from robosimstudio.robots.registry import get_embodiment, resolve_gripper

        spec = get_embodiment("ur5e_sharpa_bimanual")
        model = _model(REPO / "assets" / spec.asset.path)
        data = mujoco.MjData(model)
        joints = _names(model, "joint")
        for arm, y in ((spec.arms[0], 0.2), (spec.arms[1], -0.2)):
            hand = resolve_gripper(spec, arm)
            data.qpos[:] = 0.0
            for name, value in zip(arm.joint_names, arm.home_q, strict=True):
                data.qpos[model.jnt_qposadr[joints.index(name)]] = value
            mujoco.mj_forward(model, data)
            palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hand.tcp_body)
            rot = data.xmat[palm].reshape(3, 3)
            pinch = data.xpos[palm] + rot @ np.asarray(hand.tcp_offset)
            np.testing.assert_allclose(pinch, (0.6, y, 0.1), atol=2e-3, err_msg=arm.name)
            s, c = np.sin(np.radians(35.0)), np.cos(np.radians(35.0))
            np.testing.assert_allclose(
                rot @ np.asarray(hand.approach_axis), (-s, 0.0, -c), atol=2e-3, err_msg=f"{arm.name} approach"
            )
            # The thumb closes inward on both arms: +y on the right arm, -y on the left.
            np.testing.assert_allclose(
                rot @ np.asarray(hand.close_axis),
                (0.0, 1.0 if y < 0 else -1.0, 0.0),
                atol=2e-3,
                err_msg=f"{arm.name} close",
            )


if __name__ == "__main__":
    unittest.main()
