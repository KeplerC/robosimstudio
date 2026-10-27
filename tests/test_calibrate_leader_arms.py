# SPDX-License-Identifier: Apache-2.0
"""The leader calibration tool, driven through its page over a bare scene and a simulated bus.

**No serial port is opened and no servo is involved.** ``scripts/calibrate_leader_arms.py``
takes a ``connect`` seam for *opening* a bus, and what it is handed here is the
library's own :class:`~robosimstudio.robots.teleop.simulated.SimulatedBus`: a
table of registers whose ``Present_Position`` this file sets to what a leader
with a *known* calibration would read at each pose the tool asks for. What is
real is the rest of the tool -- the scene it builds (``yam_bimanual`` in
``bench_075``, no task), the viser page and its ghost, the state machine behind
the Start and Capture buttons, the torque-off it sends before it reads
anything, the fit, and the file it writes -- so the claim is that the procedure
recovers the calibration it was shown, keyed by serial, in the file the
examples read, having written nothing to the bus but that torque-off.

The buttons themselves are not pressed: a click only sets a flag
(``start_requested``, ``capture_requested``) for the tool's main loop to act
on, and this file sets the same flags and calls the same ``tick()``.

Three passes, in the order a person runs them: the full sweep on a fresh arm,
``--zero-only`` after that arm's horns have slipped, and ``--gripper-only``
after its finger stops moved. Then the refusals: a re-zero against a scale
nobody vouches for, and a partial pass against an arm the file does not have.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robosimstudio.robots.teleop.lerobot_controller import build_action, load_calibration
from robosimstudio.robots.teleop.simulated import SimulatedBus
from tests.test_calibrate_leader_fit import TRUE_OFFSET, TRUE_SIGN, TRUE_TPR, _ticks, _tool

TIER = "medium"
"""Builds a scene and opens a viser page on it."""

ROOT = Path(__file__).resolve().parents[1]
PORT = 0


GRIP_OPEN, GRIP_CLOSED = 1500.0, 2300.0


class _Rig:
    """One scene, one page, and a bus whose readings this test writes."""

    def __init__(self) -> None:
        from robosimstudio_studio.garments import viewer_studio

        self.cal = _tool()
        self.sim = self.cal.build_scene("yam_bimanual", "bench_075")
        self.viewer = viewer_studio(port=PORT, share=False)
        self.viewer.set_model(self.sim.model)
        self.poser = self.cal.Poser(self.sim, self.viewer)
        self.poser.show({})
        self.plan = self.cal.plan_for(self.sim, {"right": "/dev/ch340-TESTSERIAL"})
        self.index = self.plan[0][2]
        self.bus: SimulatedBus | None = None

    def connect(self, port: str, joint_names: list[str], gripper_name: str | None) -> SimulatedBus:
        motors = [*joint_names, *([gripper_name] if gripper_name else [])]
        self.bus = SimulatedBus(motors, present=dict.fromkeys(motors, 2048))
        return self.bus

    def calibrator(self, out: Path, mode: str):
        return self.cal.Calibrator(self.sim, self.poser, self.plan, out, mode=mode, connect=self.connect)

    def place(self, angles: dict[str, float], offset: float = TRUE_OFFSET) -> None:
        """Put the pretend leader's joints at these angles [rad], by motor name."""
        for motor, angle in angles.items():
            self.bus.registers[motor]["Present_Position"] = _ticks(angle, offset)

    def capture(self, calibrator) -> None:
        calibrator.capture_requested = True
        calibrator.tick()

    def close(self) -> None:
        try:
            self.viewer.close()
        except Exception:
            pass
        self.sim.close()


class TestTheCalibrationTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rig = _Rig()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "leader.json"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.rig.close()
        cls.tmp.cleanup()

    # ------------------------------------------------------------- the sweep
    def test_01_the_full_sweep_recovers_the_calibration_it_was_shown(self) -> None:
        rig, cal = self.rig, self.rig.cal
        calibrator = rig.calibrator(self.out, "full")
        self.assertEqual(calibrator.phase, "waiting")
        calibrator.start_requested = True
        calibrator.tick()
        self.assertEqual(calibrator.phase, "joint")
        self.assertIsNotNone(rig.bus, "Start opened the bus through the seam")
        self.assertEqual(
            rig.bus.calls,
            [("disable_torque", tuple(rig.bus.registers))],
            "and the tool itself took torque off every motor, before reading anything",
        )
        joints = len(rig.index.arm_coords)
        for j in range(joints):
            for point in range(len(cal.SWEEP_DEG)):
                self.assertEqual((calibrator._joint_i, calibrator._point_i), (j, point))
                rig.place({f"right_j{j}": calibrator._target_rad})
                rig.capture(calibrator)
        self.assertEqual(calibrator.phase, "gripper_open")
        rig.bus.registers["right_grip"]["Present_Position"] = GRIP_OPEN
        rig.capture(calibrator)
        self.assertEqual(calibrator.phase, "gripper_closed")
        rig.bus.registers["right_grip"]["Present_Position"] = GRIP_CLOSED
        rig.capture(calibrator)
        self.assertTrue(calibrator.done)

        written = json.loads(self.out.read_text())
        self.assertEqual(list(written), ["TESTSERIAL"], "keyed by the adapter's serial, not the arm's name")
        entry = written["TESTSERIAL"]
        self.assertEqual(entry["label"], "right")
        self.assertEqual([j["name"] for j in entry["joints"]], [f"right_j{i}" for i in range(joints)])
        for joint in entry["joints"]:
            with self.subTest(joint=joint["name"]):
                self.assertAlmostEqual(joint["ticks_per_radian"], TRUE_TPR, delta=0.5)
                self.assertEqual(joint["sign"], TRUE_SIGN)
                self.assertAlmostEqual(joint["offset_ticks"], TRUE_OFFSET, delta=0.5)
                self.assertEqual(joint["scale_source"], "fitted")
        self.assertEqual(entry["gripper"], {"name": "right_grip", "open_ticks": GRIP_OPEN, "closed_ticks": GRIP_CLOSED})
        self.assertEqual(rig.bus.calls[-1], ("disconnect",), "the bus is released when the arm is done")
        self.assertEqual(
            {call[0] for call in rig.bus.calls[1:-1]}, {"sync_read"}, "and between the two it was only ever read"
        )

    def test_02_the_file_reads_back_through_the_library_at_the_sim_s_home(self) -> None:
        calibration = load_calibration(self.out)["TESTSERIAL"]
        home = [float(q) for q in self.rig.index.home_q]
        ticks = {f"right_j{i}": _ticks(q) for i, q in enumerate(home)}
        ticks["right_grip"] = GRIP_OPEN
        action = build_action(ticks, [calibration])
        for i, q in enumerate(home):
            self.assertAlmostEqual(float(action[i]), q, places=3)
        self.assertAlmostEqual(float(action[-1]), 0.0, places=6)

    # ---------------------------------------------------------- the re-zero
    def test_03_zero_only_moves_every_offset_and_keeps_everything_else(self) -> None:
        """The horns slipped by 37 counts: one capture at home puts every zero back."""
        rig = self.rig
        before = json.loads(self.out.read_text())["TESTSERIAL"]
        calibrator = rig.calibrator(self.out, "zero")
        calibrator.start_requested = True
        calibrator.tick()
        self.assertEqual(calibrator.phase, "zero")
        slipped = TRUE_OFFSET + 37.0
        rig.place({f"right_j{i}": float(q) for i, q in enumerate(rig.index.home_q)}, offset=slipped)
        rig.capture(calibrator)
        self.assertTrue(calibrator.done)
        after = json.loads(self.out.read_text())["TESTSERIAL"]
        for was, now in zip(before["joints"], after["joints"], strict=True):
            with self.subTest(joint=now["name"]):
                self.assertAlmostEqual(now["offset_ticks"], slipped, delta=0.5)
                self.assertEqual(now["ticks_per_radian"], was["ticks_per_radian"], "the scale is kept")
                self.assertEqual(now["sign"], was["sign"])
                self.assertEqual(now["scale_source"], "fitted")
        self.assertEqual(after["gripper"], before["gripper"], "a pass that did not measure the gripper keeps it")

    def test_04_gripper_only_moves_the_endpoints_and_keeps_every_joint(self) -> None:
        rig = self.rig
        before = json.loads(self.out.read_text())["TESTSERIAL"]
        calibrator = rig.calibrator(self.out, "gripper")
        calibrator.start_requested = True
        calibrator.tick()
        self.assertEqual(calibrator.phase, "gripper_open")
        rig.bus.registers["right_grip"]["Present_Position"] = 1400.0
        rig.capture(calibrator)
        rig.bus.registers["right_grip"]["Present_Position"] = 2400.0
        rig.capture(calibrator)
        self.assertTrue(calibrator.done)
        after = json.loads(self.out.read_text())["TESTSERIAL"]
        self.assertEqual(after["gripper"], {"name": "right_grip", "open_ticks": 1400.0, "closed_ticks": 2400.0})
        self.assertEqual(after["joints"], before["joints"])

    # ------------------------------------------------------------ refusals
    def test_05_a_re_zero_refuses_a_scale_nobody_vouches_for(self) -> None:
        rig, cal = self.rig, self.rig.cal
        out = Path(self.tmp.name) / "seeded.json"
        entry = {
            "label": "right",
            "joints": [
                {"name": f"right_j{i}", "ticks_per_radian": 651.9, "sign": 1.0, "offset_ticks": 2048.0}
                for i in range(len(rig.index.arm_coords))
            ],
        }
        out.write_text(json.dumps({"TESTSERIAL": entry}))
        self.assertFalse(cal.scale_is_vouched(entry["joints"]))
        calibrator = rig.calibrator(out, "zero")
        calibrator.start_requested = True
        with self.assertRaises(SystemExit) as caught:
            calibrator.tick()
        self.assertIn("refuses", str(caught.exception))
        self.assertIn("scale_source", str(caught.exception))

    def test_06_a_partial_pass_refuses_an_arm_the_file_does_not_have(self) -> None:
        out = Path(self.tmp.name) / "empty.json"
        out.write_text("{}")
        for mode in ("zero", "gripper"):
            with self.subTest(mode=mode):
                calibrator = self.rig.calibrator(out, mode)
                calibrator.start_requested = True
                with self.assertRaises(SystemExit) as caught:
                    calibrator.tick()
                self.assertIn("Run the full sweep", str(caught.exception))

    def test_07_a_leader_this_robot_has_no_arm_for_is_refused_by_name(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self.rig.cal.plan_for(self.rig.sim, {"middle": "/dev/ch340-X"})
        self.assertIn("middle", str(caught.exception))
        self.assertIn("left", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
