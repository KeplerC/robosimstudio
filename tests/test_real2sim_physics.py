# SPDX-License-Identifier: Apache-2.0
"""`./ctl physics`: its verdict lines, its refusals, its robot table. No GPU.

The replays themselves step a built scene; ``test_real2sim_physics_replay``
covers them on the GPU.
"""

from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from robosimstudio_real2sim_agent import ctl, physics
from robosimstudio_real2sim_agent.server import ROUTINE_PLAYING, LiveServer

TIER = "fast"


def _thing(moved: float, end=(0.0, 0.0, 0.0), turn: float = 0.0, hand=((10.0, 20.0), (0.0, 0.0))) -> dict:
    return {
        "moved_max_mm": moved,
        "end": list(end),
        "turn_end_deg": turn,
        "hand_force_p50_p95_N": [list(h) for h in hand],
    }


def _run(
    outcome: str = "finished", moved: float = 300.0, end=(0.4, 0.0, 1.2), turn: float = 180.0, mu=(0.7, 0.7)
) -> dict:
    run = {"outcome": outcome, "steps": 1000, "things": {"crate": _thing(moved, end, turn)}, "deformables": {}}
    if mu:
        run["finger_mu"] = list(mu)
    return run


CLEAN = {"ties": [], "welds": [], "drives": [], "kinematic": [], "real_arm_torque": physics.ARM_TORQUE["yam"]}


def _clean_runs() -> dict:
    return {
        "written": _run(),
        "frozen": _run(moved=0.1, end=(0.4, 0.0, 0.9), turn=0.0),
        "open": _run(outcome="RuntimeError: the crate did not lift", moved=0.0, end=(0.4, 0.0, 0.9), turn=0.0),
        "built": _run(),
        "heavy": _run(),
    }


def _words(result: dict) -> dict[str, bool | None]:
    return {c["name"]: c["ok"] for c in result["checks"]}


class Verdict(unittest.TestCase):
    def test_real_physics_passes_every_line(self) -> None:
        result = physics.verdict(CLEAN, _clean_runs())
        self.assertTrue(result["passed"])
        self.assertEqual(set(_words(result).values()), {True})
        self.assertIn("PASS  frozen", result["report"])
        self.assertIn("27/10 Nm", result["report"])

    def test_a_thing_that_moves_with_the_robot_frozen_fails(self) -> None:
        runs = _clean_runs()
        runs["frozen"]["things"]["crate"]["moved_max_mm"] = 11.5
        result = physics.verdict(CLEAN, runs)
        self.assertFalse(result["passed"])
        self.assertIs(_words(result)["frozen"], False)
        self.assertIn("crate moved 11.5 mm with the robot frozen", result["report"])

    def test_a_task_done_with_the_fingers_open_is_a_note_not_a_fail(self) -> None:
        runs = _clean_runs()
        runs["open"] = _run()  # finished, and the crate ended where the written run left it
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["open"])
        self.assertTrue(result["passed"])
        self.assertIn("say in the report what carries crate", result["report"])

    def test_finished_open_but_ended_elsewhere_is_the_task_not_done(self) -> None:
        runs = _clean_runs()
        runs["open"] = _run(end=(0.4, 0.0, 0.9), turn=0.0)
        self.assertIs(_words(physics.verdict(CLEAN, runs))["open"], True)

    def test_a_routine_that_stops_as_built_fails(self) -> None:
        runs = _clean_runs()
        runs["built"] = _run(outcome="RuntimeError: the receiving hand lost the reel")
        result = physics.verdict(CLEAN, runs)
        self.assertIs(_words(result)["built"], False)
        self.assertIn("the receiving hand lost the reel", result["report"])

    def test_finished_as_built_but_slipped_fails(self) -> None:
        runs = _clean_runs()
        runs["built"] = _run(end=(0.4, 0.0, 0.9), turn=0.0)  # the routine ran out, the crate stayed on the table
        result = physics.verdict(CLEAN, runs)
        self.assertIs(_words(result)["built"], False)
        self.assertIn("did not end where the written run left them", result["report"])

    def test_built_is_not_read_when_the_written_run_failed(self) -> None:
        runs = _clean_runs()
        runs["written"]["outcome"] = runs["built"]["outcome"] = "RoutineError: the loop did not seat"
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["built"])
        self.assertIn("not read: the written replay did not finish", result["report"])

    def test_a_loop_seated_beside_its_start_is_not_one_that_never_moved(self) -> None:
        written, opened = _run(), _run()
        for run, away in ((written, 65.0), (opened, 0.0)):
            run["things"] = {}
            run["deformables"] = {
                "gasket": {"moved_max_mm": 137.0 if away else 0.0, "end": [0.0, -0.1, 0.003], "particle_end_mm": away}
            }
        runs = {**_clean_runs(), "written": written, "open": opened}
        self.assertIs(_words(physics.verdict(CLEAN, runs))["open"], True)

    def test_heavy_failing_is_a_note(self) -> None:
        runs = _clean_runs()
        runs["heavy"] = _run(outcome="RuntimeError: flower 3 slipped")
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["heavy"])
        self.assertTrue(result["passed"])

    def test_pads_grippier_than_rubber_are_a_note(self) -> None:
        runs = _clean_runs()
        runs["written"]["finger_mu"] = [2.0, 4.0]
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["grip"])
        self.assertIn("friction 2 to 4", result["report"])

    def test_a_tie_to_the_robot_fails_contact(self) -> None:
        model = {**CLEAN, "ties": ["crate by a fixed joint to the robot"]}
        result = physics.verdict(model, _clean_runs())
        self.assertIs(_words(result)["contact"], False)
        self.assertFalse(result["passed"])

    def test_a_weld_to_the_world_is_a_note(self) -> None:
        model = {**CLEAN, "welds": ["lamp welded to the world"]}
        result = physics.verdict(model, _clean_runs())
        self.assertIsNone(_words(result)["contact"])
        self.assertTrue(result["passed"])

    def test_an_unknown_robot_says_friction_only(self) -> None:
        model = {**CLEAN, "real_arm_torque": None}
        self.assertIn("no torque limits known", physics.verdict(model, _clean_runs())["report"])


def _sim(name: str, labels: list[str]):
    return types.SimpleNamespace(
        scene=types.SimpleNamespace(embodiment=types.SimpleNamespace(spec=types.SimpleNamespace(name=name))),
        model=types.SimpleNamespace(joint_label=labels),
    )


class RobotTable(unittest.TestCase):
    def test_a_yam_by_name(self) -> None:
        self.assertEqual(physics.real_arm_torque(_sim("yam_bimanual", [])), physics.ARM_TORQUE["yam"])

    def test_a_universal_robots_arm_by_its_joints_whatever_it_is_called(self) -> None:
        labels = ["pair/left_shoulder_pan_joint", "pair/left_shoulder_lift_joint", "pair/left_wrist_1_joint"]
        self.assertEqual(physics.real_arm_torque(_sim("autolab_ur_pair", labels)), physics.ARM_TORQUE["ur5"])

    def test_an_unknown_arm(self) -> None:
        self.assertIsNone(physics.real_arm_torque(_sim("some_arm", ["j1", "j2"])))


def _server(root: Path) -> LiveServer:
    live = LiveServer.__new__(LiveServer)
    live.script, live.scene_id, live.run_dir = [], "abc", root
    live.physics, live.physics_model, live.physics_scene = {}, {}, ""
    live.ns = {"routine": lambda sim, session: iter(())}
    return live


class Refusals(unittest.TestCase):
    def test_a_report_with_no_replays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reply = _server(Path(tmp)).cmd_physics(op="report")
        self.assertFalse(reply["ok"])
        self.assertIn("`./ctl physics` plays them", reply["error"])

    def test_a_report_on_replays_of_an_earlier_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            live = _server(Path(tmp))
            live.physics, live.physics_scene = {"written": _run()}, "older"
            self.assertFalse(live.cmd_physics(op="report")["ok"])

    def test_a_later_replay_before_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reply = _server(Path(tmp)).cmd_physics(mode="frozen")
        self.assertIn("replay `written` first", reply["error"])

    def test_an_unknown_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reply = _server(Path(tmp)).cmd_physics(mode="sideways")
        self.assertIn("no physics replay 'sideways'", reply["error"])

    def test_a_stopped_written_replay_is_played_again_not_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            live = _server(Path(tmp))
            live.physics, live.physics_scene = {"written": _run(outcome=physics.STOPPED)}, "abc"
            self.assertIn("replay `written` first", live.cmd_physics(mode="frozen")["error"])
            self.assertIn("was stopped before it ended", live.cmd_physics(op="report")["error"])

    def test_refused_while_a_routine_plays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            live = _server(Path(tmp))
            live.script = [object()]
            self.assertEqual(live.cmd_physics()["error"], ROUTINE_PLAYING)

    def test_no_routine_to_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            live = _server(Path(tmp))
            live.ns = {}
            self.assertIn("binds no `routine`", live.cmd_physics()["error"])

    def test_the_report_writes_physics_json_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            live = _server(Path(tmp))
            live.physics, live.physics_scene, live.physics_model = _clean_runs(), "abc", dict(CLEAN)
            reply = live.cmd_physics(op="report", note="why")
            self.assertTrue(reply["ok"] and reply["passed"])
            self.assertTrue((Path(tmp) / "physics.json").exists())
            self.assertIn('"cmd": "physics"', (Path(tmp) / "log.jsonl").read_text())


class CommandLine(unittest.TestCase):
    def test_the_replays_parse(self) -> None:
        args = ctl.build_parser().parse_args(["physics", "routine", "--only", "frozen", "open", "--settle-mm", "3"])
        self.assertEqual(
            (args.cmd, args.name, args.only, args.settle_mm), ("physics", "routine", ["frozen", "open"], 3.0)
        )

    def test_an_unknown_replay_is_refused_before_the_page_is_asked(self) -> None:
        with mock.patch.object(ctl, "request", side_effect=AssertionError("asked the page")):
            with mock.patch("sys.stderr"):
                self.assertEqual(ctl.main(["physics", "--only", "sideways"]), 2)


if __name__ == "__main__":
    unittest.main()
