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


CLEAN = {
    "ties": [],
    "welds": [],
    "drives": [],
    "drive_rows": [],
    "kinematic": [],
    "real_arm_torque": physics.ARM_TORQUE["yam"],
}


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
        self.assertIn("crate ended 300 mm from where the written run left it", result["report"])

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

    def test_ends_turned_opposite_ways_are_not_the_same_end(self) -> None:
        import math

        def quat(deg: float) -> list[float]:
            return [0.0, 0.0, math.sin(math.radians(deg) / 2), math.cos(math.radians(deg) / 2)]

        runs = _clean_runs()
        runs["written"]["things"]["crate"].update(turn_end_deg=40.0, end_quat=quat(40))
        runs["built"]["things"]["crate"].update(turn_end_deg=40.0, end_quat=quat(-40))
        self.assertIs(_words(physics.verdict(CLEAN, runs))["built"], False)

    def test_free_flight_is_a_note(self) -> None:
        runs = _clean_runs()
        runs["written"]["things"]["crate"]["free_flight_not_falling"] = 6
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["flight"])
        self.assertIs(_words(result)["written"], True)

    def test_open_is_not_read_when_the_written_run_moved_nothing(self) -> None:
        runs = _clean_runs()
        runs["written"]["things"]["crate"]["moved_max_mm"] = 3.0
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["open"])
        self.assertIn("not read: the written replay moved nothing", result["report"])

    def test_a_thing_set_back_where_it_started_is_not_one_that_never_moved(self) -> None:
        runs = _clean_runs()
        runs["written"]["things"]["crate"].update(moved_max_mm=150.0, end=[0.4, 0.0, 0.9])  # lifted and set back
        runs["open"] = _run(moved=0.0, end=(0.4, 0.0, 0.9), turn=0.0)  # finished, never moved, ends in the same place
        self.assertIs(_words(physics.verdict(CLEAN, runs))["open"], True)

    def test_a_lid_opened_is_something_moved(self) -> None:
        runs = _clean_runs()
        for run in runs.values():
            run["things"]["crate"]["moved_max_mm"] = 0.0
        runs["written"]["things"]["case#1"] = _thing(180.0, end=(0.1, 0.0, 0.9), turn=105.0)
        runs["built"]["things"]["case#1"] = _thing(2.0, end=(0.0, 0.0, 0.8), turn=0.0)  # the capped arm never opened it
        result = physics.verdict(CLEAN, runs)
        self.assertIn("moved case#1", result["report"])
        self.assertIs(_words(result)["built"], False)

    def test_a_deformable_settling_with_the_robot_frozen_is_a_note(self) -> None:
        runs = _clean_runs()
        runs["frozen"]["deformables"] = {"water": {"moved_max_mm": 3.4, "end": [0.0, 0.0, 0.8]}}
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["frozen"])
        self.assertIn("water drifted 3.4 mm: still settling", result["report"])

    def test_a_thing_left_alone_but_knocked_away_as_built_fails(self) -> None:
        runs = _clean_runs()
        for run in runs.values():
            run["things"]["cup"] = _thing(0.0)
        runs["built"]["things"]["cup"] = _thing(150.0)  # the sagging arm swept it off the bench
        result = physics.verdict(CLEAN, runs)
        self.assertIs(_words(result)["built"], False)
        self.assertIn("cup, which the written run left alone, was moved 150 mm", result["report"])

    def test_each_kind_of_arm_keeps_its_own_torque(self) -> None:
        model = {**CLEAN, "arm_torque": {"left": physics.ARM_TORQUE["yam"], "right": physics.ARM_TORQUE["panda"]}}
        result = physics.verdict(model, _clean_runs())
        self.assertIn("left 27/10 Nm; right 87/12 Nm", result["report"])

    def test_a_grasp_latch_engaged_fails_contact(self) -> None:
        runs = _clean_runs()
        runs["written"]["latched"] = ["cable"]
        self.assertIs(_words(physics.verdict(CLEAN, runs))["contact"], False)

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

    def test_a_stiff_joint_held_at_its_target_is_elasticity_not_a_motor(self) -> None:
        model = {**CLEAN, "drives": ["flower_1's revolute joint 20 (ke 3)"]}
        result = physics.verdict(model, _clean_runs())
        self.assertIs(_words(result)["contact"], True)
        self.assertIn("1 stiff joint(s) on things held at their targets", result["report"])

    def test_a_stiff_joint_whose_target_moves_is_a_motor(self) -> None:
        model = {**CLEAN, "drives": ["lid's revolute joint 3 (ke 50)"]}
        runs = _clean_runs()
        runs["written"]["thing_targets_moved"] = 1.2
        result = physics.verdict(model, runs)
        self.assertIs(_words(result)["contact"], False)
        self.assertIn("a motor", result["report"])

    def test_a_weld_to_the_world_is_a_note(self) -> None:
        model = {**CLEAN, "welds": ["lamp welded to the world"]}
        result = physics.verdict(model, _clean_runs())
        self.assertIsNone(_words(result)["contact"])
        self.assertTrue(result["passed"])

    def test_a_substep_hook_attached_while_the_routine_played_fails_contact(self) -> None:
        runs = _clean_runs()
        runs["written"]["hooks_added"] = 1
        result = physics.verdict(CLEAN, runs)
        self.assertIs(_words(result)["contact"], False)
        self.assertIn("substep hook(s) attached while the routine played", result["report"])

    def test_a_weld_to_the_robot_switched_on_while_it_played_fails_contact(self) -> None:
        model = {**CLEAN, "robot_equalities": {"3": "pair/left/grab"}}
        runs = _clean_runs()
        runs["written"]["robot_equalities_on"] = [3]
        result = physics.verdict(model, runs)
        self.assertIs(_words(result)["contact"], False)
        self.assertIn("equality row 3 (pair/left/grab) held a thing to the robot", result["report"])

    def test_a_suction_seal_is_a_note(self) -> None:
        model = {**CLEAN, "robot_equalities": {"5": "yam/right/suction/box"}}
        runs = _clean_runs()
        runs["written"]["robot_equalities_on"] = [5]
        result = physics.verdict(model, runs)
        self.assertIsNone(_words(result)["contact"])
        self.assertIn("a suction cup's seal held a thing", result["report"])

    def test_equality_rows_come_off_newton_1_7_s_mujoco_table(self) -> None:
        import numpy as np

        arr = lambda v: types.SimpleNamespace(numpy=lambda: np.asarray(v))  # noqa: E731
        table = types.SimpleNamespace(
            equality_constraint_count=2,
            equality_constraint_body1=arr([4, 1]),
            equality_constraint_body2=arr([9, 2]),
            equality_constraint_objtype=arr([1, 3]),  # a body pair, and a joint coupling
            equality_constraint_label=["hand/grab", "mimic"],
        )
        sim = types.SimpleNamespace(model=types.SimpleNamespace(mujoco=table))
        self.assertEqual(physics.equality_rows(sim), {0: (4, 9, "hand/grab")})

    def test_a_step_hook_the_routine_added_is_a_note(self) -> None:
        runs = _clean_runs()
        runs["written"]["step_hooks_added"] = 1
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["contact"])
        self.assertIn("say in the report what each does", result["report"])

    def test_a_kinematic_thing_on_a_slide_is_still_kinematic(self) -> None:
        import numpy as np

        kinds = physics._joint_kinds()
        arr = lambda v: types.SimpleNamespace(numpy=lambda: np.asarray(v))  # noqa: E731
        model = types.SimpleNamespace(
            body_count=2,
            body_inv_mass=arr([0.0, 0.0]),
            joint_type=arr([kinds["prismatic"], kinds["fixed"]]),
            joint_child=arr([0, 1]),
        )
        self.assertEqual(physics._fixed_bodies(model).tolist(), [True, False])

    def test_a_massless_base_welded_in_place_is_not_kinematic(self) -> None:
        import numpy as np

        kinds = physics._joint_kinds()
        arr = lambda v: types.SimpleNamespace(numpy=lambda: np.asarray(v))  # noqa: E731
        model = types.SimpleNamespace(
            body_count=3,
            body_inv_mass=arr([0.0, 0.0, 1.0]),
            joint_type=arr([kinds["free"], kinds["fixed"]]),
            joint_child=arr([0, 1]),
        )
        self.assertEqual(physics._fixed_bodies(model).tolist(), [True, False, False])

    def test_an_unknown_robot_says_friction_only(self) -> None:
        model = {**CLEAN, "real_arm_torque": None}
        self.assertIn("no torque limits known", physics.verdict(model, _clean_runs())["report"])


def _sim(name: str, labels: list[str], thing_labels: tuple[str, ...] = ()):
    """A robot named ``name`` whose joints (children of body 0) carry ``labels``; a thing's joints after them."""
    import numpy as np

    all_labels = list(labels) + list(thing_labels)
    child = np.array([0] * len(labels) + [1] * len(thing_labels))
    return types.SimpleNamespace(
        scene=types.SimpleNamespace(
            embodiment=types.SimpleNamespace(spec=types.SimpleNamespace(name=name)),
            handles=types.SimpleNamespace(robot_shapes=np.array([0])),
        ),
        model=types.SimpleNamespace(
            joint_label=all_labels,
            joint_child=types.SimpleNamespace(numpy=lambda: child),
            shape_body=types.SimpleNamespace(numpy=lambda: np.array([0, 1])),
        ),
    )


class RobotTable(unittest.TestCase):
    def test_a_yam_by_name(self) -> None:
        self.assertEqual(physics.real_arm_torque(_sim("yam_bimanual", [])), physics.ARM_TORQUE["yam"])

    def test_a_ur10_is_not_capped_as_a_ur5(self) -> None:
        self.assertEqual(physics.real_arm_torque(_sim("ur10e_pair", [])), physics.ARM_TORQUE["ur10"])

    def test_a_universal_robots_arm_of_unknown_size_is_left_uncapped(self) -> None:
        labels = ["pair/left_shoulder_pan_joint", "pair/left_shoulder_lift_joint", "pair/left_wrist_1_joint"]
        self.assertIsNone(physics.real_arm_torque(_sim("autolab_ur_pair", labels)))

    def test_the_size_from_the_robot_s_own_joints(self) -> None:
        self.assertEqual(
            physics.real_arm_torque(_sim("pair", ["ur5e/left_shoulder_pan_joint"])), physics.ARM_TORQUE["ur5"]
        )

    def test_a_ur30_is_not_a_ur3(self) -> None:
        self.assertIsNone(physics.real_arm_torque(_sim("ur30_cell", [])))

    def test_a_thing_named_after_a_robot_does_not_name_the_arm(self) -> None:
        self.assertIsNone(physics.real_arm_torque(_sim("arm", ["j1"], thing_labels=("panda_plush/hinge",))))


class Layout(unittest.TestCase):
    def test_a_mirrored_pair_is_held_open_at_plus_and_minus_its_opening(self) -> None:
        """A Panda's hand has no open targets of its own: open is (+w, -w), and 0 would be shut."""
        import numpy as np

        from robosimstudio.scene.handles import ArmHandle

        hand = ArmHandle(
            name="panda",
            arm_coords=tuple(range(7)),
            arm_dofs=tuple(range(7)),
            gripper_coords=(7, 8),
            gripper_dofs=(7, 8),
            gripper_open=0.04,
        )
        arr = lambda v: types.SimpleNamespace(numpy=lambda: np.asarray(v))  # noqa: E731
        sim = types.SimpleNamespace(
            control=types.SimpleNamespace(joint_target_q=types.SimpleNamespace(shape=(9,))),
            model=types.SimpleNamespace(
                joint_coord_count=9,
                joint_dof_count=9,
                joint_child=arr(range(1, 10)),
                joint_q_start=arr(range(10)),
                joint_qd_start=arr(range(10)),
                shape_body=arr([1]),
            ),
            scene=types.SimpleNamespace(
                handles=types.SimpleNamespace(arms={"panda": hand}, robot_shapes=np.array([0])),
                embodiment=types.SimpleNamespace(bodies=tuple(range(1, 10))),
            ),
        )
        lay = physics.layout(sim)
        np.testing.assert_allclose(lay["open"], [0.04, -0.04])
        self.assertEqual(lay["robot"].tolist(), list(range(9)))


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

    def test_a_routine_that_returns_no_generator_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            live = _server(Path(tmp))
            live.ns = {"routine": lambda sim: None}
            live.sim = object()
            with mock.patch.object(physics, "model_facts", return_value=dict(CLEAN)):
                reply = live.cmd_physics()
        self.assertIn("returned NoneType, not a generator", reply["error"])

    def test_a_replay_that_failed_in_its_own_code_is_not_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            live = _server(Path(tmp))
            runs = _clean_runs()
            runs["built"]["outcome"] = f"{physics.FAILED}: RuntimeError: notify"
            live.physics, live.physics_scene, live.physics_model = runs, "abc", dict(CLEAN)
            self.assertIn("did not run to a reading", live.cmd_physics(op="report")["error"])

    def test_a_stopped_written_replay_is_played_again_not_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            live = _server(Path(tmp))
            live.physics, live.physics_scene = {"written": _run(outcome=physics.STOPPED)}, "abc"
            self.assertIn("the written replay did not run to a reading", live.cmd_physics(mode="frozen")["error"])
            self.assertIn("did not run to a reading", live.cmd_physics(op="report")["error"])

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
        args = ctl.build_parser().parse_args(["physics", "--only", "frozen,open", "routine", "--settle-mm", "3"])
        self.assertEqual(
            (args.cmd, args.name, args.only, args.settle_mm), ("physics", "routine", ["frozen", "open"], 3.0)
        )

    def test_refused_while_a_routine_plays_before_any_reset(self) -> None:
        asked = []

        def page(sock, req, timeout):
            asked.append(req["cmd"])
            return {"ok": True, "playing": 1, "paused": False, "resets": 0}

        with mock.patch.object(ctl, "request", side_effect=page), mock.patch("sys.stderr"), mock.patch("sys.stdout"):
            self.assertEqual(ctl.main(["physics"]), 1)
        self.assertEqual(asked, ["status"])

    def test_an_unknown_replay_is_refused_before_the_page_is_asked(self) -> None:
        with mock.patch.object(ctl, "request", side_effect=AssertionError("asked the page")):
            with mock.patch("sys.stderr"):
                self.assertEqual(ctl.main(["physics", "--only", "sideways"]), 2)


if __name__ == "__main__":
    unittest.main()
