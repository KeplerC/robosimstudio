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

import numpy as np

from robosimstudio_real2sim_agent import ctl, physics
from robosimstudio_real2sim_agent.server import LiveServer
from robosimstudio_real2sim_agent.wire import ROUTINE_PLAYING

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


def linear_4310_clamp():
    """The registered YAM's arena hand's own servo (its 50 N clamp), or None where it has none."""
    from robosimstudio.robots.registry import get_embodiment, resolve_gripper

    yam = get_embodiment("yam_bimanual")
    hand = resolve_gripper(yam, yam.arms[0])
    return getattr(hand, "actuation", None)


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

    def test_a_hinged_part_that_settles_is_said_so(self) -> None:
        runs = _clean_runs()
        runs["frozen"]["things"]["lockbox#2"] = _thing(6.0)
        result = physics.verdict(CLEAN, runs)
        self.assertIs(_words(result)["frozen"], False)
        self.assertIn(
            "lockbox#2 moved 6.0 mm with the robot frozen: a part on a joint swung or settled", result["report"]
        )

    def test_an_attached_thing_is_named_by_frozen_and_failed_by_contact(self) -> None:
        runs = _clean_runs()
        runs["frozen"]["things"]["key"] = _thing(6.0)
        model = {**CLEAN, "ties": ["key by a fixed joint to the robot"], "tie_things": ["key"]}
        result = physics.verdict(model, runs)
        self.assertFalse(result["passed"])
        self.assertIs(_words(result)["contact"], False)
        self.assertIsNone(_words(result)["frozen"])  # the contact line fails it; frozen names it
        self.assertIn("key moved 6.0 mm, held by what the contact line names (a tie", result["report"])
        runs["frozen"]["things"]["crate"]["moved_max_mm"] = 7.0  # a push of a free thing beside it fails, and leads
        result = physics.verdict(model, runs)
        self.assertIs(_words(result)["frozen"], False)
        self.assertIn("crate moved 7.0 mm with the robot frozen: it starts inside something", result["report"])

    def test_an_attachment_is_named_whole_not_inside_another_name(self) -> None:
        runs = _clean_runs()
        runs["frozen"]["things"]["key"] = _thing(3.0)
        for other in ("keyring", "key-ring"):  # the keyring is tied, not the key
            result = physics.verdict({**CLEAN, "ties": [f"{other} by a fixed joint to the robot"]}, runs)
            self.assertIsNone(_words(result)["frozen"])
            self.assertNotIn("the attachment the contact line names", result["report"])

    def test_a_rigid_wobble_still_says_a_deformable_drifting(self) -> None:
        runs = _clean_runs()
        runs["frozen"]["things"]["crate"]["moved_max_mm"] = 3.0
        runs["frozen"]["deformables"] = {"towel": _thing(40.0)}
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["frozen"])
        self.assertIn("towel drifted 40.0 mm", result["report"])

    def test_a_hinged_part_that_settles_a_little_is_a_note_that_says_so(self) -> None:
        runs = _clean_runs()
        runs["frozen"]["things"]["lockbox#2"] = _thing(3.0)
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["frozen"])
        self.assertIn("a part on a joint swung or settled a few millimetres", result["report"])

    def test_a_few_millimetres_with_the_robot_frozen_is_a_note(self) -> None:
        runs = _clean_runs()
        runs["frozen"]["things"]["crate"]["moved_max_mm"] = 3.1  # a bottle loose in a tray, a key in a tight socket
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["frozen"])
        self.assertIn("crate moved 3.1 mm with the robot frozen: a few millimetres", result["report"])
        runs["frozen"]["things"]["crate"]["moved_max_mm"] = physics.PUSHED_MM + 0.1
        self.assertIs(_words(physics.verdict(CLEAN, runs))["frozen"], False)

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

    def test_built_that_lowered_nothing_reads_a_different_end_as_spread(self) -> None:
        runs = _clean_runs()
        runs["built"] = {**_run(end=(0.4, 0.0, 0.9), turn=0.0), "lowered": {"friction": 0, "torque": 0}}
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["built"])  # the scene was already as built: run-to-run spread, not a grip
        self.assertIn("built changed nothing", result["report"])
        self.assertIn("crate ended 300 mm from where the written run left it", result["report"])
        runs["built"] = {**_run(outcome="RuntimeError: lost it"), "lowered": {"friction": 0, "torque": 0}}
        self.assertIsNone(_words(physics.verdict(CLEAN, runs))["built"])
        runs["built"] = {**_run(), "lowered": {"friction": 0, "torque": 0}}
        self.assertIs(_words(physics.verdict(CLEAN, runs))["built"], True)
        runs["built"] = {**_run(end=(0.4, 0.0, 0.9), turn=0.0), "lowered": {"friction": 2, "torque": 0}}
        self.assertIs(_words(physics.verdict(CLEAN, runs))["built"], False)  # lowered something, and it slipped

    def test_built_cannot_tell_a_slip_where_plain_also_ends_elsewhere(self) -> None:
        runs = _clean_runs()
        runs["built"] = _run(end=(0.4, 0.0, 0.9), turn=0.0)
        runs["plain"] = _run(end=(0.6, 0.1, 0.9), turn=0.0)  # nothing changed, and it still landed elsewhere
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["built"])
        self.assertIn("this scene's outcome is not repeatable", result["report"])
        runs["plain"] = _run()  # plain ends where the written run did: the slip is built's
        self.assertIs(_words(physics.verdict(CLEAN, runs))["built"], False)
        runs["built"] = _run(outcome="RuntimeError: the tray slipped")  # a stop is not spread
        runs["plain"] = _run(end=(0.6, 0.1, 0.9), turn=0.0)
        self.assertIs(_words(physics.verdict(CLEAN, runs))["built"], False)

    def test_spread_excuses_only_the_same_thing_about_as_far(self) -> None:
        runs = _clean_runs()
        runs["built"] = _run(end=(0.4, 0.0, 0.9), turn=0.0)  # 300 mm off
        runs["plain"] = _run(end=(0.4, 0.0, 1.145))  # the crate 55 mm off: spread, but a sixth of built's miss
        self.assertIs(_words(physics.verdict(CLEAN, runs))["built"], False)
        for r in ("written", "built"):
            runs[r]["things"]["bottle"] = _thing(10.0, end=(0.1, 0.1, 0.8))
        runs["plain"] = _run()
        runs["plain"]["things"]["bottle"] = _thing(10.0, end=(0.3, 0.1, 0.8))  # plain's spread is on the bottle
        self.assertIs(_words(physics.verdict(CLEAN, runs))["built"], False)
        runs["plain"]["things"]["crate"] = _thing(300.0, end=(0.6, 0.1, 0.9))
        self.assertIsNone(_words(physics.verdict(CLEAN, runs))["built"])  # now the crate lands elsewhere in plain too
        runs["built"]["things"]["crate"]["moved_max_mm"] = 0.0  # never lifted as built: a grip lost, not spread
        self.assertIs(_words(physics.verdict(CLEAN, runs))["built"], False)

    def test_plain_within_the_tolerance_is_no_spread(self) -> None:
        runs = _clean_runs()
        runs["written"]["things"]["bottle"] = _thing(10.0, end=(0.1, 0.1, 0.8))
        runs["built"] = _run(end=(0.4, 0.0, 1.11))  # the crate 90 mm off
        runs["built"]["things"]["bottle"] = _thing(10.0, end=(0.1, 0.1, 0.8))
        runs["plain"] = _run(end=(0.4, 0.0, 1.152))  # 48 mm: within the tolerance, the same end
        runs["plain"]["things"]["bottle"] = _thing(10.0, end=(0.4, 0.1, 0.8))  # the bottle elsewhere
        self.assertIs(_words(physics.verdict(CLEAN, runs))["built"], False)

    def test_a_slip_the_other_way_from_plains_spread_is_not_spread(self) -> None:
        runs = _clean_runs()
        runs["plain"] = _run(end=(0.4, 0.0, 1.26))  # 60 mm one way
        runs["built"] = _run(end=(0.4, 0.0, 1.1))  # 100 mm the other: 160 mm from plain's end
        self.assertIs(_words(physics.verdict(CLEAN, runs))["built"], False)
        runs["built"] = _run(end=(0.4, 0.0, 1.29))  # 90 mm the same way: 30 mm from plain's end
        self.assertIsNone(_words(physics.verdict(CLEAN, runs))["built"])

    def test_a_plain_run_whose_grasp_failed_is_no_spread(self) -> None:
        runs = _clean_runs()
        runs["written"]["things"]["crate"]["end_mm"] = 300.0
        runs["plain"] = _run(moved=5.0, end=(0.4, 0.0, 0.9))  # never carried: it stayed where it started
        runs["plain"]["things"]["crate"]["end_mm"] = 5.0
        runs["built"] = _run(end=(0.4, 0.0, 1.05))  # carried, and slipped 150 mm short
        result = physics.verdict(CLEAN, runs)
        self.assertIs(_words(result)["built"], False)
        self.assertIn("crate ended 5 mm from its start where the written run took it 300", result["report"])
        runs["plain"]["things"]["crate"]["start"] = [0.4, 0.0, 0.9]
        runs["plain"]["things"]["crate"]["end"] = [0.6, 0.0, 0.9]  # shoved 200 mm aside: no nearer the written end
        self.assertIn("where the written run took it", physics.verdict(CLEAN, runs)["report"])
        runs["plain"]["things"]["crate"]["end"] = [0.4, 0.0, 1.15]  # read only at yields, but left near the end
        self.assertNotIn("where the written run took it", physics.verdict(CLEAN, runs)["report"])

    def test_built_that_lowered_nothing_beside_a_deformable_is_not_excused(self) -> None:
        runs = _clean_runs()
        for r in ("written", "built"):
            runs[r]["deformables"] = {"towel": _thing(0.0)}
        runs["built"] = {**runs["built"], **_run(end=(0.4, 0.0, 0.9)), "lowered": {"friction": 0, "torque": 0}}
        runs["built"]["deformables"] = {"towel": _thing(0.0)}
        self.assertIs(_words(physics.verdict(CLEAN, runs))["built"], False)  # a soft contact's friction is not capped

    def test_spread_compares_like_with_like(self) -> None:
        runs = _clean_runs()
        runs["built"] = _run(end=(0.4, 0.0, 0.9))  # dropped 300 mm off, turned as written
        runs["plain"] = _run(turn=0.0)  # in place, but tipped over: a turn, not a position
        self.assertIs(_words(physics.verdict(CLEAN, runs))["built"], False)

    def test_a_stop_as_built_fails_even_where_plain_stopped_too(self) -> None:
        # flower-main: held by friction 4 at a 0.7 N grip, it stops as built ("Flower 3 is outside the pitcher") and,
        # flaky, stops in plain too; excusing the one by the other lost that real catch
        runs = _clean_runs()
        runs["built"] = _run(outcome="RoutineError: the tray slipped")
        runs["plain"] = _run(outcome="RoutineError: the tray slipped")
        self.assertIs(_words(physics.verdict(CLEAN, runs))["built"], False)

    def test_a_broken_built_replay_is_not_read(self) -> None:
        runs = _clean_runs()
        runs["built"] = {**_run(outcome=physics.FAILED + ": RuntimeError: x"), "lowered": {"friction": 0, "torque": 0}}
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["built"])
        self.assertIn("not read: " + physics.FAILED, result["report"])

    def test_spread_shows_plain_in_the_measure_built_missed_by(self) -> None:
        runs = _clean_runs()
        runs["built"] = _run(turn=100.0)  # in place, turned 80 degrees
        runs["plain"] = _run(end=(0.4, 0.0, 1.14), turn=130.0)  # 60 mm off and turned 50 degrees
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["built"])
        self.assertIn(
            "played with nothing changed, crate ended turned 50° from how the written run left it too", result["report"]
        )

    def test_a_push_and_a_drift_are_both_said(self) -> None:
        runs = _clean_runs()
        runs["frozen"]["things"]["crate"]["moved_max_mm"] = 6.0
        runs["frozen"]["deformables"] = {"towel": _thing(40.0)}
        result = physics.verdict(CLEAN, runs)
        self.assertIs(_words(result)["frozen"], False)
        self.assertIn("towel drifted 40.0 mm", result["report"])

    def test_a_hold_that_names_no_thing_is_pointed_at(self) -> None:
        runs = _clean_runs()
        runs["frozen"]["things"]["crate"]["moved_max_mm"] = 3.0
        runs["written"]["hooks_added"] = 1  # only a hook names no thing
        result = physics.verdict(CLEAN, runs)
        self.assertIn("a hook the routine added, which names no thing: say if it holds this", result["report"])
        self.assertIn("crate moved 3.0 mm with the robot frozen: a few millimetres", result["report"])
        runs["written"]["hooks_added"] = 0
        runs["written"]["robot_equalities_on"] = [3]
        model = {**CLEAN, "robot_equalities": {"3": "a weld"}, "robot_equality_things": {"3": "crate"}}
        result = physics.verdict(model, runs)
        self.assertIn("crate moved 3.0 mm with the robot frozen: a few millimetres", result["report"])
        self.assertIn("the contact line names a hold on it (a latch, an equality row, a motor)", result["report"])

    def test_a_grip_lost_on_the_same_model_says_it_is_marginal(self) -> None:
        runs = _clean_runs()
        runs["built"] = {**_run(moved=20.0), "lowered": {"friction": 0, "torque": 0}}  # let go early
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["built"])
        self.assertIn("the grasp is that marginal, or the outcome differs from run to run", result["report"])

    def test_a_row_naming_another_thing_is_not_this_ones_hold(self) -> None:
        runs = _clean_runs()
        runs["frozen"]["things"]["crate"]["moved_max_mm"] = 3.0
        runs["frozen"]["things"]["key"] = _thing(0.1)
        runs["written"]["robot_equalities_on"] = [3]
        model = {**CLEAN, "robot_equalities": {"3": "key_weld"}, "robot_equality_things": {"3": "key"}}
        result = physics.verdict(model, runs)
        self.assertIn("crate moved 3.0 mm with the robot frozen: a few millimetres", result["report"])
        self.assertNotIn("names no thing", result["report"])  # the row's thing is known, and it is the key

    def test_a_thing_named_like_the_fixed_words_is_not_held(self) -> None:
        runs = _clean_runs()
        runs["frozen"]["things"]["1"] = _thing(3.0)
        runs["written"]["hooks_added"] = 1  # "1 substep hook(s) ...": names no thing
        result = physics.verdict(CLEAN, runs)
        self.assertNotIn("held by what the contact line names", result["report"])
        self.assertIn("names no thing: say if it holds this", result["report"])

    def test_a_plain_replay_cut_or_broken_is_not_read(self) -> None:
        runs = _clean_runs()
        for outcome in (physics.STOPPED, physics.FAILED + ": RuntimeError: x"):
            runs["plain"] = _run(outcome=outcome)
            result = physics.verdict(CLEAN, runs)
            self.assertIsNone(_words(result)["plain"])
            self.assertIn("not read: " + outcome[:20], result["report"])

    def test_an_open_replay_cut_or_broken_is_not_read(self) -> None:
        runs = _clean_runs()
        for outcome in (physics.STOPPED, physics.FAILED + ": RuntimeError: x"):
            runs["open"] = _run(outcome=outcome)
            result = physics.verdict(CLEAN, runs)
            self.assertIsNone(_words(result)["open"])
            self.assertIn("not read: " + outcome[:20], result["report"])

    def test_a_latched_thing_pushed_in_frozen_still_fails_and_names_the_latch(self) -> None:
        runs = _clean_runs()
        runs["frozen"]["things"]["crate"]["moved_max_mm"] = 40.0
        result = physics.verdict(CLEAN, runs)
        self.assertIs(_words(result)["frozen"], False)
        runs["written"]["latched"] = ["crate"]  # the contact line fails the latch; frozen says it may be what moved it
        result = physics.verdict(CLEAN, runs)
        self.assertIs(_words(result)["contact"], False)
        self.assertIs(_words(result)["frozen"], False)
        self.assertIn("names a hold on it (a latch, an equality row, a motor): say if that moves it", result["report"])

    def test_a_frozen_replay_cut_or_broken_is_not_read(self) -> None:
        runs = _clean_runs()
        for outcome in (physics.STOPPED, physics.FAILED + ": Broken: holding the robot's targets"):
            for moved in (0.3, 40.0):  # as the server, which reports nothing while a replay is cut or broken
                runs["frozen"] = _run(outcome=outcome, moved=moved, end=(0.4, 0.0, 0.9), turn=0.0)
                result = physics.verdict(CLEAN, runs)
                self.assertIsNone(_words(result)["frozen"])
                self.assertIn("not read: " + outcome[:20], result["report"])

    def test_a_lost_grip_after_another_miss_is_said(self) -> None:
        runs = _clean_runs()
        runs["written"]["things"]["bottle"] = _thing(300.0, end=(0.1, 0.1, 0.8))
        runs["built"] = {**_run(end=(0.4, 0.0, 0.9)), "lowered": {"friction": 0, "torque": 0}}
        runs["built"]["things"]["bottle"] = _thing(20.0, end=(0.1, 0.1, 0.8))  # let go on the way
        result = physics.verdict(CLEAN, runs)
        self.assertIn("bottle moved 20 mm where the written run moved it 300", result["report"])
        self.assertIn("the grasp is that marginal", result["report"])

    def test_a_part_of_a_tied_thing_that_swings_still_fails(self) -> None:
        runs = _clean_runs()
        runs["frozen"]["things"]["box#1"] = _thing(20.0)  # the lid on its own joint
        model = {**CLEAN, "ties": ["box by a fixed joint to the robot"], "tie_things": ["box"]}
        result = physics.verdict(model, runs)
        self.assertIs(_words(result)["frozen"], False)
        self.assertIn("box#1 moved 20.0 mm with the robot frozen: a part on a joint swung", result["report"])

    def test_a_marginal_grasp_names_a_handled_thing_not_a_bystander(self) -> None:
        runs = _clean_runs()
        runs["written"]["things"]["mug"] = _thing(6.0, end=(0.2, 0.2, 0.8))
        runs["plain"] = _run(moved=5.0, end=(0.4, 0.0, 0.9))
        runs["plain"]["things"]["mug"] = _thing(1.0, end=(0.2, 0.2, 0.8))
        runs["built"] = _run(end=(0.4, 0.0, 1.05))
        runs["built"]["things"]["mug"] = _thing(60.0, end=(0.26, 0.2, 0.8))  # knocked
        result = physics.verdict(CLEAN, runs)
        self.assertNotIn("mug was not carried either", result["report"])

    def test_a_slip_twice_as_far_as_plains_spread_is_not_spread(self) -> None:
        runs = _clean_runs()
        runs["plain"] = _run(end=(0.4, 0.0, 1.1))  # 100 mm off
        runs["built"] = _run(end=(0.4, 0.0, 1.01))  # 190 mm off the same way: 90 mm from plain's end
        self.assertIs(_words(physics.verdict(CLEAN, runs))["built"], False)

    def test_a_swinging_part_of_a_tied_thing_is_not_blamed_on_a_latch(self) -> None:
        runs = _clean_runs()
        runs["frozen"]["things"]["lockbox#2"] = _thing(8.0)
        model = {**CLEAN, "ties": ["lockbox by a fixed joint to the robot"], "tie_things": ["lockbox"]}
        self.assertNotIn("a latch, an equality row, a motor", physics.verdict(model, runs)["report"])

    def test_settle_does_not_read_a_cut_frozen_run(self) -> None:
        model = {**CLEAN, "settle": {"crate": {"rise_mm": 0.0, "aside_mm": 0.0, "turn_deg": 0.0, "overlap_mm": 3.0}}}
        runs = _clean_runs()
        runs["frozen"] = _run(outcome=physics.STOPPED, moved=3.1, end=(0.4, 0.0, 0.9), turn=0.0)
        result = physics.verdict(model, runs)
        self.assertNotIn("stays there with the robot frozen", result["report"])

    def test_built_beside_another_solver_is_not_excused_for_lowering_nothing(self) -> None:
        runs = _clean_runs()
        runs["written"]["contacts_unread"] = {"crate": "stepped by SuperDex"}
        runs["built"] = {**_run(end=(0.4, 0.0, 0.9)), "lowered": {"friction": 0, "torque": 0}}
        self.assertIs(_words(physics.verdict(CLEAN, runs))["built"], False)

    def test_settle_says_why_a_cut_frozen_run_tells_it_nothing(self) -> None:
        model = {**CLEAN, "settle": {"crate": {"rise_mm": 0.0, "aside_mm": 0.0, "turn_deg": 0.0, "overlap_mm": 2.0}}}
        runs = _clean_runs()
        runs["frozen"] = _run(outcome=physics.STOPPED, moved=0.5, end=(0.4, 0.0, 0.9), turn=0.0)
        self.assertIn("the frozen replay was not read whole", physics.verdict(model, runs)["report"])

    def test_a_replay_whose_log_failed_is_not_read(self) -> None:
        runs = _clean_runs()
        runs["frozen"] = {**_run(moved=0.0), "log_error": "RuntimeError: x", "things": {}}
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["frozen"])
        self.assertIn("not read: its log failed: RuntimeError: x", result["report"])

    def test_a_written_replay_cut_short_is_not_read(self) -> None:
        runs = _clean_runs()
        runs["written"]["outcome"] = physics.STOPPED
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["written"])
        self.assertIn("not read: " + physics.STOPPED, result["report"])

    def test_plain_carried_a_deformable_it_left_far_from_its_start(self) -> None:
        written = {"deformables": {"gasket": {"moved_max_mm": 536.0, "end_mm": 477.0, "end": [0.0, 0.2, 0.3]}}}
        left = {"deformables": {"gasket": {"moved_max_mm": 138.0, "end_mm": 400.0, "end": [0.0, 0.2, 0.3]}}}
        stayed = {"deformables": {"gasket": {"moved_max_mm": 138.0, "end_mm": 17.0, "end": [0.0, 0.0, 0.0]}}}
        self.assertTrue(physics._carried(left, written, "gasket", sampled=True))
        self.assertFalse(physics._carried(stayed, written, "gasket", sampled=True))
        self.assertFalse(physics._carried(left, written, "gasket"))  # read every step: its largest move decides

    def test_a_part_of_a_tied_thing_says_what_holds_the_thing(self) -> None:
        runs = _clean_runs()
        runs["frozen"]["things"]["box#1"] = _thing(6.0)
        model = {**CLEAN, "ties": ["box by a fixed joint to the robot"], "tie_things": ["box"]}
        self.assertIn("the thing it is part of is held", physics.verdict(model, runs)["report"])

    def test_a_written_run_whose_log_failed_says_why_it_was_not_read(self) -> None:
        runs = _clean_runs()
        runs["written"]["log_error"] = "Traceback\nRuntimeError: rows"
        result = physics.verdict(CLEAN, runs)
        self.assertIn("not read: its log failed: RuntimeError: rows", result["report"])
        self.assertEqual(physics.cut_short({"log_error": "\n"}), "its log failed: no message")

    def test_frozen_pass_says_the_largest_move_of_any_rigid_thing(self) -> None:
        runs = _clean_runs()
        runs["frozen"]["things"]["key"] = _thing(1.8)
        model = {**CLEAN, "ties": ["key by a fixed joint to the robot"], "tie_things": ["key"]}
        self.assertIn("nothing moved more than 1.8 mm", physics.verdict(model, runs)["report"])

    def test_a_built_replay_the_page_cut_is_not_read(self) -> None:
        runs = _clean_runs()
        runs["built"] = {**_run(outcome=physics.STOPPED), "lowered": {"friction": 0, "torque": 0}}
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["built"])
        self.assertIn("not read: " + physics.STOPPED, result["report"])

    def test_built_says_what_it_could_check(self) -> None:
        runs = _clean_runs()
        runs["built"] = {**_run(), "lowered": {"friction": 0, "torque": 0}}
        result = physics.verdict({k: v for k, v in CLEAN.items() if k != "real_arm_torque"}, runs)
        self.assertIn(
            "no rigid shape's friction here is above 1 (no torque limits known for every arm)", result["report"]
        )
        runs["built"] = {**_run(end=(0.4, 0.0, 0.9), turn=0.0), "lowered": None}  # apply() did not finish
        self.assertIs(_words(physics.verdict(CLEAN, runs))["built"], False)

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

    def test_a_thing_another_solver_steps_is_named_not_cleared(self) -> None:
        runs = _clean_runs()
        runs["written"]["things"]["crate"]["hand_force_p50_p95_N"] = [[0.0, 0.0], [0.0, 0.0]]
        runs["written"]["contacts_unread"] = {"crate": "stepped by SolverVBD"}
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["grip"])  # a note to explain, not a pass on a grip nobody read
        self.assertIn(
            "no finger force read; crate (stepped by SolverVBD): not touched by the fingers", result["report"]
        )
        self.assertIn("its contacts are not in MuJoCo's list", result["report"])
        self.assertNotIn("no finger force on a rigid thing", result["report"])

    def test_an_unread_thing_that_did_not_move_is_not_named(self) -> None:
        runs = _clean_runs()
        runs["written"]["contacts_unread"] = {"bystander": "stepped by SolverVBD"}
        runs["written"]["things"]["bystander"] = _thing(4.0)
        result = physics.verdict(CLEAN, runs)
        self.assertIs(_words(result)["grip"], True)
        self.assertNotIn("bystander", result["report"])

    def test_an_unread_thing_nudged_past_the_flight_floor_or_by_a_part_is_named(self) -> None:
        runs = _clean_runs()
        runs["written"]["contacts_unread"] = {"bin": "stepped by SolverVBD", "box": "stepped by SolverVBD"}
        runs["written"]["things"]["bin"] = _thing(12.0, hand=((0.0, 0.0), (0.0, 0.0)))  # under the 20 mm "moved" floor
        runs["written"]["things"]["box"] = _thing(0.0, hand=((0.0, 0.0), (0.0, 0.0)))
        runs["written"]["things"]["box#1"] = _thing(60.0, hand=((0.0, 0.0), (0.0, 0.0)))  # its lid swung
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["grip"])
        self.assertIn("bin (stepped by SolverVBD)", result["report"])
        self.assertIn("box (stepped by SolverVBD)", result["report"])

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
            equality_constraint_type=arr([1, 2]),
            equality_constraint_label=["hand/grab", "mimic"],
        )
        sim = types.SimpleNamespace(model=types.SimpleNamespace(mujoco=table))
        self.assertEqual(physics.equality_rows(sim), {0: (4, 9, "hand/grab")})

    def test_a_weld_left_at_the_default_object_type_is_a_body_weld(self) -> None:
        """Newton's MuJoCo solver reads an object type of -1 (the default) as a body."""
        import numpy as np

        arr = lambda v: types.SimpleNamespace(numpy=lambda: np.asarray(v))  # noqa: E731
        table = types.SimpleNamespace(
            equality_constraint_count=1,
            equality_constraint_body1=arr([4]),
            equality_constraint_body2=arr([9]),
            equality_constraint_objtype=arr([-1]),
            equality_constraint_type=arr([1]),
            equality_constraint_label=["glue"],
        )
        sim = types.SimpleNamespace(model=types.SimpleNamespace(mujoco=table))
        self.assertEqual(physics.equality_rows(sim), {0: (4, 9, "glue")})

    def test_a_step_hook_the_routine_added_is_a_note(self) -> None:
        runs = _clean_runs()
        runs["written"]["step_hooks_added"] = 1
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["contact"])
        self.assertIn("say in the report what each does", result["report"])

    def test_a_massless_link_on_a_hinge_is_moved_by_its_joint(self) -> None:
        import numpy as np

        kinds = physics._joint_kinds()
        arr = lambda v: types.SimpleNamespace(numpy=lambda: np.asarray(v))  # noqa: E731
        model = types.SimpleNamespace(
            body_count=1, body_inv_mass=arr([0.0]), joint_type=arr([kinds["revolute"]]), joint_child=arr([0])
        )
        self.assertEqual(physics._fixed_bodies(model).tolist(), [False])

    def test_yaml_is_not_a_yam(self) -> None:
        self.assertIsNone(physics.real_arm_torque(_sim("arm_from_yaml", [])))
        self.assertEqual(physics.real_arm_torque(_sim("dual_ur5e", [])), physics.ARM_TORQUE["ur5"])

    def test_a_kinematic_thing_on_a_slide_is_still_kinematic(self) -> None:
        import numpy as np

        kinds = physics._joint_kinds()
        arr = lambda v: types.SimpleNamespace(numpy=lambda: np.asarray(v))  # noqa: E731
        model = types.SimpleNamespace(
            body_count=2,
            body_inv_mass=arr([1.0, 0.0]),
            joint_type=arr([kinds["prismatic"], kinds["fixed"]]),
            joint_child=arr([0, 1]),
            body_flags=arr([int(__import__("newton").BodyFlags.KINEMATIC), 0]),
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

    def test_a_replay_whose_log_failed_is_not_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            live = _server(Path(tmp))
            runs = _clean_runs()
            runs["frozen"]["log_error"] = "Traceback ... ValueError"
            live.physics, live.physics_scene, live.physics_model = runs, "abc", dict(CLEAN)
            self.assertIn("did not run to a reading", live.cmd_physics(op="report")["error"])

    def test_a_refusal_shows_the_log_s_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            live = _server(Path(tmp))
            runs = _clean_runs()
            runs["written"]["log_error"] = "Traceback (most recent call last):\n  ...\nKeyError: 'geoms'"
            live.physics, live.physics_scene, live.physics_model = runs, "abc", dict(CLEAN)
            self.assertIn("its log failed: KeyError: 'geoms'", live.cmd_physics(mode="frozen")["error"])

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

    def test_a_plain_that_does_not_end_unwatched_goes_on_to_the_report(self) -> None:
        asked, resets = [], [0]

        def page(sock, req, timeout):
            asked.append((req["cmd"], req.get("op", ""), req.get("mode", "")))
            if req["cmd"] == "reset":
                resets[0] += 1
            if req["cmd"] == "physics" and req.get("op") == "report":
                return {"ok": True, "passed": False, "report": "FAIL  plain  ..."}
            return {"ok": True, "playing": 0, "paused": True, "resets": resets[0], "serial": 1}

        def wait(sock, started, quiet, skip, limit):
            mode = asked[-1][2]
            return {"ok": mode != "plain", "timed_out": mode == "plain", "error": "still playing"}

        with (
            mock.patch.object(ctl, "request", side_effect=page),
            mock.patch.object(ctl, "wait_for_script", side_effect=wait),
            mock.patch.object(ctl, "poll", return_value={"ok": True}),
            mock.patch.object(ctl, "wait_until_free", return_value={"ok": True}),
            mock.patch("sys.stderr"),
            mock.patch("sys.stdout"),
        ):
            ctl.main(["physics", "--only", "plain"])
        self.assertIn(("physics", "report", ""), asked)

    def test_an_unknown_replay_is_refused_before_the_page_is_asked(self) -> None:
        with mock.patch.object(ctl, "request", side_effect=AssertionError("asked the page")):
            with mock.patch("sys.stderr"):
                self.assertEqual(ctl.main(["physics", "--only", "sideways"]), 2)


class UnreadThings(unittest.TestCase):
    """Which things' contacts MuJoCo's list cannot show: those another island steps, or all with no probe."""

    def _sim(self, *, probe=True, coupled=True, owners=None):
        mujoco, vbd = object(), object()
        owners = owners if owners is not None else {"box": vbd, "lid": mujoco}

        def solver_for(name):
            if name not in owners:
                raise KeyError(name)
            return owners[name]

        program = types.SimpleNamespace(coupled=coupled, solver_for=solver_for)
        return types.SimpleNamespace(
            contacts=types.SimpleNamespace(solver=mujoco) if probe else None,
            pipeline=types.SimpleNamespace(program=program),
        )

    def test_a_thing_in_another_island_is_unread(self) -> None:
        self.assertEqual(physics.unread_things(self._sim(), ["box", "lid"]), {"box": "stepped by object"})

    def test_a_single_island_scene_reads_everything(self) -> None:
        self.assertEqual(physics.unread_things(self._sim(coupled=False), ["box", "lid"]), {})

    def test_no_probe_reads_nothing(self) -> None:
        self.assertEqual(
            physics.unread_things(self._sim(probe=False), ["box"]), {"box": "the scene has no MuJoCo contact list"}
        )

    def test_a_thing_whose_solver_cannot_be_placed_is_unread_not_read(self) -> None:
        self.assertEqual(physics.unread_things(self._sim(), ["ghost"]), {"ghost": "its solver could not be placed"})
        sim = self._sim()
        sim.pipeline.program.solver_for = mock.Mock(side_effect=AttributeError("no solver accessor"))
        self.assertEqual(physics.unread_things(sim, ["box"]), {"box": "its solver could not be placed"})


class StudyLines(unittest.TestCase):
    """The lines the physics study added: model, settle, a lasting flight, drop, plain, and drift that is elasticity."""

    def test_a_gravity_off_the_earths_fails_and_a_featherweight_is_a_note(self) -> None:
        model = {**CLEAN, "gravity": 1.962, "things": {"crate": {"mass_kg": 0.002, "volume_l": 0.096, "density": 20.8}}}
        words = _words(physics.verdict(model, _clean_runs()))
        self.assertIs(words["model"], False)
        model["gravity"] = 9.81
        result = physics.verdict(model, _clean_runs())
        self.assertIsNone(_words(result)["model"])
        self.assertIn("lighter than expanded polystyrene", result["report"])

    def test_an_invisible_collider_touching_a_handled_thing_is_a_note(self) -> None:
        runs = _clean_runs()
        runs["written"]["things"]["crate"]["hidden_touch"] = ["shelf"]
        result = physics.verdict({**CLEAN, "gravity": 9.81}, runs)
        self.assertIsNone(_words(result)["model"])
        self.assertIn("shelf: a collider nothing draws touched a handled thing", result["report"])

    def test_a_thing_the_settle_pushed_up_was_put_inside_something(self) -> None:
        model = {**CLEAN, "settle": {"crate": {"rise_mm": 12.6, "aside_mm": 0.0, "turn_deg": 0.0}}}
        self.assertIsNone(_words(physics.verdict(model, _clean_runs()))["settle"])  # higher: pushed out, or re-posed
        model["settle"]["crate"]["overlap_mm"] = 12.6
        self.assertIs(_words(physics.verdict(model, _clean_runs()))["settle"], False)
        model["settle"] = {"crate": {"rise_mm": -4.0, "aside_mm": 0.1, "turn_deg": 0.0}}  # it fell into place
        self.assertIs(_words(physics.verdict(model, _clean_runs()))["settle"], True)

    def test_a_flight_that_lasts_fails(self) -> None:
        runs = _clean_runs()
        crate = runs["written"]["things"]["crate"]
        crate.update(free_flight_not_falling=physics.FLIGHT_STEPS, free_flight_longest=physics.FLIGHT_STEPS)
        self.assertIs(_words(physics.verdict(CLEAN, runs))["flight"], False)
        # as many steps, but in short dropouts (a contact that comes and goes between reads): a note
        crate.update(free_flight_not_falling=3 * physics.FLIGHT_STEPS, free_flight_longest=4)
        self.assertIsNone(_words(physics.verdict(CLEAN, runs))["flight"])

    def _drop_runs(self, *, fell: bool, robot_steps: int = 0, robot_n: float = 0.0) -> dict:
        runs = _clean_runs()
        runs["written"].update(carried=["crate"], apex_step=300)
        end = (0.9, 0.5, 0.9) if fell else (0.4, 0.0, 1.2)
        runs["drop"] = _run(end=end)
        runs["drop"].update(held_rows=1, release_rows=200, release_skip=10)
        runs["drop"]["things"]["crate"].update(robot_after_release_steps=robot_steps, robot_after_release_N=robot_n)
        return runs

    def test_drop_passes_what_falls_and_fails_what_rides_on_untouched(self) -> None:
        model = {**CLEAN, "things": {"crate": {"mass_kg": 1.0}}}
        self.assertIs(_words(physics.verdict(model, self._drop_runs(fell=True)))["drop"], True)
        self.assertIs(_words(physics.verdict(model, self._drop_runs(fell=False)))["drop"], False)

    def test_drop_notes_a_thing_the_robot_still_bears_and_fails_one_it_only_brushes(self) -> None:
        model = {**CLEAN, "things": {"crate": {"mass_kg": 1.0}}}
        squeezed = physics.verdict(model, self._drop_runs(fell=False, robot_steps=40, robot_n=30.0))
        self.assertIsNone(_words(squeezed)["drop"])
        brushed = physics.verdict(model, self._drop_runs(fell=False, robot_steps=40, robot_n=0.4))
        self.assertIs(_words(brushed)["drop"], False)

    def test_drop_passes_when_nothing_was_carried(self) -> None:
        runs = self._drop_runs(fell=True)
        runs["written"].pop("carried")
        self.assertIs(_words(physics.verdict(CLEAN, runs))["drop"], True)

    def test_drop_reads_a_fall_off_the_heights_after_the_release(self) -> None:
        # dropped right over where the written run set it down: it ends there too, but it fell while the written
        # run's hand still held it up
        model = {**CLEAN, "things": {"crate": {"mass_kg": 1.0}}}
        runs = self._drop_runs(fell=False)
        runs["written"]["z_after"] = {"crate": [1.2] * 40}
        runs["drop"]["z_after"] = {"crate": [1.2] * 5 + [1.0] * 35}
        self.assertIs(_words(physics.verdict(model, runs))["drop"], True)

    def test_frozen_reads_a_small_straight_sink_as_settling(self) -> None:
        runs = _clean_runs()
        runs["frozen"]["things"]["crate"].update(
            moved_max_mm=3.4, start=[0.4, 0.0, 0.9], end=[0.4, 0.0, 0.8966], rise_max_mm=0.0
        )
        self.assertIsNone(_words(physics.verdict(CLEAN, runs))["frozen"])
        runs["frozen"]["things"]["crate"]["end"] = [0.4034, 0.0, 0.9]  # the same 3.4 mm, aside: a wobble to explain
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["frozen"])
        self.assertIn("a few millimetres", result["report"])
        runs["frozen"]["things"]["crate"].update(moved_max_mm=6.0, end=[0.406, 0.0, 0.9])  # farther aside: pushed
        self.assertIs(_words(physics.verdict(CLEAN, runs))["frozen"], False)

    def test_plain_notes_an_end_that_differs_and_fails_a_flight_unwatched(self) -> None:
        runs = _clean_runs()
        runs["written"]["things"]["crate"]["end_quat"] = [0.0, 0.0, 0.0, 1.0]
        runs["plain"] = {"outcome": "finished", "things": {"crate": _thing(300.0, end=(0.4, 0.0, 1.2))}}
        runs["plain"]["things"]["crate"]["end_quat"] = [0.0, 0.0, 0.0, 1.0]
        self.assertIs(_words(physics.verdict(CLEAN, runs))["plain"], True)
        runs["plain"]["things"]["crate"]["end"] = [0.6, 0.0, 1.2]  # the sim is not deterministic: a note
        self.assertIsNone(_words(physics.verdict(CLEAN, runs))["plain"])
        runs["plain"]["things"]["crate"].update(end=[0.4, 0.0, 1.2], free_flight_longest=physics.FLIGHT_STEPS)
        self.assertIs(_words(physics.verdict(CLEAN, runs))["plain"], False)  # what nondeterminism cannot do

    def test_an_elastic_part_or_one_another_solver_steps_drifts_without_failing_frozen(self) -> None:
        runs = _clean_runs()
        runs["frozen"]["things"]["stem#11"] = _thing(6.2)
        self.assertIs(_words(physics.verdict(CLEAN, runs))["frozen"], False)
        self.assertIsNone(_words(physics.verdict({**CLEAN, "elastic_parts": ["stem#11"]}, runs))["frozen"])
        runs["written"]["contacts_unread"] = {"stem": "stepped by SolverVBD"}
        self.assertIs(_words(physics.verdict(CLEAN, runs))["frozen"], False)  # 6.2 mm is more than a sag
        runs["frozen"]["things"]["stem#11"] = _thing(3.4)
        self.assertIsNone(_words(physics.verdict(CLEAN, runs))["frozen"])

    def test_a_round_things_spin_about_its_axis_is_not_a_different_end(self) -> None:
        runs = _clean_runs()
        spun = [0.0, 0.0, float(np.sin(np.radians(69))), float(np.cos(np.radians(69)))]  # 138 degrees about z
        runs["written"]["things"]["crate"]["end_quat"] = [0.0, 0.0, 0.0, 1.0]
        runs["built"]["things"]["crate"].update(turn_end_deg=138.0, end_quat=spun)
        self.assertIs(_words(physics.verdict(CLEAN, runs))["built"], False)
        model = {**CLEAN, "things": {"crate": {"round_axis": [0.0, 0.0, 1.0]}}}
        self.assertIs(_words(physics.verdict(model, runs))["built"], True)
        model["things"]["crate"]["round_axis"] = [1.0, 0.0, 0.0]  # round about x: a turn about z tilts that axis
        self.assertIs(_words(physics.verdict(model, runs))["built"], False)

    def test_a_pose_that_stops_being_a_number_fails(self) -> None:
        runs = _clean_runs()
        runs["written"]["nonfinite"] = {"crate": 212}
        result = physics.verdict(CLEAN, runs)
        self.assertIs(_words(result)["finite"], False)
        self.assertIn("stopped being a number at step 212", result["report"])

    def test_plain_that_did_not_finish_unwatched_is_a_note_and_a_round_spin_passes(self) -> None:
        runs = _clean_runs()
        runs["written"]["things"]["crate"]["end_quat"] = [0.0, 0.0, 0.0, 1.0]
        runs["plain"] = {"outcome": physics.PLAIN_LONG, "things": {}}
        self.assertIsNone(_words(physics.verdict(CLEAN, runs))["plain"])
        spun = [0.0, 0.0, float(np.sin(np.radians(69))), float(np.cos(np.radians(69)))]
        runs["plain"] = {
            "outcome": "finished",
            "things": {"crate": {**_thing(300.0, end=(0.4, 0.0, 1.2)), "end_quat": spun}},
        }
        model = {**CLEAN, "things": {"crate": {"round_axis": [0.0, 0.0, 1.0]}}}
        self.assertIs(_words(physics.verdict(model, runs))["plain"], True)

    def test_drop_is_not_fooled_by_a_bystander_or_a_late_brush(self) -> None:
        model = {**CLEAN, "things": {"crate": {"mass_kg": 1.0}}}
        runs = self._drop_runs(fell=False)
        runs["written"]["things"]["bystander"] = _thing(0.0, hand=((0.0, 0.0), (0.0, 0.0)))
        runs["drop"]["things"]["bystander"] = _thing(120.0, hand=((0.0, 0.0), (0.0, 0.0)))  # knocked away here
        self.assertIs(_words(physics.verdict(model, runs))["drop"], False)
        brushed = self._drop_runs(fell=False, robot_steps=3, robot_n=40.0)  # touched in 3 of the moments after
        self.assertIs(_words(physics.verdict(model, brushed))["drop"], False)

    def test_a_thing_the_settle_slid_aside_is_a_note(self) -> None:
        model = {**CLEAN, "settle": {"ball": {"rise_mm": 0.0, "aside_mm": 4.0, "turn_deg": 20.0}}}
        self.assertIsNone(_words(physics.verdict(model, _clean_runs()))["settle"])
        model["settle"]["ball"]["overlap_mm"] = 6.0  # deeper than any soft contact sinks
        self.assertIs(_words(physics.verdict(model, _clean_runs()))["settle"], False)

    def test_a_shallow_rest_the_frozen_run_leaves_is_a_note_and_one_it_pushes_out_fails(self) -> None:
        model = {**CLEAN, "settle": {"crate": {"rise_mm": 0.0, "aside_mm": 0.0, "turn_deg": 0.0, "overlap_mm": 3.0}}}
        self.assertIsNone(_words(physics.verdict(model, _clean_runs()))["settle"])
        runs = _clean_runs()
        runs["frozen"]["things"]["crate"]["moved_max_mm"] = 3.1  # the frozen run pushed it out: a start inside
        self.assertIs(_words(physics.verdict(model, runs))["settle"], False)
        self.assertIsNone(_words(physics.verdict(model, runs))["frozen"])  # the move alone is a wobble

    def test_a_squeeze_no_real_hand_gives_fails_grip(self) -> None:
        runs = _clean_runs()
        runs["written"]["things"]["crate"]["hand_force_p50_p95_N"] = [[900.0, 950.0], [0.0, 0.0]]
        result = physics.verdict({**CLEAN, "hands": ["robotiq_2f85", "robotiq_2f85"]}, runs)
        self.assertIs(_words(result)["grip"], False)
        self.assertIn("1.9x the most a Robotiq 2F-85 squeezes (235 N a jaw)", result["report"])
        # an impact spike sets the p95, not the median while touching (crate-pr: 23 N median, 200 N p95)
        runs["written"]["things"]["crate"]["hand_force_p50_p95_N"] = [[23.0, 200.0], [0.0, 0.0]]
        self.assertIsNot(_words(physics.verdict({**CLEAN, "hands": ["yam", "yam"]}, runs))["grip"], False)
        # a YAM squeezing its real 50 N a jaw reads about 100 N over its two fingers: a real squeeze
        runs["written"]["things"]["crate"]["hand_force_p50_p95_N"] = [[98.0, 104.0], [0.0, 0.0]]
        self.assertIsNot(_words(physics.verdict({**CLEAN, "hands": ["yam", "yam"]}, runs))["grip"], False)
        runs["written"]["things"]["crate"]["hand_force_p50_p95_N"] = [[200.0, 229.0], [0.0, 0.0]]
        self.assertIsNot(_words(physics.verdict({**CLEAN, "hands": ["robotiq_2f85"] * 2}, runs))["grip"], False)

    def test_which_real_hand(self) -> None:
        def sim(labels, name="pair"):
            return types.SimpleNamespace(
                scene=types.SimpleNamespace(embodiment=types.SimpleNamespace(spec=types.SimpleNamespace(name=name))),
                model=types.SimpleNamespace(joint_label=labels),
            )

        arm = types.SimpleNamespace()
        with mock.patch.object(physics, "_robot_joint_labels", side_effect=lambda s: list(s.model.joint_label)):
            robotiq = sim(["l_hand_right_driver_joint", "l_hand_right_follower_joint"])
            self.assertEqual(physics.real_hand(robotiq, arm), "robotiq_2f85")
            self.assertEqual(physics.real_hand(sim([], name="yam_bimanual"), arm), "yam")
            self.assertIsNone(physics.real_hand(sim(["j1"], name="arm"), arm))

    def test_a_ball_ends_the_same_however_it_turned(self) -> None:
        runs = _clean_runs()
        runs["written"]["things"]["crate"]["end_quat"] = [0.0, 0.0, 0.0, 1.0]
        runs["built"]["things"]["crate"].update(turn_end_deg=90.0, end_quat=[0.7071068, 0.0, 0.0, 0.7071068])
        self.assertIs(
            _words(physics.verdict({**CLEAN, "things": {"crate": {"round_axis": [0.0, 0.0, 0.0]}}}, runs))["built"],
            True,
        )

    def test_settle_honours_the_bar_it_is_given(self) -> None:
        model = {**CLEAN, "settle": {"crate": {"rise_mm": 3.0, "aside_mm": 0.0, "turn_deg": 0.0}}}
        self.assertIsNone(_words(physics.verdict(model, _clean_runs()))["settle"])
        self.assertIs(_words(physics.verdict(model, _clean_runs(), settle_mm=5.0))["settle"], True)

    def test_a_blow_up_outside_the_written_run_is_named_and_not_read(self) -> None:
        runs = _clean_runs()
        runs["heavy"]["nonfinite"] = {"crate": 400}
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["finite"])
        self.assertIsNone(_words(result)["heavy"])
        self.assertIn("the solver blew up in heavy (crate at step 400)", result["report"])

    def test_a_flight_not_judged_says_so(self) -> None:
        runs = _clean_runs()
        runs["written"]["things"]["crate"]["flight_unread"] = "the replay's collision pass overflowed its buffer"
        result = physics.verdict(CLEAN, runs)
        self.assertIsNone(_words(result)["flight"])
        self.assertIn("not judged for crate", result["report"])

    def test_a_release_too_near_the_end_is_not_read(self) -> None:
        runs = self._drop_runs(fell=False)
        runs["drop"]["release_rows"] = 12
        self.assertIsNone(_words(physics.verdict({**CLEAN, "things": {"crate": {"mass_kg": 1.0}}}, runs))["drop"])

    def test_density_is_bounded_both_ways_against_overlapping_colliders(self) -> None:
        # two overlapping colliders halve a summed density: light only if the largest alone is light too
        model = {
            **CLEAN,
            "gravity": 9.81,
            "things": {"crate": {"mass_kg": 0.05, "volume_l": 2.0, "density": 25.0, "density_max": 50.0}},
        }
        self.assertIs(_words(physics.verdict(model, _clean_runs()))["model"], True)
        model["things"]["crate"]["density_max"] = 26.0
        self.assertIsNone(_words(physics.verdict(model, _clean_runs()))["model"])

    def test_a_shallow_rest_without_a_frozen_run_is_a_note(self) -> None:
        model = {**CLEAN, "settle": {"crate": {"rise_mm": 0.0, "aside_mm": 0.0, "turn_deg": 0.0, "overlap_mm": 2.0}}}
        runs = _clean_runs()
        runs.pop("frozen")
        result = physics.verdict(model, runs)
        self.assertIsNone(_words(result)["settle"])
        self.assertIn("play frozen to see whether it is pushed out", result["report"])

    def test_drop_names_a_carry_it_did_not_release(self) -> None:
        model = {**CLEAN, "things": {"crate": {"mass_kg": 1.0}}}
        runs = self._drop_runs(fell=True)
        runs["written"]["carried_later"] = ["mug"]
        result = physics.verdict(model, runs)
        self.assertIsNone(_words(result)["drop"])
        self.assertIn("mug, carried later, was not released", result["report"])

    def test_a_flight_not_judged_is_said_beside_one_that_flew(self) -> None:
        runs = _clean_runs()
        runs["written"]["things"]["crate"].update(free_flight_not_falling=4, free_flight_longest=4)
        runs["written"]["things"]["box"] = {**_thing(0.0), "flight_unread": "the replay's collision pass failed"}
        result = physics.verdict(CLEAN, runs)
        self.assertIn("not judged for box", result["report"])

    def test_drop_fails_a_read_thing_riding_on_beside_an_unread_one(self) -> None:
        model = {**CLEAN, "things": {"crate": {"mass_kg": 1.0}, "pot": {"mass_kg": 1.0}}}
        runs = self._drop_runs(fell=False)
        runs["written"]["carried"] = ["crate", "pot"]
        runs["written"]["things"]["pot"] = _thing(300.0, end=(0.4, 0.0, 1.2))
        runs["drop"]["things"]["pot"] = {**_thing(300.0, end=(0.4, 0.0, 1.2)), "robot_touch_read": False}
        result = physics.verdict(model, runs)
        self.assertIs(_words(result)["drop"], False)
        self.assertIn("pot also stayed", result["report"])

    def test_a_plain_cut_on_a_slow_page_is_not_read(self) -> None:
        runs = _clean_runs()
        runs["plain"] = {"outcome": physics.PLAIN_SLOW, "things": {}}
        self.assertIsNone(_words(physics.verdict(CLEAN, runs))["plain"])

    def test_a_bendy_things_base_drifts_with_its_stem(self) -> None:
        runs = _clean_runs()
        runs["frozen"]["things"]["stem"] = _thing(7.0)
        self.assertIs(_words(physics.verdict(CLEAN, runs))["frozen"], False)
        stem = {**CLEAN, "elastic_parts": ["stem#1", "stem#2", "stem#3"]}  # a chain of bending joints
        self.assertIsNone(_words(physics.verdict(stem, runs))["frozen"])
        door = {**CLEAN, "elastic_parts": ["stem#1"]}  # one sprung joint (a door) is not a stem
        self.assertIs(_words(physics.verdict(door, runs))["frozen"], False)

    def test_the_robot_line_holds_the_robot_to_the_registered_one(self) -> None:
        runs = _clean_runs()
        same = {
            "name": "yam_bimanual",
            "registered": True,
            "file": "",
            "arm": "",
            "gripper": "",
            "hand": "",
            "other": "",
        }
        self.assertIs(_words(physics.verdict({**CLEAN, "robot_spec": same}, runs))["robot"], True)
        copied = {**same, "file": "its file is twin/assets/yam/arms.xml, not the registered embodiments/yam/..."}
        result = physics.verdict({**CLEAN, "robot_spec": copied}, runs)
        self.assertIs(_words(result)["robot"], False)
        self.assertFalse(result["passed"])
        servo = {**same, "gripper": "its gripper servo is the twin's own (gains 2000/40, 25 limit; registered: ...)"}
        self.assertIn(
            "gripper servo is the twin's own", physics.verdict({**CLEAN, "robot_spec": servo}, runs)["report"]
        )
        inline = {"name": "ur5e_gasket", "registered": False, "file": "assets/ur5e.xml"}
        self.assertIsNone(_words(physics.verdict({**CLEAN, "robot_spec": inline}, runs))["robot"])
        self.assertNotIn("robot", _words(physics.verdict(CLEAN, runs)), "no robot facts read, no line")

    def test_a_robot_the_twin_registered_from_its_own_folder_is_its_own_file(self) -> None:
        from dataclasses import replace

        import robosimstudio as rss
        from robosimstudio.robots.registry import EMBODIMENTS, get_embodiment, register_embodiment

        EMBODIMENTS.load()
        yam = get_embodiment("yam_bimanual")
        here = Path(__file__).resolve().parent / "twin_robot_probe" / "arms.xml"
        own = replace(yam, name="study_twin_registered_yam", asset=rss.AssetSpec(path=str(here)))
        register_embodiment(own)
        self.addCleanup(EMBODIMENTS.forget, "study_twin_registered_yam")

        def sim(spec):
            return types.SimpleNamespace(scene=types.SimpleNamespace(embodiment=types.SimpleNamespace(spec=spec)))

        self.assertEqual(
            physics.robot_departures(sim(yam)),
            {
                "name": "yam_bimanual",
                "registered": True,
                "as": "yam_bimanual",
                "file": "",
                "arm": "",
                "gripper": "",
                "hand": "",
                "other": "",
            },
        )
        renamed = replace(
            yam,
            name="yam_of_the_twin",
            gripper_actuation=rss.ActuationSpec(target_ke=2000, target_kd=40, effort_limit=500),
        )
        read = physics.robot_departures(sim(renamed))
        self.assertEqual((read["registered"], read["as"]), (True, "yam_bimanual"), "a renamed YAM is still the YAM")
        self.assertIn("gripper servo is the twin's own", read["gripper"])
        swapped = physics.robot_departures(sim(replace(yam, enable_self_collisions=True, gripper="franka_hand")))
        self.assertIn("enable_self_collisions", swapped["other"])
        self.assertEqual(swapped["hand"], "", "each YAM arm names its own hand, so the robot-wide one is not used")
        same_file = replace(yam, asset=rss.AssetSpec(path=str(physics._robot_file(yam))))
        self.assertEqual(physics.robot_departures(sim(same_file))["file"], "", "the same file, however it is named")
        hand = linear_4310_clamp()
        if hand is not None:
            clamp = physics.robot_departures(sim(replace(yam, gripper_actuation=hand)))
            self.assertEqual(clamp["gripper"], "", "gripper= the hand's own servo is no departure")
        read = physics.robot_departures(sim(own))
        self.assertIs(read["registered"], False, "registered by the twin, from a file the library does not ship")
        self.assertEqual(read["file"], str(here))

    def test_drawn_parts_that_do_not_collide_fail_past_a_centimetre(self) -> None:
        runs = _clean_runs()
        result = physics.verdict({**CLEAN, "gravity": 9.81, "drawn_only_mm": {"flower_1": 62.0}}, runs)
        self.assertIs(_words(result)["model"], False)
        self.assertIn("flower_1: drawn parts that do not collide reach 62 mm outside its colliders", result["report"])
        self.assertIsNone(
            _words(physics.verdict({**CLEAN, "gravity": 9.81, "drawn_only_mm": {"cap": 5.0}}, runs))["model"]
        )

    def test_drawn_only_reach_measures_past_the_colliders(self) -> None:
        import newton

        arr = lambda v: types.SimpleNamespace(numpy=lambda: np.asarray(v))  # noqa: E731
        box = int(newton.GeoType.BOX)
        vis, col = int(newton.ShapeFlags.VISIBLE), int(newton.ShapeFlags.COLLIDE_SHAPES)
        # a stem collider 1 x 1 x 5 cm; a drawn-only leaf 4 cm long reaching 3 cm past it in x; a drawn-only label inside
        model = types.SimpleNamespace(
            shape_flags=arr([col, vis, vis]),
            shape_type=arr([box, box, box]),
            shape_scale=arr([[0.005, 0.005, 0.025], [0.02, 0.002, 0.005], [0.004, 0.004, 0.01]]),
            shape_transform=arr([[0, 0, 0, 0, 0, 0, 1], [0.015, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]]),
            shape_body=arr([0, 0, 0]),
            shape_source=[None, None, None],
            body_count=1,
            joint_count=0,
        )
        flower = types.SimpleNamespace(bodies=[0], body=0, shapes=[0, 1, 2], static=False)
        sim = types.SimpleNamespace(
            model=model, scene=types.SimpleNamespace(handles=types.SimpleNamespace(objects={"flower": flower}))
        )
        self.assertAlmostEqual(physics.drawn_only_reach(sim)["flower"], 30.0, places=1)
        flower.shapes = [0, 2]
        self.assertEqual(physics.drawn_only_reach(sim), {}, "a drawn part inside its collider")
        flower.shapes = [0, 1, 2]
        model.joint_count = 1  # the stem welded to the world: a vase on the bench, scenery
        model.joint_type = arr([int(newton.JointType.FIXED)])
        model.joint_parent, model.joint_child = arr([-1]), arr([0])
        self.assertEqual(physics.drawn_only_reach(sim), {}, "scenery welded to the world is left out")

    def test_shape_volumes(self) -> None:
        import newton

        g = newton.GeoType
        arr = lambda v: types.SimpleNamespace(numpy=lambda: np.asarray(v))  # noqa: E731
        m = types.SimpleNamespace(
            shape_type=arr([int(g.BOX), int(g.SPHERE)]), shape_scale=arr([[0.1, 0.2, 0.3], [0.5, 0, 0]])
        )
        self.assertAlmostEqual(physics._shape_volume(m, 0), 0.048)
        self.assertAlmostEqual(physics._shape_volume(m, 1), 4.0 / 3.0 * np.pi * 0.125)


if __name__ == "__main__":
    unittest.main()
