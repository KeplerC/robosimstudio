# SPDX-License-Identifier: Apache-2.0
"""A routine, step by step: where ``reach`` puts the cursor and when, and what the Run panel does with it.

``reach`` glides an arm's gizmo to a pose at a hand's pace, eased at both ends,
then lets the drag go and waits for the hand; ``grip`` presses the Grip
control; ``wait`` lets the page step. :class:`PanelRoutine` starts, advances
and stops one generator a step, holds the arm panels off for a routine that
drives the scene itself, and gives them back however it ends. None of that is
physics: the arms here are :class:`~tests.fixtures.studio_fakes.FakeArmPanel`,
whose hand is wherever the cursor was put (or, for one that cannot arrive,
stays where it was). That the hello-world pick lifts its cube through all of it
is :mod:`tests.test_studio_routine`'s, on a real page.
"""

from __future__ import annotations

import contextlib
import io
import math
import unittest

import numpy as np

from robosimstudio_studio.routine import PanelRoutine, RoutineError, _slerp, grip, hand, reach, wait
from tests.fixtures.studio_fakes import FakeArmPanel, FakeEnv, FakeSession

TIER = "fast"
"""No GPU and no scene: fake arm panels whose hand is where the test says."""

HZ = 50.0
DT = 1.0 / HZ
IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _session(**arm) -> FakeSession:
    """A page at 50 Hz with one arm, ``right``, at (0.3, 0, 0.2) and unrotated."""
    return FakeSession(FakeEnv(control_hz=HZ), arms={"right": FakeArmPanel("right", **arm)})


def _about_z(angle: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)])


def _quiet() -> contextlib.ExitStack:
    """The panel prints where a routine ended and a traceback when one raises; the tests read its status instead."""
    stack = contextlib.ExitStack()
    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
    return stack


class TestReach(unittest.TestCase):
    def test_it_ends_with_the_drag_let_go_exactly_at_the_target(self) -> None:
        session = _session()
        target = np.array([0.4, 0.1, 0.25])
        list(reach(session, "right", target))
        aims = session.arms["right"].aims
        np.testing.assert_allclose(aims[-1][0], target)
        self.assertFalse(aims[-1][2], "the last aim ends the drag")
        self.assertTrue(all(moving for _, _, moving in aims[:-1]), "every aim before it is mid-drag")

    def test_the_glide_takes_distance_over_speed_in_control_steps(self) -> None:
        session = _session()
        list(reach(session, "right", (0.3 + 0.31, 0.0, 0.2), speed=0.15))  # 310 mm at 150 mm/s: 2.07 s
        self.assertEqual(104, len(session.arms["right"].aims) - 1, "103.3 control steps, rounded up")

    def test_a_turn_slower_than_the_glide_sets_the_pace(self) -> None:
        session = _session()
        list(reach(session, "right", (0.3, 0.0, 0.2), _about_z(math.pi / 2), turn=0.5))  # pi/2 at 0.5 rad/s: 3.14 s
        self.assertEqual(158, len(session.arms["right"].aims) - 1, "157.1 control steps, rounded up")

    def test_a_reach_to_where_the_hand_is_still_takes_a_step(self) -> None:
        session = _session()
        steps = len(list(reach(session, "right", (0.3, 0.0, 0.2))))
        self.assertEqual(1, steps, "one gliding step, and the hand is already there")
        self.assertEqual(2, len(session.arms["right"].aims))

    def test_the_glide_is_eased_at_both_ends_and_halfway_at_half_time(self) -> None:
        session = _session()
        list(reach(session, "right", (0.5, 0.0, 0.2), speed=0.2))  # 200 mm in 50 steps
        xs = np.array([aim[0][0] for aim in session.arms["right"].aims[:-1]]) - 0.3
        gaps = np.diff(np.concatenate([[0.0], xs]))
        self.assertTrue(np.all(gaps > 0.0), "it only ever moves toward the target")
        self.assertLess(gaps[0], gaps[len(gaps) // 2] / 10.0, "it starts slowly")
        self.assertLess(gaps[-1], gaps[len(gaps) // 2] / 10.0, "and arrives slowly")
        self.assertAlmostEqual(0.1, xs[len(xs) // 2 - 1], places=9)

    def test_no_orientation_given_keeps_the_cursors_own(self) -> None:
        session = _session(quat=_about_z(0.3))
        list(reach(session, "right", (0.35, 0.0, 0.2)))
        for _, quat, _ in session.arms["right"].aims:
            np.testing.assert_allclose(quat, _about_z(0.3), atol=1e-12)

    def test_an_orientation_given_is_reached_and_normalised(self) -> None:
        session = _session()
        list(reach(session, "right", (0.3, 0.0, 0.2), 3.0 * _about_z(0.8)))
        np.testing.assert_allclose(session.arms["right"].aims[-1][1], _about_z(0.8), atol=1e-12)

    def test_a_hand_that_cannot_arrive_raises_saying_how_far_short(self) -> None:
        session = _session(tracks=False)  # the hand stays at x = 0.3
        steps = 0
        with self.assertRaisesRegex(RoutineError, "the right hand stopped 105 mm short of where it was sent"):
            for _ in reach(session, "right", (0.405, 0.0, 0.2), speed=1.0, settle=0.21):
                steps += 1
        # 105 mm at 1 m/s is 5.25 steps of gliding, and 0.21 s of settling 10.5 more: each rounded up
        self.assertEqual(6 + 11, steps, "it waited out the whole settle before saying so")

    def test_a_hand_within_tolerance_has_arrived(self) -> None:
        session = _session(tracks=False)
        arm = session.arms["right"]
        arm.ik.position = np.array([0.41, 0.0, 0.2])  # 5 mm from where it is sent, inside the 6 mm
        steps = len(list(reach(session, "right", (0.405, 0.0, 0.2), speed=1.0)))
        self.assertEqual(6, steps, "the glide's 5.25 steps and no settle step: it was there at once")


class TestTheShortWayRound(unittest.TestCase):
    def test_a_quaternion_and_its_negative_are_one_orientation(self) -> None:
        q = _about_z(0.4)
        np.testing.assert_allclose(_slerp(q, -q, 0.5), q, atol=1e-12)

    def test_halfway_between_two_turns_is_the_turn_between(self) -> None:
        np.testing.assert_allclose(_slerp(_about_z(0.0), _about_z(1.2), 0.5), _about_z(0.6), atol=1e-12)

    def test_nearly_the_same_orientation_stays_unit(self) -> None:
        mid = _slerp(_about_z(0.0), _about_z(1e-4), 0.5)
        self.assertAlmostEqual(1.0, float(np.linalg.norm(mid)), places=12)


class TestGripAndWait(unittest.TestCase):
    def test_grip_asks_the_panel_and_lets_the_fingers_move(self) -> None:
        session = _session()
        self.assertEqual(math.ceil(0.6 * HZ), len(list(grip(session, "right", 1.0))))
        self.assertEqual(1.0, session.arms["right"].grip_request)

    def test_wait_is_seconds_of_control_steps_and_never_none(self) -> None:
        session = _session()
        self.assertEqual(5, len(list(wait(session, 0.1))))
        self.assertEqual(1, len(list(wait(session, 0.0))))

    def test_a_page_with_no_control_rate_paces_at_one_hertz(self) -> None:
        session = FakeSession(FakeEnv(control_hz=0.0))
        self.assertEqual(2, len(list(wait(session, 2.0))))

    def test_hand_is_the_tool_centre_now(self) -> None:
        session = _session()
        position, quat = hand(session, "right")
        np.testing.assert_allclose(position, (0.3, 0.0, 0.2))
        np.testing.assert_allclose(quat, IDENTITY)


def _panel(routines: dict, **arms) -> tuple[PanelRoutine, FakeSession]:
    session = FakeSession(FakeEnv(control_hz=HZ), arms=arms or {"right": FakeArmPanel("right")})
    panel = PanelRoutine(session, routines)
    panel.build(session.server.gui)
    return panel, session


def _steps(n: int):
    def routine(sim, session):
        for _ in range(n):
            yield

    return routine


class TestTheRunPanel(unittest.TestCase):
    def test_a_button_per_routine_and_a_stop_and_the_click_only_asks(self) -> None:
        panel, session = _panel({"Pick": _steps(3), "Place": _steps(3)})
        gui = session.server.gui
        self.assertEqual(["Pick", "Place", "Stop", "Playing"], [label for label, _ in gui.widgets])
        self.assertEqual("nothing", gui.find("Playing").value)
        gui.find("Place").fire()
        self.assertEqual("Place", panel.requested)
        self.assertIsNone(panel.gen, "nothing starts on viser's thread")

    def test_it_plays_one_step_a_drain_and_says_when_it_is_done(self) -> None:
        panel, session = _panel({"Pick": _steps(3)})
        panel.requested = "Pick"
        with _quiet():
            for expected in (1, 2, 3):
                panel.drain()
                self.assertEqual(expected, panel.steps)
            panel.drain()
        self.assertIsNone(panel.gen)
        self.assertEqual("Pick: done", session.server.gui.find("Playing").value)

    def test_a_hand_that_cannot_arrive_stops_the_routine_and_says_so(self) -> None:
        def lost(sim, session):
            yield from reach(session, "right", (5.0, 5.0, 5.0), speed=10.0, settle=0.2)

        panel, _ = _panel({"Lost": lost}, right=FakeArmPanel("right", tracks=False))
        panel.requested = "Lost"
        with _quiet():
            for _ in range(200):
                panel.drain()
        self.assertIsNone(panel.gen)
        self.assertIn("short of where it was sent", panel._status.value)

    def test_a_routine_that_drives_the_scene_itself_has_the_panels_held_off(self) -> None:
        def hands_off(sim):
            for _ in range(25):
                yield

        held = FakeArmPanel("left", suspended=True)  # a leader arm has this one already
        panel, session = _panel({"Hands off": hands_off}, left=held, right=FakeArmPanel("right"))
        panel.requested = "Hands off"
        with _quiet():
            panel.drain()
            self.assertTrue(all(p.suspended for p in session.arms.values()))
            while panel.gen is not None:
                panel.drain()
        self.assertFalse(session.arms["right"].suspended, "the hands were not given back")
        self.assertTrue(held.suspended, "and one it never took is not its to give")
        self.assertEqual([], held.calls)

    def test_a_routine_that_takes_the_page_leaves_the_panels_driving(self) -> None:
        panel, session = _panel({"Pick": _steps(3)})
        panel.requested = "Pick"
        with _quiet():
            panel.drain()
        self.assertEqual([], session.arms["right"].calls)

    def test_stop_and_reset_end_it_where_it_is(self) -> None:
        panel, _ = _panel({"Pick": _steps(100)})
        with _quiet():
            panel.requested = "Pick"
            for _ in range(5):
                panel.drain()
            panel.stop_requested = True
            panel.drain()
            self.assertIsNone(panel.gen)
            self.assertEqual("Pick: stopped", panel._status.value)
            panel.requested = "Pick"
            for _ in range(5):
                panel.drain()
            panel.on_reset({}, {})
        self.assertIsNone(panel.gen)
        self.assertEqual("Pick: reset", panel._status.value)

    def test_a_routine_that_returns_no_generator_is_refused_and_the_arms_given_back(self) -> None:
        panel, session = _panel({"Flat": lambda sim: None})
        panel.requested = "Flat"
        with _quiet():
            panel.drain()
        self.assertIsNone(panel.gen)
        self.assertEqual("did not start: Flat returned NoneType, not a generator", panel._status.value)
        self.assertEqual(["suspend", "resume"], session.arms["right"].calls)

    def test_a_routine_that_raises_as_it_starts_did_not_start(self) -> None:
        def broken(sim):
            raise ValueError("no cube")

        panel, session = _panel({"Broken": broken})
        panel.requested = "Broken"
        with _quiet():
            panel.drain()
        self.assertEqual("did not start: no cube", panel._status.value)
        self.assertFalse(session.arms["right"].suspended)

    def test_a_routine_that_raises_mid_way_is_stopped_saying_why(self) -> None:
        def breaks(sim):
            yield
            raise RuntimeError("the cube slipped")

        panel, session = _panel({"Breaks": breaks})
        panel.requested = "Breaks"
        with _quiet():
            panel.drain()
            panel.drain()
        self.assertIsNone(panel.gen)
        self.assertEqual("Breaks: stopped: the cube slipped", panel._status.value)
        self.assertEqual(["suspend", "resume"], session.arms["right"].calls)

    def test_starting_another_closes_the_one_playing(self) -> None:
        closed = []

        def first(sim, session):
            try:
                while True:
                    yield
            finally:
                closed.append("first")

        panel, _ = _panel({"First": first, "Second": _steps(3)})
        with _quiet():
            panel.requested = "First"
            panel.drain()
            panel.requested = "Second"
            panel.drain()
        self.assertEqual(["first"], closed)
        self.assertEqual("Second", panel.playing)


if __name__ == "__main__":
    unittest.main()
