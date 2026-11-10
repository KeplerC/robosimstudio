# SPDX-License-Identifier: Apache-2.0
"""``examples/08_demos/unscrew_the_cap``: two YAM arms take a cap off a bottle that nothing holds but fingers.

The demo's routine is played the way its page plays it, through the Run button
(:mod:`tests.fixtures.grasp`): the right hand takes the bottle's base, the left
turns the cap four strokes and lifts it. Nothing joins the cap to the bottle but
their threads, the bottle stands free, and the routine writes no pose.

The assertions are a thread's: the cap comes off after the two turns its ridge
is wound and not before, its height over the rim follows pitch times turns to
within the thread's own play while it is on, the fingers do not let it turn in
them, and the bottle stays where the other hand holds it.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
import unittest

import numpy as np

from tests.fixtures.examples import ROOT
from tests.fixtures.grasp import open_session, tick

TIER = "medium"
"""Builds the scene and plays the 33 s routine through a page with no browser on it: about 70 s."""

DEMO = ROOT / "examples" / "08_demos" / "unscrew_the_cap" / "unscrew_the_cap.py"


def _load():
    name = "_rss_demo_unscrew_the_cap"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, DEMO)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestTheRoutineUnscrewsTheCap(unittest.TestCase):
    """One build, one press of Run, and every reading taken along the way."""

    @classmethod
    def setUpClass(cls) -> None:
        from robosimstudio.core.math import quat_yaw

        demo = cls.demo = _load()
        sim = cls.sim = demo.build()
        lid, flask = sim["cap"], sim["bottle"]
        cls.thread = flask.entity.spec.thread
        rim = float(flask.entity.spec.rim)
        hz = float(sim.request.physics.control_hz)
        left = sim.arm("left")
        roll = left.arm_coords[-1]

        session = cls.session = open_session(sim)
        session.add_routines({"Unscrew the cap": demo.routine})
        panel = session.panels[-1]
        tick(session, 10)
        r = cls.readings = {"contacts": 0, "finite": True}
        cap, bottle = lid.pose, flask.pose
        last = [quat_yaw(cap[3:7]), quat_yaw(bottle[3:7]), float(sim.state.joint_q.numpy()[roll])]
        turns = [0.0, 0.0, 0.0]  # the cap's, the bottle's and the left wrist's, each from where it started
        rows = []
        panel.requested = "Unscrew the cap"
        tick(session, 1)
        steps = 0
        while panel.gen is not None and steps < round(60 * hz):
            tick(session, 1)
            steps += 1
            cap, bottle = lid.pose, flask.pose
            now = [quat_yaw(cap[3:7]), quat_yaw(bottle[3:7]), float(sim.state.joint_q.numpy()[roll])]
            for k in range(3):
                turns[k] += math.remainder(now[k] - last[k], 2.0 * math.pi) / (2.0 * math.pi)
            last = now
            closed = float(session.env.gripper("left").command) > 0.99
            rows.append((turns[0] - turns[1], cap[2] - bottle[2] - rim, turns[2], closed))
            if steps % 10 == 0:
                r["contacts"] = max(r["contacts"], int(sim.pipeline.contacts.rigid_contact_count.numpy()[0]))
                r["finite"] = r["finite"] and bool(np.isfinite(cap).all())
        cls.rows = np.asarray(rows, dtype=np.float64)
        r["seconds"] = steps / hz
        r["said"] = str(panel._status.value) if panel._status is not None else ""
        cls.measures = demo.measures(sim)
        print(f"\n[unscrew] {r} {cls.measures} turned {cls.rows[-1, 0]:.3f}", flush=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def _strokes(self) -> list[np.ndarray]:
        """The rows of each stroke: the left hand shut and its wrist turning."""
        shut = self.rows[:, 3] > 0.5
        wrist = np.abs(np.gradient(self.rows[:, 2])) > 1.0e-5
        turning = np.flatnonzero(shut & wrist)
        return [chunk for chunk in np.split(turning, np.flatnonzero(np.diff(turning) > 30) + 1) if len(chunk) > 60]

    def test_the_routine_plays_to_its_end(self) -> None:
        self.assertIn("done", self.readings["said"])
        self.assertTrue(self.readings["finite"] and self.sim.state_is_finite())

    def test_the_cap_comes_off_and_stays_in_the_hand(self) -> None:
        m = self.measures
        self.assertGreater(m["cap_over_rim"], 0.030, "the cap was not lifted off the bottle")
        self.assertLess(m["cap_off_axis"], 0.015, "it was carried straight up")
        self.assertAlmostEqual(2.22, float(self.rows[-1, 0]), delta=0.12, msg="four strokes of 200 degrees")

    def test_it_was_still_on_after_three_strokes(self) -> None:
        """Two turns of ridge: at 1.67 turns the cap is on the helix, a pitch for each turn up, not loose."""
        strokes = self._strokes()
        self.assertEqual(4, len(strokes), "four strokes")
        turns, height = self.rows[strokes[2][-1], :2]
        self.assertAlmostEqual(1.667, turns, delta=0.08)
        self.assertAlmostEqual(self.thread.pitch * turns, height, delta=0.5 * self.thread.axial_play + 0.15e-3)

    def test_its_height_follows_pitch_times_turns_within_the_threads_play(self) -> None:
        """While the ridges are engaged: the cap rides a flank, so it is off the helix by the play at most."""
        turns, height = self.rows[:, 0], self.rows[:, 1]
        on = turns < 1.85
        off = height[on] - self.thread.pitch * turns[on]
        play = 0.5 * self.thread.axial_play
        self.assertGreater(float(off.min()), -play - 0.15e-3)
        self.assertLess(float(off.max()), play + 0.25e-3, "kicked up into the ridge above it")
        turning = np.concatenate(self._strokes()[:3])
        slope = float(np.polyfit(turns[turning], height[turning], 1)[0])
        self.assertAlmostEqual(self.thread.pitch, slope, delta=0.05 * self.thread.pitch)

    def test_the_fingers_turned_the_cap_and_it_did_not_slip_in_them(self) -> None:
        """Each stroke the wrist turns 200 degrees and the cap, against the bottle, within 12 of that."""
        for k, stroke in enumerate(self._strokes()):
            with self.subTest(stroke=k + 1):
                wrist = abs(self.rows[stroke[-1], 2] - self.rows[stroke[0], 2]) * 360.0
                cap = (self.rows[stroke[-1], 0] - self.rows[stroke[0], 0]) * 360.0
                self.assertAlmostEqual(200.0, wrist, delta=4.0)
                self.assertAlmostEqual(wrist, cap, delta=12.0, msg="the cap, or the bottle under it, slipped")

    def test_the_other_hand_held_the_bottle(self) -> None:
        m = self.measures
        self.assertLess(m["bottle_moved"], 0.006, "the bottle was dragged along the bench")
        self.assertLess(m["bottle_lean_deg"], 1.5, "or tipped")
        self.assertLess(abs(m["bottle_turned_deg"]), 20.0, "or turned round in the right hand's fingers")

    def test_the_contacts_fit_the_buffer_the_table_allows(self) -> None:
        self.assertGreater(self.readings["contacts"], 150)
        self.assertLess(self.readings["contacts"], self.sim.request.physics.collision.rigid_contact_max // 2)


class TestTheDemoHasTheShapeOfOne(unittest.TestCase):
    """The rules ``tests/test_examples.py`` holds the numbered examples to, at a demo's length."""

    MAX_LINES = 300

    def setUp(self) -> None:
        self.source = DEMO.read_text()
        self.tree = ast.parse(self.source)

    def test_it_is_a_script_with_a_main_that_ends_on_the_studio(self) -> None:
        self.assertLessEqual(len(self.source.splitlines()), self.MAX_LINES)
        self.assertNotIn("argparse", self.source)
        for withheld in ("examples.tasks", "taskbuild", "robosimstudio_bench", "solutions"):
            self.assertNotIn(withheld, self.source, "a demo stands on the public library alone")
        for hold in ("held_by", "latch(", "weld", "body_q.assign", "joint_q.assign", "static=True"):
            self.assertNotIn(hold, self.source, "nothing holds the cap or the bottle but fingers and a thread")
        main = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        self.assertEqual([], main.args.args)
        self.assertEqual("rss.view", ast.unparse(main.body[-1].value.func))
        self.assertEqual("__name__ == '__main__'", ast.unparse(self.tree.body[-1].test))

    def test_its_docstring_is_the_page_it_becomes(self) -> None:
        blocks = [block.strip() for block in (ast.get_docstring(self.tree) or "").split("\n\n")]
        self.assertTrue(2 <= len(blocks[0].split()) <= 5 and not blocks[0].endswith("."), blocks[0])
        self.assertNotIn("\n", blocks[1])
        self.assertTrue(blocks[1].endswith("."))
        self.assertIn(len(blocks), (2, 3), "at most one paragraph")

    def test_it_says_which_reads_are_privileged(self) -> None:
        """The routine reads the cap's and the bottle's poses to aim the hands, and says so where it does."""
        reads = [line for line in self.source.splitlines() if ".position" in line and "sim[" not in line]
        self.assertGreaterEqual(len(reads), 2)
        for line in reads:
            self.assertIn("a privileged read", line)


if __name__ == "__main__":
    unittest.main()
