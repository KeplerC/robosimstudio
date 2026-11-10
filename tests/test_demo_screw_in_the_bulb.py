# SPDX-License-Identifier: Apache-2.0
"""``examples/08_demos/screw_in_the_bulb``: a YAM screws a lamp into its holder until its base contact closes.

The demo's routine is played the way its page plays it, through the Run button
(:mod:`tests.fixtures.grasp`). The lamp starts loose in the holder's mouth;
the left hand takes the glass and turns it clockwise, a stroke of the wrist at a
time, and the routine ends the turning on the force it reads between the lamp's
base contact and the holder's pad. Nothing joins the two but their threads.

The assertions: the lamp went down one pitch a turn, the contact closed after
the two turns between loose and home and stays closed under the lamp's own
weight once the hand has gone, and the glass did not slip in the fingers.
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
"""Builds the scene and plays the 32 s routine through a page with no browser on it: about 65 s."""

DEMO = ROOT / "examples" / "08_demos" / "screw_in_the_bulb" / "screw_in_the_bulb.py"


def _load():
    name = "_rss_demo_screw_in_the_bulb"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, DEMO)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestTheRoutineScrewsInTheBulb(unittest.TestCase):
    """One build, one press of Run, and every reading taken along the way."""

    @classmethod
    def setUpClass(cls) -> None:
        from robosimstudio.core.math import quat_yaw

        demo = cls.demo = _load()
        sim = cls.sim = demo.build()
        lamp, holder = sim["bulb"], sim["socket"]
        cls.thread = lamp.entity.spec.thread
        cls.weight = 9.81 * float(sim.model.body_mass.numpy()[int(lamp.entity.body)])
        pair = ([holder.entity.contact_shape], [lamp.entity.contact_shape])
        hz = float(sim.request.physics.control_hz)
        roll = sim.arm("left").arm_coords[-1]

        session = cls.session = open_session(sim)
        session.add_routines({"Screw in the bulb": demo.routine})
        panel = session.panels[-1]
        tick(session, 30)
        cls.loose = demo.measures(sim)
        r = cls.readings = {"contacts": 0, "finite": True, "force": 0.0}
        start = lamp.pose
        last = [quat_yaw(start[3:7]), float(sim.state.joint_q.numpy()[roll])]
        turns = [0.0, 0.0]  # the lamp's and the left wrist's
        rows = []
        panel.requested = "Screw in the bulb"
        tick(session, 1)
        steps = 0
        while panel.gen is not None and steps < round(60 * hz):
            tick(session, 1)
            steps += 1
            pose = lamp.pose
            now = [quat_yaw(pose[3:7]), float(sim.state.joint_q.numpy()[roll])]
            for k in range(2):
                turns[k] += math.remainder(now[k] - last[k], 2.0 * math.pi) / (2.0 * math.pi)
            last = now
            closed = float(session.env.gripper("left").command) > 0.99
            rows.append((-turns[0], start[2] - pose[2], turns[1], closed))
            # Every step, as the routine reads it: the force that stops the hand lasts a step or two.
            sim.contacts.update()
            r["force"] = max(r["force"], float(np.linalg.norm(sim.contacts.pairwise_force(*pair))))
            if steps % 5 == 0:
                r["contacts"] = max(r["contacts"], int(sim.pipeline.contacts.rigid_contact_count.numpy()[0]))
                r["finite"] = r["finite"] and bool(np.isfinite(pose).all())
        cls.rows = np.asarray(rows, dtype=np.float64)
        r["seconds"] = steps / hz
        r["said"] = str(panel._status.value) if panel._status is not None else ""
        cls.measures = demo.measures(sim)
        print(f"\n[bulb] loose {cls.loose}\n  {r} {cls.measures} turned {cls.rows[-1, 0]:.3f}", flush=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def _strokes(self) -> list[np.ndarray]:
        """The rows of each stroke: the hand shut and its wrist turning."""
        shut = self.rows[:, 3] > 0.5
        wrist = np.abs(np.gradient(self.rows[:, 2])) > 1.0e-5
        turning = np.flatnonzero(shut & wrist)
        return [chunk for chunk in np.split(turning, np.flatnonzero(np.diff(turning) > 30) + 1) if len(chunk) > 30]

    def test_the_routine_plays_to_its_end(self) -> None:
        self.assertIn("done", self.readings["said"])
        self.assertTrue(self.readings["finite"] and self.sim.state_is_finite())

    def test_the_lamp_started_loose_in_the_holders_mouth(self) -> None:
        """2.15 turns out: standing on the ridge's end 6.2 mm over the floor, its contact open."""
        self.assertAlmostEqual(self.thread.pitch * self.demo.OUT, self.loose["tip_over_floor"], delta=0.4e-3)
        self.assertEqual(0.0, self.loose["contact_force"])

    def test_it_went_down_one_pitch_a_turn(self) -> None:
        turns, down = self.rows[:, 0], self.rows[:, 1]
        turning = np.concatenate(self._strokes())
        on = turning[(turns[turning] > 0.2) & (turns[turning] < 1.8)]
        slope = float(np.polyfit(turns[on], down[on], 1)[0])
        self.assertAlmostEqual(self.thread.pitch, slope, delta=0.05 * self.thread.pitch)

    def test_the_base_contact_closed_after_two_turns_and_stopped_the_hand(self) -> None:
        m = self.measures
        # 1.97 turns to touch the pad; the thread's play is a sixth of a turn more before it clamps.
        self.assertGreater(float(self.rows[-1, 0]), 1.9, "it stopped short of the pad")
        self.assertLess(float(self.rows[-1, 0]), 2.2, "it turned on with the lamp home")
        self.assertAlmostEqual(m["pad_height"], m["tip_over_floor"], delta=0.05e-3, msg="the tip is on the pad")
        self.assertGreater(self.readings["force"], self.demo.SEATED, "the contact was never pressed")
        self.assertLess(self.readings["force"], 60.0, "the hand turned on into a closed contact")
        self.assertLess(len(self._strokes()), self.demo.STROKES, "it stopped on the contact, not on the count")

    def test_the_contact_stays_closed_once_the_hand_has_gone(self) -> None:
        m = self.measures
        self.assertGreater(m["contact_force"], 0.5 * self.weight, "the lamp's weight rests on the contact")
        self.assertLess(m["lean_deg"], 1.0)
        self.assertFalse(bool(self.rows[-1, 3]), "the hand let go")

    def test_the_glass_did_not_slip_in_the_fingers(self) -> None:
        for k, stroke in enumerate(self._strokes()):
            with self.subTest(stroke=k + 1):
                wrist = abs(self.rows[stroke[-1], 2] - self.rows[stroke[0], 2]) * 360.0
                lamp = abs(self.rows[stroke[-1], 0] - self.rows[stroke[0], 0]) * 360.0
                self.assertAlmostEqual(wrist, lamp, delta=6.0)

    def test_the_contacts_fit_the_buffer_the_table_allows(self) -> None:
        self.assertGreater(self.readings["contacts"], 100)
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
        for hold in ("held_by", "latch(", "weld", "body_q.assign", "joint_q.assign"):
            self.assertNotIn(hold, self.source, "nothing holds the lamp but fingers and a thread")
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

    def test_it_says_which_read_is_privileged_and_that_the_holder_is_a_fixture(self) -> None:
        reads = [line for line in self.source.splitlines() if ".position" in line and "sim[" not in line]
        self.assertGreaterEqual(len(reads), 1)
        for line in reads:
            self.assertIn("a privileged read", line)
        self.assertIn("The holder is a fixture", ast.get_docstring(self.tree))


if __name__ == "__main__":
    unittest.main()
