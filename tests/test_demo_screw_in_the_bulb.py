# SPDX-License-Identifier: Apache-2.0
"""``examples/08_demos/screw_in_the_bulb``: a YAM takes a lamp from its tray and screws it into its holder until it lights.

The demo's routine is played the way its page plays it, through the Run button
(:mod:`tests.fixtures.grasp`), with the demo's own per-step reading of the
centre contact lighting the lamp. The lamp is a household one, an A19 bulb on a
medium Edison screw; it starts standing in a tray, and the left hand takes it
by the glass, sets it in the holder's mouth with its thread on the holder's
helix, and turns it clockwise a stroke of the wrist at a time, its jaws just
off the glass for each turn back. Nothing joins lamp and holder but their
threads.

The assertions: the lamp left the tray in the hand and was set in where its
thread starts; it went down one pitch a turn, inside the thread's play; the
centre contact closed, lit the lamp and stopped the hand, and stays closed
under the lamp's own weight once the hand has gone; and the glass did not slip
in the fingers.
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
"""Builds the scene and plays the 41 s routine through a page with no browser on it: about 55 s."""

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
        import robosimstudio as rss
        from robosimstudio.core.math import quat_yaw

        demo = cls.demo = _load()
        sim = cls.sim = demo.build()
        lamp, holder = sim["bulb"], sim["socket"]
        cls.lamp = lamp.entity.spec.lamp
        cls.weight = 9.81 * float(sim.model.body_mass.numpy()[int(lamp.entity.body)])
        pair = ([holder.entity.contact_shape], [lamp.entity.contact_shape])
        hz = float(sim.request.physics.control_hz)
        roll = sim.arm("left").arm_coords[-1]
        home = float(rss.get_cell(demo.CELL).surface_z) + holder.entity.spec.rim
        session = cls.session = open_session(sim)
        session.each_step(demo.circuit(sim))
        session.add_routines({"Screw in the bulb": demo.routine})
        panel = session.panels[-1]
        tick(session, 30)
        cls.before = demo.measures(sim)
        r = cls.readings = {"contacts": 0, "finite": True, "force": 0.0}
        last = [quat_yaw(lamp.pose[3:7]), float(sim.state.joint_q.numpy()[roll])]
        turns = [last[0] / (2.0 * math.pi), 0.0]  # the lamp's, from how it stands, and the left wrist's
        rows = []
        panel.requested = "Screw in the bulb"
        tick(session, 1)
        steps = 0
        while panel.gen is not None and steps < round(80 * hz):
            tick(session, 1)
            steps += 1
            pose = lamp.pose
            now = [quat_yaw(pose[3:7]), float(sim.state.joint_q.numpy()[roll])]
            for k in range(2):
                turns[k] += math.remainder(now[k] - last[k], 2.0 * math.pi) / (2.0 * math.pi)
            last = now
            closed = float(session.env.gripper("left").command) > 0.99
            # Every step, as the routine reads it: the force that stops the hand lasts a step or two.
            sim.contacts.update()
            force = float(np.linalg.norm(sim.contacts.pairwise_force(*pair)))
            rows.append((turns[0], pose[2] - home, turns[1], closed, bool(lamp.entity.lit), force))
            r["force"] = max(r["force"], force)
            if steps % 5 == 0:
                r["contacts"] = max(r["contacts"], int(sim.pipeline.contacts.rigid_contact_count.numpy()[0]))
                r["finite"] = r["finite"] and bool(np.isfinite(pose).all())
        cls.rows = np.asarray(rows, dtype=np.float64)
        r["seconds"] = steps / hz
        r["said"] = str(panel._status.value) if panel._status is not None else ""
        tick(session, 60)
        cls.measures = demo.measures(sim)
        strokes = cls._strokes(cls)
        # The lamp's turn in the hand and its height say which turn of the helix it was set in on.
        first = strokes[0][0]
        cls.whole = round(cls.rows[first, 1] / cls.lamp.pitch - cls.rows[first, 0])
        cls.out = cls.rows[:, 0] + cls.whole
        print(f"\n[bulb] before {cls.before}\n  {r} {cls.measures} set in {cls.out[first]:.3f} turns out", flush=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def _strokes(self) -> list[np.ndarray]:
        """The rows of each stroke: one per closing of the hand, the lamp in the holder and the wrist screwing it in."""
        shut = self.rows[:, 3] > 0.5
        grasp = np.cumsum(np.diff(shut.astype(int), prepend=0) == 1)
        wrist = np.gradient(self.rows[:, 2])
        way = np.sign(self.rows[-1, 2] - self.rows[np.flatnonzero(shut)[-1] - 30, 2])
        in_holder = (self.rows[:, 1] < 0.02) & (self.rows[:, 1] > -0.001)
        turning = np.flatnonzero(shut & (wrist * way > 1.0e-5) & in_holder)
        strokes = [turning[grasp[turning] == k] for k in np.unique(grasp[turning])]
        return [stroke for stroke in strokes if len(stroke) > 30]

    def test_the_routine_plays_to_its_end(self) -> None:
        self.assertIn("done", self.readings["said"])
        self.assertTrue(self.readings["finite"] and self.sim.state_is_finite())

    def test_the_lamp_started_in_its_tray_dark(self) -> None:
        b = self.before
        self.assertAlmostEqual(
            -self.sim["socket"].entity.spec.rim, b["eyelet_over_contact"], delta=0.3e-3, msg="on the bench"
        )
        self.assertAlmostEqual(math.dist(self.demo.AT, self.demo.TRAY), b["off_axis"], delta=1.0e-3)
        self.assertLess(b["lean_deg"], 2.0)
        self.assertEqual(0.0, b["contact_force"])
        self.assertFalse(b["lit"])

    def test_the_hand_carried_it_over_the_holder_s_mouth(self) -> None:
        self.assertGreater(float(self.rows[:, 1].max()), self.lamp.well + 0.01, "its eyelet cleared the mouth")

    def test_it_was_set_in_where_its_thread_starts(self) -> None:
        """On the holder's helix, between the 3.25 turns at which the threads first meet and a turn more."""
        first = self._strokes()[0][0]
        self.assertGreaterEqual(self.out[first], self.lamp.catch / self.lamp.pitch)
        self.assertLess(self.out[first], self.lamp.catch / self.lamp.pitch + 1.0)
        set_in = self.lamp.pitch * self.out[first] - self.demo.PRESS
        self.assertAlmostEqual(set_in, self.rows[first, 1], delta=0.1e-3, msg="on the helix, less the press")

    def test_it_went_down_one_pitch_a_turn(self) -> None:
        turning = np.concatenate(self._strokes())
        # From the second stroke on: the first starts pressed 0.3 mm under the helix and comes up onto it.
        on = turning[(self.out[turning] < 2.7) & (self.out[turning] > 0.25)]
        self.assertGreater(float(np.ptp(self.out[on])), 2.4, "measured over two and a half turns")
        slope = float(np.polyfit(self.out[on], self.rows[on, 1], 1)[0])
        self.assertAlmostEqual(self.lamp.pitch, slope, delta=0.02 * self.lamp.pitch)

    def test_turning_it_stayed_inside_the_thread_s_play(self) -> None:
        """Centred, the lamp has 0.18 mm either way; set in pressed 0.3 mm down, it starts 0.06 under that."""
        turning = np.concatenate(self._strokes())
        on = turning[(self.out[turning] < 3.2) & (self.out[turning] > 0.1)]
        off = self.rows[on, 1] - self.lamp.pitch * self.out[on]
        print(
            f"\n[bulb] height off the helix, turning: {1e3 * off.min():+.3f} to {1e3 * off.max():+.3f} mm", flush=True
        )
        self.assertLess(float(np.abs(off).max()), 0.5 * self.lamp.axial_play + 0.1e-3)

    def test_the_centre_contact_closed_and_stopped_the_hand(self) -> None:
        m = self.measures
        self.assertAlmostEqual(0.0, m["eyelet_over_contact"], delta=0.05e-3, msg="the eyelet is on the contact")
        self.assertLess(m["off_axis"], 0.3e-3)
        self.assertGreater(self.readings["force"], self.demo.SEATED, "the contact was never pressed")
        self.assertLess(self.readings["force"], 10.0, "the hand turned on into a closed contact")
        self.assertEqual(6, len(self._strokes()), "home comes in the sixth stroke")
        # A tenth of a turn past home is the thread's play taken up; more is a hand that did not stop.
        self.assertLess(abs(float(self.out[-1])), 0.1)
        self.assertAlmostEqual(self.demo.OUT, 5.0 * self.demo.STROKE / (2.0 * math.pi) + 0.38, delta=0.01)

    def test_the_lamp_lit_as_the_contact_closed_and_not_before(self) -> None:
        lit = self.rows[:, 4] > 0.5
        self.assertTrue(lit.any())
        # A lamp leaning a degree or two in the hand touches by its eyelet's edge, its middle still this far off.
        self.assertLess(float(self.rows[np.argmax(lit), 1]), 0.25e-3, "dark until its eyelet was on the contact")
        self.assertTrue(lit[-1])

    def test_the_contact_stays_closed_once_the_hand_has_gone(self) -> None:
        m = self.measures
        # All of its weight, or most: a lamp that settled a little off upright leaves a tenth of it on the thread.
        self.assertGreater(m["contact_force"], 0.75 * self.weight)
        self.assertLess(m["contact_force"], 1.1 * self.weight)
        self.assertLess(m["lean_deg"], 0.5)
        self.assertTrue(m["lit"])
        self.assertFalse(bool(self.rows[-1, 3]), "the hand let go")

    def test_the_glass_did_not_slip_in_the_fingers(self) -> None:
        for k, stroke in enumerate(self._strokes()):
            with self.subTest(stroke=k + 1):
                wrist = abs(self.rows[stroke[-1], 2] - self.rows[stroke[0], 2]) * 360.0
                lamp = abs(self.rows[stroke[-1], 0] - self.rows[stroke[0], 0]) * 360.0
                self.assertAlmostEqual(wrist, lamp, delta=2.0)

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
        routine = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "routine")
        reads = [line for line in ast.get_source_segment(self.source, routine).splitlines() if "lamp.pose" in line]
        self.assertGreaterEqual(len(reads), 3)
        for line in reads:
            self.assertIn("a privileged read", line)
        self.assertIn("The holder is a fixture", ast.get_docstring(self.tree))


if __name__ == "__main__":
    unittest.main()
