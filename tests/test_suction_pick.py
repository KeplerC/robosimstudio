# SPDX-License-Identifier: Apache-2.0
"""The suction examples' task, done the way their pages ask: down onto the plate, **Vacuum**, lift, carry, let go.

``examples/04_extending/02_suction`` and ``06_tasks/05_suction_scored`` promise
that a hand dragged down onto a plate lying flat, with the right arm's
**Vacuum** box ticked, picks the plate up, carries it 200 mm and sets it down in
the square when the box is unticked. This drives the arm panel through that
gesture with no browser (:mod:`tests.fixtures.grasp`): the cursor the gizmo
writes, the request the box writes, the session's own setpoint law. Nothing is
pinned or teleported, so the plate comes up because the cup holds it.

Three more things the cup has to show, each a claim in
:mod:`robosimstudio.core.suction`: a cup tilted 20 degrees onto the plate does
not seal; holding and letting go keep the captured graph (no eager step); and
under a 2 kg plate both tiers keep hold while the YAM sags, the wrench tier
lets go when its pull passes the break force, and on :data:`GENERIC`, which
puts the plate on VBD, the weld is refused and the wrench holds instead.
"""

from __future__ import annotations

import math
import time
import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""Four scenes built on one GPU, each stepped through the arm panel for a few seconds of scene time."""

BENCH = np.array([0.6, 0.0, 0.75])
"""``bench_075``'s surface frame in world [m]; the examples' numbers are in it."""

SQUARE_LO = BENCH[:2] + np.array([-0.11, -0.22])
SQUARE_HI = BENCH[:2] + np.array([0.11, 0.0])
"""The square as the examples state it, in world x and y [m]."""

HALF_THICKNESS = 0.003
"""The plate's half-thickness [m]: 6 mm thick."""

PRESS = 0.004
"""How far into the plate the cursor is put [m].

Walked down at 50 mm/s the tool centre trails the cursor by 3.5 mm, and the panel puts the cursor back on
the hand when a drag ends with the hand within ``POS_TOLERANCE`` (4 mm) of it. So a cursor put on the
plate's top leaves the face 3.5 mm over it, too far to seal; put 2 mm in, 1.5 mm over. Put 4 mm in, the face
comes down onto the plate and stops there, 3.8 to 3.9 mm short of the cursor: inside the tolerance too, so
the panel puts the cursor back on the hand, and the face rests on the plate with a seal gap under 0.02 mm
(``press_down.py`` on ``04_extending/02_suction``, GPU 3, two runs each; at 4 mm also after the tilted try)."""

LIFT, CARRY, SPEED = 0.08, 0.20, 0.10
"""The lift and the carry [m], and the cursor's walk [m/s]."""

SET_DOWN = 0.002
"""How far over the bench a held plate's underside is stopped before the cup lets go [m].

The plate hangs under the cursor by the seal gap and the arm's sag, so walking the cursor back down the way
it came up pressed the welded plate into the bench, and MuJoCo printed linesearch warnings until the cup let
go. The walk down is measured on the plate instead (:func:`_set_down`)."""


def _tilted(quat, degrees: float) -> np.ndarray:
    """``quat`` turned ``degrees`` about world x first, xyzw."""
    half = math.radians(degrees) / 2
    ax, aw = math.sin(half), math.cos(half)
    bx, by, bz, bw = (float(v) for v in quat)
    return np.array([aw * bx + ax * bw, aw * by - ax * bz, aw * bz + ax * by, aw * bw - ax * bx])


def _walk(session, arm: str, delta, *, speed: float = SPEED, watch=None) -> None:
    """Drag the cursor by ``delta`` [m] at ``speed``, as a mouse does, calling ``watch()`` each step."""
    panel = session.arms[arm]
    start = np.array(panel.cursor_pos, dtype=np.float64)
    end = start + np.asarray(delta, dtype=np.float64)
    legs = max(
        1, math.ceil(float(np.linalg.norm(end - start)) / (speed / float(session.env.request.physics.control_hz)))
    )
    for k in range(1, legs + 1):
        panel.cursor_pos, panel.dragging = start + (end - start) * (k / legs), k < legs
        tick(session, 1)
        if watch is not None:
            watch()
    panel.dragging = False


def _down_onto(session, plate) -> float:
    """Over the plate, then down until the cursor is :data:`PRESS` into it; the tool centre's miss [m]."""
    lying = np.asarray(plate.position, dtype=np.float64)
    top = lying[2] + HALF_THICKNESS
    go(session, "right", np.array([lying[0], lying[1], top + 0.03]), speed=SPEED)
    return go(session, "right", np.array([lying[0], lying[1], top - PRESS]), speed=0.05, steps=120, tolerance=0.0)


def _set_down(session, plate) -> float:
    """Walk the cursor down until the held plate's underside is :data:`SET_DOWN` over the bench; that height [m]."""
    over = float(plate.position[2]) - HALF_THICKNESS - float(BENCH[2])
    _walk(session, "right", (0.0, 0.0, -(over - SET_DOWN)))
    tick(session, 30)
    return float(plate.position[2]) - HALF_THICKNESS - float(BENCH[2])


def _pick(session, sim) -> dict:
    """The page's gesture: down, Vacuum, lift, carry toward the middle of the bench, down, untick, up."""
    panel, plate, cup = session.arms["right"], sim["plate"], sim.suction("right")
    out = {"miss": _down_onto(session, plate), "eager": sim.pipeline.steps_eager}
    rest = np.asarray(plate.position, dtype=np.float64).copy()
    panel.vacuum_request = True
    tick(session, 10)
    panel.refresh()
    out.update(tier=cup.tier, mass=cup.held_mass, readout=panel._seal_readout.value)

    def hand() -> np.ndarray:
        return panel.ik.ee_pose(session.env.state)[0]

    offset = np.asarray(plate.position) - hand()
    drift, held = [0.0], [True]

    def watch() -> None:
        drift[0] = max(drift[0], float(np.linalg.norm(np.asarray(plate.position) - hand() - offset)))
        held[0] = held[0] and cup.holding

    _walk(session, "right", (0.0, 0.0, LIFT + PRESS), watch=watch)
    tick(session, 20)
    out["rise"] = float(plate.position[2] - rest[2])
    _walk(session, "right", (0.0, CARRY, 0.0), watch=watch)
    tick(session, 30)
    out.update(drift=drift[0], held=held[0])
    out["set_down"] = _set_down(session, plate)
    panel.vacuum_request = False
    tick(session, 10)
    out["let_go"] = not cup.holding
    _walk(session, "right", (0.0, 0.0, 0.06))
    tick(session, 60)
    out.update(
        plate=np.asarray(plate.position, dtype=np.float64),
        quat=np.asarray(sim.state.body_q.numpy()[int(sim.scene.by_name["plate"].body)][3:7], dtype=np.float64),
        rest=rest,
        eager_after=sim.pipeline.steps_eager,
        captured=sim.pipeline.captured,
        finite=bool(sim.state_is_finite()),
    )
    return out


class _Pick:
    """Build the example, open its page, pick the plate; the subclasses say which script."""

    EXAMPLE = ""
    PORT = 0

    @classmethod
    def setUpClass(cls) -> None:
        cls.opened, kwargs = _run(cls.EXAMPLE)
        cls.sim = getattr(cls.opened, "sim", cls.opened)
        cls.sim.run(20).sync()
        t = time.perf_counter()
        cls.sim.run(100).sync()
        cls.ms_per_step = 1e3 * (time.perf_counter() - t) / 100
        cls.session = open_session(cls.opened, cls.PORT, kwargs.get("page"))
        tick(cls.session, 30)
        cls.before_tilt()
        cls.result = _pick(cls.session, cls.sim)

    @classmethod
    def before_tilt(cls) -> None:
        """Nothing before the pick; the page's own test tries a tilted cup first."""

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_scene_steps_captured_inside_its_budget(self) -> None:
        self.assertTrue(self.sim.pipeline.captured, self.sim.pipeline.capture_error)
        self.assertLess(self.ms_per_step, 20.0, f"{self.ms_per_step:.1f} ms per control step; the docstring says 3.5")

    def test_the_plate_rests_in_the_square(self) -> None:
        result = self.result
        self.assertTrue(result["finite"])
        self.assertTrue(
            0.0 < result["set_down"] < 2.0 * SET_DOWN,
            f"the held plate was let go {result['set_down'] * 1e3:.1f} mm over the bench, not {SET_DOWN * 1e3:.0f}",
        )
        self.assertTrue(result["let_go"], "the cup still holds the plate after Vacuum was unticked")
        centre = result["plate"]
        self.assertTrue(np.all(centre[:2] >= SQUARE_LO) and np.all(centre[:2] <= SQUARE_HI), f"plate at {centre}")
        top = float(centre[2] + HALF_THICKNESS - BENCH[2])
        self.assertLess(top, 0.010, f"the plate's top is {top * 1e3:.1f} mm over the bench")
        self.assertGreater(abs(float(result["quat"][3])), math.cos(math.radians(3) / 2), "the plate is not lying flat")
        self.assertGreater(float(centre[1] - result["rest"][1]), CARRY - 0.03, "the plate was not carried")


class TestThePageLiftsThePlate(_Pick, unittest.TestCase):
    """``04_extending/02_suction``: a tilted cup is refused, then the level one seals by weld and carries the plate."""

    EXAMPLE = "04_extending/02_suction.py"
    PORT = 9630

    @classmethod
    def before_tilt(cls) -> None:
        session, plate, cup = cls.session, cls.sim["plate"], cls.sim.suction("right")
        panel = session.arms["right"]
        home = np.array(panel.cursor_quat, dtype=np.float64)
        panel.cursor_quat = _tilted(home, 20.0)
        lying = float(plate.position[2])
        _down_onto(session, plate)
        panel.vacuum_request = True
        tick(session, 10)
        cls.tilted = {"holding": cup.holding, "why": cup.last.why, "tilt": cup.last.tilt_rad}
        _walk(session, "right", (0.0, 0.0, 0.05))
        tick(session, 20)
        cls.tilted["rise"] = float(plate.position[2]) - lying
        panel.vacuum_request = False
        tick(session, 5)
        panel.cursor_quat = home
        go(session, "right", np.array(panel.cursor_pos), steps=60)

    def test_a_tilted_cup_does_not_seal(self) -> None:
        self.assertFalse(self.tilted["holding"], "a cup 20 degrees off the plate sealed")
        self.assertIn("degrees off the face", self.tilted["why"])
        self.assertGreater(math.degrees(self.tilted["tilt"]), 15.0)
        self.assertLess(abs(self.tilted["rise"]), 0.002, "the plate came up under a cup that did not seal")

    def test_the_cup_seals_by_weld(self) -> None:
        self.assertLess(self.result["miss"], 0.006, "the hand did not get down onto the plate")
        self.assertEqual("weld", self.result["tier"])
        self.assertAlmostEqual(0.3, self.result["mass"], places=3)
        self.assertTrue(self.result["readout"].startswith("sealed"), self.result["readout"])

    def test_the_plate_comes_up_with_the_cup_and_stays_on_it(self) -> None:
        self.assertGreater(self.result["rise"], 0.06, f"the plate rose {self.result['rise'] * 1e3:.1f} mm")
        self.assertTrue(self.result["held"], "the seal let go during the lift or the carry")
        self.assertLess(self.result["drift"], 0.003, f"the plate moved {self.result['drift'] * 1e3:.1f} mm on the cup")

    def test_hold_and_release_keep_the_graph(self) -> None:
        self.assertTrue(self.result["captured"])
        self.assertEqual(self.result["eager"], self.result["eager_after"], "a hold or a release dropped the graph")


class TestTheTaskGoalTurnsTrue(_Pick, unittest.TestCase):
    """``06_tasks/05_suction_scored``: the same pick on the task's page, and ``evaluate`` says success."""

    EXAMPLE = "06_tasks/05_suction_scored.py"
    PORT = 9631

    def test_the_goal_is_met(self) -> None:
        _, _, info = self.opened.evaluate()
        self.assertTrue(info["success"], f"metrics: {info['metrics']}")


class TestTheTiersUnderLoad(unittest.TestCase):
    """A 2 kg plate: both tiers hold it while the arm sags, and the wrench lets go past its break force."""

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        scene = rss.Scene(cell="bench_075", name="heavy_plate", physics=rss.MUJOCO_500HZ)
        scene.add(rss.robot("yam_bimanual_suction"))
        scene.add(rss.box(name="plate", half=(0.075, 0.075, HALF_THICKNESS), at=(0.0, -0.31, 0.003), mass=2.0))
        cls.sim = scene.build(control_mode="joint_position")
        cls.session = open_session(cls.sim, 9632)
        tick(cls.session, 30)
        cls.cup = cls.sim.suction("right")
        cls.body = int(cls.sim.scene.by_name["plate"].body)
        cls.default_break = cls.cup.break_force
        cls.lifts = {tier: cls.lift(tier) for tier in ("weld", "wrench")}
        cls.cup.break_force = 10.0
        cls.lifts["broken"] = cls.lift("wrench")
        cls.cup.break_force = cls.default_break

    @classmethod
    def lift(cls, tier: str) -> dict:
        session, plate, cup = cls.session, cls.sim["plate"], cls.cup
        _down_onto(session, plate)
        rest = np.asarray(plate.position, dtype=np.float64).copy()
        report = cup.hold(cls.body, tier=tier)
        tick(session, 10)
        panel = session.arms["right"]
        hand = panel.ik.ee_pose(session.env.state)[0]
        _walk(session, "right", (0.0, 0.0, LIFT + PRESS))
        tick(session, 30)
        out = {
            "sealed": report.sealed,
            "holding": cup.holding,
            "broke": cup.broke,
            "rise": float(plate.position[2] - rest[2]),
            "sag": float(panel.cursor_pos[2] - panel.ik.ee_pose(session.env.state)[0][2]),
            "hand_rise": float(panel.ik.ee_pose(session.env.state)[0][2] - hand[2]),
        }
        out["set_down"] = _set_down(session, plate)
        cup.release()
        _walk(session, "right", (0.0, 0.0, 0.05))
        tick(session, 30)
        return out

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_weld_holds_and_the_arm_sags(self) -> None:
        weld = self.lifts["weld"]
        self.assertTrue(weld["sealed"] and weld["holding"], weld)
        self.assertAlmostEqual(weld["rise"], weld["hand_rise"], delta=0.003, msg="the plate left the cup")
        self.assertGreater(weld["sag"], 0.02, f"a 2 kg plate on the YAM's wrist sagged only {weld['sag'] * 1e3:.1f} mm")

    def test_the_wrench_holds_below_its_break_force_and_loads_the_arm(self) -> None:
        wrench = self.lifts["wrench"]
        self.assertTrue(wrench["holding"] and not wrench["broke"], wrench)
        self.assertGreater(self.default_break, 2.0 * 9.81, "the default break force is under a 2 kg plate's weight")
        self.assertAlmostEqual(wrench["rise"], wrench["hand_rise"], delta=0.012)
        self.assertGreater(wrench["sag"], 0.02, "the arm did not feel the plate: the wrench has no reaction")

    def test_the_wrench_lets_go_past_its_break_force(self) -> None:
        broken = self.lifts["broken"]
        self.assertTrue(broken["sealed"], "the seal test itself failed")
        self.assertTrue(broken["broke"] and not broken["holding"], broken)
        self.assertLess(broken["rise"], 0.01, "a plate past the break force came up")


class TestGenericHoldsByWrench(unittest.TestCase):
    """``GENERIC`` puts the plate on VBD: no weld to it, and an automatic hold is the wrench."""

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        scene = rss.Scene(cell="bench_075", name="generic_plate")
        scene.add(rss.robot("yam_bimanual_suction"))
        scene.add(rss.box(name="plate", half=(0.075, 0.075, HALF_THICKNESS), at=(0.0, -0.31, 0.003), mass=0.3))
        cls.sim = scene.build(control_mode="joint_position")
        cls.session = open_session(cls.sim, 9633)
        tick(cls.session, 30)
        cls.cup = cls.sim.suction("right")
        cls.body = int(cls.sim.scene.by_name["plate"].body)
        cls.miss = _down_onto(cls.session, cls.sim["plate"])
        cls.report = cls.cup.hold(cls.body)
        cls.tier = cls.cup.tier
        rest = float(cls.sim["plate"].position[2])
        _walk(cls.session, "right", (0.0, 0.0, LIFT + PRESS))
        tick(cls.session, 20)
        cls.rise = float(cls.sim["plate"].position[2]) - rest
        cls.cup.release()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_weld_is_refused(self) -> None:
        self.assertFalse(self.cup.weld_available(self.body))
        with self.assertRaises(RuntimeError):
            self.cup.hold(self.body, tier="weld", check=False)

    def test_the_automatic_hold_is_the_wrench_and_lifts(self) -> None:
        self.assertTrue(self.report.sealed, self.report)
        self.assertEqual("wrench", self.tier)
        self.assertGreater(self.rise, 0.06)


if __name__ == "__main__":
    unittest.main()
