# SPDX-License-Identifier: Apache-2.0
"""The water example's gesture, made through its arm panel: the hand pours the water into the glass past its line.

``examples/03_materials/14_water`` promises that the hand, dragged down over the
cup and closed, lifts it, carries it over the glass and pours it when the gizmo is
turned. This drives the example's own page the way :mod:`tests.fixtures.grasp`
does -- the cursor the gizmo writes, the grip the Close button writes, the
session's own setpoint law -- and turns the cursor the way the gizmo's rings do.
Nothing is pinned and no handle is down: the cup comes up because the pads hold
it, and the water leaves it because the wrist tips it. Two controls drive the same
gesture with one thing changed: a shallow tilt, which pours nothing, and the glass
missed by 100 mm, which puts the water on the bench.

The water is Genesis's SPH, so this module needs the ``[genesis]`` extra, which
lives in a virtual environment of its own (``pyproject.toml``); it skips without it,
as ``tests/test_genesis.py`` does.
"""

from __future__ import annotations

import importlib.util
import math
import time
import unittest

import numpy as np

TIER = "medium"
"""One GPU, the [genesis] extra: the examples built four times, each stepped through its page for about 15 s."""

BENCH = np.array([0.6, 0.0, 0.75])
"""``bench_075``'s surface frame in world [m]; the example's numbers are in it."""
GLASS_LO = BENCH + np.array([0.09, -0.16, 0.0])
GLASS_HI = BENCH + np.array([0.15, -0.10, 0.055])
"""The glass's inside as the example states it: 60 mm square about bench (0.12, -0.13), 55 mm tall."""
LINE = 0.025
"""The example's fill line over the glass's floor [m]."""
SPACING = 0.004
"""The water's particle spacing [m]."""

GRASP = 0.055
"""Tool centre over the cup's bottom at the close [m]: the pads from 48 mm up to the 78 mm rim."""
LIFT_TO = 0.17
"""Where the cursor lifts the tool centre to, over the bench [m]; the loaded arm hangs about 27 mm under it, and
the cup's bottom clears the 55 mm glass by about 35 mm."""
ACROSS = 0.125
"""How far the cup is carried toward the glass [m]. Over three drives each (``p7_drive.py``): 0.125 put 99.8-100%
in the glass, 0.135 put 98.8%, 0.14 put 96.3-96.9% (the stream's front lands past the far wall), 0.11 put 96% and
0.155 put 80%."""
TILT = 120.0
"""The gizmo's turn about world x [deg]; the wrist stops at about 115."""
SPEED = 0.10
"""The cursor's walk for the moves [m/s]."""


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


HAS_GENESIS = _importable("genesis")


def _turn(session, axis, degrees: float, seconds: float, watch) -> None:
    """Turn the right arm's cursor ``degrees`` about the world ``axis`` at a steady rate, as a ring drag does."""
    from robosimstudio.core.math import quat_from_axis_angle, quat_multiply
    from tests.fixtures.grasp import tick

    panel = session.arms["right"]
    pos0 = np.array(panel.cursor_pos, dtype=np.float64)
    q0 = np.array(panel.cursor_quat, dtype=np.float64)
    n = max(1, round(seconds * float(session.env.request.physics.control_hz)))
    for k in range(1, n + 1):
        turn = quat_from_axis_angle(np.asarray(axis, dtype=np.float64), math.radians(degrees) * k / n)
        panel.cursor_pos, panel.cursor_quat, panel.dragging = pos0, quat_multiply(turn, q0), k < n
        tick(session, 1)
        watch()


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0))))


def _in_cup(cup, points: np.ndarray) -> int:
    """How many particles are inside the cup's cavity, in the cup's own frame."""
    from robosimstudio.core.math import quat_to_mat

    pose = np.asarray(cup.pose, dtype=np.float64)
    rel = (points - pose[:3]) @ quat_to_mat(pose[3:7])
    return int(np.all((rel > [-0.025, -0.025, 0.005]) & (rel < [0.025, 0.025, 0.08]), axis=1).sum())


def _pour(sim, session, *, tilt: float = TILT, aside: float = 0.0) -> dict:
    """Down onto the cup, **Close**, up, across (and ``aside`` toward the robot), tip, hold, and back upright."""
    from robosimstudio.core.math import quat_to_mat
    from robosimstudio_materials.mpm.measures import spill_fraction
    from robosimstudio_materials.water import fill_level
    from tests.fixtures.grasp import go, tick

    panel = session.arms["right"]
    cup, water = sim["cup"], sim["water"]
    bridge = sim.scene.program.solver.solver("genesis").bridge
    body = int(cup.bodies[0])
    hz = float(session.env.request.physics.control_hz)
    out: dict = {"slip_mm": 0.0, "carried_fz": []}

    def hand():
        pos, quat = panel.ik.ee_pose(session.env.state)
        return pos, quat_to_mat(quat)

    def in_hand():
        pos, rot = hand()
        pose = np.asarray(cup.pose, dtype=np.float64)
        return rot.T @ (pose[:3] - pos)

    grip = {"at": None}

    def watch():
        if grip["at"] is not None:
            out["slip_mm"] = max(out["slip_mm"], 1e3 * float(np.linalg.norm(in_hand() - grip["at"])))

    standing = np.asarray(cup.position, dtype=np.float64)
    at = standing + np.array([0.0, 0.0, GRASP])
    out["miss_grasp"] = go(session, "right", at, speed=SPEED)
    panel.grip_request = 1.0
    tick(session, round(hz))
    out["width"] = float(session.env.gripper("right").width(session.env.state))
    grip["at"] = in_hand()
    go(session, "right", np.array([at[0], at[1], BENCH[2] + LIFT_TO]), speed=SPEED, steps=60)
    go(session, "right", np.array(panel.cursor_pos) + np.array([-aside, ACROSS, 0.0]), speed=SPEED, steps=60)
    for _ in range(round(0.5 * hz)):
        tick(session, 1)
        watch()
        out["carried_fz"].append(float(bridge.last_force.get(body, np.zeros(3))[2]))
    out["carried_z"] = float(np.asarray(cup.position)[2] - BENCH[2])
    out["carried_in_cup"] = _in_cup(cup, np.asarray(water.particle_q, dtype=np.float64))
    before = hand()[1]
    _turn(session, (1.0, 0.0, 0.0), -tilt, 2.0, watch)
    out["turned_deg"] = _angle(before, hand()[1])
    for _ in range(round(2.5 * hz)):
        tick(session, 1)
        watch()
    _turn(session, (1.0, 0.0, 0.0), tilt, 2.0, watch)
    tick(session, round(hz))
    points = np.asarray(water.particle_q, dtype=np.float64)
    out["fill"] = fill_level(points, GLASS_LO, GLASS_HI, SPACING)
    out["spilled"] = spill_fraction(points, GLASS_LO, GLASS_HI, floor=BENCH[2], band=2.0 * SPACING)
    out["in_cup"] = _in_cup(cup, points)
    out["count"] = len(points)
    out["off_bench"] = int((points[:, 2] < BENCH[2] - 0.005).sum())
    out["width_end"] = float(session.env.gripper("right").width(session.env.state))
    out["finite"] = bool(session.env.state_is_finite())
    out["weight"] = float(sim.scene.model.particle_mass.numpy()[water.particles].sum()) * 9.81
    return out


def _drive(port: int, example: str = "03_materials/14_water.py", **kw) -> tuple:
    """Build ``example``, time its step, open its page and pour. Returns ``(opened, ms, result)``.

    ``opened`` is what the example handed the studio: a ``Sim``, or the scored example's ``Env``.
    """
    from tests.fixtures.examples import _run
    from tests.fixtures.grasp import open_session, tick

    opened, kwargs = _run(example)
    sim = getattr(opened, "sim", opened)
    sim.run(20).sync()
    t = time.perf_counter()
    sim.run(60).sync()
    ms = 1e3 * (time.perf_counter() - t) / 60
    session = open_session(opened, port, kwargs.get("page"))
    tick(session, 30)
    try:
        result = _pour(sim, session, **kw)
    finally:
        session.teardown()
    return opened, ms, result


@unittest.skipUnless(HAS_GENESIS, "the [genesis] extra is not installed")
class TestTheHandPoursTheWater(unittest.TestCase):
    """The cup in the fingers, over the glass, tipped: the water ends in the glass, past the line."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim, cls.ms_per_step, cls.result = _drive(9671)

    def test_the_scene_steps_inside_its_budget(self) -> None:
        """At the precision the arrangement states. A Genesis process is initialised once, and an IPC scene built
        earlier in it (``tests/test_genesis``, in one tier run) leaves it at float64, where this pour measured
        71.7 ms a step: right, and about 8x slower in the SPH."""
        from robosimstudio.engines.genesis.runtime import _ENGINE

        if _ENGINE["precision"] != "32":
            self.skipTest(
                f"Genesis runs at precision {_ENGINE['precision']} in this process, not the 32 it was timed at"
            )
        self.assertLess(self.ms_per_step, 40.0, f"{self.ms_per_step:.1f} ms per control step; the docstring says 24")

    def test_the_fingers_closed_on_the_cup(self) -> None:
        self.assertLess(self.result["miss_grasp"], 0.006, "the hand did not get down around the cup")
        self.assertAlmostEqual(0.066, self.result["width"], delta=0.003, msg="the jaws are not on the cup's walls")

    def test_the_cup_was_carried_full_and_stayed_in_the_fingers(self) -> None:
        r = self.result
        self.assertTrue(r["finite"])
        self.assertGreater(r["carried_z"], 0.065, "the cup's bottom does not clear the 55 mm glass")
        self.assertGreaterEqual(r["carried_in_cup"], 0.99 * r["count"], "water left the cup before the tilt")
        self.assertLess(r["slip_mm"], 10.0, f"the cup moved {r['slip_mm']:.1f} mm in the hand")
        self.assertAlmostEqual(0.066, r["width_end"], delta=0.003, msg="the jaws are no longer on the cup")

    def test_the_cup_feels_the_waters_weight(self) -> None:
        """The two-way harvest, against a number that is not a simulation outcome: carried still, the water's
        reaction on the cup is its weight."""
        fz = -float(np.mean(self.result["carried_fz"]))
        self.assertAlmostEqual(fz / self.result["weight"], 1.0, delta=0.15, msg=f"{fz:.3f} N on the cup")

    def test_the_wrist_tipped_the_cup(self) -> None:
        self.assertGreater(self.result["turned_deg"], 105.0, f"the hand turned {self.result['turned_deg']:.1f} deg")

    def test_the_glass_fills_past_its_line(self) -> None:
        fill = self.result["fill"]
        self.assertGreaterEqual(fill["fraction"], 0.9, f"{fill['fraction']:.3f} of the water is in the glass")
        self.assertGreaterEqual(fill["height"], LINE, f"the water stands {1e3 * fill['height']:.1f} mm")

    def test_little_is_spilled(self) -> None:
        self.assertLessEqual(self.result["spilled"], 0.05, f"{self.result['spilled']:.3f} spilled on the bench")
        self.assertLessEqual(self.result["off_bench"], 10, "water ran off the bench")

    def test_the_water_kept_its_volume(self) -> None:
        """The height it stands at is the height its count fills: no particle lost, none squeezed away."""
        fill = self.result["fill"]
        self.assertEqual(self.result["count"], 1728)
        self.assertAlmostEqual(fill["height"] / fill["expected"], 1.0, delta=0.1, msg=str(fill))

    def test_a_reset_puts_the_water_back_in_the_cup(self) -> None:
        sim = self.sim
        sim.reset_scene()
        targets = getattr(sim.pipeline.control, "joint_target_q", None)
        if targets is not None:
            targets.assign(sim.pipeline.state_0.joint_q.numpy().copy())
        sim.run(10).sync()
        points = np.asarray(sim["water"].particle_q, dtype=np.float64)
        self.assertGreaterEqual(_in_cup(sim["cup"], points), 0.99 * len(points), "the water did not come back")


@unittest.skipUnless(HAS_GENESIS, "the [genesis] extra is not installed")
class TestAShallowTiltPoursNothing(unittest.TestCase):
    """The same gesture with the gizmo turned 60 degrees: the water stays in the cup."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim, _ms, cls.result = _drive(9672, tilt=60.0)

    def test_the_water_stayed_in_the_cup(self) -> None:
        r = self.result
        self.assertLess(r["turned_deg"], 65.0)
        self.assertGreaterEqual(r["in_cup"], 0.99 * r["count"], f"{r['in_cup']} of {r['count']} in the cup")
        # One drive in ten lets a few particles over the rim: 0.1% in the glass, 1,724 of 1,728 in the cup.
        self.assertLessEqual(r["fill"]["fraction"], 0.005, f"{r['fill']['fraction']:.4f} in the glass")


@unittest.skipUnless(HAS_GENESIS, "the [genesis] extra is not installed")
class TestPouringAsideSpillsIt(unittest.TestCase):
    """The same pour 100 mm toward the robot, beside the glass: the water lands on the bench."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim, _ms, cls.result = _drive(9673, aside=0.10)

    def test_the_water_is_on_the_bench_not_in_the_glass(self) -> None:
        r = self.result
        self.assertLessEqual(r["fill"]["fraction"], 0.02, f"{r['fill']['fraction']:.3f} in the glass")
        self.assertGreaterEqual(r["spilled"], 0.8, f"{r['spilled']:.3f} spilled")


@unittest.skipUnless(HAS_GENESIS, "the [genesis] extra is not installed")
class TestTheTaskGoalTurnsTrue(unittest.TestCase):
    """``06_tasks/10_water_scored``: the same pour on the task's page, from the YAM's home, and ``evaluate`` says success."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.env, cls.ms_per_step, cls.result = _drive(9674, "06_tasks/10_water_scored.py")

    def test_the_goal_is_met(self) -> None:
        _, _, info = self.env.evaluate()
        self.assertTrue(info["success"], f"metrics: {info['metrics']}")
        self.assertGreaterEqual(float(info["metrics"]["height"]), LINE)
        self.assertGreaterEqual(self.result["fill"]["fraction"], 0.9)


if __name__ == "__main__":
    unittest.main()
