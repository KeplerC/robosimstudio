# SPDX-License-Identifier: Apache-2.0
"""The paper fold example's headline task, done by both of the robot's hands the way its take plays it.

``examples/03_materials/05_paper`` promises a gesture and a number. This plays the FOLD take's own drive
(``scripts/recording/takes.py``'s :func:`fold_drive`) through the example's arm panels, a control step per
step, with no browser, and reads the same ``crease_angle`` the page shows. Nothing is pinned to a hand: each
hand presses **Close** on the overhanging edge from its own corner and its fingers stall on the sheet at the
grip cap, the contact forces fed back along each finger's slide; the two carry the edge up to upright turning
88 degrees, lay it on the far half and let go; then each runs a closed fingertip along its half of the fold.

What it holds the drive to is what made the old fold unrealistic: there the arms passed 60 mm into each
other, a hand turned 180 degrees holding the edge and the edge stayed clamped while the other hand stamped
the fold; a later version held the edge in a position vise the sheet read 30 to 120 N in. Here the arms,
which collide with each other, keep 20 mm apart and each to its own side of the middle at every control
step; a hand turns at most 90 degrees while it holds and its grip stalls at its cap; nothing but the creasing
fingertip and the box touches the sheet while the fold is creased; the edges end within 10 mm of each other
and the fold keeps its crease once the hands have gone, the sheet where it was laid.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import open_session, tick

TIER = "medium"
"""One GPU: the example built once and its take's drive played through its page (about 11,000 control steps)."""

FOLD_PORT = 9734
GAP = 0.020
"""The closest the two arms' colliders may come [m]."""


class TestTheHandsFoldTheSheet(unittest.TestCase):
    """Pinch the overhang from both corners, carry it up and lay it over, let go, crease it: it stays folded in half."""

    @classmethod
    def setUpClass(cls) -> None:
        from robosimstudio.assetlib.semantics.snapshot import Snapshot
        from robosimstudio.assetlib.vocabulary.measures import cloth_fold
        from robosimstudio_materials.paper import crease_angle

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from recording import takes
        from recording.lockstep import Film

        cls.takes = takes
        cls.sim, kwargs = _run("03_materials/05_paper.py")
        sim = cls.sim
        solver = sim.pipeline.program.solver
        sim.run(10).sync()
        t = time.perf_counter()
        sim.run(60).sync()
        cls.ms_per_step = 1e3 * (time.perf_counter() - t) / 60
        cls.session = session = open_session(sim, FOLD_PORT, kwargs.get("page"))
        film = Film(takes.find("03_materials/05_paper"), port=0, scratch=None, software=False, probe=False, log=print)
        film.sim, film.session = sim, session
        lay = takes._fold_layout(film)
        line = {"crease": {"axis": 0, "at": lay["line"]}}
        index = sim.scene.embodiment.arms
        body = sim.model.shape_body.numpy()
        fingers = {arm: [np.flatnonzero(np.isin(body, side)) for side in index[arm].finger_bodies] for arm in index}
        labels = [str(n).rsplit("/", 1)[-1] for n in sim.model.joint_label]
        q_start = sim.model.joint_q_start.numpy()
        driven = {arm: int(q_start[labels.index(f"{arm}_left_finger")]) for arm in index}
        actuator = cls._finger_actuators(solver.rigid)
        bench_top = float(film.bench((0.0, 0.0, 0.0))[2])
        cls.rest = rest = np.asarray(sim["paper"].particle_q, dtype=np.float64)
        base = rest[:, 0] > lay["fold"] + 0.0106
        cls.pushes, cls.finger_q, cls.servo, cls.lowest, cls.drift, cls.marks = {}, {}, {}, np.inf, {}, {}
        takes._fold_watch(film)

        def measures():
            sim["paper"].entity.update_probe(sim.state, sim.step_count)
            snapshot = Snapshot(sim.make_context())
            return float(crease_angle(snapshot, "paper", line)["angle"]), float(
                cloth_fold(snapshot.ctx, "paper", axis=0)[0]
            )

        for phase, script, mark in takes.fold_drive():
            for _ in script(film):
                tick(session, 1)
                points = np.asarray(sim["paper"].particle_q, dtype=np.float64)
                on_box = points[:, 0] >= lay["box_edge"]
                under = np.where(on_box, points[:, 2] - lay["top"], points[:, 2] - bench_top)
                cls.lowest = min(cls.lowest, float(under.min()))
                moved = np.abs(points[base, :2].mean(axis=0) - rest[base, :2].mean(axis=0))
                cls.drift[phase] = np.maximum(cls.drift.get(phase, 0.0), moved)
                forces = solver.shape_contact_forces()
                joint_q = sim.state.joint_q.numpy()
                servo = solver.rigid.mjw_data.qfrc_actuator.numpy()[0]
                for arm in index:
                    push = [float(np.linalg.norm(forces[side].sum(axis=0))) for side in fingers[arm]]
                    whole = float(np.linalg.norm(sum(forces[side].sum(axis=0) for side in fingers[arm])))
                    cls.pushes.setdefault((phase, arm), []).append((max(push), whole))
                    cls.finger_q.setdefault((phase, arm), []).append(float(joint_q[driven[arm]]))
                    cls.servo.setdefault((phase, arm), []).append(abs(float(servo[actuator[arm]])))
            if mark:
                cls.marks[mark] = measures()
        cls.watch = film.fold_watch
        cls.points = points = np.asarray(sim["paper"].particle_q, dtype=np.float64)
        near, far = rest[:, 0] < rest[:, 0].min() + 0.003, rest[:, 0] > rest[:, 0].max() - 0.003
        cls.edges = [
            float(points[near & (rest[:, 1] < 0), 0].mean() - points[far & (rest[:, 1] < 0), 0].mean()),
            float(points[near & (rest[:, 1] > 0), 0].mean() - points[far & (rest[:, 1] > 0), 0].mean()),
        ]
        cls.effort = sim.model.joint_effort_limit.numpy()[
            [sim.model.joint_qd_start.numpy()[labels.index(f"{a}_left_finger")] for a in index]
        ]
        print(
            f"\n[fold] arms {cls.watch['gap'] * 1e3:.1f} mm apart at the closest, reaching "
            f"{cls.watch['left_y'] * 1e3:+.1f}/{cls.watch['right_y'] * 1e3:+.1f} mm across the middle, turns "
            f"{cls.watch['turn']}, edges {np.round(np.array(cls.edges) * 1e3, 1)} mm, marks {cls.marks}, lowest "
            f"{cls.lowest * 1e3:.2f} mm, drift { ({k: np.round(v * 1e3, 1).tolist() for k, v in cls.drift.items()}) }, "
            f"{cls.ms_per_step:.1f} ms",
            flush=True,
        )

    @staticmethod
    def _finger_actuators(rigid) -> dict[str, int]:
        """Each hand's driven finger's DOF in MuJoCo's own model, where its actuator's force is read."""
        import mujoco

        mj_model = rigid.mj_model
        out = {}
        for i in range(mj_model.nu):
            joint = int(mj_model.actuator_trnid[i, 0])
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
            for arm in ("left", "right"):
                if name.endswith(f"{arm}_left_finger"):
                    out[arm] = int(mj_model.jnt_dofadr[joint])
        return out

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_scene_steps_captured_inside_its_budget(self) -> None:
        self.assertTrue(self.sim.pipeline.captured, self.sim.pipeline.capture_error)
        self.assertLess(self.ms_per_step, 60.0, f"{self.ms_per_step:.1f} ms per control step")

    def test_the_arms_collide_and_never_come_within_20_mm(self) -> None:
        self.assertTrue(self.sim.scene.embodiment.spec.cross_arm_collisions)
        self.assertGreaterEqual(self.watch["gap"], GAP, f"the arms came {self.watch['gap'] * 1e3:.1f} mm apart")

    def test_each_arm_stays_on_its_own_side_of_the_middle(self) -> None:
        self.assertGreaterEqual(self.watch["left_y"], 0.0, f"the left arm reached {self.watch['left_y'] * 1e3:.1f} mm")
        self.assertLessEqual(self.watch["right_y"], 0.0, f"the right arm reached {self.watch['right_y'] * 1e3:.1f} mm")

    def test_a_hand_turns_at_most_90_degrees_while_it_holds(self) -> None:
        for arm in ("left", "right"):
            turn = self.watch["turn"].get(arm, 0.0)
            self.assertGreater(turn, 80.0, f"the {arm} hand carried the edge over")
            self.assertLessEqual(turn, 90.0, f"the {arm} hand turned {turn:.1f} degrees holding it")

    def test_the_grip_stalls_on_the_sheet_at_its_cap(self) -> None:
        """Close drives the fingers to shut; the sheet's push, fed back along their slides, stops them short, and
        the servo asking 100 kN/m of them saturates at its 20 N cap: 10 N a finger, the driven one and its twin."""
        np.testing.assert_allclose(self.effort, 20.0)
        for arm in ("left", "right"):
            servo = self.servo[("pinched", arm)] + self.servo[("carry", arm)]
            self.assertGreater(float(np.median(servo)), 19.9, f"the {arm} grip binds at its cap")
            self.assertLessEqual(max(servo), 20.0 + 1.0e-3, f"and never passes it ({max(servo):.2f} N)")
            held = self.finger_q[("pinched", arm)] + self.finger_q[("carry", arm)]
            self.assertGreater(float(np.median(held)), 0.0, f"the {arm} fingers stop on the sheet, short of shut")
            self.assertGreater(np.median([p for p, _ in self.pushes[("carry", arm)]]), 1.0, "the pads hold the edge")

    def test_only_the_creasing_fingertip_touches_the_sheet_while_it_is_creased(self) -> None:
        self.assertEqual(0.0, max(p for p, _ in self.pushes[("sweep_right", "left")]), "the left hand touched it")
        self.assertEqual(0.0, max(p for p, _ in self.pushes[("sweep_left", "right")]), "the right hand touched it")
        for arm in ("left", "right"):
            pressed = [w for _, w in self.pushes[(f"sweep_{arm}", arm)] if w > 1.0]
            self.assertGreater(len(pressed), 500, f"the {arm} fingertip pressed the fold")
            self.assertLess(
                np.median(pressed), self.takes.FOLD_PUSH + 4.0, f"and pressed about {self.takes.FOLD_PUSH} N"
            )

    def test_nothing_is_pinned(self) -> None:
        entity = self.sim["paper"].entity
        lo, hi = entity.range.particles
        self.assertEqual(0, int((self.sim.model.particle_mass.numpy()[lo:hi] == 0.0).sum()))

    def test_the_edges_are_laid_on_each_other(self) -> None:
        for name, gap in zip(("right", "left"), self.edges, strict=True):
            self.assertLess(abs(gap), 0.010, f"the {name} corners ended {gap * 1e3:.1f} mm apart")

    def test_the_creased_fold_stays_once_the_hands_have_gone(self) -> None:
        gone, later = self.marks["gone"][0], self.marks["plus3s"][0]
        self.assertGreaterEqual(gone, self.takes.FOLD_CREASE, f"the fold kept {gone:.3f} rad once the hands had gone")
        self.assertLess(abs(later - gone), 0.02, f"and {later:.3f} three seconds later")
        self.assertGreater(gone, self.marks["released"][0] + 0.3, "the strokes, not the lay alone, creased it")

    def test_the_sheet_lies_folded_in_half_where_it_was_laid(self) -> None:
        extent = self.marks["plus3s"][1]
        self.assertLess(extent, 0.6, f"its span is {extent:.3f} of the flat sheet's")
        self.assertGreater(extent, 0.45, "half over, not more")
        # A particle's centre at the surface is the sheet half sunk into it (its contact radius is 0.5 mm); a
        # creasing press has put one 1.6 um under, which is the solve's tolerance, not the sheet going through.
        self.assertGreaterEqual(
            self.lowest, -5.0e-5, f"a particle went {-self.lowest * 1e3:.3f} mm under the surface it lay on"
        )
        worst = max(float(v.max()) for v in self.drift.values())
        self.assertLess(worst, 0.015, f"the far half slid {worst * 1e3:.1f} mm")
        self.assertTrue(self.sim.state_is_finite())


if __name__ == "__main__":
    unittest.main()
