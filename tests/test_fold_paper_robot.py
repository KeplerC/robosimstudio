# SPDX-License-Identifier: Apache-2.0
"""The paper fold example's headline task, done by both of the robot's hands the way its take plays it.

``examples/03_materials/05_paper`` promises a gesture and a number. This plays the FOLD take's own drive
(``scripts/recording/takes.py``'s :func:`fold_drive`) through the example's arm panels, a control step per
step, with no browser, and reads the same ``crease_angle`` the page shows. Nothing is pinned to a hand: each
hand presses **Close** on the overhanging edge from its own corner and its fingers stall on the sheet at the
grip's 10 N cap, the contact forces fed back along each finger's slide; the two carry the edge up to upright
turning 88 degrees and on over the far half, and open over it; then each runs its lower finger along its half
of the fold, the sheet pushing that finger up its slide against the same 10 N cap.

What it holds the drive to is what made the earlier folds unrealistic: the arms passing into each other, a
hand turned 180 degrees holding the edge, the edge clamped while the other hand stamped the fold, a position
vise the sheet read 30 to 120 N in, a 20 N grip the sheet pried open and slipped out of before **Open**, and a
fingertip that ran under the fold and lifted it. Here the arms, which collide with each other, keep 20 mm
apart and each to its own side of the middle at every control step; a hand turns at most 90 degrees while it
holds; no finger of either hand pushes more than 15 N at any control step, holding, letting go or pressing;
the pads still hold the edge, nearly shut, when **Open** is pressed; nothing but the pressing finger and the
box touches the sheet while it is creased, and the strokes lift no part of it; the edges end within 10 mm of
each other and the fold keeps a 2.8 rad crease once the hands have gone, the sheet where it was laid.
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
"""One GPU: the example built once and its take's drive played through its page (about 10,500 control steps)."""

FOLD_PORT = 9734
GAP = 0.020
"""The closest the two arms' colliders may come [m]."""
CAP = 10.0
"""The example's grip cap [N]: the most one hand's motor pushes along its driven finger's slide; the other finger
follows that one through a joint equality."""
FINGER = 15.0
"""The most any one finger may push on the sheet, holding it or pressing it [N]: the realism ceiling."""


class TestTheHandsFoldTheSheet(unittest.TestCase):
    """Pinch the overhang from both corners, carry it over, open, press the fold: it stays folded in half."""

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
        actuator = cls._finger_actuators(solver.rigid)
        bench_top = float(film.bench((0.0, 0.0, 0.0))[2])
        cls.rest = rest = np.asarray(sim["paper"].particle_q, dtype=np.float64)
        base = rest[:, 0] > lay["fold"] + 0.0106
        cls.pushes, cls.servo, cls.gap, cls.lowest, cls.drift, cls.marks = {}, {}, {}, np.inf, {}, {}
        cls.highest, cls.others, cls.idle = {}, {}, {}
        from robosimstudio_materials.cloth import contact_shapes

        box = contact_shapes(sim, "box")
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
                cls.highest.setdefault(phase, []).append(float(points[:, 2].max() - lay["top"]))
                moved = np.abs(points[base, :2].mean(axis=0) - rest[base, :2].mean(axis=0))
                cls.drift[phase] = np.maximum(cls.drift.get(phase, 0.0), moved)
                forces = solver.shape_contact_forces()
                servo = solver.rigid.mjw_data.qfrc_actuator.numpy()[0]
                for arm in index:
                    push = [float(np.linalg.norm(forces[side].sum(axis=0))) for side in fingers[arm]]
                    whole = float(np.linalg.norm(sum(forces[side].sum(axis=0) for side in fingers[arm])))
                    cls.pushes.setdefault((phase, arm), []).append((max(push), whole))
                    if phase == f"sweep_{arm}":
                        rest_of_it = np.ones(len(forces), dtype=bool)
                        rest_of_it[np.concatenate([*fingers[arm], box])] = False
                        other = float(np.linalg.norm(forces[rest_of_it], axis=1).max(initial=0.0))
                        cls.others[phase] = max(cls.others.get(phase, 0.0), other)
                        cls.idle[phase] = max(cls.idle.get(phase, 0.0), min(push))
                    cls.servo.setdefault((phase, arm), []).append(abs(float(servo[actuator[arm]])))
                    if phase in ("pinched", "carry"):
                        centres, _, normals = takes._fold_pads(film, arm)
                        apart = abs(float(np.dot(centres[1] - centres[0], normals[0]))) - 0.004  # face to face
                        cls.gap.setdefault(arm, []).append(apart)
            if mark:
                cls.marks[mark] = measures()
        cls.watch = film.fold_watch
        cls.points = points = np.asarray(sim["paper"].particle_q, dtype=np.float64)
        near, far = rest[:, 0] < rest[:, 0].min() + 0.003, rest[:, 0] > rest[:, 0].max() - 0.003
        cls.edges = [
            float(points[near & (rest[:, 1] < 0), 0].mean() - points[far & (rest[:, 1] < 0), 0].mean()),
            float(points[near & (rest[:, 1] > 0), 0].mean() - points[far & (rest[:, 1] > 0), 0].mean()),
        ]
        labels = [str(n).rsplit("/", 1)[-1] for n in sim.model.joint_label]
        cls.effort = sim.model.joint_effort_limit.numpy()[
            [sim.model.joint_qd_start.numpy()[labels.index(f"{a}_left_finger")] for a in index]
        ]
        print(
            f"\n[fold] arms {cls.watch['gap'] * 1e3:.1f} mm apart at the closest, reaching "
            f"{cls.watch['left_y'] * 1e3:+.1f}/{cls.watch['right_y'] * 1e3:+.1f} mm across the middle, turns "
            f"{cls.watch['turn']}, edges {np.round(np.array(cls.edges) * 1e3, 1)} mm, marks {cls.marks}, lowest "
            f"{cls.lowest * 1e3:.2f} mm, drift { ({k: np.round(v * 1e3, 1).tolist() for k, v in cls.drift.items()}) }, "
            f"pushes { ({k: round(max(p for p, _ in v), 1) for k, v in cls.pushes.items()}) }, pad gap "
            f"{ ({k: (round(min(v) * 1e3, 2), round(max(v) * 1e3, 1)) for k, v in cls.gap.items()}) } mm, "
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
        the servo asking 100 kN/m of them saturates at its 10 N cap, which one motor spends on both fingers."""
        np.testing.assert_allclose(self.effort, CAP)
        for arm in ("left", "right"):
            pinched = self.servo[("pinched", arm)]
            self.assertGreater(float(np.median(pinched)), CAP - 0.1, f"the {arm} grip binds at its cap")
            held = pinched + self.servo[("carry", arm)]
            # MuJoCo clamps the actuator to its forcerange, so this holds for as long as effort_limit reaches it.
            self.assertLessEqual(max(held), CAP + 1.0e-3, f"and never passes it ({max(held):.2f} N)")
            self.assertGreater(np.median([p for p, _ in self.pushes[("carry", arm)]]), 1.0, "the pads hold the edge")
            # On the sheet, not on each other: the pads' faces stay apart by the sheet between them.
            narrowest = min(self.gap[arm])
            self.assertGreater(narrowest, 0.0005, f"the {arm} pads closed to {narrowest * 1e3:.2f} mm, shut")

    def test_no_finger_ever_pushes_more_than_15_n(self) -> None:
        """Each finger's own contact force, its friction included, at every control step of every phase, the let-go
        and the strokes among them: at most 15 N, not one step over."""
        for (phase, arm), rows in self.pushes.items():
            worst = max(p for p, _ in rows)
            self.assertLessEqual(worst, FINGER, f"a {arm} finger pushed {worst:.1f} N in the {phase}")

    def test_the_pads_still_pinch_the_edge_when_open_is_pressed(self) -> None:
        """The edge is let go, not lost: the pads touch it on the carry's last step and stood nearly shut on it."""
        for arm in ("left", "right"):
            last = self.pushes[("carry", arm)][-1][1]
            self.assertGreater(last, 1.0, f"the {arm} pads held {last:.2f} N of the edge as Open was pressed")
            widest = max(self.gap[arm])
            self.assertLess(widest, 0.010, f"the {arm} pads stood {widest * 1e3:.1f} mm apart on the sheet")

    def test_only_the_pressing_finger_touches_the_sheet_while_it_is_creased(self) -> None:
        """While a hand creases, the sheet feels its pressing finger and the box and nothing else: not the other hand,
        not the pressing hand's other finger, no other link of either arm, not the bench. Read, like every force
        here, off the coupler's records of the last substep of each control step."""
        for arm in ("left", "right"):
            phase = f"sweep_{arm}"
            self.assertLessEqual(self.others[phase], 1.0e-3, f"something else pushed the sheet in the {phase}")
            self.assertLessEqual(self.idle[phase], 1.0e-3, f"the {arm} hand's other finger pushed it too")
            pressed = [i for i, (_, w) in enumerate(self.pushes[(f"sweep_{arm}", arm)]) if w > 1.0]
            self.assertGreater(len(pressed), 500, f"the {arm} finger pressed the fold")
            servo = [self.servo[(f"sweep_{arm}", arm)][i] for i in pressed]
            self.assertGreater(float(np.median(servo)), CAP - 0.1, f"the {arm} press is its grip's cap")

    def test_the_strokes_lift_no_part_of_the_sheet(self) -> None:
        """The finger runs on top of the fold: nothing of the sheet rises above where it stood when the hands let go."""
        laid = max(self.highest["let_go"][-30:])
        for arm in ("left", "right"):
            top = max(self.highest[f"sweep_{arm}"])
            self.assertLessEqual(top, laid + 0.001, f"the {arm} strokes raised the sheet to {top * 1e3:.1f} mm")

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
        self.assertGreaterEqual(later, self.takes.FOLD_CREASE, f"and {later:.3f} three seconds later")
        self.assertGreater(gone, self.marks["released"][0] + 0.3, "the strokes, not the lay alone, creased it")

    def test_the_sheet_lies_folded_in_half_where_it_was_laid(self) -> None:
        extent = self.marks["plus3s"][1]
        self.assertLess(extent, 0.6, f"its span is {extent:.3f} of the flat sheet's")
        self.assertGreater(extent, 0.45, "half over, not more")
        # A particle's centre at the surface is the sheet half sunk into it (its contact radius is 0.5 mm); a
        # press has put one 1.6 um under, which is the solve's tolerance, not the sheet going through.
        self.assertGreaterEqual(
            self.lowest, -5.0e-5, f"a particle went {-self.lowest * 1e3:.3f} mm under the surface it lay on"
        )
        worst = max(float(v.max()) for v in self.drift.values())
        self.assertLess(worst, 0.015, f"the far half slid {worst * 1e3:.1f} mm")
        self.assertTrue(self.sim.state_is_finite())


if __name__ == "__main__":
    unittest.main()
