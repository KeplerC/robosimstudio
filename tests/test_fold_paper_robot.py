# SPDX-License-Identifier: Apache-2.0
"""The paper fold example's headline task, done by both of the robot's hands the way its take plays it.

``examples/03_materials/05_paper`` promises a gesture and a number. This plays the FOLD take's own
drive (``scripts/recording/takes.py``: ``_fold_reach``, ``_fold_pinch``, ``_fold_carry``,
``_fold_let_go``, ``_fold_sweep``) through the example's arm panels, a control step per step, with no
browser, and reads the same ``crease_angle`` the page shows. Nothing is pinned to a hand: each hand's
pads close on the overhanging edge from its own corner, the two carry it over on matching arcs turning
90 degrees, let go over the far edge and leave; then each runs a closed fingertip along its half of
the fold.

What it holds the drive to is what made the old fold unrealistic: there the arms passed 60 mm into
each other, a hand turned 180 degrees holding the edge, and the edge stayed clamped while the other
hand stamped the fold. Here the two arms, which collide with each other, keep 20 mm apart at every
control step; a hand turns at most 90 degrees while it holds; nothing but the sweeping fingertip and
the box touches the sheet while the fold is creased; the edges end within 10 mm of each other.
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
"""One GPU: the example built once and its take's drive played through its page."""

FOLD_PORT = 9734
GAP = 0.020
"""The closest the two arms' colliders may come [m]."""


def _segments(model):
    """Each arm's colliders as capsules ``(body, a, b, radius)`` in body frames; a box split along its long side."""
    import newton

    from robosimstudio.core.math import quat_to_mat

    shape_body = model.shape_body.numpy()
    labels = [str(b).rsplit("/", 1)[-1] for b in model.body_label]
    flags, kinds = model.shape_flags.numpy(), model.shape_type.numpy()
    scale, xform = model.shape_scale.numpy(), model.shape_transform.numpy()
    out = {"left": [], "right": []}
    for s in range(model.shape_count):
        b = int(shape_body[s])
        if b < 0 or not int(flags[s]) & int(newton.ShapeFlags.COLLIDE_SHAPES):
            continue
        side = next((a for a in out if labels[b].startswith(a + "_")), None)
        if side is None:
            continue
        rot, at, sc, kind = quat_to_mat(xform[s][3:7]), xform[s][:3], scale[s], int(kinds[s])
        if kind == int(newton.GeoType.CAPSULE):
            pieces = [(np.array([0, 0, -sc[1]]), np.array([0, 0, sc[1]]), sc[0])]
        elif kind == int(newton.GeoType.SPHERE):
            pieces = [(np.zeros(3), np.zeros(3), sc[0])]
        elif kind == int(newton.GeoType.BOX):
            h = np.asarray(sc[:3], dtype=np.float64)
            thin, mid, long = np.argsort(h)
            n = int(np.ceil(max(h[mid] - h[thin], 0) / (h[thin] / 2))) + 1
            pieces = []
            for off in np.linspace(-(h[mid] - h[thin]), h[mid] - h[thin], n) if n > 1 else [0.0]:
                p0, p1 = np.zeros(3), np.zeros(3)
                p0[long], p1[long] = -(h[long] - h[thin]), h[long] - h[thin]
                p0[mid] = p1[mid] = off
                pieces.append((p0, p1, h[thin]))
        else:
            raise AssertionError(f"{labels[b]} has a collider of type {kind}")
        out[side] += [(b, at + rot @ p0, at + rot @ p1, float(r)) for p0, p1, r in pieces]
    return out


def _segment_distance(p1, q1, p2, q2):
    """Closest distance between segments, over broadcast leading axes (Ericson 5.1.9)."""
    d1, d2, rr = q1 - p1, q2 - p2, p1 - p2
    a, e, f = (d1 * d1).sum(-1), (d2 * d2).sum(-1), (d2 * rr).sum(-1)
    c, b = (d1 * rr).sum(-1), (d1 * d2).sum(-1)
    den = a * e - b * b
    sa, se = np.where(a > 1e-12, a, 1.0), np.where(e > 1e-12, e, 1.0)
    s = np.where(den > 1e-12, np.clip((b * f - c * e) / np.where(den > 1e-12, den, 1.0), 0, 1), 0.0)
    t = (b * s + f) / se
    s = np.where(t < 0, np.clip(-c / sa, 0, 1), np.where(t > 1, np.clip((b - c) / sa, 0, 1), s))
    t = np.clip(t, 0, 1)
    s, t = np.where(a > 1e-12, s, 0.0), np.where(a > 1e-12, t, np.clip(f / se, 0, 1))
    t = np.where(e > 1e-12, t, 0.0)
    s = np.where(e > 1e-12, s, np.where(a > 1e-12, np.clip(-c / sa, 0, 1), 0.0))
    return np.linalg.norm(p1 + d1 * s[..., None] - p2 - d2 * t[..., None], axis=-1)


def _arm_gap(segments, body_q) -> float:
    """The smallest distance between any left-arm and any right-arm collider surface [m]."""
    from robosimstudio.core.math import quat_to_mat

    def world(side):
        rows = []
        for b, a, c, r in segments[side]:
            rot = quat_to_mat(body_q[b][3:7])
            rows.append((body_q[b][:3] + rot @ a, body_q[b][:3] + rot @ c, r))
        return [np.array([row[i] for row in rows]) for i in range(3)]

    la, lb, lr = world("left")
    ra, rb, rr = world("right")
    d = _segment_distance(la[:, None], lb[:, None], ra[None], rb[None]) - lr[:, None] - rr[None]
    return float(d.min())


class TestTheHandsFoldTheSheet(unittest.TestCase):
    """Pinch the overhang from both corners, carry it over together, let go, crease it: it stays folded in half."""

    @classmethod
    def setUpClass(cls) -> None:
        from robosimstudio.assetlib.semantics.snapshot import Snapshot
        from robosimstudio.assetlib.vocabulary.measures import cloth_fold
        from robosimstudio_materials.paper import crease_angle

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from recording import takes
        from recording.lockstep import Film

        cls.sim, kwargs = _run("03_materials/05_paper.py")
        sim = cls.sim
        solver = sim.pipeline.program.solver
        solver.enable_contact_forces()  # before the first step captures the graph
        sim.run(10).sync()
        t = time.perf_counter()
        sim.run(60).sync()
        cls.ms_per_step = 1e3 * (time.perf_counter() - t) / 60
        cls.session = session = open_session(sim, FOLD_PORT, kwargs.get("page"))
        film = Film(takes.find("03_materials/05_paper"), port=0, scratch=None, software=False, probe=False, log=print)
        film.sim, film.session = sim, session
        lay = takes._fold_layout(film)
        line = {"crease": {"axis": 0, "at": lay["line"]}}
        segments = _segments(sim.model)
        index = sim.scene.embodiment.arms
        fingers = {
            arm: [np.flatnonzero(np.isin(sim.model.shape_body.numpy(), side)) for side in index[arm].finger_bodies]
            for arm in ("left", "right")
        }
        wrist = {arm: sim.arm(arm).ee_body for arm in ("left", "right")}
        cls.gaps, cls.turns, cls.pushes, cls.lowest = [], {"left": [0.0], "right": [0.0]}, {}, np.inf
        held_from = {}
        bench_top = float(film.bench((0.0, 0.0, 0.0))[2])

        def measures():
            sim["paper"].entity.update_probe(sim.state, sim.step_count)
            snapshot = Snapshot(sim.make_context())
            return float(crease_angle(snapshot, "paper", line)["angle"]), float(
                cloth_fold(snapshot.ctx, "paper", axis=0)[0]
            )

        def play(script, phase):
            for _ in script:
                tick(session, 1)
                body_q = sim.state.body_q.numpy()
                cls.gaps.append(_arm_gap(segments, body_q))
                points = np.asarray(sim["paper"].particle_q, dtype=np.float64)
                on_box = points[:, 0] >= lay["box_edge"]
                under = np.where(on_box, points[:, 2] - lay["top"], points[:, 2] - bench_top)
                cls.lowest = min(cls.lowest, float(under.min()))
                forces = solver.shape_contact_forces()
                for arm in ("left", "right"):
                    push = [float(np.linalg.norm(forces[side].sum(axis=0))) for side in fingers[arm]]
                    cls.pushes.setdefault((phase, arm), []).append(max(push))
                    holding = abs(session.arms[arm].gripper_command - takes.FOLD_GRIP) < 1e-6
                    if holding:
                        q = body_q[wrist[arm]][3:7]
                        q0 = held_from.setdefault(arm, q)
                        cls.turns[arm].append(np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(q0, q)))))))

        tick(session, 78)  # the take's warm-up and opening wait
        cls.rest = np.asarray(sim["paper"].particle_q, dtype=np.float64)
        play(takes._fold_reach(film), "reach")
        play(takes._fold_pinch(film), "pinch")
        play((None for _ in range(96)), "pinched")
        play(takes._fold_carry(film), "carry")
        play(takes._fold_let_go(film), "let go")
        cls.released = measures()
        play(takes._fold_sweep("right")(film), "right sweeps")
        play(takes._fold_sweep("left")(film), "left sweeps")
        cls.gone = measures()
        cls.after = []
        for _ in range(6):
            play((None for _ in range(30)), "watch")
            cls.after.append(measures())
        cls.points = points = np.asarray(sim["paper"].particle_q, dtype=np.float64)
        rest = cls.rest
        near, far = rest[:, 0] < rest[:, 0].min() + 0.003, rest[:, 0] > rest[:, 0].max() - 0.003
        cls.edges = [
            float(points[near & (rest[:, 1] < 0), 0].mean() - points[far & (rest[:, 1] < 0), 0].mean()),
            float(points[near & (rest[:, 1] > 0), 0].mean() - points[far & (rest[:, 1] > 0), 0].mean()),
        ]
        print(
            f"\n[fold] arms {min(cls.gaps) * 1e3:.1f} mm apart at the closest, turns {max(cls.turns['right']):.1f} "
            f"{max(cls.turns['left']):.1f} deg, edges {np.round(np.array(cls.edges) * 1e3, 1)} mm, released "
            f"{cls.released}, gone {cls.gone}, after {cls.after[-1]}, lowest {cls.lowest * 1e3:.2f} mm, "
            f"{cls.ms_per_step:.1f} ms",
            flush=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_scene_steps_captured_inside_its_budget(self) -> None:
        self.assertTrue(self.sim.pipeline.captured, self.sim.pipeline.capture_error)
        self.assertLess(self.ms_per_step, 60.0, f"{self.ms_per_step:.1f} ms per control step")

    def test_the_arms_never_come_within_20_mm(self) -> None:
        self.assertGreaterEqual(min(self.gaps), GAP, f"the arms came {min(self.gaps) * 1e3:.1f} mm apart")

    def test_a_hand_turns_at_most_90_degrees_while_it_holds(self) -> None:
        for arm, turns in self.turns.items():
            self.assertGreater(len(turns), 100, f"the {arm} hand held the sheet")
            self.assertLessEqual(max(turns), 90.0, f"the {arm} hand turned {max(turns):.1f} degrees holding it")

    def test_the_pads_carry_the_edge_by_contact(self) -> None:
        for arm in ("left", "right"):
            self.assertGreater(np.median(self.pushes[("carry", arm)]), 1.0, f"the sheet pushes the {arm} pads")
        self.assertGreater(self.released[0], 0.5, "the carry bent the flap over")

    def test_only_the_sweeping_fingertip_touches_the_sheet_while_it_is_creased(self) -> None:
        self.assertEqual(0.0, max(self.pushes[("right sweeps", "left")]), "the left hand touched the sheet")
        self.assertEqual(0.0, max(self.pushes[("left sweeps", "right")]), "the right hand touched the sheet")
        self.assertGreater(max(self.pushes[("right sweeps", "right")]), 1.0, "the right fingertip pressed the fold")
        self.assertGreater(max(self.pushes[("left sweeps", "left")]), 1.0, "the left fingertip pressed the fold")

    def test_the_edges_are_laid_on_each_other(self) -> None:
        for name, gap in zip(("right", "left"), self.edges, strict=True):
            self.assertLess(abs(gap), 0.010, f"the {name} corners ended {gap * 1e3:.1f} mm apart")

    def test_the_swept_fold_stays(self) -> None:
        angles = [a for a, _ in self.after]
        self.assertGreaterEqual(min(angles), 2.2, f"the fold kept {np.round(angles, 3)} rad after the hands left")
        self.assertLess(max(angles) - min(angles), 0.05, "and does not relax")
        self.assertGreater(min(angles), self.released[0] + 0.8, "the sweep, not the carry, creased it")

    def test_the_sheet_lies_folded_in_half_on_the_box(self) -> None:
        extent = self.after[-1][1]
        self.assertLess(extent, 0.6, f"its span is {extent:.3f} of the flat sheet's")
        self.assertGreater(extent, 0.45, "half over, not more")
        self.assertGreaterEqual(
            self.lowest, 0.0, f"a particle went {-self.lowest * 1e3:.2f} mm under the surface it lay on"
        )
        self.assertTrue(self.sim.state_is_finite())


if __name__ == "__main__":
    unittest.main()
