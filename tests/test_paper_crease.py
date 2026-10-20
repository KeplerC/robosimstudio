# SPDX-License-Identifier: Apache-2.0
"""``robosimstudio_materials/paper.py``: a hinge that yields, in VBD's own angle convention.

Three claims, two of them cheap:

* **The kernel.** One hinge of four particles bent to 90 degrees with a yield of 0.2 rad and
  rate 1 leaves a rest angle of ``pi/2 - 0.2`` after one launch; with a yield of 0 it leaves
  the rest angle alone. The angle it reads is the one Newton computes: the builder's
  ``add_edge`` rest angle and a line-for-line port of the VBD bending kernel agree with it on
  several poses, sign included. CPU warp, no GPU.
* **The entity.** A sheet builds on the CPU, its hinges are found, the plastic sliders land
  in the kernel's array, and the flat names resolve. A CPU scene refuses paper on ``vbd`` by
  name, so the sheet names ``newton.solvers.SolverVBD`` by path, the slow way it runs there.
* **The fold.** On the GPU: a strip of PAPER pinned at one end, its free end carried over 180
  degrees by a pin and held, the standing loop pressed flat by a third pin, everything released.
  The creased strip stays folded and its fold line reads as a crease; the same strip with
  ``yield_angle`` 0 springs open; a reset flattens the crease. The press is what makes a fold a
  crease, in the sim as on a desk: carried over and merely laid down, the loop stands 30 mm high,
  three hinge columns share its turn, the line reads 1.5 rad and the strip springs open.
  And the array the kernel writes is the one the sheet's solver bends against, under the
  one-way coupler and under Newton's coupled solver alike.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

TIER = "medium"
"""The hinge and entity classes run on the CPU; the fold classes need a GPU and build scenes."""


def _hinge(angle: float) -> np.ndarray:
    """Four points: an edge along y, one opposite vertex flat at -x, the other folded up by ``angle``."""
    return np.array(
        [
            [0.0, 0.0, 0.0],  # v0, edge start
            [0.0, 1.0, 0.0],  # v1, edge end
            [-1.0, 0.5, 0.0],  # opposite 0
            [math.cos(angle), 0.5, math.sin(angle)],  # opposite 1
        ],
        dtype=np.float32,
    )


ROW = np.array([[2, 3, 0, 1]], dtype=np.int32)
"""The edge row, ``(opposite0, opposite1, v0, v1)`` as the builder lays it out."""


def _vbd_angle(pos: np.ndarray, row: np.ndarray) -> float:
    """The first lines of Newton's ``evaluate_dihedral_angle_based_bending_force_hessian``, ported."""
    x0, x1, x2, x3 = (pos[int(i)].astype(np.float64) for i in row)
    x02, x03, x13, x12, e = x2 - x0, x3 - x0, x3 - x1, x2 - x1, x3 - x2
    n1, n2 = np.cross(x02, x03), np.cross(x13, x12)
    n1_hat, n2_hat, e_hat = n1 / np.linalg.norm(n1), n2 / np.linalg.norm(n2), e / np.linalg.norm(e)
    return math.atan2(float(np.dot(np.cross(n1_hat, n2_hat), e_hat)), float(np.dot(n1_hat, n2_hat)))


def _launch(pos: np.ndarray, *, rest: float, params: tuple[float, float, float], dt: float = 0.0) -> float:
    """One launch of the crease kernel over one hinge, on the CPU. Returns the rest angle after."""
    import warp as wp

    from robosimstudio_materials.paper import _crease_kernel

    device = "cpu"
    rows = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device)
    edges = wp.array(ROW, dtype=wp.int32, device=device)
    prm = wp.array(np.array(params, dtype=np.float32), dtype=wp.float32, device=device)
    q = wp.array(pos, dtype=wp.vec3, device=device)
    rest_wp = wp.array(np.array([rest], dtype=np.float32), dtype=wp.float32, device=device)
    wp.launch(_crease_kernel, dim=1, inputs=[rows, edges, prm, float(dt), q], outputs=[rest_wp], device=device)
    return float(rest_wp.numpy()[0])


class TestTheHingeKernel(unittest.TestCase):
    """One hinge, on the CPU."""

    def test_a_right_angle_past_a_yield_of_a_fifth_keeps_all_but_the_yield(self) -> None:
        theta = _vbd_angle(_hinge(math.pi / 2), ROW[0])
        self.assertAlmostEqual(abs(theta), math.pi / 2, places=5)
        got = _launch(_hinge(math.pi / 2), rest=0.0, params=(0.2, 1.0, 0.0))
        self.assertAlmostEqual(abs(got), math.pi / 2 - 0.2, places=5)
        self.assertEqual(np.sign(got), np.sign(theta), "the rest angle moves toward the bend, on the bend's side")

    def test_a_yield_of_zero_is_the_off_switch(self) -> None:
        self.assertEqual(0.0, _launch(_hinge(math.pi / 2), rest=0.0, params=(0.0, 1.0, 0.0)))
        self.assertAlmostEqual(0.3, _launch(_hinge(math.pi / 2), rest=0.3, params=(0.0, 1.0, 0.0)), places=6)

    def test_below_the_yield_nothing_is_kept(self) -> None:
        self.assertEqual(0.0, _launch(_hinge(0.15), rest=0.0, params=(0.2, 1.0, 0.0)))

    def test_the_rate_scales_the_step(self) -> None:
        got = _launch(_hinge(math.pi / 2), rest=0.0, params=(0.2, 0.25, 0.0))
        self.assertAlmostEqual(abs(got), 0.25 * (math.pi / 2 - 0.2), places=5)

    def test_recovery_relaxes_the_rest_angle_toward_flat(self) -> None:
        # A yield wide enough that nothing plastic happens, so only the recovery term acts.
        got = _launch(_hinge(0.0), rest=1.0, params=(3.0, 1.0, 2.0), dt=0.1)
        self.assertAlmostEqual(0.8, got, places=5)

    def test_a_fold_past_flat_on_itself_continues_the_same_crease(self) -> None:
        """At a dihedral near +-pi the sign flips with contact noise; the excess is wrapped, so a
        crease at 2.8 rad reading -3.0 rad moves on toward pi rather than to a mirrored -2.8."""
        theta = _vbd_angle(_hinge(-3.0), ROW[0])
        self.assertGreater(theta, 2.9, "folding opposite 1 by -3 rad reads +3 rad in this convention")
        # The crease was written at -2.8, the same fold read before the sign flipped across pi.
        got = _launch(_hinge(-3.0), rest=-2.8, params=(0.2, 1.0, 0.0))
        self.assertLess(got, -2.8, "it continues toward -pi")
        self.assertGreater(got, -math.pi - 0.1)
        self.assertAlmostEqual(-2.8 + ((3.0 + 2.8) - 2.0 * math.pi) + 0.2, got, places=4)

    def test_the_angle_is_newtons_on_four_poses(self) -> None:
        """The kernel's angle against the builder's ``add_edge`` rest angle and the VBD port."""
        import newton
        import warp as wp

        from robosimstudio_materials.paper import dihedral_angles

        for angle in (0.0, math.radians(60.0), math.radians(-135.0), math.pi / 2):
            with self.subTest(angle_deg=round(math.degrees(angle))):
                pos = _hinge(angle)
                builder = newton.ModelBuilder()
                for p in pos:
                    builder.add_particle(wp.vec3(*[float(c) for c in p]), wp.vec3(0.0), 1.0)
                i, j, k, m = (int(v) for v in ROW[0])
                builder.add_triangle(i, k, m)
                builder.add_triangle(j, m, k)
                builder.add_edge(i, j, k, m)
                newtons = float(builder.edge_rest_angle[0])
                ported = _vbd_angle(pos, ROW[0])
                mine = dihedral_angles(
                    wp.array(pos, dtype=wp.vec3, device="cpu"), wp.array(ROW, dtype=wp.int32, device="cpu")
                )
                self.assertAlmostEqual(ported, newtons, places=5)
                self.assertAlmostEqual(float(mine[0]), newtons, places=5)
                if angle:
                    self.assertAlmostEqual(abs(float(mine[0])), abs(angle), places=5)


class TestTheEntityOnTheCpu(unittest.TestCase):
    """A sheet built on the CPU: its hinges, its sliders, its names."""

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        physics = rss.GENERIC.with_backend("paper", rss.backend("newton.solvers.SolverVBD"))
        scene = rss.Scene(cell="void", name="paper_cpu", physics=physics)
        scene.add(rss.paper(name="leaf", size=(0.04, 0.04), cells=(4, 4), at=(0.0, 0.0, 0.1), settle_frames=0))
        cls.sim = scene.build(device="cpu")
        cls.entity = cls.sim.scene.by_name["leaf"]

    def test_the_flat_names_resolve_to_the_pack(self) -> None:
        import robosimstudio as rss
        from robosimstudio_materials.paper import PAPER, papers

        for name in ("paper", "PaperSpec", "PaperMaterialSpec"):
            self.assertEqual("robosimstudio_materials.paper", getattr(rss, name).__module__, name)
        self.assertIs(rss.papers, papers)
        self.assertIs(PAPER, rss.PaperSpec().material)
        self.assertEqual("paper", rss.paper().entity)

    def test_the_sheet_owns_every_interior_hinge(self) -> None:
        # A 4 x 4 grid of right triangles has 2 * 4 * 4 = 32 triangles and 56 edges, 40 of them interior.
        model = self.sim.model
        self.assertEqual(56, len(self.entity.rows))
        self.assertEqual(model.edge_count, len(self.entity.rows))
        np.testing.assert_array_equal(np.zeros(56), self.entity.crease_angles())
        self.assertAlmostEqual(0.01, self.entity.cell_size(), places=3)
        self.assertEqual((56, 3), self.entity.hinge_midpoints().shape)

    def test_the_params_array_holds_the_material_and_takes_a_slider(self) -> None:
        from robosimstudio_materials.paper import PAPER

        np.testing.assert_allclose([0.35, 1.0, 0.0], self.entity._params.numpy(), rtol=1e-6)
        self.assertEqual(PAPER.yield_angle, 0.35)
        (material,) = self.entity.tunables()
        self.assertTrue(material.write("yield_angle", 0.0))
        self.assertTrue(material.write("plastic_recovery", 0.5))
        self.assertTrue(material.write("edge_ke", 12.0), "cloth's own fields still go through LiveMaterial")
        self.assertFalse(material.write("not_a_field", 1.0))
        np.testing.assert_allclose([0.0, 1.0, 0.5], self.entity._params.numpy(), rtol=1e-6)
        material.write("yield_angle", 0.35)

    def test_it_steps_and_the_crease_atom_reads_zero(self) -> None:
        from robosimstudio.assetlib.semantics.snapshot import Snapshot
        from robosimstudio_materials.paper import crease_angle

        self.sim.run(3).sync()
        self.assertTrue(self.sim.state_is_finite())
        got = crease_angle(Snapshot(self.sim.read()), "leaf", {"crease": {"axis": 0, "at": 0.02}})
        self.assertEqual(0.0, got["angle"])
        self.assertGreater(got["edges"], 0)
        self.assertLess(got["edges"], 56)

    def test_the_domain_reads_as_cloth_too(self) -> None:
        from robosimstudio.specs.physics import get_domain

        self.assertEqual(("paper", "cloth"), get_domain("paper").reads)
        self.assertEqual("deformables", get_domain("paper").role)


# ---------------------------------------------------------------------------
# The fold, on the GPU
# ---------------------------------------------------------------------------

LENGTH, WIDTH = 0.2, 0.05
CELL = 0.01
CARRY, HOLD, PRESS, FREE = 90, 30, 30, 120
PRESS_TO = 0.004
"""How far above the bench the loop's crest is pressed [m]. At 0.002 the two layers are forced
through each other and the elastic control stays locked shut; at 0.004 and 0.008 it springs open."""


def _fold_a_strip(yield_angle: float) -> dict:
    """Pin one end of a strip of PAPER, carry the other over 180 degrees, hold, release.

    Returns the fold ratio and the crease along the fold line after the free run, then the same
    after a reset, and whether the kernel's array is the solver's. ``open`` is the distance between
    the strip's two end rows over its length: the sprung elastic strip lands flat but yawed, so its
    span along x says less than the span between its ends.
    """
    import robosimstudio as rss
    from robosimstudio.assetlib.semantics.snapshot import Snapshot
    from robosimstudio.assetlib.vocabulary.measures import cloth_fold
    from robosimstudio.core.draggers import ParticleDragger
    from robosimstudio_materials.paper import PAPER, crease_angle

    material = rss.replace(PAPER, yield_angle=yield_angle)
    scene = rss.Scene(cell="bench_075", physics=rss.GENERIC, name=f"paper_fold_{yield_angle}")
    scene.add(
        rss.paper(
            name="strip",
            size=(LENGTH, WIDTH),
            cells=(round(LENGTH / CELL), round(WIDTH / CELL)),
            at=(-LENGTH / 2, -WIDTH / 2, 0.005),
            material=material,
        )
    )
    sim = scene.build()
    entity = sim.scene.by_name["strip"]
    model = sim.model
    same_array = entity._rest.ptr == model.edge_rest_angle.ptr

    ids = np.asarray(entity.particles, dtype=np.int64)
    rest_q = sim.state.particle_q.numpy().astype(np.float64)[ids]
    root_row = rest_q[:, 0] <= rest_q[:, 0].min() + 1e-6
    tip_row = rest_q[:, 0] >= rest_q[:, 0].max() - 1e-6
    root, tip = rest_q[root_row].mean(axis=0), rest_q[tip_row].mean(axis=0)
    centre = 0.5 * (root + tip)
    radius = 0.5 * float(tip[0] - root[0])

    def positions() -> np.ndarray:
        return sim.state.particle_q.numpy().astype(np.float64)[ids]

    def openness() -> float:
        q = positions()
        return float(np.linalg.norm(q[tip_row].mean(axis=0) - q[root_row].mean(axis=0)) / LENGTH)

    def measure():
        entity.update_probe(sim.state, sim.step_count)
        return Snapshot(sim.read())

    anchor = ParticleDragger(model, radius=0.03, max_particles=64, mode="pin")
    hand = ParticleDragger(model, radius=0.012, max_particles=16, mode="pin")
    up = (0.0, 0.0, 0.0, 1.0)
    anchor.grab(sim.state, root, up, candidates=ids)
    hand.grab(sim.state, tip, up, candidates=ids)
    for dragger in (anchor, hand):
        sim.pipeline.hooks.post_solve.append(dragger.apply)
    sim.pipeline.invalidate_graph()

    for step in range(CARRY):
        a = math.pi * (step + 1) / CARRY
        position = centre + radius * np.array([math.cos(a), 0.0, math.sin(a)])
        hand.set_target(position, (0.0, -math.sin(a / 2), 0.0, math.cos(a / 2)))
        sim.run(1)
    sim.run(HOLD).sync()
    line = {"crease": {"axis": 0, "at": 0.0}}
    folded_ratio = float(cloth_fold(measure().ctx, "strip", axis=0)[0])
    laid = float(crease_angle(measure(), "strip", line)["angle"])

    # Press the standing loop flat, as a finger run along a fold does.
    q = positions()
    crest = q[np.argmax(q[:, 2])].copy()
    crest[1] = root[1]
    press = ParticleDragger(model, radius=0.03, max_particles=16, mode="pin")
    press.grab(sim.state, crest, up, candidates=ids)
    sim.pipeline.hooks.post_solve.append(press.apply)
    sim.pipeline.invalidate_graph()
    floor = np.array([crest[0], crest[1], rest_q[:, 2].min() + PRESS_TO])
    for step in range(PRESS):
        f = (step + 1) / PRESS
        press.set_target(crest * (1.0 - f) + floor * f, up)
        sim.run(1)
    sim.run(HOLD // 2).sync()
    pressed = float(crease_angle(measure(), "strip", line)["angle"])

    for dragger in (hand, anchor, press):
        dragger.release()
    sim.run(FREE).sync()

    snap = measure()
    ratio = float(cloth_fold(snap.ctx, "strip", axis=0)[0])
    crease = crease_angle(snap, "strip", line)
    whole = crease_angle(snap, "strip")
    finite = bool(sim.state_is_finite())
    open_after = openness()

    sim.reset()
    sim.run(5).sync()
    after = measure()
    return {
        "same_array": same_array,
        "folded_ratio": folded_ratio,
        "laid_crease": laid,
        "pressed_crease": pressed,
        "ratio": ratio,
        "open": open_after,
        "crease": float(crease["angle"]),
        "peak": float(crease["peak"]),
        "crease_edges": int(crease["edges"]),
        "whole": float(whole["angle"]),
        "finite": finite,
        "reset_ratio": float(cloth_fold(after.ctx, "strip", axis=0)[0]),
        "reset_open": openness(),
        "reset_crease": float(crease_angle(after, "strip")["angle"]),
        "reset_restores_rest": bool(np.allclose(entity.crease_angles(), entity.rest_angles)),
    }


class TestAFoldedStripStaysFolded(unittest.TestCase):
    """PAPER keeps the fold; the elastic control springs open; a reset flattens the crease."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.paper = _fold_a_strip(0.35)
        cls.elastic = _fold_a_strip(0.0)
        print(f"\n[paper] {cls.paper}\n[elastic] {cls.elastic}", flush=True)

    def test_the_kernel_writes_the_array_the_solver_reads(self) -> None:
        self.assertTrue(self.paper["same_array"])

    def test_the_carry_folded_the_strip(self) -> None:
        for name in ("paper", "elastic"):
            with self.subTest(name):
                self.assertLess(getattr(self, name)["folded_ratio"], 0.7)
                self.assertTrue(getattr(self, name)["finite"])

    def test_paper_stays_folded_and_reads_as_creased(self) -> None:
        self.assertLess(self.paper["ratio"], 0.7)
        self.assertLess(self.paper["open"], 0.5)
        self.assertGreater(self.paper["crease"], 2.2, "the atom's gate")
        self.assertGreater(self.paper["peak"], 1.0)
        self.assertGreater(self.paper["crease_edges"], 0)

    def test_the_press_is_what_makes_the_crease(self) -> None:
        """Carried over and laid down, the loop stands 30 mm high and its hinges share the turn; pressed, it is a fold."""
        self.assertLess(self.paper["laid_crease"], self.paper["pressed_crease"] - 1.0)
        self.assertLess(self.paper["laid_crease"], 2.0)
        self.assertGreater(self.paper["pressed_crease"], 2.2)

    def test_the_elastic_control_reopens_and_reads_no_crease(self) -> None:
        self.assertGreater(self.elastic["open"], 0.9)
        self.assertEqual(0.0, self.elastic["crease"])
        self.assertEqual(0.0, self.elastic["whole"])

    def test_a_reset_flattens_the_crease_with_the_fold(self) -> None:
        self.assertTrue(self.paper["reset_restores_rest"])
        self.assertEqual(0.0, self.paper["reset_crease"])
        self.assertGreater(self.paper["reset_ratio"], 0.95)
        self.assertGreater(self.paper["reset_open"], 0.95)


class TestTheCoupledSolverReadsTheSameArray(unittest.TestCase):
    """With a robot in the scene the sheet is solved behind a coupler; the kernel writes what that solver reads.

    Measured at Newton 1.6.0: the one-way coupler hands VBD the pipeline model itself, so the
    arrays are one; Newton's coupled solver (``proxy``) hands it a ``ModelView`` that holds its
    own copy of ``edge_rest_angle``, 1.2 MB away from the model's. ``solver_rest_angles`` finds
    the copy, and a crease written to the model's array would bend nothing.
    """

    def test_under_the_proxy_and_the_one_way_coupler(self) -> None:
        import robosimstudio as rss
        from robosimstudio.robots.registry import get_embodiment

        one_way = rss.replace(
            rss.GENERIC, coupling=rss.coupler("one_way"), collision=rss.CollisionSpec(full_surface_contact=True)
        )
        for label, physics in (("proxy", rss.GENERIC), ("one_way", one_way)):
            with self.subTest(coupler=label):
                scene = rss.Scene(cell="bench_0728", physics=physics, name=f"paper_{label}")
                scene.add(rss.robot(get_embodiment("yam_bimanual")))
                scene.add(rss.paper(name="sheet", size=(0.1, 0.1), cells=(10, 10), at=(-0.05, -0.2, 0.01)))
                sim = scene.build(control_mode="joint_position")
                entity = sim.scene.by_name["sheet"]
                solver = sim.pipeline.program.solver_for("sheet")
                theirs = solver.model.edge_rest_angle
                self.assertEqual(entity._rest.ptr, theirs.ptr, "the kernel writes an array the solver never reads")
                self.assertEqual(entity._rest.shape, sim.model.edge_rest_angle.shape)
                self.assertEqual(label == "one_way", entity._rest.ptr == sim.model.edge_rest_angle.ptr)
                self.assertTrue(sim.pipeline.is_coupled)
                # The write reaches the solve: a rest angle of 1 rad on every hinge curls the sheet.
                sim.run(5).sync()
                before = sim.state.particle_q.numpy()[entity.particles].copy()
                entity._rest.assign(np.full(entity._rest.shape, 1.0, dtype=np.float32))
                sim.run(30).sync()
                moved = float(np.abs(sim.state.particle_q.numpy()[entity.particles] - before).max())
                self.assertGreater(moved, 0.01, f"{label}: the solver did not see the rest angles the kernel writes")
                self.assertTrue(sim.state_is_finite())


if __name__ == "__main__":
    unittest.main()
