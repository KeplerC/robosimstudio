# SPDX-License-Identifier: Apache-2.0
"""Every material kind on a CPU: it builds and steps, or the build refuses it by name before anything is built.

What ``robosimstudio[cpu]`` promises for the materials, on a machine whose GPUs
are hidden (``scripts/testrun.py --tier cpu``), with ``device`` left at
``auto``. Cloth, the sponge and the cable are :mod:`tests.test_cpu_scenes`'s,
through the examples a reader runs; this module holds the rest:

* **Built and stepped.** A bag lands on SuperDex beside the cloth, under
  ``GENERIC`` and under ``rss.BAG_60HZ`` alike, and lies on the bench. The
  cutting kind's cage runs on Warp's CPU device: a knife driven down through a
  block of tofu parts it in two, and the YAM's right hand closes on the knife's
  20 mm handle and stalls on it, as it does on MuJoCo-Warp.
* **Refused by name.** Paper, tape, the brush, a heap, a liquid, a dough and
  water have no backend a CPU runs -- VBD, the implicit MPM and Genesis are
  CUDA only -- and ``rss.CPU`` names nothing else for them. Under ``GENERIC``
  and under each kind's own arrangement, the build raises before an entity is
  made, naming the entity, its kind, the backend, why that backend is not run
  on a CPU, and that the scene needs a GPU; a scene holding one of them beside
  a kind a CPU does step names only the one it cannot. Water under ``GENERIC``,
  which leaves it on VBD, and every one of them under ``rss.CPU``, which leaves
  it on MuJoCo, is told the arrangement that steps it instead, since a GPU
  refuses those arrangements too.
* **What a served kind cannot bring along, refused by name.** A cloth's contact
  forces are VBD's, so reading them on a CPU, where the cloth is on SuperDex,
  says it needs a GPU rather than falling through to MuJoCo's solver.

Built from the library's constructors rather than the example scripts, so the
scenes say what they are without depending on where an example lives; each
material's example and scored variant, the two capsule-chain rope examples that
build with ``device="cuda"`` among them, is :mod:`tests.test_cpu_examples`'s. Run
by hand on a GPU box without the runner, the module sets
``ROBOSIMSTUDIO_DEVICE=cpu`` for itself, as :mod:`tests.test_cpu_scenes` does.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import traceback
import unittest
import unittest.mock

import numpy as np

TIER = "cpu"
"""No GPU, real scenes; the bag needs the ``[cpu]`` extra. About a minute warm, half of it the cutter's walks."""

HAS_SUPERDEX = importlib.util.find_spec("superdex") is not None
NO_SUPERDEX = "needs the [cpu] extra (SuperDex)"

_hidden = unittest.mock.patch.dict(os.environ, {"ROBOSIMSTUDIO_DEVICE": "cpu"})
_device_before: list = []


def setUpModule() -> None:
    """Hide the GPU by the rule's own variable when the runner did not hide it by the driver's."""
    import warp as wp

    _device_before.append(wp.get_device())
    if wp.is_cuda_available():
        _hidden.start()


def tearDownModule() -> None:
    import warp as wp

    with contextlib.suppress(RuntimeError):  # never started
        _hidden.stop()
    if _device_before:
        wp.set_device(_device_before.pop())


def _paper_arrangement(rss):
    """The fold example's: one-way coupling and full-surface contact, which let the pads pinch a sheet."""
    return rss.replace(
        rss.GENERIC, coupling=rss.coupler("one_way"), collision=rss.CollisionSpec(full_surface_contact=True)
    )


REFUSED = (
    # kind, entity, the backend it is on, what builds it, its own arrangement
    ("paper", "paper", "vbd", lambda rss: [rss.paper(name="paper", size=(0.1, 0.1), cells=(5, 5), at=(0.0, 0.0, 0.01))],
     _paper_arrangement),
    ("tape", "label", "vbd", lambda rss: [rss.box(name="box", half=(0.05, 0.05, 0.02), at=(0.0, 0.0, 0.02)),
                                          rss.tape(name="label", on="box", at=(-0.02, -0.03, 0.04))],
     lambda rss: rss.TAPE_60HZ),
    ("brush", "brush", "vbd", lambda rss: [rss.brush(name="brush", at=(0.0, 0.0, 0.0), rows=4, columns=4)],
     lambda rss: rss.BRUSH_60HZ),
    ("granular", "rice", "mpm", lambda rss: [rss.heap(name="rice", size=(0.04, 0.04, 0.02), at=(0.0, 0.0, 0.0))],
     lambda rss: rss.GRANULAR_60HZ),
    ("liquid", "honey", "mpm", lambda rss: [rss.liquid(name="honey", size=(0.03, 0.03, 0.03), at=(0.0, 0.0, 0.0))],
     lambda rss: rss.LIQUID_120HZ),
    ("dough", "dough", "mpm", lambda rss: [rss.dough(name="dough", size=(0.04, 0.04, 0.02), at=(0.0, 0.0, 0.0))],
     lambda rss: rss.DOUGH_60HZ),
    ("water", "water", "genesis", lambda rss: [rss.water(name="water", at=(0.0, 0.0, 0.0))],
     lambda rss: rss.WATER_60HZ),
)  # fmt: skip
"""The kinds nothing on a CPU steps: what each is on, a small one of it, and the arrangement its examples stage."""


class TestTheKindsACpuRefuses(unittest.TestCase):
    """Refused in the compiler, before an entity is made, with the kind and the reason in one sentence."""

    def refusal(self, physics, things) -> tuple[str, list[traceback.FrameSummary]]:
        import robosimstudio as rss
        from robosimstudio.scene.physics import CompositionError

        scene = rss.Scene(cell="bench_075", **({"physics": physics} if physics is not None else {}))
        scene.add(rss.robot("panda"))
        for thing in things:
            scene.add(thing)
        try:
            scene.build()
        except CompositionError as refused:  # by hand: assertRaises drops the traceback this reads
            return str(refused), traceback.extract_tb(refused.__traceback__)
        self.fail("a CPU built it")

    def test_each_kind_under_generic_and_under_its_own_arrangement(self) -> None:
        import robosimstudio as rss

        for kind, name, solver, make, own in REFUSED:
            for label, physics in (("GENERIC", None), ("its own", own(rss))):
                with self.subTest(kind=kind, arrangement=label):
                    message, frames = self.refusal(physics, make(rss))
                    if (kind, label) == ("water", "GENERIC"):
                        # GENERIC's "*", VBD, steps no water on a GPU either: the route is what works.
                        self.assertIn("'water' (water) is on 'vbd', which steps no water on any device", message)
                        route = "as rss.WATER_60HZ does (it needs a GPU and the [genesis] extra)."
                        self.assertTrue(message.endswith(route), message)
                    else:
                        self.assertIn(f"'{name}' ({kind}) needs a GPU: {kind} is on {solver!r}", message)
                        self.assertIn("which runs on cuda only (", message, "the backend's reason is given")
                        self.assertIn("rss.CPU puts cloth, bags, rope and soft bodies on SuperDex", message)
                        extra = " with the [genesis] extra" if kind == "water" else ""
                        self.assertTrue(message.endswith(f"Build it on a GPU{extra}."), message)
                    # Nothing was built: the refusal is the compiler's, before the arrangement is adapted.
                    self.assertEqual(frames[-1].name, "_for_platform", [f.name for f in frames])

    def test_under_rss_cpu_staged_by_hand_each_is_refused_not_handed_to_mujoco(self) -> None:
        """``rss.CPU``'s ``"*"`` is MuJoCo, which steps none of them on either device: each is told its route, not a GPU."""
        import robosimstudio as rss

        things = [thing for _, _, _, make, _ in REFUSED for thing in make(rss)]
        message, frames = self.refusal(rss.CPU, things)
        for kind, name, solver, *_ in REFUSED:
            self.assertIn(
                f"'{name}' ({kind}) is on 'mujoco', which steps no {kind} on any device: route the {kind!r} kind to "
                f"backend({solver!r})",
                message,
            )
        self.assertNotIn("Build it on a GPU", message)
        self.assertEqual(frames[-1].name, "_for_platform", [f.name for f in frames])

    def test_beside_a_kind_a_cpu_steps_only_the_one_it_cannot_is_named(self) -> None:
        import robosimstudio as rss

        things = [
            rss.sheet(name="towel", size=(0.1, 0.1), cells=(4, 4), at=(-0.1, -0.1, 0.02)),
            rss.box(name="cube", half=(0.02, 0.02, 0.02), at=(0.15, 0.1, 0.02)),
            rss.heap(name="rice", size=(0.04, 0.04, 0.02), at=(0.0, 0.1, 0.0)),
        ]
        message, _ = self.refusal(None, things)
        self.assertIn("'rice' (granular) needs a GPU", message)
        self.assertNotIn("towel", message)
        self.assertNotIn("cube", message)


class TestWhatACpuServes(unittest.TestCase):
    """The rule, without a scene: which kinds an arrangement leaves a CPU nothing for."""

    def test_only_the_gpu_kinds_under_generic(self) -> None:
        from robosimstudio.scene.physics.devices import unserved
        from robosimstudio.specs.physics import GENERIC

        owners = {kind: [kind] for kind in ("cloth", "soft", "rod", "bag", "rigid", "cutting", "static", "articulated")}
        owners.update({kind: [name] for kind, name, *_ in REFUSED})
        refused = unserved(GENERIC, owners)
        self.assertEqual([sentence.split(" (")[1].split(")")[0] for sentence in refused], [k for k, *_ in REFUSED])

    def test_every_gpu_kind_on_an_arrangement_whose_star_is_mujoco(self) -> None:
        from robosimstudio.scene.physics.devices import unserved
        from robosimstudio.specs.physics import CPU, MUJOCO_500HZ

        owners = {kind: [name] for kind, name, *_ in REFUSED}
        owners.update({kind: [kind] for kind in ("cloth", "soft", "rod", "bag", "rigid", "cutting")})
        for arrangement in (CPU, MUJOCO_500HZ):
            refused = unserved(arrangement, owners)
            self.assertEqual([sentence.split(" (")[1].split(")")[0] for sentence in refused], [k for k, *_ in REFUSED])

    def test_a_dotted_path_states_no_devices_and_is_served(self) -> None:
        """``newton.solvers.SolverVBD`` by path is the way to run paper on a CPU anyway, slowly."""
        from robosimstudio.scene.physics.devices import unserved
        from robosimstudio.specs.physics import GENERIC, backend

        self.assertEqual(
            unserved(GENERIC.with_backend("paper", backend("newton.solvers.SolverVBD")), {"paper": ["p"]}), ()
        )

    def test_an_entity_naming_a_gpu_only_backend_itself_is_told_there_is_no_other(self) -> None:
        """Not pointed at ``rss.CPU``'s MuJoCo, which is what ``CPU`` would otherwise suggest for a heap."""
        from robosimstudio.scene.physics import CompositionError
        from robosimstudio.scene.physics.devices import on_device
        from robosimstudio.specs.physics import backend

        with self.assertRaises(CompositionError) as caught:
            on_device(backend("mpm"), "cpu", owner="rice", domain="granular", declared=True)
        message = str(caught.exception)
        self.assertIn("'rice' (granular) is on 'mpm', which runs on cuda only (Newton's implicit MPM", message)
        self.assertIn("nothing steps granular. Build it on a GPU.", message)
        self.assertNotIn("mujoco", message)

    def test_every_cuda_only_backend_says_why(self) -> None:
        from robosimstudio.specs.physics import get_backend

        for alias in ("vbd", "mpm", "genesis"):
            info = get_backend(alias)
            self.assertEqual(info.devices, ("cuda",), alias)
            self.assertTrue(info.cpu_note, f"{alias} has no cpu_note for the refusal to give")


@unittest.skipUnless(HAS_SUPERDEX, NO_SUPERDEX)
class TestAClothsContactForcesOnACpu(unittest.TestCase):
    """The scrub example's readout: staged as on a GPU, the towel lands on SuperDex, and reading it is refused."""

    def test_the_reading_says_it_needs_a_gpu(self) -> None:
        import robosimstudio as rss
        from robosimstudio_materials.cloth import ForceGatedWipe, contact_force

        forces = rss.coupler("one_way", contact_forces=True)
        scene = rss.Scene(cell="bench_075", physics=rss.replace(rss.GENERIC, coupling=forces))
        scene.add(rss.robot("panda"))
        scene.add(rss.sheet(name="towel", size=(0.1, 0.1), cells=(4, 4), at=(-0.05, -0.05, 0.02)))
        sim = scene.build()
        try:
            self.assertTrue(sim.platform.is_cpu)
            for read in (lambda: contact_force(sim, "towel"), lambda: ForceGatedWipe(sim, "towel", "s", "arm", (1, 2))):
                with self.assertRaises(RuntimeError) as caught:
                    read()
                message = str(caught.exception)
                self.assertIn("contact forces need a GPU", message)
                self.assertIn("rss.CPU puts the cloth on SuperDex", message)
                self.assertTrue(message.endswith("Build it on a GPU."), message)
        finally:
            sim.close()


@unittest.skipUnless(HAS_SUPERDEX, NO_SUPERDEX)
class TestABagOnACpu(unittest.TestCase):
    """SuperDex's self-colliding shell, which is what a bag is on a GPU too: on its island, on the bench."""

    def test_under_generic_and_under_bag_60hz(self) -> None:
        import robosimstudio as rss
        from robosimstudio_materials.bag import EntityBagSuperdex

        for label, physics in (("GENERIC", None), ("BAG_60HZ", rss.BAG_60HZ)):
            with self.subTest(arrangement=label):
                scene = rss.Scene(cell="bench_075", **({"physics": physics} if physics is not None else {}))
                scene.add(rss.robot("panda"))
                scene.add(rss.bag(name="bag", size=(0.16, 0.12), cells=(8, 6), at=(0.0, -0.1, 0.0)))
                sim = scene.build()
                try:
                    self.assertTrue(sim.platform.is_cpu)
                    self.assertIsInstance(sim["bag"].entity, EntityBagSuperdex)
                    program = sim.pipeline.program
                    host = next(i for i in program.islands if i.backend.solver == "superdex")
                    self.assertIn("bag", host.entities, program.summary())
                    sim.run(20).sync()
                    q = sim.pipeline.state_0.particle_q.numpy()[np.asarray(sim["bag"].particles, dtype=int)]
                    self.assertTrue(np.isfinite(q).all(), "the bag diverged on a CPU")
                    surface = float(sim.scene.handles.surface_z)
                    self.assertAlmostEqual(float(q[:, 2].min()), surface, delta=0.006, msg="the bag is off the bench")
                finally:
                    sim.close()


BLADE_X, TOFU_Y = 0.04, -0.16
"""The cutting example's blade and tofu middles across the bench [m]."""
BOARD_TOP = 0.012
"""The board's top over the bench [m]."""
EDGE = 0.04
"""The edge's depth under the handle's middle, the knife's frame [m]."""


def _tofu_scene(rss, *, knife_at, tofu: bool = True):
    """The cutting example's board and tofu, and a knife standing on its edge with its handle's middle at ``knife_at``."""
    knife = rss.knife(name="knife", at=knife_at)
    scene = rss.Scene(cell="bench_075", physics=rss.CUTTING_50HZ, name="cut_tofu")
    scene.add(knife)
    board = (0.07, 0.08, 0.006)
    scene.add(rss.slab(name="board", half=board, at=(BLADE_X, TOFU_Y, board[2])))
    if tofu:
        at = (BLADE_X, TOFU_Y, BOARD_TOP)
        material = rss.cuttables.TOFU
        scene.add(rss.cuttable(name="tofu", knife=knife, size=(0.06, 0.10, 0.04), at=at, material=material))
    return scene


class TestTheTofuOnACpu(unittest.TestCase):
    """The cage cutter on Warp's CPU device: a knife written down through the tofu to the board parts it in two.

    The knife is moved by its free joint, 4 mm a control step, rather than by a
    hand, so what is measured is the cut and the coupling that reads the
    knife's pose off the state MuJoCo's C library steps; the hand is the next
    class's. Each step the edge advances costs 1 to 2 s here, against 7.6 ms on
    a GPU: the cutter's walks run on one CPU thread.
    """

    @classmethod
    def setUpClass(cls) -> None:
        import newton

        import robosimstudio as rss

        top = 0.10  # the handle's middle: the edge 8 mm over the tofu's 52 mm top
        cls.sim = sim = _tofu_scene(rss, knife_at=(BLADE_X - 0.105, TOFU_Y, top)).build()
        cls.tofu = tofu = sim["tofu"].entity
        knife = sim.scene.objects["knife"]
        start = int(knife.coord_start)
        q0 = sim.pipeline.state_0.joint_q.numpy().copy()
        lift = q0[start + 2] - top  # the knife's free joint, in world, over its bench-frame height
        cls.over = (tofu.pieces(), tofu.cut_fraction())
        cls.halfway = None
        heights = np.arange(top, BOARD_TOP + EDGE + 0.0005 - 1.0e-9, -0.004)
        for z in (*heights, *[heights[-1]] * 5):
            q = sim.pipeline.state_0.joint_q.numpy().copy()
            q[start : start + 7] = q0[start : start + 7]
            q[start + 2] = z + lift
            for state in (sim.pipeline.state_0, sim.pipeline.state_1):
                state.joint_q.assign(q)
                state.joint_qd.zero_()
                newton.eval_fk(sim.model, state.joint_q, state.joint_qd, state)
            sim.run(1).sync()
            if cls.halfway is None and z - EDGE < BOARD_TOP + 0.02:
                cls.halfway = (tofu.pieces(), tofu.cut_fraction())
        cls.done = (tofu.pieces(), tofu.cut_fraction(), tofu.cuts())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def test_it_is_on_the_cpu_with_the_knife_on_mujocos_c_library(self) -> None:
        self.assertTrue(self.sim.platform.is_cpu)
        program = self.sim.pipeline.program
        self.assertEqual([island.backend.solver for island in program.islands], ["mujoco"])
        self.assertTrue(program.islands[0].backend.get("use_mujoco_cpu"))

    def test_over_the_tofu_nothing_is_cut(self) -> None:
        self.assertEqual(self.over, (1, 0.0))

    def test_halfway_down_the_cut_runs_and_the_tofu_is_one_piece(self) -> None:
        pieces, fraction = self.halfway
        self.assertEqual(pieces, 1)
        self.assertGreater(fraction, 0.2)

    def test_at_the_board_the_tofu_is_in_two(self) -> None:
        self.assertEqual(self.done, (2, 1.0, 1))
        self.assertTrue(np.isfinite(self.sim.pipeline.state_0.body_q.numpy()).all())


class TestTheHandHoldsTheKnifeOnACpu(unittest.TestCase):
    """The cutting example's grasp: the YAM's right hand closes on the 20 mm handle and stalls on it.

    MuJoCo-Warp stalls it at 20.4 mm. On MuJoCo's C library the left finger went
    through to 3.8 mm until its collision tree was refitted to the inertial frame
    Newton's notify gives it (:func:`robosimstudio.core.mjdata.refit_bvh`).
    """

    def test_the_pads_stall_on_the_handle(self) -> None:
        import robosimstudio as rss
        from robosimstudio.robots.registry import get_embodiment
        from tests.fixtures.grasp import go, open_session, tick

        over_knife = (0.0, 1.792, 1.376, -1.154, 0.0, 0.0)  # the example's right arm, over the handle
        yam = get_embodiment("yam_bimanual")
        yam = rss.replace(yam, arms=(yam.arms[0], rss.replace(yam.arms[1], home_q=over_knife)))
        scene = _tofu_scene(rss, knife_at=(BLADE_X - 0.105, -0.31, 0.04), tofu=False)
        scene.add(rss.robot(yam, gripper=rss.ActuationSpec(target_ke=1500.0, target_kd=60.0, effort_limit=12.0)))
        for name, y in (("rest_a", -0.31 - 0.00625), ("rest_b", -0.31 + 0.00625)):
            scene.add(rss.slab(name=name, half=(0.05, 0.004, 0.012), at=(BLADE_X, y, 0.012)))
        sim = scene.build(control_mode="joint_position")
        session = open_session(sim, 8983)
        try:
            self.assertTrue(sim.platform.is_cpu)
            tick(session, 30)
            miss = go(session, "right", np.asarray(sim["knife"].position, dtype=np.float64), speed=0.04)
            self.assertLess(miss, 0.008, f"the hand ended {miss * 1e3:.1f} mm short of the handle")
            session.arms["right"].grip_request = 1.0
            tick(session, 50)
            width = float(session.env.gripper("right").width(session.env.state))
            self.assertGreater(width, 0.018, f"the jaws closed to {width * 1e3:.1f} mm on a 20 mm handle")
        finally:
            session.teardown()


if __name__ == "__main__":
    unittest.main()
