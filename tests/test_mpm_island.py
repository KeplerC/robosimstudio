# SPDX-License-Identifier: Apache-2.0
"""An MPM island couples to the robot's: a test-local heap of particles beside the YAM, on the GPU.

No shipped kind is used -- the three particle materials are built on top of
this backend in ``robosimstudio_materials`` -- so the heap here is a kind this
module registers for itself, the way ``examples/04_extending/01_new_kind``
does: a domain, a spec and an entity that registers the MPM's custom
attributes and adds a block of particles with the rice preset from the Phase 0
probes. What is claimed is the composer's wiring, each item measured on the
built scene:

* two islands, the robot's and the MPM's, the MPM entry stepped **in place**
  and **no main collision pipeline** (the MPM detects its own contacts);
* the heap falls to the ground and rests there, finite;
* the substep block **captures** into a CUDA graph with the MPM stepping at
  its own rate inside it;
* a hand driven into the heap displaces it;
* ``sim.reset()`` puts the particles back;
* a two-world build composes with ``separate_worlds`` on.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TIER = "medium"
"""A GPU, one small scene stepped a few hundred times, and a second at two worlds."""

RICE = {
    "mpm:young_modulus": 1.0e15,
    "mpm:poisson_ratio": 0.3,
    "mpm:friction": 0.68,
    "mpm:yield_pressure": 1.0e12,
    "mpm:tensile_yield_ratio": 0.0,
}
"""Newton 1.6's rice: the shipped granular example's numbers, as ``probe_a`` used them."""

DOMAIN = "granular_probe"
ENTITY = "heap_probe"


@dataclass(frozen=True, kw_only=True)
class HeapProbeSpec:
    """A block of particles with the rice preset, bottom face centred at ``at`` in the surface frame."""

    entity: str = ENTITY
    name: str = "heap"
    size: tuple[float, float, float] = (0.08, 0.08, 0.03)
    spacing: float = 0.003
    density: float = 1500.0
    at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sim: Any = field(default_factory=lambda: _requirements())  # noqa: PLW0108 -- the import happens at call time


def _requirements():
    import robosimstudio as rss

    return rss.SimRequirements(domain=DOMAIN, contacts=rss.contacts_with("surface", "hands"))


def _register() -> None:
    """The domain and the entity, once per process; a second call is the registries' identical re-registration."""
    import warp as wp

    import robosimstudio as rss
    from robosimstudio.domains.capabilities import ParticleMixin
    from robosimstudio.scene.entity import Entity
    from robosimstudio.specs.physics import DOMAINS, get_domain

    get_domain("cloth")  # the fallback layout must exist before a chain names it
    if DOMAIN not in DOMAINS:
        rss.register_domain(
            DOMAIN,
            state=("particles",),
            reads=(DOMAIN, "cloth"),
            role="deformables",
            contacts=("surface", "hands"),
            summary="A heap of particles, for the island test.",
        )

    class EntityHeapProbe(ParticleMixin, Entity):
        domain = DOMAIN

        def __init__(self, spec, *, world: int = 0) -> None:
            super().__init__(spec, world=world)
            self.particles = np.zeros(0, dtype=np.int32)
            self.sets = {}
            self.probe = None
            self.rest = {}

        def build(self, builder, episode) -> None:
            del episode
            from newton.solvers import SolverImplicitMPM

            # Before the particles: an attribute registered after they were
            # added is one the builder never kept for them (the rod does the
            # same for VBD's). The compiler registered it from the arrangement
            # already; a second registration is accepted.
            SolverImplicitMPM.register_custom_attributes(builder)
            spec = self.spec
            placed = self.placed((*spec.at, 0.0, 0.0, 0.0, 1.0))
            dims = [max(1, int(round(s / spec.spacing))) for s in spec.size]
            cell = [s / d for s, d in zip(spec.size, dims, strict=True)]
            lo = (placed[0] - 0.5 * spec.size[0], placed[1] - 0.5 * spec.size[1], placed[2] + 0.5 * cell[2])
            builder.add_particle_grid(
                pos=wp.vec3(*[float(v) for v in lo]),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=dims[0],
                dim_y=dims[1],
                dim_z=dims[2],
                cell_x=cell[0],
                cell_y=cell[1],
                cell_z=cell[2],
                mass=float(np.prod(cell)) * spec.density,
                jitter=0.25 * spec.spacing,
                radius_mean=0.5 * max(cell),
                custom_attributes=dict(RICE),
            )

        def handle(self, model):
            handle = super().handle(model)
            handle.point_radius = 0.5 * float(self.spec.spacing)
            return handle

        def on_scene_ready(self, pipeline) -> None:
            self.join_settle(pipeline, None)

        def reset(self, pipeline, episode, *, world: int = 0) -> None:
            """The particles back where they rested, and the MPM's history with them.

            ``reset_scene`` resets the solvers with ``flags=0`` -- the joint
            coordinates and nothing else -- and the implicit MPM keeps its
            particle history (plastic volume, strain, stress) on the state,
            where only a ``PARTICLE``-flagged reset clears it. A particle kind
            built on this backend asks for that here, as the plan has the
            shipped kinds do.
            """
            import newton

            super().reset(pipeline, episode, world=world)
            pipeline.reset_solvers(flags=int(newton.StateFlags.PARTICLE))

        def update_probe(self, state, step=None, *, readback=None):
            if readback is not None and getattr(readback, "particle_q", None) is not None and len(readback.particle_q):
                self.probe.update_arrays(readback.particle_q, readback.particle_qd, step)
            else:
                self.probe.update(state, step)
            return self.probe

    from robosimstudio.scene.registry import ENTITIES

    if ENTITY not in ENTITIES:
        rss.register_entity(ENTITY)(EntityHeapProbe)


def _physics():
    import robosimstudio as rss

    # GENERIC routes ``granular``; this test's own domain has to be routed the same way.
    return rss.GENERIC.with_backend(DOMAIN, rss.backend("mpm"))


def _build(*, num_worlds: int = 1, **spec):
    import robosimstudio as rss

    _register()
    scene = rss.Scene(cell="void", name="mpm_island", physics=_physics())
    scene.add(rss.robot("yam_bimanual"))
    heap = scene.add(HeapProbeSpec(**spec))
    sim = scene.build(control_mode="joint_position", num_worlds=num_worlds)
    return sim, heap


class TestTheIslandCouplesToTheRobot(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim, cls.heap = _build()
        cls.program = cls.sim.pipeline.program

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def _particles(self) -> np.ndarray:
        self.sim.pipeline.sync()
        return np.asarray(self.sim.pipeline.state_0.particle_q.numpy(), dtype=np.float64)

    def test_two_islands_the_mpm_one_in_place_and_no_main_pipeline(self) -> None:
        names = [(island.name, island.backend.solver) for island in self.program.islands]
        self.assertEqual([("mujoco", "mujoco"), ("mpm", "mpm")], names)
        configs = self.program.solver._entry_configs_by_name
        self.assertTrue(configs["mpm"].in_place, "the registered steps_in_place reached the entry")
        self.assertFalse(configs["mujoco"].in_place)
        self.assertIsNone(
            self.program.collision, "the MPM detects its own contacts; no main pipeline over its particles"
        )
        self.assertEqual(
            {}, dict(getattr(self.program.solver, "_proxy_collision_configs", {})), "and no channel pipeline"
        )
        self.assertEqual(1, len(self.program.channels))
        self.assertEqual(("mujoco", "mpm"), (self.program.channels[0].source, self.program.channels[0].destination))

    def test_the_solver_is_the_adapter_at_the_registered_rate(self) -> None:
        from robosimstudio.engines.newton.mpm import SolverMPM

        solver = self.program.solver_of_island("mpm")
        self.assertIsInstance(solver, SolverMPM)
        self.assertEqual(60.0, solver.step_hz)
        self.assertEqual(16, solver.stride, "sixteen substeps at 60 Hz and a 60 Hz step: one step per control step")
        self.assertTrue(solver.project_outside_after_step)
        self.assertTrue(solver._sparse_rebuildable, "the sparse grid is the capacity-bounded, capturable one")
        self.assertIn("mpm_grid_check:mpm", self.sim.pipeline.schedule.names()[-1])

    def test_an_overflowed_grid_is_named_by_the_check_and_a_reset_clears_it(self) -> None:
        """The after-step check names the island and the knobs; Newton's reset clears the status, so the scene runs on.

        The status is set between a step and its check: the next step's grid
        rebuild writes the word afresh.
        """
        import warp as wp

        stage = next(s for s in self.sim.pipeline.schedule.phases["after"] if s.name == "mpm_grid_check:mpm")
        solver = self.program.solver_of_island("mpm")
        self.sim.run(1)
        self.assertIsNotNone(solver._grid_status, "the first step allocates the status word")
        try:
            solver._grid_status.fill_(wp.Volume.REBUILD_VOXEL_CAPACITY_EXCEEDED)
            with self.assertRaisesRegex(RuntimeError, "island 'mpm'.*max_active_cell_count"):
                stage.fn(None, 0.0)
        finally:
            self.sim.pipeline.reset_solvers(flags=0)
        stage.fn(None, 0.0)
        self.sim.run(1)

    def test_the_graph_captured(self) -> None:
        """``Executor.captured`` is whether ``warmup_and_capture`` kept a graph; the error it saw is beside it."""
        self.assertTrue(self.sim.pipeline.captured, self.sim.pipeline.capture_error)

    def test_the_heap_falls_and_rests_on_the_ground(self) -> None:
        self.sim.run(90)
        q = self._particles()
        self.assertTrue(np.isfinite(q).all())
        self.assertGreater(float(q[:, 2].min()), -0.005, "a particle went through the ground plane")
        self.assertLess(float(q[:, 2].min()), 0.005, "the bottom layer rests within 5 mm of the surface")
        self.assertLess(float(q[:, 2].max()), 0.03, "the block relaxed rather than standing as built")
        self.assertTrue(self.sim.pipeline.captured, "stepping did not drop the graph")

    def test_a_hand_driven_into_the_heap_displaces_it(self) -> None:
        sim = self.sim
        ik = sim.arm("right")
        gripper = sim.gripper("right")
        hz = float(sim.request.physics.control_hz)
        q = self._particles()
        centre = q[:, :2].mean(axis=0)
        top = float(q[:, 2].max())
        spacing = float(self.heap.entity.spec.spacing) if hasattr(self.heap, "entity") else 0.003

        def go(start, goal, seconds: float) -> None:
            frames = int(seconds * hz)
            for k in range(frames):
                a = (k + 1) / frames
                ik.set_target(start * (1.0 - a) + goal * a, quat)
                ik.solve_into(sim.state.joint_q, sim.control.joint_target_q)
                gripper.apply(sim.control.joint_target_q)
                sim.run(1)

        pos, quat = ik.ee_pose(sim.state)
        gripper.command = 1.0  # fingers together, so the hand is one blade over the middle of the heap
        hover = np.array([centre[0], centre[1], top + 0.05])
        go(pos, hover, 1.5)
        before = self._particles()
        press = np.array([centre[0], centre[1], top - 0.015])
        go(hover, press, 1.5)
        after = self._particles()
        self.assertTrue(np.isfinite(after).all())
        moved = np.linalg.norm(after - before, axis=1)
        self.assertGreater(float(moved.max()), spacing, "no particle moved by more than the spacing")
        self.assertGreater(int((moved > spacing).sum()), 20, "the hand pushed a few grains, not a heap")

    def test_reset_restores_the_particles(self) -> None:
        self.sim.run(30)
        rest = np.asarray(self.heap.entity.rest["particle_q"] if hasattr(self.heap, "entity") else None)
        self.sim.reset()
        q = self._particles()
        np.testing.assert_allclose(q, rest, atol=1.0e-6)
        state = self.sim.pipeline.state_0
        jp = state.mpm.particle_Jp.numpy()
        np.testing.assert_allclose(jp, 1.0, atol=1.0e-5, err_msg="the plastic volume history was not reset")


class TestTwoWorlds(unittest.TestCase):
    def test_a_replicated_build_isolates_the_worlds(self) -> None:
        sim, _heap = _build(num_worlds=2, size=(0.06, 0.06, 0.024), spacing=0.004)
        try:
            program = sim.pipeline.program
            solver = program.solver_of_island("mpm")
            self.assertTrue(solver._separate_worlds, "a replicated model gets one grid per world")
            self.assertEqual(2, solver._environment_count)
            per_world = sim.pipeline.model.particle_count // 2
            sim.run(60)
            sim.pipeline.sync()
            q = np.asarray(sim.pipeline.state_0.particle_q.numpy(), dtype=np.float64)
            self.assertTrue(np.isfinite(q).all())
            for world in range(2):
                block = q[world * per_world : (world + 1) * per_world]
                self.assertGreater(float(block[:, 2].min()), -0.005, f"world {world} fell through the ground")
                self.assertLess(float(block[:, 2].min()), 0.005, f"world {world} is not resting on the ground")
        finally:
            sim.close()


if __name__ == "__main__":
    unittest.main()
