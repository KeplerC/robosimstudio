# SPDX-License-Identifier: Apache-2.0
"""``FleetEnv``'s refusals, made before anything is built: an undeclared pairing, a wave
whose documents are different models, and a scene whose scoring cannot be read per world;
and what counts as different physics and how a staged value is named in a refusal."""

import dataclasses
import unittest
from types import SimpleNamespace

from robosimstudio.scene import compiler
from robosimstudio.scene.compiler import CannotReplicate
from robosimstudio.scene.entity import Entity
from robosimstudio.specs import TaskSpec
from robosimstudio_bench.envs import fleet

TIER = "fast"
"""Specs, unbuilt entities and a bare FleetEnv: no scene, no GPU."""


class TestABatchedEnvRefusesAnUndeclaredPairing(unittest.TestCase):
    def test_scene_override_on_a_task_without_bindings(self):
        from robosimstudio_bench.envs.fleet import FleetEnv

        spec = TaskSpec(name="t", scene="bench_a", embodiment="franka")
        with self.assertRaises(KeyError) as caught:
            FleetEnv(spec, num_worlds=2, scene="bench_b", device="cpu")
        self.assertIn("declares no binding", str(caught.exception))
        # Not fixed by narrowing the grid, so not answered the way a replication refusal is.
        self.assertNotIsInstance(caught.exception, CannotReplicate)


def _with_params(arrangement, **params):
    """``arrangement`` with its one backend's ``params`` updated."""
    ((domain, backend),) = arrangement.backends
    merged = tuple(sorted({**dict(backend.params), **params}.items()))
    return dataclasses.replace(arrangement, backends=((domain, dataclasses.replace(backend, params=merged)),))


class TestWhatCountsAsDifferentPhysics(unittest.TestCase):
    """``_material_physics``: what an arrangement integrates, with the buffer sizes left out."""

    def setUp(self):
        from robosimstudio_bench.taskbuild.arrangements import PHYSICS

        self.base = PHYSICS.get("mujoco_500hz@v1")

    def test_a_bigger_constraint_buffer_is_the_same_physics(self):
        bigger = _with_params(self.base, njmax=8192, nconmax=4096, rigid_contact_max=10_000)
        self.assertEqual(fleet._material_physics(bigger), fleet._material_physics(self.base))

    def test_the_rate_the_substeps_and_a_solver_parameter_are_not(self):
        for changed in (
            dataclasses.replace(self.base, control_hz=100.0),
            dataclasses.replace(self.base, substeps=20),
            _with_params(self.base, enable_multiccd=False),
        ):
            with self.subTest(changed=changed):
                self.assertNotEqual(fleet._material_physics(changed), fleet._material_physics(self.base))

    def test_no_arrangement_is_nothing(self):
        self.assertEqual(fleet._material_physics(None), ())


class TestAStagedValueIsNamed(unittest.TestCase):
    def test_a_registered_arrangement_is_its_name(self):
        from robosimstudio_bench.taskbuild.arrangements import PHYSICS

        self.assertEqual(fleet._staged_name("physics", PHYSICS.get("mujoco_500hz@v1")), "'mujoco_500hz@v1'")

    def test_an_unregistered_one_is_cut_to_eighty_characters(self):
        from robosimstudio_bench.taskbuild.arrangements import PHYSICS

        unregistered = dataclasses.replace(PHYSICS.get("mujoco_500hz@v1"), control_hz=123.0)
        named = fleet._staged_name("physics", unregistered)
        self.assertEqual(len(named), 80)
        self.assertTrue(named.endswith("..."))

    def test_any_other_field_is_its_repr(self):
        self.assertEqual(fleet._staged_name("scene", "bench_075"), "'bench_075'")


class TestAWaveHoldsOneModel(unittest.TestCase):
    """``_check_documents``: a document that is a different model is refused, naming what differs."""

    def _fleet(self, **staged):
        env = fleet.FleetEnv.__new__(fleet.FleetEnv)
        env.build_task = TaskSpec(name="t", **staged)
        return env

    def test_documents_that_stage_the_same_model_share_a_wave(self):
        env = self._fleet(scene="bench_075")
        env._check_documents([None, TaskSpec(name="t", scene="bench_075"), TaskSpec(name="t", scene="bench_075")])

    def test_a_document_of_another_base_is_refused(self):
        env = self._fleet()
        with self.assertRaisesRegex(
            CannotReplicate, "world 1 runs 'other', whose fleet base is 'other'; this wave's model is 't'"
        ):
            env._check_documents([None, TaskSpec(name="other")])

    def test_the_field_two_documents_disagree_on_is_named(self):
        env = self._fleet(scene="bench_075")
        with self.assertRaisesRegex(CannotReplicate, "stages scene='bench_b' against world 0's 'bench_075'"):
            env._check_documents([TaskSpec(name="t", scene="bench_075"), TaskSpec(name="t", scene="bench_b")])

    def test_a_base_built_with_bigger_buffers_still_runs_its_instances(self):
        from robosimstudio_bench.taskbuild.arrangements import PHYSICS

        base = PHYSICS.get("mujoco_500hz@v1")
        env = self._fleet(physics=_with_params(base, njmax=8192))
        env._check_documents([TaskSpec(name="t", physics=base)])
        env = self._fleet(physics=dataclasses.replace(base, control_hz=100.0))
        with self.assertRaisesRegex(CannotReplicate, "may not change the integration"):
            env._check_documents([TaskSpec(name="t", physics=base)])


class _Publisher(Entity):
    def published_states(self, snapshot) -> dict[str, float]:
        return {"bond_fraction": 0.0}


class _WorldAwarePublisher(_Publisher):
    world_aware = True


class _Hooked(Entity):
    """An entity that puts kernels in the substep block, written for one world."""

    def contribute_hooks(self, hooks) -> None:
        pass


class _Probed(Entity):
    """Particles, and a probe opened per world over them."""

    def probe_for(self, world: int):
        return None


def _entity(cls, name: str, particles: int = 0):
    entity = cls(SimpleNamespace(name=name))
    entity.range.particles = (0, particles)
    return entity


class TestWhatAWaveCanScore(unittest.TestCase):
    """``scoring_refusal``: the branches beyond the plain refusals ``TestTheEmptyContextIsCoveredByTheRefusal`` pins."""

    def test_a_rigid_scene_is_scored_per_world(self):
        self.assertEqual(fleet.scoring_refusal(SimpleNamespace(by_name={"cube": _entity(Entity, "cube")})), "")

    def test_particles_with_a_probe_per_world_are_scored_per_world(self):
        towel = _entity(_Probed, "towel", particles=64)
        self.assertEqual(fleet.scoring_refusal(SimpleNamespace(by_name={"towel": towel})), "")

    def test_a_publisher_made_of_particles_is_refused_for_what_it_publishes(self):
        film = _entity(_Publisher, "film", particles=64)
        refusal = fleet.scoring_refusal(SimpleNamespace(by_name={"film": film}))
        self.assertIn("publishing states (film)", refusal)
        self.assertNotIn("made of particles", refusal)

    def test_every_refused_entity_is_named_in_order(self):
        scene = SimpleNamespace(
            by_name={"z_towel": _entity(Entity, "z_towel", 8), "a_towel": _entity(Entity, "a_towel", 8)}
        )
        self.assertIn("(a_towel, z_towel)", fleet.scoring_refusal(scene))


class TestCannotReplicate(unittest.TestCase):
    """The refusal a page answers by narrowing rather than by refusing.

    Most scenes cannot be built at more than one world yet. On an interactive
    page that is not a failed load -- the selection is a *task*, and the world
    count is a preference on it -- so the studio catches this one type and
    rebuilds at one world. Which only works if it stays distinguishable from
    every other way a compile can fail, and stays catchable by the handlers
    written before it had a name.
    """

    def test_it_is_a_kind_of_not_implemented_error(self):
        """A wave the fleet refuses is caught by a handler written for ``NotImplementedError``."""
        from robosimstudio_bench.envs.fleet import FleetEnv

        fleet = FleetEnv.__new__(FleetEnv)
        fleet.build_task = TaskSpec(name="t", control_mode="joint_position")
        with self.assertRaises(NotImplementedError) as caught:
            fleet._check_documents([fleet.build_task, TaskSpec(name="t", control_mode="ee_delta")])
        self.assertIsInstance(caught.exception, CannotReplicate)


class TestTheEmptyContextIsCoveredByTheRefusal(unittest.TestCase):
    """`FleetEnv.context` fills no `particles`, and only per-world `entities`, safely.

    Safely only because `fleet.scoring_refusal` declines every scene that would
    read anything else; an entity is handed over only when it declares
    `world_aware` (the gear board: per-world latch and spin windows, and
    a `published_states` that reads `WorldState.world`). Lifting the rest with
    `entities` short would take every published state with it: `bind._published`
    reads that dict, so a peel would raise `BindError` on every episode.
    Both refusals are pure functions of the entities, so they are asked here of
    entities that were never built.
    """

    def test_the_refusal_still_declines_scenes_with_deformables(self):
        """It asks what a thing *is*, not which view held it.

        The clause read ``scene.deformables`` -- a bin -- so a publisher that
        owned a handle sat in ``objects`` and slipped past it. Both halves ask
        the entity now, which is what this pins: the scene below has no bins at
        all, only ``by_name``."""
        from robosimstudio_bench.envs.fleet import scoring_refusal

        towel = Entity(SimpleNamespace(name="towel"))
        towel.range.particles = (0, 64)
        self.assertIn("made of particles (towel)", scoring_refusal(SimpleNamespace(by_name={"towel": towel})))
        film = _Publisher(SimpleNamespace(name="film"))
        self.assertIn("publishing states (film)", scoring_refusal(SimpleNamespace(by_name={"film": film})))
        # And the one way through: a publisher that answers per world says so.
        board = _WorldAwarePublisher(SimpleNamespace(name="board"))
        self.assertEqual(scoring_refusal(SimpleNamespace(by_name={"board": board})), "")

    def test_the_compilers_refusal_is_about_the_physics_alone(self):
        """The two refusals answer different questions and must stay apart.

        A deformable's physics replicates -- four worlds to within 2e-10 -- so
        the compiler has nothing to say about it; what cannot be replicated is a
        substep kernel with world-0 indices baked into it. Scoring is this
        class's problem, and saying so twice would leave the fleet's own gap
        depending on the compiler remembering it.
        """
        towel = Entity(SimpleNamespace(name="towel"))
        towel.range.particles = (0, 64)
        no_physics = SimpleNamespace(physics=None)
        self.assertEqual(compiler._replication_refusal(SimpleNamespace(entities=[towel]), no_physics), "")
        hooked = _Hooked(SimpleNamespace(name="cutter"))
        refusal = compiler._replication_refusal(SimpleNamespace(entities=[towel, hooked]), no_physics)
        self.assertIn("substep physics written for one world: cutter", refusal)

    def test_a_cutting_block_is_refused_for_what_its_goal_reads(self):
        """The cutter's published ``pieces``, not the eight corners its block is drawn at.

        Asked of the particles first, a one-world wave of
        ``examples/06_tasks/09_cutting_scored`` was refused for the corners, with
        "the physics replicates correctly", and the cutter's does not: at two
        worlds the compiler refuses its one-world substep hooks.
        """
        from tests.fixtures.packs import has_pack

        if not has_pack("cutting"):
            self.skipTest("the cutting pack is not installed")
        import robosimstudio as rss
        from robosimstudio_bench.envs.fleet import scoring_refusal
        from robosimstudio_materials.cutting.entity import EntityCuttable

        tofu = EntityCuttable(rss.cuttable(name="tofu", knife=rss.knife(name="knife"), size=(0.06, 0.1, 0.04)))
        tofu.range.particles = (0, 8)  # the corners, as its build adds them
        self.assertTrue(tofu.carries_particles)
        refusal = scoring_refusal(SimpleNamespace(by_name={"tofu": tofu}))
        self.assertIn("publishing states (tofu)", refusal)
        self.assertNotIn("replicates correctly", refusal)


if __name__ == "__main__":
    unittest.main()
