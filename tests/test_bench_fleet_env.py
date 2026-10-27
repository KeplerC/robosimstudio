# SPDX-License-Identifier: Apache-2.0
"""``FleetEnv``'s refusals, made before anything is built: an undeclared pairing, a wave
whose documents are different models, and a scene whose scoring cannot be read per world;
and what counts as different physics and how a staged value is named in a refusal."""

import dataclasses
import unittest
from types import SimpleNamespace

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


class _Probed(Entity):
    """Particles, and a probe opened per world over them."""

    def probe_for(self, world: int):
        return None


def _entity(cls, name: str, particles: int = 0):
    entity = cls(SimpleNamespace(name=name))
    entity.range.particles = (0, particles)
    return entity


class TestWhatAWaveCanScore(unittest.TestCase):
    """``scoring_refusal``: the branches beyond the plain refusals ``test_fleet`` pins."""

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


if __name__ == "__main__":
    unittest.main()
