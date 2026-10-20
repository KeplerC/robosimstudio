# SPDX-License-Identifier: Apache-2.0
"""``sim.attach``/``sim.detach``: the deformable hold three suites each wrote.

``latch_cloth``/``release_cloth``, ``latch_tape``/``release_tape`` and
``hold_cable``/``release_cable`` are one tool: a pinning ``ParticleDragger`` at
a hand, re-aimed every control step, over a declared particle set. Their return
shapes already agreed. The only thing that differed was which particles, and a
feature name says that.

What this guards is not the physics -- that needs a scene, and the suites'
own rollouts exercise it -- but the three properties a copy is likely to get
wrong when the seventh caller arrives:

* the tool appears only where there is something to hold, so a rigid cell does
  not offer a graph a tool that raises the moment it is believed -- nor does a
  cuttable's cell, whose eight particles are the corners its cage writes back
  for a page to draw, with no probe, the way ``Sim.deformables`` picks;
* both entries are tagged privileged, because seizing particles has no
  real-robot analogue and a graph that attaches will not transfer;
* the hook is registered **and the CUDA graph invalidated**, in that order.
  Getting the second wrong is silent: the grab bookkeeping is host-side, so the
  tool reports the particles it seized while nothing holds them.

``sim.get_particle_state`` picks the same way: offered the cuttable first, it
raised ``AttributeError`` on its ``.probe``. And ``sim.attach`` with no entity
took the task's first ``deformables``-stage spec, where a cuttable is staged, so
a tofu declared before a towel had the hand pin its cage's corners and report
``attached``; now both tools and ``Sim.deformables`` share one rule,
:func:`~robosimstudio.sim.sim.scene_deformables`. A name off that list is a
``ToolRefusal`` naming the ones there are, not a bare ``KeyError``, and one world
of a fleet reads its own world's probe, not world 0's.

No GPU: a stub env is the whole fixture. It has no ``deformables`` of its own,
as one world of a fleet (a ``WorldView``) has none.
"""

from __future__ import annotations

import types
import unittest
import unittest.mock

from robosimstudio.sim import Sim
from robosimstudio_policy_agent.sim_tools import ClothAttachment, SimTools, ToolRefusal, register_attachment

TIER = "fast"
"""No GPU. Gates every commit."""


class _Registry:
    """Records what was claimed and under which tags."""

    def __init__(self) -> None:
        self.calls: dict[str, dict] = {}

    def register_callable(self, name, fn, **kwargs):
        if name in self.calls:
            raise AssertionError(f"duplicate registration of {name!r}")
        self.calls[name] = {"fn": fn, **kwargs}
        return fn


class _Bus:
    """This thread owns the env, so a call is just a call."""

    def call(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


def _env(*, deformables: bool, cuttable: bool = False) -> types.SimpleNamespace:
    """A tape carrying a probe when ``deformables``, behind a cuttable's eight corners when ``cuttable``."""
    probe = types.SimpleNamespace(summary=lambda key: {"set": key, "z_min": 0.10})  # a summary, the band of one
    thing = types.SimpleNamespace(carries_particles=True, probe=probe)
    tofu = types.SimpleNamespace(carries_particles=True, particles=list(range(8)))
    by_name = {**({"tofu": tofu} if cuttable else {}), **({"tape": thing} if deformables else {})}
    return types.SimpleNamespace(scene=types.SimpleNamespace(by_name=by_name, embodiment=None), handles=None)


class TestItAppearsOnlyWhereItApplies(unittest.TestCase):
    def test_a_scene_with_a_deformable_gets_both_entries(self) -> None:
        registry = _Registry()
        attachment = register_attachment(registry, _env(deformables=True), _Bus())
        self.assertIsInstance(attachment, ClothAttachment)
        self.assertEqual(sorted(registry.calls), ["sim.attach", "sim.detach"])

    def test_a_rigid_scene_gets_neither(self) -> None:
        registry = _Registry()
        self.assertIsNone(register_attachment(registry, _env(deformables=False), _Bus()))
        self.assertEqual(registry.calls, {})

    def test_a_cuttable_alone_gets_neither(self) -> None:
        registry = _Registry()
        self.assertIsNone(register_attachment(registry, _env(deformables=False, cuttable=True), _Bus()))
        self.assertEqual(registry.calls, {}, "the tofu's corners are nothing a pin moves")

    def test_both_entries_are_privileged(self) -> None:
        registry = _Registry()
        register_attachment(registry, _env(deformables=True), _Bus())
        for name, call in registry.calls.items():
            self.assertIn("privileged", tuple(call.get("tags") or ()), f"{name} is not tagged privileged")


class TestTheParticleStateReadsADeformable(unittest.TestCase):
    def test_the_cuttable_is_passed_over(self) -> None:
        got = SimTools(_env(deformables=True, cuttable=True), _Bus()).get_particle_state()
        self.assertEqual("tape", got["cloth"], "the tofu came first and has no probe")
        self.assertEqual(0.10, got["z_min"])

    def test_a_cuttable_alone_is_no_deformable(self) -> None:
        with self.assertRaises(ToolRefusal):
            SimTools(_env(deformables=False, cuttable=True), _Bus()).get_particle_state()


class TestOneWorldOfAFleetReadsItsOwnParticles(unittest.TestCase):
    """A ``WorldView`` carries its ``world``; the tool read ``entity.probe``, world 0's, for every world."""

    def _view(self, world: int) -> types.SimpleNamespace:
        probes = [types.SimpleNamespace(summary=lambda key, z=z: {"set": key, "z_min": z}) for z in (0.10, 0.30)]
        towel = types.SimpleNamespace(carries_particles=True, probe=probes[0], probe_for=lambda w: probes[w])
        scene = types.SimpleNamespace(by_name={"towel": towel}, embodiment=None)
        return types.SimpleNamespace(scene=scene, handles=None, world=world)

    def test_world_one_reads_world_ones_towel(self) -> None:
        self.assertEqual(0.30, SimTools(self._view(1), _Bus()).get_particle_state()["z_min"], "world 0's towel")

    def test_world_zero_still_reads_its_own(self) -> None:
        self.assertEqual(0.10, SimTools(self._view(0), _Bus()).get_particle_state()["z_min"])

    def test_an_env_with_no_world_reads_world_zero(self) -> None:
        self.assertEqual(0.10, SimTools(_env(deformables=True), _Bus()).get_particle_state()["z_min"])


class TestAnUnknownDeformableIsRefused(unittest.TestCase):
    """The refusal ``get_rgbd`` gives an unknown camera: the name, and the ones there are."""

    def test_the_cuttable_by_name_is_refused_with_the_list(self) -> None:
        with self.assertRaisesRegex(ToolRefusal, r"no deformable 'tofu'; have \['tape'\]"):
            SimTools(_env(deformables=True, cuttable=True), _Bus()).get_particle_state(cloth="tofu")

    def test_a_typo_is_refused(self) -> None:
        with self.assertRaisesRegex(ToolRefusal, r"no deformable 'towle'"):
            SimTools(_env(deformables=True), _Bus()).get_particle_state(cloth="towle")


class TestTheHookOrdering(unittest.TestCase):
    """The invalidation has to follow the append, or the pin never launches."""

    def _scene(self) -> tuple[types.SimpleNamespace, list[str]]:
        order: list[str] = []
        dragger_state = types.SimpleNamespace(grabbed=0)

        class _Dragger:
            radius = 0.0
            last_distance = 0.25
            active = False

            def __init__(self, *_a, **_k) -> None:
                pass

            def apply(self, *_a, **_k) -> None:
                pass

            def grab(self, *_a, **_k) -> int:
                dragger_state.grabbed += 1
                return 4

            def release(self) -> None:
                order.append("release")

        hooks = types.SimpleNamespace(post_solve=_Recorder(order, "append-hook"))
        pipeline = types.SimpleNamespace(hooks=hooks, invalidate_graph=lambda: order.append("invalidate"))
        arm = types.SimpleNamespace(ee_pose=lambda _state: ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
        env = types.SimpleNamespace(
            model=object(),
            state=object(),
            pipeline=pipeline,
            post_step_hooks=[],
            handles=types.SimpleNamespace(arms={"left": object()}, work_arm="left"),
            task=types.SimpleNamespace(deformables=[types.SimpleNamespace(name="tape")]),
            scene=types.SimpleNamespace(
                by_name={"tape": types.SimpleNamespace(particles=[0, 1, 2, 3], carries_particles=True, probe=object())}
            ),
            arm=lambda _name: arm,
        )
        return env, order, _Dragger

    def _behind_a_cuttable(self, seized: list) -> tuple[types.SimpleNamespace, type]:
        """The cutting spec's stage bin: a tofu declared before a towel, each staged as a deformable."""
        env, _order, dragger = self._scene()
        tofu = types.SimpleNamespace(name="tofu", carries_particles=True, particles=list(range(8)))
        towel = types.SimpleNamespace(name="towel", carries_particles=True, particles=[8, 9, 10], probe=object())
        env.task = types.SimpleNamespace(deformables=[tofu, towel])
        env.scene = types.SimpleNamespace(by_name={"tofu": tofu, "towel": towel}, embodiment=None)
        dragger.grab = lambda self, *a, candidates=(), **k: seized.append(list(candidates)) or len(candidates)  # type: ignore[assignment]
        return env, dragger

    def test_no_entity_holds_the_deformable_the_tools_list(self) -> None:
        seized: list = []
        env, dragger = self._behind_a_cuttable(seized)
        with unittest.mock.patch("robosimstudio.core.draggers.ParticleDragger", dragger):
            result = ClothAttachment(env, _Bus()).attach()
        self.assertEqual("towel", result["entity"], "the tofu's cage corners, attached=True and holding nothing")
        self.assertEqual([[8, 9, 10]], seized)
        self.assertEqual(["towel"], list(Sim.deformables(env)), "the rule Sim.deformables answers by")

    def test_the_cuttable_by_name_is_refused(self) -> None:
        env, dragger = self._behind_a_cuttable([])
        with unittest.mock.patch("robosimstudio.core.draggers.ParticleDragger", dragger):
            with self.assertRaisesRegex(ToolRefusal, r"no deformable 'tofu'; have \['towel'\]"):
                ClothAttachment(env, _Bus()).attach(entity="tofu")

    def test_the_graph_is_invalidated_after_the_hook_is_appended(self) -> None:
        env, order, dragger = self._scene()
        attachment = ClothAttachment(env, _Bus())
        with unittest.mock.patch("robosimstudio.core.draggers.ParticleDragger", dragger):
            result = attachment.attach()
        self.assertEqual(order, ["append-hook", "invalidate"], "the pin's kernel is captured before it exists")
        self.assertTrue(result["attached"])
        self.assertEqual(result["n_grabbed"], 4)

    def test_a_miss_reports_the_distance_that_missed(self) -> None:
        env, _order, dragger = self._scene()
        dragger.grab = lambda self, *a, **k: 0  # type: ignore[assignment]
        attachment = ClothAttachment(env, _Bus())
        with unittest.mock.patch("robosimstudio.core.draggers.ParticleDragger", dragger):
            result = attachment.attach()
        self.assertFalse(result["attached"])
        self.assertEqual(result["n_grabbed"], 0)
        self.assertGreater(result["nearest_m"], 0.0, "a miss that does not say how far off it was is not routable")


class _Recorder(list):
    """A list that records the fact of an append, for ordering assertions."""

    def __init__(self, order: list[str], label: str) -> None:
        super().__init__()
        self._order, self._label = order, label

    def append(self, item) -> None:
        self._order.append(self._label)
        super().append(item)


if __name__ == "__main__":
    unittest.main()
