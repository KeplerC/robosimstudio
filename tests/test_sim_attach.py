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

import numpy as np

from robosimstudio.core.worlds import WorldMap
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
    probe = types.SimpleNamespace(
        summary=lambda key: {"set": key, "z_min": 0.10}, update=lambda _state, *_step, **_rows: None
    )
    thing = types.SimpleNamespace(carries_particles=True, probe=probe)
    tofu = types.SimpleNamespace(carries_particles=True, particles=list(range(8)))
    by_name = {**({"tofu": tofu} if cuttable else {}), **({"tape": thing} if deformables else {})}
    scene = types.SimpleNamespace(by_name=by_name, embodiment=None)
    return types.SimpleNamespace(scene=scene, handles=None, state=None)


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
        probes = [
            types.SimpleNamespace(
                summary=lambda key, z=z: {"set": key, "z_min": z}, update=lambda _state, *_step, **_rows: None
            )
            for z in (0.10, 0.30)
        ]
        towel = types.SimpleNamespace(carries_particles=True, probe=probes[0], probe_for=lambda w: probes[w])
        scene = types.SimpleNamespace(by_name={"towel": towel}, embodiment=None)
        return types.SimpleNamespace(scene=scene, handles=None, world=world, state=None)

    def test_world_one_reads_world_ones_towel(self) -> None:
        self.assertEqual(0.30, SimTools(self._view(1), _Bus()).get_particle_state()["z_min"], "world 0's towel")

    def test_world_zero_still_reads_its_own(self) -> None:
        self.assertEqual(0.10, SimTools(self._view(0), _Bus()).get_particle_state()["z_min"])

    def test_an_env_with_no_world_reads_world_zero(self) -> None:
        self.assertEqual(0.10, SimTools(_env(deformables=True), _Bus()).get_particle_state()["z_min"])


class TestOneWorldOfAFleetHoldsItsOwnParticles(unittest.TestCase):
    """``sim.attach`` on world 1 of a fleet pins world 1's particles, and world 0's do not move.

    The candidates are the template's indices. ``WorldView.state`` slices world 1's particles out
    for the grab to read, so they find world 1's sheet; the pin then runs on the fleet's whole
    state, where those indices are world 0's rows. On a real two-world granular wave
    ``attach("rice")`` on world 1 seized eight rows under the 16000-particle stride and moved world
    0's rice 13 mm toward world 1's hand in one step.

    A real ``WorldView`` and a real pinning dragger on the CPU, over two worlds of four particles
    that share space, as a wave's worlds do: world 1's sheet sits 50 mm over world 0's.
    """

    PER_WORLD = 4

    def setUp(self) -> None:
        import warp as wp

        from robosimstudio_bench.envs.world_view import WorldView

        sheet = np.array([[0.0, 0.0, 0.10], [0.01, 0.0, 0.10], [0.0, 0.01, 0.10], [0.01, 0.01, 0.10]])
        rows = np.concatenate([sheet, sheet + np.array([0.0, 0.0, 0.05])]).astype(np.float32)
        self.state = types.SimpleNamespace(
            particle_q=wp.array(rows, dtype=wp.vec3, device="cpu"),
            particle_qd=wp.zeros(len(rows), dtype=wp.vec3, device="cpu"),
        )
        self.hand = [(0.005, 0.005, 0.155), (0.0, 0.0, 0.0, 1.0)]  # over world 1's sheet, 5 mm up
        ik = types.SimpleNamespace(ee_pose=lambda _state: (self.hand[0], self.hand[1]))
        towel = types.SimpleNamespace(carries_particles=True, probe=object(), particles=list(range(self.PER_WORLD)))
        handles = types.SimpleNamespace(arms={"left": object()}, work_arm="left")
        self.fleet = types.SimpleNamespace(
            model=types.SimpleNamespace(device="cpu"),
            pipeline=types.SimpleNamespace(
                state_0=self.state, hooks=types.SimpleNamespace(post_solve=[]), invalidate_graph=lambda: None
            ),
            scene=types.SimpleNamespace(by_name={"towel": towel}, embodiment=types.SimpleNamespace(iks={"left": ik})),
            worlds=WorldMap(2, 0, 0, 0, 0, 0, self.PER_WORLD),
            handles_for=lambda _world: handles,
        )
        self.view = WorldView(self.fleet, 1)
        self.before = rows.copy()

    def _attach(self) -> tuple[dict, ClothAttachment]:
        attachment = ClothAttachment(self.view, _Bus())
        return attachment.attach("towel", radius=0.02, max_particles=8), attachment

    def test_it_seizes_world_ones_rows(self) -> None:
        result, attachment = self._attach()
        self.assertTrue(result["attached"])
        held = attachment.draggers["left"].grabbed_indices.tolist()
        self.assertEqual(self.PER_WORLD, len(held))
        self.assertTrue(all(self.PER_WORLD <= row < 2 * self.PER_WORLD for row in held), f"world 0's rows {held}")

    def test_world_zero_does_not_move_and_world_one_follows_the_hand(self) -> None:
        _result, attachment = self._attach()
        self.hand[0] = (0.005, 0.005, 0.165)  # the hand lifts 10 mm
        for hook in list(self.view.post_step_hooks):
            hook(self.view)
        for _ in range(40):  # 0.5 mm a substep at most, so 20 substeps carry the 10 mm
            for apply in self.fleet.pipeline.hooks.post_solve:
                apply(self.state, 1.0 / 600.0)
        after = self.state.particle_q.numpy()
        np.testing.assert_array_equal(self.before[: self.PER_WORLD], after[: self.PER_WORLD], "world 0's sheet moved")
        lift = after[self.PER_WORLD :, 2] - self.before[self.PER_WORLD :, 2]
        np.testing.assert_allclose(lift, 0.010, atol=1e-5, err_msg="world 1's sheet did not ride the hand")
        attachment.detach()
        self.assertFalse(attachment.draggers["left"].active)


class TestTheParticleStateIsReadNow(unittest.TestCase):
    """``sim.get_particle_state`` reads the particles as they stand when it is called.

    It read the probe's summary as the probe last held it, and nothing refreshes a probe on a
    tool's behalf: a fleet refreshes world k's only when it scores that world, every ``eval_every``
    steps, and a bare ``Sim``'s holds what ``snapshot_rest`` left. So a sheet that had moved read
    where it was at the last score. Real probes over a CPU state: the particles move with nothing
    scored, then the tool reads.
    """

    def _state(self, rows: np.ndarray) -> types.SimpleNamespace:
        import warp as wp

        return types.SimpleNamespace(
            particle_q=wp.array(rows.astype(np.float32), dtype=wp.vec3, device="cpu"),
            particle_qd=wp.zeros(len(rows), dtype=wp.vec3, device="cpu"),
        )

    def _probes(self, worlds: int, per_world: int) -> list:
        from robosimstudio.core.probes.particle import ParticleProbe

        model = types.SimpleNamespace(particle_count=worlds * per_world)
        probes = []
        for world in range(worlds):
            probe = ParticleProbe(model, world=world)
            probe.add_set("towel", np.arange(per_world) + world * per_world)
            probes.append(probe)
        return probes

    def test_an_env_reads_the_sheet_where_it_is(self) -> None:
        state = self._state(np.zeros((4, 3)))
        (probe,) = self._probes(1, 4)
        probe.update(state)  # the last score, with the sheet at z = 0
        state.particle_q.assign(np.full((4, 3), 0.25, dtype=np.float32))  # moved, nothing scored
        towel = types.SimpleNamespace(carries_particles=True, probe=probe, probe_for=lambda _w: probe)
        scene = types.SimpleNamespace(by_name={"towel": towel}, embodiment=None)
        env = types.SimpleNamespace(scene=scene, handles=None, state=state)
        self.assertAlmostEqual(0.25, SimTools(env, _Bus()).get_particle_state()["centroid_z"], places=6)

    def test_world_one_of_a_fleet_reads_its_sheet_where_it_is(self) -> None:
        from robosimstudio_bench.envs.world_view import WorldView

        state = self._state(np.zeros((8, 3)))
        probes = self._probes(2, 4)
        for probe in probes:
            probe.update(state)  # each world's last score
        rows = np.zeros((8, 3), dtype=np.float32)
        rows[4:, 2] = 0.30  # world 1's sheet lifted 300 mm since
        state.particle_q.assign(rows)
        towel = types.SimpleNamespace(carries_particles=True, probe=probes[0], probe_for=lambda w: probes[w])
        fleet = types.SimpleNamespace(
            pipeline=types.SimpleNamespace(state_0=state),
            scene=types.SimpleNamespace(by_name={"towel": towel}, embodiment=None),
            worlds=WorldMap(2, 0, 0, 0, 0, 0, 4),
            handles_for=lambda _world: None,
        )
        got = [SimTools(WorldView(fleet, w), _Bus()).get_particle_state()["centroid_z"] for w in (0, 1)]
        np.testing.assert_allclose([0.0, 0.30], got, atol=1e-6)

    def test_a_fleet_world_copies_its_own_rows_once_a_step(self) -> None:
        """It copied the whole model's particles, every world's, under the stream lock, to summarise one
        world's set, and read them with no step, so the fleet's own read of that world at the same step
        copied them all again. Counted here by the rows each copy to the host carries."""
        from robosimstudio_bench.envs.world_view import WorldView

        copied: list[int] = []

        class Counted:
            """A device array whose copies to the host are counted, by rows."""

            def __init__(self, array) -> None:
                self.array = array

            def __getitem__(self, rows):
                return Counted(self.array[rows])

            def numpy(self) -> np.ndarray:
                copied.append(len(self.array))
                return self.array.numpy()

        rows = np.zeros((8, 3), dtype=np.float32)
        rows[4:, 2] = 0.30  # world 1's sheet
        device = self._state(rows)
        state = types.SimpleNamespace(particle_q=Counted(device.particle_q), particle_qd=Counted(device.particle_qd))
        probes = self._probes(2, 4)
        towel = types.SimpleNamespace(carries_particles=True, probe=probes[0], probe_for=lambda w: probes[w])
        fleet = types.SimpleNamespace(
            pipeline=types.SimpleNamespace(state_0=state),
            scene=types.SimpleNamespace(by_name={"towel": towel}, embodiment=None),
            worlds=WorldMap(2, 0, 0, 0, 0, 0, 4),
            handles_for=lambda _world: None,
            step_count=7,
        )
        tools = SimTools(WorldView(fleet, 1), _Bus())
        self.assertAlmostEqual(0.30, tools.get_particle_state()["centroid_z"], places=6)
        self.assertEqual([4, 4], copied, "world 1's positions and velocities, 4 rows each, and nothing else")
        probes[1].update(state, 7)  # the fleet scoring world 1 at the same step
        self.assertAlmostEqual(0.30, tools.get_particle_state()["centroid_z"], places=6)
        self.assertEqual([4, 4], copied, "copied again at the step it was read at")
        fleet.step_count = 8
        tools.get_particle_state()
        self.assertEqual([4, 4, 4, 4], copied, "the next step is read")


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
