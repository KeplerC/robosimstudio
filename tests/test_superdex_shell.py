# SPDX-License-Identifier: Apache-2.0
"""One SuperDex shell builder: every sheet about its own centroid, its contact and damping its own.

The engine is float32 and solves in the coordinates it is given. The bag was built about its
centroid, with the offset in its root transform, after a bag in bench-height world coordinates
never converged; the bridge's own shell builder, which every cloth on SuperDex goes through, kept
building in world coordinates at an identity root, read back without the root, and took the
engine's contact (mu 0.5) whatever the cloth said. The bag's builder was a copy of it with those
three things fixed. Now the bridge's builder is the one (``SuperdexBridge.build_shell``) and the
bag's is that builder under its own defaults.

Checked against a stand-in engine, so no GPU and no ``[superdex]`` wheel: the rest goes in about
the centroid and the root is the centroid, the nodes read back through the root, a cloth's friction
and falloff reach the contact while an unstated key keeps the engine's own, damping is set only when
stated, a kind's defaults sit under the record's own, and the declarations name every key the
builder now reads. ``tests/test_superdex_sheet.py`` builds the same on the engine.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

TIER = "fast"
"""A stand-in engine and model. No GPU, no wheel."""

ENGINE_MU, ENGINE_FALLOFF, ENGINE_PENALTY = 0.5, 0.01, 1.0e9
"""The engine's own contact, as a shell is built with when nothing is said (``p1_facts.py``)."""


class _Arr:
    def __init__(self, a) -> None:
        self.a = np.asarray(a)

    def numpy(self):
        return self.a


class _Quaternion:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0) -> None:
        self.xyzw = np.array([x, y, z, w])

    def tolist(self) -> list[float]:
        return self.xyzw.tolist()


class _Transform:
    def __init__(self, rotation=None, translation=(0.0, 0.0, 0.0)) -> None:
        self.rotation = rotation if rotation is not None else _Quaternion()
        self.translation = list(translation)


class _Contact:
    def __init__(
        self,
        penalty_coefficient: float = ENGINE_PENALTY,
        coulomb_friction_coefficient: float = ENGINE_MU,
        penalty_threshold_extra_padding: float = 0.0,
    ) -> None:
        self.penalty_coefficient = penalty_coefficient
        self.coulomb_friction_coefficient = coulomb_friction_coefficient
        self.penalty_threshold_extra_padding = penalty_threshold_extra_padding
        self.friction_falloff_vel = ENGINE_FALLOFF
        self.penalty_threshold_default = 1.0e-3
        self.penalty_smoothing_half_distance = 5.0e-3


class _Shell:
    """``ShellActorParams``: what the builder hands the engine."""

    def __init__(self, *, name, shape, material, world_from_local) -> None:
        self.name, self.shape, self.material, self.world_from_local = name, shape, material, world_from_local
        self.point_cloud_collider = SimpleNamespace(radius=0.0, self_contact=False)
        self.contact = _Contact()


class _EngineError(RuntimeError):
    """``superdex.physics.Error``."""


class _Actor:
    """A built shell: its root is what it was built at, and its local nodes are its rest, moved by ``drift``."""

    def __init__(self, params: _Shell) -> None:
        self.params = params
        self.drift = np.zeros(3)
        self.root_reads = 0

    def get_root_transform(self) -> _Transform:
        self.root_reads += 1
        return self.params.world_from_local

    def get_node_positions_local(self) -> np.ndarray:
        return (np.asarray(self.params.shape.coordinates).reshape(-1, 3) + self.drift).ravel()

    def register_query(self, query) -> None:
        del query

    def set_recentering_params(self, params) -> None:
        """As the engine answers for a shell: it has no recentering to set (``p1_facts.py``)."""
        del params
        raise _EngineError("CRecenteringParams component not found")


def _engine() -> SimpleNamespace:
    """Just enough of ``superdex.physics`` to build a shell and say what it was handed."""
    built: list[_Actor] = []

    def create_shell_actor(scene, params):
        del scene
        built.append(_Actor(params))
        return built[-1]

    def material(young, poisson, density, thickness):
        return SimpleNamespace(
            young=young,
            poisson=poisson,
            density=density,
            thickness=thickness,
            mass_damping_coefficient=0.0,
            stiffness_damping_coefficient=0.0,
        )

    return SimpleNamespace(
        built=built,
        Error=_EngineError,
        RecenteringParams=lambda use_recentering: SimpleNamespace(use_recentering=use_recentering),
        TransformRT=_Transform,
        Quaternion=_Quaternion,
        ContactParams=_Contact,
        QueryType=SimpleNamespace(NODE_POSITIONS="node_positions"),
        uses_double_precision=lambda: False,
        create_tri_mesh_shape=lambda coordinates, connectivity: SimpleNamespace(
            coordinates=np.asarray(coordinates), connectivity=np.asarray(connectivity)
        ),
        experimental=SimpleNamespace(
            shell_material_params_from3d_isotropic=material,
            ShellActorParams=_Shell,
            create_shell_actor=create_shell_actor,
        ),
    )


def _sheet(at=(0.6, 0.0, 0.75), n: int = 4, size: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """A flat sheet of ``n`` x ``n`` cells, its corner at ``at``: vertices and triangles."""
    xs = np.linspace(0.0, size, n + 1)
    vertices = np.array([(x, y, 0.0) for x in xs for y in xs]) + np.asarray(at)
    tris = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            b = a + n + 1
            tris += [(a, b, a + 1), (b, b + 1, a + 1)]
    return vertices, np.asarray(tris)


def _bridge(vertices: np.ndarray, faces: np.ndarray, **recipe):
    """A one-world bridge over a model that holds only this sheet, on the stand-in engine."""
    from robosimstudio.engines.superdex.bridge import SuperdexBridge

    bridge = object.__new__(SuperdexBridge)
    bridge.physics = _engine()
    bridge.model = SimpleNamespace(particle_q=_Arr(vertices), tri_indices=_Arr(faces), tri_count=len(faces))
    bridge.recipe = {"shell_young": 1.0e5, "shell_poisson": 0.3, **recipe}
    bridge.scenes, bridge.num_worlds = [object()], 1
    return bridge


def _record(vertices: np.ndarray, **params) -> dict:
    return {"name": "towel", "kind": "shell", "particle_start": 0, "particle_count": len(vertices), "params": params}


def _shell_builder():
    from robosimstudio.engines.geometry import actor_builder

    return actor_builder("superdex", "shell")


class TestASheetIsBuiltAboutItsCentroid(unittest.TestCase):
    """Through the builder the registry holds for ``shell``: the one every cloth on SuperDex takes."""

    def setUp(self) -> None:
        self.vertices, self.faces = _sheet()
        self.bridge = _bridge(self.vertices, self.faces)
        self.entry = _shell_builder()(self.bridge, _record(self.vertices, friction=0.2), None, None, None)
        self.actor = self.bridge.physics.built[-1]

    def test_the_rest_goes_in_about_the_centroid_and_the_centroid_is_the_root(self) -> None:
        centre = self.vertices.mean(axis=0)
        local = self.actor.params.shape.coordinates.reshape(-1, 3)
        np.testing.assert_allclose(local, self.vertices - centre, atol=1e-12)
        np.testing.assert_allclose(self.actor.params.world_from_local.translation, centre, atol=1e-12)
        np.testing.assert_allclose(self.actor.params.world_from_local.rotation.tolist(), [0.0, 0.0, 0.0, 1.0])

    def test_its_nodes_read_back_in_the_world_through_its_root(self) -> None:
        self.actor.drift = np.array([0.0, 0.01, -0.002])
        np.testing.assert_allclose(self.entry["read"](), self.vertices + self.actor.drift, atol=1e-12)
        # As a placement writes them: a root turned a quarter about z turns what it reads.
        half = math.sqrt(0.5)
        make = self.bridge.physics.experimental.create_shell_actor

        def turned(scene, params):
            params.world_from_local = _Transform(_Quaternion(0.0, 0.0, half, half), (0.6, 0.0, 0.75))
            params.shape.coordinates = np.tile([0.01, 0.0, 0.0], len(self.vertices))
            return make(scene, params)

        self.bridge.physics.experimental.create_shell_actor = turned
        entry = _shell_builder()(self.bridge, _record(self.vertices), None, None, None)
        np.testing.assert_allclose(entry["read"](), np.tile([0.6, 0.01, 0.75], (len(self.vertices), 1)), atol=1e-12)

    def test_its_root_is_read_as_it_stands_at_every_read(self) -> None:
        """Held from the build, a root moved later read stale; read every call, the cost is inside the control
        step's spread (``c4_root.py``)."""
        before = self.actor.root_reads
        for _ in range(5):
            self.entry["read"]()
        self.assertEqual(before + 5, self.actor.root_reads)
        self.actor.params.world_from_local = _Transform(translation=(0.0, 0.0, 0.1))
        moved = self.vertices - self.vertices.mean(axis=0) + np.array([0.0, 0.0, 0.1])
        np.testing.assert_allclose(self.entry["read"](), moved, atol=1e-12)

    def test_the_rest_it_reports_is_the_world_rest(self) -> None:
        """``prev`` is what the first tick's speed and every weld's nearest node are measured from."""
        np.testing.assert_allclose(self.entry["prev"], self.vertices, atol=1e-6)
        self.assertEqual((0, len(self.vertices)), (self.entry["start"], self.entry["count"]))


class TestItsContactAndDampingAreItsOwn(unittest.TestCase):
    def _build(self, defaults=None, **params):
        vertices, faces = _sheet()
        bridge = _bridge(vertices, faces)
        if defaults is None:
            _shell_builder()(bridge, _record(vertices, **params), None, None, None)
        else:
            bridge.build_shell(_record(vertices, **params), defaults=defaults)
        return bridge.physics.built[-1].params

    def test_a_cloths_friction_and_falloff_reach_the_engine(self) -> None:
        """A towel at the fabric default of 0.2 was built at the engine's 0.5."""
        contact = self._build(friction=0.2, friction_falloff=2.0e-3).contact
        self.assertAlmostEqual(0.2, contact.coulomb_friction_coefficient)
        self.assertAlmostEqual(2.0e-3, contact.friction_falloff_vel)
        self.assertAlmostEqual(ENGINE_PENALTY, contact.penalty_coefficient, msg="unstated: the engine's own")

    def test_a_sheet_that_says_nothing_keeps_the_engines_contact(self) -> None:
        contact = self._build().contact
        self.assertEqual(
            (ENGINE_MU, ENGINE_FALLOFF, ENGINE_PENALTY, 0.0),
            (
                contact.coulomb_friction_coefficient,
                contact.friction_falloff_vel,
                contact.penalty_coefficient,
                contact.penalty_threshold_extra_padding,
            ),
        )

    def test_damping_is_set_when_stated_and_left_at_the_engines_otherwise(self) -> None:
        stated = self._build(mass_damping=2.0, stiffness_damping=1.0e-3).material
        self.assertEqual((2.0, 1.0e-3), (stated.mass_damping_coefficient, stated.stiffness_damping_coefficient))
        unstated = self._build().material
        self.assertEqual((0.0, 0.0), (unstated.mass_damping_coefficient, unstated.stiffness_damping_coefficient))

    def test_a_kinds_defaults_sit_under_the_records_own(self) -> None:
        defaults = {"friction": 0.7, "particle_radius": 4.0e-3, "self_contact": True, "young": 1.0e6}
        params = self._build(defaults=defaults, friction=0.3)
        self.assertAlmostEqual(0.3, params.contact.coulomb_friction_coefficient, msg="the record's own wins")
        self.assertEqual((4.0e-3, True), (params.point_cloud_collider.radius, params.point_cloud_collider.self_contact))
        self.assertEqual(1.0e6, params.material.young, "a kind's modulus over the island's")


class TestTheBagIsTheSameBuilder(unittest.TestCase):
    def test_the_bag_builds_through_the_shell_builder_under_its_canvas(self) -> None:
        """Its modulus under the shell's names, its canvas where its record says nothing."""
        from robosimstudio.engines.geometry import actor_builder
        from robosimstudio_materials.bag import CANVAS

        vertices, faces = _sheet()
        bridge = _bridge(vertices, faces)
        record = {**_record(vertices, young=2.0e6), "kind": "bag"}
        entry = actor_builder("superdex", "bag")(bridge, record, None, None, None)
        params = bridge.physics.built[-1].params
        self.assertEqual(2.0e6, params.material.young)
        self.assertAlmostEqual(CANVAS.friction, params.contact.coulomb_friction_coefficient)
        self.assertEqual(
            (CANVAS.particle_radius, True),
            (params.point_cloud_collider.radius, params.point_cloud_collider.self_contact),
        )
        np.testing.assert_allclose(params.world_from_local.translation, vertices.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(entry["read"](), vertices, atol=1e-12)


class TestTheDeclarations(unittest.TestCase):
    def test_the_shell_builder_reads_every_key_a_cloth_hands_it(self) -> None:
        """``friction`` crossed in every cloth's actor and the shell builder declared it unread."""
        from robosimstudio.engines.geometry import actor_consumes
        from robosimstudio.specs.physics import get_backend
        from robosimstudio_materials.cloth import ClothMaterialSpec, EntityCloth, sheet

        get_backend("superdex")  # loads the engine bundle, which registers its builders
        material = ClothMaterialSpec(shell_young=1.0e6, shell_poisson=0.4)
        actor = EntityCloth(sheet(name="towel", material=material)).host_actor()
        self.assertEqual("shell", actor.kind)
        self.assertLessEqual(set(actor.params), set(actor_consumes("superdex", "shell")))

    def test_the_bag_reads_what_the_shell_builder_reads(self) -> None:
        """Its modulus too: it states ``young``/``poisson`` as a sheet does, and reads the two names a record stated
        it under before (``shell_young``/``shell_poisson``), which its builder declares beside the shell's."""
        from robosimstudio.engines.geometry import actor_consumes
        from robosimstudio.specs.physics import get_backend
        from robosimstudio_materials.bag import EntityBagSuperdex, bag

        get_backend("superdex")
        shell = set(actor_consumes("superdex", "shell"))
        self.assertEqual(shell | {"shell_young", "shell_poisson"}, set(actor_consumes("superdex", "bag")))
        self.assertEqual(len(shell) + 2, len(actor_consumes("superdex", "bag")), "each name once")
        params = EntityBagSuperdex(bag(name="bag")).host_actor().params
        self.assertEqual((1.0e6, 0.3), (params["young"], params["poisson"]), "the canvas, under the shell's names")

    def test_a_weld_stated_on_a_sheet_or_a_bag_is_one_its_builder_reads(self) -> None:
        """The bridge's weld pass joins a shell's stated nodes as a soft body's, so neither is told it is unread."""
        import importlib
        import warnings

        from robosimstudio.engines.geometry import actor_builder, actor_consumes
        from robosimstudio.specs.physics import HostActor, get_backend

        compose = importlib.import_module("robosimstudio.scene.physics.compose")
        get_backend("superdex")  # loads the engine bundle, which registers its builders
        actor_builder("superdex", "bag")  # the lookup scans the packs, which registers the bag's
        island = SimpleNamespace(backend=SimpleNamespace(solver="superdex"))
        welds = [{"to": "plate", "nodes": [0], "contact": False}]
        for kind in ("shell", "bag"):
            with self.subTest(kind=kind), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertIn("welds", actor_consumes("superdex", kind))
                actor = HostActor(kind=kind, params={"thickness": 1.0e-3, "welds": welds})
                compose._warn_lost_material(SimpleNamespace(name="towel", entity=None), actor, island)
                self.assertEqual([], [str(w.message) for w in caught])

    def test_a_key_the_shell_builder_learns_is_the_bags_too(self) -> None:
        """The bag's declaration is read off the shell builder's, not copied from it key by key.

        A fresh process, since the bag registers when it is first imported: the shell builder
        learns a key first, and the bag that imports after it reads it. The superdex wheel stays
        out: the bag reaches the bundle's declaration, not the engine.
        """
        import json
        import os
        import subprocess
        import sys
        from pathlib import Path

        script = (
            "import json, sys\n"
            "import robosimstudio.engines.superdex\n"
            "from robosimstudio.specs.physics import actor_consumes, declare_consumes\n"
            "declare_consumes('superdex', 'shell', (*actor_consumes('superdex', 'shell'), 'learned'))\n"
            "import robosimstudio_materials.bag\n"
            "print(json.dumps({'bag': actor_consumes('superdex', 'bag'), 'wheel': 'superdex' in sys.modules}))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
        got = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertIn("learned", got["bag"])
        self.assertFalse(got["wheel"], "importing the bag loaded the superdex wheel")

    def test_a_fabrics_modulus_is_not_called_lost_where_it_crosses_as_young(self) -> None:
        """The composer's entity-side check looks a field up by the name it crosses under.

        A towel stating ``shell_young`` on SuperDex was told the field never reached the engine,
        which built its shell at that modulus; the bag kept its material's names to dodge the same.
        """
        import importlib
        import warnings

        from robosimstudio.specs.physics import get_backend
        from robosimstudio_materials.bag import EntityBagSuperdex, bag
        from robosimstudio_materials.cloth import ClothMaterialSpec, EntityCloth, sheet

        compose = importlib.import_module("robosimstudio.scene.physics.compose")
        get_backend("superdex")
        island = SimpleNamespace(backend=SimpleNamespace(solver="superdex"))
        towel = EntityCloth(sheet(name="towel", material=ClothMaterialSpec(shell_young=1.0e6, shell_poisson=0.4)))
        for name, entity in (("towel", towel), ("bag", EntityBagSuperdex(bag(name="bag")))):
            with self.subTest(entity=name), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                compose._warn_lost_material(SimpleNamespace(name=name, entity=entity), entity.host_actor(), island)
                self.assertEqual([], [str(w.message) for w in caught])
        slab = EntityCloth(
            sheet(name="slab", material=ClothMaterialSpec(shell="slab", thickness=4e-3, shell_young=1e6))
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compose._warn_lost_material(SimpleNamespace(name="slab", entity=slab), slab.host_actor(), island)
        lost = [str(w.message) for w in caught if "never reach" in str(w.message)]
        self.assertTrue(lost and "shell_young" in lost[0], "a slab is a tet body, which takes no shell modulus")


if __name__ == "__main__":
    unittest.main()
