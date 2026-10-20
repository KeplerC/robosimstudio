# SPDX-License-Identifier: Apache-2.0
"""A static object is welded to the world: one fixed joint from the world, and a kinematic body.

``ObjectSpec.static`` says *welded to the world rather than free*. Every path
``EntityRigid`` built a static object by made a bare ``add_link`` instead: a
body with no joint, flagged DYNAMIC. VBD integrated it like a free body, and
MuJoCo refused to convert it. ``tests/test_a_static_box_stays_put.py`` measures
both on a GPU. A geometry-only USD stage built a *free* body for
``static=True``, with a FREE joint.

Checked here on a CPU builder, finalized and never stepped, for each of the
three paths: a primitive, a mesh file and a geometry-only USD stage. A free
object's FREE joint is checked too, and is unchanged. And no entity that builds
on ``EntityRigid`` welds a static body a second time.

A layout that moves a static object is refused by name, by a reset and by a
fleet's wave, which asks only the statics and keeps nothing it refuses: a static
the scene's arena declares as well as a welded one, and an instance document
whose own start poses put a welded static elsewhere than its fleet base does.
"""

from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import newton
import numpy as np

from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.scene.compiler import CannotReplicate
from robosimstudio.scene.entity import label_index
from robosimstudio.specs import AssetSpec, ObjectSpec, TaskSpec
from robosimstudio.specs.asset import ShapeSpec
from tests.fixtures.packs import has_pack

TIER = "fast"
"""Builds one object into a CPU builder and finalizes it; no solver, no GPU."""

POSE = (0.1, -0.2, 0.3)

MOVED = {"object_poses": {"box": (POSE[0] + 0.05, POSE[1], POSE[2], 0.0)}}
TURNED = {"object_poses": {"box": (*POSE, 0.5)}}
STAYS = {"object_poses": {"box": (*POSE, 0.0)}}
"""Three episodes for ``box``: 50 mm along x, turned half a radian, and where it was built."""

ROOT = Path(__file__).resolve().parents[1]

WIPE_THE_BAR = ROOT / "examples" / "08_demos" / "wipe_the_bar"
"""The demo whose ``GrabBar`` extends ``EntityRigid``'s build of a static object with elbows and legs."""

_USD_CUBE = """#usda 1.0
def Mesh "cube"
{
    int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
    int[] faceVertexIndices = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]
    point3f[] points = [(-0.05, -0.05, -0.05), (0.05, -0.05, -0.05), (0.05, 0.05, -0.05), (-0.05, 0.05, -0.05),
                        (-0.05, -0.05, 0.05), (0.05, -0.05, 0.05), (0.05, 0.05, 0.05), (-0.05, 0.05, 0.05)]
}
"""
"""A 10 cm cube with no physics schema: geometry, which ``_build_usd`` treats as an ``.obj``."""


def _built(spec: ObjectSpec, entity_class=EntityRigid):
    entity = entity_class(spec)
    builder = newton.ModelBuilder()
    entity.build_into(builder, episode=None)
    return entity, builder.finalize(device="cpu")


def _world_joints(model, body: int) -> list[int]:
    """The joints from the world to ``body``."""
    parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
    return [int(j) for j in np.flatnonzero((children == body) & (parents == -1))]


def _fleet(objects: dict, *, base: TaskSpec | None = None):
    """A two-world fleet holding no model: what ``reset_wave``'s checks read, and a wave already laid down."""
    from robosimstudio_bench.envs.fleet import FleetEnv

    fleet = FleetEnv.__new__(FleetEnv)
    fleet.num_worlds = 2
    fleet.scene = SimpleNamespace(objects=objects)
    fleet.task = fleet.build_task = base or TaskSpec(name="doc")
    fleet._declared = {name: entity.declared_q for name, entity in objects.items()}
    fleet._declared_docs = {}
    fleet.episodes, fleet.documents = [STAYS, STAYS], [None, None]
    return fleet


class _ThreePaths(unittest.TestCase):
    """A primitive, a mesh file and a geometry-only USD stage, each a 10 cm cube named ``box`` at :data:`POSE`."""

    @classmethod
    def setUpClass(cls) -> None:
        import trimesh

        cls._dir = tempfile.TemporaryDirectory()
        root = Path(cls._dir.name)
        trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(root / "cube.obj")
        (root / "cube.usda").write_text(_USD_CUBE)
        cls.paths = {"primitive": None, "mesh": root / "cube.obj", "usd": root / "cube.usda"}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._dir.cleanup()

    def _spec(self, how: str, *, static: bool) -> ObjectSpec:
        path = self.paths[how]
        if path is None:
            return ObjectSpec(name="box", shape=ShapeSpec(half_extents=(0.05,) * 3), pose=POSE, static=static)
        return ObjectSpec(name="box", asset=AssetSpec(path=str(path)), pose=POSE, static=static, obb_grasp=False)


class TestAStaticObjectIsWelded(_ThreePaths):
    def test_each_build_path_bolts_it_to_the_world_where_it_stands(self) -> None:
        for how in self.paths:
            with self.subTest(how=how):
                entity, model = _built(self._spec(how, static=True))
                body = entity.body
                joints = np.flatnonzero(model.joint_child.numpy() == body)
                self.assertEqual(1, len(joints), "one joint holds it")
                j = int(joints[0])
                self.assertEqual(int(newton.JointType.FIXED), int(model.joint_type.numpy()[j]), "a weld, not free")
                self.assertEqual(-1, int(model.joint_parent.numpy()[j]), "to the world")
                np.testing.assert_allclose(model.joint_X_p.numpy()[j], model.body_q.numpy()[body], atol=1.0e-7)
                np.testing.assert_allclose(model.body_q.numpy()[body][:3], POSE, atol=1.0e-7)
                self.assertEqual(
                    int(newton.BodyFlags.KINEMATIC), int(model.body_flags.numpy()[body]), "nothing pushes it"
                )
                self.assertEqual((-1, 0), (entity.coord_start, model.joint_coord_count), "no coordinates to reset")

    def test_a_free_object_keeps_its_free_joint(self) -> None:
        for how in self.paths:
            with self.subTest(how=how):
                entity, model = _built(self._spec(how, static=False))
                joints = np.flatnonzero(model.joint_child.numpy() == entity.body)
                self.assertEqual([int(newton.JointType.FREE)], [int(model.joint_type.numpy()[j]) for j in joints])
                self.assertEqual(int(newton.BodyFlags.DYNAMIC), int(model.body_flags.numpy()[entity.body]))
                self.assertEqual(0, entity.coord_start)


class TestAnEpisodeThatMovesAStaticObjectIsRefused(_ThreePaths):
    """A static object stands where its model welded it, so an episode that moves it is refused by name.

    The weld has no coordinates, so ``EntityRigid.reset`` returned early and the
    object stood at the build episode's pose whatever a later episode on the same
    model said, and a fleet's wave stood every world's copy there. On a GPU,
    ``tests/test_a_static_box_stays_put.py`` runs both, and the way out the
    refusal names.
    """

    def test_a_reset_refuses_an_episode_that_moves_or_turns_it(self) -> None:
        for how in self.paths:
            entity, _ = _built(self._spec(how, static=True))
            for name, episode in (("moved", MOVED), ("turned", TURNED)):
                with self.subTest(how=how, episode=name), self.assertRaisesRegex(ValueError, "'box' is static"):
                    entity.reset(None, episode)

    def test_a_reset_takes_an_episode_that_leaves_it_where_it_was_built(self) -> None:
        for how in self.paths:
            with self.subTest(how=how):
                entity, _ = _built(self._spec(how, static=True))
                for episode in (None, STAYS, {"object_poses": {"other": (0.0, 0.0, 0.0, 1.0)}}):
                    entity.reset(None, episode)

    def test_a_wave_refuses_a_world_whose_episode_moves_it(self) -> None:
        entity, _ = _built(self._spec("primitive", static=True))
        fleet = _fleet({"box": entity})
        fleet._check_statics([STAYS, None, STAYS], [None, None, None])
        with self.assertRaisesRegex(CannotReplicate, "world 1 runs 'doc', and object 'box' is static"):
            fleet._check_statics([STAYS, MOVED], [None, None])

    def test_the_reset_and_the_wave_name_one_remedy_written_once(self) -> None:
        """The way out was written in ``EntityRigid.reset`` and again in ``FleetEnv._check_statics``."""
        import inspect

        from robosimstudio.domains.rigid import entity as rigid
        from robosimstudio_bench.envs import fleet as fleet_module

        entity, _ = _built(self._spec("primitive", static=True))
        with self.assertRaises(ValueError) as serial:
            entity.reset(None, MOVED)
        with self.assertRaises(CannotReplicate) as wave:
            _fleet({"box": entity})._check_statics([STAYS, MOVED], [None, None])
        for refusal in (serial.exception, wave.exception):
            self.assertTrue(str(refusal).endswith(entity.move_remedy), str(refusal))
        written = sum(inspect.getsource(m).count("(:model-key-fields ...)") for m in (rigid, fleet_module))
        self.assertEqual(1, written, "the remedy is written in one place")


class TestAWaveAsksOnlyItsStatics(unittest.TestCase):
    """``_check_statics`` composed every free object's start pose in every world, for a refusal that was always empty.

    Twenty free cubes and one static box in twelve worlds: 252 composes, now 12 (``e5_composes.py``).
    """

    def test_no_free_object_is_composed(self) -> None:
        cube, _ = _built(ObjectSpec(name="cube", shape=ShapeSpec(half_extents=(0.02,) * 3), pose=POSE))
        box, _ = _built(ObjectSpec(name="box", shape=ShapeSpec(half_extents=(0.05,) * 3), pose=POSE, static=True))
        composed = []
        cube.episode_start_q = lambda *args, **kwargs: composed.append(args) or cube.declared_q
        _fleet({"cube": cube, "box": box})._check_statics([STAYS, STAYS], [None, None])
        self.assertEqual([], composed)


class TestARefusedWaveKeepsNothing(unittest.TestCase):
    """A wave refused by name leaves the fleet holding the wave before it.

    ``reset_wave`` kept the wave's episodes and documents and then checked them,
    so a caller that caught ``CannotReplicate`` held a fleet whose episodes named
    the refused wave while its state held the one before.
    """

    def _refused(self, episodes, documents=None):
        entity, _ = _built(ObjectSpec(name="box", shape=ShapeSpec(half_extents=(0.05,) * 3), pose=POSE, static=True))
        fleet = _fleet({"box": entity})
        before = (fleet.episodes, fleet.documents)
        with self.assertRaises(CannotReplicate):
            fleet.reset_wave(episodes, documents)
        self.assertIs(before[0], fleet.episodes)
        self.assertIs(before[1], fleet.documents)

    def test_by_its_statics(self) -> None:
        self._refused([STAYS, MOVED])

    def test_by_its_documents(self) -> None:
        self._refused([STAYS, STAYS], [TaskSpec(name="another_model"), None])


class TestARefusedResetMovesNothing(unittest.TestCase):
    """``Sim.reset_scene`` asks every entity before it moves anything.

    It placed entity by entity, and a static object's reset refused only when its turn came: a
    free cube declared before a static block had gone to the refused episode's x = 0.70 when the
    block said no, with ``env.episode`` already the refused one (``e3_partial_reset.py``).
    ``tests/test_a_static_box_stays_put.py`` runs a real ``Env`` through it on a GPU.
    """

    def test_a_free_cube_before_a_moved_static_box_stays_where_it_was(self) -> None:
        from robosimstudio.sim import Sim

        cube = EntityRigid(ObjectSpec(name="cube", shape=ShapeSpec(half_extents=(0.02,) * 3), pose=(0.0, 0.2, 0.02)))
        box = EntityRigid(ObjectSpec(name="box", shape=ShapeSpec(half_extents=(0.05,) * 3), pose=POSE, static=True))
        builder = newton.ModelBuilder()
        for entity in (cube, box):
            entity.build_into(builder, episode=None)
        model = builder.finalize(device="cpu")
        solved: list[dict] = []
        pipeline = SimpleNamespace(
            state_0=model.state(),
            state_1=model.state(),
            reset_solvers=lambda **flags: solved.append(flags),
            mark_dirty=lambda: None,
        )
        sim = Sim.__new__(Sim)
        # A start articulation unlike the state's, so that writing it is seen as well as the cube's reset.
        sim.pipeline, sim.model, sim._start_joint_q = pipeline, model, np.zeros(model.joint_coord_count, np.float32)
        sim.scene = SimpleNamespace(entities=[cube, box])
        sim.episode = {"object_poses": {"cube": (0.1, 0.2, 0.02, 0.0), **MOVED["object_poses"]}}
        before = [state.joint_q.numpy().copy() for state in (pipeline.state_0, pipeline.state_1)]
        with self.assertRaisesRegex(ValueError, "'box' is static"):
            sim.reset_scene()
        for state, was in zip((pipeline.state_0, pipeline.state_1), before, strict=True):
            np.testing.assert_array_equal(was, state.joint_q.numpy(), "the cube went to the refused layout")
        self.assertEqual([], solved, "the solvers were reset for a layout that was refused")


class TestADocumentStatesAStaticWhereItsBaseDoes(unittest.TestCase):
    """A superset's instance runs on its base's model, so its own start pose for a static must be the base's.

    Its serial runs build its own model, with the static where its start poses put
    it -- the lowering's default when they omit it -- and a wave's rows are to be
    those rows. A wave that ran it with the static where the base welded it would
    score a layout the document does not have, so it is refused, and the refusal
    says the start poses moved it: a model key, the episode's remedy, never
    separates two documents of one base.
    """

    SPEC = ObjectSpec(name="box", shape=ShapeSpec(half_extents=(0.05,) * 3), pose=POSE, static=True)

    def _wave(self, document: TaskSpec):
        entity, _ = _built(self.SPEC)
        fleet = _fleet({"box": entity}, base=TaskSpec(name="base", suite="s", entities=(self.SPEC,)))
        fleet._check_statics([{}, {}], [fleet.build_task, document])

    def test_one_stating_its_base_pose_runs(self) -> None:
        self._wave(TaskSpec(name="t00", suite="s", entities=(self.SPEC,)))

    def test_one_stating_another_or_none_is_refused_by_its_start_poses(self) -> None:
        from robosimstudio.specs import replace

        for pose in ((POSE[0] + 0.05, POSE[1], POSE[2]), (0.0, 0.0, 0.0)):
            document = TaskSpec(name="t01", suite="s", entities=(replace(self.SPEC, pose=pose),))
            with (
                self.subTest(pose=pose),
                self.assertRaisesRegex(
                    CannotReplicate, r"world 1 runs 's/t01', and object 'box' is static.*state it where 's/base' does"
                ),
            ):
                self._wave(document)


ARENA = (0.4, 0.1, 0.8)
"""Where the arena stands ``bin``, in world [m]: not its spec's pose, which places nothing for an arena's object."""

ARENA_MOVED = {"object_xforms": {"bin": (ARENA[0] + 0.05, ARENA[1], ARENA[2], 0.0, 0.0, 0.0, 1.0)}}
ARENA_TURNED = {"object_poses": {"bin": (*ARENA, 0.5)}}
ARENA_STAYS = {"object_xforms": {"bin": (*ARENA, 0.0, 0.0, 0.0, 1.0)}}
"""Three episodes for ``bin``: 50 mm along x, turned half a radian, and named where the arena stands it."""


def _arena_bin(*, fixed: bool = False):
    """A ``bin`` the arena declares and ``EntityRigid`` finds: a dynamic body on the arena's own free joint.

    ``fixed``: held by the arena's own fixed joint to the world instead, a link with no free one.
    """
    import warp as wp

    builder = newton.ModelBuilder()
    xform = wp.transform(ARENA, wp.quat_identity())
    if fixed:
        body = builder.add_link(xform=xform, label="bin")
        builder.add_joint_fixed(-1, body, parent_xform=xform, label="bin_fixed")
    else:
        body = builder.add_body(xform=xform, label="bin")
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    entity = EntityRigid(ObjectSpec(name="bin", static=True, obb_grasp=False))
    entity.build_into(builder, episode=None)
    model = builder.finalize(device="cpu")
    entity.bind(model)
    return entity


class TestAStaticTheArenaDeclares(unittest.TestCase):
    """A static object the scene's arena declares: no asset and no shape, found by ``find_in_scene``.

    It only located the body, so it held no pose: a reset whose episode named it
    50 mm off returned without a word, and a wave stood every world's copy where
    the arena had it, each scored as though it stood where asked. Its spec's pose
    places nothing, as the import put the body where it is, so only what an episode
    names can ask for it elsewhere. And its mass panel was dropped on ``static``
    alone, though no weld made the body kinematic: here it is dynamic.
    """

    def test_a_reset_refuses_an_episode_that_names_it_elsewhere(self) -> None:
        entity = _arena_bin()
        for name, episode in (("moved", ARENA_MOVED), ("turned", ARENA_TURNED)):
            with self.subTest(name), self.assertRaisesRegex(ValueError, "'bin' is static, where the scene's arena"):
                entity.reset(None, episode)

    def test_a_reset_takes_an_episode_that_names_it_where_it_stands_or_not_at_all(self) -> None:
        entity = _arena_bin()
        self.assertFalse(np.allclose(entity.declared_q[:3], ARENA), "its spec states the default origin")
        for episode in (None, {}, ARENA_STAYS, {"object_poses": {"other": (0.0, 0.0, 0.0, 1.0)}}):
            entity.reset(None, episode)

    def test_a_wave_refuses_a_world_whose_episode_names_it_elsewhere(self) -> None:
        entity = _arena_bin()
        fleet = _fleet({"bin": entity})
        fleet._check_statics([ARENA_STAYS, {}], [None, None])
        with self.assertRaisesRegex(CannotReplicate, "world 1 runs 'doc', and object 'bin' is static"):
            fleet._check_statics([{}, ARENA_MOVED], [None, None])

    def test_a_document_stating_another_pose_for_it_runs(self) -> None:
        """Its spec's pose places nothing in either model, so a document's differing one moves nothing."""
        entity = _arena_bin()
        fleet = _fleet({"bin": entity}, base=TaskSpec(name="base", suite="s", entities=(entity.spec,)))
        document = TaskSpec(name="t01", suite="s", entities=(ObjectSpec(name="bin", static=True, pose=ARENA),))
        fleet._check_statics([{}, {}], [fleet.build_task, document])

    def test_it_offers_its_mass(self) -> None:
        self.assertEqual(["contact", "mass"], [t.name for t in _arena_bin().tunables()])

    def test_held_by_the_arenas_fixed_joint_it_offers_no_mass(self) -> None:
        """Gated on no weld of its own, it offered a mass slider that moved nothing: its body is held, not free."""
        self.assertEqual(["contact"], [t.name for t in _arena_bin(fixed=True).tunables()])


class TestNoStaticBodyIsWeldedTwice(unittest.TestCase):
    """One joint to the world per static body, however an entity extends ``EntityRigid``'s build.

    ``EntityRigid`` welds a static object itself, so a subclass that welds it
    again after ``super().build()`` gives it a second fixed joint to the world
    under the same label. The wipe-the-bar demo's ``GrabBar`` did: the bar had two
    ``bar_weld`` joints, ``SolverMuJoCo`` made the first the mocap root and the
    second a weld equality, and ``label_index`` found the label twice.
    """

    @staticmethod
    def _bar():
        if str(WIPE_THE_BAR) not in sys.path:
            sys.path.insert(0, str(WIPE_THE_BAR))
        import bar

        return _built(bar.BarSpec(), bar.GrabBar)

    def test_the_grab_bar_has_one_weld_and_its_label_names_it(self) -> None:
        entity, model = self._bar()
        welds = _world_joints(model, entity.body)
        self.assertEqual(1, len(welds), "the bar had two, both 'bar_weld'")
        self.assertEqual(welds[0], label_index(model.joint_label, "bar_weld"))

    def test_every_static_body_the_shipped_rigid_entities_build(self) -> None:
        builds = {"primitive": _built(ObjectSpec(name="box", shape=ShapeSpec(half_extents=(0.05,) * 3), static=True))}
        builds["grab_bar"] = self._bar()
        if has_pack("cutting"):
            import robosimstudio as rss
            from robosimstudio_materials.cutting.entity import EntityKnife

            builds["knife"] = _built(rss.knife(name="knife", static=True), EntityKnife)
        for name, (entity, model) in builds.items():
            with self.subTest(name):
                kinematic = np.flatnonzero(model.body_flags.numpy() == int(newton.BodyFlags.KINEMATIC))
                self.assertEqual([entity.body], [int(b) for b in kinematic], "the one static body")
                self.assertEqual(1, len(_world_joints(model, entity.body)), "one weld")


class TestAStaticObjectStatesNoMass(unittest.TestCase):
    """A static body's mass is read by no solver, so the build does not write one and no panel offers one.

    The mesh path already skipped ``set_body_mass`` for a static object. The
    primitive path, the knife's among them, still wrote ``mass`` onto the
    kinematic body, which VBD gives an inverse mass of zero and MuJoCo makes a
    mocap body, and :meth:`EntityRigid.tunables` offered a ``mass`` that moved
    nothing.
    """

    def _builds(self):
        box = ObjectSpec(name="box", shape=ShapeSpec(half_extents=(0.05,) * 3), pose=POSE, static=True, mass=5.0)
        out = {"box": (box, EntityRigid)}
        if has_pack("cutting"):
            import robosimstudio as rss
            from robosimstudio_materials.cutting.entity import EntityKnife

            out["knife"] = (rss.knife(name="knife", static=True, mass=0.5), EntityKnife)
        return out

    def test_a_static_primitive_weighs_what_it_would_with_no_mass_stated(self) -> None:
        from robosimstudio.specs import replace

        for name, (spec, entity_class) in self._builds().items():
            with self.subTest(name):
                entity, model = _built(spec, entity_class)
                unstated, bare = _built(replace(spec, mass=None), entity_class)
                mass = float(model.body_mass.numpy()[entity.body])
                self.assertNotEqual(float(spec.mass), mass, "it wrote the stated mass")
                self.assertEqual(float(bare.body_mass.numpy()[unstated.body]), mass)

    def test_a_static_object_offers_its_contact_and_no_mass(self) -> None:
        for name, (spec, entity_class) in self._builds().items():
            with self.subTest(name):
                entity, model = _built(spec, entity_class)
                entity.bind(model)
                self.assertEqual(["contact"], [t.name for t in entity.tunables()])

    def test_a_free_object_still_offers_its_mass(self) -> None:
        from robosimstudio.specs import replace

        spec, _ = self._builds()["box"]
        entity, model = _built(replace(spec, static=False))
        entity.bind(model)
        self.assertAlmostEqual(5.0, float(model.body_mass.numpy()[entity.body]), places=5)
        self.assertEqual(["contact", "mass"], [t.name for t in entity.tunables()])


WELDS_ELSEWHERE = {
    "robosimstudio/domains/static/welded.py": "weld_to_world, the recipe itself",
    "robosimstudio_materials/rod/entity.py": (
        "a cable's anchored end: a capsule inside the rod's own articulation, tied to the world by a weld "
        "VBD solves, not a static body of its own"
    ),
}
"""The files that may add a fixed joint from the world themselves, and why."""


def _world_welds(path: Path) -> list[int]:
    """The lines of ``path`` that call ``add_joint_fixed`` with the world as the parent."""

    def is_world(node) -> bool:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return isinstance(node.operand, ast.Constant) and node.operand.value == 1
        return isinstance(node, ast.Constant) and node.value == -1

    lines = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_joint_fixed"):
            continue
        parent = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "parent"), None)
        if parent is not None and is_world(parent):
            lines.append(node.lineno)
    return lines


class TestOneRecipeWeldsABodyToTheWorld(unittest.TestCase):
    """Every fixed joint from the world in the shipped code is ``weld_to_world``'s.

    The recipe, a kinematic link and a fixed joint from the world at its pose, was
    written out in ``EntityRigid``, in ``WeldedFixture.weld_in_place`` and in the
    wipe-the-bar demo, and the demo's copy welded a body ``EntityRigid`` had
    already welded. Read by AST, so it stays in the fast tier.
    """

    PACKAGES = (
        "robosimstudio",
        "robosimstudio_bench",
        "robosimstudio_materials",
        "robosimstudio_policy_agent",
        "robosimstudio_real2sim_agent",
        "robosimstudio_studio",
        "examples",
        "scripts",
    )

    def test_nothing_else_adds_a_fixed_joint_from_the_world(self) -> None:
        found = {}
        for package in self.PACKAGES:
            for path in sorted((ROOT / package).rglob("*.py")):
                relative = path.relative_to(ROOT).as_posix()
                if "__pycache__" in path.parts or relative.startswith("examples/live/run/"):
                    continue
                if (lines := _world_welds(path)) and relative not in WELDS_ELSEWHERE:
                    found[relative] = lines
        self.assertEqual({}, found, "weld a body to the world through weld_to_world")

    def test_the_allowed_files_still_weld_to_the_world(self) -> None:
        """An entry nothing needs any more is one that would let the next copy through."""
        for relative in WELDS_ELSEWHERE:
            with self.subTest(relative):
                self.assertTrue(_world_welds(ROOT / relative))


if __name__ == "__main__":
    unittest.main()
