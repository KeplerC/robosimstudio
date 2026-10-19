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
"""

from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path

import newton
import numpy as np

from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.scene.entity import label_index
from robosimstudio.specs import AssetSpec, ObjectSpec
from robosimstudio.specs.asset import ShapeSpec
from tests.fixtures.packs import has_pack

TIER = "fast"
"""Builds one object into a CPU builder and finalizes it; no solver, no GPU."""

POSE = (0.1, -0.2, 0.3)

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


class TestAStaticObjectIsWelded(unittest.TestCase):
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
