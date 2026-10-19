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
object's FREE joint is checked too, and is unchanged.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import newton
import numpy as np

from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.specs import AssetSpec, ObjectSpec
from robosimstudio.specs.asset import ShapeSpec

TIER = "fast"
"""Builds one object into a CPU builder and finalizes it; no solver, no GPU."""

POSE = (0.1, -0.2, 0.3)

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


def _built(spec: ObjectSpec):
    entity = EntityRigid(spec)
    builder = newton.ModelBuilder()
    entity.build_into(builder, episode=None)
    return entity, builder.finalize(device="cpu")


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


if __name__ == "__main__":
    unittest.main()
