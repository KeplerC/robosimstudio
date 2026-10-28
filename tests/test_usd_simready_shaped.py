# SPDX-License-Identifier: Apache-2.0
"""A SimReady-shaped USD prop imports as one rigid body, on the stage's Newton variant.

NVIDIA's SimReady assets carry a variant set per physics engine with PhysX
enabled, and most of their multi-part props hold the parts with fixed joints that
belong to no articulation, which Newton's ``finalize`` refuses. The import selects
the Newton variant, gives such joints one articulation root, and folds a prop
whose joints are all fixed into one body -- all on a throwaway session layer, and
folding only what this import added. A stage with moving joints is refused by the
rigid path with a pointer to ``ArticulatedSpec``, and a stage whose jointed parts
form two trees is refused naming the stage.

Built on a CPU builder from stages :mod:`tests.fixtures.simready` writes; nothing
is stepped.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import newton
import numpy as np

from robosimstudio.domains.articulated.entity import EntityArticulated
from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.scene.build.imports import add_usd_stage, articulate, open_session_stage, select_engine_variant
from robosimstudio.specs import ArticulatedSpec, AssetSpec, ObjectSpec
from tests.fixtures.simready import NEWTON_MASS, PHYSX_MASS, Joint, Part, write_stage

TIER = "fast"
"""Builds stages into a CPU builder and finalizes them; no solver, no GPU."""


def _rigid(path: Path, name: str = "prop") -> EntityRigid:
    return EntityRigid(ObjectSpec(name=name, asset=AssetSpec(path=str(path)), pose=(0.0, 0.0, 0.1), obb_grasp=False))


class _Stages(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._dir = tempfile.TemporaryDirectory()
        root = Path(cls._dir.name)
        cls.box = write_stage(root / "box.usda", [Part("box_01")])
        cls.bottle = write_stage(
            root / "bottle.usda",
            [
                Part("body_01", (0.0, 0.0, 0.08), (0.03, 0.03, 0.08)),
                Part("cap_01", (0.0, 0.0, 0.17), (0.02, 0.02, 0.01)),
            ],
            [Joint("cap_01", "fixed", "body_01", "cap_01", (0.0, 0.0, 0.16))],
        )
        cls.cabinet = write_stage(
            root / "cabinet.usda",
            [
                Part("body_01", (0.0, 0.0, 0.3), (0.2, 0.2, 0.3)),
                Part("drawer_01", (0.25, 0.0, 0.4), (0.05, 0.15, 0.05)),
            ],
            [Joint("drawer_01", "prismatic", "body_01", "drawer_01", (0.25, 0.0, 0.4), "X", (0.0, 0.3))],
        )
        cls.split = write_stage(
            root / "split.usda",
            [
                Part("body_01"),
                Part("hinge_01", (0.1, 0.0, 0.0)),
                Part("lid_01", (0.0, 0.0, 0.2)),
                Part("lock_01", (0.1, 0.0, 0.2)),
            ],
            [
                Joint("hinge_01", "revolute", "body_01", "hinge_01", (0.1, 0.0, 0.0), "Y", (-90.0, 90.0)),
                Joint("lock_01", "revolute", "lid_01", "lock_01", (0.1, 0.0, 0.2), "Y", (-90.0, 90.0)),
            ],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._dir.cleanup()


class TestTheNewtonVariantIsSelected(_Stages):
    def test_the_rigid_path_takes_the_newton_mass(self) -> None:
        entity = _rigid(self.box)
        builder = newton.ModelBuilder()
        entity.build_into(builder, None)
        self.assertAlmostEqual(float(builder.body_mass[entity.body]), NEWTON_MASS, places=5)

    def test_no_engine_variant_keeps_the_shipped_selection(self) -> None:
        builder = newton.ModelBuilder()
        add_usd_stage(builder, str(self.box), floating=True, engine_variant=None)
        self.assertAlmostEqual(float(builder.body_mass[0]), PHYSX_MASS, places=5)

    def test_a_stage_without_engine_variants_is_not_touched(self) -> None:
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        stage.SetDefaultPrim(stage.DefinePrim("/RootNode", "Xform"))
        self.assertEqual([], select_engine_variant(stage, "Newton"))


class TestAFixedJointPropIsOneBody(_Stages):
    def test_the_bottle_and_its_cap_are_one_body_with_both_masses(self) -> None:
        entity = _rigid(self.bottle)
        builder = newton.ModelBuilder()
        entity.build_into(builder, None)
        model = builder.finalize(device="cpu")
        self.assertEqual(1, model.body_count)
        self.assertEqual([int(newton.JointType.FREE)], model.joint_type.numpy().tolist())
        self.assertAlmostEqual(float(model.body_mass.numpy()[0]), 2 * NEWTON_MASS, places=4)

    def test_the_fold_leaves_what_was_built_before_it_alone(self) -> None:
        cabinet = lambda: EntityArticulated(ArticulatedSpec(name="cabinet", asset=AssetSpec(path=str(self.cabinet))))  # noqa: E731
        alone = newton.ModelBuilder()
        cabinet().build_into(alone, None)
        both = newton.ModelBuilder()
        cabinet().build_into(both, None)
        prop = _rigid(self.bottle)
        prop.build_into(both, None)
        bodies, joints, shapes = alone.body_count, alone.joint_count, alone.shape_count
        np.testing.assert_allclose(np.asarray(alone.body_q), np.asarray(both.body_q)[:bodies])
        self.assertEqual(list(alone.joint_type), list(both.joint_type)[:joints])
        self.assertEqual(list(alone.joint_parent), list(both.joint_parent)[:joints])
        self.assertEqual(list(alone.shape_body), list(both.shape_body)[:shapes])
        self.assertEqual(bodies, prop.body)
        both.finalize(device="cpu")


class TestAJointedStageIsNotARigidBody(_Stages):
    def test_it_is_refused_with_the_spec_that_takes_it(self) -> None:
        with self.assertRaisesRegex(ValueError, "ArticulatedSpec"):
            _rigid(self.cabinet).build_into(newton.ModelBuilder(), None)

    def test_static_is_refused_with_the_same_pointer(self) -> None:
        spec = ObjectSpec(name="box", asset=AssetSpec(path=str(self.box)), static=True, obb_grasp=False)
        with self.assertRaisesRegex(ValueError, "ArticulatedSpec with fixed_base=True"):
            EntityRigid(spec).build_into(newton.ModelBuilder(), None)


class TestTheFileIsNotEdited(_Stages):
    def test_the_root_layer_is_clean_and_the_bytes_unchanged(self) -> None:
        from pxr import Sdf

        before = self.cabinet.read_bytes()
        builder = newton.ModelBuilder()
        add_usd_stage(builder, str(self.cabinet), floating=False)
        self.assertEqual(before, self.cabinet.read_bytes())
        self.assertFalse(Sdf.Layer.FindOrOpen(str(self.cabinet)).dirty)

    def test_a_second_open_sees_the_stage_as_shipped(self) -> None:
        from pxr import UsdPhysics

        first = open_session_stage(self.cabinet)
        self.assertEqual(["/RootNode"], articulate(first))
        second = open_session_stage(self.cabinet)
        self.assertFalse(second.GetDefaultPrim().HasAPI(UsdPhysics.ArticulationRootAPI))


class TestTwoTreesAreNamed(_Stages):
    def test_a_lid_joined_to_nothing_is_refused_naming_the_stage(self) -> None:
        with self.assertRaisesRegex(ValueError, r"split\.usda.*more than one tree"):
            add_usd_stage(newton.ModelBuilder(), str(self.split), floating=True)


if __name__ == "__main__":
    unittest.main()
