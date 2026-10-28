# SPDX-License-Identifier: Apache-2.0
"""A drawer driven open on MuJoCo opens, its cabinet stays bolted, and a reset closes it.

A SimReady-shaped cabinet on a fixed base, robot-less, on ``MUJOCO_500HZ``: one
drawer gets a position drive and a target 250 mm out. After two seconds it is
past 200 mm, read off where its bodies are; the other drawer is where it
started; the cabinet's root part has not moved a tenth of a millimetre; and a
reset brings the driven drawer back to its start. :mod:`tests.test_articulated_object`
checks the same entity on a CPU builder without stepping.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.fixtures.simready import Joint, Part, write_stage

TIER = "medium"

TARGET = 0.25
"""Where the driven drawer is sent [m]."""


class TestADrivenDrawerOpens(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        cls._dir = tempfile.TemporaryDirectory()
        path = write_stage(
            Path(cls._dir.name) / "cabinet.usda",
            [
                Part("body_01", (0.0, 0.0, 0.3), (0.2, 0.2, 0.3)),
                Part("drawer_01", (0.21, 0.0, 0.45), (0.05, 0.15, 0.04)),
                Part("drawer_02", (0.21, 0.0, 0.25), (0.05, 0.15, 0.04)),
            ],
            [
                Joint("drawer_01", "prismatic", "body_01", "drawer_01", (0.21, 0.0, 0.45), "X", (0.0, 0.3)),
                Joint("drawer_02", "prismatic", "body_01", "drawer_02", (0.21, 0.0, 0.25), "X", (0.0, 0.3)),
            ],
        )
        scene = rss.Scene(cell="void", name="drawer", physics=rss.MUJOCO_500HZ)
        scene.add(
            rss.articulated(
                name="cabinet",
                asset=str(path),
                at=(0.5, 0.0, 0.0),
                joints=(rss.JointSpec(name="drawer_01", q=0.0, target_ke=200.0, target_kd=20.0),),
            )
        )
        cls.sim = scene.build()
        cls.handle = cls.sim.handles.objects["cabinet"]
        cls.root = cls.sim.pipeline.state_0.body_q.numpy()[cls.handle.body].copy()
        drawer = cls.handle.joints["drawer_01"]
        targets = cls.sim.pipeline.control.joint_target_q.numpy()
        targets[drawer.q_start] = TARGET
        cls.sim.pipeline.control.joint_target_q.assign(targets)
        cls.sim.run(100).sync()
        cls.after = cls.sim.pipeline.state_0.body_q.numpy().copy()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._dir.cleanup()

    def test_the_driven_drawer_opens(self) -> None:
        self.assertGreater(self.handle.joints["drawer_01"].coordinate_from_bodies(self.after), 0.2)

    def test_the_other_drawer_stays_shut(self) -> None:
        self.assertLess(abs(self.handle.joints["drawer_02"].coordinate_from_bodies(self.after)), 0.005)

    def test_the_cabinet_is_bolted(self) -> None:
        np.testing.assert_allclose(self.after[self.handle.body], self.root, atol=1.0e-4)
        self.assertTrue(np.isfinite(self.after).all())

    def test_a_reset_closes_it(self) -> None:
        self.sim.reset()
        body_q = self.sim.pipeline.state_0.body_q.numpy()
        self.assertLess(abs(self.handle.joints["drawer_01"].coordinate_from_bodies(body_q)), 1.0e-4)


if __name__ == "__main__":
    unittest.main()
