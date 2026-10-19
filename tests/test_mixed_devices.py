# SPDX-License-Identifier: Apache-2.0
"""A GPU scene with a material that only runs on the CPU: per kind, and per object.

The device is the scene's, and the engine is the material's. SuperDex is a CPU
engine, co-simulated beside whatever device the model is on, so a GPU scene can
hold an arm on MuJoCo-Warp and a towel on VBD -- both on the GPU, inside one
captured CUDA graph -- beside a sponge and a second towel on SuperDex, stepped on
the CPU once a control step. The arrangement puts a kind on an engine
(``with_backend("soft", ...)``); a spec's ``sim=SimRequirements(backend=...)``
puts one object there; and the build says, on its platform line, what it steps
on the CPU.
"""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

TIER = "medium"
"""A GPU and the [superdex] (or [cpu]) extra."""


@unittest.skipUnless(importlib.util.find_spec("superdex") is not None, "needs the [superdex] extra")
class TestACpuMaterialInAGpuScene(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import warp as wp

        import robosimstudio as rss

        if not wp.is_cuda_available():
            raise unittest.SkipTest("needs a GPU: this is a GPU scene with a CPU material in it")
        scene = rss.Scene(cell="bench_075", name="mixed")  # GENERIC: soft on SuperDex, cloth on VBD
        scene.add(rss.robot("yam_bimanual"))
        scene.add(rss.sheet(name="towel_gpu", size=(0.2, 0.2), cells=(12, 12), at=(-0.25, -0.1, 0.02)))
        on_the_cpu = rss.SimRequirements(domain="cloth", backend=rss.backend("superdex", substeps=2))
        scene.add(rss.sheet(name="towel_cpu", size=(0.2, 0.2), cells=(12, 12), at=(0.05, -0.1, 0.02), sim=on_the_cpu))
        scene.add(rss.sponge(name="sponge", size=(0.09, 0.05, 0.035), cells=(9, 5, 4), at=(0.0, 0.2, 0.0)))
        cls.sim = scene.build()
        cls.sim.run(20).sync()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def engine_of(self, name: str) -> str:
        program = self.sim.pipeline.program
        return next(i.backend.solver for i in program.islands if name in i.entities)

    def test_each_thing_is_on_the_engine_it_was_put_on(self) -> None:
        self.assertEqual(self.engine_of("towel_gpu"), "vbd", "the kind's engine, from GENERIC")
        self.assertEqual(self.engine_of("towel_cpu"), "superdex", "this one object's own engine")
        self.assertEqual(self.engine_of("sponge"), "superdex", "the kind's engine, from GENERIC")

    def test_the_gpu_part_is_still_one_captured_graph(self) -> None:
        self.assertFalse(self.sim.platform.is_cpu)
        self.assertTrue(self.sim.pipeline.captured, self.sim.pipeline.capture_error)

    def test_the_platform_line_says_what_runs_on_the_cpu(self) -> None:
        line = self.sim.platform.readout()
        self.assertIn("on superdex, stepped on the CPU each control step", line)
        self.assertIn("towel_cpu", line)
        self.assertIn("superdex (on the CPU)", self.sim.pipeline.program.summary())

    def test_every_material_lands_and_stays_finite(self) -> None:
        q = self.sim.pipeline.state_0.particle_q.numpy()
        surface = float(self.sim.scene.handles.surface_z)
        for name in ("towel_gpu", "towel_cpu", "sponge"):
            nodes = np.asarray(self.sim[name].particles)
            self.assertTrue(np.isfinite(q[nodes]).all(), name)
            self.assertAlmostEqual(float(q[nodes, 2].min()), surface, delta=0.004, msg=f"{name} is off the bench")


if __name__ == "__main__":
    unittest.main()
