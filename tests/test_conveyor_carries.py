# SPDX-License-Identifier: Apache-2.0
"""A slat belt carries a rigid box on MuJoCo and a SuperDex mailer at its own speed, by contact.

Both ride the same slats: the box through MuJoCo's contacts with the slats' written velocity, the
mailer through SuperDex's sweep of their written poses. Measured over two seconds after a settle,
on a 0.2 m/s belt: the box at 0.2000 m/s, the mailer at 0.1997 (``p1_smoke.py``). Stopped, the
belt holds what it carries where it is; reset, it starts from the beginning of its travel.
"""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

TIER = "medium"
"""One GPU: a YAM, a belt, a box and a mailer, about 200 control steps."""

HAS_SUPERDEX = importlib.util.find_spec("superdex") is not None


@unittest.skipUnless(HAS_SUPERDEX, "needs the [superdex] extra")
class TestTheBeltCarries(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss
        from robosimstudio_materials.bag import _BAG_HOST

        physics = rss.MUJOCO_500HZ.with_backend("bag", _BAG_HOST)
        scene = rss.Scene(cell="void", physics=physics, name="belt_carries")
        scene.add(rss.robot("yam_bimanual"))
        cls.belt = scene.add(rss.conveyor(at=(0.1, 0.0, 0.06), length=1.4, speed=0.2, rail_height=0.03))
        cls.box = scene.add(rss.box(name="box", half=(0.10, 0.075, 0.04), at=(-0.45, 0.10, 0.105), mass=0.4))
        cls.mailer = scene.add(rss.mailer(at=(-0.45, -0.10, 0.064)))
        cls.sim = scene.build(control_mode="joint_position")
        cls.hz = physics.control_hz
        cls.sim.run(30).sync()

    def _track(self, steps: int) -> np.ndarray:
        rows = []
        for k in range(steps):
            self.sim.run(1)
            if k % 10 == 0:
                self.sim.sync()
                rows.append((k / self.hz, self.box.position[0], np.asarray(self.mailer.particle_q)[:, 0].mean()))
        return np.asarray(rows, dtype=np.float64)

    def test_1_both_ride_at_the_belt_speed(self) -> None:
        track = self._track(100)
        box_v = np.polyfit(track[:, 0], track[:, 1], 1)[0]
        mailer_v = np.polyfit(track[:, 0], track[:, 2], 1)[0]
        self.assertAlmostEqual(0.2, box_v, delta=0.002)
        self.assertAlmostEqual(0.2, mailer_v, delta=0.01)
        self.assertAlmostEqual(
            0.100, float(self.box.position[2]), delta=0.002, msg="the box rides on the top: 60 mm + its 40 mm half"
        )
        self.assertGreater(float(np.asarray(self.mailer.particle_q)[:, 2].min()), 0.058, "the mailer sank")

    def test_2_a_stopped_belt_holds_what_it_carries(self) -> None:
        self.belt.stop()
        self.sim.run(10).sync()
        x0 = float(self.box.position[0])
        m0 = float(np.asarray(self.mailer.particle_q)[:, 0].mean())
        travel = self.belt.travel
        self.sim.run(50).sync()
        self.assertAlmostEqual(x0, float(self.box.position[0]), delta=0.001)
        self.assertAlmostEqual(m0, float(np.asarray(self.mailer.particle_q)[:, 0].mean()), delta=0.002)
        self.assertAlmostEqual(travel, self.belt.travel, places=9)
        self.belt.run()
        self.assertAlmostEqual(0.2, self.belt.speed)

    def test_3_a_reset_starts_the_belt_over(self) -> None:
        self.belt.speed = 0.05
        self.sim.reset()
        self.assertEqual(0.2, self.belt.speed, "a reset restores the spec's speed")
        self.sim.run(5).sync()
        self.assertAlmostEqual(5 / self.hz * 0.2, self.belt.travel, delta=0.004)


if __name__ == "__main__":
    unittest.main()
