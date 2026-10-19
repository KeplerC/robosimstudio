# SPDX-License-Identifier: Apache-2.0
"""The wipe example's gesture, made the way its page makes it, moves the puddle off its patch.

``tests/scenes/01_wipe_spill`` promises that a hand dragged down onto the
wiper, closed, and dragged toward the middle of the bench pushes the ketchup off
the patch it settled on. This drives the example's own arm panel through
:mod:`tests.fixtures.grasp` -- the cursor the gizmo writes, the grip the Close
button writes, the session's own setpoint law -- and reads the same count the
page shows. Nothing is pinned: the wiper stays in the fingers because the pads
hold it, and the ketchup moves because the wiper's face moves it.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import drag_to_grasp, go, open_session, tick

TIER = "medium"
"""One GPU: the example built once and stepped through its page for about 12 s of scene time."""

PORT = 9622
SWEEP = 0.18
"""How far the hand is dragged toward the middle of the bench [m]: the puddle's far edge and 30 mm past it."""
SPEED = 0.09
"""The cursor's walk [m/s]; the panel's setpoint law paces the hand under it, and both are under the
0.36 m/s a 4 mm voxel allows a tool at 60 Hz."""


class TestTheWipeByHand(unittest.TestCase):
    """Down onto the wiper, Close, across the puddle: at least half the ketchup leaves its patch."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim, kwargs = _run("tests/scenes/01_wipe_spill.py")
        cls.session = open_session(cls.sim, PORT, kwargs.get("page"))
        tick(cls.session, 30)
        cls.sauce, cls.wiper = cls.sim["ketchup"], cls.sim["wiper"]
        rest = cls.sauce.entity.rest["particle_q"][cls.sauce.particles]
        pad = 0.5 * float(cls.sauce.spec.particle_spacing)
        cls.lo = rest.min(axis=0) - np.array((pad, pad, 0.0))
        cls.hi = rest.max(axis=0) + np.array((pad, pad, 0.03))
        cls.floor = float(cls.sauce.entity.floor_z)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def _pushed_off(self) -> float:
        from robosimstudio_materials.mpm.measures import spill_fraction

        return spill_fraction(self.sauce.particle_q, self.lo, self.hi)

    def test_the_hand_starts_over_the_wiper_pointing_down(self) -> None:
        pos, quat = self.session.arms["right"].ik.home_pose
        standing = np.asarray(self.wiper.position, dtype=np.float64)
        self.assertLess(float(np.linalg.norm(pos[:2] - standing[:2])), 0.01, "the home tool centre is over the wiper")
        self.assertGreater(float(pos[2] - standing[2]), 0.05, "and above it")
        # The tool's +z in world is the third column of the rotation: pointing down is (0, 0, -1).
        x, y, z, w = quat
        tool_z = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        self.assertLess(float(tool_z[2]), -0.95, f"the tool points down: {tool_z.round(3)}")

    def test_the_sweep_pushes_the_puddle_off_its_patch(self) -> None:
        session, wiper = self.session, self.wiper
        self.assertEqual(0.0, self._pushed_off(), "nothing has left the patch before the hand moves")
        standing = np.asarray(wiper.position, dtype=np.float64)
        at = np.array([standing[0], standing[1], self.floor + 0.0165])  # the pads' tips on the bench
        grasp = drag_to_grasp(
            session, "right", lambda: wiper.position, at=at, hover=0.05, lift=0.0, close=0.75, hold=0.25
        )
        self.assertTrue(grasp.finite)
        self.assertAlmostEqual(0.020, grasp.width, delta=0.003, msg="the fingers are on the wiper's 20 mm faces")
        self.assertLess(grasp.drift, 0.005, "the wiper did not slide in the fingers")
        hand = session.arms["right"].ik.ee_pose(session.env.state)[0]
        miss = go(session, "right", hand + np.array([0.0, SWEEP, 0.0]), speed=SPEED, steps=120)
        self.assertLess(miss, 0.015, f"the hand got across: {miss * 1e3:.1f} mm short")
        tick(session, 60)
        after = np.asarray(wiper.position, dtype=np.float64)
        self.assertGreater(float(after[1] - standing[1]), SWEEP - 0.03, "the wiper came along with the hand")
        self.assertLess(abs(float(after[2] - standing[2])), 0.01, "and stayed on its edge")
        pushed = self._pushed_off()
        self.assertGreaterEqual(pushed, 0.5, f"{pushed:.3f} of the ketchup left its patch; the page asks for half")
        self.assertTrue(self.sim.state_is_finite())
        self.assertTrue(self.sim.pipeline.captured, "the gesture kept the graph")


if __name__ == "__main__":
    unittest.main()
