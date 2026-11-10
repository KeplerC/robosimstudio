# SPDX-License-Identifier: Apache-2.0
"""A bonded joint loaded at 80% of its strength for ten seconds breaks nothing and does not creep.

A block hangs from the world by four bonds of 2 N, under gravity. A force is ramped over half a
second to 80% of the joint's 8 N, the block's weight counted in, and held for ten seconds: straight
down, which is tension, and along the plane of the bonds, which is shear (applied in that plane, so
the bonds carry no moment). A 100 g block and a 1 g one, because a MuJoCo constraint is as stiff as the
lighter body is heavy. No bond may break, none may read more than its share, and the block may not
move 0.1 mm between half a second after the ramp and the end; how far the ramp itself moved it is the
joint's elastic stretch, and is under 0.05 mm for the 1 g block.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.fixtures.bonds import Wrenches

TIER = "medium"
"""Four one-block scenes on one GPU, each stepped for eleven seconds of scene time."""

STRENGTH, BONDS = 2.0, 4
HALF = (0.02, 0.02, 0.01)
RAMP, SETTLE, HOLD = 0.5, 0.5, 10.0
"""Seconds: the ramp to 80%, the wait before the creep is measured from, and the hold."""
GRAVITY = 9.81


def _hold(case: str, mass: float) -> dict:
    import robosimstudio as rss

    top = 0.5
    scene = rss.Scene(cell="void", name="hold", physics=rss.MUJOCO_500HZ)
    scene.add(rss.box(name="block", half=HALF, at=(0.0, 0.0, top - HALF[2]), mass=mass))
    angles = 2.0 * np.pi * (np.arange(BONDS) + 0.5) / BONDS
    ring = [(0.012 * np.cos(a), 0.012 * np.sin(a), top) for a in angles]
    scene.add(rss.bond(name="glue", a="block", at=ring, normal=(0.0, 0.0, -1.0), strength=STRENGTH))
    sim = scene.build()
    glue = sim["glue"].entity
    block = int(sim["block"].entity.body)
    wrenches = Wrenches(sim)
    weight = mass * GRAVITY
    load = 0.8 * BONDS * STRENGTH
    start = np.array(sim["block"].position, dtype=np.float64)
    rest, creep, peak = None, 0.0, 0.0
    while sim.time < RAMP + SETTLE + HOLD:
        s = min(sim.time / RAMP, 1.0)
        if case == "tension":
            wrenches.set(block, force=(0.0, 0.0, -s * (load - weight)))
        else:
            # Along x in the plane of the bonds, 10 mm over the block's centre: the force and the couple that puts it there.
            across = s * float(np.sqrt(load**2 - weight**2))
            wrenches.set(block, force=(across, 0.0, 0.0), torque=(0.0, across * HALF[2], 0.0))
        sim.run(1)
        peak = max(peak, float(glue.loads().max()))
        here = np.array(sim["block"].position, dtype=np.float64)
        if rest is None and sim.time >= RAMP + SETTLE:
            rest = here
        if rest is not None:
            creep = max(creep, float(np.linalg.norm(here - rest)))
    return {
        "broken": int(glue.let_go().sum()),
        "active": int(glue.active().sum()),
        "stretch": float(np.linalg.norm(rest - start)),
        "creep": creep,
        "peak": peak,
        "captured": bool(sim.pipeline.captured),
    }


class TestAJointHoldsAtEightyPercent(unittest.TestCase):
    def test_nothing_breaks_and_nothing_creeps(self) -> None:
        for case, mass in (("tension", 0.1), ("shear", 0.1), ("tension", 0.001), ("shear", 0.001)):
            with self.subTest(case=case, mass=mass):
                got = _hold(case, mass)
                self.assertEqual(0, got["broken"], "a bond let go at 80% of its strength")
                self.assertEqual(BONDS, got["active"])
                self.assertLess(got["creep"], 1.0e-4, f"the joint crept {got['creep'] * 1e3:.4f} mm in ten seconds")
                self.assertLess(
                    got["stretch"], 5.0e-5, f"the joint stretched {got['stretch'] * 1e3:.4f} mm under the load"
                )
                self.assertLess(got["peak"], 0.8 * STRENGTH * 1.05, "a bond read more than its share of the load")
                self.assertGreater(got["peak"], 0.8 * STRENGTH * 0.95, "the bonds did not carry the load")
                self.assertTrue(got["captured"])


if __name__ == "__main__":
    unittest.main()
