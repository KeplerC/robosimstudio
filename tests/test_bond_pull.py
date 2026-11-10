# SPDX-License-Identifier: Apache-2.0
"""Two blocks joined by N bonds and drawn apart come apart at N times the bond's strength.

``robosimstudio_materials/bond.py`` says a bond holds to its strength and then lets go. This is that
claim on the simplest joint there is: one block on another, no gravity, the pair's contact off (bonded
bodies do not collide), N bonds of 2 N in the plane between them, and equal and opposite forces along
the bonds' normal ramped from nothing over six seconds, 0.33% of N x strength a control step. Nothing
may let go before 99% of N x strength, everything has let go by 101%, the solver's own rows are off
with them, and the graph captured at the build is the one that ran. A 1 g pair is here because a
constraint between light bodies is the soft case (the module's "How stiff").
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.fixtures.bonds import Wrenches

TIER = "medium"
"""Four two-block scenes on one GPU, each stepped for six seconds of scene time."""

STRENGTH = 2.0
"""Each bond's strength [N]."""
HALF = (0.02, 0.02, 0.01)
"""The blocks' half-extents [m]."""
RAMP = 6.0
"""Seconds for the pull to reach N x strength."""


def _pull(bonds: int, mass: float) -> dict:
    """Build the pair, ramp the pull, and say when the bonds went."""
    import robosimstudio as rss

    scene = rss.Scene(cell="void", name="pull", physics=rss.replace(rss.MUJOCO_500HZ, gravity=0.0))
    scene.add(rss.box(name="lower", half=HALF, at=(0.0, 0.0, 0.5), mass=mass))
    scene.add(rss.box(name="upper", half=HALF, at=(0.0, 0.0, 0.5 + 2 * HALF[2]), mass=mass))
    angles = 2.0 * np.pi * np.arange(bonds) / bonds
    ring = (
        [(0.012 * np.cos(a), 0.012 * np.sin(a), 0.5 + HALF[2]) for a in angles]
        if bonds > 1
        else [(0.0, 0.0, 0.5 + HALF[2])]
    )
    scene.add(rss.bond(name="glue", a="upper", b="lower", at=ring, normal=(0.0, 0.0, 1.0), strength=STRENGTH))
    sim = scene.build()
    glue = sim["glue"].entity
    upper, lower = int(sim["upper"].entity.body), int(sim["lower"].entity.body)
    wrenches = Wrenches(sim)
    whole = bonds * STRENGTH
    out = {"whole": whole, "first": None, "last": None, "stretch": None}
    gap0 = float(sim["upper"].position[2] - sim["lower"].position[2])
    while sim.time < 1.5 * RAMP:
        pull = whole * sim.time / RAMP
        wrenches.set(upper, force=(0.0, 0.0, pull))
        wrenches.set(lower, force=(0.0, 0.0, -pull))
        sim.run(1)
        gone = glue.let_go()
        if out["stretch"] is None and pull >= 0.8 * whole:
            out["stretch"] = float(sim["upper"].position[2] - sim["lower"].position[2]) - gap0
        if gone.any() and out["first"] is None:
            out["first"] = pull
        if gone.all():
            out["last"] = pull
            break
    sim.run(10)
    out.update(
        active=glue.active(),
        broke_at=glue.broke_at(),
        gap=float(sim["upper"].position[2] - sim["lower"].position[2]) - gap0,
        captured=bool(sim.pipeline.captured),
    )
    return out


class TestBlocksPartAtNTimesTheStrength(unittest.TestCase):
    def test_the_pair_parts_within_one_percent(self) -> None:
        for bonds, mass in ((1, 0.1), (4, 0.1), (8, 0.1), (4, 0.001)):
            with self.subTest(bonds=bonds, mass=mass):
                got = _pull(bonds, mass)
                whole = got["whole"]
                self.assertIsNotNone(got["last"], f"the bonds held past {1.5 * whole:.1f} N")
                self.assertGreaterEqual(got["first"], 0.99 * whole, "a bond let go early")
                self.assertLessEqual(got["last"], 1.01 * whole, "a bond held past its strength")
                self.assertFalse(got["active"].any(), "the solver still holds a row of a broken bond")
                np.testing.assert_array_less(STRENGTH, got["broke_at"], err_msg="each went past its own strength")
                # The first to go went at its strength; the rest took its share within a substep and read more.
                self.assertLess(got["broke_at"].min(), 1.02 * STRENGTH)
                self.assertLess(abs(got["stretch"]), 1.0e-4, "the joint stretched 0.1 mm before it broke")
                self.assertGreater(got["gap"], 1.0e-3, "the blocks did not part")
                self.assertTrue(got["captured"], "the break rule dropped the captured graph")


if __name__ == "__main__":
    unittest.main()
