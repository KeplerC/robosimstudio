# SPDX-License-Identifier: Apache-2.0
"""A loose light piece in a liquid, mirrored at its own mass: a 0.4 g chip let fall into a cup of thick liquid.

The particle arrangements mirror every rigid body into the MPM at 1000 times its mass, which keeps a held
tool from being pushed about by what it works and is wrong for a free piece lighter than the liquid a grid
cell holds: the liquid meets a mirror that weighs 0.4 kg and answers a chip that weighs 0.4 g.
``SimRequirements.proxy_mass_scale`` lets the one entity state its own (``robosimstudio/specs/declare.py``).
The scene is the cup of ``examples/03_materials/12_liquid`` cut down, on a 1.5 mm grid stepped at 240 Hz, and
the chip is 12 by 10 by 3 mm: a shard of a bonded shell. Nothing is pinned; it is let go 20 mm over the liquid.
"""

from __future__ import annotations

import unittest

import numpy as np

import robosimstudio as rss
from robosimstudio_materials.liquid import HONEY, cup

TIER = "medium"
"""One GPU: two scenes of 38,400 particles at 240 Hz, three seconds each."""

BENCH = np.array([0.6, 0.0, 0.75])
INNER, WALL, DEPTH = (0.03, 0.03, 0.03), 0.006, 0.018
"""The cup's cavity, its walls and the liquid's depth [m]."""
CHIP_HALF, CHIP_MASS = (0.006, 0.005, 0.0015), 0.0004
"""[m] and [kg]: denser than the liquid, so left alone it sinks to the cup's floor."""
SECONDS = 3.0


def _drop(own: float | None) -> dict:
    """Let the chip fall and watch it for :data:`SECONDS`; ``own`` is its proxy mass scale, or ``None``."""
    grid = {"voxel_size": 0.0015, "step_hz": 240.0, "max_active_cell_count": 1 << 17, "max_leaf_node_count": 8192}
    mpm = rss.backend("mpm", project_outside=True, **grid)
    table = rss.LIQUID_120HZ.with_backend("liquid", mpm)
    hard = rss.CollisionSpec(contact_ke=6.0e4, contact_kd=500.0, rigid_gap=2.0e-3)
    scene = rss.Scene(cell="bench_075", name="chip_in_a_cup", physics=rss.replace(table, collision=hard))
    scene.add(cup(name="cup", inner=INNER, wall=WALL, at=(0.0, 0.0, 0.0)))
    liquid = scene.add(
        rss.liquid(
            name="filling",
            size=(INNER[0], INNER[1], DEPTH),
            at=(0.0, 0.0, WALL + 0.0005),
            material=rss.replace(HONEY.spec, density=1030.0, viscosity=2.0),
            particle_spacing=0.00075,
            containers=("cup", "chip"),
            container_margin=0.00075,
        )
    )
    declared = None if own is None else rss.SimRequirements(proxy_mass_scale=own)
    scene.add(
        rss.box(
            name="chip", half=CHIP_HALF, at=(0.0, 0.0, WALL + DEPTH + 0.02), mass=CHIP_MASS, friction=0.2, sim=declared
        )
    )
    sim = scene.build()
    chip = sim["chip"]
    fastest, finite = 0.0, True
    for _ in range(round(SECONDS * float(sim.request.physics.control_hz))):
        sim.run(1)
        velocity = np.asarray(chip.velocity, dtype=np.float64)[:3]
        finite = finite and bool(np.isfinite(velocity).all())
        if not finite:
            break
        fastest = max(fastest, float(np.linalg.norm(velocity)))
    at = np.asarray(chip.position, dtype=np.float64) - BENCH
    inside = bool(abs(at[0]) < INNER[0] / 2 and abs(at[1]) < INNER[1] / 2 and WALL - 0.001 < at[2] < WALL + INNER[2])
    return {
        "scales": sorted(float(m.mass_scale) for m in sim.pipeline.program.solver._proxy_mappings),
        "finite": finite and bool(np.isfinite(np.asarray(liquid.particle_q)).all()),
        "fastest": fastest,
        "at": at,
        "in_the_cup": finite and inside,
    }


class TestALoosePieceCarriesItsOwnProxyMass(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.own, cls.shared = _drop(1.0), _drop(None)

    def test_only_the_chip_is_mirrored_at_its_own_scale(self) -> None:
        self.assertEqual([1.0, 1000.0], self.own["scales"])
        self.assertEqual([1000.0], self.shared["scales"])

    def test_at_its_own_mass_the_chip_sinks_and_stays_in_the_cup(self) -> None:
        """Let go 20 mm up it meets the liquid at 0.63 m/s, and nothing after that is faster."""
        self.assertTrue(self.own["finite"])
        self.assertLess(self.own["fastest"], 1.0, "measured 0.65 m/s")
        self.assertTrue(self.own["in_the_cup"], f"it ended at {np.round(1e3 * self.own['at'], 1)} mm")
        self.assertLess(float(self.own["at"][2]), WALL + 0.004, "on the cup's floor: it is denser than the liquid")

    def test_at_a_thousand_times_its_mass_the_liquid_throws_it(self) -> None:
        """The failure the field is for, kept as the control: out of the cup, or faster than it fell."""
        thrown = not self.shared["in_the_cup"] or self.shared["fastest"] > 1.0
        self.assertTrue(
            thrown, f"fastest {self.shared['fastest']:.2f} m/s, ended at {np.round(1e3 * self.shared['at'], 1)} mm"
        )


if __name__ == "__main__":
    unittest.main()
