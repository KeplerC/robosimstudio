# SPDX-License-Identifier: Apache-2.0
"""What a paste sticks to: a pat of butter put against the underside of a plate hangs there or falls.

Newton's implicit MPM decides whether a paste holds on to a surface by the collider's adhesion [Pa], reads the
friction off the shape, and has nowhere to read the adhesion from: it is zero. ``sticks_to`` on a particle body
is where a scene says it (``robosimstudio_materials/mpm/spec.py``), and the body writes it into the solver's
collider materials once the solver exists. The check is the one with no friction in it: a 7.5 mm pat, 65 Pa of
its own weight, against the underside of a plate 60 mm over the bench. Nothing is pinned.
"""

from __future__ import annotations

import unittest

import numpy as np

import robosimstudio as rss

TIER = "medium"
"""One GPU: three scenes of 768 particles, two of them stepped for a second."""

ADHESION = 2000.0
"""[Pa]; thirty times the pat's weight per area."""
PLATE_UNDER = 0.06
"""The plate's underside over the bench [m]."""
PAT = (0.04, 0.04, 0.0075)


def _scene(sticks_to):
    scene = rss.Scene(cell="bench_075", name="pat_under_a_plate", physics=rss.DOUGH_60HZ)
    scene.add(rss.slab(name="plate", half=(0.04, 0.04, 0.005), at=(0.0, 0.0, PLATE_UNDER + 0.005)))
    scene.add(rss.slab(name="saucer", half=(0.04, 0.04, 0.005), at=(0.12, 0.0, 0.005)))
    pat = scene.add(
        rss.dough(
            name="pat",
            size=PAT,
            at=(0.0, 0.0, PLATE_UNDER - PAT[2]),
            material="butter",
            settle_frames=0,
            sticks_to=sticks_to,
        )
    )
    return scene.build(), pat


def _fall(sim, pat, seconds: float = 1.0) -> np.ndarray:
    """How far each particle is under where it began after ``seconds`` [m]."""
    began = np.asarray(pat.particle_q)[:, 2].copy()
    sim.run(round(seconds * float(sim.request.physics.control_hz)))
    return began - np.asarray(pat.particle_q)[:, 2]


class TestAPasteSticksToWhatItNames(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loose_sim, loose = _scene({})
        cls.stuck_sim, stuck = _scene({"plate": ADHESION})
        cls.loose_table = loose.entity.solver._mpm_model.collider.material_adhesion.numpy().copy()
        cls.stuck_table = stuck.entity.solver._mpm_model.collider.material_adhesion.numpy().copy()
        cls.loose, cls.stuck = _fall(cls.loose_sim, loose), _fall(cls.stuck_sim, stuck)

    def test_nothing_named_leaves_every_collider_at_zero(self) -> None:
        self.assertEqual(0.0, float(np.abs(self.loose_table).max()))

    def test_the_named_plate_alone_takes_the_adhesion(self) -> None:
        """One material a collider shape behind Newton's default one: the bench, the plate and the saucer."""
        self.assertEqual([ADHESION], [float(v) for v in self.stuck_table if v != 0.0])

    def test_with_none_the_pat_falls_to_the_bench(self) -> None:
        self.assertTrue(np.isfinite(self.loose).all())
        self.assertGreater(float(self.loose.mean()), 0.045, "52 mm down to the bench, measured 51.8")
        self.assertEqual(0.0, float((self.loose < 0.002).mean()), "and none of it stays behind")

    def test_with_it_the_pat_hangs_under_the_plate(self) -> None:
        self.assertTrue(np.isfinite(self.stuck).all())
        self.assertLess(float(np.abs(self.stuck).max()), 0.002, "every particle within 2 mm of where it began")

    def test_a_name_with_no_collider_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "sticks_to names 'spoon'"):
            _scene({"spoon": ADHESION})


if __name__ == "__main__":
    unittest.main()
