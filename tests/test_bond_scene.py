# SPDX-License-Identifier: Apache-2.0
"""A set of bonds in a built scene: where it works, what a reset does, and what it does to the bodies it joins.

``tests/test_bond_pull``, ``test_bond_peel`` and ``test_bond_hold`` are the rule's numbers. These are
the seams around it, each a sentence of ``robosimstudio_materials/bond.py``:

* **A reset makes every bond again.** Broken is for the episode: after ``sim.reset()`` the rows are on
  in the solver, nothing is remembered, and the same pull breaks them at the same force.
* **Bonded bodies do not collide with each other** unless asked, so two pieces built overlapping stay
  where they were built; asked to collide, the same pair is pushed apart and its bonds break.
* **The pieces take the stated contact impedance**, in the solver's own model.
* **Both bodies have to be on MuJoCo.** ``rss.GENERIC`` puts free bodies on VBD, and the build says so
  rather than hold nothing.
* **The module's own example is run.** The kind ships with no example script, so the snippet in its
  docstring is taken out of the docstring and executed here, and what it prints is what it says.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.fixtures.bonds import Wrenches

TIER = "medium"
"""Six scenes of two blocks on one GPU, stepped a few seconds of scene time between them."""

HALF = (0.02, 0.02, 0.01)


def _pair(*, overlap: float = 0.0, physics=None, gravity: float = 0.0, **bond):
    import robosimstudio as rss

    physics = rss.replace(rss.MUJOCO_500HZ, gravity=gravity) if physics is None else physics
    scene = rss.Scene(cell="void", name="pair", physics=physics)
    scene.add(rss.box(name="lower", half=HALF, at=(0.0, 0.0, 0.5), mass=0.1))
    scene.add(rss.box(name="upper", half=HALF, at=(0.0, 0.0, 0.5 + 2 * HALF[2] - overlap), mass=0.1))
    seam = 0.5 + HALF[2] - 0.5 * overlap
    ring = [(0.012 * np.cos(a), 0.012 * np.sin(a), seam) for a in 2.0 * np.pi * np.arange(4) / 4]
    scene.add(rss.bond(name="glue", a="upper", b="lower", at=ring, normal=(0.0, 0.0, 1.0), strength=2.0, **bond))
    return scene


def _gap(sim) -> float:
    return float(sim["upper"].position[2] - sim["lower"].position[2])


class TestAResetMakesEveryBondAgain(unittest.TestCase):
    def test_broken_is_for_the_episode(self) -> None:
        sim = _pair().build()
        glue = sim["glue"].entity
        upper, lower = int(sim["upper"].entity.body), int(sim["lower"].entity.body)
        wrenches = Wrenches(sim)

        def pull_apart() -> float:
            start = sim.time
            for _ in range(400):
                pull = 8.0 * (sim.time - start) / 3.0
                wrenches.set(upper, force=(0.0, 0.0, pull))
                wrenches.set(lower, force=(0.0, 0.0, -pull))
                sim.run(1)
                if glue.let_go().all():
                    return pull
            self.fail("the bonds never broke")

        first = pull_apart()
        self.assertEqual(0, glue.held())
        wrenches.set(upper)
        wrenches.set(lower)
        sim.reset()
        self.assertEqual(4, glue.held(), "a reset left a bond broken")
        self.assertTrue(glue.active().all(), "a reset left a row off in the solver")
        np.testing.assert_array_equal(glue.peaks(), 0.0, err_msg="a reset kept the last episode's loads")
        sim.run(25)
        self.assertEqual(4, glue.held())
        self.assertAlmostEqual(2 * HALF[2], _gap(sim), delta=1.0e-4, msg="the remade bonds do not hold the pair")
        again = pull_apart()
        self.assertAlmostEqual(first, again, delta=0.02 * first, msg="the remade bonds broke at another force")


class TestBondedBodiesDoNotCollide(unittest.TestCase):
    def test_an_overlapping_pair_stays_as_built(self) -> None:
        sim = _pair(overlap=0.004).build()
        sim.run(50)
        glue = sim["glue"].entity
        self.assertEqual(4, glue.held())
        self.assertAlmostEqual(2 * HALF[2] - 0.004, _gap(sim), delta=1.0e-4)
        self.assertLess(float(glue.peaks().max()), 0.05, "the pair's own contact loads its bonds")

    def test_asked_to_collide_the_same_pair_is_pushed_apart(self) -> None:
        sim = _pair(overlap=0.004, collide=True).build()
        sim.run(50)
        glue = sim["glue"].entity
        self.assertGreater(float(glue.peaks().max()), 0.5, "4 mm of overlap put no load on the bonds")


class TestThePiecesTakeTheStatedImpedance(unittest.TestCase):
    def test_it_reaches_the_solver(self) -> None:
        from robosimstudio_materials.bond import PIECE_IMPEDANCE

        sim = _pair(piece_impedance=PIECE_IMPEDANCE, piece_mix=50.0).build()
        solver = sim["glue"].entity.solver
        solimp = np.asarray(solver.mjw_model.geom_solimp.numpy()).reshape(-1, 5)
        solmix = np.asarray(solver.mjw_model.geom_solmix.numpy()).reshape(-1)
        stiff = np.isclose(solimp[:, 0], PIECE_IMPEDANCE[0])
        self.assertEqual(2, int(stiff.sum()), "the two blocks' boxes, and nothing else in the scene")
        np.testing.assert_allclose(solimp[stiff], [PIECE_IMPEDANCE] * 2, rtol=1e-6)
        np.testing.assert_allclose(solmix[stiff], 50.0)
        np.testing.assert_allclose(solmix[~stiff], 1.0)


class TestBothBodiesHaveToBeOnMuJoCo(unittest.TestCase):
    def test_free_bodies_on_vbd_are_refused_at_the_build(self) -> None:
        import robosimstudio as rss

        with self.assertRaisesRegex(RuntimeError, "MUJOCO_500HZ"):
            _pair(physics=rss.GENERIC).build()


class TestTheModulesOwnExample(unittest.TestCase):
    """The usage snippet in ``robosimstudio_materials/bond.py``'s docstring, executed as it is written."""

    @classmethod
    def setUpClass(cls) -> None:
        import contextlib
        import io
        import itertools
        import textwrap

        from robosimstudio_materials import bond

        block = bond.__doc__.split("::\n\n", 1)[1].splitlines()
        indented = itertools.takewhile(lambda line: not line.strip() or line.startswith("    "), block)
        cls.source = textwrap.dedent("\n".join(indented))
        cls.names: dict = {}
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            exec(compile(cls.source, "robosimstudio_materials/bond.py docstring", "exec"), cls.names)
        cls.printed = printed.getvalue().splitlines()[-2:]

    def test_it_is_the_whole_of_a_scene(self) -> None:
        self.assertIn("import robosimstudio as rss", self.source)
        self.assertIn("rss.bond(", self.source)
        self.assertIn("scene.build()", self.source)

    def test_the_light_block_hangs_and_the_heavy_one_fell(self) -> None:
        sim, hooks = self.names["sim"], self.names["hooks"].entity
        self.assertEqual([False, True], hooks.let_go().tolist())
        surface = 0.75  # bench_075's top in the world [m]
        self.assertAlmostEqual(surface + 0.3, float(sim["light"].position[2]), delta=0.001)
        self.assertAlmostEqual(surface + 0.02, float(sim["heavy"].position[2]), delta=0.002, msg="on the bench")

    def test_it_prints_what_its_comments_say(self) -> None:
        self.assertEqual("[False  True]", self.printed[0])
        loads = self.names["hooks"].entity.loads()
        self.assertAlmostEqual(0.1 * 9.81, float(loads[0]), delta=0.02, msg="the light block's weight")
        self.assertEqual(0.0, float(loads[1]))
        self.assertEqual("[0.98 0.  ]", self.printed[1])


if __name__ == "__main__":
    unittest.main()
