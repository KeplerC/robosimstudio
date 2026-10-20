# SPDX-License-Identifier: Apache-2.0
"""A declaration made from a constructor's spec builds through one value, :class:`SpecFactory`.

The registry keeps the first of two declarations under one name when they say the
same thing, and compares a ``build`` factory by its name -- or, for a value with
none, by its ``repr``. A value factory therefore makes the same thing declared
twice one asset, and two different things under one name a clash; a lambda would
make every re-declaration the first. Each kind's ``from_spec`` builds through the
one class in :mod:`robosimstudio.assetlib.record`.
"""

from __future__ import annotations

import unittest

TIER = "fast"
"""Specs and records only: nothing is built."""


def _declarations(size: float):
    """The same kinds of thing at two sizes: ``(kind, record)`` pairs."""
    import robosimstudio as rss
    from robosimstudio.domains.rigid import Body
    from robosimstudio_materials.bag import Bag
    from robosimstudio_materials.brush import Brush
    from robosimstudio_materials.cutting import Cuttable
    from robosimstudio_materials.mpm.declare import Heap
    from robosimstudio_materials.rod import Rod
    from robosimstudio_materials.rod.band import band
    from robosimstudio_materials.tape import Tape
    from robosimstudio_materials.water import Water

    knife = rss.knife(name="knife", blade_length=4 * size)
    return (
        ("rigid", Body.from_spec(rss.box(name="cube", half=(size,) * 3), name="cube")),
        ("granular", Heap.from_spec(rss.heap(name="rice", size=(size, size, size)), name="rice")),
        ("water", Water.from_spec(rss.water(name="pour", size=(size, size, size)), name="pour")),
        ("bag", Bag.from_spec(rss.bag(name="bag", size=(8 * size, 8 * size)), name="bag")),
        ("rod", Rod.from_spec(band(name="band", diameter=2 * size), name="band")),
        ("brush", Brush.from_spec(rss.brush(name="brush", sole=(4 * size, 2 * size, 0.008)), name="brush")),
        ("cuttable", Cuttable.from_spec(rss.cuttable(name="tofu", knife=knife, size=(size,) * 3), name="tofu")),
        ("knife", Cuttable.knife(knife, name="knife")),
        ("tape", Tape.from_spec(rss.tape(name="label", on="box", size=(2 * size, 4 * size)), name="label")),
    )


class TestADeclarationFromASpecIsAValue(unittest.TestCase):
    def test_every_kind_builds_through_the_one_factory(self) -> None:
        from robosimstudio.assetlib.record import SpecFactory

        for kind, record in _declarations(0.025):
            with self.subTest(kind=kind):
                self.assertIsInstance(record.build, SpecFactory)
                self.assertEqual(record.build.spec, record.build())

    def test_the_same_spec_twice_is_one_asset_and_two_specs_are_a_clash(self) -> None:
        from robosimstudio.assetlib.registry import _same_record

        for (kind, first), (_, again), (_, other) in zip(
            _declarations(0.025), _declarations(0.025), _declarations(0.03), strict=True
        ):
            with self.subTest(kind=kind):
                self.assertTrue(_same_record(first, again))
                self.assertFalse(_same_record(first, other))

    def test_a_rod_laid_from_an_array_is_a_value_too(self) -> None:
        """``overhand()`` returns an array: the spec holds tuples, so it compares, hashes and registers twice."""
        from unittest import mock

        import robosimstudio as rss
        from robosimstudio.assetlib.registry import ASSETS, register_asset
        from robosimstudio_materials.rod import Rod
        from robosimstudio_materials.rod.knot import overhand

        nodes = overhand(radius=0.006, size=0.10)
        first, again = (rss.RodSpec(name="rope", nodes=n, stress_free=n, radius=0.006) for n in (nodes, nodes.copy()))
        self.assertEqual(first, again)
        self.assertEqual(hash(first), hash(again))
        self.assertIsInstance(first.nodes[0][0], float)
        self.assertIsInstance(first.stress_free, tuple)
        self.assertEqual(Rod.from_spec(first, name="rope").build, Rod.from_spec(again, name="rope").build)
        with mock.patch.dict(ASSETS._items), mock.patch.dict(ASSETS._origins):
            for spec in (first, again):
                register_asset(Rod.from_spec(spec, name="knotted_rope"))

    def test_a_rod_from_its_spec_states_its_length_radius_and_self_contact(self) -> None:
        from robosimstudio_materials.rod import Rod
        from robosimstudio_materials.rod.band import band

        loop = band(diameter=0.08, self_contact=True)
        record = Rod.from_spec(loop, name="rubber_band")
        self.assertEqual("rod", record.sim.domain)
        self.assertTrue(record.sim.self_contact)
        self.assertIn(f"{loop.length:.3g} m", record.summary)
        with self.assertRaises(TypeError):
            Rod.from_spec(object(), name="rubber_band")

    def test_a_rods_record_states_the_requirements_the_rod_runs(self) -> None:
        """Read as ``EntityRodSuperdex.sim_requirements`` reads the spec: a declared ``sim`` first, else its self-contact.

        Self-contact is the spec's, which the rod is built from, so one passed beside it is refused
        rather than stated on the record alone; so are contacts beside a spec that declares them.
        """
        from robosimstudio.specs.physics import SimRequirements
        from robosimstudio_materials.rod import Rod
        from robosimstudio_materials.rod.band import band
        from robosimstudio_materials.rod.superdex import EntityRodSuperdex

        for on in (True, False):
            for declared in (None, SimRequirements(domain="rod", self_contact=on)):
                with self.subTest(self_contact=on, declared=declared is not None):
                    spec = band(diameter=0.08, self_contact=on if declared is None else not on, sim=declared)
                    record = Rod.from_spec(spec, name="rubber_band")
                    self.assertIs(EntityRodSuperdex(spec).sim_requirements().self_contact, record.sim.self_contact)
                    self.assertIs(on, record.sim.self_contact)
                    if declared is not None:
                        self.assertEqual(declared, record.sim)
        with self.assertRaisesRegex(TypeError, "self_contact is the spec's"):
            Rod.from_spec(band(diameter=0.08), name="rubber_band", self_contact=True)
        declares = band(diameter=0.08, sim=SimRequirements(domain="rod"))
        with self.assertRaisesRegex(ValueError, "state touches there"):
            Rod.from_spec(declares, name="rubber_band", touches=("hands",))


class TestTheScoredBandExampleDeclaresItsBandAsAValue(unittest.TestCase):
    """``examples/06_tasks/07_band_scored``'s ``main()`` again with another band: refused, not the first kept."""

    def test_a_second_main_with_a_different_band_is_a_clash(self) -> None:
        from unittest import mock

        from robosimstudio.assetlib.registry import ASSETS
        from tests.fixtures.examples import _load

        module = _load("06_tasks/07_band_scored.py")
        real = module.band
        with (
            mock.patch.dict(ASSETS._items),
            mock.patch.dict(ASSETS._origins),
            mock.patch.object(module.bench, "Env", lambda task: mock.Mock()),
            mock.patch.object(module.rss, "view", lambda *args, **kwargs: None),
        ):
            module.main()
            module.main()  # the same band again is the same asset
            with mock.patch.object(module, "band", lambda **kw: real(**{**kw, "diameter": 0.06})):
                with self.assertRaisesRegex(ValueError, "rubber_band"):
                    module.main()


if __name__ == "__main__":
    unittest.main()
