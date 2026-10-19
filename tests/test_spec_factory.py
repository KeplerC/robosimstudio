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
    from robosimstudio_materials.mpm.declare import Heap
    from robosimstudio_materials.rod import Rod
    from robosimstudio_materials.rod.band import band

    return (
        ("rigid", Body.from_spec(rss.box(name="cube", half=(size,) * 3), name="cube")),
        ("granular", Heap.from_spec(rss.heap(name="rice", size=(size, size, size)), name="rice")),
        ("bag", Bag.from_spec(rss.bag(name="bag", size=(8 * size, 8 * size)), name="bag")),
        ("rod", Rod.from_spec(band(name="band", diameter=2 * size), name="band")),
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


class TestTheScoredBandExampleDeclaresItsBandAsAValue(unittest.TestCase):
    """``examples/18_bands/01_band_on_pegs_scored``'s ``main()`` again with another band: refused, not the first kept."""

    def test_a_second_main_with_a_different_band_is_a_clash(self) -> None:
        from unittest import mock

        from robosimstudio.assetlib.registry import ASSETS
        from tests.fixtures.examples import _load

        module = _load("18_bands/01_band_on_pegs_scored.py")
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
