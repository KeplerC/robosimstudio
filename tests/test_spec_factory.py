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

    return (
        ("rigid", Body.from_spec(rss.box(name="cube", half=(size,) * 3), name="cube")),
        ("granular", Heap.from_spec(rss.heap(name="rice", size=(size, size, size)), name="rice")),
        ("bag", Bag.from_spec(rss.bag(name="bag", size=(8 * size, 8 * size)), name="bag")),
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


if __name__ == "__main__":
    unittest.main()
