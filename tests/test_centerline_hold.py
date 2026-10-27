# SPDX-License-Identifier: Apache-2.0
"""Every rod storage answers ``hold`` with the one signature ``Centerline`` states.

``Centerline.hold`` is the affordance; the storage decides the mechanism. To the
caller -- a routine, a graph tool -- the capsule chain and the SuperDex rod are
one method with one signature, and the same ``held`` flag comes back in
``published_states``. That the hold works on a built scene is
``tests.test_hold_per_storage``'s; this is the contract, read off the classes.
"""

from __future__ import annotations

import unittest

TIER = "fast"
"""Reads two classes' signatures: no scene, no GPU."""


class TestTheContractIsOneMethod(unittest.TestCase):
    def test_both_implementations_answer_hold(self) -> None:
        import inspect

        from robosimstudio_materials.rod import Centerline, EntityRod, EntityRodSuperdex

        for cls in (EntityRod, EntityRodSuperdex):
            with self.subTest(cls=cls.__name__):
                self.assertIn("hold", vars(cls), "each storage answers for itself")
                self.assertEqual(
                    list(inspect.signature(cls.hold).parameters),
                    list(inspect.signature(Centerline.hold).parameters),
                    "one signature, so a caller need not know which it holds",
                )
                self.assertIn("published_states", dir(cls))


if __name__ == "__main__":
    unittest.main()
