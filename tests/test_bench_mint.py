# SPDX-License-Identifier: Apache-2.0
"""The mint gate over the atom catalog."""

import unittest
from unittest import mock

from robosimstudio_bench.goals import mint

TIER = "fast"
"""The registered atoms' metadata only: no scene, no GPU."""


class TestTheMintGateListsTheCatalogOnce(unittest.TestCase):
    def test_once(self):
        """One report per registered atom, in catalog order, from one listing of the catalog."""
        with mock.patch.object(mint, "list_atoms", wraps=mint.list_atoms) as list_atoms:
            reports = mint.mint_gate(exempt={})
        self.assertEqual(list_atoms.call_count, 1)
        self.assertEqual([report.atom for report in reports], mint.list_atoms())
        self.assertGreater(len(reports), 10)


if __name__ == "__main__":
    unittest.main()
