# SPDX-License-Identifier: Apache-2.0
"""The mint gate over the atom catalog."""

import unittest
from types import SimpleNamespace
from unittest import mock

from robosimstudio_bench.goals import mint

TIER = "fast"
"""The registered atoms' metadata only: no scene, no GPU."""


class TestTheMintGateListsTheCatalogOnce(unittest.TestCase):
    def test_once(self):
        with (
            mock.patch.object(mint, "list_atoms", return_value=["a", "b", "c"]) as list_atoms,
            mock.patch.object(mint, "get_atom", return_value=SimpleNamespace()),
            mock.patch.object(mint, "mint_problems", return_value=[]) as problems,
            mock.patch.object(mint, "MintReport", side_effect=lambda *a, **k: None),
        ):
            mint.mint_gate(exempt={})
        self.assertEqual(list_atoms.call_count, 1)
        self.assertEqual({call.kwargs["among"] for call in problems.call_args_list}, {("a", "b", "c")})


if __name__ == "__main__":
    unittest.main()
