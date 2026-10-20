# SPDX-License-Identifier: Apache-2.0
"""No two tests open a page on the same port.

The GPU tiers run sharded, a module per process across the cards, so two modules
that claim one port are two pages that cannot both come up when the shards meet.
``tests.test_cut_tofu`` took ``ALL_PORT = 9843``, which was already
``tests.test_cut_tofu_scored``'s ``PORT``. There is no ledger beside the tests:
this reads every claim -- a ``PORT``-named constant, at module or class level, a
number or a tuple of them, and a ``port=`` given as a number -- off the source.
"""

from __future__ import annotations

import ast
import re
import unittest
from collections import defaultdict
from pathlib import Path

TIER = "fast"
"""The tests' source, read by AST: nothing imported."""

TESTS = Path(__file__).resolve().parent


def claims() -> dict[int, list[str]]:
    """Each port a test module claims, and where."""
    found: dict[int, list[str]] = defaultdict(list)
    for path in sorted(TESTS.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Assign):
                named = [t.id for t in node.targets if isinstance(t, ast.Name) and re.search(r"(^|_)PORTS?$", t.id)]
                values = node.value.elts if isinstance(node.value, (ast.Tuple, ast.List)) else [node.value]
            elif isinstance(node, ast.keyword) and node.arg == "port":
                named, values = ["port="], [node.value]
            else:
                continue
            for name in named:
                for value in values:
                    if isinstance(value, ast.Constant) and isinstance(value.value, int) and value.value:
                        found[value.value].append(f"{path.relative_to(TESTS)}:{node.lineno} {name}")
    return found


class TestEveryPortIsClaimedOnce(unittest.TestCase):
    def test_no_two_claims_share_a_port(self) -> None:
        shared = {port: where for port, where in claims().items() if len(where) > 1}
        self.assertEqual({}, shared)

    def test_the_cutting_pages_are_found(self) -> None:
        """The reader finds what it is for: the cutting tests' four pages."""
        found = claims()
        self.assertTrue(all(found.get(port) for port in (9841, 9842, 9843, 9844)), "the scan missed a cutting page")


if __name__ == "__main__":
    unittest.main()
