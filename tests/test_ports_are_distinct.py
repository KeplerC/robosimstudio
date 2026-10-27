# SPDX-License-Identifier: Apache-2.0
"""No two tests open a page on the same port.

The GPU tiers run sharded, a module per process across the cards, so two modules
that claim one port are two pages that cannot both come up when the shards meet.
``tests.test_cut_tofu`` took ``ALL_PORT = 9843``, which was already
``tests.test_cut_tofu_scored``'s ``PORT``. A page that asks for port 0 gets any
free one and claims nothing, which is what a test should do; for one that still
names a number there is no ledger beside the tests: this reads every claim -- a
``PORT``-named constant, at module or class level, a number or a tuple of them, a
``port=`` given as a number, and one derived from such a constant by adding a
number to it (``PORT + 1``) or by counting it up (``Cls.PORT += 1``) -- off the
source.

What it cannot see is how far a counter runs: that is the number of pages the
module opens at run time, so the scan claims the counter's first port alone. And
a derived port is read only where its constant has one value in its module;
where it has several, the scan raises rather than guess. The reader itself is
checked on trees of its own, so what it finds does not depend on which modules
still name a number.
"""

from __future__ import annotations

import ast
import re
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

TIER = "fast"
"""The tests' source, read by AST: nothing imported."""

TESTS = Path(__file__).resolve().parent


def _port_name(node: ast.AST) -> str:
    """The name of a ``PORT``-named variable or attribute, ``PORT``/``Cls.PORT``/``self.PORT``, or ``""``."""
    name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""
    return name if re.search(r"(^|_)PORTS?$", name) else ""


def _number(node: ast.AST) -> int:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, int) else 0


def _derived(node: ast.AST, stated: dict[str, set[int]], where: object) -> tuple[str, int] | None:
    """``PORT + k`` or ``k + PORT`` (a page handed its own), ``PORT += k`` (a counter's first page), else ``None``."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        pairs = ((node.left, node.right), (node.right, node.left))
        name, step = next(((_port_name(a), _number(b)) for a, b in pairs if _port_name(a) and _number(b)), ("", 0))
        label = f"{name} + {step}"
    elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
        name, step = _port_name(node.target), _number(node.value)
        label = f"{name} += {step}"
    else:
        return None
    if not (name and step):
        return None
    if len(stated.get(name, ())) != 1:
        raise ValueError(f"{where}:{node.lineno} {label}: {name} is {sorted(stated.get(name, ()))} here; which one?")
    (base,) = stated[name]
    return label, base + step


def claims(root: Path = TESTS) -> dict[int, list[str]]:
    """Each port a test module under ``root`` claims, and where."""
    found: dict[int, list[str]] = defaultdict(list)
    for path in sorted(root.rglob("*.py")):
        tree, where = ast.parse(path.read_text()), path.relative_to(root)
        # Each PORT-named constant's numbers in this module, for what is derived from it.
        stated: dict[str, set[int]] = defaultdict(set)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and _number(node.value):
                for target in node.targets:
                    if _port_name(target):
                        stated[_port_name(target)].add(_number(node.value))
        for node in ast.walk(tree):
            derived = _derived(node, stated, where)
            if derived is not None:
                label, port = derived
                found[port].append(f"{where}:{node.lineno} {label}")
                continue
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
                        found[value.value].append(f"{where}:{node.lineno} {name}")
    return found


class TestEveryPortIsClaimedOnce(unittest.TestCase):
    def test_no_two_claims_share_a_port(self) -> None:
        shared = {port: where for port, where in claims().items() if len(where) > 1}
        self.assertEqual({}, shared)

    def test_every_way_of_stating_a_port_is_found(self) -> None:
        """The reader finds what it is for, on a tree of its own: a module's ``PORT``, a class's tuple of
        ``PORTS``, and a ``port=`` keyword -- and a port of 0, which asks for any free one, claims nothing."""
        found = self._scan(
            a="PORT = 9841\nANY_PORT = 0\n",
            b="class T:\n    PAGE_PORTS = (9842, 9843)\n\nopen_page(port=9844)\nopen_page(port=0)\n",
        )
        self.assertEqual(
            {9841: ["a.py:1 PORT"], 9842: ["b.py:2 PAGE_PORTS"], 9843: ["b.py:2 PAGE_PORTS"], 9844: ["b.py:4 port="]},
            found,
        )

    def test_the_derived_ports_are_found(self) -> None:
        """``PORT + 1`` handed to a page, and a class counter's ``PORT += 1``: neither is an ``Assign`` or a ``port=``."""
        found = self._scan(
            sliders="PORT = 9801\n_open('a', PORT + 1)\n_open('b', 2 + PORT)\n",
            examples="class T:\n    PORT = 8981\n\n    def page(self):\n        T.PORT += 1\n",
        )
        self.assertEqual(["sliders.py:2 PORT + 1"], found[9802])
        self.assertEqual(["sliders.py:3 PORT + 2"], found[9803])
        self.assertIn("PORT += 1", " ".join(found.get(8982, [])), "a counter's first page")

    def test_two_modules_on_one_port_are_both_named(self) -> None:
        found = self._scan(one="PORT = 9843\n", two="ALL_PORT = 9843\n")
        self.assertEqual(["one.py:1 PORT", "two.py:1 ALL_PORT"], found[9843])

    def _scan(self, **modules: str) -> dict[int, list[str]]:
        with tempfile.TemporaryDirectory() as root:
            for name, source in modules.items():
                (Path(root) / f"{name}.py").write_text(source)
            return dict(claims(Path(root)))

    def test_a_derived_port_on_a_claimed_one_is_caught(self) -> None:
        """What the reader is for, on a module of its own: ``PORT + 1`` landing on another page's ``PORT``."""
        stated = {"PORT": {9801}}
        node = ast.parse("_open('a.py', PORT + 1)").body[0].value.args[1]
        self.assertEqual(("PORT + 1", 9802), _derived(node, stated, "a.py"))
        with self.assertRaisesRegex(ValueError, "which one"):
            _derived(node, {"PORT": {9801, 9811}}, "a.py")


if __name__ == "__main__":
    unittest.main()
