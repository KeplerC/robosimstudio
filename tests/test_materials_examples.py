# SPDX-License-Identifier: Apache-2.0
"""Two material examples held to what they say: the chain damps its own links only, and the
cable's capsules are spaced above twice their radius.

No GPU: the chain's kernel runs on Warp's CPU device and the cable is read off its source.
"""

from __future__ import annotations

import ast
import unittest
from types import SimpleNamespace

import numpy as np

from tests.fixtures.examples import EXAMPLES, _load

TIER = "fast"
"""Warp on its CPU device and a source file: no GPU."""


class TestTheChainDampsOnlyItsLinks(unittest.TestCase):
    """``04_extending/01_new_kind``: the drag hook belongs to the chain, not the robot beside it."""

    def test_other_bodies_keep_their_velocity(self) -> None:
        import warp as wp

        module = _load("04_extending/01_new_kind.py")
        qd = np.ones((6, 6), dtype=np.float32)
        state = SimpleNamespace(body_qd=wp.array(qd, dtype=wp.spatial_vector, device="cpu"))
        chain = SimpleNamespace(bodies=[2, 4], spec=SimpleNamespace(damping=0.5))
        module.MyChain._drag(chain, state, 0.001)
        got = state.body_qd.numpy()
        np.testing.assert_allclose(got[[2, 4]], 0.5, err_msg="the chain's links are damped")
        np.testing.assert_allclose(got[[0, 1, 3, 5]], 1.0, err_msg="a body that is not the chain's was damped")


class TestTheCableKeepsItsOwnSpacingRule(unittest.TestCase):
    """``03_materials/00_rope``: "their spacing must stay above twice the radius"."""

    def test_spacing_is_above_twice_the_radius(self) -> None:
        tree = ast.parse((EXAMPLES / "03_materials/00_rope.py").read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        values = {
            n.targets[0].id: ast.literal_eval(n.value)
            for n in main.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)
        }
        spacing = values["length"] / values["segments"]
        self.assertGreater(spacing, 2 * values["radius"], f"spacing {spacing * 1e3:.2f} mm")


if __name__ == "__main__":
    unittest.main()
