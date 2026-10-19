# SPDX-License-Identifier: Apache-2.0
"""The SuperDex backend's solver options by name: the engine's enum member, or a refusal naming the choices.

``integration``, ``linear_solver`` and ``preconditioner`` are spelled as the
engine's enums name them, in any case. A misspelling used to surface as a bare
``AttributeError`` from inside the bridge's build; it is refused with the names
there are, before the engine's parameters are touched. A stand-in engine, so
this runs with no GPU and no ``[superdex]`` wheel; the wheel's own enums are
checked too when it is installed.
"""

from __future__ import annotations

import enum
import importlib.util
import unittest
from types import SimpleNamespace

TIER = "fast"
"""A stand-in engine and scene. No GPU, no scene build."""

HAS_SUPERDEX = importlib.util.find_spec("superdex") is not None


class _Scene:
    """The engine scene's solver parameters, and whether they were written back."""

    def __init__(self) -> None:
        self.params = SimpleNamespace(
            integration_method=None, linear_solver=SimpleNamespace(), non_linear_solver=SimpleNamespace()
        )
        self.written = False

    def get_solver_params(self):
        return self.params

    def set_solver_params(self, params) -> None:
        self.written = True


def _bridge():
    """Enough of a bridge for ``_tune_solver``: the engine's three enums."""
    return SimpleNamespace(
        physics=SimpleNamespace(
            IntegrationMethod=enum.Enum("IntegrationMethod", "BACKWARD_EULER BDF2 COUNT"),
            LinearSolverType=enum.Enum("LinearSolverType", "CG MINRES COUNT"),
            PreconditionerType=enum.Enum("PreconditionerType", "BLOCK_JACOBI PER_ACTOR COUNT"),
        )
    )


class TestASolverOptionIsAnEngineName(unittest.TestCase):
    def test_a_name_in_any_case_is_the_engines_member(self) -> None:
        from robosimstudio.engines.superdex.bridge import SuperdexBridge

        bridge, scene = _bridge(), _Scene()
        SuperdexBridge._tune_solver(bridge, scene, "cg", "Block_Jacobi", None, None, "bdf2")
        self.assertIs(bridge.physics.IntegrationMethod.BDF2, scene.params.integration_method)
        self.assertIs(bridge.physics.LinearSolverType.CG, scene.params.linear_solver.solver_type)
        self.assertIs(bridge.physics.PreconditionerType.BLOCK_JACOBI, scene.params.linear_solver.preconditioner_type)
        self.assertTrue(scene.written)

    def test_an_unknown_name_is_refused_with_the_names_there_are(self) -> None:
        from robosimstudio.engines.superdex.bridge import SuperdexBridge

        for args, option, names in (
            ((None, None, None, None, "bdf9"), "integration", "backward_euler, bdf2"),
            (("cholesky", None, None, None, None), "linear_solver", "cg, minres"),
            ((None, "ilu", None, None, None), "preconditioner", "block_jacobi, per_actor"),
        ):
            with self.subTest(option=option):
                scene = _Scene()
                with self.assertRaises(ValueError) as caught:
                    SuperdexBridge._tune_solver(_bridge(), scene, *args)
                self.assertIn(option, str(caught.exception))
                self.assertTrue(str(caught.exception).endswith(names), caught.exception)
                self.assertFalse(scene.written, "a refused option leaves the engine's parameters alone")

    @unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed")
    def test_the_wheels_integrators_are_named_without_their_sentinel(self) -> None:
        import superdex.physics as physics

        from robosimstudio.engines.superdex.bridge import _named

        self.assertIs(physics.IntegrationMethod.BDF2, _named(physics.IntegrationMethod, "bdf2", "integration"))
        with self.assertRaisesRegex(ValueError, "backward_euler") as caught:
            _named(physics.IntegrationMethod, "count", "integration")
        self.assertNotIn("count,", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
