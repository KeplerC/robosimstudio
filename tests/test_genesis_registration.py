# SPDX-License-Identifier: Apache-2.0
"""The Genesis backend's registration, with Genesis not installed.

Everything here runs with the ``[genesis]`` extra absent, on purpose --- it is
the half of the story ``test_genesis.py`` cannot tell, and the half that matters
most for a checkout that will never build a Genesis island.

The claim checked here is the extension seam's: registering a backend imports
nothing, an alias resolves from a cold process, the engine is asked for only
when a scene actually stages one --- not at import, and never as a silent
fallback --- and the solver is a host solver that writes ``open`` and nothing
else.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest

TIER = "fast"
"""No GPU, no engine, no scene."""


class TestRegistrationImportsNothing(unittest.TestCase):
    def test_the_alias_resolves_and_is_a_host_engine(self):
        from robosimstudio.specs.physics import get_backend

        info = get_backend("genesis")
        self.assertTrue(info.host_engine)
        self.assertEqual(info.supports, ("particles",))
        self.assertEqual(info.cls, "robosimstudio.engines.genesis.solver.SolverGenesis")

    def test_it_resolves_from_a_cold_process(self):
        """``GENERIC`` may name an alias nothing has imported; ``get_backend``
        scans the engine bundles once and retries before giving up."""
        code = "from robosimstudio.specs.physics import get_backend\nassert get_backend('genesis').host_engine\n"
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    @unittest.skipIf(importlib.util.find_spec("genesis") is not None, "genesis is installed here")
    def test_a_missing_wheel_is_refused_by_name(self):
        from robosimstudio.engines.genesis.runtime import ensure_initialized

        with self.assertRaises(ImportError) as caught:
            ensure_initialized()
        message = str(caught.exception)
        self.assertIn("[genesis]", message, "the refusal names the extra to install")
        self.assertIn("genesis-world", message)


class TestTheSolverIsAHostSolver(unittest.TestCase):
    def test_it_inherits_the_shared_contract(self):
        from robosimstudio.engines.genesis.solver import SolverGenesis
        from robosimstudio.engines.host import SolverHost

        self.assertTrue(issubclass(SolverGenesis, SolverHost))
        self.assertTrue(SolverGenesis.host_engine)
        for name in (
            "step",
            "notify_model_changed",
            "update_contacts",
            "reset",
            "attach_pipeline",
            "capture_rest",
            "host_tick",
        ):
            self.assertTrue(hasattr(SolverGenesis, name), f"the composer calls {name}")

    def test_a_bridge_states_no_default_the_solver_does_not(self):
        """The solver passes every option; a default of a bridge's own that differs from its own is one that only a
        bridge built by hand ever sees."""
        import inspect

        from robosimstudio.engines.genesis.bridge import GenesisBridge
        from robosimstudio.engines.genesis.solver import SolverGenesis
        from robosimstudio.engines.genesis.sph import SphBridge

        solver = inspect.signature(SolverGenesis.open).parameters
        for bridge in (GenesisBridge, SphBridge):
            for name, param in inspect.signature(bridge.__init__).parameters.items():
                if param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is not inspect.Parameter.empty:
                    with self.subTest(bridge=bridge.__name__, option=name):
                        self.assertEqual(solver[name].default, param.default)

    def test_open_is_the_only_thing_it_writes(self):
        """The point of SolverHost: a second engine costs one method."""
        from robosimstudio.engines.genesis.solver import SolverGenesis

        own = {n for n, v in vars(SolverGenesis).items() if callable(v)}
        self.assertEqual(own, {"open"}, f"SolverGenesis defines more than open(): {sorted(own)}")


if __name__ == "__main__":
    unittest.main()
