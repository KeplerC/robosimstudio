# SPDX-License-Identifier: Apache-2.0
"""``FleetEnv``'s refusals, made before anything is built."""

import unittest

from robosimstudio.specs import TaskSpec

TIER = "fast"
"""Refused before a model is built: no scene, no GPU."""


class TestABatchedEnvRefusesAnUndeclaredPairing(unittest.TestCase):
    def test_scene_override_on_a_task_without_bindings(self):
        from robosimstudio_bench.envs.fleet import FleetEnv

        spec = TaskSpec(name="t", scene="bench_a", embodiment="franka")
        with self.assertRaises(KeyError) as caught:
            FleetEnv(spec, num_worlds=2, scene="bench_b", device="cpu")
        self.assertIn("declares no binding", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
