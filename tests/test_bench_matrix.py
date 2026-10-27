# SPDX-License-Identifier: Apache-2.0
"""The pairing matrix: a robot's own hand admitted however it is spelled, and found
as the declared binding."""

import unittest
from unittest import mock

from robosimstudio.robots.registry import get_embodiment, hands_of
from robosimstudio.specs import Binding, TaskRequirements, TaskSpec
from robosimstudio_bench.taskbuild.matrix import admits, declared_key

TIER = "fast"
"""Task specs and the robot registry: no scene, no GPU."""


class TestAdmitsTheRobotsOwnHand(unittest.TestCase):
    def test_naming_the_own_hand_is_the_native_pairing(self):
        task = TaskSpec(name="t", scene="bench_0728", embodiment="panda", requires=TaskRequirements(monolithic=True))
        (own,) = hands_of(get_embodiment("panda"))
        self.assertEqual(admits(task, "bench_0728", "panda", ""), (True, ""))
        self.assertEqual(admits(task, "bench_0728", "panda", own), (True, ""))

    def test_another_pairing_is_still_refused(self):
        task = TaskSpec(name="t", scene="bench_0728", embodiment="panda", requires=TaskRequirements(monolithic=True))
        ok, why = admits(task, "bench_0728", "panda_robotiq", "")
        self.assertFalse(ok)
        self.assertIn("one import", why)


class TestTheOwnHandFindsTheDeclaredBinding(unittest.TestCase):
    def _task(self):
        native = Binding(scene="bench_0728", embodiment="panda", graphs=("pick",))
        return TaskSpec(
            name="t",
            scene="bench_0728",
            embodiment="panda",
            bindings=(native,),
            requires=TaskRequirements(monolithic=True),
        )

    def test_the_own_hand_is_the_declared_raw_key(self):
        task = self._task()
        (own,) = hands_of(get_embodiment("panda"))
        self.assertEqual(declared_key(task, "bench_0728", "panda", own), ("bench_0728", "panda", ""))
        self.assertIs(task.binding_for(*declared_key(task, "bench_0728", "panda", own)), task.bindings[0])

    def test_empty_arguments_are_the_tasks_own(self):
        self.assertEqual(declared_key(self._task()), ("bench_0728", "panda", ""))

    def test_an_undeclared_pairing_is_none(self):
        self.assertIsNone(declared_key(self._task(), "bench_0728", "panda_robotiq", ""))
        self.assertIsNone(declared_key(self._task(), "bench_0728", "no_such_robot", ""))

    def test_the_fleet_base_takes_the_own_hand(self):
        from robosimstudio.specs import replace
        from robosimstudio_bench.run import fleet as run_fleet

        base = self._task()
        (own,) = hands_of(get_embodiment("panda"))
        instance = replace(base, name="instance", gripper=own, fleet_base="base")
        with mock.patch("robosimstudio_bench.taskbuild.registry.get_task", return_value=base):
            built = run_fleet.fleet_build_task(instance)
        self.assertEqual((built.scene, built.embodiment), ("bench_0728", "panda"))
        self.assertIs(built.binding_for(), built.bindings[0])


if __name__ == "__main__":
    unittest.main()
