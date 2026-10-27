# SPDX-License-Identifier: Apache-2.0
"""The connector: the privilege boundary of its registry, and an approach that stops on
a failed pre-approach."""

from __future__ import annotations

import unittest
from unittest import mock

TIER = "fast"
"""A fake registry and a stub servo: no GPU, no scene."""


class _FakeRegistry:
    def __init__(self) -> None:
        self.names: list[str] = []

    def register_callable(self, name, fn, **kwargs):
        self.names.append(name)

    def register_rpc(self, name, client, **kwargs):
        self.names.append(name)

    def discover_pending(self, *a, **k):
        raise AssertionError("reached the real registry")

    def __contains__(self, name):
        return name in self.names

    def __len__(self):
        return len(self.names)


class TestNonPrivilegedRegistry(unittest.TestCase):
    def setUp(self) -> None:
        from robosimstudio_policy_agent.connector import NonPrivilegedRegistry

        self.inner = _FakeRegistry()
        self.reg = NonPrivilegedRegistry(self.inner)

    def test_register_rpc_drops_sim_and_privileged(self) -> None:
        self.reg.register_rpc("sim.get_object_pose", object(), summary="")
        self.reg.register_rpc("look", object(), tags=("privileged",))
        self.reg.register_rpc("robot.home", object(), tags=("robot",))
        self.assertEqual(self.inner.names, ["robot.home"])
        self.assertEqual(self.reg.dropped, ["sim.get_object_pose", "look"])

    def test_other_registration_doors_are_refused(self) -> None:
        for attr in (
            "discover_pending",
            "discover",
            "_discover_python",
            "_register_python",
            "register",
            "_tools",
            "python_adapter",
            "rpc_adapter",
        ):
            with self.assertRaises(AttributeError, msg=attr):
                getattr(self.reg, attr)

    def test_readers_pass_through(self) -> None:
        self.inner.get = lambda name: name
        self.assertEqual(self.reg.get("robot.home"), "robot.home")
        self.reg.register_callable("robot.home", lambda: None)
        self.assertEqual(len(self.reg), 1)

    def test_contains_delegates(self) -> None:
        self.reg.register_callable("robot.home", lambda: None)
        self.assertIn("robot.home", self.reg)
        self.assertNotIn("sim.x", self.reg)


class TestApproachFailureStops(unittest.TestCase):
    def test_contact_on_the_approach_is_reported(self) -> None:
        try:
            import robosimstudio_policy_agent.connector as C
        except ImportError as exc:  # pragma: no cover
            self.skipTest(str(exc))

        class Servo:
            def __init__(self) -> None:
                self.calls: list[float] = []

            def move(self, pose, arm_id=0, **kw):
                self.calls.append(pose["position"]["z"])
                if len(self.calls) == 1:
                    return {"outcome": "contact", "remaining_distance": 0.08, "steps": 40}
                return {"outcome": "reached", "remaining_distance": 0.001, "steps": 90}

        class Base:
            def __init__(self, *a, **k) -> None:
                pass

            def go_to_pose(self, *a, **k):
                return "base"

        servo = Servo()
        cfg = mock.MagicMock()
        cfg.default_cameras = []
        with mock.patch.object(C, "_connector_base", lambda: Base):
            try:
                conn = C.make_connector(
                    mock.MagicMock(),
                    config=cfg,
                    facade=mock.MagicMock(),
                    ik=mock.MagicMock(),
                    servo=servo,
                    bus=mock.MagicMock(),
                )
            except ImportError as exc:  # pragma: no cover
                self.skipTest(str(exc))
        pose = {"position": {"x": 0.4, "y": 0.0, "z": 0.1}, "rotation": {"x": 1, "y": 0, "z": 0, "w": 0}}
        out = conn.go_to_pose(pose, z_approach=0.1)
        self.assertEqual(len(servo.calls), 1)
        self.assertEqual(out["outcome"], "contact")
        self.assertEqual(out["phase"], "approach")
        self.assertFalse(out["reached"])


if __name__ == "__main__":
    unittest.main()
