# SPDX-License-Identifier: Apache-2.0
"""A scene put together by hand, and finding things in it by name.

``Scene.add`` binds a ref for whatever it accepts and refuses a second of
anything; ``Sim`` answers ``in`` and ``names()`` from the same tables
``__getitem__`` reads, regions and statics among them.
"""

from __future__ import annotations

import types
import unittest
from types import SimpleNamespace

TIER = "fast"
"""CPU only: specs and stand-in handles, no scene build."""


def _region(name: str):
    from robosimstudio.specs import RegionSpec

    return RegionSpec(name=name, frame="", xy_ranges=(-0.1, -0.1, 0.1, 0.1))


class TestSceneAddBindsAndRefuses(unittest.TestCase):
    def _scene(self):
        from robosimstudio.sim.scene import Scene

        return Scene(cell="bench_075")

    def test_robot_region_and_camera_refs_are_bound_by_build(self) -> None:
        from robosimstudio.sim.scene import robot
        from robosimstudio.specs import CameraSpec

        scene = self._scene()
        refs = [
            scene.add(robot("panda")),
            scene.add(_region("bin")),
            scene.add(CameraSpec(name="top")),
        ]
        # Sim binds exactly the refs the scene holds, so a ref left out stays on None.
        self.assertEqual([scene._refs[r.name] for r in refs], refs)
        compiled = types.SimpleNamespace(embodiment="the robot")
        refs[0]._scene = compiled
        self.assertEqual(refs[0].entity, "the robot")
        self.assertEqual(refs[2].spec.name, "top")
        with self.assertRaisesRegex(ValueError, "camera"):
            _ = refs[2].pose

    def test_a_second_robot_is_refused(self) -> None:
        from robosimstudio.sim.scene import robot

        scene = self._scene()
        scene.add(robot("panda"))
        with self.assertRaisesRegex(ValueError, "already has a robot"):
            scene.add(robot("ur5e"))

    def test_duplicate_names_are_refused_for_every_kind(self) -> None:
        from robosimstudio.domains.rigid import box
        from robosimstudio.specs import CameraSpec

        scene = self._scene()
        scene.add(CameraSpec(name="top"))
        scene.add(_region("bin"))
        for spec in (CameraSpec(name="top"), _region("bin"), box(name="top")):
            with self.assertRaises(ValueError):
                scene.add(spec)

    def test_remove_takes_out_whatever_add_put_in(self) -> None:
        from robosimstudio.sim.scene import robot
        from robosimstudio.specs import CameraSpec

        scene = self._scene()
        scene.add(robot("panda"))
        scene.add(_region("bin"))
        scene.add(CameraSpec(name="top"))
        for name in ("panda", "bin", "top"):
            scene.remove(name)
        self.assertEqual((scene._robot, scene.regions, scene.cameras, scene._refs), (None, [], [], {}))
        scene.add(robot("ur5e"))  # the slot is free again


class TestNamesAgreeWithGetitem(unittest.TestCase):
    def test_a_region_is_in_the_sim(self) -> None:
        from robosimstudio.sim.sim import Sim

        handles = SimpleNamespace(
            objects={"cube": object()}, deformables={}, statics={"post": object()}, regions={"goal": object()}
        )
        handles.find = lambda name: next(
            (t[name] for t in (handles.objects, handles.deformables, handles.statics, handles.regions) if name in t),
            None,
        )
        sim = Sim.__new__(Sim)
        sim.scene = SimpleNamespace(by_name={}, handles=handles)
        sim.refs = {}
        self.assertIn("goal", sim)
        self.assertIn("post", sim)
        self.assertNotIn("nothing", sim)
        self.assertEqual(["cube", "goal", "post"], sim.names())


class TestTwoWaysToSayWhere(unittest.TestCase):
    def test_at_and_a_binding_mount_together_are_refused(self) -> None:
        import robosimstudio as rss
        from robosimstudio.specs import Binding

        binding = Binding(scene="bench_075", embodiment="panda", mount=(0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0))
        scene = rss.Scene(cell="bench_075", binding=binding)
        scene.add(rss.robot("panda", at=(-0.5, 0.0, 0.0)))
        with self.assertRaisesRegex(ValueError, "both say where the robot stands"):
            _ = scene.request


if __name__ == "__main__":
    unittest.main()
