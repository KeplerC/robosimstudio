# SPDX-License-Identifier: Apache-2.0
"""Genesis's SPH bridge on its own: where spilled water rests, and what an empty mirror reads while it turns.

``tests/test_pour_water.py`` is the bridge in the example's scene; these are two
things that scene never does. Water on a void cell has nothing under it but the
ground plane, which the bridge does not build as a body; and the cup is boxes,
whose centre of mass is their origin, so a mesh mirror turned off its centre is
never harvested there.

Needs the ``[genesis]`` extra, which lives in a virtual environment of its own
(``pyproject.toml``); it skips without it, as ``tests/test_genesis.py`` does.
"""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

TIER = "medium"
"""One GPU, the [genesis] extra: a 64-particle scene, and two bridges built by hand."""


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


HAS_GENESIS = _importable("genesis")


@unittest.skipUnless(HAS_GENESIS, "the [genesis] extra is not installed")
class TestSpilledWaterRestsOnTheGround(unittest.TestCase):
    """A block of water dropped 30 mm onto a void cell's ground plane lies on it, not 218 mm under it."""

    def test_the_water_lies_on_the_plane(self) -> None:
        import robosimstudio as rss

        scene = rss.Scene(cell="void", physics=rss.WATER_60HZ)
        water = scene.add(rss.water(name="water", size=(0.016, 0.016, 0.016), at=(0.0, 0.0, 0.03), settle_frames=0))
        sim = scene.build()
        sim.run(90).sync()
        z = np.asarray(water.particle_q, dtype=np.float64)[:, 2]
        self.assertTrue(sim.state_is_finite())
        self.assertGreaterEqual(float(z.min()), 0.0, f"water stands {1e3 * z.min():.1f} mm, under the ground")
        self.assertLess(float(z.max()), 0.02, f"water stands {1e3 * z.max():.1f} mm, still falling")


@unittest.skipUnless(HAS_GENESIS, "the [genesis] extra is not installed")
class TestAnEmptyMirrorTurnedOffItsCentreFeelsNothing(unittest.TestCase):
    """A 50 mm cube mesh whose centre is 50 mm off its body's origin, turned about the origin at 1 rad/s with the
    water 0.3 m away: its harvest is the water's, none. Read at the origin it was ``m w^2 r``, 6.2 N for its 125 kg."""

    def test_its_harvest_is_zero(self) -> None:
        import newton
        import trimesh
        import warp as wp

        from robosimstudio.core.math import quat_from_axis_angle, quat_multiply
        from robosimstudio.engines.genesis.sph import SphBridge

        cube = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
        cube.apply_translation((0.05, 0.0, 0.0))
        builder = newton.ModelBuilder()
        lattice = np.stack(np.meshgrid(*[np.arange(2) * 0.004] * 3, indexing="ij"), -1).reshape(-1, 3)
        builder.add_particles(
            pos=[tuple(p) for p in lattice + np.array([0.3, 0.3, 0.05])],
            vel=[(0.0, 0.0, 0.0)] * 8,
            mass=[6.4e-5] * 8,
            radius=[0.002] * 8,
        )
        start = quat_from_axis_angle(np.array([1.0, 0.3, 0.0]) / np.linalg.norm([1.0, 0.3, 0.0]), 0.4)
        origin = np.array([0.0, 0.0, 0.5])
        body = builder.add_body(xform=wp.transform(wp.vec3(*origin), wp.quat(*start)), mass=1.0)
        builder.add_shape_mesh(body, mesh=newton.Mesh(cube.vertices.astype(np.float32), cube.faces.astype(np.int32)))
        model = builder.finalize()
        recipe = {
            "island": "spin",
            "control_hz": 60.0,
            "actors": [{"particle_start": 0, "particle_count": 8, "kind": "sph", "name": "water", "params": {}}],
            "mirrors": [{"shape": 0, "body": body, "two_way": True}],
        }
        bridge = SphBridge(model, recipe, substeps=20, precision="32")
        bridge.apply_wrenches = lambda wrench: None

        state, dt = model.state(), 1.0 / 60.0
        axis = np.array([0.2, -0.3, 1.0]) / np.linalg.norm([0.2, -0.3, 1.0])
        quat, forces = start, []
        for _ in range(30):
            quat = quat_multiply(quat_from_axis_angle(axis, dt), quat)
            state.body_q.assign(np.asarray([[*origin, *quat]], dtype=np.float32))
            bridge.tick(state, dt)
            forces.append(float(np.linalg.norm(bridge.last_force[body])))
        self.assertLess(max(forces[1:]), 0.05, f"an empty mirror turning reads {max(forces[1:]):.3f} N")


def _water(builder, centre: np.ndarray, n: int = 4, spacing: float = 0.004) -> int:
    """An ``n``-cubed block of particles at one spacing about ``centre``; its count."""
    lattice = np.stack(np.meshgrid(*[(np.arange(n) - 0.5 * (n - 1)) * spacing] * 3, indexing="ij"), -1).reshape(-1, 3)
    count = len(lattice)
    builder.add_particles(
        pos=[tuple(p) for p in lattice + centre],
        vel=[(0.0, 0.0, 0.0)] * count,
        mass=[1000.0 * spacing**3] * count,
        radius=[0.5 * spacing] * count,
    )
    return count


@unittest.skipUnless(HAS_GENESIS, "the [genesis] extra is not installed")
class TestWaterOnAMirrorOffItsCentre(unittest.TestCase):
    """Water resting on a 40 mm mesh block whose centre is 50 mm off its body's origin. Genesis's midpoint rule for a
    lone free body left the water's push out of its motion, so it read 0.0000 N."""

    @classmethod
    def setUpClass(cls) -> None:
        import newton
        import trimesh
        import warp as wp

        from robosimstudio.engines.genesis.sph import SphBridge

        block = trimesh.creation.box(extents=(0.06, 0.06, 0.04))
        block.apply_translation((0.05, 0.0, 0.0))
        cls.origin = origin = np.array([0.0, 0.0, 0.5])
        builder = newton.ModelBuilder()
        count = _water(builder, origin + np.array([0.05, 0.0, 0.02 + 0.008]))
        body = builder.add_body(xform=wp.transform(wp.vec3(*origin), wp.quat_identity()), mass=1.0)
        builder.add_shape_mesh(body, mesh=newton.Mesh(block.vertices.astype(np.float32), block.faces.astype(np.int32)))
        cls.model = model = builder.finalize()
        recipe = {
            "island": "block",
            "control_hz": 60.0,
            "actors": [{"particle_start": 0, "particle_count": count, "kind": "sph", "name": "water", "params": {}}],
            "mirrors": [{"shape": 0, "body": body, "two_way": True}],
        }
        bridge = SphBridge(model, recipe, substeps=20, precision="32")
        wrenches: list[np.ndarray] = []
        bridge.apply_wrenches = lambda wrench: wrenches.append(np.array(wrench[body], dtype=np.float64))
        state, cls.carried = model.state(), []
        for _ in range(40):
            bridge.tick(state, 1.0 / 60.0)
            cls.carried.append(-float(bridge.last_force[body][2]))
        cls.body, cls.wrench = body, wrenches[-1]
        cls.weight = count * 1000.0 * 0.004**3 * 9.81

    def test_it_carries_the_water(self) -> None:
        carried = float(np.mean(self.carried[10:]))
        self.assertGreater(carried, 0.6 * self.weight, f"the block carries {carried:.4f} N of {self.weight:.4f} N")


if __name__ == "__main__":
    unittest.main()
