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
"""One GPU, the [genesis] extra: a 64-particle scene, and a bridge built by hand."""


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


if __name__ == "__main__":
    unittest.main()
