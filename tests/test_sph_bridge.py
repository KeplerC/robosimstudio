# SPDX-License-Identifier: Apache-2.0
"""Genesis's SPH bridge on its own: where spilled water rests, what a mirror off its centre reads, and the box's walls.

``tests/test_pour_water.py`` is the bridge in the example's scene; these are
things that scene never does. Water on a void cell has nothing under it but the
ground plane, which the bridge does not build as a body; the cup is boxes, whose
centre of mass is their origin, so a mesh mirror off its centre is never harvested
there; its water never reaches the solver's box; and it holds water, never the oil
preset, whose cohesion takes more substeps than the arrangement states.

Needs the ``[genesis]`` extra, which lives in a virtual environment of its own
(``pyproject.toml``); it skips without it, as ``tests/test_genesis.py`` does.
"""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

TIER = "medium"
"""One GPU, the [genesis] extra: a 64-particle scene, four bridges built by hand, and a cup of oil."""


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
        bridge.restore_rest()
        self.assertEqual({}, bridge.last_force, "a reset leaves the last reaction for a reader")


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
    lone free body left the water's push out of its motion, so it read 0.0000 N; and the reaction, read at the
    block's centre, was applied at the origin, a couple of ``(origin - centre) x F`` that no water made."""

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

    def test_the_torque_is_the_force_at_its_centre(self) -> None:
        force, torque = self.wrench[:3], self.wrench[3:]
        self.assertGreater(-force[2], 0.2 * self.weight, "a force to place")
        com = self.origin + self.model.body_com.numpy()[self.body].astype(np.float64)
        centre = self.origin + np.array([0.05, 0.0, 0.0])
        np.testing.assert_allclose(torque, np.cross(centre - com, force), atol=1.0e-3 * abs(force[2]))


@unittest.skipUnless(HAS_GENESIS, "the [genesis] extra is not installed")
class TestWaterOnACapsuleMirror(unittest.TestCase):
    """Water walled in on a capsule lying along x. Its tessellated centre comes back about 1e-18 m off its origin,
    and Genesis's midpoint rule tests the centre exactly: off the bridge's zero-gain servo it read 0.0 N of the
    water's 0.040 N."""

    def test_it_carries_the_water(self) -> None:
        import newton
        import warp as wp

        from robosimstudio.engines.genesis.sph import SphBridge

        origin = np.array([0.0, 0.0, 0.5])
        builder = newton.ModelBuilder()
        count = _water(builder, origin + np.array([0.0, 0.0, 0.03]))
        body = builder.add_body(xform=wp.transform(wp.vec3(*origin), wp.quat_identity()), mass=1.0)
        along_x = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2)
        builder.add_shape_capsule(
            body, xform=wp.transform(wp.vec3(0.0, 0.0, -0.08), along_x), radius=0.1, half_height=0.05
        )
        walls = []
        for x, y, hx, hy in ((0.014, 0.0, 0.004, 0.018), (-0.014, 0.0, 0.004, 0.018), (0.0, 0.014, 0.01, 0.004)):
            for sign in (1.0, -1.0) if y else (1.0,):
                at = wp.vec3(*(origin + np.array([x, sign * y, 0.043])))
                walls.append(
                    builder.add_shape_box(-1, xform=wp.transform(at, wp.quat_identity()), hx=hx, hy=hy, hz=0.027)
                )
        model = builder.finalize()
        recipe = {
            "island": "capsule",
            "control_hz": 60.0,
            "actors": [{"particle_start": 0, "particle_count": count, "kind": "sph", "name": "water", "params": {}}],
            "mirrors": [{"shape": 0, "body": body, "two_way": True}] + [{"shape": s, "body": -1} for s in walls],
        }
        bridge = SphBridge(model, recipe, substeps=20, precision="32")
        bridge.apply_wrenches = lambda wrench: None
        centre = np.asarray(bridge.mirrors[0]["entity"].links[0].inertial_pos, dtype=np.float64)
        self.assertTrue(centre.any() and np.linalg.norm(centre) < 1.0e-9, f"the capsule's centre is {centre}")
        state, carried = model.state(), []
        for _ in range(60):
            bridge.tick(state, 1.0 / 60.0)
            carried.append(-float(bridge.last_force[body][2]))
        weight = count * 1000.0 * 0.004**3 * 9.81
        self.assertAlmostEqual(1.0, float(np.mean(carried[20:])) / weight, delta=0.15, msg="the capsule carries it")


@unittest.skipUnless(HAS_GENESIS, "the [genesis] extra is not installed")
class TestWaterAtTheSolversWallIsSaid(unittest.TestCase):
    """Genesis holds a particle at its box's face, so water carried past ``sph_reach`` stops at a wall nobody drew:
    the bridge says so, once, naming the island and the reach."""

    def test_a_box_the_water_outgrows_is_warned_once(self) -> None:
        import newton

        from robosimstudio.engines.genesis import sph

        builder = newton.ModelBuilder()
        count = _water(builder, np.array([0.0, 0.0, 0.05]))
        model = builder.finalize()
        recipe = {
            "island": "tight",
            "control_hz": 60.0,
            "sph_reach": 0.004,
            "sph_reach_down": 0.01,
            "actors": [{"particle_start": 0, "particle_count": count, "kind": "sph", "name": "water", "params": {}}],
            "mirrors": [],
        }
        bridge = sph.SphBridge(model, recipe, substeps=20, precision="32")
        state, dt = model.state(), 1.0 / 60.0
        with self.assertLogs(sph.log, "WARNING") as logs:
            for _ in range(60):
                bridge.tick(state, dt)
        said = [line for line in logs.output if "sph_reach" in line]
        self.assertEqual(1, len(said), logs.output)
        self.assertIn("'tight'", said[0])


@unittest.skipUnless(HAS_GENESIS, "the [genesis] extra is not installed")
class TestOilRestsInTheCupAtItsOwnSubsteps(unittest.TestCase):
    """The oil preset in the pour example's cup: refused at the water's 20 substeps, held still at the 35 it states.

    At 20 all of its 1,728 particles left the cup and at 27, 353 did (``oil_rest.py``): its cohesion, three times
    water's, sets the substep."""

    WALL = 0.008
    INNER = (0.05, 0.05, 0.07)

    def _scene(self, physics):
        import robosimstudio as rss
        from robosimstudio_materials.liquid import cup

        scene = rss.Scene(cell="bench_075", physics=physics)
        scene.add(cup(name="cup", inner=self.INNER, wall=self.WALL))
        oil = scene.add(rss.water(name="oil", at=(0.0, 0.0, self.WALL + 0.001), material="oil", settle_frames=0))
        return scene, oil

    def test_the_waters_arrangement_refuses_it_by_name(self) -> None:
        import robosimstudio as rss

        scene, _ = self._scene(rss.WATER_60HZ)
        with self.assertRaisesRegex(ValueError, r"'oil'.*surface tension.*substeps=35"):
            scene.build()

    def test_its_own_substeps_hold_it_in_the_cup(self) -> None:
        import robosimstudio as rss

        physics = rss.WATER_60HZ.with_backend("water", rss.backend("genesis", substeps=35, precision="32"))
        scene, oil = self._scene(physics)
        sim = scene.build()
        sim.run(90).sync()
        bridge, body = sim.scene.program.solver.solver("genesis").bridge, int(sim["cup"].bodies[0])
        carried = []
        for _ in range(30):
            sim.run(1).sync()
            carried.append(-float(bridge.last_force[body][2]))
        bench = np.asarray(rss.get_cell("bench_075").surface_frame[:3], dtype=np.float64)
        lo = bench + np.array([-0.5 * self.INNER[0], -0.5 * self.INNER[1], self.WALL]) - 0.002
        hi = bench + np.array([0.5 * self.INNER[0], 0.5 * self.INNER[1], self.WALL + self.INNER[2]]) + 0.002
        q = np.asarray(oil.particle_q, dtype=np.float64)
        inside = int(np.all((q >= lo) & (q <= hi), axis=1).sum())
        weight = oil.spec.mass * 9.81
        self.assertTrue(sim.state_is_finite())
        self.assertEqual(len(q), inside, f"{len(q) - inside} of {len(q)} particles left the cup")
        self.assertAlmostEqual(1.0, float(np.mean(carried)) / weight, delta=0.1, msg="the cup carries the oil")


if __name__ == "__main__":
    unittest.main()
