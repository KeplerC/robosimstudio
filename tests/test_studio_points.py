# SPDX-License-Identifier: Apache-2.0
"""A particle kind is drawn: points or a surface on the studio page, and in the path tracer.

A heap of particles has no triangle, no tube and no order, so until now the
page drew nothing for it and the path tracer logged nothing. Its handle now
says how it is drawn -- a positive ``point_radius``, and optionally a
``surface`` callable that regrows a mesh over the live positions -- and both
renderers read those two fields through :mod:`robosimstudio.core.render.particles`.

No shipped kind is used (the three particle materials are built in
``robosimstudio_materials`` on the same backend), so the heap is a test-local
kind the way ``tests/test_mpm_island`` builds one, with a ``render`` switch that
either leaves the handle as points or hands it a
:class:`~robosimstudio.core.render.particles.ParticleSurfacer`. Held here, each
on a built scene on the GPU:

* the page draws the points as one point cloud of the particle count at the
  particle diameter, rewrites it in place as the heap falls, and keeps drawing
  it under the ``flat`` look;
* with a surface, the page draws a mesh over the heap instead, whose vertex
  count changes between frames and which stays within the heap's extent;
* the path tracer's frame is not blank where the heap is, as points and as a
  surface, in the entity's own colour.
"""

from __future__ import annotations

import time
import unittest
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TIER = "medium"
"""A GPU: three small MPM scenes, two viser servers on spare ports, and the path tracer where it runs."""

RICE = {
    "mpm:young_modulus": 1.0e15,
    "mpm:poisson_ratio": 0.3,
    "mpm:friction": 0.68,
    "mpm:yield_pressure": 1.0e12,
    "mpm:tensile_yield_ratio": 0.0,
}
"""Newton 1.6's rice, as ``tests/test_mpm_island`` and the Phase 0 probes used it."""

DOMAIN = "granular_render_probe"
ENTITY = "heap_render_probe"
COLOR = (0.9, 0.15, 0.1)
"""A red nothing else in a void cell has, so a frame can be asked where the heap is."""


@dataclass(frozen=True, kw_only=True)
class HeapSpec:
    """A block of particles with the rice preset, bottom face centred at ``at`` in the surface frame.

    ``render`` is what the kinds will offer: ``"points"`` leaves the handle as
    points; ``"surface"`` hands it a :class:`ParticleSurfacer` over the radius.
    """

    entity: str = ENTITY
    name: str = "heap"
    size: tuple[float, float, float] = (0.08, 0.08, 0.03)
    spacing: float = 0.003
    density: float = 1500.0
    at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    color: tuple[float, float, float] = COLOR
    render: str = "points"
    sim: Any = field(default_factory=lambda: _requirements())  # noqa: PLW0108 -- the import happens at call time


def _requirements():
    import robosimstudio as rss

    return rss.SimRequirements(domain=DOMAIN, contacts=rss.contacts_with("surface"))


def _register() -> None:
    """The domain and the entity, once per process; a second call is the registries' identical re-registration."""
    import warp as wp

    import robosimstudio as rss
    from robosimstudio.core.render.particles import ParticleSurfacer
    from robosimstudio.domains.capabilities import ParticleMixin
    from robosimstudio.scene.entity import Entity
    from robosimstudio.specs.physics import DOMAINS, get_domain

    get_domain("cloth")  # the fallback layout must exist before a chain names it
    if DOMAIN not in DOMAINS:
        rss.register_domain(
            DOMAIN,
            state=("particles",),
            reads=(DOMAIN, "cloth"),
            role="deformables",
            contacts=("surface",),
            summary="A heap of particles, for the render tests.",
        )

    class EntityHeap(ParticleMixin, Entity):
        domain = DOMAIN

        def __init__(self, spec, *, world: int = 0) -> None:
            super().__init__(spec, world=world)
            self.particles = np.zeros(0, dtype=np.int32)
            self.sets = {}
            self.probe = None
            self.rest = {}

        def build(self, builder, episode) -> None:
            del episode
            from newton.solvers import SolverImplicitMPM

            SolverImplicitMPM.register_custom_attributes(builder)
            spec = self.spec
            placed = self.placed((*spec.at, 0.0, 0.0, 0.0, 1.0))
            dims = [max(1, int(round(s / spec.spacing))) for s in spec.size]
            cell = [s / d for s, d in zip(spec.size, dims, strict=True)]
            lo = (placed[0] - 0.5 * spec.size[0], placed[1] - 0.5 * spec.size[1], placed[2] + 0.5 * cell[2])
            builder.add_particle_grid(
                pos=wp.vec3(*[float(v) for v in lo]),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=dims[0],
                dim_y=dims[1],
                dim_z=dims[2],
                cell_x=cell[0],
                cell_y=cell[1],
                cell_z=cell[2],
                mass=float(np.prod(cell)) * spec.density,
                jitter=0.25 * spec.spacing,
                radius_mean=0.5 * max(cell),
                custom_attributes=dict(RICE),
            )

        def handle(self, model):
            handle = super().handle(model)
            handle.point_radius = 0.5 * float(self.spec.spacing)
            if self.spec.render == "surface":
                handle.surface = ParticleSurfacer(handle.point_radius, device=model.device)
            return handle

        def on_scene_ready(self, pipeline) -> None:
            self.join_settle(pipeline, None)

        def reset(self, pipeline, episode, *, world: int = 0) -> None:
            import newton

            super().reset(pipeline, episode, world=world)
            pipeline.reset_solvers(flags=int(newton.StateFlags.PARTICLE))

        def update_probe(self, state, step=None, *, readback=None):
            if readback is not None and getattr(readback, "particle_q", None) is not None and len(readback.particle_q):
                self.probe.update_arrays(readback.particle_q, readback.particle_qd, step)
            else:
                self.probe.update(state, step)
            return self.probe

    from robosimstudio.scene.registry import ENTITIES

    if ENTITY not in ENTITIES:
        rss.register_entity(ENTITY)(EntityHeap)


def build(render: str = "points", **spec):
    """A heap alone in a void cell, its domain routed to the ``mpm`` backend the way ``GENERIC`` routes ``granular``."""
    import robosimstudio as rss

    _register()
    physics = rss.GENERIC.with_backend(DOMAIN, rss.backend("mpm"))
    scene = rss.Scene(cell="void", name=f"heap_{render}", physics=physics)
    scene.add(HeapSpec(render=render, **spec))
    return scene.build(control_mode="joint_position")


def camera():
    """A close three-quarter view of the heap, small enough that a frame is cheap."""
    import robosimstudio as rss

    return rss.CameraSpec(
        name="probe", width=320, height=240, eye=(0.22, -0.22, 0.16), look_at=(0.0, 0.0, 0.01), far=3.0
    )


def heap_pixels(rgb: np.ndarray) -> int:
    """How many pixels are the heap's red: red well above both green and blue."""
    r, g, b = (rgb[..., i].astype(np.int32) for i in range(3))
    return int(((r > g + 40) & (r > b + 40)).sum())


def _session(sim, port: int):
    from robosimstudio_studio.session import StudioConfig, StudioSession

    return StudioSession(sim, config=StudioConfig(port=port, panels="drive"))


def _rtx_runs() -> bool:
    from robosimstudio.solvers.render import rt_cores, rtx_available

    return rtx_available() and rt_cores()[0]


class TestThePageDrawsThePoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = build("points")
        cls.session = _session(cls.sim, 8975)
        cls.viewer = cls.session.viewer
        cls.key = cls.viewer._qualify("/model/garments/heap/points")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()
        cls.viewer.close()

    def _points(self) -> np.ndarray:
        self.session.draw()
        return np.asarray(self.viewer._garment_nodes[self.key].points, dtype=np.float64)

    def test_the_premise_is_particles_with_no_surface(self) -> None:
        handle = self.sim.handles.deformables["heap"]
        self.assertEqual(0, self.sim.model.tri_count)
        self.assertEqual(0, len(handle.triangles))
        self.assertEqual(0.0, handle.tube_radius)
        self.assertAlmostEqual(0.0015, handle.point_radius)
        self.assertIsNone(handle.surface)

    def test_one_point_cloud_of_the_particle_count_at_the_particle_diameter(self) -> None:
        self.session.reset_env()
        points = self._points()
        self.assertEqual(self.sim.model.particle_count, len(points))
        node = self.viewer._garment_nodes[self.key]
        self.assertAlmostEqual(0.003, float(node.point_size))
        # viser keeps a point cloud at float16 (the fuzz is sent the same way):
        # 1.5e-5 m off within 0.1 m of the origin, a twentieth of a grain.
        np.testing.assert_allclose(points, np.asarray(self.sim.state.particle_q.numpy(), dtype=np.float64), atol=1e-4)
        self.assertNotIn(self.viewer._qualify("/model/garments/heap"), self.viewer._garment_nodes, "no mesh node")
        self.assertNotIn(self.viewer._qualify("/model/garments/heap/surface"), self.viewer._garment_nodes)

    def test_it_follows_the_heap_in_the_one_node(self) -> None:
        self.session.reset_env()
        before = self._points()
        node = self.viewer._garment_nodes[self.key]
        for _ in range(30):
            self.sim.step_teleop()
        after = self._points()
        self.assertIs(node, self.viewer._garment_nodes[self.key], "a frame re-added the cloud instead of moving it")
        self.assertGreater(float(before[:, 2].mean() - after[:, 2].mean()), 0.002, "the drawn heap did not settle")

    def test_the_flat_look_still_draws_it(self) -> None:
        style = self.viewer.garment_style
        try:
            self.viewer.set_garment_style("flat")
            self.assertNotIn(self.key, self.viewer._garment_nodes)
            self.assertEqual(self.sim.model.particle_count, len(self._points()))
        finally:
            self.viewer.set_garment_style(style)


class TestThePageDrawsTheSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = build("surface")
        cls.session = _session(cls.sim, 8976)
        cls.viewer = cls.session.viewer
        cls.key = cls.viewer._qualify("/model/garments/heap/surface")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()
        cls.viewer.close()

    def _surface(self) -> tuple[np.ndarray, np.ndarray]:
        self.session.draw()
        node = self.viewer._garment_nodes[self.key]
        return np.asarray(node.vertices, dtype=np.float64), np.asarray(node.faces, dtype=np.int64)

    def test_a_mesh_over_the_heap_and_no_points(self) -> None:
        """The drawn mesh is the surfacer's own over the live positions, not the particles relabelled.

        Checked against a second extraction here because a first version of
        the page reached for ``GarmentVisual.surface`` -- the sheet method --
        instead of the handle's callable, drew the 7,290 particles as 7,290
        "vertices" over faces made of cast normals, and passed a looser test.
        """
        from robosimstudio.core.render.particles import surface_mesh

        self.session.reset_env()
        vertices, faces = self._surface()
        q = np.asarray(self.sim.state.particle_q.numpy(), dtype=np.float64)
        self.assertGreater(len(faces), 100)
        self.assertNotEqual(len(vertices), len(q), "one vertex per particle is the particles, not a surface")
        self.assertLess(int(faces.max()), len(vertices))
        self.assertTrue((np.sort(faces, axis=1)[:, :-1] != np.sort(faces, axis=1)[:, 1:]).all(), "degenerate faces")
        lo, hi = q.min(axis=0) - 0.006, q.max(axis=0) + 0.006
        self.assertTrue(((vertices >= lo) & (vertices <= hi)).all(), "the surface left the heap's extent")
        own, own_faces, _ = surface_mesh(self.sim.handles.deformables["heap"].surface, q)
        self.assertEqual((len(own), len(own_faces)), (len(vertices), len(faces)))
        # Two extractions over one position set agree as sets, not as arrays:
        # the sparse marching cubes hands out vertex slots in whatever order
        # its blocks finish, so the same mesh comes back in another numbering.
        keys = {tuple(k) for k in np.round(own / 1e-4).astype(np.int64)}
        found = sum(tuple(k) in keys for k in np.round(vertices / 1e-4).astype(np.int64))
        self.assertGreater(
            found / len(vertices), 0.99, f"{len(vertices) - found} drawn vertices are not the surfacer's"
        )
        points_key = self.viewer._qualify("/model/garments/heap/points")
        node = self.viewer._garment_nodes.get(points_key)
        self.assertTrue(node is None or not node.visible, "the points show under the surface")

    def test_it_is_regrown_as_the_heap_moves(self) -> None:
        self.session.reset_env()
        before, _ = self._surface()
        for _ in range(30):
            self.sim.step_teleop()
        after, faces = self._surface()
        self.assertGreater(len(faces), 100)
        self.assertGreater(float(before[:, 2].max() - after[:, 2].max()), 0.002, "the drawn surface did not settle")
        self.assertTrue(self.viewer._garment_nodes[self.key].visible)


@unittest.skipUnless(_rtx_runs(), "the path tracer needs ovrtx and a GPU with RT cores")
class TestTheTracerDrawsTheHeap(unittest.TestCase):
    """The frame is not blank where the heap is, drawn either way.

    Timed because the point path is one sphere instance per particle and the
    surface path regrows a mesh on the host each frame; the numbers are printed
    for the record rather than asserted, since they depend on the GPU.
    """

    def _frame(self, render: str) -> tuple[np.ndarray, float]:
        sim = build(render)
        try:
            renderer = sim.make_renderer("rtx", cameras=[camera()])
            try:
                renderer.render(sim.state)  # the first frame builds the stage
                sim.run(10)
                t0 = time.perf_counter()
                frame = renderer.render(sim.state)
                dt = time.perf_counter() - t0
            finally:
                renderer.close()
        finally:
            sim.close()
        return np.asarray(frame["probe"]["rgb"]), dt

    def test_points(self) -> None:
        rgb, dt = self._frame("points")
        count = heap_pixels(rgb)
        print(f"[rtx points] {count} heap pixels of {rgb.shape[0] * rgb.shape[1]}, {1e3 * dt:.1f} ms a frame")
        self.assertGreater(count, 300, "the heap is not in the frame")

    def test_surface(self) -> None:
        rgb, dt = self._frame("surface")
        count = heap_pixels(rgb)
        print(f"[rtx surface] {count} heap pixels of {rgb.shape[0] * rgb.shape[1]}, {1e3 * dt:.1f} ms a frame")
        self.assertGreater(count, 300, "the heap is not in the frame")


if __name__ == "__main__":
    unittest.main()
