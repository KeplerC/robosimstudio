# SPDX-License-Identifier: Apache-2.0
"""The entity behind every particle kind: sample a region, hand the particles to the MPM, read them back.

:class:`EntityParticleBody` is what ``rss.heap``, ``rss.liquid`` and
``rss.dough`` build. The three kinds subclass it for their domain and nothing
else: the builder calls, the settle, the reset, the handle and the tunables are
the same for a heap of rice and a pool of honey, because to the implicit MPM
they are the same thing with different numbers per particle.

What is particular to this backend, and why each piece is here:

* **The MPM's custom attributes are registered before the particles are
  added.** An attribute registered after is one the builder never kept for the
  particles already there (the rod does the same for VBD's). The compiler has
  registered them from the arrangement already; a second registration is
  accepted.
* **A reset asks the solvers for a particle reset.** ``Sim.reset_scene`` resets
  every solver with ``flags=0`` -- the joint coordinates and nothing else -- and
  the implicit MPM keeps its history (plastic volume ``Jp``, elastic strain,
  stress, the velocity gradient) on the state, where only a ``PARTICLE``-flagged
  reset clears it. After a press ``Jp`` had reached 1e5 in the island test; the
  particles were back where they started and the material was not.
* **Container margins are written at build.** A liquid leaks through a wall
  thinner than about two voxels under the graph-capturable collider basis;
  the cup probe measured a 3 mm wall losing 60% of its water in 2 s at
  rest and 5% once every shape the liquid touches carried a signed-distance
  margin of half a voxel. The margin is a per-shape builder field the solver
  reads at construction, so it is written here, onto the shapes of the
  entities the spec names as its containers.
"""

from __future__ import annotations

import math
import re
import warnings
import zlib
from typing import TYPE_CHECKING, Any, ClassVar

import newton
import numpy as np
import warp as wp

from robosimstudio.assetlib.materials import MaterialRecord, get_material
from robosimstudio.core.math import quat_to_mat
from robosimstudio.core.probes.particle import ParticleProbe
from robosimstudio.core.render.particles import ParticleSurfacer
from robosimstudio.domains.capabilities import ParticleMixin
from robosimstudio.domains.selection import resolve_selector
from robosimstudio.scene.entity import Entity

from .spec import _STRESS_KEYS, MPM_KEYS, NEWTON_MPM_STRESS_SCALE, ParticleBodySpec, ParticleMaterialSpec

if TYPE_CHECKING:
    from robosimstudio.core.pipeline import ScenePipeline

__all__ = [
    "DEFAULT_CONTAINER_MARGIN",
    "EntityParticleBody",
    "island_solver",
    "resolve_material",
    "sample_positions",
]

DEFAULT_CONTAINER_MARGIN = 0.002
"""Half the registered 4 mm voxel [m]: the signed-distance margin a container's
shapes get when a spec names no other (``probe_c_water``: 1.5-2 mm at 3-4 mm
voxels cut the leak from 60% to 5% of the water)."""

VOXEL_RATIO_BAND = (1.5, 4.0)
"""Samples per voxel axis (``voxel_size / particle_spacing``) outside which the
entity warns: below, cells go unsampled and the surface tears; above, memory is
spent for no change in motion, since the step's cost is the colliders'."""


def resolve_material(material: Any) -> ParticleMaterialSpec:
    """A material as a spec, from a spec, a :class:`MaterialRecord` or a registered name."""
    if isinstance(material, ParticleMaterialSpec):
        return material
    if isinstance(material, MaterialRecord):
        material = material.spec
        if not isinstance(material, ParticleMaterialSpec):
            raise TypeError(f"material record holds a {type(material).__name__}, not a ParticleMaterialSpec")
        return material
    if isinstance(material, str):
        record = get_material(material)
        if not isinstance(record.spec, ParticleMaterialSpec):
            raise TypeError(
                f"material {material!r} is a {type(record.spec).__name__}, not a particle material; "
                "a particle body takes one of the granular kind's records (rice, sand, water, honey, dough, ...)"
            )
        return record.spec
    raise TypeError(
        f"material must be a ParticleMaterialSpec, a MaterialRecord or a name, got {type(material).__name__}"
    )


def _lattice(extent: tuple[float, float, float], spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Lattice points filling ``[-x/2, x/2] x [-y/2, y/2] x [0, z]``, and the cell they stand in.

    One sample per cell of ``extent / round(extent / spacing)``, at the cell's
    centre, so the lattice ends on the faces rather than overhanging them.
    """
    dims = np.array([max(1, int(round(float(s) / spacing))) for s in extent], dtype=np.int64)
    cell = np.asarray(extent, dtype=np.float64) / dims
    axes = [(np.arange(n) + 0.5) * c for n, c in zip(dims, cell, strict=True)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    grid[:, 0] -= 0.5 * float(extent[0])
    grid[:, 1] -= 0.5 * float(extent[1])
    return grid, cell


def sample_positions(spec: ParticleBodySpec) -> np.ndarray:
    """Particle positions for ``spec`` in its own frame, ``[n, 3]``: bottom-face centre at the origin, z up.

    A box is the lattice; a cylinder keeps the lattice points inside the
    inscribed ellipse; a mesh keeps the points ``trimesh.contains`` says are
    inside, over the mesh's own bound. The jitter is drawn once per spec name,
    so two builds of one spec sample the same positions.
    """
    return _sample(spec)[0]


def _sample(spec: ParticleBodySpec) -> tuple[np.ndarray, np.ndarray]:
    """:func:`sample_positions`, and the lattice cell each particle stands in ``[3]``."""
    spacing = float(spec.particle_spacing)
    if spec.shape == "mesh":
        points, cell = _mesh_points(spec, spacing)
    else:
        points, cell = _lattice(tuple(float(v) for v in spec.size), spacing)
        if spec.shape == "cylinder":
            rx, ry = 0.5 * float(spec.size[0]), 0.5 * float(spec.size[1])
            inside = (points[:, 0] / rx) ** 2 + (points[:, 1] / ry) ** 2 <= 1.0
            points = points[inside]
    if spec.jitter > 0.0 and len(points):
        rng = np.random.default_rng(zlib.crc32(spec.name.encode("utf-8")))
        points = points + (rng.random(points.shape) - 0.5) * (2.0 * float(spec.jitter)) * cell
    return np.ascontiguousarray(points, dtype=np.float64), cell


def _mesh_points(spec: ParticleBodySpec, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Lattice points inside the asset's mesh, shifted so its bound's bottom centre is the origin."""
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover -- trimesh is a dependency of the library
        raise ImportError("filling a mesh with particles needs trimesh") from exc
    from robosimstudio.paths import resolve_asset

    mesh = trimesh.load(resolve_asset(spec.asset), force="mesh")
    scale = float(getattr(spec.asset, "scale", 1.0) or 1.0)
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh.faces), process=False)
    lo, hi = mesh.bounds
    extent = tuple(float(v) for v in (hi - lo))
    points, cell = _lattice(extent, spacing)
    # The lattice is centred in x and y and starts at z = 0; the mesh's bound is not.
    offset = np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), lo[2]])
    inside = mesh.contains(points + offset)
    points = points[inside]
    if not len(points):
        raise ValueError(
            f"particle body {spec.name!r}: no lattice point of spacing {spacing} m lies inside the mesh "
            f"(bound {extent}); the mesh may not be watertight, or the spacing is larger than it"
        )
    return points, cell


def island_solver(pipeline: Any, entity_name: str) -> Any | None:
    """The MPM solver stepping ``entity_name``'s island, or ``None`` when it is not on one.

    The composed program knows which island a name landed in and which solver
    steps it; an uncoupled program (a heap alone) answers with the one solver
    it has. Duck-typed against the program, as :func:`host_bridge` is.
    """
    from robosimstudio.engines.newton.mpm import SolverMPM

    program = getattr(pipeline, "program", None)
    if program is None:
        return None
    island = None
    island_of = getattr(program, "island_of", None)
    if callable(island_of):
        try:
            island = island_of(entity_name)
        except (KeyError, LookupError):
            island = None
    candidates = []
    if island is not None:
        of_island = getattr(program, "solver_of_island", None)
        if callable(of_island):
            try:
                candidates.append(of_island(island.name))
            except (KeyError, LookupError):
                pass
    solver_for = getattr(program, "solver_for", None)
    if callable(solver_for):
        try:
            candidates.append(solver_for(entity_name))
        except (KeyError, LookupError):
            pass
    candidates.append(getattr(program, "solver", None))
    for candidate in candidates:
        if isinstance(candidate, SolverMPM):
            return candidate
        inner = getattr(candidate, "solvers", None)
        if isinstance(inner, dict):
            for value in inner.values():
                if isinstance(value, SolverMPM):
                    return value
    return None


class EntityParticleBody(ParticleMixin, Entity):
    """One particle body, built from a :class:`ParticleBodySpec` into world ``world``.

    The kinds subclass this and set :attr:`domain`; nothing else differs.
    """

    spec: ParticleBodySpec
    domain: ClassVar[str] = "granular"

    def __init__(self, spec: ParticleBodySpec, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.particles: np.ndarray = np.zeros(0, dtype=np.int32)
        """Global indices of this body's particles (the template's, in a replicated build)."""
        self.sets: dict[str, np.ndarray] = {}
        self.probe: ParticleProbe | None = None
        self.rest: dict[str, np.ndarray] = {}
        self.material: ParticleMaterialSpec = resolve_material(spec.material)
        """The material as resolved from the spec: a name becomes its record's spec here, once."""
        self.solver: Any = None
        """The MPM solver stepping this body, found at scene-ready; ``None`` on another backend."""
        self.surface: ParticleSurfacer | None = None
        """The :class:`~robosimstudio.core.render.particles.ParticleSurfacer` over this body, when ``render == "surface"``."""
        self._handle: Any = None
        self._count = 0
        self._particle_volume = float(spec.particle_volume)

    # -------------------------------------------------------------- phase 1
    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        from newton.solvers import SolverImplicitMPM

        SolverImplicitMPM.register_custom_attributes(builder)
        spec = self.spec
        local, cell = _sample(spec)
        placed = self.placed((*spec.at, *spec.quat))
        rotation = quat_to_mat(np.asarray(placed[3:7], dtype=np.float64))
        world = local @ rotation.T + np.asarray(placed[:3], dtype=np.float64)
        n = len(world)
        # One cell's volume per particle, as mass and as radius: Newton's
        # implicit MPM takes a particle's volume to be ``8 r^3``, so the radius
        # is half the cell's cube root and the density it derives is the material's.
        self._particle_volume = float(np.prod(cell))
        mass = self._particle_volume * float(self.material.density)
        attributes = {key: [value] * n for key, value in self.material.custom_attributes().items()}
        builder.add_particles(
            pos=[tuple(float(c) for c in row) for row in world],
            vel=[(0.0, 0.0, 0.0)] * n,
            mass=[mass] * n,
            radius=[self.particle_radius] * n,
            flags=[int(newton.ParticleFlags.ACTIVE)] * n,
            custom_attributes=attributes,
        )
        self._count = n
        self._apply_container_margins(builder)

    def _apply_container_margins(self, builder: newton.ModelBuilder) -> None:
        """Give every collision shape of the named container entities a signed-distance margin.

        Found by label, and by the whole name: a container's shapes are the
        ones on the body that carries its name (a rigid body's, with a shell
        collider's floor and walls) and the world-fixed ones a static builds,
        labelled with the name, ``<name>_collision`` or ``<name>_collision_<i>``,
        each followed by ``_convex_<i>`` for the hulls past the first that
        Newton's convex decomposition adds (a static bowl with
        ``approximate="coacd"`` is ``bowl`` and ``bowl_convex_1`` to ``_26``). A
        prefix is not enough, since ``cup`` would then also take ``cup_holder``
        and ``cup_2``. A name that matches nothing is a warning rather than an error, because
        the container may be built in a later stage of a scene that will be
        compiled again with it -- but a liquid poured into a wall with no
        margin leaks, and the warning says which name was missed.
        """
        if not self.spec.containers:
            return
        margin = float(self.spec.container_margin or DEFAULT_CONTAINER_MARGIN)
        labels = list(getattr(builder, "shape_label", []))
        bodies = list(getattr(builder, "body_label", []))
        for name in self.spec.containers:
            owned = {i for i, label in enumerate(bodies) if label == name}
            own = re.compile(rf"{re.escape(name)}(?:_collision(?:_\d+)?)?(?:_convex_\d+)?")
            shapes = [s for s in range(len(labels)) if builder.shape_body[s] in owned or own.fullmatch(labels[s])]
            if not shapes:
                warnings.warn(
                    f"particle body {self.spec.name!r}: container {name!r} has no shapes in the builder yet, "
                    "so none got a margin; a container is built in the scenery or objects stage, before a "
                    "deformable",
                    stacklevel=2,
                )
                continue
            for s in shapes:
                builder.shape_margin[s] = max(float(builder.shape_margin[s]), margin)

    # -------------------------------------------------------------- phase 2
    def handle(self, model: newton.Model) -> Any:
        """The particles as a point body: no triangles, drawn at the radius the model holds.

        ``surface`` is installed at scene-ready, once the solver exists to
        reconstruct one; a renderer meeting the handle before then draws points.
        """
        handle = super().handle(model)
        handle.point_radius = self.particle_radius
        if self.surface is not None:
            handle.surface = self.surface_mesh  # the compiler may ask for the handle after scene-ready
        self._handle = handle
        return handle

    def tunables(self, pipeline: Any = None):
        """This body's material, as something a panel may change while the scene runs.

        Read back from ``model.mpm.*`` at the first particle, so the panel shows
        what the solver holds. A write lands in the model's per-particle arrays
        for every world's copy of this body and tells the solvers the model's
        properties changed, which the implicit MPM answers by re-reading its
        material -- and which drops a captured graph, so it is ``RECAPTURE``.
        Density is ``REBUILD``: it became particle mass at build.
        """
        from robosimstudio.core.tuning import Tunable

        model = self.stepping_model or self.model
        if model is None or not len(self.particles):
            return ()
        return (
            Tunable(
                name="material",
                spec=self.read_material(model),
                write=lambda field, value: self.write_material(field, value, pipeline),
                summary=f"What {self.name} is made of.",
            ),
        )

    def read_material(self, model: Any = None) -> ParticleMaterialSpec:
        """The material as the model holds it, at this body's first particle."""
        model = model or self.stepping_model or self.model
        first = int(self.particles[0])
        values: dict[str, float] = {}
        for key in MPM_KEYS:
            array = getattr(getattr(model, "mpm", None), key, None)
            if array is None:
                values[key] = float(getattr(self.material, key))
                continue
            value = float(array.numpy()[first])
            if key in _STRESS_KEYS:
                value /= NEWTON_MPM_STRESS_SCALE
            values[key] = value
        values["density"] = float(model.particle_mass.numpy()[first]) / self._particle_volume
        return ParticleMaterialSpec(color=self.material.color, **values)

    def _all_world_indices(self) -> np.ndarray:
        worlds = getattr(self, "worlds", None)
        count = int(getattr(worlds, "num_worlds", 1) or 1)
        stride = int(getattr(worlds, "particles_per_world", 0) or 0)
        base = np.asarray(self.particles, dtype=np.int64)
        return np.concatenate([base + w * stride for w in range(count)]) if count > 1 else base

    def write_material(self, field: str, value: Any, pipeline: Any = None) -> bool:
        """Write one material field into the model for this body's particles. ``False`` for density."""
        from robosimstudio.core.tuning import notify

        if field not in MPM_KEYS:
            return False
        model = self.stepping_model or self.model
        namespace = getattr(model, "mpm", None)
        array = getattr(namespace, field, None)
        if array is None:
            return False
        scaled = float(value) * (NEWTON_MPM_STRESS_SCALE if field in _STRESS_KEYS else 1.0)
        rows = self._all_world_indices()
        targets = [array]
        # The island's solver holds a view of the model; where the view copied
        # the attribute rather than sliced it, the copy is written too.
        view = getattr(getattr(getattr(self.solver, "model", None), "mpm", None), field, None)
        if view is not None and getattr(view, "ptr", None) != getattr(array, "ptr", None) and len(view) == len(array):
            targets.append(view)
        for target in targets:
            host = target.numpy().copy()
            host[rows] = scaled
            target.assign(host)
        notify(pipeline, int(newton.ModelFlags.MODEL_PROPERTIES))
        return True

    # -------------------------------------------------------------- phase 4
    def on_scene_ready(self, pipeline: ScenePipeline) -> None:
        """Resolve the sets, settle, find the solver, and offer a surface if asked."""
        self.resolve_sets(pipeline.state_0)
        self.join_settle(pipeline, (int(self.spec.settle_frames), int(self.spec.settle_substeps)))
        self.solver = island_solver(pipeline, self.name)
        self._check_voxel_ratio()
        if self.spec.render == "surface" and self.solver is not None:
            self._install_surface()

    def _check_voxel_ratio(self) -> None:
        config = getattr(self.solver, "config", None)
        voxel = getattr(config, "voxel_size", None)
        if voxel is None:
            return
        ratio = float(voxel) / float(self.spec.particle_spacing)
        lo, hi = VOXEL_RATIO_BAND
        if not lo <= ratio <= hi:
            warnings.warn(
                f"particle body {self.spec.name!r}: {ratio:.2f} samples per voxel axis (voxel {float(voxel) * 1e3:.1f} mm, "
                f"spacing {float(self.spec.particle_spacing) * 1e3:.2f} mm) is outside [{lo}, {hi}]; below it cells go "
                "unsampled and the surface tears, above it memory is spent for no change in motion",
                stacklevel=2,
            )

    def _install_surface(self) -> None:
        """A surfacer over this body at the solver's resolution, offered on the handle as a callable.

        The context ``SolverImplicitMPM.create_particle_surface`` would make:
        0.45 of the solver's voxel unless the spec names one, a kernel of
        three of those voxels, one per world of the solver's model.
        """
        voxel = self.spec.surface_voxel
        voxel = 0.45 * float(self.solver.config.voxel_size) if voxel is None else float(voxel)
        self.surface = ParticleSurfacer(
            self.particle_radius,
            voxel_size=voxel,
            device=self.solver.model.device,
            kernel_radius=3.0 * voxel,
            world_count=max(int(self.solver.model.world_count), 1),
        )
        if self._handle is not None:
            self._handle.surface = self.surface_mesh

    def surface_mesh(self, positions: Any) -> tuple[Any, Any, Any]:
        """``(vertices, triangles, normals)`` over ``positions`` ``[n, 3]`` of this body's particles.

        What ``DeformableHandle.surface`` points at. The vertices come back on
        the host, the triangles and normals on the device. Extracted by
        :attr:`surface`, which moves the particles into the positive octant
        first: on a bench they sit at negative y, where the extraction returned
        about a third of the surface, and moved to straddle z = 0 it never came
        back. The vertex count varies per call, which is why a renderer that cannot
        re-log a mesh per frame draws the points instead. Empty arrays when the
        reconstruction finds no triangles (a body scattered thinner than a voxel).
        """
        device = self.surface.device
        if isinstance(positions, wp.array):
            positions = positions.numpy()
        vertices, indices, normals = self.surface(positions)
        if vertices is None or indices is None or len(indices) == 0:
            empty = wp.empty(0, dtype=wp.vec3, device=device)
            return empty, wp.empty((0, 3), dtype=wp.int32, device=device), wp.empty(0, dtype=wp.vec3, device=device)
        return vertices, indices.reshape((-1, 3)), normals

    def settle(self, pipeline: ScenePipeline) -> None:
        """Let the body come to rest. Where it lands is the start state."""
        spec = self.spec
        if spec.settle_frames <= 0:
            return
        if spec.settle_substeps > 0:
            with pipeline.substeps_at(int(spec.settle_substeps)):
                for _ in range(int(spec.settle_frames)):
                    pipeline.step()
        else:
            for _ in range(int(spec.settle_frames)):
                pipeline.step()
        pipeline.sim_time = 0.0
        pipeline.step_count = 0

    # -------------------------------------------------------------- phase 5
    def reset(self, pipeline: ScenePipeline, episode, *, world: int = 0) -> None:
        """The particles back where they rested, and the MPM's history with them.

        ``reset_scene`` resets the solvers with ``flags=0`` before the entities
        run; that leaves the implicit MPM's per-particle history alone, so it is
        asked for a ``PARTICLE`` reset here, after the positions are restored,
        which also refreshes its cached collider poses from the restored state.
        """
        super().reset(pipeline, episode, world=world)
        if self.rest:
            pipeline.reset_solvers(flags=int(newton.StateFlags.PARTICLE))

    # ---------------------------------------------------------------- runtime
    def resolve_sets(self, state: newton.State) -> dict[str, np.ndarray]:
        """Resolve the spec's particle selectors against the rest layout, in the work-surface frame."""
        if not len(self.particles):
            return {}
        rest = np.asarray(state.particle_q.numpy(), dtype=np.float64)[self.particles] - np.asarray(self.frame[:3])
        for set_name, selector in self.spec.particle_sets:
            key = f"{self.spec.name}/{set_name}"
            indices = resolve_selector(selector, rest, self.particles)
            self.sets[key] = indices
            self.add_set_every_world(key, indices)
        return dict(self.sets)

    def set_indices(self, name: str) -> np.ndarray:
        """One named set's global particle indices, resolved inside this body."""
        key = name if "/" in name else f"{self.spec.name}/{name}"
        if key not in self.sets:
            raise KeyError(f"particle body {self.spec.name!r} has no particle set {name!r}; has {sorted(self.sets)}")
        return self.sets[key]

    def substrate_points(self, snapshot) -> np.ndarray | None:
        """This body's particles, ``[n, 3]``, from the snapshot's probe for it."""
        probe = getattr(snapshot, "particles", {}).get(self.name) if snapshot is not None else None
        if probe is None:
            return None
        return np.asarray(probe.points(self.name), dtype=np.float64)

    @property
    def particle_mass(self) -> float:
        """Mass of one particle [kg]: its lattice cell's volume times the density."""
        return self._particle_volume * float(self.material.density)

    @property
    def particle_radius(self) -> float:
        """Radius of one particle [m]: half its lattice cell's cube root, as the model holds it.

        What the solver reads back as the particle's volume (``8 r^3``), and so
        what the points are drawn at and the surface is grown over: the wipe's
        puddle, 2.5 mm across and 3 mm deep a cell, is 1.33 mm, not the 1.25
        mm of half the spacing.
        """
        return 0.5 * self._particle_volume ** (1.0 / 3.0)

    @property
    def floor_z(self) -> float:
        """The work surface this body was placed on, in world z [m]."""
        return float(self.frame[2]) + float(self.spec.at[2])

    def update_probe(self, state: newton.State, step: int | None = None, *, readback=None) -> ParticleProbe:
        """Read this step's particle positions, once per control step."""
        if self.probe is None:
            raise RuntimeError(f"particle body {self.spec.name!r} is not bound yet")
        if readback is not None and getattr(readback, "particle_q", None) is not None and len(readback.particle_q):
            self.probe.update_arrays(readback.particle_q, readback.particle_qd, step)
        else:
            self.probe.update(state, step)
        return self.probe


def voxel_ratio(voxel_size: float, spacing: float) -> float:
    """Samples per voxel axis: what :data:`VOXEL_RATIO_BAND` bounds."""
    return float(voxel_size) / float(spacing) if spacing > 0.0 else math.inf
