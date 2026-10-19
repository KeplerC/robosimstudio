# SPDX-License-Identifier: Apache-2.0
"""Genesis's SPH liquid under its legacy coupler, as a host engine beside the Newton islands.

The second Genesis bridge. :class:`~robosimstudio.engines.genesis.bridge.GenesisBridge`
runs the IPC coupler (cloth and FEM bodies); this one runs the **legacy coupler**,
which is the only Genesis coupler that steps SPH. A Genesis scene holds exactly one
coupler (``Simulator`` builds one of SAP, legacy or IPC from its options), and an
SPH body under the IPC coupler does not compile: its first step raises a
``QuadrantsCompilationError`` reading ``_coupler._rigid_sph``, an attribute only the
legacy coupler has (``p4_ipc_sph.py``, genesis-world 1.3.3). So an island picks its
coupler from its actors' kinds -- :func:`~robosimstudio.engines.genesis.bridge.coupler_for`
-- and an island that holds an ``sph`` actor beside a ``shell`` or ``tet`` one is
refused by name.

Measured in the pour probes (``p1_rest.py``, ``p3_pour.py``, ``p5_nowhere.py``; 4 mm
particles, a 50 mm cube of water in a 66 mm cup with 8 mm walls, RTX PRO 6000,
genesis-world 1.3.3), and each one decides something below:

* **Every mirror is a free, heavy body, held by velocity.** The coupler gives the
  water's pressure to a rigid wall only when the wall's link is not fixed (the
  buoyancy term in ``legacy_coupler.py`` skips fixed links). Against fixed walls the
  poured water packed to 69% of its height (21.0 mm in a 60 mm square glass where its
  count predicts 30.6); against free walls held in place it stood 29.1 mm (1e4 Pa, 20
  substeps). So a world-fixed shape is a free body of :data:`MIRROR_DENSITY` held at
  its pose with zero velocity every control step, and a moving one is placed at the
  start of the step and given the velocity that carries it to its target by the end,
  which Genesis's rigid solver integrates across the substeps. Posed with ``set_pos``
  and a zero velocity every substep instead, the walls the water saw would stand still
  while they jumped, and every substep would be a round trip from the host.
* **The reaction is the velocity the water gave the mirror.** A heavy mirror is
  pushed by microns, and ``mass * dv / dt`` is the force over the whole control step:
  the water at rest in the cup reads 1.003 to 1.009 of its weight at float32 in five
  runs. It stays honest while the hand moves: a cup tipped 60 degrees and turned back
  upright still full reads 1.09 N through the turn, the water's weight, and a cup
  turned back empty reads 0.0000 N (``p12_harvest.py``). Only the force is read: a
  heavy wall turned 3 degrees in one step ends 0.02 degrees off its aim with no water
  at all, from its own gyroscopic term (``p11_track.py``), so a mirror's couple is
  left out and the torque on the Newton body is each mirror's force about the body's
  centre of mass.
* **float32, not float64.** Genesis's legacy reaction is a momentum change, not the
  IPC coupler's difference of two nearly equal coordinates, so the 64-bit rule of
  :mod:`~robosimstudio.engines.genesis.runtime` does not apply to it, and on this GPU
  float64 cost 8x: one SPH substep of 1,728 particles took 1.9 ms at float64 and
  0.24 ms at float32 (``p2_cost.py``). The arrangement states ``precision="32"``.
* **The particles are Newton's.** The SPH entity is built empty (``Nowhere``) and
  Newton's own lattice written into it after build, so particle ``i`` of the actor is
  particle ``start + i`` of the model; at rest in the cup its top particle stands
  40.65 and 40.99 mm in two runs, where Genesis's own sampling stands 40.75 mm
  (``p5_nowhere.py``).
* **The mass is the volume it fills.** Genesis gives a particle ``0.8 * size**3 * rho``
  of mass (``sph_solver.py``, "an empirical value"), but settled water fills ``size**3``
  per particle (the 29.1 mm above). The builder passes ``rho / 0.8`` and
  ``stiffness / 0.8``: pressure accelerates a particle by stiffness over rest density
  and nothing else in the solve reads the density, so the motion is unchanged and
  the water weighs its density times the volume it fills.

The water is never a Newton solver's: its island is this engine, its particles are
written back every control step, and a renderer or a measure reads them as any
particle state.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from robosimstudio.core.math import quat_inverse, quat_multiply
from robosimstudio.engines.cosim import CosimRig
from robosimstudio.engines.feedback import WrenchAccumulator
from robosimstudio.engines.geometry import (
    ShapeGeometry,
    actor_builder,
    actor_geometry,
    iter_mirrors,
    register_actor_builder,
)
from robosimstudio.engines.pose import compose_many

from .runtime import ensure_initialized

__all__ = [
    "GENESIS_MASS_VOLUME",
    "MIRROR_DENSITY",
    "SPH_CONSUMES",
    "SphBridge",
    "liquid_args",
]

log = logging.getLogger(__name__)

MIRROR_DENSITY = 1.0e6
"""Density [kg/m^3] every mirror is built at. The cup's 66 x 66 x 8 mm floor weighs
35 kg here, so the water's 1 N moves it 9 microns in a control step, and the
velocity that motion leaves is the reaction read back."""

GENESIS_MASS_VOLUME = 0.8
"""Genesis's particle mass over ``size**3 * rho`` (``SPHSolver._particle_volume``)."""

SPH_CONSUMES = ("density", "exponent", "spacing", "stiffness", "surface_tension", "viscosity")
"""What an ``sph`` actor's params may state, and the builder reads."""


def liquid_args(params: dict) -> dict[str, float]:
    """``gs.materials.SPH.Liquid`` arguments for an ``sph`` actor's params.

    The density and the stiffness are divided by :data:`GENESIS_MASS_VOLUME` (see the
    module docstring); viscosity and surface tension are Genesis's own ``mu`` and
    ``gamma``, passed as they stand.
    """
    return {
        "rho": float(params.get("density", 1000.0)) / GENESIS_MASS_VOLUME,
        "stiffness": float(params.get("stiffness", 1.0e4)) / GENESIS_MASS_VOLUME,
        "exponent": float(params.get("exponent", 7.0)),
        "mu": float(params.get("viscosity", 5.0e-4)),
        "gamma": float(params.get("surface_tension", 0.01)),
    }


def _wxyz(q: np.ndarray) -> np.ndarray:
    """``xyzw`` rows as Genesis's ``wxyz``."""
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    return q[:, [3, 0, 1, 2]]


def _angular_velocity(start: np.ndarray, end: np.ndarray, dt: float) -> np.ndarray:
    """The constant angular velocity [rad/s] turning ``start`` into ``end`` (``xyzw``) in ``dt``, in the body's frame.

    The body's frame, not the world's, because that is how Genesis integrates a free
    joint's angular dofs (``q_next = q * exp(w dt)``): a box yawed 90 degrees about z
    and given (1, 0, 0) rad/s for 0.5 s turned 0.5 rad about world y
    (``p10_angvel.py``, genesis-world 1.3.3). A tilted, yawed wall turned 3 degrees
    about a skew axis in one control step ends 0.02 degrees off its target with this
    rate and 1.23 degrees off with the world-frame one (``p11_track.py``).
    """
    rel = quat_multiply(quat_inverse(start), end)
    if rel[3] < 0.0:
        rel = -rel
    s = float(np.linalg.norm(rel[:3]))
    if s < 1.0e-12:
        return np.zeros(3)
    return (2.0 * np.arctan2(s, float(rel[3])) / s) * rel[:3] / float(dt)


class SphBridge:
    """One Genesis scene under the legacy coupler: SPH bodies, heavy free mirrors, the per-step tick.

    Args:
        model: The finalized Newton model.
        recipe: The manifest's host-island record. Island-wide scalars ride it, as
            :class:`~robosimstudio.engines.genesis.solver.SolverGenesis` folds them in.
        substeps: Genesis substeps per control step. One ``scene.step()`` runs them
            all (``SimOptions.substeps``): the legacy coupler takes its ``dt`` per
            substep, which is the IPC bridge's reason for stepping one at a time
            and is not this one's.
        precision: ``"32"`` (see the module docstring); ``"64"`` runs, 8x slower.
        feedback: ``"auto"`` or ``"off"``.
        pose_source: ``"fresh"`` or ``"snapshot"``.
    """

    def __init__(
        self,
        model: Any,
        recipe: dict,
        *,
        substeps: int = 20,
        precision: str = "32",
        feedback: str = "auto",
        pose_source: str = "fresh",
    ) -> None:
        gs = ensure_initialized(precision, coupler="legacy")
        self.gs = gs
        self.model = model
        self.recipe = dict(recipe)
        self.substeps = max(1, int(substeps))
        self.feedback = str(feedback)
        self.control_dt = 1.0 / float(recipe.get("control_hz", 60.0))

        geoms = [actor_geometry(model, record) for record in recipe.get("actors", ())]
        spacing = sorted({float(g.params.get("spacing", 0.004)) for g in geoms})
        if len(spacing) != 1:
            raise ValueError(
                f"genesis island {recipe.get('island')!r}: its sph actors state particle spacings {spacing}; "
                "Genesis's SPH solver has one particle size per scene, so every liquid in an island states the same"
            )
        self.spacing = spacing[0]
        lo, hi = self._domain(geoms)
        flags = dict.fromkeys(
            ("rigid_mpm", "rigid_pbd", "rigid_fem", "mpm_sph", "mpm_pbd", "fem_mpm", "fem_sph"), False
        )
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.control_dt,
                substeps=self.substeps,
                gravity=tuple(float(v) for v in recipe.get("gravity", (0.0, 0.0, -9.81))),
            ),
            # Newton resolves rigid contact; the mirrors are its bodies seen twice.
            rigid_options=gs.options.RigidOptions(enable_collision=False, enable_self_collision=False),
            sph_options=gs.options.SPHOptions(
                particle_size=self.spacing,
                pressure_solver=str(recipe.get("pressure_solver", "WCSPH")),
                lower_bound=tuple(float(v) for v in lo),
                upper_bound=tuple(float(v) for v in hi),
            ),
            coupler_options=gs.options.LegacyCouplerOptions(rigid_sph=True, **flags),
            show_viewer=False,
        )

        self.mirrors: list[dict] = []
        for geom in iter_mirrors(model, recipe.get("mirrors", ())):
            entry = self._add_mirror(geom)
            if entry is not None:
                self.mirrors.append(entry)
        self.actors: list[dict] = []
        for geom in geoms:
            build = actor_builder("genesis", geom.kind)
            if build is None:
                raise ValueError(f"genesis cannot build a {geom.kind!r} actor ({geom.name!r}) under its legacy coupler")
            entity = self.scene.add_entity(
                material=build(self, geom), morph=gs.morphs.Nowhere(n_particles=int(geom.count))
            )
            self.actors.append({"name": geom.name, "entity": entity, "geom": geom, "prev": geom.rest.copy()})

        self.scene.build()
        for entry in self.actors:
            rest = entry["geom"].rest.astype(np.float32)
            entry["entity"].set_particles_pos(rest)
            entry["entity"].set_particles_active(np.ones(len(rest), dtype=bool))
            entry["rest"] = rest
        self._index_mirrors()
        self.rig = CosimRig(
            [m["geom"].body for m in self.mirrors if m["geom"].moving],
            substeps=self.substeps,
            pose_source=str(pose_source),
        )
        self.feedback_enabled = (
            self.feedback != "off"
            and any(m["geom"].two_way and m["geom"].moving for m in self.mirrors)
            and int(getattr(model, "body_count", 0)) > 0
        )
        self.apply_wrenches = None
        """Set by the owning solver: ``(wrench[n_bodies, 6] float32) -> None``."""
        self._acc = (
            WrenchAccumulator(
                model,
                alpha=float(recipe.get("feedback_alpha", 0.35)),
                dv_max=float(recipe.get("feedback_dv_max", 0.5)),
            )
            if self.feedback_enabled
            else None
        )
        self.last_force: dict[int, np.ndarray] = {}
        """The last control step's reaction per Newton body [N], before shaping. For a reader."""

    # ----------------------------------------------------------------- build
    def _domain(self, geoms) -> tuple[np.ndarray, np.ndarray]:
        """The SPH solver's box: the liquids' rest bounds grown by the island's reach.

        Its walls hold a particle in (Genesis clamps one that crosses), so it is
        where water that falls off the bench ends; its size costs nothing, since
        the spatial hash is capped at 150 cells a side (an SPH substep of 1,728
        particles took 0.50 ms with 88 x 88 x 46 cells and 0.49 ms with 25 x 25 x 46,
        ``p2_cost.py`` at float32).
        """
        rest = np.concatenate([g.rest for g in geoms]) if geoms else np.zeros((1, 3))
        side = float(self.recipe.get("sph_reach", 0.5))
        down = float(self.recipe.get("sph_reach_down", 0.25))
        lo = rest.min(axis=0) - np.array([side, side, down])
        hi = rest.max(axis=0) + np.array([side, side, side])
        return lo, hi

    def _add_mirror(self, geom: ShapeGeometry) -> dict | None:
        """One Newton shape as a free, heavy, gravity-compensated Genesis body, or ``None`` for a plane.

        A plane is left out: the solver's box has a floor, and a plane mirrored as a
        box would be a body the size of the room. Each mirror is built at its pose,
        so the first hold is no jump.
        """
        gs = self.gs
        if geom.kind == "plane":
            return None
        place = geom.local if not geom.moving else compose_many(self._body_pose(geom.body), geom.local)[0]
        common = {
            "pos": tuple(float(v) for v in place[:3]),
            "quat": tuple(float(v) for v in _wxyz(place[3:7])[0]),
            "fixed": False,
        }
        scale = geom.scale
        if geom.kind == "sphere":
            morph = gs.morphs.Sphere(radius=float(scale[0]), **common)
        elif geom.kind == "box":
            morph = gs.morphs.Box(size=tuple(float(v) for v in 2.0 * scale[:3]), **common)
        elif geom.kind == "cylinder":
            morph = gs.morphs.Cylinder(radius=float(scale[0]), height=2.0 * float(scale[1]), **common)
        else:
            mesh = geom.mesh()
            if mesh is None:
                return None
            from .bridge import write_obj

            morph = gs.morphs.Mesh(
                file=write_obj(self.recipe, geom, mesh),
                align=False,
                convexify=False,
                decimate=False,
                watertighten=None,
                recompute_inertia=True,
                **common,
            )
        entity = self.scene.add_entity(
            morph=morph,
            material=gs.materials.Rigid(
                rho=MIRROR_DENSITY,
                gravity_compensation=1.0,
                coup_friction=float(self.recipe.get("wall_friction", 0.1)),
            ),
        )
        return {"entity": entity, "geom": geom, "place": np.asarray(place, dtype=np.float64)}

    def _body_pose(self, body: int) -> np.ndarray:
        return np.asarray(self.model.body_q.numpy()[int(body)], dtype=np.float64)

    def _index_mirrors(self) -> None:
        """The rigid solver's coordinate and velocity indices for every mirror, in one batch each."""
        q_idx, d_idx = [], []
        for entry in self.mirrors:
            entity = entry["entity"]
            q_idx.append(np.arange(entity.q_start, entity.q_start + 7))
            d_idx.append(np.arange(entity.dof_start, entity.dof_start + 6))
            entry["mass"] = float(entity.get_mass())
        self._q_idx = np.concatenate(q_idx) if q_idx else np.zeros(0, np.int64)
        self._d_idx = np.concatenate(d_idx) if d_idx else np.zeros(0, np.int64)
        self._moving = [i for i, m in enumerate(self.mirrors) if m["geom"].moving]
        self._locals = np.asarray([self.mirrors[i]["geom"].local for i in self._moving]).reshape(-1, 7)
        self._qpos = np.zeros((len(self.mirrors), 7))
        self._vel = np.zeros((len(self.mirrors), 6))
        for i, entry in enumerate(self.mirrors):
            self._qpos[i, :3] = entry["place"][:3]
            self._qpos[i, 3:] = _wxyz(entry["place"][3:7])[0]

    # ------------------------------------------------------------------ tick
    def attach_pipeline(self, pipeline: Any) -> None:
        self.rig.attach_pipeline(pipeline)

    def tick(self, state: Any, dt: float) -> None:
        """Hold the statics, aim the moving mirrors, step the substeps, harvest, write the particles back."""
        rig = self.rig
        targets = rig.body_poses(state)
        if self._moving:
            starts = rig.start_poses(targets)
            bodies = [self.mirrors[i]["geom"].body for i in self._moving]
            begin = compose_many(np.asarray([starts[b] for b in bodies]), self._locals)
            end = compose_many(np.asarray([targets[b] for b in bodies]), self._locals)
            step = float(self.control_dt)
            for row, i in enumerate(self._moving):
                self._qpos[i, :3] = begin[row, :3]
                self._qpos[i, 3:] = _wxyz(begin[row, 3:7])[0]
                self._vel[i, :3] = (end[row, :3] - begin[row, :3]) / step
                self._vel[i, 3:] = _angular_velocity(begin[row, 3:7], end[row, 3:7], step)
        if len(self.mirrors):
            solver = self.scene.sim.rigid_solver
            solver.set_qpos(self._qpos.reshape(-1), qs_idx=self._q_idx)
            solver.set_dofs_velocity(self._vel.reshape(-1), dofs_idx=self._d_idx)
        self.scene.step()
        rig.commit(targets)

        if self.feedback_enabled and self.apply_wrenches is not None:
            self._harvest(targets)

        for entry in self.actors:
            positions = np.asarray(entry["entity"].get_particles_pos().cpu()).reshape(-1, 3)
            entry["prev"] = rig.write_particles(
                state, entry["geom"].start, entry["geom"].count, positions, dt, entry["prev"]
            )

    def _harvest(self, targets: dict[int, np.ndarray]) -> None:
        """Each two-way mirror's force: its mass times the velocity the water gave it, over the step."""
        acc = self._acc
        acc.begin()
        velocity = np.asarray(
            self.scene.sim.rigid_solver.get_dofs_velocity(dofs_idx=self._d_idx).cpu(), dtype=np.float64
        ).reshape(-1, 6)
        self.last_force = {}
        for i, entry in enumerate(self.mirrors):
            geom = entry["geom"]
            if not (geom.two_way and geom.moving) or geom.body not in targets:
                continue
            force = entry["mass"] * (velocity[i, :3] - self._vel[i, :3]) / float(self.control_dt)
            acc.add_at_shape(geom.body, force, targets[geom.body], geom.local)
            self.last_force[geom.body] = self.last_force.get(geom.body, np.zeros(3)) + force
        self.apply_wrenches(acc.shape(self.control_dt))

    # ----------------------------------------------------------------- reset
    def capture_rest(self) -> None:
        """The settled particles, as what a reset returns to."""
        for entry in self.actors:
            entry["rest"] = np.asarray(entry["entity"].get_particles_pos().cpu(), dtype=np.float32).reshape(-1, 3)

    def restore_rest(self) -> None:
        """The particles back where they settled, at rest; the next tick places the mirrors afresh."""
        for entry in self.actors:
            entity, rest = entry["entity"], entry["rest"]
            entity.set_particles_pos(rest)
            entity.set_particles_vel(np.zeros_like(rest))
            entry["prev"] = rest.astype(np.float64).copy()
        self.rig.mark_teleport()
        if self.feedback_enabled and self.apply_wrenches is not None:
            self.apply_wrenches(self._acc.zero())


def _build_sph(bridge: Any, geom: Any) -> Any:
    """The ``SPH.Liquid`` material for an actor; :class:`SphBridge` builds the entity itself."""
    return bridge.gs.materials.SPH.Liquid(**liquid_args(geom.params), sampler="regular")


# The ``sph`` kind's reader is the library's own (``engines/geometry.py``); the builder is this engine's.
register_actor_builder(
    "genesis",
    "sph",
    _build_sph,
    consumes=SPH_CONSUMES,
    summary="Genesis builds a liquid (SPH.Liquid) from an sph actor, under its legacy coupler.",
)
