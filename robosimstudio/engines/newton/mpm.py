# SPDX-License-Identifier: Apache-2.0
"""Newton's implicit MPM as the ``mpm`` backend: flat kwargs, a stepping rate, and the in-place contract.

:class:`newton.solvers.SolverImplicitMPM` takes a ``Config`` dataclass, which
is why it could not be handed an arrangement's parameters directly:
:func:`~robosimstudio.scene.physics.backends.build_solver` constructs every
backend as ``cls(model=view, **kwargs)``. :class:`SolverMPM` is that
constructor over the same solver, and three things besides:

* **A stepping rate of its own.** The arrangement's substep loop calls every
  solver once per substep -- sixteen times a control step under
  :data:`~robosimstudio.specs.GENERIC`, which is 960 Hz -- and an implicit MPM
  step is the most expensive thing in the scene, and its cost is the colliders'
  rather than the particles' (20k, 30k and 50k particles cost the same in
  ``probe_a``). Measured on an RTX PRO 6000 with a 28,812-particle rice bed
  and the YAM bimanual on ``bench_075`` under ``GENERIC``, graph-captured: one
  60 Hz control step costs 163 ms with the MPM stepped at 960 Hz, 25.9 ms at
  120 Hz and 15.6 ms at 60 Hz (the robot alone is 4.2 ms), with no visible
  difference in how the bed settles. So the solver steps at ``step_hz``
  (60 by registration) and skips the calls in between, taking the skipped
  calls' time into the one step it does take (``dt * stride``). The skipping
  is host-side Python, which a captured graph bakes in -- which is correct
  exactly when a control step holds a whole number of MPM steps, and the
  composer checks that before anything is built
  (:func:`~robosimstudio.scene.physics.compose.compose`). The solver checks
  the other half itself: a capture that begins mid-stride raises rather than
  record a graph whose first step covers less time than the rest.
* **``separate_worlds`` on for a replicated model.** With one shared grid,
  rice in one world fell through the ground plane of another (``probe_c``,
  4 worlds, ``separate_worlds=False``); isolated per-world grids cost
  1.0-1.17x one world. Stated explicitly, an arrangement's value wins.
* **``project_outside`` after each step.** A 3 mm paddle through 4 mm voxels
  leaves 14-16% of the rice it sweeps behind it with the ``S2`` collider
  basis alone and 9.9% with a projection after every step (``probe_b``,
  32,890 particles). For a liquid the projection is wrong -- it pushes
  particles into the container wall and cost 6x in the cup probe -- so the
  liquid kind states ``project_outside=False``, and ``step_hz=120``, because
  what is poured falls faster than the band above allows at 60 Hz.

**What cannot be captured.** The ``"pic"`` collider basis copies to the host
inside its step (``CUDA error 906`` during capture, and the context is
poisoned afterwards); it runs only with ``use_cuda_graph=False`` on the
arrangement, at 49.5 ms per frame against 16.9 ms captured for ``S2`` in the
same scene. Nothing here guards against it beyond the schema's note, because
the failure is Newton's and immediate.

**Measures.** The water preset loses volume at tabletop scale in Newton 1.6
(a 60 mm column in a thick tank is 34% of its volume after 2 s), so a measure
of a liquid counts particles in a region rather than reading a height.
Recorded here because the kinds built over this solver inherit the limit.

**Separate worlds bin their particles through a corrected cell lookup.** With
``separate_worlds`` each world's cells are packed into one grid at an offset
along x, and Warp 1.18's environment lookup (``NanogridBase.
make_filtered_cell_lookup``) has two faults, both in rounding. It adds the
offset to the particle's float grid coordinate before flooring, where the grid
was built from the floored coordinate plus the offset in integers, so a
particle within a float ulp under a cell face asks for the next cell; and the
search it falls back on spans ``vec3i(uvw -+ 0.5)``, which truncates toward
zero, so for a negative coordinate the window sits one cell high and leaves
out the cell the particle is in. Together they bin the particle into a
neighbour up to half a cell away, its shape weights go negative, and so can a
node's mass. Measured on ``TestAHeapAlone.test_two_worlds_isolate``'s two
worlds (2 x 1,350 particles, RTX PRO 6000, Newton 1.7.0, Warp 1.18.0): world 1,
packed at +34, went non-finite in 12 of 245 trials (captured and eager, with
and without its one stray particle), always at the same control step, never
world 0 (+10, a finer ulp) and never in 50 trials of three worlds. In the
failing step a particle at local index (-2.0000007, 4.58, 2.72) went into cell
(-2, 5, 2) at y = -0.42, one node's inverse mass came out -0.016, three strain
nodes' Delassus blocks went NaN and the Gauss-Seidel solve carried it through
world 1's heap. Replaying that step's input went NaN 100 times in 100; with
:func:`_environment_cell_lookup` installed, 0 in 100 eager and 0 in 100
captured, and 0 in 360 trials (``probe3``, ``replay1``). It floors before
packing and floors the window, and is installed on Warp's ``Nanogrid`` at
import only while Warp's own lookup still has the faulty lines;
``tests/test_mpm_kinds``' ``TestTwoWorldsBinAlike`` is the four-particle case.
"""

import dataclasses
import inspect
from typing import Any

import newton
import warp as wp
import warp.fem as fem
from newton.solvers import SolverImplicitMPM

__all__ = ["SolverMPM"]


class SolverMPM(SolverImplicitMPM):
    """:class:`newton.solvers.SolverImplicitMPM` behind the backend constructor contract.

    Args:
        model: The model, or the coupled solver's view of it.
        step_hz: The rate the solver steps at [Hz], as a ceiling: called more
            often than this, it steps every ``round(1 / (step_hz * dt))``-th
            call with the accumulated ``dt``; called less often, it steps on
            every call at the caller's ``dt``. The class default is 120 (two
            steps per 60 Hz control step, 25.9 ms on the bed above); the
            registered ``mpm`` backend states 60 (15.6 ms). Keep a tool's
            travel per step under ~1.5 voxels, which at 4 mm and 60 Hz is
            0.36 m/s: the SDF query band is 1.7 voxels and a faster tool
            tunnels (``probe_b``, grid travel). A falling liquid is its own
            fast body -- it passes 0.36 m/s after 7 mm of free fall -- and
            honey poured from 100 mm went through the bowl's floor and the
            bench at 60 Hz (85 of 8,000 particles below the bench) and not at
            120 Hz; from 170 mm, 406 and 5 (``pour_probe``). The liquid kind
            states 120.
        project_outside: Project particles out of the colliders after every
            step, at ``project_gap`` (``None`` is Newton's ``voxel * sqrt(3)``).
        project_gap: Maximum closest-point query distance for the projection
            [m]. A pour at 2.4 m/s passed through a 10 cm slab at 120 Hz with the
            default gap and stayed on it with 30 mm (``probe_b``, grid travel).
        verbose: Newton's own; ``None`` follows ``wp.config.log_level``.
        **config: The fields of :class:`newton.solvers.SolverImplicitMPM.Config`.
            An unknown name raises ``TypeError`` naming it.

    Raises:
        TypeError: For a parameter that is neither one of this class's own nor
            a ``Config`` field, naming it and listing what is accepted.
    """

    OWN_PARAMETERS = ("step_hz", "project_outside", "project_gap", "verbose")
    """The constructor's parameters that are not ``Config`` fields."""

    def __init__(
        self,
        model: Any,
        *,
        step_hz: float = 120.0,
        project_outside: bool = False,
        project_gap: float | None = None,
        verbose: bool | None = None,
        **config: Any,
    ) -> None:
        fields = {f.name for f in dataclasses.fields(SolverImplicitMPM.Config)}
        unknown = sorted(set(config) - fields)
        if unknown:
            raise TypeError(
                f"SolverMPM: unknown parameter(s) {unknown}. The parameters are {sorted(self.OWN_PARAMETERS)} and "
                f"the fields of SolverImplicitMPM.Config: {sorted(fields)}."
            )
        if "separate_worlds" not in config and int(getattr(model, "world_count", 1) or 1) > 1:
            config["separate_worlds"] = True
        cfg = SolverImplicitMPM.Config(**config)
        super().__init__(model, cfg, verbose=verbose, enable_timers=False)
        self.config = cfg
        """The ``Config`` this solver was built from, for a reader and the manifest."""
        if not float(step_hz) > 0.0:
            raise ValueError(f"SolverMPM: step_hz must be positive, got {step_hz!r}")
        self.step_hz = float(step_hz)
        self.project_outside_after_step = bool(project_outside)
        self.project_gap = None if project_gap is None else float(project_gap)
        self._calls = 0
        self._call_dt: float | None = None
        self._stride = 1
        self._steps = 0
        self._step_dt = 0.0
        self._was_capturing = False
        self._held_wrenches: dict[int, wp.array] = {}
        self._harvested_at: dict[int, int] = {}

    # ----------------------------------------------------------------- rate
    @property
    def stride(self) -> int:
        """How many calls one step currently spans; 1 until the first call."""
        return int(self._stride)

    def _stride_for(self, dt: float) -> int:
        """The stride for calls of ``dt``; a new ``dt`` starts a new cadence from phase zero."""
        if self._call_dt is None or abs(float(dt) - self._call_dt) > 1.0e-12:
            self._call_dt = float(dt)
            self._calls = 0
            self._stride = max(1, int(round(1.0 / (self.step_hz * float(dt)))))
        return self._stride

    def _carry(self, state_in: newton.State, state_out: newton.State) -> None:
        """Copy the particle state forward on a skipped call between two distinct states.

        A bare solver is stepped between the pipeline's two states, which swap
        every substep; a skipped call has to hand the next one the same
        particles or it would read the state from two substeps ago. The
        ``mpm`` namespace carries the history (strain, plastic volume, stress,
        velocity gradient, frame), so it travels too. Under a coupled solver
        the entry is in place and this is never reached.
        """
        for name in ("particle_q", "particle_qd"):
            src, dst = getattr(state_in, name, None), getattr(state_out, name, None)
            if isinstance(src, wp.array) and isinstance(dst, wp.array) and src is not dst:
                wp.copy(dst, src)
        src_ns, dst_ns = getattr(state_in, "mpm", None), getattr(state_out, "mpm", None)
        if src_ns is None or dst_ns is None:
            return
        for name, src in vars(src_ns).items():
            if name.startswith("_") or not isinstance(src, wp.array):
                continue
            dst = getattr(dst_ns, name, None)
            if isinstance(dst, wp.array) and dst is not src:
                wp.copy(dst, src)

    # ----------------------------------------------------------------- step
    def step(
        self, state_in: newton.State, state_out: newton.State, control, contacts, dt: float, *, observables=None
    ) -> None:
        stride = self._stride_for(dt)
        capturing = bool(getattr(self.model.device, "is_capturing", False))
        if capturing and not self._was_capturing and self._calls % stride:
            raise RuntimeError(
                f"SolverMPM: graph capture began {self._calls % stride} call(s) into a {stride}-call step "
                f"(step_hz={self.step_hz:g}, dt={dt:g}); the graph would replay a first step shorter than the rest. "
                "Capture at a control-step boundary, and keep the step rate dividing the substep rate."
            )
        self._was_capturing = capturing
        self._calls += 1
        if self._calls % stride:
            if state_out is not state_in:
                self._carry(state_in, state_out)
            return
        step_dt = float(dt) * stride
        self._steps += 1
        self._step_dt = step_dt
        asked = {} if observables is None else {"observables": observables}  # a pre-1.7 parent takes no such keyword
        super().step(state_in, state_out, control, contacts, step_dt, **asked)
        if self.project_outside_after_step:
            self.project_outside(state_out, state_out, step_dt, gap=self.project_gap)

    def reset(self, state: newton.State, world_mask=None, flags=None) -> None:
        """Newton's reset, and the cadence starts again from phase zero with no wrench held.

        A reset happens between control steps, where the phase is already zero
        when the rate fits (which the composer checked); stating it keeps a
        reset taken mid-settle from carrying a partial step into the run. The
        held wrenches are zeroed and count as harvested, so the coupled steps
        before the first MPM step of the new episode -- fifteen of every
        sixteen calls at the registered stride -- read no load rather than the
        last contact of the episode before. Zeroing is a device memset, so it
        stays capturable; a ``world_mask`` reset zeroes every world's, which
        the next MPM step, at most one stride later, harvests again.
        """
        self._calls = 0
        for ptr, held in self._held_wrenches.items():
            held.zero_()
            self._harvested_at[ptr] = self._steps
        super().reset(state, world_mask=world_mask, flags=flags)

    # ------------------------------------------------------------- coupling
    def coupling_harvest_proxy_wrenches(self, body_local_to_proxy_global, out_body_f, *, dt: float, **kwargs) -> None:
        """The collider impulses as wrenches over the *step* this solver took, held through the skipped calls.

        The proxy coupler harvests once per coupled step, at that step's ``dt``,
        and the entry calls this solver ``substeps`` times inside it at
        ``dt / substeps``; the impulse is the last MPM step's, so it is divided
        by that step's own length (``stride`` calls of the entry's ``dt``), not
        by the coupler's ``dt`` times the stride. With
        ``backend("mpm", substeps=2)`` the coupler's ``dt`` was twice the
        call's, and a 50 mm cube of honey at rest in a free cup, 1.717 N of
        it, read -0.858 N on the cup at two substeps against -1.719 N at one;
        it now reads -1.719 N at two (``probe_c2_weight``, the honey-cup scene
        of ``tests/test_mpm_kinds`` under LIQUID_120HZ, 1 s at rest after 2 s,
        RTX PRO 6000, Newton 1.6.0).
        Whether to harvest is whether this solver stepped since this array's
        last harvest, not whether the entry's last call stepped: with three
        substeps and a stride of two, a step can fall on a call other than the
        last. On a coupled step with no MPM step in it the wrench is
        **copied from a persistent array**
        rather than harvested again, and that is load-bearing under a captured
        graph, not a saving: Newton's impulse field lives in the step's
        temporaries, so a harvest on a skipped call reads the *previous* step's
        buffers -- which, for the skipped calls at the start of a captured
        control step, are buffers allocated before the capture and freed when
        the first captured step replaced them. The graph then replayed reads of
        freed memory: an illegal memory access on the first replay at 29k
        particles, intermittent at 7k, and none at all with a stride of 1,
        where every harvest follows its own step (measured while this was
        built, RTX PRO 6000, Newton 1.6.0). The held copy is allocated on the
        first harvest, which the pipeline's eager warm-up makes an eager one.
        """
        del dt  # the coupled step's; the impulse is the last MPM step's
        held = self._held_wrenches.get(out_body_f.ptr)
        if held is None:
            held = self._held_wrenches[out_body_f.ptr] = wp.zeros_like(out_body_f)
        if self._harvested_at.get(out_body_f.ptr, 0) == self._steps:
            wp.copy(out_body_f, held)
            return
        self._harvested_at[out_body_f.ptr] = self._steps
        super().coupling_harvest_proxy_wrenches(body_local_to_proxy_global, out_body_f, dt=self._step_dt, **kwargs)
        wp.copy(held, out_body_f)


# ------------------------------------------------------------ cell lookup
# Module scope and no ``from __future__ import annotations`` in this file: Warp
# reads a function's annotations as objects, and the nested functions below
# annotate with their closure's types.


@wp.func
def _floor3(v: Any):
    return wp.vec3i(int(wp.floor(v[0])), int(wp.floor(v[1])), int(wp.floor(v[2])))


_WARP_CELL_LOOKUP = fem.Nanogrid.make_filtered_cell_lookup
"""Warp's own factory, which still answers for a grid of one environment."""

_FAULTY_LINES = ("packed_uvw = uvw + vec3_type(env_offset)", "wp.vec3i(uvw - offset * scales) + env_offset")
"""The two lines of Warp 1.18's environment lookup that the module docstring describes."""

_FAULTY_WARPS = ("1.18.",)
"""The Warp releases the workaround was written against: a lookup there without both lines is said by name."""


# _environment_cell_lookup is adapted from NVIDIA Warp 1.18.0 (https://github.com/NVIDIA/warp),
# NanogridBase.make_filtered_cell_lookup in warp/_src/fem/geometry/nanogrid.py, under the Apache License 2.0:
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Changed here: the environment lookup packs floor(uvw) + env_offset in integers and floors its fallback
# search window rather than truncating it (the module docstring says why), and a grid of one environment
# keeps Warp's own lookup.
def _environment_cell_lookup(grid_geo, filter_func: wp.Function = None):
    """``NanogridBase.make_filtered_cell_lookup`` with the environment lookup floored before it packs.

    Warp 1.18's, line for line, but for the two rounding faults in the module
    docstring: the cell is ``floor(uvw) + env_offset`` in integers, as the grid
    was built, and the fallback window is floored rather than truncated. A grid
    of one environment keeps Warp's own lookup.
    """
    if grid_geo.environment_count() <= 1:
        return _WARP_CELL_LOOKUP(grid_geo, filter_func)
    suffix = (f"{grid_geo.name}{filter_func.key if filter_func is not None else ''}", "floored")
    scalar = grid_geo.scalar_type
    vec3_type = wp.vec3d if scalar == wp.float64 else wp.vec3
    CoordsType = grid_geo.coords_type

    @fem.cache.dynamic_func(suffix=(suffix, "env"))
    def cell_lookup_env(
        args: grid_geo.CellArg, pos: vec3_type, max_dist: float, filter_data: Any, filter_target: Any, env_index: int
    ):
        grid = args.cell_grid
        uvw = wp.volume_world_to_index(grid, pos) + vec3_type(scalar(0.5))
        env_offset = args.env_offsets[env_index]
        packed = _floor3(uvw) + env_offset
        cell_index = grid_geo._lookup_cell_index(args, packed[0], packed[1], packed[2])
        if cell_index != -1:
            if args.cell_env[cell_index] == env_index:
                coords = grid_geo._cell_coordinates_local(args, cell_index, uvw)
                if wp.static(filter_func is None):
                    return fem.make_free_sample(cell_index, coords)
                else:
                    if filter_func(filter_data, cell_index) == filter_target:
                        return fem.make_free_sample(cell_index, coords)

        cell_size = vec3_type(
            wp.length(wp.volume_index_to_world_dir(grid, vec3_type(scalar(1.0), scalar(0.0), scalar(0.0)))),
            wp.length(wp.volume_index_to_world_dir(grid, vec3_type(scalar(0.0), scalar(1.0), scalar(0.0)))),
            wp.length(wp.volume_index_to_world_dir(grid, vec3_type(scalar(0.0), scalar(0.0), scalar(1.0)))),
        )
        offset = scalar(0.5)
        min_cell_size = wp.min(cell_size)
        max_offset = wp.ceil(max_dist / min_cell_size)
        scales = wp.cw_div(vec3_type(min_cell_size), vec3_type(cell_size))
        closest_cell = int(fem.NULL_ELEMENT_INDEX)
        closest_coords = CoordsType()
        while closest_cell == fem.NULL_ELEMENT_INDEX:
            uvw_min = _floor3(uvw - offset * scales) + env_offset
            uvw_max = _floor3(uvw + offset * scales) + wp.vec3i(1) + env_offset
            closest_dist = min_cell_size * min_cell_size * scalar(offset * offset)
            for i in range(uvw_min[0], uvw_max[0]):
                for j in range(uvw_min[1], uvw_max[1]):
                    for k in range(uvw_min[2], uvw_max[2]):
                        cell_index = grid_geo._lookup_cell_index(args, i, j, k)
                        if cell_index == -1:
                            continue
                        if args.cell_env[cell_index] != env_index:
                            continue
                        if wp.static(filter_func is not None):
                            if filter_func(filter_data, cell_index) != filter_target:
                                continue
                        dist, coords = grid_geo._cell_closest_point_local(args, cell_index, uvw)
                        if dist <= closest_dist:
                            closest_dist = dist
                            closest_coords = coords
                            closest_cell = cell_index
            if offset >= max_offset:
                break
            offset = wp.min(scalar(3.0) * offset, max_offset)
        return fem.make_free_sample(closest_cell, closest_coords)

    @fem.cache.dynamic_func(suffix=suffix, allow_overloads=True)
    def cell_lookup(
        args: grid_geo.CellArg, pos: vec3_type, max_dist: float, filter_data: Any, filter_target: Any, env_index: int
    ):
        return cell_lookup_env(args, pos, max_dist, filter_data, filter_target, env_index)

    return cell_lookup


def _install_environment_cell_lookup() -> bool:
    """Put :func:`_environment_cell_lookup` on Warp's ``Nanogrid`` while Warp's own has the faults; whether it did.

    Checked against the installed Warp's source, so a Warp that has fixed either
    line keeps its own lookup. At import, which is before any solver here bins a
    particle: Warp caches the functions a lookup is built into under the grid's
    name, so one built before this would be reused.

    Standing down is said, by name, when it may leave a fault in place: one of
    the two lines there without the other (the lookup changed around it), or
    neither found on a Warp the workaround was written for (:data:`_FAULTY_WARPS`).
    A later Warp with neither line has fixed both, and is quiet. A source that
    cannot be read says so on any Warp, since nothing then says which it is.
    """
    import warnings

    version = str(wp.config.version)
    where = "Warp's Nanogrid environment cell lookup (warp.fem NanogridBase.make_filtered_cell_lookup)"
    try:
        source = inspect.getsource(_WARP_CELL_LOOKUP)
    except (OSError, TypeError) as exc:
        warnings.warn(
            f"Warp {version}: the source of {where} cannot be read ({exc}), so whether it has the rounding faults "
            "robosimstudio corrects there is unknown; the correction is not installed, and separate MPM worlds bin "
            "their particles through Warp's own lookup",
            stacklevel=2,
        )
        return False
    found = [line in source for line in _FAULTY_LINES]
    if all(found):
        fem.Nanogrid.make_filtered_cell_lookup = _environment_cell_lookup
        return True
    if any(found):
        kept, gone = _FAULTY_LINES if found[0] else _FAULTY_LINES[::-1]
        warnings.warn(
            f"Warp {version}: {where} still has {kept!r} but not {gone!r}, so it changed around the rounding "
            "faults robosimstudio corrects there; the correction is not installed, and separate MPM worlds bin "
            "their particles through Warp's own lookup",
            stacklevel=2,
        )
    elif version.startswith(_FAULTY_WARPS):
        warnings.warn(
            f"Warp {version}: neither faulty line robosimstudio corrects in {where} is in its source, on a "
            "release the correction was written for; it is not installed, and separate MPM worlds bin their "
            "particles through Warp's own lookup",
            stacklevel=2,
        )
    return False


ENVIRONMENT_LOOKUP_CORRECTED = _install_environment_cell_lookup()
"""Whether this module replaced Warp's environment cell lookup (Warp 1.18.0 has both faults)."""
