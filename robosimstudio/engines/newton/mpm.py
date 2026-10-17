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
  liquid kind states ``project_outside=False``.

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
"""

from __future__ import annotations

import dataclasses
from typing import Any

import newton
import warp as wp
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
            tunnels (``probe_b``, grid travel).
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
        self._stepped = False
        self._was_capturing = False
        self._held_wrenches: dict[int, wp.array] = {}

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
    def step(self, state_in: newton.State, state_out: newton.State, control, contacts, dt: float) -> None:
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
        self._stepped = not self._calls % stride
        if not self._stepped:
            if state_out is not state_in:
                self._carry(state_in, state_out)
            return
        step_dt = float(dt) * stride
        super().step(state_in, state_out, control, contacts, step_dt)
        if self.project_outside_after_step:
            self.project_outside(state_out, state_out, step_dt, gap=self.project_gap)

    def reset(self, state: newton.State, world_mask=None, flags=None) -> None:
        """Newton's reset, and the cadence starts again from phase zero.

        A reset happens between control steps, where the phase is already zero
        when the rate fits (which the composer checked); stating it keeps a
        reset taken mid-settle from carrying a partial step into the run.
        """
        self._calls = 0
        super().reset(state, world_mask=world_mask, flags=flags)

    # ------------------------------------------------------------- coupling
    def coupling_harvest_proxy_wrenches(self, body_local_to_proxy_global, out_body_f, *, dt: float, **kwargs) -> None:
        """The collider impulses as wrenches over the *step* this solver took, held through the skipped calls.

        The proxy coupler harvests after every call and divides the impulse by
        the call's ``dt``; the impulse was accumulated over ``stride`` calls, so
        on the call that stepped the wrench is the mean force over that step.
        On a skipped call the wrench is **copied from a persistent array**
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
        held = self._held_wrenches.get(out_body_f.ptr)
        if held is None:
            held = self._held_wrenches[out_body_f.ptr] = wp.zeros_like(out_body_f)
        if not self._stepped:
            wp.copy(out_body_f, held)
            return
        super().coupling_harvest_proxy_wrenches(
            body_local_to_proxy_global, out_body_f, dt=float(dt) * self._stride, **kwargs
        )
        wp.copy(held, out_body_f)
