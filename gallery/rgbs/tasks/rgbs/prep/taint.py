# SPDX-License-Identifier: Apache-2.0
"""The taint on the bar: an irreversible per-cell Preston-law removal field.

Polish's mark field, moved onto a cylinder and fed by a foreign engine. The law
is the same one -- ``d(intensity)/dt = -k_p * pressure * sliding speed``,
monotone, clamped at zero, nothing here can un-wipe a cell -- and two things
about *getting* its inputs are not.

**Where the pressure comes from, and why it is not ``ContactProbe``.** Polish
reads its press off MuJoCo's own contact table. That table is geom pairs, and a
SuperDex sponge has no MuJoCo geom: its Newton particles are built with
``ParticleFlags.ACTIVE`` cleared, so Newton's collider never sees them either.
A field copied from ``marks.py`` with a sponge substituted reads a press force
of **identically zero**, forever, without raising -- which is this repo's
characteristic failure and was the first thing checked here. The press instead
comes from the co-simulation bridge's own per-sample contact records
(:meth:`~robosimstudio.engines.superdex.bridge.SuperdexBridge.contact_points`),
which report position, force, both contact-point velocities and a quadrature
area for every sample the foam has against the bar.

**So pressure is measured, not nominal.** Polish divides its force by a
*stated* patch area and fits ``k_p`` against that stated number. Here the engine
reports each sample's own quadrature weight in square metres -- checked against
a sponge resting on a bench, where the weights sum to 7.000e-3 m2 against a
100 x 70 mm footprint -- so ``pressure = |force| / area`` per sample is the real
thing and there is no nominal to keep in step with a coefficient.

**One hook, in ``after``, not a kernel per substep.** Polish must run its
removal inside the captured substep block, which is why every gate in its
kernel is device data: a host-side ``if`` around a launch vanishes at capture.
Here the *input* only changes once per control step -- SuperDex steps in the
eager ``after`` phase and the contact query is filled by that step -- so
integrating per substep would integrate the same contact patch sixteen times.
The whole update therefore lives in ``after``, on the host, where an ordinary
``if`` is legal. **Do not move it back into ``post_solve``**; it would not be
more accurate, it would be sixteen times the same number.

**A cell takes the nearest sample, never the sum.** Contact samples are
quadrature points and several cover one 2.5 mm cell; adding their rates would
make removal a function of how finely the sponge happens to be tessellated.
Preston's law is local, so a cell is wiped at the rate of the sample nearest it.

**There is a pressure floor and it is not optional.** SuperDex's penalty
contact carries a padding band -- 2 mm by default -- so samples exist, with
positions and areas and essentially zero force, before anything is touching.
Measured on the 22 mm calibration pair: at zero overdrive there are 292 samples
spanning 43 deg of the bar at 0.002 N total. Without ``p_min`` a policy cleans
the bar by holding the sponge near it.

**And the field is the sponge's speedometer.** The goal's set-down clause is
``settle-speed sponge``, a library measure over Newton's particle velocities --
which the host island never writes, so it reads 0.0 through a 0.1 m/s carry.
The bridge does write the *positions* back every control step, and this entity
already holds the bridge and the sponge's name; so it publishes the centroid's
speed off those positions as ``sponge_speed``, and the document thresholds that.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import newton
import numpy as np
import warp as wp

from robosimstudio.core.render.finish import Finish
from robosimstudio.scene.entity import Entity, label_index
from robosimstudio.scene.registry import register_entity

from ..batched import HostField, instant
from ..look import INK_BLUE, STEEL
from .bar import CHROME
from .layout import LAYOUT, Layout

__all__ = ["MAX_SAMPLES", "PATCH_DONE", "REACHABLE_HALF_ANGLE", "EntityPrepTaint", "SpecPrepTaint", "prep_taint"]

PATCH_DONE = 0.9
"""Cleared share at which one taint patch counts as done, for the trace."""

REACHABLE_HALF_ANGLE = math.radians(60.0)
"""How far off the crown a press can reach, as a half-angle [rad].

**Measured, and a hard constraint on where taint may be authored.** Foam wrap
against press overdrive on the 22 mm bar and 70 x 45 mm sponge this field was
calibrated with, from ``rgbs/scripts/rgbs_prep_bench.py wrap``: 43 deg of total
span at first touch, 87 at 3 mm, 126 at 8 mm. Half of that widest reading is
63 deg, and this keeps 3 deg of margin off it. The 25 mm grab bar wraps a little
less per millimetre of press; the shipped stripe is 9 deg wide and nowhere near
the ceiling either way.

It is enforced in :meth:`EntityPrepTaint.build` because the failure it prevents
is silent and expensive: a patch whose outer cells sit past the wrap simply
never clears, the field caps below the document's gate, and the suite ships a
task no policy can finish. That is not a hypothetical -- the first authored
layout put patches 30 deg wide at 40 deg off the crown, and a full 8 mm press
stalled one of them at exactly 0.80 and stayed there for six more strokes."""

FORCE_WINDOW = 10
"""Control steps of press smoothing -- polish's number, for polish's reason:
the solver drops a settled pair for several consecutive steps, and a mean of
the nonzero readings scales a rate honestly where a raw sample does not."""

MAX_SAMPLES = 2048
"""Device capacity for one step's contact samples.

Measured worst case on the calibration sponge is 712 against the bar at 8 mm of
overdrive, plus whatever the gripper pads contribute -- but the pads are
filtered out before they get here, by body. Overflow keeps the hardest-pressing
samples and says so once per episode rather than dropping the tail in silence,
because a rate that depends on how many samples happened to fit is exactly the
kind of number that stays plausible while it is wrong."""

_PI = math.pi
_TWO_PI = 2.0 * math.pi


@wp.kernel(enable_backward=False)
def _wipe_kernel(
    cell_ut: wp.array(dtype=wp.vec2),  # (axial, angle) per cell, bar frame
    intensity: wp.array(dtype=float),
    sample_ut: wp.array(dtype=wp.vec2),  # (axial, angle) per contact sample
    sample_rate: wp.array(dtype=float),  # k_p * pressure * slide speed [1/s]
    counts: wp.array(dtype=int),  # [n_samples]
    law: wp.array(dtype=float),  # [reach^2, radius]
    dt: float,
    offset: int,  # where this world's block of the field starts
):
    """One cell's Preston update, from the nearest contact sample that reaches it.

    Distance is measured on the tube's surface: axial millimetres and arc
    millimetres, with the angle difference wrapped, so a cell near the crown and
    a sample just past it are as close as the surface says they are and not as
    close as their raw angles suggest.

    ``offset`` is this world's block of the field. The *cells* are shared -- one
    bar's surface, the same in every world -- and only what has been wiped off
    them is per world, so one launch per world over the same lattice writes a
    different block of one array.
    """
    i = wp.tid()
    cell_index = offset + i
    if intensity[cell_index] <= 0.0:
        return
    n = counts[0]
    reach2 = law[0]
    radius = law[1]
    best = float(0.0)
    best_d2 = float(1.0e9)
    cell = cell_ut[i]
    for k in range(n):
        s = sample_ut[k]
        du = s[0] - cell[0]
        da = s[1] - cell[1]
        if da > _PI:
            da = da - _TWO_PI
        if da < -_PI:
            da = da + _TWO_PI
        arc = da * radius
        d2 = du * du + arc * arc
        if d2 < best_d2 and d2 <= reach2:
            best_d2 = d2
            best = sample_rate[k]
    if best > 0.0:
        intensity[cell_index] = wp.max(0.0, intensity[cell_index] - best * dt)


@wp.kernel(enable_backward=False)
def _decal_color_kernel(
    intensity: wp.array(dtype=float),
    base: wp.vec3,
    clear: wp.vec3,
    start: int,
    shape_color: wp.array(dtype=wp.vec3),
    offset: int,  # where this world's block of the field starts
):
    """Fade one decal from the taint's colour toward the bar's own.

    Two offsets, and they are not the same number: ``offset`` indexes the field
    and ``start`` indexes the *shapes*, which stride by the model's own
    per-world shape count. A world that faded its own patches using its
    neighbour's decals would repaint the wrong bar -- and on a vision cell that
    is the whole observation.
    """
    i = wp.tid()
    t = 1.0 - intensity[offset + i]
    shape_color[start + i] = base * (1.0 - t) + clear * t


@dataclass(frozen=True, kw_only=True)
class SpecPrepTaint:
    """A marker stripe on the tube, as cells in the bar's own (axial, angle) surface.

    The geometry of the bar is **restated** here rather than reached for, so a
    field can be constructed with no bar in hand -- and :meth:`EntityPrepTaint.bind`
    asserts the restatement against what was actually built, which is polish's
    discipline and the reason a drifted radius cannot quietly rescale every
    pressure in the task. The defaults are the layout's, so the two agree
    unless somebody restates one by hand.
    """

    entity: str = "prep_taint"
    name: str = "taint"

    bar_object: str = "bar"
    sponge: str = "sponge"

    bar_radius: float = LAYOUT.tube_radius
    """Restated from the bar, and checked against it at bind."""

    bar_half_len: float = LAYOUT.straight_half
    """Restated from the bar: half the straight run [m], so a patch can be
    refused for running off the end of it before anything is built."""

    patches: tuple[tuple[float, float], ...] = ((0.0, 0.0),)
    """Patch centres as (axial [m], angle [rad]) on the tube: one, on the crown,
    mid-run -- the clip's marker stripe.

    One patch where there were three. The earlier layout spread three 30 mm
    patches 90 mm apart with two of them 40 deg off the crown, to make a
    policy vary its press depth; the clip has a single line drawn along the
    crown, and what it asks of a policy is travel -- the stripe is 200 mm long
    against a 75 mm sponge -- plus a real press, since the pressure floor cuts
    the feathered edge of a light one."""

    patch_half: tuple[float, float] = (LAYOUT.stripe_half, math.radians(LAYOUT.stripe_half_angle_deg))
    """Half-extent of the stripe: 100 mm along the bar by 9 deg of arc (2 mm) --
    a ~4 mm marker line down the middle 200 mm of the 340 mm run.

    Shorter than the photograph's stripe on purpose: the working hand's far
    stroke has to stay clear of the hand steadying the far elbow, and a stroke
    to the end of a longer stripe shoved that hand off."""

    cell_pitch: float = 0.0025
    """Cell size [m], on the surface.

    Not finer than the contact can resolve: samples are quadrature points on
    the foam's surface triangles, about three per triangle, which runs ~2 mm
    apart on the shipped lattice. Cells finer than that measure tessellation.
    2.5 mm rather than polish's 3 so a 4 mm wide stripe is three cells across
    instead of one."""

    sample_reach: float = 0.004
    """How far a contact sample reaches, on the surface [m].

    Slightly over the ~2 mm sample spacing, so the stripe has no unwiped
    lattice holes in it, and well under its length."""

    k_p: float = 1.2e-2
    """Preston coefficient [1/(Pa*m)].

    Fitted so a firm wipe clears a patch in a couple of seconds: at the
    measured 1.2 kPa of a 4 mm press and a 30 mm/s stroke this removes a unit
    of intensity in about 2.3 s. Re-fit it against ``rgbs/scripts/rgbs_prep_bench.py`` if
    the sponge's modulus or the bar's radius moves; unlike polish's, it is NOT
    tied to a stated patch area, because the pressure here is measured."""

    p_min: float = 300.0
    """Pressure floor [Pa]; below it a sample removes nothing.

    Not a tuning knob -- a correctness gate. See the module docstring: the
    penalty contact's padding band produces hundreds of samples at ~0 N before
    anything touches. 300 Pa sits an order of magnitude above the 1 mm-press
    reading (30-60 Pa) and an order below the 3 mm one (510-690 Pa)."""

    surface_tol: float = 0.004
    """How far off the tube's radius a sample may sit and still count [m].

    The elbows and legs are on the same body and the same radius, so this does
    not tell them apart from the straight run -- it does not have to: the
    stripe's cells end 70 mm short of the elbows and a sample reaches 4 mm.
    What it excludes is a sample against a gripper pad or the block's face
    that happens to share the body's contact table."""

    cleared_below: float = 0.1
    """Intensity at or under which a cell counts as cleared."""

    decal_lift: float = 2.0e-4
    """How far the decals stand off the tube [m]."""

    decal_thickness: float = 1.0e-4

    taint_color: tuple[float, float, float] = INK_BLUE
    """A marker streak on the bar, blue-black as in the reference clip. Stated
    because it disagrees with every role default: this is the thing the task is
    about and no palette entry means 'dirt'."""

    clear_color: tuple[float, float, float] = CHROME
    """What a cleared cell fades to: the bar's own stated colour, imported from
    the bar rather than transcribed, so a decal cannot leave a ghost of the
    field the moment somebody repaints the tube."""

    decal_finish: Finish | None = STEEL
    """The decals' finish under the path tracer: the steel they sit on, so a
    cleared cell reads as bare chrome rather than as a matte grey strip. A
    marker line on chrome is glossy too, so the same finish serves both ends of
    the fade."""


class _LazyRows:
    """A device array brought to the host on first use, and once for the step.

    Rows of it are wanted by a *world* -- one body's pose out of the fleet's --
    but the transfer is the whole array whoever asks, so three worlds asking is
    three copies of the same bytes and three stream syncs. This makes it one.
    Reading it up front instead would be simpler and slightly wrong: for most of
    an episode no world is touching anything, and that is exactly when nobody
    needs the poses.
    """

    __slots__ = ("_read", "_rows")

    def __init__(self, read) -> None:
        self._read = read
        self._rows: Any = None

    def __getitem__(self, index):
        if self._rows is None:
            self._rows = self._read()
        return self._rows[index]


@register_entity("prep_taint")
class EntityPrepTaint(Entity):
    """Decal shapes on the bar, one device field, one removal kernel, one feed."""

    spec: SpecPrepTaint

    def __init__(self, spec: SpecPrepTaint, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        # The sponge's particle slice (bound once the bridge is), its last
        # centroid, and the speed the field publishes off the two.
        self._sponge_range: tuple[int, int] | None = None
        self._sponge_prev: np.ndarray | None = None
        self._sponge_speed = 0.0
        self.bar_body: int = -1
        self.decal_shape_start: int = -1
        self._cell_ut_np: np.ndarray | None = None
        self._cell_patch_np: np.ndarray | None = None
        self._cell_ut: wp.array | None = None
        self._intensity: wp.array | None = None
        self._sample_ut: wp.array | None = None
        self._sample_rate: wp.array | None = None
        self._counts: wp.array | None = None
        self._ut_buf: np.ndarray | None = None
        self._rate_buf: np.ndarray | None = None
        self._law: wp.array | None = None
        self._rest_intensity: np.ndarray | None = None
        self._bridge = None
        self._windows: list[deque[float]] = []
        """One force window per world, sized in :meth:`bind`."""
        self._press_now: list[float] = [0.0]
        self._slide_now: list[float] = [0.0]
        self._touching: list[bool] = [False]
        self._overflows: int = 0
        self._can_recolor: bool = False
        self._published_at: tuple[Any, dict[str, Any]] | None = None
        self._field_host: HostField | None = None
        """The whole field on the host, read once per instant. See ``..batched``."""

    # ------------------------------------------------------------------ build
    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec = self.spec
        self.bar_body = label_index(list(builder.body_label), spec.bar_object)

        # Cells: a rectangle of pitch-sized cells per patch, measured on the
        # SURFACE -- axial metres by arc metres -- so a cell is the same size
        # wherever it sits and the grid does not stretch toward the crown.
        radius = float(spec.bar_radius)
        pitch = float(spec.cell_pitch)
        half_u, half_a = float(spec.patch_half[0]), float(spec.patch_half[1])
        for index, (cu, ca) in enumerate(spec.patches):
            reach = abs(float(ca)) + half_a
            if reach > REACHABLE_HALF_ANGLE:
                raise ValueError(
                    f"taint patch {index} reaches {math.degrees(reach):.0f} deg off the crown, and "
                    f"a full press wraps to {math.degrees(REACHABLE_HALF_ANGLE):.0f}. Its outer "
                    "cells can never be touched, so the field caps below the document's gate and "
                    "the task is unachievable -- silently, as a policy that stalls."
                )
            if abs(float(cu)) + half_u > float(spec.bar_half_len):
                raise ValueError(
                    f"taint patch {index} runs off the straight run "
                    f"({(abs(float(cu)) + half_u) * 1e3:.0f} mm from centre against a "
                    f"{float(spec.bar_half_len) * 1e3:.0f} mm half-length)"
                )
        nu = max(1, int(round(half_u / pitch)))
        na = max(1, int(round(half_a * radius / pitch)))
        cells: list[tuple[float, float]] = []
        patch_of: list[int] = []
        for index, (cu, ca) in enumerate(spec.patches):
            for iu in range(-nu, nu + 1):
                for ia in range(-na, na + 1):
                    u = float(cu) + iu * pitch
                    a = float(ca) + (ia * pitch) / radius
                    cells.append((u, a))
                    patch_of.append(index)
        self._cell_ut_np = np.asarray(cells, dtype=np.float64)
        self._cell_patch_np = np.asarray(patch_of, dtype=np.int32)

        decal_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0, has_shape_collision=False, has_particle_collision=False
        )
        self.decal_shape_start = builder.shape_count
        lift = radius + float(spec.decal_lift)
        for i, (u, a) in enumerate(cells):
            centre = wp.vec3(lift * math.cos(a), lift * math.sin(a), u)
            # +z onto the outward radial: one quarter turn about the tangent.
            quat = wp.quat_from_axis_angle(wp.vec3(-math.sin(a), math.cos(a), 0.0), math.pi / 2.0)
            builder.add_shape_box(
                self.bar_body,
                xform=wp.transform(centre, quat),
                hx=pitch / 2.0,
                hy=pitch / 2.0,
                hz=float(spec.decal_thickness),
                cfg=decal_cfg,
                color=spec.taint_color,
                label=f"{spec.name}_cell_{i:03d}",
            )

    world_aware = True
    """Every phase of this field carries a world axis.

    The field is one array of ``worlds x cells`` and both kernels take the block
    to write; the contact patch is read per world off the host bridge, which
    lays the worlds out somewhere of its own; and what is published is read off
    the world the context names.

    The decal colours are the half worth naming, because this is a *vision*
    cell. A patch that faded using another world's field would repaint the wrong
    bar, and the policy would be looking at a picture of somebody else's
    progress -- which is not a scoring bug that a rate would show, but an
    observation that is simply of the wrong episode.
    """

    @property
    def _num_worlds(self) -> int:
        """Worlds in this build. 1 when nothing replicated it."""
        return int(getattr(self.worlds, "num_worlds", 1) or 1)

    @property
    def _cells(self) -> int:
        """Cells in one world's block of the field."""
        return len(self._cell_ut_np) if self._cell_ut_np is not None else 0

    @property
    def _fleet_model(self):
        """The model that steps, which is the template unless one replicated it."""
        return self.stepping_model if self.stepping_model is not None else self.model

    def _bar_body(self, world: int) -> int:
        """This world's copy of the bar."""
        return self.worlds.body(self.bar_body, world) if self.worlds is not None else self.bar_body

    def _decal_start(self, world: int) -> int:
        """Where this world's decal shapes begin."""
        if self.worlds is None:
            return self.decal_shape_start
        return self.worlds.shape(self.decal_shape_start, world)

    # ------------------------------------------------------------------- bind
    def apply_display(self, builder: newton.ModelBuilder) -> None:
        """The decals finished as the steel they sit on. See ``decal_finish``."""
        super().apply_display(builder)
        finish = self.spec.decal_finish
        if finish is None:
            return
        for shape in range(*self.range.shapes):
            self.finishes[shape] = finish

    def bind(self, model: newton.Model) -> None:
        super().bind(model)
        spec = self.spec
        device = model.device
        n = len(self._cell_ut_np)
        worlds = self._num_worlds
        self._cell_ut = wp.array(self._cell_ut_np.astype(np.float32), dtype=wp.vec2, device=device)
        # One array, a block per world. The *cells* are shared -- one bar's
        # surface, identical in every world -- so only what has been wiped off
        # them needs the axis.
        self._intensity = wp.array(np.ones(n * worlds, dtype=np.float32), dtype=float, device=device)
        self._field_host = HostField(self._intensity.numpy)
        self._rest_intensity = np.ones(n * worlds, dtype=np.float32)
        self._sample_ut = wp.zeros(MAX_SAMPLES, dtype=wp.vec2, device=device)
        self._sample_rate = wp.zeros(MAX_SAMPLES, dtype=float, device=device)
        self._counts = wp.zeros(1, dtype=int, device=device)
        # Staged host-side and uploaded whole. Allocating a device array per
        # control step would be a fresh allocation inside the step loop, and
        # 2048 samples is 24 KB -- cheaper to send than to allocate.
        self._ut_buf = np.zeros((MAX_SAMPLES, 2), dtype=np.float32)
        self._rate_buf = np.zeros(MAX_SAMPLES, dtype=np.float32)
        self._law = wp.array([float(spec.sample_reach) ** 2, float(spec.bar_radius)], dtype=float, device=device)
        self._windows = [deque(maxlen=FORCE_WINDOW) for _ in range(worlds)]
        self._press_now = [0.0] * worlds
        self._slide_now = [0.0] * worlds
        self._touching = [False] * worlds
        self._assert_restated(model)
        sc = getattr(model, "shape_color", None)
        self._can_recolor = isinstance(sc, wp.array)
        if not self._can_recolor:
            print("[prep_taint] model.shape_color is not a device array; decals will not fade")

    def _assert_restated(self, model: newton.Model) -> None:
        """The bar this field measures must be the bar the spec describes.

        A radius restated 1 mm small rescales every pressure in the task by the
        ratio and leaves the numbers looking entirely reasonable. The first
        shape on the bar's body is the straight run -- the spec's own cylinder,
        built before the elbows -- which is why the check reads it.
        """
        spec = self.spec
        shape_body = model.shape_body.numpy()
        radii = model.shape_scale.numpy()
        mine = [int(s) for s in np.flatnonzero(shape_body == self.bar_body)]
        barrel = [s for s in mine if s < self.decal_shape_start]
        if not barrel:
            raise RuntimeError(f"prep_taint found no shapes on body {spec.bar_object!r}")
        built = float(radii[barrel[0]][0])
        if abs(built - float(spec.bar_radius)) > 1.0e-4:
            raise ValueError(
                f"prep_taint restates a bar radius of {spec.bar_radius * 1e3:.1f} mm but the "
                f"tube it is measuring is {built * 1e3:.1f} mm; every cell position and every "
                "arc distance in the removal law is wrong by that ratio"
            )

    # ------------------------------------------------------------------ hooks
    @property
    def recolors_live(self) -> bool:
        return self._can_recolor

    def contribute_hooks(self, hooks) -> None:
        """One stage, in ``after``. See the module docstring for why only one."""
        hooks.add("after", self._wipe, name="prep-taint-wipe", cadence=1)

    def on_scene_ready(self, pipeline) -> None:
        """Find the co-simulation bridge, and refuse to run without it."""
        program = getattr(pipeline, "program", None)
        island = None
        island_of = getattr(program, "island_of", None)
        if callable(island_of):
            try:
                island = island_of(self.spec.sponge)
            except (KeyError, ValueError):
                island = None
        if island is not None:
            solver = program.solver
            lookup = getattr(solver, "solver", None)
            inner = lookup(island.name) if callable(lookup) else solver
            self._bridge = getattr(inner, "bridge", None)
        if self._bridge is None:
            raise RuntimeError(
                f"prep_taint needs {self.spec.sponge!r} on a host-engine island to read its "
                "contact patch from; the arrangement must route the 'soft' domain to the "
                "superdex backend. On a Newton solver the sponge has no contact records at all "
                "and every wipe would measure exactly zero."
            )
        # The sponge's particle range, off the bridge's own actor table: the
        # slice of `particle_q` the tick writes back each control step, which
        # is what `sponge_speed` is read from.
        entry = next((e for e in getattr(self._bridge, "actors", ()) if e.get("name") == self.spec.sponge), None)
        if entry is not None:
            start = int(entry["start"])
            self._sponge_range = (start, start + int(entry["count"]))
        self._rest_intensity = self._intensity.numpy().copy()

    def _track_sponge(self, state, dt: float) -> None:
        """The sponge's centroid speed off the positions the bridge wrote back."""
        if self._sponge_range is None or not float(dt) > 0.0:
            return
        start, stop = self._sponge_range
        q = np.asarray(state.particle_q.numpy()[start:stop], dtype=np.float64)
        if not len(q):
            return
        centroid = q.mean(axis=0)
        if self._sponge_prev is not None:
            self._sponge_speed = float(np.linalg.norm(centroid - self._sponge_prev) / float(dt))
        self._sponge_prev = centroid

    def _wipe(self, state, dt) -> None:
        """One control step of removal, for every world.

        ``dt`` is the control period; if this ever runs at a cadence above one,
        the integration must be scaled by it -- the schedule passes the step's
        own dt and not the elapsed time between firings.

        A loop rather than one fused launch, and deliberately: the expensive
        part is the *contact patch*, which the host engine answers per actor and
        therefore per world whatever this does. What is left is a few thousand
        cells against a few hundred samples, once per world -- small enough that
        fusing it would buy a launch and cost the per-world sample buffers this
        reuses.

        The one device read is hoisted out of that loop, and it is
        :class:`_LazyRows` rather than an eager ``numpy()``: a slice of a device
        array still transfers the whole array, so reading one per world is three
        copies of the same bytes and three stream syncs a control step for one
        row apiece -- and for most of an episode no world is touching its bar at
        all, which is exactly when nobody needs the poses.
        """
        # The speedometer first, off the positions this tick wrote back, so the
        # published `sponge_speed` is the carry the document thresholds.
        self._track_sponge(state, dt)
        del state
        if self._bridge is None:
            # See below: the settle runs this before the bridge is found.
            return
        body_q = _LazyRows(self._fleet_model.body_q.numpy)
        for world in range(self._num_worlds):
            self._wipe_world(body_q, dt, world)

    def _wipe_world(self, body_q, dt, world: int) -> None:
        """One world's removal. Every index below is that world's own."""
        spec = self.spec
        if self._bridge is None:
            # Not yet wired, which happens exactly once per build and is not the
            # same thing as a missing bridge. `EntitySoft.on_scene_ready` settles
            # the sponge by STEPPING THE PIPELINE, and those steps run this hook
            # -- so the first ~30 firings land before this entity's own
            # `on_scene_ready` has looked the bridge up. Nothing is being wiped
            # during a settle in any case. A sponge that is genuinely on the
            # wrong solver still fails loudly, in `on_scene_ready`.
            return
        bar_body = self._bar_body(world)
        samples = self._bridge.contact_points(self.spec.sponge, against=bar_body, world=world)
        n_raw = len(samples["pos"])
        self._press_now[world] = 0.0
        self._slide_now[world] = 0.0
        self._touching[world] = False
        if not n_raw:
            self._windows[world].append(0.0)
            return

        # World -> the bar's own frame. `body_q` is (xyz, xyzw); the inverse
        # rotation is the conjugate, applied to the offset. It is the STEPPING
        # model's array -- ``bar_body`` is that model's index. ``self.model`` is
        # the template every entity binds against, and a world's bar past the
        # first sits beyond its end, which is an IndexError the moment that
        # world's sponge first touches its bar and not before.
        pose = body_q[bar_body]
        origin = np.asarray(pose[:3], dtype=np.float64)
        rot = _quat_matrix(np.asarray(pose[3:7], dtype=np.float64))
        local = (samples["pos"] - origin) @ rot  # rot^T applied, as a right-multiply

        radial = np.hypot(local[:, 0], local[:, 1])
        on_barrel = np.abs(radial - float(spec.bar_radius)) <= float(spec.surface_tol)

        area = np.maximum(samples["area"], 1.0e-9)
        force = np.linalg.norm(samples["force"], axis=1)
        pressure = force / area

        # Sliding, and only sliding: the component of the relative velocity in
        # the contact plane. Pressing straight down at a kilonewton removes
        # nothing, which is what makes this a wipe and not a stamp.
        rel = samples["v_soft"] - samples["v_other"]
        normal = samples["force"] / np.maximum(force, 1.0e-12)[:, None]
        slide = np.linalg.norm(rel - normal * np.sum(rel * normal, axis=1)[:, None], axis=1)

        live = on_barrel & (pressure >= float(spec.p_min))
        self._press_now[world] = float(np.sum(force[on_barrel]))
        self._windows[world].append(self._press_now[world])
        self._touching[world] = bool(live.any())
        if not self._touching[world]:
            return
        self._slide_now[world] = float(np.max(slide[live]))

        keep = np.flatnonzero(live)
        if keep.size > MAX_SAMPLES:
            # Keep the hardest-pressing samples, and SAY SO once. Dropping the
            # tail silently would make the removal rate a function of how many
            # samples happened to fit, which is the kind of number that stays
            # plausible while it is wrong.
            if not self._overflows:
                print(
                    f"[prep_taint] {keep.size} live contact samples against a device capacity of "
                    f"{MAX_SAMPLES}; keeping the hardest-pressing ones. Raise MAX_SAMPLES, or "
                    "coarsen the sponge lattice -- the patch is finer than this field can spend."
                )
            self._overflows += 1
            keep = keep[np.argsort(-pressure[keep])[:MAX_SAMPLES]]
        ut = np.stack([local[keep, 2], np.arctan2(local[keep, 1], local[keep, 0])], axis=1)
        rate = float(spec.k_p) * pressure[keep] * slide[keep]

        self._ut_buf[: len(keep)] = ut
        self._rate_buf[: len(keep)] = rate
        self._sample_ut.assign(self._ut_buf)
        self._sample_rate.assign(self._rate_buf)
        self._counts.assign(np.array([len(keep)], dtype=np.int32))
        wp.launch(
            _wipe_kernel,
            dim=len(self._cell_ut_np),
            inputs=[
                self._cell_ut,
                self._intensity,
                self._sample_ut,
                self._sample_rate,
                self._counts,
                self._law,
                float(dt),
                int(world * self._cells),
            ],
            device=self._intensity.device,
        )
        if self._can_recolor:
            self._recolor(world)

    def _recolor(self, world: int = 0) -> None:
        """Fade one world's decals to one world's field."""
        wp.launch(
            _decal_color_kernel,
            dim=self._cells,
            inputs=[
                self._intensity,
                wp.vec3(*(float(v) for v in self.spec.taint_color)),
                wp.vec3(*(float(v) for v in self.spec.clear_color)),
                int(self._decal_start(world)),
                self._fleet_model.shape_color,
                int(world * self._cells),
            ],
            device=self._intensity.device,
        )

    # ------------------------------------------------------------------- read
    @property
    def cells_ut(self) -> np.ndarray:
        """Every cell's (axial [m], angle [rad]) on the tube, in the order :meth:`read`'s
        ``intensity`` lists them: where a policy aims what is still blue."""
        return self._cell_ut_np

    def read(self, world: int = 0, *, at: Any = None) -> dict[str, Any]:
        """One world's field, read back to the host. One device copy per instant.

        The copy is the whole array and the slice is taken on the host: it is
        one transfer either way. ``at`` is the scoring instant, and it is what
        makes a wave pay for one transfer rather than one per world -- see
        :mod:`tasks.rgbs.assets.batched`. Without it every call re-reads, which
        is what a tool asking outside the scoring path wants.
        """
        cells = self._cells
        intensity = self._field_host.rows(at)[world * cells : (world + 1) * cells]
        cleared = intensity <= float(self.spec.cleared_below)
        patches = []
        for p in range(len(self.spec.patches)):
            mask = self._cell_patch_np == p
            patches.append(float(cleared[mask].mean()) if mask.any() else 1.0)
        live = [v for v in self._windows[world] if v > 0.0]
        return {
            "intensity": intensity,
            "cleared_fraction": float(cleared.mean()) if len(cleared) else 1.0,
            "patch_fractions": patches,
            "wipe_force": float(np.mean(live)) if live else 0.0,
            "slide_speed": self._slide_now[world],
            "touching": self._touching[world],
            "sponge_speed": self._sponge_speed,
        }

    def published_states(self, snapshot) -> dict[str, Any]:
        context = getattr(snapshot, "ctx", snapshot)
        cached = self._published_at
        if cached is not None and cached[0] is context:
            return cached[1]
        state = self.read(int(getattr(context, "world", 0)), at=instant(context))
        done = sum(1 for f in state["patch_fractions"] if f >= PATCH_DONE)
        out: dict[str, Any] = {
            "cleaned_fraction": state["cleared_fraction"],
            "patches_remaining": float(len(self.spec.patches) - done),
            "wipe_force": float(state["wipe_force"]),
            "slide_speed": float(state["slide_speed"]),
            "sponge_on_bar": bool(state["touching"]),
            "sponge_speed": float(state["sponge_speed"]),
        }
        for p, frac in enumerate(state["patch_fractions"]):
            out[f"patch{p}_cleared"] = float(frac)
        self._published_at = (context, out)
        return out

    # ------------------------------------------------------------------ reset
    def reset(self, pipeline, episode, *, world: int = 0) -> None:
        """Bit-exact: the recorded rest field, no press, fresh decals.

        This world's block of the field and this world's scalars. A wave resets
        its worlds one at a time, so restoring the whole array on every call
        would put the worlds already reset back to *their* rest -- the same
        answer only because every world starts alike, and it would stop being so
        the moment one did not.
        """
        del pipeline, episode
        if self._intensity is None:
            return
        cells = self._cells
        field = self._intensity.numpy()
        field[world * cells : (world + 1) * cells] = self._rest_intensity[world * cells : (world + 1) * cells]
        self._intensity.assign(field)
        self._field_host.invalidate()
        self._counts.zero_()
        self._windows[world].clear()
        self._press_now[world] = 0.0
        self._slide_now[world] = 0.0
        self._touching[world] = False
        self._overflows = 0
        self._published_at = None
        # The bridge teleports the sponge back to its rest; the first speed
        # reading after that must not be the teleport.
        self._sponge_prev = None
        self._sponge_speed = 0.0
        if self._can_recolor:
            self._recolor(world)


def prep_taint(layout: Layout = LAYOUT) -> SpecPrepTaint:
    """The stripe along the crown, restating the layout's bar. Built after the
    bar by stage: it resolves the bar's body label."""
    return SpecPrepTaint(
        name="taint",
        bar_object="bar",
        sponge="sponge",
        bar_radius=layout.tube_radius,
        bar_half_len=layout.straight_half,
        patches=((0.0, 0.0),),
        patch_half=(layout.stripe_half, math.radians(layout.stripe_half_angle_deg)),
    )


def _quat_matrix(q: np.ndarray) -> np.ndarray:
    """The body-to-world rotation of an xyzw quaternion.

    Returned in that direction on purpose, because the caller wants the other
    one and numpy rows supply it for free: for row vectors ``v @ R`` is
    ``R.T @ v``, so right-multiplying an offset by this matrix rotates world
    into the body's frame. Writing the transpose here and left-multiplying
    would be the same arithmetic spelled twice.
    """
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
