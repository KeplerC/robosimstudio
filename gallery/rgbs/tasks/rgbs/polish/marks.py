# SPDX-License-Identifier: Apache-2.0
"""The blue marks on the strip, as an irreversible per-cell removal field.

The peel suite's bond field, inverted: peel's per-element state breaks on
*opening* displacement because tape pressed onto its box must not unstick;
polishing is removal **by** pressing and sliding, so each mark cell loses
intensity by **Preston's law** -- the standard glass-polishing/CMP model --

    d(intensity)/dt = k_p * pressure * |v_rel|

evaluated per cell, per substep, on device, only while the cell sits inside the
spinning drum's **barrel** patch and the smoothed strip<->bit normal force is
above a press gate. Everything about the law that matters for the task falls
out rather than being scripted:

* the drum is clamped SIDEWAYS, so the patch is a band -- 10 mm along the
  spindle axis (the drum's own length), and ``2*sqrt(2*R*d - d*d)`` across it
  for a strip riding ``d`` into the drum, which measures 12-15 mm at the press
  this rig delivers. Against a mark that is 18 mm across the strip and 8 mm
  along it, that covers the mark's LENGTH and rather over half its WIDTH:
  measured, a press held at any overdrive from 1 to 6 mm clears 0.68 of a mark
  and then stops. The rest is reached by RUBBING the strip along the spindle;
* the drum's surface at that band runs across the axis -- along the strip's own
  long axis -- so its drag pulls the strip out of an end-on pinch. A stroke in
  that direction cannot be held and clears nothing; a stroke ALONG the axis is
  perpendicular to the drag and nearly free. Neither is written down here: both
  are what Coulomb friction does to this geometry;
* the gate plus the barrel window means a wrong-face press (marks down) or an
  incidental brush removes nothing;
* removal is monotone and irreversible; reset restores the recorded field
  bit-exactly (the peel discipline).

Pressure comes from the solver, not from a model: an ``after``-phase hook (the
eager, once-per-control-step slot) reads the strip<->bit pairwise contact force
off a private :class:`ContactProbe` and feeds a 10-step mean-of-nonzero into a
device scalar the captured kernel reads live -- the placement handler's
FORCE_WINDOW treatment of MuJoCo's contact dropouts, with a mean rather than a
max because this number scales a rate rather than gating a verdict. It is
projected on the **working radial**, not on the spindle axis: a barrel contact
pushes the strip along the radius, perpendicular to the axis, so the axial
projection the nose-down build used reads ~0 here and no press would ever open
the gate.

Visually, each cell is a non-colliding decal box on the strip's top face whose
``model.shape_color`` is lerped mark-blue -> strip-white as its intensity
falls, by a tiny in-graph kernel. The tiled ray tracer and the GL viewer read
``shape_color`` live; the studio's viser page repacks colours only when a scene
declares that it repaints itself, which is what :attr:`recolors_live` below is
for; an offline re-render replays no color channel and shows the marks frozen
(known limit).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import newton
import numpy as np
import warp as wp

from robosimstudio.core.render.palette import DEFORMABLE
from robosimstudio.scene.entity import Entity, label_index
from robosimstudio.scene.registry import register_entity

from ..batched import HostField, instant

__all__ = ["EntityPolishMarks", "SpecPolishMarks"]

FORCE_WINDOW = 10
"""Control steps of press smoothing -- the placement handler's measured answer
to the solver dropping a settled pair for up to six consecutive steps."""

SPOT_DONE = 0.95
"""A spot counts as finished when this fraction of its cells has cleared."""


@wp.kernel(enable_backward=False)
def _preston_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    cell_local: wp.array(dtype=wp.vec3),
    intensity: wp.array(dtype=float),
    press: wp.array(dtype=float),
    law: wp.array(dtype=float),  # [k_p, f_min, gap_tol, 1/patch_area]
    pair: wp.array(dtype=int),  # [strip_body, spindle_body]
    frame: wp.array(dtype=wp.vec3),  # [axis_point, axis_dir, work_normal]
    geom: wp.array(dtype=float),  # [bit_radius, bit_half]
    dt: float,
    cells_per_world: int,
    bodies_per_world: int,
):
    """One cell's Preston update. Launched unconditionally; every gate is
    device-side data, which is what keeps the substep block capturable.

    The contact test is done in WORLD, against the spindle axis rather than in
    the drum's own frame. The axis is a revolute to the world, so its point and
    direction never move; both the axial and the radial coordinate are then
    invariant under the drum's spin, and the test costs one dot and one length
    instead of an inverse transform per cell per substep.

    **One launch for the whole fleet, a thread per world-cell.** The field is
    ``worlds x cells`` in world-major order, so the thread id *is* the field
    index and ``i / cells_per_world`` is the world it belongs to. ``pair``
    carries the template body indices, ``bodies_per_world`` carries the world,
    and ``press`` is one reading per world -- one world pressing its strip onto
    its drum is not a reason for another world's marks to come off. At one world
    every stride multiplies by zero and this is the arithmetic it always did.

    The cell *geometry* is shared, because every world's strip is the same strip
    with the same marks drawn on it; only what is left of them is per world.
    """
    i = wp.tid()
    if intensity[i] <= 0.0:
        return
    world = i / cells_per_world
    local = i - world * cells_per_world
    n = press[world]
    if n < law[1]:
        return
    base = world * bodies_per_world
    X_s = body_q[pair[0] + base]
    X_b = body_q[pair[1] + base]
    p_w = wp.transform_point(X_s, cell_local[local])
    d = p_w - frame[0]
    axial = wp.dot(d, frame[1])
    if wp.abs(axial) > geom[1]:
        return
    radial = d - axial * frame[1]
    r = wp.length(radial)
    # In contact = within gap_tol OUTSIDE the barrel, or anywhere inside it.
    # The solver's soft contact lets a pressed strip ride *into* the drum, so a
    # window pinned to the barrel surface would reject exactly the presses that
    # polish. A wrong-face press still measures the strip's 4 mm thickness
    # outside the barrel and stays inert.
    if r > geom[0] + law[2]:
        return
    # The working side only. A cell that is inside the barrel on the far side
    # is not being polished, it is the drum passing over the strip's shadow.
    if wp.dot(radial, frame[2]) <= 0.0:
        return
    # Point velocities. Newton's body_qd is (linear-at-COM, angular), world
    # frame; both bodies here are built with COM at their origin (a box and a
    # centered prism), so the origin stands in for the COM exactly.
    vb = body_qd[pair[1] + base]
    vs = body_qd[pair[0] + base]
    v_bit = wp.spatial_top(vb) + wp.cross(wp.spatial_bottom(vb), p_w - wp.transform_get_translation(X_b))
    v_str = wp.spatial_top(vs) + wp.cross(wp.spatial_bottom(vs), p_w - wp.transform_get_translation(X_s))
    v_rel = wp.length(v_bit - v_str)
    di = law[0] * (n * law[3]) * v_rel * dt
    intensity[i] = wp.max(0.0, intensity[i] - di)


@wp.kernel(enable_backward=False)
def _decal_color_kernel(
    intensity: wp.array(dtype=float),
    base: wp.vec3,
    clear: wp.vec3,
    start: int,
    shape_color: wp.array(dtype=wp.vec3),
    cells_per_world: int,
    shapes_per_world: int,
):
    """Fade every world's decals to every world's field, in one launch.

    Two strides and they are not the same number: the field strides by cells and
    the decals stride by the model's own per-world **shape** count. A world that
    faded its own marks through its neighbour's decals would repaint the wrong
    strip, and on a vision cell that is the whole observation rather than a
    scoring slip a rate would show.
    """
    i = wp.tid()
    world = i / cells_per_world
    local = i - world * cells_per_world
    t = 1.0 - intensity[i]
    shape_color[start + world * shapes_per_world + local] = base * (1.0 - t) + clear * t


@dataclass(frozen=True, kw_only=True)
class SpecPolishMarks:
    """The mark field's knobs.

    ``spots`` are mark-spot centers on the strip's **top face, in the strip's
    own frame** [m]; each becomes an ellipse of ``cell_pitch``-sized cells. The
    geometry constants restate the machine's bit (a spec must build without
    the machine entity in hand) and are asserted against it at build.
    """

    entity: str = "polish_marks"
    name: str = "marks"
    strip_object: str = "strip"
    machine: str = "machine"
    """The machine entity's name; the spindle link is ``{machine}_spindle``."""
    spots: tuple[tuple[float, float], ...] = ((-0.033, 0.0), (0.0, 0.0), (0.033, 0.0))
    spot_half: tuple[float, float] = (0.004, 0.009)
    """Each mark's half-extents in the strip's frame [m]: along its length, then
    ACROSS its width.

    8 x 18 mm, and the second number is the one that makes the task a task. The
    drum is 10 mm long, so its contact patch spans 10 mm of the strip's width --
    a little over half the mark. The rest is reached only by traversing the
    strip along the spindle axis, which is the rub. Along the strip's LENGTH
    there is nothing to reach: the patch's band is 12-15 mm at the press this
    rig delivers, already wider than the mark's 8 mm, so rubbing that way adds
    no cells at all. It is also the direction the drum's drag runs in.

    Both halves were measured rather than chosen. Round 10 mm spots -- what this
    first shipped with -- sit entirely inside the patch at any real press
    (``rgbs/scripts/rgbs_polish_bench.py --press``: 1 mm of overdrive gives 1.7 N, 0.67 mm
    of ride-in and an 8 mm band), so a spot cleared standing still and the whole
    technique was decoration.

    The width came DOWN from 20 mm, and that is a lesson about the arms rather
    than about the drum. At 20 mm the mark's far cells sit 10 mm out against a
    reach of ``STROKE_AMP + bit_half`` = 11 mm, and on a rigid bench holder that
    is enough -- but two compliant arms do not deliver the whole commanded
    stroke, so one mark in three stalled a cell or two short and the schedule
    spent four rounds on it instead of three. That cost 3800 control steps, and
    what actually failed was not the polishing: the extra rubbing eroded the
    grip past ``GripHold``'s budget and the strip came down beside its rest.
    18 mm leaves 2 mm of reach in hand and the episode finishes in three."""
    cell_pitch: float = 0.002
    strip_half: tuple[float, float, float] = (0.075, 0.0125, 0.002)
    mark_color: tuple[float, float, float] = (0.13, 0.32, 0.80)
    clear_color: tuple[float, float, float] = DEFORMABLE
    """What a cell fades to as it is polished away: the strip's own colour.

    Stated as the palette entry rather than as a copy of it, because these two
    are not independently chosen -- a cleared cell has to become invisible
    against the strip, and a strip repainted without this following it leaves a
    ghost of the mark field on the finished surface."""
    decal_lift: float = 2.0e-4
    bit_radius: float = 0.0125
    bit_half: float = 0.005
    nominal_band: float = 0.013
    """Nominal contact-band width across the spindle axis [m].

    Preston's law scales with PRESSURE, so the measured normal force has to be
    divided by a patch area, and a barrel's true patch depends on how far the
    strip rides in. Rather than recompute it per step from a penetration this
    entity cannot see, the pressure is referred to a stated nominal patch --
    ``2*bit_half * nominal_band``, 10 x 13 mm -- and ``k_p`` is fitted against
    that. 13 mm is the band measured at the 3 mm overdrive the arms deliver
    (2.4 mm of ride-in on the 25 mm drum). Change one of the two and re-fit the
    other; ``rgbs/scripts/rgbs_polish_bench.py --press`` reports both."""
    k_p: float = 6.0e-4
    """Preston coefficient [1/(Pa*m)].

    Fitted against the press this rig actually delivers, ~3.1 N, over the nine
    out-and-back rubs the shipped schedule spends on a mark. Swept with
    ``rgbs/scripts/rgbs_polish_bench.py --rub --cycles 9 --kps 6e-5,2e-4,6e-4``, on the
    mark's own cleared fraction:

        k_p        rub ALONG the spindle    rub ACROSS it
        6.0e-05            0.94                  0.55
        2.0e-04            1.00                  0.55
        6.0e-04            1.00                  0.55

    (swept on the 20 mm mark this shipped with for one day; the across column
    reads 0.68 on the 18 mm one, and is just as flat.)

    Two things to read there. It has to clear 0.95, so 6e-5 is under-fitted --
    the cells at the far edge of a mark are only in the band for about a
    twentieth of each pass, and they run out of rubs before they run out of
    intensity. And **the cross-axis column does not move**: it sits where the
    contact band ends, not where the rate does, so the task's central claim is
    a fact about the geometry and cannot be tuned into or out of existence.
    6.0e-4 is twice over the 2.0e-4 that first reaches 1.00 on the bench, and
    the margin is spent on the arms rather than on the bench: a compliant pair
    delivers less stroke and a wobblier press than a rigid holder, and the cells
    at a mark's far edge are only in the band for about a twentieth of each
    pass. What the margin must NOT be spent on is more rounds -- see
    ``spot_half`` for the episode that failed by rubbing too long.

    History, because the number has moved a long way. At 2.2e-5, sized for a
    hypothetical 8-12 N press, a three-second press cleared nothing. 5.0e-4 was
    where the nose-down build left it, after halving the spindle and then
    losing half the press force. Turning the drum on its side changed the
    reference patch it is divided by -- a 10 x 13 mm band against the old
    face's 490 mm^2 disc -- so the fit had to be redone from scratch rather
    than scaled."""
    f_min: float = 0.8
    """Press gate [N]: below this, contact polishes nothing.

    Lowered from 2.0 when the hold gained orientation control. A six-degree
    hold carries part of the press as torque about the pinch rather than as
    force through the drum, and measured press dropped from ~4 N to 1.6-1.8 N
    -- under the old gate, so a rollout pressed correctly and polished nothing.
    0.8 N still refuses an incidental brush (a hand merely near the part reads
    hundredths of a newton) while admitting the press this rig delivers."""
    gap_tol: float = 1.0e-3
    """How far OUTSIDE the barrel a cell still counts as in contact [m].

    The strip is 4 mm thick, so a wrong-face press measures a 4 mm gap and is
    physically inert. It also adds to the band budget: the band across the
    spindle is ``2*sqrt(2*R*d - d*d)`` for a ride-in ``d``, and this window adds
    to ``d``. It does not decide whether a rub is needed -- that is the drum's
    10 mm LENGTH against the mark's 20 mm width, which no window touches -- but
    it does decide whether the mark's own 8 mm length stays covered, and a mark
    whose length stopped being covered would start rewarding a rub along the
    strip. On the 25 mm drum this window alone is worth a 9.8 mm band."""
    cleared_below: float = 0.1
    """A cell counts as cleared once its intensity falls under this."""


@register_entity("polish_marks")
class EntityPolishMarks(Entity):
    """Decal shapes on the strip, device state, one removal kernel, one feed."""

    spec: SpecPolishMarks

    def __init__(self, spec: SpecPolishMarks, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.strip_body: int = -1
        self.spindle_body: int = -1
        self.decal_shape_start: int = -1
        self._cell_local_np: np.ndarray | None = None
        self._cell_spot_np: np.ndarray | None = None
        self._cell_local: wp.array | None = None
        self._intensity: wp.array | None = None
        self._press: wp.array | None = None
        self._law: wp.array | None = None
        self._pair: wp.array | None = None
        self._frame: wp.array | None = None
        self._geom: wp.array | None = None
        self._rest_intensity: np.ndarray | None = None
        self._strip_shapes: list[int] = []
        self._bit_shapes: list[int] = []
        self._axis_point: np.ndarray | None = None
        self._axis_dir: np.ndarray | None = None
        self._work_normal: np.ndarray | None = None
        self._probe = None
        self._windows: list[deque[float]] = []
        """One press-smoothing window per world, sized in :meth:`bind`."""
        self._press_now: list[float] = [0.0]
        self._press_buf: np.ndarray | None = None
        """The host staging row for :attr:`_press`, so a per-step upload is one
        ``assign`` of an array that already exists rather than a fresh one."""
        self._can_recolor: bool = False
        self._published_at: tuple[Any, dict[str, Any]] | None = None
        self._field_host: HostField | None = None
        """The whole field on the host, read once per instant. See ``..batched``."""

    world_aware = True
    """Every phase of this field carries a world axis.

    Four of them, and none has a failure mode that looks like a failure: the
    Preston kernel runs a thread per world-cell against that world's strip and
    that world's drum, the press feed is a reading per world off that world's
    contact rows, the decals fade through that world's shapes, and what is
    published is read off the world the context names. A field that kept one
    press would polish every world at whichever world happened to be pressing
    hardest -- plausible motion, in the wrong episode.
    """

    @property
    def _num_worlds(self) -> int:
        """Worlds in this build. 1 when nothing replicated it."""
        return int(getattr(self.worlds, "num_worlds", 1) or 1)

    @property
    def _cells(self) -> int:
        """Cells in one world's block of the field."""
        return len(self._cell_local_np) if self._cell_local_np is not None else 0

    @property
    def _fleet_model(self):
        """The model that steps, which is the template unless one replicated it."""
        return self.stepping_model if self.stepping_model is not None else self.model

    @property
    def _bodies_per_world(self) -> int:
        """The stride the Preston kernel offsets a template body index by."""
        if self.worlds is not None:
            return int(self.worlds.bodies_per_world)
        return int(self.model.body_count) if self.model is not None else 0

    @property
    def _shapes_per_world(self) -> int:
        """The stride the decal kernel offsets the first decal shape by."""
        if self.worlds is not None:
            return int(self.worlds.shapes_per_world)
        return int(self.model.shape_count) if self.model is not None else 0

    # ------------------------------------------------------------------ build
    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec = self.spec
        labels = list(builder.body_label)
        self.strip_body = label_index(labels, spec.strip_object)
        self.spindle_body = label_index(labels, f"{spec.machine}_spindle")

        # Cell grid: an ellipse of pitch-sized cells per spot, on the top face.
        # Wider ACROSS the strip than along it -- see `spot_half`.
        cells: list[tuple[float, float, float]] = []
        spot_of: list[int] = []
        pitch = float(spec.cell_pitch)
        hx_spot, hy_spot = (float(v) for v in spec.spot_half)
        z = float(spec.strip_half[2]) + float(spec.decal_lift)
        nx = int(math.ceil(hx_spot / pitch))
        ny = int(math.ceil(hy_spot / pitch))
        for s, (cx, cy) in enumerate(spec.spots):
            for ix in range(-nx, nx + 1):
                for iy in range(-ny, ny + 1):
                    dx, dy = ix * pitch, iy * pitch
                    if (dx / hx_spot) ** 2 + (dy / hy_spot) ** 2 > 1.0:
                        continue
                    x, y = float(cx) + dx, float(cy) + dy
                    if abs(x) > spec.strip_half[0] - pitch or abs(y) > spec.strip_half[1] - pitch:
                        raise ValueError(f"mark spot {s} at {(cx, cy)} runs off the strip's top face")
                    cells.append((x, y, z))
                    spot_of.append(s)
        self._cell_local_np = np.asarray(cells, dtype=np.float64)
        self._cell_spot_np = np.asarray(spot_of, dtype=np.int32)

        decal_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0, has_shape_collision=False, has_particle_collision=False
        )
        self.decal_shape_start = builder.shape_count
        for i, (x, y, zc) in enumerate(cells):
            builder.add_shape_box(
                self.strip_body,
                xform=wp.transform(wp.vec3(float(x), float(y), float(zc)), wp.quat_identity()),
                hx=pitch / 2.0,
                hy=pitch / 2.0,
                hz=1.0e-4,
                cfg=decal_cfg,
                color=spec.mark_color,
                label=f"{spec.name}_cell_{i:03d}",
            )
        print(
            f"[polish_marks] {len(cells)} cells over {len(spec.spots)} spots "
            f"(pitch {pitch * 1e3:.0f} mm, {2 * hx_spot * 1e3:.0f} x {2 * hy_spot * 1e3:.0f} mm each)"
        )

    # ------------------------------------------------------------------ bind
    def bind(self, model: newton.Model) -> None:
        super().bind(model)
        spec = self.spec
        device = model.device
        n = len(self._cell_local_np)
        worlds = self._num_worlds
        # The cell *geometry* is shared -- one strip's top face, the same marks
        # drawn on it in every world -- so only what is left of them carries the
        # world axis. One array, a block per world, in world-major order.
        self._cell_local = wp.array(self._cell_local_np.astype(np.float32), dtype=wp.vec3, device=device)
        self._intensity = wp.array(np.ones(n * worlds, dtype=np.float32), dtype=float, device=device)
        self._field_host = HostField(self._intensity.numpy)
        self._rest_intensity = np.ones(n * worlds, dtype=np.float32)
        self._press = wp.zeros(worlds, dtype=float, device=device)
        self._press_buf = np.zeros(worlds, dtype=np.float32)
        self._windows = [deque(maxlen=FORCE_WINDOW) for _ in range(worlds)]
        self._press_now = [0.0] * worlds
        patch_area = 2.0 * float(spec.bit_half) * float(spec.nominal_band)
        self._law = wp.array(
            [float(spec.k_p), float(spec.f_min), float(spec.gap_tol), 1.0 / patch_area],
            dtype=float,
            device=device,
        )
        self._pair = wp.array([self.strip_body, self.spindle_body], dtype=int, device=device)
        self._geom = wp.array([float(spec.bit_radius), float(spec.bit_half)], dtype=float, device=device)

        shape_body = model.shape_body.numpy()
        flags = model.shape_flags.numpy()
        collides = (flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
        self._strip_shapes = [int(s) for s in np.flatnonzero(shape_body == self.strip_body) if collides[int(s)]]
        self._bit_shapes = [int(s) for s in np.flatnonzero(shape_body == self.spindle_body) if collides[int(s)]]
        if not self._strip_shapes or not self._bit_shapes:
            raise RuntimeError(
                f"polish_marks found no colliding shapes for the pair "
                f"(strip: {self._strip_shapes}, bit: {self._bit_shapes})"
            )
        # The spin axis never moves (a revolute to the world): read it once.
        # The prism is centred on its link origin, so the body's translation IS
        # a point on the axis.
        #
        # ``model`` is the template and ``spindle_body`` is a template index, and
        # one axis serves every world: replication superimposes the worlds at the
        # origin (``spacing=(0, 0, 0)``), so every world's drum is welded to the
        # same world-frame axis. There is nothing per world to store here, which
        # is why ``_frame`` below stays one row of three vectors.
        X = model.body_q.numpy()[self.spindle_body]
        q = X[3:7]
        # z-column of the rotation matrix, from the quaternion directly.
        x, y, zc, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        axis_dir = np.array(
            [2.0 * (x * zc + w * y), 2.0 * (y * zc - w * x), 1.0 - 2.0 * (x * x + y * y)], dtype=np.float64
        )
        axis_dir /= np.linalg.norm(axis_dir)
        axis_point = np.asarray(X[:3], dtype=np.float64)
        # The working radial: down, projected perpendicular to the axis. This is
        # the direction the drum presses the strip in, and the one the contact
        # force is measured along. Derived here rather than read off the machine
        # entity so the field still builds with no machine in hand -- the two
        # agree by construction because both come from the same spindle frame.
        down = np.array([0.0, 0.0, -1.0])
        work = down - float(np.dot(down, axis_dir)) * axis_dir
        norm = float(np.linalg.norm(work))
        if norm < 1.0e-6:
            raise RuntimeError(
                "polish_marks: the spindle axis is vertical, so the barrel has no working "
                "radial -- the machine is presenting its end face (SpecPolishMachine.tilt_deg)"
            )
        # The drum's real dimensions, off its own collider. The spec restates
        # them so that a field can be built with no machine in hand; measuring
        # the built drum is what stops the restatement from drifting, and the
        # two numbers are exactly the ones the whole band arithmetic rests on.
        source = getattr(model, "shape_source", None)
        mesh = source[self._bit_shapes[0]] if source is not None else None
        verts = getattr(mesh, "vertices", None)
        if verts is not None:
            v = np.asarray(verts, dtype=np.float64)
            built_r = float(np.max(np.linalg.norm(v[:, :2], axis=1)))
            built_h = float(np.max(np.abs(v[:, 2])))
            for what, said, built in (
                ("bit_radius", float(spec.bit_radius), built_r),
                ("bit_half", float(spec.bit_half), built_h),
            ):
                if abs(said - built) > 1.0e-4:
                    raise ValueError(
                        f"polish_marks restates {what}={said * 1e3:.1f} mm but the drum it is "
                        f"measuring is {built * 1e3:.1f} mm; the contact band and the Preston "
                        f"patch are both wrong by that ratio"
                    )
        self._axis_point = axis_point
        self._axis_dir = axis_dir
        self._work_normal = work / norm
        self._frame = wp.array(
            np.stack([axis_point, axis_dir, self._work_normal]).astype(np.float32),
            dtype=wp.vec3,
            device=device,
        )
        sc = getattr(model, "shape_color", None)
        self._can_recolor = isinstance(sc, wp.array)
        if not self._can_recolor:
            print("[polish_marks] model.shape_color is not a device array; decals will not fade")

    @property
    def recolors_live(self) -> bool:
        """Yes, whenever this model can be recoloured at all.

        Read by the studio at model swap. It is ``_can_recolor`` rather than a
        constant because a model whose ``shape_color`` is not a device array
        never gets the fading hook either, and a page told to repack colours
        that nothing writes would pay the readback for a picture that cannot
        change.
        """
        return self._can_recolor

    # ------------------------------------------------------------------ hooks
    def contribute_hooks(self, hooks) -> None:
        hooks.add("post_solve", self._remove, name="polish-marks-preston", cadence=1)
        if self._can_recolor:
            hooks.add("post_solve", self._recolor, name="polish-marks-decals", cadence=1)
        hooks.add("after", self._feed, name="polish-marks-press", cadence=1)

    def _remove(self, state, dt) -> None:
        """Every world's cells, in one launch. See :func:`_preston_kernel`."""
        wp.launch(
            _preston_kernel,
            dim=self._num_worlds * self._cells,
            inputs=[
                state.body_q,
                state.body_qd,
                self._cell_local,
                self._intensity,
                self._press,
                self._law,
                self._pair,
                self._frame,
                self._geom,
                float(dt),
                self._cells,
                self._bodies_per_world,
            ],
            device=self._fleet_model.device,
        )

    def _recolor(self, state, dt) -> None:
        del state, dt
        wp.launch(
            _decal_color_kernel,
            dim=self._num_worlds * self._cells,
            inputs=[
                self._intensity,
                wp.vec3(*self.spec.mark_color),
                wp.vec3(*self.spec.clear_color),
                int(self.decal_shape_start),
                # The model that steps, not the template every entity bound
                # against: ``decal_shape_start`` plus a world's stride is an
                # index into the fleet's shapes, and the template's array ends
                # after world 0's.
                self._fleet_model.shape_color,
                self._cells,
                self._shapes_per_world,
            ],
            device=self._fleet_model.device,
        )

    def _feed(self, state, dt) -> None:
        """Once per control step, eagerly: solver contact force -> device row.

        One reading per world, from **one** batched contact read. The probe is
        the fleet's own form, which pulls every world's contact rows across in
        the same three copies the single-world probe costs and hands each world
        its slice with the shape indices rebased to the template -- so the
        template shape sets resolved in :meth:`bind` are asked unchanged.
        """
        del state, dt
        if self._probe is None:
            return
        if self.worlds is None:
            self._probe.update(step=None)
            views = [self._probe]
        else:
            self._probe.update_all(step=None)
            views = [self._probe.view(w) for w in range(self._num_worlds)]
        for world, view in enumerate(views):
            f = view.pairwise_force(self._strip_shapes, self._bit_shapes)
            # On the WORKING RADIAL. A barrel contact pushes along the radius,
            # which is perpendicular to the spindle axis, so projecting on the
            # axis (what the nose-down build did) reads ~0 and the press gate
            # never opens.
            fn = abs(float(np.dot(f, self._work_normal)))
            self._windows[world].append(fn)
            live = [v for v in self._windows[world] if v > 0.0]
            self._press_now[world] = float(np.mean(live)) if live else 0.0
            self._press_buf[world] = self._press_now[world]
        self._press.assign(self._press_buf)

    # ------------------------------------------------------------------ ready
    def on_scene_ready(self, pipeline) -> None:
        solver = getattr(getattr(pipeline, "program", None), "mujoco_solver", None)
        if solver is None:
            raise RuntimeError(
                "polish_marks needs the MuJoCo solver's contact pipeline for its press feed "
                "(arrangement must run backend 'mujoco' with use_mujoco_contacts=True)"
            )
        from robosimstudio.core.probes.contact import ContactProbe, FleetContactProbe

        dt = float(pipeline.sim_dt)
        # The fleet's probe when there is a fleet: it reads every world's contact
        # rows in one pass and rebases each world's shape indices to the
        # template, so the sets resolved in ``bind`` stay the right question to
        # ask. A single-world probe here would answer every world with world 0's
        # press -- and a press is what opens this field's gate at all.
        self._probe = (
            ContactProbe(solver, self.model, dt)
            if self.worlds is None
            else FleetContactProbe(solver, self._fleet_model, dt, worlds=self.worlds)
        )
        self._rest_intensity = self._intensity.numpy().copy()

    # ---------------------------------------------------------------- queries
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
        spots = []
        for s in range(len(self.spec.spots)):
            in_spot = self._cell_spot_np == s
            frac = float(cleared[in_spot].mean()) if in_spot.any() else 1.0
            spots.append(frac)
        return {
            "intensity": intensity,
            "cleared_fraction": float(cleared.mean()) if len(cleared) else 1.0,
            "spot_fractions": spots,
            "press_force": self._press_now[world],
        }

    def spot_centres_world(self, body_q: np.ndarray) -> list[list[float]]:
        """Each mark spot's centroid in world coordinates.

        What a press has to put under the drum's contact band. The band is
        10 mm along the spindle axis, so a mark landing even a centimetre off
        contributes no cleared cells at all -- which is exactly how a press
        reading real force can polish nothing.
        """
        pts = self.cell_positions_world(body_q)
        out = []
        for s in range(len(self.spec.spots)):
            rows = pts[self._cell_spot_np == s]
            out.append([float(v) for v in rows.mean(axis=0)] if len(rows) else [float("nan")] * 3)
        return out

    def cell_positions_world(self, body_q: np.ndarray) -> np.ndarray:
        """Every cell's world position given the strip's current 7-pose."""
        from robosimstudio.core.math import transform_matrix

        m = transform_matrix(np.asarray(body_q, dtype=np.float64))
        pts = np.hstack([self._cell_local_np, np.ones((len(self._cell_local_np), 1))])
        return (pts @ m.T)[:, :3]

    # ------------------------------------------------------------- published
    def published_states(self, snapshot) -> dict[str, Any]:
        context = getattr(snapshot, "ctx", snapshot)
        cached = self._published_at
        if cached is not None and cached[0] is context:
            return cached[1]
        state = self.read(int(getattr(context, "world", 0)), at=instant(context))
        done = sum(1 for f in state["spot_fractions"] if f >= SPOT_DONE)
        out = {
            "polished_fraction": state["cleared_fraction"],
            "spots_remaining": float(len(self.spec.spots) - done),
            "press_force": float(state["press_force"]),
            "strip_on_bit": bool(state["press_force"] >= float(self.spec.f_min)),
        }
        for s, frac in enumerate(state["spot_fractions"]):
            out[f"spot{s}_cleared"] = float(frac)
        self._published_at = (context, out)
        return out

    # ----------------------------------------------------------------- reset
    def reset(self, pipeline, episode, *, world: int = 0) -> None:
        """Bit-exact: this world's recorded rest field, its press zeroed, its decals fresh.

        One world's rows, because a wave resets its worlds one at a time:
        restoring the whole field on every call would put the worlds already
        reset back to *their* rest -- the same answer only because every world
        starts alike, and it would stop being so the moment one did not. The
        decal repaint is one launch over the whole fleet, which is idempotent
        for the worlds it repaints to the colours they already have.
        """
        del episode
        if self._intensity is None:
            return
        cells = self._cells
        field = self._intensity.numpy()
        field[world * cells : (world + 1) * cells] = self._rest_intensity[world * cells : (world + 1) * cells]
        self._intensity.assign(field)
        self._field_host.invalidate()
        self._press_buf[world] = 0.0
        self._press.assign(self._press_buf)
        self._windows[world].clear()
        self._press_now[world] = 0.0
        self._published_at = None
        if self._can_recolor:
            self._recolor(None, None)
