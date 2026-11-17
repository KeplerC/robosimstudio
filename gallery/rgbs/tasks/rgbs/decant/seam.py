# SPDX-License-Identifier: Apache-2.0
"""The sealing tape: two rigid strips across the join, and an irreversible per-cell severance field on them.

``rgbs/tasks/rgbs/polish/marks.py`` re-shaped. Polish removes a mark by Preston's
law -- ``d(intensity)/dt = k_p * pressure * |v_rel|`` -- evaluated per cell, per
substep, on device, gated on a contact force the solver actually measured. A cut
is the same object with **an offset in the force term**::

    d(intact)/dt = -k_c * (press - f_pierce) * |v_draw|     for press >= f_pierce
                 =  0                                       otherwise

and that offset is the whole task. It makes the press a **threshold** rather
than a dial, so the two things the errand is about fall out of the law rather
than being scripted:

* **Under the gate the blade skates.** Not "cuts slowly" -- cuts *nothing*. A
  blade drawn the full length of the tape at a shy press leaves it sealed and
  the run reads as a clean traverse with ``cut_fraction`` 0.00. :attr:`blade_slip`
  exists to make that legible, so a trace says "skated 100 mm, cut 0.00" rather
  than going quiet.
* **Above the gate with no travel it still does nothing.** ``v_draw`` is the
  blade's speed *along its own edge*, so a stationary press severs only the one
  or two cells under the tip however hard it leans. A knife cuts by being drawn.

**The tape is geometry as well as a field, and that is new.** The carton's
flaps stand 12 mm apart so a fingertip can be put into the join
(:attr:`~.carton.SpecShippingCarton.seam_gap`), and a blade meeting a 12 mm slot
cuts nothing -- the shipped 0-gap carton was measured against exactly that
failure. So the tape is built as two RIGID 1 mm strips, one on each flap link,
each running from the flap's top face across half the gap so the two meet at
the centreline with zero gap: the blade lands on a continuous plane over the
taped 100 mm, and the press feed reads the blade against the flaps' colliders,
which the strips are. The severance cells are decals on those strips, and
``cut_fraction`` is over the tape, not over the seam.

**This module also owns what holds the flaps shut**, and that is deliberate: the
thing that knows how much tape is left is the thing severing it. The hold is a
hinge torque written into ``control.joint_f``, capped by that flap's own
remaining intact fraction. So a half-cut seam holds weakly and *can* be forced --
which the goal's ``cut-fraction`` gate then refuses to reward -- and a fully cut
one holds nothing. There is no second mechanism and no scripted release: opening
the box is what the cut leaves behind.

``control.joint_f`` on an **unactuated** revolute was the one assumption this
design rested on and it is measured, not assumed: on the built carton, 0.05 N*m
swings a flap to 99 degrees and 0.20 N*m puts it on its 110-degree stop. The
alternatives were weighed and rejected -- ``joint_target_ke`` and
``joint_friction`` are both RECAPTURE-tier (``robosimstudio/core/tuning.py``), so
neither can be continuous; a ``body_f`` wrench is a world-frame force at the COM
and needs its reaction applied to the base by hand or it propels the carton
across the mat, which is the exact quantity ``carton_shift`` exists to measure.

Visually each cell is a non-colliding decal on its strip, lerped tape-yellow ->
join-dark as it parts, by a tiny in-graph kernel, so a cut seam shows as the
dark line down the middle of the yellow the clip has. The tiled ray tracer and
the GL viewer read ``shape_color`` live, and the studio's viser page does once
:attr:`recolors_live` declares that this scene repaints itself; an offline
re-render replays no colour channel and shows the seam frozen (marks' known
limit, inherited).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import newton
import numpy as np
import warp as wp

from robosimstudio.scene.entity import Entity, label_index
from robosimstudio.scene.registry import register_entity

from ..batched import HostField, instant
from ..look import TAPE

__all__ = ["SEAL_TAPE", "EntitySealTape", "SpecSealTape"]

FORCE_WINDOW = 10
"""Control steps of press smoothing -- marks' measured answer to MuJoCo dropping
a settled contact pair for up to six consecutive steps. A mean rather than a max
because this number scales a rate rather than gating a verdict."""

CUT_BELOW = 0.05
"""A cell counts as severed once its intact value falls under this."""


# --------------------------------------------------------------------- kernels
@wp.kernel(enable_backward=False)
def _sever_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    cell_body: wp.array(dtype=int),
    cell_local: wp.array(dtype=wp.vec3),
    intact: wp.array(dtype=float),
    press: wp.array(dtype=float),
    edge_local: wp.array(dtype=wp.vec3),  # [p0, dir] in the knife's own frame
    law: wp.array(dtype=float),  # [k_c, f_pierce, kerf, edge_len]
    knife: wp.array(dtype=int),  # [knife_body]
    dt: float,
    cells_per_world: int,
    bodies_per_world: int,
):
    """One tape cell's severance. Launched unconditionally; every gate is
    device-side data, which is what keeps the substep block capturable.

    The blade edge is reconstructed here from the knife's pose rather than fed
    in from the host, because the host feed runs once per control step and this
    runs sixteen times -- 0.8 mm of stale edge per control step at the design
    draw, against a 6 mm kerf. Survivable, and not worth having when a
    ``transform_point`` costs less than the staleness.

    With the blade held tilted, the edge is a line rising away from its tip,
    and ``|perp| <= kerf`` then admits the cells within ``kerf / sin(tilt)`` of
    the tip along the tape -- about 10 mm at 35 degrees. The window travels
    with the tip; that is what a drawn blade does.

    **One launch for the whole fleet, a thread per world-cell.** The field is
    ``worlds x cells`` in world-major order, so the thread id *is* the field
    index and ``i / cells_per_world`` is the world it belongs to. The cell
    geometry, the flap it belongs to and the blade's own edge are shared -- every
    world is the same carton with the same seam and the same knife -- while the
    bodies, the press and what is left of the tape are that world's. At one
    world ``bodies_per_world`` multiplies by zero and this is the arithmetic it
    always did.
    """
    i = wp.tid()
    if intact[i] <= 0.0:
        return
    world = i / cells_per_world
    local = i - world * cells_per_world
    n = press[world]
    if n < law[1]:
        return  # THE PENETRATION GATE -- below it, nothing parts, ever
    base = world * bodies_per_world
    X_k = body_q[knife[0] + base]
    p0 = wp.transform_point(X_k, edge_local[0])
    u = wp.transform_vector(X_k, edge_local[1])
    X_f = body_q[cell_body[local] + base]
    p_w = wp.transform_point(X_f, cell_local[local])
    d = p_w - p0
    t = wp.dot(d, u)
    if t < 0.0 or t > law[3]:
        return  # off the end of the blade
    if wp.length(d - t * u) > law[2]:
        return  # outside the kerf
    q = p0 + t * u
    vk = body_qd[knife[0] + base]
    vf = body_qd[cell_body[local] + base]
    v_k = wp.spatial_top(vk) + wp.cross(wp.spatial_bottom(vk), q - wp.transform_get_translation(X_k))
    v_f = wp.spatial_top(vf) + wp.cross(wp.spatial_bottom(vf), p_w - wp.transform_get_translation(X_f))
    # Along the EDGE only. Dragging a knife sideways scrapes; drawing it along
    # its own edge cuts. A knife held at 45 degrees to the seam then cuts at
    # cos(45) -- a graded penalty, which is what an oblique cut really is.
    v_draw = wp.abs(wp.dot(v_k - v_f, u))
    intact[i] = wp.max(0.0, intact[i] - law[0] * (n - law[1]) * v_draw * dt)


@wp.kernel(enable_backward=False)
def _hold_kernel(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    intact: wp.array(dtype=float),
    cell_flap: wp.array(dtype=int),
    flap_coord: wp.array(dtype=int),
    flap_dof: wp.array(dtype=int),
    gains: wp.array(dtype=float),  # [kp, kd, tau_max, cells_per_flap]
    joint_f: wp.array(dtype=float),
    cells_per_world: int,
    coords_per_world: int,
    dofs_per_world: int,
):
    """The tape, as a hinge torque whose ceiling is what is left of it.

    Launched ``dim = worlds * 2`` every substep in ``rigid_force``. The intact
    sum is read *here* rather than passed in, so a seam that becomes
    half-severed holds half as hard on that same substep -- no host round trip,
    no recapture, and the grading is a property of the integration rather than
    of a state machine.

    A generalized force on a hinge DOF is an **internal** torque: MuJoCo applies
    it to parent and child correctly, so action and reaction are exact by
    construction and the carton cannot propel itself.

    **The sum runs over this world's cells only**, which is the whole of what
    makes this world aware: an unbounded sweep of ``intact`` would add every
    world's remaining tape into every world's hinge, so one world cutting its
    seam would slacken all of them and a wave's flaps would open together.
    ``gains[3]`` is the per-flap cell count of ONE world, for the same reason.
    """
    tid = wp.tid()
    world = tid / 2
    f = tid - world * 2
    held = float(0.0)
    offset = world * cells_per_world
    for i in range(cells_per_world):
        if cell_flap[i] == f:
            held += intact[offset + i]
    frac = held / gains[3]
    coord = flap_coord[f] + world * coords_per_world
    dof = flap_dof[f] + world * dofs_per_world
    q = joint_q[coord]  # 0 is shut, by construction
    qd = joint_qd[dof]
    cap = gains[2] * frac
    joint_f[dof] = wp.clamp(frac * (-gains[0] * q - gains[1] * qd), -cap, cap)


@wp.kernel(enable_backward=False)
def _slip_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    edge_local: wp.array(dtype=wp.vec3),
    law: wp.array(dtype=float),  # [k_c, f_pierce, kerf, edge_len, f_touch]
    knife: wp.array(dtype=int),  # [knife_body, carton_body]
    press: wp.array(dtype=float),
    slip: wp.array(dtype=float),  # [arc_len, on_flag] per world
    dt: float,
    bodies_per_world: int,
):
    """The skate, accumulated on device because the gate that defines it is.

    Arc length travelled while the blade is touching and **under** the pierce
    gate. A failed rollout has a lot of this and no cut, which is the pair of
    numbers that tells a shy press apart from a bad aim.

    One thread per world, each over its own knife, its own carton and its own
    two-float row of ``slip``. A shared accumulator would report the sum of
    every world's skating as each world's, which on a wave of failures reads as
    one policy that skated three times as far as it did.
    """
    world = wp.tid()
    row = world * 2
    n = press[world]
    if n <= law[4] or n >= law[1]:
        slip[row + 1] = 0.0
        return
    slip[row + 1] = 1.0
    base = world * bodies_per_world
    X_k = body_q[knife[0] + base]
    p0 = wp.transform_point(X_k, edge_local[0])
    u = wp.transform_vector(X_k, edge_local[1])
    mid = p0 + 0.5 * law[3] * u
    vk = body_qd[knife[0] + base]
    vc = body_qd[knife[1] + base]
    X_c = body_q[knife[1] + base]
    v_k = wp.spatial_top(vk) + wp.cross(wp.spatial_bottom(vk), mid - wp.transform_get_translation(X_k))
    v_c = wp.spatial_top(vc) + wp.cross(wp.spatial_bottom(vc), mid - wp.transform_get_translation(X_c))
    slip[row] = slip[row] + wp.abs(wp.dot(v_k - v_c, u)) * dt


@wp.kernel(enable_backward=False)
def _decal_color_kernel(
    intact: wp.array(dtype=float),
    tape: wp.vec3,
    severed: wp.vec3,
    start: int,
    shape_color: wp.array(dtype=wp.vec3),
    cells_per_world: int,
    shapes_per_world: int,
):
    """Fade every world's decals to every world's field, in one launch.

    Two strides and they are not the same number: the field strides by cells and
    the decals stride by the model's own per-world **shape** count. A world that
    showed its own cut through its neighbour's decals would be a picture of
    somebody else's episode -- which on a vision cell is the observation, not a
    scoring slip a rate would show.
    """
    i = wp.tid()
    world = i / cells_per_world
    local = i - world * cells_per_world
    t = 1.0 - intact[i]
    shape_color[start + world * shapes_per_world + local] = tape * (1.0 - t) + severed * t


SEAL_TAPE = replace(TAPE, opacity=1.0, roughness=0.32)
"""How the tape strips reflect light.

The suite's shared :data:`~..look.TAPE` is CLEAR BOPP at 0.62 opacity, which is
right for the film ``peel`` lifts and wrong here: the clip's strip is a
pigmented masking tape, so at 0.62 the kraft showed through and the strip came
back the colour it was before it was made yellow. Opaque, with a slight sheen
between matte paper and gloss film -- glossy-matte."""


# ------------------------------------------------------------------------ spec
@dataclass(frozen=True, kw_only=True)
class SpecSealTape:
    """The tape's knobs: its geometry on the flaps, and the field's law.

    Every law number below is a **seed** until ``rgbs/scripts/rgbs_decant_bench.py``
    has replaced it, and each carries the reasoning that set it so a re-fit
    knows what it is arguing against.
    """

    entity: str = "seal_tape"
    name: str = "seam"
    carton: str = "carton"
    """The carton object's name; its flap links are ``{carton}_flap_near/far``."""
    knife_object: str = "knife"
    """The cutter object's name; its blade shape is ``{knife}_blade``."""

    # --- geometry, RESTATED and then asserted against what was built --------
    # polish's discipline: a spec must build with no sibling entity in hand, and
    # measuring the built thing at bind is what stops the restatement drifting.
    # The loader fills these from `carton_frame()` and `blade_edge_local()`, so
    # there is still exactly one place each number is derived.
    flap_half: tuple[float, float, float] = (0.100, 0.0325, 0.0015)
    seam_gap: float = 0.012
    seam_len: float = 0.200
    flap_travel: float = 1.92
    """The flaps' hinge travel [rad], restated from the carton so the field can
    normalise ``flap_open`` without reaching for the carton's spec."""
    blade_p0: tuple[float, float, float] = (0.050, 0.0, -0.009)
    blade_dir: tuple[float, float, float] = (1.0, 0.0, 0.0)
    blade_len: float = 0.015
    """The cutting edge in the KNIFE's own frame [m]: start, unit direction,
    length. Read from ``knife.blade_edge_local`` by the loader."""

    # --- the strips ----------------------------------------------------------
    tape_len: float = 0.100
    """How much of the seam is taped [m], centred on it.

    The clip's strip: a short piece across the middle of the join, four grid
    squares long against a parcel eight squares long, with the join open on
    either side of it. It is also the DRAW: the cut is over the tape, and the
    routine's passes run its length and no further."""

    tape_width: float = 0.0125
    """One strip's width across the seam [m]: 6.5 mm on the flap's top face
    and 6 mm over the gap, so the two strips meet at the centreline. 25 mm of
    tape across the join is the clip's, measured against its 25 mm grid."""

    tape_thickness: float = 0.001
    """A rigid strip's thickness [m]. Not a tape's (0.15 mm): the strip is a
    collider the blade presses on across a 12 mm slot, and a box under a
    millimetre against a 0.2 mm contact gap is the ill-conditioned pair the
    blade's own thickness floor exists to avoid."""

    tape_friction: float = 0.4
    """Tape against the blade. MuJoCo takes the larger of a pair's
    coefficients, so this sits just over the knife's 0.35 and is what the draw's
    drag on the carton is."""

    cell_pitch: float = 0.004
    """Cell spacing along the tape [m].

    24 stations x 2 rows on a 100 mm strip, so ``cut_fraction`` moves in about
    2% steps -- fine enough for a 0.95 gate. Not aliasing-bound: at the design
    draw of 0.05 m/s the edge advances 0.052 mm per substep against a 12 mm
    kerf window, 240x of margin."""

    row_inset: float = 0.0015
    """Cell-row centre, inboard of the seam's CENTRELINE, on the strip [m].

    The two rows sit +-1.5 mm across the centreline. It was 3 mm, and the
    difference is what separates ``cut_fraction`` from ``cut_run``: a station
    counts as run only when BOTH its rows are parted, so a blade held ``d`` off
    the centreline stops cutting the far row at ``d = kerf - row_inset`` while
    it goes on cutting the near one for another ``2 * row_inset``. At 3 mm that
    band is 6 mm wide and a scored episode came back with a fraction of 0.604
    against a run of 0.250 -- the document's own signature for a seam nicked in
    places rather than drawn through. Measured on the arm, the blade tip's
    lateral offset from the TCP moves about 20 mm between passes as the tool
    settles in the jaws, so the routine re-measures it every pass; 1.5 mm
    halves what is left to a per-pass error.

    Not zero, which would make the two rows one and ``near_intact`` and
    ``far_intact`` the same number."""

    kerf: float = 0.007
    """How far off the seam centreline the blade still parts the tape [m].

    **Not the width of the slit a real blade opens** (that is ~0). It is the
    lateral tolerance the *technique* has: hold the knife within 7 mm of the
    seam and the tape parts, wander further and it does not -- and with
    :attr:`row_inset` at 1.5 mm, within 5.5 mm and BOTH rows part, which is
    what a travelling front needs. Bounded below, exactly, by
    ``row_inset + cell_pitch/2`` = 3.5 mm -- at anything less a blade held
    perfectly on the centreline cannot reach a cell row's outer edge and
    ``cut_fraction`` could never reach 1. Asserted at build."""

    f_pierce: float = 2.0
    """The penetration gate [N]. **Bracketed from above by what the hand can
    deliver through the tool, which is a measurement.**

    The floor is the easy half: above an incidental brush (polish measures a
    hand merely near a part at hundredths of a newton) and 5x the cutter's own
    weight resting on the seam (0.060 kg -> 0.59 N), so a knife *laid down* on
    the tape severs nothing and the law needs no special case for it.

    The ceiling is what set the number. A hand holding a 100 mm tool by its
    middle can press the tape only until the tool's own nose corner lands on
    it, and that stand-off is ``knife.tip_lead`` -- 11.2 mm at the 50-degree
    hold. Measured on the arm, seating a millimetre at a time and reading the
    blade against the flaps and the BODY against the flaps separately:

        tip below the tape [mm]   -2.8  -4.7  -5.4  -7.1  -8.7  -9.1  -9.5
        blade [N]                  0.0   0.0   1.8   2.8   4.7   5.2   4.8
        body [N]                   0.0   0.0   0.0   0.0   0.0   0.0   1.4

    and one millimetre further the body carries 6.7, then 13.6, then 20.3 N
    while the blade reads 0.00 -- the tool is riding on its own nose and the
    knife is levered round in the jaws (its tip went from 38 mm below the TCP
    to 12 mm above it). So this grasp delivers **3-5 N through the blade** and
    nothing more, and a 6.0 N gate -- which the bench's stiff holder clears at
    1.5 mm of overdrive -- was a gate this hand could not reach, whatever the
    aim. It shipped at 6.0 and the bench had verified it sat on the knee; what
    the bench cannot measure is the grasp, which is the honest limit
    ``rgbs/scripts/rgbs_decant_bench.py`` states in its own docstring.

    Two, not three, because the pressed number is not the drawn one: the bench
    (``--cut``, pinned, with the tilted holder) reads 0.69 N at 0.5 mm of
    overdrive and cuts exactly 0.000 over 128 mm of skate, 3.50 N at 1.0 mm and
    still cuts 0.000, and 13.0 N at 1.5 mm and cuts 0.750. A gate at 3 would
    sit on top of what the arm delivers; at 2 the arm's 4-5 N is a clear 2-3 N
    of overdrive and the negative control still has a newton and a half of
    room under it. It is still 3.4x the cutter's own resting weight."""

    k_c: float = 90.0
    """Severance coefficient [1/(N*m)].

    Analytic single-pass value: a cell sits inside the kerf window for
    ``2*kerf`` = 12 mm of draw, which at 0.05 m/s is 0.24 s, so taking intact
    1 -> 0 at an overdrive of ``d`` newtons needs ``1/(d*0.05*0.24)`` = ``83/d``.
    At 90 a press **one newton over the gate** cuts a cell through in one pass,
    which is the margin the arm has (it delivers 4-5 N against a 3 N gate).
    Was 40 against a 6 N gate, where the same statement needed 2 N of
    overdrive. The binding constraint stays the **gate** rather than the rate:
    a knife either pierces or it does not, and the honest model makes the force
    axis a threshold rather than a dial."""

    f_touch: float = 0.3
    """Contact floor for :attr:`blade_slip` [N]: under it the blade is not on the
    tape at all and its travel is not a skate, it is a flight."""

    hold_kp: float = 120.0
    """Tape's hinge stiffness [N*m/rad]. SEED.

    Stability-bounded: an explicit generalized force has ``kp < I_eff/dt^2``,
    and with the flap's 5.0e-4 armature at dt = 1/960 that ceiling is 465. At
    120 the clamp bites at ``tau_max/kp`` = 0.05 rad, so an intact flap sags
    3 degrees under a hard pull and then holds -- which is what tape does."""

    hold_kd: float = 0.35
    """Critical for the armature above: ``2*sqrt(kp*I)`` = 0.348. Bound is
    ``I/dt`` = 0.48."""

    hold_tau_max: float = 6.0
    """What a fully-taped flap can resist [N*m]. MEASURED on the shipped
    carton, re-verified on this one by ``tests/test_rgbs_decant_cut.py``.

    Sized against the arm, not against gravity: gravity's closing torque on a
    10 g flap is 0.0032 N*m and 0.05 N*m alone swings one to its stop, so the
    hold exists entirely to refuse an *arm*.

    Swept on the shipped carton -- an opening torque about each hinge, 90
    control steps, against the field held at a fixed level (``--hold``):

        tau [N*m]   at the flap edge   intact 1.0   0.5    0.2    0.0
          0.05           0.7 N            0.0deg  0.0    0.1    110
          0.50           6.7 N            0.2     0.5    1.2    110
          1.50          20.0 N            0.7     1.4    110    110
          3.00          40.0 N            1.4     2.9    110    110
          6.00          80.0 N            4.4     110    110    110
          9.00         120.0 N            110     110    110    110

    Three things to read there, and all three are the task. A **sealed** carton
    needs 120 N at the flap edge against the ~40 N a YAM hand makes, so it
    cannot be forced and the knife is the only way in. A **20%-cut** seam yields
    at 20 N, which an arm *can* do -- which is exactly the case the goal's
    ``cut-fraction`` gate has to refuse to reward, and why that gate is 0.95 and
    not something softer. And the sub-yield sag is 1.4 degrees at 40 N on an
    intact seam: tape gives a little and then holds, rather than being a weld.
    The lever is 65 mm now rather than 75, so the same torques read ~15%
    higher at the edge."""

    tape_color: tuple[float, float, float] = (0.92, 0.88, 0.36)
    """Intact tape. **Yellow, sampled off the clip.**

    ``sample_videos/box_decanting.mp4`` seals its carton with a short strip of
    yellow masking tape: the lit strip reads sRGB 255/252/174 (linear
    1.0/0.97/0.42; ``look.py`` records the same sample) and its shaded end
    242/227/75. This sits just under the lit sample, because the strip in the
    clip is a *lit* surface and the path tracer supplies that lighting itself.

    The colour is not decoration here: it is the only thing that says WHERE the
    draw has to go, and :meth:`EntitySealTape._recolor` blends the cells toward
    :attr:`severed_color` one by one, so a viewer reads the cut off it."""

    severed_color: tuple[float, float, float] = (0.18, 0.16, 0.15)
    """Parted tape: the dark slit the blade leaves, the colour of the open join
    beside it. In the clip this is a black line straight down the middle of the
    yellow, which is exactly what a row of cells flipping to this colour draws."""

    cell_half_width: float = 0.002
    """Half the DRAWN width of a cell across the seam [m], anchored on the
    centreline: the two rows together draw a 4 mm dark line down the middle of
    the tape as they part. Appearance only, and deliberately decoupled from
    :attr:`row_inset`: a cell is a **sample of the cut law** evaluated at its
    centre against :attr:`kerf`; nothing in it reads the decal's extent."""

    decal_lift: float = 2.0e-4


# ---------------------------------------------------------------------- entity
@register_entity("seal_tape")
class EntitySealTape(Entity):
    """Two strips on the flaps, decal cells on the strips, one severance kernel, one hold, one feed."""

    spec: SpecSealTape

    def __init__(self, spec: SpecSealTape, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.carton_body: int = -1
        self.knife_body: int = -1
        self.flap_bodies: list[int] = []
        self.flap_coords: list[int] = []
        self.flap_dofs: list[int] = []
        self.strip_shapes: list[int] = []
        self.decal_shape_start: int = -1
        self._cell_local_np: np.ndarray | None = None
        self._cell_flap_np: np.ndarray | None = None
        self._cell_station_np: np.ndarray | None = None
        self._n_stations: int = 0
        self._cell_local = None
        self._cell_body = None
        self._cell_flap = None
        self._intact = None
        self._press = None
        self._slip = None
        self._law = None
        self._slip_law = None
        self._gains = None
        self._knife_pair = None
        self._knife_only = None
        self._edge_local = None
        self._flap_coord = None
        self._flap_dof = None
        self._rest_intact: np.ndarray | None = None
        self._blade_shapes: list[int] = []
        self._flap_shapes: list[int] = []
        self._probe = None
        self._joint_f = None
        self._windows: list[deque[float]] = []
        """One press-smoothing window per world, sized in :meth:`bind`."""
        self._press_now: list[float] = [0.0]
        self._press_buf: np.ndarray | None = None
        """Host staging for :attr:`_press`, so a per-step upload is one ``assign``
        of an array that already exists rather than a fresh one."""
        self._can_recolor: bool = False
        self._rest_xy: list[np.ndarray] = [np.zeros(2)]
        self._rest_yaw: list[float] = [0.0]
        self._published_at: tuple[Any, dict[str, Any]] | None = None
        self._intact_host: HostField | None = None
        self._slip_host: HostField | None = None
        """The two device arrays a read needs, each brought to the host once per
        instant rather than once per world. See ``..batched``."""

    world_aware = True
    """Every phase of this seam carries a world axis.

    Five of them, and the hinge hold is the one with a failure nobody would read
    as one: the severance kernel runs a thread per world-cell against that
    world's knife and flaps, the press feed is a reading per world off that
    world's contact rows, the skate accumulator is a row per world, the decals
    fade through that world's shapes, and the hold sums **only this world's**
    remaining tape. A hold that summed the whole field would let one world's cut
    slacken every world's hinges -- three cartons falling open because one of
    them was opened, which looks exactly like a task that is easier than it is.
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
        """The stride a template body index is offset by, per world."""
        if self.worlds is not None:
            return int(self.worlds.bodies_per_world)
        return int(self.model.body_count) if self.model is not None else 0

    @property
    def _coords_per_world(self) -> int:
        if self.worlds is not None:
            return int(self.worlds.coords_per_world)
        return int(self.model.joint_coord_count) if self.model is not None else 0

    @property
    def _dofs_per_world(self) -> int:
        if self.worlds is not None:
            return int(self.worlds.dofs_per_world)
        return int(self.model.joint_dof_count) if self.model is not None else 0

    @property
    def _shapes_per_world(self) -> int:
        if self.worlds is not None:
            return int(self.worlds.shapes_per_world)
        return int(self.model.shape_count) if self.model is not None else 0

    # ------------------------------------------------------------- geometry
    def strip_frame(self) -> dict[str, float]:
        """Where a strip and its cell row sit in a FLAP's frame (local ``+y``
        toward the free edge, the same on both flaps). One derivation, read by
        :meth:`build`, the bench and the tests."""
        spec = self.spec
        free_y = 2.0 * float(spec.flap_half[1])
        centre_y = free_y + 0.5 * float(spec.seam_gap)
        flap_top = 2.0 * float(spec.flap_half[2])
        return {
            "free_y": free_y,
            "centre_y": centre_y,
            "strip_y": centre_y - 0.5 * float(spec.tape_width),
            "strip_hy": 0.5 * float(spec.tape_width),
            "strip_z": flap_top + 0.5 * float(spec.tape_thickness),
            "strip_hz": 0.5 * float(spec.tape_thickness),
            "strip_top": flap_top + float(spec.tape_thickness),
            "row_y": centre_y - float(spec.row_inset),
            "draw_y": centre_y - float(spec.cell_half_width),
        }

    # ------------------------------------------------------------------ build
    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec = self.spec
        labels = list(builder.body_label)
        self.carton_body = label_index(labels, spec.carton)
        self.flap_bodies = [label_index(labels, f"{spec.carton}_flap_{s}") for s in ("near", "far")]

        # The flap hinges, by label -- the builder carries both the labels and
        # the coordinate starts, so this needs no sibling entity.
        jl = list(builder.joint_label)
        self.flap_coords = [
            int(builder.joint_q_start[label_index(jl, f"{spec.carton}_flap_{s}")]) for s in ("near", "far")
        ]
        self.flap_dofs = [
            int(builder.joint_qd_start[label_index(jl, f"{spec.carton}_flap_{s}")]) for s in ("near", "far")
        ]

        g = self.strip_frame()
        floor = float(spec.row_inset) + 0.5 * float(spec.cell_pitch)
        if float(spec.kerf) < floor - 1.0e-9:
            raise ValueError(
                f"kerf {spec.kerf * 1e3:.1f} mm is under the {floor * 1e3:.1f} mm its own cell "
                "layout needs: a blade held exactly on the seam centreline could not reach a "
                "row's outer edge and cut_fraction could never reach 1"
            )
        if not (g["strip_y"] - g["strip_hy"] < g["row_y"] < g["centre_y"]):
            raise ValueError(
                f"the cell row at {g['row_y'] * 1e3:.1f} mm is off the strip "
                f"({(g['strip_y'] - g['strip_hy']) * 1e3:.1f}..{g['centre_y'] * 1e3:.1f} mm); "
                "row_inset must keep the law's samples on the tape"
            )
        if float(spec.tape_width) <= 0.5 * float(spec.seam_gap) + 1.0e-9:
            raise ValueError(
                f"a {spec.tape_width * 1e3:.1f} mm strip does not reach across half a "
                f"{spec.seam_gap * 1e3:.0f} mm gap; the two strips must meet at the centreline"
            )
        if float(spec.tape_len) > 2.0 * float(spec.flap_half[0]) - 1.0e-9:
            raise ValueError(f"{spec.tape_len * 1e3:.0f} mm of tape on a {spec.flap_half[0] * 2e3:.0f} mm flap")

        # --- the two strips: rigid, colliding, on their flap links ------------
        tape_cfg = builder.default_shape_cfg.copy()
        tape_cfg.density = 0.0  # the flap's declared mass and armature carry it
        tape_cfg.mu = float(spec.tape_friction)
        ident = wp.quat_identity()
        for side, link in zip(("near", "far"), self.flap_bodies, strict=True):
            self.strip_shapes.append(
                int(
                    builder.add_shape_box(
                        link,
                        xform=wp.transform(wp.vec3(0.0, float(g["strip_y"]), float(g["strip_z"])), ident),
                        hx=0.5 * float(spec.tape_len),
                        hy=float(g["strip_hy"]),
                        hz=float(g["strip_hz"]),
                        cfg=tape_cfg,
                        color=spec.tape_color,
                        label=f"{spec.name}_strip_{side}",
                    )
                )
            )

        # --- the cells: samples of the law, and the slit they draw ------------
        pitch = float(spec.cell_pitch)
        n = int(math.floor((float(spec.tape_len) - pitch) / pitch))
        self._n_stations = n
        xs = (np.arange(n) - 0.5 * (n - 1)) * pitch
        z = float(g["strip_top"]) + float(spec.decal_lift)
        cells: list[tuple[float, float, float]] = []
        of_flap: list[int] = []
        station: list[int] = []
        # Station-major so the decal shapes are one contiguous run even though
        # `cell_body` alternates: `builder.shape_count` increments per call
        # regardless of which body a shape lands on.
        for s, x in enumerate(xs):
            for flap in (0, 1):
                cells.append((float(x), float(g["row_y"]), z))
                of_flap.append(flap)
                station.append(s)
        self._cell_local_np = np.asarray(cells, dtype=np.float64)
        self._cell_flap_np = np.asarray(of_flap, dtype=np.int32)
        self._cell_station_np = np.asarray(station, dtype=np.int32)

        decal = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        self.decal_shape_start = builder.shape_count
        for i, (x, _y_law, zc) in enumerate(cells):
            builder.add_shape_box(
                self.flap_bodies[of_flap[i]],
                xform=wp.transform(wp.vec3(float(x), float(g["draw_y"]), float(zc)), ident),
                hx=pitch / 2.0,
                hy=float(spec.cell_half_width),
                hz=1.0e-4,
                cfg=decal,
                color=spec.tape_color,
                label=f"{spec.name}_cell_{i:03d}",
            )
        print(
            f"[seal_tape] {len(cells)} cells over {n} stations on {spec.tape_len * 1e3:.0f} mm of tape "
            f"(pitch {pitch * 1e3:.0f} mm, kerf {spec.kerf * 1e3:.0f} mm), two {spec.tape_width * 1e3:.1f} mm "
            f"strips meeting over a {spec.seam_gap * 1e3:.0f} mm gap"
        )

    # ---------------------------------------------------------------- display
    def apply_display(self, builder: newton.ModelBuilder) -> None:
        """Tape's finish on the strips and the decals.

        ``finish_for`` keys off the colour a shape ends up with, and a saturated
        yellow is not a role it knows, so the strip went out matte and read as
        painted-on card. :data:`SEAL_TAPE` is the suite's own BOPP finish made
        opaque, because a *coloured* tape's carrier is pigmented and does not
        show the kraft through it the way the clear one ``peel`` models does.
        """
        super().apply_display(builder)
        for s in self.strip_shapes:
            self.finishes[s] = SEAL_TAPE
        n = 0 if self._cell_local_np is None else int(len(self._cell_local_np))
        for i in range(n):
            self.finishes[self.decal_shape_start + i] = SEAL_TAPE

    # ------------------------------------------------------------------- bind
    def bind(self, model: newton.Model) -> None:
        super().bind(model)
        spec = self.spec
        device = model.device
        labels = list(model.body_label)
        self.knife_body = label_index(labels, spec.knife_object)
        n = len(self._cell_local_np)

        self._cell_local = wp.array(self._cell_local_np.astype(np.float32), dtype=wp.vec3, device=device)
        self._cell_body = wp.array(
            np.asarray([self.flap_bodies[f] for f in self._cell_flap_np], dtype=np.int32),
            dtype=int,
            device=device,
        )
        self._cell_flap = wp.array(self._cell_flap_np, dtype=int, device=device)
        worlds = self._num_worlds
        # The cell *geometry* is shared -- one carton, one seam, the same strip
        # of tape drawn on it in every world -- so only what is left of it, the
        # press that cuts it and the skate carry the world axis.
        self._intact = wp.array(np.ones(n * worlds, dtype=np.float32), dtype=float, device=device)
        self._intact_host = HostField(self._intact.numpy)
        self._rest_intact = np.ones(n * worlds, dtype=np.float32)
        self._press = wp.zeros(worlds, dtype=float, device=device)
        self._press_buf = np.zeros(worlds, dtype=np.float32)
        self._slip = wp.zeros(2 * worlds, dtype=float, device=device)
        self._slip_host = HostField(self._slip.numpy)
        self._windows = [deque(maxlen=FORCE_WINDOW) for _ in range(worlds)]
        self._press_now = [0.0] * worlds

        p0 = np.asarray(spec.blade_p0, dtype=np.float64)
        u = np.asarray(spec.blade_dir, dtype=np.float64)
        u = u / float(np.linalg.norm(u))
        edge_len = float(spec.blade_len)
        self._edge_local = wp.array(np.stack([p0, u]).astype(np.float32), dtype=wp.vec3, device=device)
        self._law = wp.array(
            [float(spec.k_c), float(spec.f_pierce), float(spec.kerf), float(edge_len)],
            dtype=float,
            device=device,
        )
        self._slip_law = wp.array(
            [float(spec.k_c), float(spec.f_pierce), float(spec.kerf), float(edge_len), float(spec.f_touch)],
            dtype=float,
            device=device,
        )
        self._knife_only = wp.array([self.knife_body], dtype=int, device=device)
        self._knife_pair = wp.array([self.knife_body, self.carton_body], dtype=int, device=device)
        self._gains = wp.array(
            [float(spec.hold_kp), float(spec.hold_kd), float(spec.hold_tau_max), 0.5 * float(n)],
            dtype=float,
            device=device,
        )

        self._flap_coord = wp.array(np.asarray(self.flap_coords, dtype=np.int32), dtype=int, device=device)
        self._flap_dof = wp.array(np.asarray(self.flap_dofs, dtype=np.int32), dtype=int, device=device)

        shape_body = model.shape_body.numpy()
        flags = model.shape_flags.numpy()
        collides = (flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
        shape_labels = list(model.shape_label)
        self._blade_shapes = [
            i for i, lab in enumerate(shape_labels) if lab.endswith(f"{spec.knife_object}_blade") and collides[i]
        ]
        # The flaps' colliders: the boards AND the strips on them. The strips
        # are where the blade lands over the tape, so a feed that read the
        # boards alone would report a press of exactly zero on every cut.
        self._flap_shapes = [int(s) for s in np.flatnonzero(np.isin(shape_body, self.flap_bodies)) if collides[int(s)]]
        if not self._blade_shapes or not all(s in self._flap_shapes for s in self.strip_shapes):
            raise RuntimeError(
                f"seal_tape found no colliding shapes for the press pair (blade: "
                f"{self._blade_shapes}, flaps+strips: {self._flap_shapes}). The press feed reads the "
                "BLADE against the FLAPS' colliders and nothing else -- see _feed."
            )
        self._assert_restated(model)
        sc = getattr(model, "shape_color", None)
        self._can_recolor = isinstance(sc, wp.array)

    def _assert_restated(self, model: newton.Model) -> None:
        """The restated geometry, against what was actually built.

        The spec restates the flap and the blade so a field can be built with no
        carton and no knife in hand; measuring the built ones here is what stops
        the restatement drifting. It is the assertion that would catch somebody
        resizing the carton, or swapping the knife's blade for a mesh -- both of
        which would leave every cell in the wrong place while the field went on
        reporting a plausible fraction.
        """
        scale = getattr(model, "shape_scale", None)
        types = getattr(model, "shape_type", None)
        if scale is None or types is None:
            return
        scale = scale.numpy()
        labels = list(model.shape_label)
        for want, tail in ((self.spec.flap_half, f"{self.spec.carton}_flap_near"),):
            hits = [i for i, lab in enumerate(labels) if lab.endswith(tail)]
            if not hits:
                continue
            built = np.asarray(scale[hits[0]][:3], dtype=np.float64)
            if not np.allclose(built, np.asarray(want, dtype=np.float64), atol=1.0e-4):
                raise ValueError(
                    f"seal_tape restates the flap as {np.round(np.asarray(want) * 1e3, 1)} mm "
                    f"but the one it is laying cells on is {np.round(built * 1e3, 1)} mm; every "
                    "cell is in the wrong place by that difference"
                )

    @property
    def recolors_live(self) -> bool:
        """Yes, whenever this model can be recoloured at all -- see marks'."""
        return self._can_recolor

    # ------------------------------------------------------------------ hooks
    def contribute_hooks(self, hooks) -> None:
        hooks.add("rigid_force", self._hold, name="seal-tape-hold", cadence=1)
        hooks.add("post_solve", self._sever, name="seal-tape-sever", cadence=1)
        hooks.add("post_solve", self._slip_step, name="seal-tape-slip", cadence=1)
        if self._can_recolor:
            hooks.add("post_solve", self._recolor, name="seal-tape-decals", cadence=1)
        hooks.add("after", self._feed, name="seal-tape-press", cadence=1)

    def _hold(self, state, dt) -> None:
        """Every world's two hinges, in one launch. See :func:`_hold_kernel`."""
        del dt
        if self._joint_f is None:
            return
        wp.launch(
            _hold_kernel,
            dim=2 * self._num_worlds,
            inputs=[
                state.joint_q,
                state.joint_qd,
                self._intact,
                self._cell_flap,
                self._flap_coord,
                self._flap_dof,
                self._gains,
                self._joint_f,
                self._cells,
                self._coords_per_world,
                self._dofs_per_world,
            ],
            device=self._fleet_model.device,
        )

    def _sever(self, state, dt) -> None:
        """Every world's cells, in one launch. See :func:`_sever_kernel`."""
        wp.launch(
            _sever_kernel,
            dim=self._num_worlds * self._cells,
            inputs=[
                state.body_q,
                state.body_qd,
                self._cell_body,
                self._cell_local,
                self._intact,
                self._press,
                self._edge_local,
                self._law,
                self._knife_only,
                float(dt),
                self._cells,
                self._bodies_per_world,
            ],
            device=self._fleet_model.device,
        )

    def _slip_step(self, state, dt) -> None:
        """One thread per world. See :func:`_slip_kernel`."""
        wp.launch(
            _slip_kernel,
            dim=self._num_worlds,
            inputs=[
                state.body_q,
                state.body_qd,
                self._edge_local,
                self._slip_law,
                self._knife_pair,
                self._press,
                self._slip,
                float(dt),
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
                self._intact,
                wp.vec3(*self.spec.tape_color),
                wp.vec3(*self.spec.severed_color),
                # The model that steps, not the template every entity bound
                # against: the first decal plus a world's stride is an index into
                # the fleet's shapes, and the template's array ends after
                # world 0's.
                int(self.decal_shape_start),
                self._fleet_model.shape_color,
                self._cells,
                self._shapes_per_world,
            ],
            device=self._fleet_model.device,
        )

    def _feed(self, state, dt) -> None:
        """Once per control step, eagerly: solver contact force -> device row.

        **The blade alone against the flaps' colliders alone.** ``pairwise_force``
        sums impulse *vectors*, which is what made a whole-hand grasp read 0.16 N
        at 44 N a pad. A blade pressing down on the two strips has every contact
        pointing the same way, so the net vector IS the press -- but fold in the
        body, or fold in the base the support hand is pinning, and the
        cancellation is back.

        Projected on the **carton's own lid normal**, not world +z: a shoved
        carton tilts, and the component that pierces is the one normal to the
        lid. Polish's nose-down build projected on the wrong axis, read ~0, and
        no press ever opened its gate.

        One reading per world, out of **one** batched contact read and **one**
        host copy of ``body_q``. Reading either per world would be the same bytes
        transferred once per world and a stream sync apiece, every control step.
        """
        del dt
        if self._probe is None:
            return
        if self.worlds is None:
            self._probe.update(step=None)
            views = [self._probe]
        else:
            self._probe.update_all(step=None)
            views = [self._probe.view(w) for w in range(self._num_worlds)]
        body_q = state.body_q.numpy()
        for world, view in enumerate(views):
            f = view.pairwise_force(self._blade_shapes, self._flap_shapes)
            # The stepping model's index: ``carton_body`` is the template's and a
            # world past the first sits beyond the template array's end.
            quat = body_q[self._carton_body(world)][3:7]
            lid = _lid_normal(quat)
            self._windows[world].append(abs(float(np.dot(f, lid))))
            live = [v for v in self._windows[world] if v > 0.0]
            self._press_now[world] = float(np.mean(live)) if live else 0.0
            self._press_buf[world] = self._press_now[world]
        self._press.assign(self._press_buf)

    def _carton_body(self, world: int) -> int:
        """This world's copy of the carton's base body."""
        return self.worlds.body(self.carton_body, world) if self.worlds is not None else self.carton_body

    # ------------------------------------------------------------------ ready
    def on_scene_ready(self, pipeline) -> None:
        solver = getattr(getattr(pipeline, "program", None), "mujoco_solver", None)
        if solver is None:
            raise RuntimeError(
                "seal_tape needs the MuJoCo solver's contact pipeline for its press feed "
                "(arrangement must run backend 'mujoco' with use_mujoco_contacts=True)"
            )
        from robosimstudio.core.probes.contact import ContactProbe, FleetContactProbe

        # The fleet's probe when there is a fleet: it reads every world's contact
        # rows in one pass and rebases each world's shape indices to the
        # template, so the blade and flap sets resolved in ``bind`` stay the
        # right question to ask. A single-world probe would answer every world
        # with world 0's press -- and the press is what opens the pierce gate.
        dt = float(pipeline.sim_dt)
        self._probe = (
            ContactProbe(solver, self.model, dt)
            if self.worlds is None
            else FleetContactProbe(solver, self._fleet_model, dt, worlds=self.worlds)
        )
        self._joint_f = pipeline.control.joint_f
        self._rest_intact = self._intact.numpy().copy()
        # One host copy for the whole fleet, then each world's own carton out of
        # it: the rest pose every later shift and yaw is measured against.
        body_q = pipeline.state_0.body_q.numpy()
        self._rest_xy = [
            np.asarray(body_q[self._carton_body(w)][:3], dtype=np.float64)[:2].copy() for w in range(self._num_worlds)
        ]
        self._rest_yaw = [_yaw_of(body_q[self._carton_body(w)][3:7]) for w in range(self._num_worlds)]

    # ---------------------------------------------------------------- queries
    def read(self, world: int = 0, *, at: Any = None) -> dict[str, Any]:
        """One world's field, read back to the host. Two device copies per instant.

        Each copy is a whole array and the slice is taken on the host: it is one
        transfer either way. ``at`` is the scoring instant, and it is what makes
        a wave pay for two transfers rather than two per world -- see
        :mod:`tasks.rgbs.assets.batched`. Without it every call re-reads, which
        is what a tool asking outside the scoring path wants.
        """
        cells = self._cells
        intact = self._intact_host.rows(at)[world * cells : (world + 1) * cells]
        cut = intact <= CUT_BELOW
        per_flap = [
            float(intact[self._cell_flap_np == f].mean()) if (self._cell_flap_np == f).any() else 0.0 for f in (0, 1)
        ]
        # The longest CONTIGUOUS run of fully-cut stations. A seam cut in
        # scattered nicks and one cut through in a single draw score the same
        # fraction and are not the same outcome -- only the second opens.
        done = np.array([bool(cut[self._cell_station_np == s].all()) for s in range(self._n_stations)])
        best = run = 0
        for d in done:
            run = run + 1 if d else 0
            best = max(best, run)
        slip = self._slip_host.rows(at)[world * 2 : world * 2 + 2]
        return {
            "intact": intact,
            "cut_fraction": float(cut.mean()) if len(cut) else 0.0,
            "intact_fraction": float(intact.mean()) if len(intact) else 1.0,
            "cut_run": float(best) / float(max(1, self._n_stations)),
            "per_flap_intact": per_flap,
            "press_force": self._press_now[world],
            "blade_slip": float(slip[0]),
            "blade_on_seam": bool(slip[1] > 0.5),
        }

    # ------------------------------------------------------------- published
    def published_states(self, snapshot) -> dict[str, Any]:
        context = getattr(snapshot, "ctx", snapshot)
        cached = self._published_at
        if cached is not None and cached[0] is context:
            return cached[1]
        world = int(getattr(context, "world", 0))
        state = self.read(world, at=instant(context))
        out = {
            **self._carton_states(getattr(context, "body_q", None), world),
            "tape_cut_fraction": state["cut_fraction"],
            "intact_fraction": state["intact_fraction"],
            "cut_run": state["cut_run"],
            "press_force": float(state["press_force"]),
            "blade_slip": float(state["blade_slip"]),
            "blade_on_seam": bool(state["blade_on_seam"]),
            "near_intact": float(state["per_flap_intact"][0]),
            "far_intact": float(state["per_flap_intact"][1]),
            "flaps_free": float(max(state["per_flap_intact"]) <= 0.15),
        }
        self._published_at = (context, out)
        return out

    def _carton_states(self, body_q, world: int = 0) -> dict[str, float]:
        """What the carton is doing, published by the FIELD.

        ``body_q`` is the *context's*, which a wave has already sliced to the
        asking world and indexed like the template -- so the body indices this
        resolved at build stay the right ones and only the rest pose it measures
        against has to be picked per world.

        The carton is an ``ObjectSpec`` and so lands in the ``objects`` bin,
        but ``_published`` in ``robosimstudio_bench/goals/bind.py`` resolves a published state
        only out of ``snapshot.extras`` -- so an object cannot publish one at
        all, and a record that declared one would score every episode zero
        while looking perfectly correct. (The init check catches it, which is
        what it is for.) This entity is an extra, it already resolves the base
        and both flaps, and how open the flaps are is a fact about the seal as
        much as about the box. So it publishes them.

        The hinge angle is read as the angle between a flap's own +z and the
        base's -- at ``q = 0`` they are parallel by construction and they
        diverge by exactly the hinge angle. That needs no joint coordinate and
        no knowledge of which way each flap was anchored.
        """
        if body_q is None or max([self.carton_body, *self.flap_bodies], default=-1) >= len(body_q):
            return {}
        base = np.asarray(body_q[self.carton_body], dtype=np.float64)
        up = _lid_normal(base[3:7])
        travel = float(self.spec.flap_travel)
        angles = []
        for link in self.flap_bodies:
            n = _lid_normal(np.asarray(body_q[link], dtype=np.float64)[3:7])
            angles.append(float(np.arccos(np.clip(float(np.dot(n, up)), -1.0, 1.0))))
        return {
            "flap_near_open": angles[0] / travel,
            "flap_far_open": angles[1] / travel,
            "flap_open": 0.5 * (angles[0] + angles[1]) / travel,
            "carton_shift": float(np.linalg.norm(base[:2] - self._rest_xy[world])),
            "carton_yaw": _wrap(_yaw_of(base[3:7]) - self._rest_yaw[world]),
        }

    # ----------------------------------------------------------------- reset
    def reset(self, pipeline, episode, *, world: int = 0) -> None:
        """Bit-exact: this world's rest field, its press zeroed, its skate cleared.

        One world's rows, because a wave resets its worlds one at a time:
        restoring the whole field on every call would re-seal the carton of a
        world that had already been reset and started cutting. The decal repaint
        is one launch over the whole fleet, which is idempotent for the worlds it
        repaints to the colours they already have.
        """
        del episode
        if self._intact is None:
            return
        cells = self._cells
        field = self._intact.numpy()
        field[world * cells : (world + 1) * cells] = self._rest_intact[world * cells : (world + 1) * cells]
        self._intact.assign(field)
        self._intact_host.invalidate()
        self._press_buf[world] = 0.0
        self._press.assign(self._press_buf)
        skate = self._slip.numpy()
        skate[world * 2 : world * 2 + 2] = 0.0
        self._slip.assign(skate)
        self._slip_host.invalidate()
        self._windows[world].clear()
        self._press_now[world] = 0.0
        self._published_at = None
        if self._can_recolor:
            self._recolor(None, None)


def _lid_normal(quat_xyzw) -> np.ndarray:
    """The carton's own +z in world, from an xyzw quaternion."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    return np.array(
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
        dtype=np.float64,
    )


def _yaw_of(quat_xyzw) -> float:
    """Heading about +z, from an xyzw quaternion."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
