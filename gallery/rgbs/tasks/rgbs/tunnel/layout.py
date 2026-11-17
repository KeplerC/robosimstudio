# SPDX-License-Identifier: Apache-2.0
"""Where ``thread`` stands its cable box and lays its cable: the placements the records build.

The fifth suite over a chain and the first that puts one **inside** something.
Three things about it are worth knowing before reading the constants:

**The layout is a world axis, and that is a constraint rather than a taste.** A
goal region is a world-axis-aligned box offset by its target's origin, with no
rotation applied (:class:`~robosimstudio.scene.handles.RegionHandle`). A box
laid on a diagonal could not be described by the document that scores it, so the
run is +y and the box stands square to the bench. The box is a free body now,
so the region *follows* it -- ``origin_now`` re-reads the body pose -- but it
still cannot turn: a box the feed had spun would carry its regions round with
it as world-axis boxes, which is one more reason the support hand holds it.

**The tip starts inside, and that is a spawn position.** ``(:init)`` is empty:
nothing holds the cable, both arms are at the embodiment's own home, and the
cable is laid in the shape the clip's own cable has -- flat on the mat behind
the box, up the hang onto the box's top, then along the bore axis with
:data:`SEED` of it already past the entry face (:func:`cable_nodes`). It is a
**rest** shape, which is what makes it a spawn position and not a pose the
first second of the episode undoes: the channel is 100 mm up, and a rod laid
straight along its axis has 360 mm of itself in mid-air, falls, and levers the
tip back out before a hand has moved.

**What the records build**, and why each cannot be in a document: the box's
world coordinates (BDDL has no coordinates), the rod's spec with its dimensions
and its derived bend stiffness, and the friction label set the rod has to carry
so that the box it is fed through is not the one thing in the scene it cannot
slide against -- see :func:`tunnel_cable`, which is where the sharpest edge in
this family is written down.
"""

from __future__ import annotations

import itertools
import math

from robosimstudio.core.render.finish import GLOSS
from robosimstudio.scene.frames import point_from_surface
from robosimstudio.specs import SimRequirements, contacts_with

from ..look import BLACK
from ..staging import CELL, rgbs_views
from .box import BOX_HEIGHT, BOX_LENGTH, FLARE, MAT_TOP, RUN, SpecCableBox, box_frame, shape_labels
from .rod import LENGTH, RADIUS, SEGMENTS, RodSpec

__all__ = [
    "BORE_FLOOR_Z",
    "BOX_NAME",
    "BOX_SIM",
    "EXIT_CLEAR",
    "HANG_AT_PINCH",
    "MIN_TAIL",
    "MU_SURFACE",
    "RAMP_RUN",
    "SEED",
    "SETTLE_CLEAR",
    "TAIL_ON_MAT",
    "THREAD_AIM",
    "THREAD_VIEWS",
    "TUNNEL_X",
    "TUNNEL_Y",
    "cable_box",
    "cable_nodes",
    "slack",
    "tunnel_cable",
]

BOX_NAME = "box"
"""What the document calls the object. Read here rather than written twice: the
rod's friction pass finds a shape by its label's tail, and those labels are this
name with :data:`~box.PARTS` appended."""

TUNNEL_X = 0.55
"""The bore's line [m, world x].

Surface x -0.05, which puts the box's front face 16.5 mm nearer the arms at
surface -0.0665: two and a half grid squares in front of the bench's middle,
where frame 0 of the clip has it. The conduit stood on ``cable_route``'s
``WEAVE_PEG_X`` (0.52) because that x had a reach solve behind it; this box is
30 mm further out and its reach was solved for its own points instead --
``rgbs/scripts/rgbs_tunnel_bench.py reach``, and the sweep that set this: every point the
routine drives (the lid pinch and its approach, the plug grab and the lifted
draw, the feed pinch on the drape and the carry) solves to 0.00 mm at world x
0.52, 0.535 and 0.55, and at 0.565 the support hand's approach over the lid
comes back 18 mm short. So 0.55 is the last square the left arm reaches."""

TUNNEL_Y = -0.080
"""The centre of the lidded run [m, world y], so the entry face is at y -0.205
and the exit at +0.045; the box itself runs from -0.205 to +0.195, centred
5 mm right of the bench's centreline with its lid strip offset further right,
as the clip's inset shows it (the box's centre 7 mm right of the frame's, the
strip's 40-50 mm right of the box's).

**Placed by the rod's tail.** The rod spawns straight behind the entry and
its tail lands on the table; 0.44 m less the seed is 0.36 m of tail, which from
an entry at -0.205 lands at about -0.565 against a table edge at -0.65. Moving
the box right puts the tail off the table; moving it left puts the support
hand's hold at the far end of the left arm's reach (``rgbs/scripts/rgbs_tunnel_bench.py
reach`` measures it at the lid's left end, y +0.03, x 0.52: the pinch solves,
a waypoint 60 mm over it does not, 30 does)."""

SETTLE_CLEAR = 0.0005
"""The box spawns this far above the table [m] and comes to rest on it. Derived
rather than typed, like the carton's: a spawn that is not where the box comes
to rest puts the bore somewhere the routine did not measure."""

BORE_FLOOR_Z = BOX_HEIGHT - MAT_TOP + SETTLE_CLEAR
"""The bore floor's height above the work surface [m]: +100.5 mm. The box's
origin is ON the bore axis at floor level, so this is the box's spawn z, and
the rod's axis is one radius above it."""

SEED = 0.080
"""How far the tip starts inside the entry face [m].

Nearly seven capsules. Short of the middle of the run, which is what keeps the
reward's idle floor at zero -- it is read over the bore's FAR half, and this
seed does not reach it -- and far enough in that the bore, rather than the
lead-in, is what holds the tip while the rest of the cable settles. Measured on
the built scene: 74 mm in at t = 0 (the tip is a capsule centre, half a pitch
behind the rod's end), 75 mm after the 90-step settle, 69 mm after 300 -- the
seed is where the cable stays, which is the point of seeding the hang."""

RAMP_RUN = 0.130
"""How far back from the flare's lip the cable's rest shape takes to come up
from the mat to the bore [m].

**The cable is seeded BENT, and that is the whole difference between this scene
and a rod hanging in mid-air.** The channel is at the top of a 100 mm box, so a
rod laid straight along the bore axis has 360 mm of itself unsupported: it
falls the instant the episode starts, the fall levers the tip back out of the
bore (80 mm seeded, 47 in after the settle and still creeping), and it leaves
the feeding hand chasing a cable that is somewhere else by the time it arrives.
The clip's cable does not do that: it comes off the box's right end and drapes
onto the mat, which is a curve.

**130 mm because that is the curve this rod actually holds, measured rather
than designed.** At material parity the section is EI 5.6e-5 N*m^2, which is
nothing against 27.5 g/m: a 250 mm ramp -- the first attempt, and the shape a
stiffer cable would keep -- collapsed within 90 steps to a hang that is steep
near the mouth and flat on the mat 130 mm back, and dragged the tip 24 mm out
of the bore doing it. So the seed is that hang. Sampled off the settled rod:

    back from the entry   -8    24    46    64    89   111   133 mm
    above the mat        101    89    63    33    10     2     0 mm

A smoothstep over 130 mm reproduces it to a few millimetres, which is what
takes the settle's tip creep from 24 mm to :data:`SETTLE_SLIP`."""

HANG_AT_PINCH = 0.088
"""How far under the bore axis the settled cable lies where the hand pinches
it [m], 90 mm back from the entry face.

Measured on the built scene after 300 idle steps, off the seed above: the
cable leaves the mouth, falls steeply (60 degrees at its steepest, 50 mm back)
and is within 10 mm of the mat by here. Read by the arithmetic that has to know
the column's chord (``tests/test_rgbs_tunnel.py``) and by the rig; the routine
itself never uses it, because it reads the cable live -- a constant for where a
cable hangs is a constant that is wrong the moment a hand has touched it."""

TAIL_ON_MAT = 0.150
"""How much of the cable lies flat on the mat behind the ramp [m]. What is left
of :data:`~rod.LENGTH` once the seed, the lead-in and the ramp are taken;
asserted rather than fitted, so a longer ramp or a deeper seed is refused by
:func:`cable_nodes` instead of silently eating the tail. It is also what the
feeding hand has to grip: the pinch is 90 mm back from the entry, where the
cable is 10 mm above the mat and nearly flat."""

EXIT_CLEAR = 0.080
"""How far past the exit face the scoring region begins [m].

The document states this as part of ``beyond_exit``'s near edge and
``tests/test_rgbs_tunnel.py`` checks the two agree. Set from the pull's
measured length on the conduit rather than guessed: the feed alone carried
``cable.end_b`` about 34 mm past the exit, the support hand's full draw carried
it 70 mm further, and at 80 mm success needs 46 mm of the draw -- so the move
the task is named for is most of what you see at the end, and the goal is a
state the pull is needed to reach rather than one the feed passes through."""

MIN_TAIL = 0.10
"""Rod that must still be outside the entry when the tip clears the exit [m].

Not a comfort margin -- it is what a hand has left to hold. A layout that needs
more cable than it has builds, steps and scores zero for ever, which looks
exactly like a hard task; :func:`slack` is what refuses it instead."""

MU_SURFACE = 0.0011
"""The rod's friction against the table and the box, as a ``mu_surface``.

**The number this task turns on, and the arithmetic that sets it is worth
following, because at the library default the task is not merely hard -- it is
impossible, and impossible in a way that looks like a policy problem.**

Friction combines as ``sqrt(mu_a * mu_b)`` and this rod's own capsules are 60,
so ``mu_surface`` is squared-inverse in what the contact actually reads: the
library's 0.02 gives 1.10, which is right for a weave -- a routed cable has to
stay where it was laid -- and is a cable cemented into a pipe here.

A pushed rod is a column, and a column takes ``pi^2 * EI / L^2`` before it
folds. This rod's section is ``bend_ke * pitch`` = 5.6e-5 N*m^2, and the free
span between a grip and the mouth cannot come inside 90 mm -- the flare is
solid and the hand has width -- which is **68 mN**. The conduit's ledger set
its 0.0015 (mu 0.30) against 200 mm of run and 0.40 m of rod: 42 mN, 1.6x
inside the limit. This layout has 250 mm of run and 0.44 m of rod, 602 mm
engaged against 520, so the same 0.30 would ask 49 mN and cut the margin to
1.4x. 0.0011 reads 0.26 at the contact -- a PVC jacket in a plastic channel --
and asks 42 mN again: the conduit's margin, kept, at the longer run.

It sets the table and the box together, which is not a compromise but the
same claim twice: this suite has no route to retain and nothing that must stay
where it was put down, so a cable that slides in the channel should slide on
the table its tail lies on. **What it also sets is the box on the table**: the
rod's ``scene_friction`` pass rewrites every shape in the model, so the box's
foot meets the table at ``mu_base`` (2.0) against this, and a 350 g box holds
still under the feed's tens of millinewtons on that alone. See
``rgbs/scripts/rgbs_tunnel_bench.py hold`` for the measurement.

Order-of-magnitude, all of it. What settles it is the bench rig, which reads
the force the hand actually applies and the advance it actually buys."""

BOX_SIM = SimRequirements(
    domain="rigid",
    contacts=contacts_with("surface", "robot", "objects", "deformables"),
    notes=(
        "A free body in the arms' MuJoCo island, mirrored into the rod's VBD island as a "
        "mass-scaled proxy through the same channel as the fingers, two-way, so the cable feels "
        "its channel and the channel feels the cable. What the feed's tens of millinewtons do to "
        "a 350 g box is measured by rgbs/scripts/rgbs_tunnel_bench.py hold and pinned by "
        "tests/test_rgbs_tunnel_feed.py."
    ),
)
"""What the box needs of physics, stated once for the spec AND the record.

The composer reads a **spec's** ``sim`` (``interactions._requirements``), and
the document loader does not copy a record's declaration onto the spec its
factory returns -- so a declaration that lived on the record alone would be
catalogue prose while the built box composed as a bare ``rigid`` with no
backend under it. ``cable_box`` puts this on the spec; ``records.CABLE_BOX``
quotes the same object."""

THREAD_AIM = point_from_surface(CELL, (TUNNEL_X, TUNNEL_Y, CELL.surface_z + BORE_FLOOR_Z + RADIUS))
"""What the rig looks at [m, surface frame]: the rod's axis, mid-run, 97 mm up.

The centre of the lidded channel rather than the centre of the *work*, and the
two are not the same place -- the hands work the 90 mm just outside the entry,
at y -0.20, and the goal is met 300 mm the other way. Mid-run is what puts both
inside one frame: the overhead reaches y +0.31 from here and the plan view
+0.39, against a drawn connector that gets to about +0.20."""

THREAD_VIEWS = rgbs_views(THREAD_AIM)
"""The suite's rig aimed mid-run: what ``thread`` stages as
``(:cameras assets.tunnel.layout:THREAD_VIEWS)``.
"""


def cable_nodes() -> tuple[tuple[float, float, float], ...]:
    """The cable's rest shape, as ``SEGMENTS + 1`` points [m, surface frame].

    Tail first, tip last -- a rod's ``start`` is its trailing end -- and sampled
    at equal arc length so the capsule pitch is uniform and
    :data:`~rod.LENGTH` is the polyline's own arc length rather than a second
    claim about it.

    **In the cell's work-surface frame, like every other pose here, and the
    conversion is not a formality.** :data:`TUNNEL_X` and :data:`TUNNEL_Y` are
    WORLD coordinates (``cable_route``'s peg line, kept), and this cell's
    surface frame is offset from the world by its centre -- 0.6 m in x. Stated
    raw, every node lands 0.6 m beyond the bench's far edge: measured, the
    whole cable free-fell 850 mm past the table to the floor, at exactly g,
    with no contact at all, which reads as a solver that has lost the rod's
    collisions rather than as a rod in the wrong place. So each point is built
    in world coordinates and converted, exactly as the ``start`` of every rod
    in this tree is.

    Three pieces, which is what the reference clip shows: the cable lies flat
    on the mat behind the box, comes up a :data:`RAMP_RUN` ramp onto the box's
    top, and runs straight along the bore axis from the flare's lip to
    :data:`SEED` past the entry face. The ramp is a smoothstep, so it leaves
    the mat and meets the axis tangentially at both ends -- a circular arc
    would put a kink at each join, and a kink in a REST shape is a permanent
    bend the cable carries for ever.

    **This is a rest shape, so the cable holds it with no energy.** That is a
    claim about a real cable -- one pulled off a reel and laid up onto a box
    does keep a gentle set -- and it is the claim that makes the scene stand
    still at t = 0 instead of collapsing into the pose the routine then has to
    chase.
    """
    axis_z = BORE_FLOOR_Z + RADIUS
    mat_z = RADIUS
    entry_y = TUNNEL_Y - RUN / 2
    lip_y = entry_y - FLARE
    straight = SEED + FLARE  # tip to the flare's lip, on the axis

    def curve(t: float) -> tuple[float, float]:
        """One point of the designed centreline at ``t`` metres back from the tip,
        as ``(world y, height above the work surface)``."""
        if t <= straight:
            return entry_y + SEED - t, axis_z
        u = t - straight
        if u <= RAMP_RUN:
            # Smoothstep in the horizontal, so both ends are tangential.
            f = u / RAMP_RUN
            return lip_y - u, mat_z + (axis_z - mat_z) * (1.0 - f * f * (3.0 - 2.0 * f))
        return lip_y - RAMP_RUN - (u - RAMP_RUN), mat_z

    # Dense sample, cumulative arc length, then resample at the rod's own pitch.
    dense = [curve(0.0005 * i) for i in range(int((straight + RAMP_RUN + 2.0 * TAIL_ON_MAT) / 0.0005))]
    arc, total = [0.0], 0.0
    for a, b in itertools.pairwise(dense):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
        arc.append(total)
    if total < LENGTH:
        raise ValueError(
            f"the designed centreline is {total:.3f} m and the cable is {LENGTH:.3f}: lengthen TAIL_ON_MAT"
        )
    pitch = LENGTH / SEGMENTS
    points, k = [], 0
    for i in range(SEGMENTS + 1):
        want = i * pitch
        while k + 1 < len(arc) and arc[k + 1] < want:
            k += 1
        span = max(arc[k + 1] - arc[k], 1.0e-12)
        f = (want - arc[k]) / span
        y = dense[k][0] + (dense[k + 1][0] - dense[k][0]) * f
        z = dense[k][1] + (dense[k + 1][1] - dense[k][1]) * f
        points.append(point_from_surface(CELL, (TUNNEL_X, y, CELL.surface_z + z)))
    tail = LENGTH - straight - _ramp_length()
    if tail < 0.5 * TAIL_ON_MAT:
        raise ValueError(
            f"only {tail * 1e3:.0f} mm of cable reaches the mat behind the ramp, against "
            f"{TAIL_ON_MAT * 1e3:.0f} planned: shorten RAMP_RUN or SEED, or lengthen the rod"
        )
    # Tail first: a rod's `start` is its trailing end.
    return tuple(reversed(points))


def _ramp_length() -> float:
    """The smoothstep ramp's own arc length [m]."""
    steps = 400
    rise = BORE_FLOOR_Z
    total = 0.0
    # The ramp starts at the TOP, beside the bore: (0, rise), not the origin.
    prev = (0.0, rise)
    for i in range(1, steps + 1):
        f = i / steps
        point = (f * RAMP_RUN, rise * (1.0 - f * f * (3.0 - 2.0 * f)))
        total += math.hypot(point[0] - prev[0], point[1] - prev[1])
        prev = point
    return total


def slack() -> float:
    """Rod left over once the whole route is taken up [m].

    A guard, not a statistic, and the analogue of ``cable_route``'s
    ``_weave_slack``. The route is the run, plus the clearance the tip has to
    reach past the far face, plus the tail a hand still needs behind the mouth.
    """
    return LENGTH - (RUN + EXIT_CLEAR + MIN_TAIL)


def _bend() -> tuple[float, float]:
    """This rod's bend stiffness and damping [N*m, N*m*s], at material parity.

    The same derivation ``cable_route``'s ``_weave_bend`` makes, against the same
    reference, and it is a derivation rather than a constant because both suites
    vary the rod: a cable joint's bend stiffness is the section's ``EI`` over the
    pitch it spans, ``EI`` goes as the fourth power of the radius, and the shorter
    pitch multiplies back. Writing the answer down would be right for one length.

    **What this task does with the answer is the opposite of what a weave does
    with it.** A weave wants a rod that bends; a feed wants a rod that does
    *not*, because a pushed rod is a column and a column that bends has
    buckled. Parity is where this starts because it is the one figure with a
    justification behind it; if the free span outside the mouth folds, the first
    move is to shorten that span in the routine and not to stiffen the rod here.
    A rod stiffened until it feeds is a rod that would feed through anything.

    **The drape is the price of parity here.** At this EI a 10 mm rod sags
    about 30 mm in the first 90 mm off the mouth, so the routine reads the
    cable's height where it pinches rather than assuming the axis.
    """
    reference = RodSpec(scene_friction=True)
    reference_pitch = reference.length / reference.segments
    pitch = LENGTH / SEGMENTS
    scale = (RADIUS / reference.radius) ** 4 * (reference_pitch / pitch)
    return reference.bend_ke * scale, reference.bend_kd * scale


def cable_box() -> SpecCableBox:
    """The cable box, at rest on the bench, in the scene's work-surface frame.

    Its origin is on the bore axis at the centre of the run, at floor level --
    and every region in the document is measured from exactly here. Moving it
    moves the goal. The box body hangs off that origin toward -x (the front
    face), +x (the mat and the far edge), and +y (the ledge and the left end);
    see :func:`~box.box_frame`.
    """
    x, y, _ = point_from_surface(CELL, (TUNNEL_X, TUNNEL_Y, CELL.surface_z))
    return SpecCableBox(name=BOX_NAME, pose=(x, y, BORE_FLOOR_Z), sim=BOX_SIM)


def tunnel_cable() -> RodSpec:
    """The rod at t = 0: on the mat, up the ramp, into the bore, tip past the mouth.

    Spawned from a **polyline rest shape** rather than a start and a direction
    (:func:`cable_nodes`), because the channel is 100 mm up and a straight rod
    laid along its axis hangs in mid-air -- see :data:`RAMP_RUN`. The first node
    is the trailing end, the last is the tip, and :data:`SEED` is what puts that
    past the entry face.

    **The friction labels are the sharp edge in this suite, and they are here
    because nothing else could hold them.** ``EntityRod.apply_model_materials``
    runs a post-finalize pass over every shape in the *whole model* and assigns
    each one a friction by the tail of its label: a gripper gets ``mu_gripper``,
    a named surface gets ``mu_surface``, and everything else falls through to
    ``mu_base`` at 2.0. Friction combines as ``sqrt(mu_a * mu_b)`` and the rod's
    own capsules are 60, so a box shape that is not named here is one the
    cable meets at **mu ~= 10.9** -- and it does not fail, it simply never
    slides. The set comes from :func:`~box.shape_labels` rather than from a
    dozen string literals, so a shape renamed in the entity is renamed here too.
    ``"table"`` stays: the tail lies on it.
    """
    if slack() < 0.0:
        raise ValueError(
            f"thread needs {RUN + EXIT_CLEAR + MIN_TAIL:.3f} m of cable and has {LENGTH:.3f}: "
            "the route is longer than the rod, so no policy could finish it. Lengthen "
            "tunnel/rod.py's LENGTH, shorten tunnel/box.py's RUN, or bring MIN_TAIL in."
        )
    bend_ke, bend_kd = _bend()
    return RodSpec(
        scene_friction=True,  # this cell is the cable's; see RodSpec.scene_friction
        color=BLACK,
        finish=GLOSS,
        name="cable",
        entity="tunnel_cable",
        # A polyline rest shape rather than a straight line: see `cable_nodes`,
        # and `RAMP_RUN` for what a straight one did. `start`, `direction` and
        # `length` are echoes of it (RodSpec.nodes), stated for the readers
        # that want them without a polyline in hand.
        nodes=cable_nodes(),
        start=cable_nodes()[0],
        direction=(0.0, 1.0, 0.0),
        length=LENGTH,
        radius=RADIUS,
        segments=SEGMENTS,
        bend_ke=bend_ke,
        bend_kd=bend_kd,
        mu_surface=MU_SURFACE,
        surface_labels=("table", *shape_labels(BOX_NAME)),
    )


# Read once at import so a layout that has drifted off its own box refuses to
# load rather than staging a run the document cannot score.
assert box_frame()["y_left"] - box_frame()["y_right"] == BOX_LENGTH
