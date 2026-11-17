# SPDX-License-Identifier: Apache-2.0
"""What ``rgbs``'s documents stage: five arrangements, and **one** camera rig.

The suite is nine variants that were five, and the merge is asymmetric on
purpose. Two of the three things a document stages are now shared and one is
not, and which is which is the whole argument for merging at all.

**The cell is shared, and it is now a fact rather than a coincidence.** Every
variant stands in ``bench_075``. Four of them always did; the four peel variants
stood in ``bench_0728`` -- the same table 22 mm lower -- and moved here. Nothing
in a peel layout had to be re-derived for that, because every constant it stands
on is stated in the cell's *work-surface frame* and both embodiments mount
against that frame rather than against the floor. What did have to be re-derived
is the reach, because the peel variants changed robot at the same time; see
``peel/layout.py``.

**The rig is shared, and that is the change with teeth.** Each of the five
suites aimed its own cameras at its own work, so nine variants produced nine
different observation spaces: two views here, three there, four for the tunnel,
`rgb` alone for two of them and `rgb`+`depth` for the rest. A policy trained
across the suite saw a different lens for every task it was asked to do. There
is one rig now -- :func:`rgbs_views` -- and every variant gets the same four
cameras, at the same resolution, the same fields of view and the same channels,
differing only in **where they are aimed**.

The geometry is ``polish``'s, and it is the one worth keeping because it is the
only one that was *measured*: :data:`T_OVERHEAD_BOARD` is a real overhead camera
solved against a calibration board on the real bench. The other four suites' rigs
were hand-aimed. So the rig that survives the merge is the one with a
provenance, and the hand-aimed ones are gone.

**What the shared rig costs, stated rather than glossed.** ``tunnel_route``
shipped a fourth camera, ``exit``, square onto the far mouth of the bore, and
called it "the one view that answers the question the goal asks". It is gone,
and the plan view is what replaces it: at 0.50 m over the aim and 50 degrees it
frames 0.47 x 0.62 m, which holds the whole 200 mm run, the 80 mm of exit
clearance and the drawn connector past that. ``box_decant``'s ``seam`` view --
low and along the cut line -- is likewise gone, and ``close`` is what replaces
it. Both losses are real; a per-variant camera is exactly the thing this merge
exists to remove, so neither is bought back.

**The arrangements are NOT shared, and must not be.** An arrangement is a
calibration: which solver owns which domain, at what rate, with what contact
stiffness. ``peel``'s sheet is integrated by VBD at 1/1200 s, ``surface_prep``'s
sponge is a foreign engine's on a host island, and ``tunnel_route``'s rod is a
coupled proxy. Those are not interchangeable and folding them together would not
be a merge, it would be a claim that a film and a foam sponge are the same
physics. So all five arrive here **under the names they already had**, verbatim.
A versioned name is a promise about a number somebody measured, and renaming
``polish_960hz@v1`` to ``rgbs_polish@v1`` would break every record that quotes it
while changing nothing about the physics.
"""

from __future__ import annotations

import numpy as np

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.scene.builtin.benches import BENCH_075
from robosimstudio.scene.frames import in_surface
from robosimstudio.specs import ArrangementSpec, CollisionSpec, backend, coupler
from robosimstudio_bench.taskbuild.arrangements import register_arrangement, register_camera_rig

__all__ = [
    "CELL",
    "CLOTH_VBD",
    "DECANT_PHYSICS",
    "POLISH_PHYSICS",
    "RGBS_RIG",
    "RGBS_RIG_NAME",
    "SURFACE_PREP_PHYSICS",
    "TUNNEL_PHYSICS",
    "rgbs_views",
]

CELL = BENCH_075
"""The one cell every variant stands in.

Named here and imported by the builders rather than each naming its own, which
is the one place this suite departs from the corpus's redundancy rule -- and it
may, because these are no longer five task directories that have to travel
apart. They are one, and one directory disagreeing with itself about which table
it stands on is not redundancy, it is a bug waiting.
"""

# ---------------------------------------------------------------------------
# The arrangements, one per family, under the names they were measured at.
# ---------------------------------------------------------------------------

CLOTH_VBD = register_arrangement(
    "cloth_vbd@ref",
    # peel's, verbatim. 60 Hz over 20 substeps -- a 1/1200 s dt -- with 80 VBD
    # iterations. Both numbers are stability floors rather than choices: at a
    # coarser dt the sheet tunnels through what it is stuck to, and a
    # teleoperated drag on top of that escalates to NaN. A *thinner* film than
    # the tape's makes that worse rather than better. ``nconmax`` is sized for
    # an arm reaching into the scene rather than the resting one, because an
    # overflow there is an illegal memory access rather than a dropped contact.
    #
    # Two islands, coupled one way: MuJoCo owns the arm and detects its own
    # contacts with the box or the panel; VBD owns the sheet, its bonds ride as
    # pre-solve stages and the handle pin as a post-solve one; the arm is the
    # sheet's moving boundary and never feels it.
    ArrangementSpec(
        control_hz=60.0,
        substeps=20,
        backends=(
            ("articulated", backend("mujoco", nconmax=2048, njmax=2048)),
            (
                "cloth",
                backend(
                    "vbd",
                    iterations=80,
                    particle_self_contact_radius=0.002,
                    particle_self_contact_margin=0.002,
                ),
            ),
        ),
        coupling=coupler("one_way"),
        collision=CollisionSpec(soft_contact_margin=0.008),
    ),
)

POLISH_PHYSICS = register_arrangement(
    "polish_960hz@v1",
    # The contact-rich yam rates (60 Hz x 16 substeps, dt = 1/960), with three
    # deliberate choices of its own, each probed before this was written:
    #
    # * `use_mujoco_contacts=True`: the mark field's press feed is a
    #   ContactProbe over MuJoCo's own contact table, and without this flag
    #   there is no geom table to read.
    # * `enable_multiccd=True`, load-bearing twice: the bit's convex-mesh drum
    #   against the box strip gets a multi-point face manifold instead of one
    #   rocking point, and so do the YAM's mesh fingertips on the strip.
    # * contact (6e4, 500) = 4 ms solref, the stiffest this substep admits
    #   (tool_sort's measurement; MuJoCo clamps below 2*dt). Even so the
    #   impedance is mass-scaled: a 16 g strip takes a ~1.6 kN/m contact and
    #   rides millimetres INTO the drum under a firm press -- measured ~5 mm at
    #   9 N -- which is why pressing is force-servoed rather than positioned.
    # * `impratio=10`. Two hands carry one rigid part for the whole episode
    #   against a drum whose drag is the same order as the pads' friction
    #   budget, which is precisely the regime MuJoCo's own guidance names for a
    #   large impratio. Measured on the two-handed carry: at 1 the pads' force
    #   wanders +-40% between control steps and the strip creeps across them
    #   until it leaves; at 10 the same carry holds and the set-down lands on
    #   the rest in 4 runs of 4 against 2 of 3.
    ArrangementSpec(
        control_hz=60.0,
        substeps=16,
        backends=(
            (
                "*",
                backend(
                    "mujoco",
                    use_mujoco_contacts=True,
                    nconmax=2048,
                    njmax=4096,
                    iterations=100,
                    ls_iterations=50,
                    cone="elliptic",
                    impratio=10.0,
                    enable_multiccd=True,
                ),
            ),
        ),
        collision=CollisionSpec(contact_ke=6.0e4, contact_kd=500.0, rigid_contact_max=8192, rigid_gap=2.0e-4),
    ),
)

DECANT_PHYSICS = register_arrangement(
    "box_decant_960hz@v1",
    # Numerically this is the polish arrangement with a bigger contact budget,
    # and that is a finding rather than a copy -- the same regime wants the same
    # numbers. What must not be inherited by accident is the reasoning, so the
    # two lines that are this family's own carry it:
    #
    # * `rigid_gap=2.0e-4` is the one that must not drift. The blade is 1.5 mm
    #   thick, so Newton's centimetre default would open contact six blade
    #   thicknesses out and every press number would be fiction.
    # * `nconmax` is raised over polish's 2048: the carton is a floor, four
    #   walls and two flaps against two solid arms and a knife, and a contact
    #   overflow is silently a dropped pair, which the press feed reads as zero.
    ArrangementSpec(
        control_hz=60.0,
        substeps=16,
        backends=(
            (
                "*",
                backend(
                    "mujoco",
                    use_mujoco_contacts=True,
                    nconmax=4096,
                    njmax=4096,
                    iterations=100,
                    ls_iterations=50,
                    cone="elliptic",
                    impratio=10.0,
                    enable_multiccd=True,
                ),
            ),
        ),
        collision=CollisionSpec(contact_ke=6.0e4, contact_kd=500.0, rigid_contact_max=8192, rigid_gap=2.0e-4),
    ),
)

SURFACE_PREP_PHYSICS = register_arrangement(
    "surface_prep_superdex@v1",
    # Two engines, and the split is the point: the arms, the bar and the bench
    # are Newton's, stepped through MuJoCo; the sponge is SuperDex's, on a host
    # island the composer splits out before the couplers.
    #
    # * `use_mujoco_contacts=True` is load-bearing here for a different reason
    #   than in polish. The taint field does NOT read MuJoCo's contact table --
    #   it cannot, the sponge has no geom in it -- but the `gripped` atom and
    #   any grip-force read do, and this task turns on two grasps.
    # * `enable_multiccd=True` for the box-box manifolds that carry the whole
    #   task: cube on rest pad, cube in the jaws. Note that mujoco_warp warns
    #   CYLINDER-BOX has no multi-contact support and generates one point --
    #   which is exactly why the bar rests on its cubes and its barrel meets
    #   nothing but the foam.
    # * `substeps=2` on the superdex side is the tunnelling margin: contact
    #   detection is at engine-step endpoints, and a mirror sweeping more than
    #   ~3-4 mm per step goes through. Measured the hard way -- a bar dropped
    #   64 mm onto the foam moves 9 mm per engine step and lands on the bench,
    #   having passed through the sponge without a contact.
    ArrangementSpec(
        control_hz=60.0,
        substeps=16,
        backends=(
            (
                "*",
                backend(
                    "mujoco",
                    use_mujoco_contacts=True,
                    nconmax=2048,
                    njmax=4096,
                    iterations=100,
                    ls_iterations=50,
                    cone="elliptic",
                    impratio=10.0,
                    enable_multiccd=True,
                ),
            ),
            # The foam's push on a body is capped at what moves that body
            # `feedback_dv_max` in a control step. A YAM fingertip is 71 g, so the
            # default 0.5 m/s is 2.1 N against the jaws' 8-14 N and they close
            # through the sponge; 5 m/s is 21 N (measured for
            # examples/08_demos/wipe_the_bar, the same scene).
            ("soft", backend("superdex", substeps=2, feedback_dv_max=5.0)),
        ),
        collision=CollisionSpec(contact_ke=6.0e4, contact_kd=500.0, rigid_contact_max=8192, rigid_gap=2.0e-4),
    ),
)

_TUNNEL_MUJOCO = backend(
    "mujoco",
    solver="newton",
    integrator="implicitfast",
    cone="elliptic",
    iterations=100,
    ls_iterations=20,
    # Newton detects the arm's contacts with the scenery and the composer routes
    # them to this island; MuJoCo only solves them.
    use_mujoco_contacts=False,
    njmax=1024,
    nconmax=512,
)
"""The arms' backend, and the cable box's. Bound to a name because two domains
have to share one spec OBJECT rather than two equal copies -- see the
:data:`TUNNEL_PHYSICS` entry that uses it twice."""


TUNNEL_PHYSICS = register_arrangement(
    "cable_tunnel_960hz@v1",
    # MuJoCo owns the arms, rigid VBD owns the rod, and the gripper bodies
    # mirror into the VBD side as mass-scaled kinematic proxies at 20x, because
    # MuJoCo has no rod joint and VBD should not solve an arm.
    #
    # The contact budget is the line that is this variant's own. A weave's rod
    # touches one spool at a time; a rod lying inside a four-walled bore touches
    # four shapes along its whole engaged length at once, and the buffer fails
    # by silently DROPPING contacts rather than by raising -- which reads out as
    # a cable that moved 0.1 mm. Hence 512 against the weaves' 256.
    ArrangementSpec(
        control_hz=60.0,
        substeps=16,
        backends=(
            ("articulated", _TUNNEL_MUJOCO),
            # THE SAME SPEC OBJECT, for the cable box. `thread`'s fixture became
            # a free body on 2026-09-11 -- the clip's channel is a box a hand
            # has to hold down, not a tunnel welded to the bench -- and a
            # `rigid` domain with no backend under it makes the composer refuse
            # the scene outright. It has to join the ARMS' island rather than
            # get a second MuJoCo one: the support hand pinches the box as a
            # native contact, the rod slides through it as a proxy contact
            # mirrored into VBD through the same channel that carries the
            # fingers, and a third island is what the proxy coupler refuses.
            # The island partition keys on the spec object, so the two domains
            # share one rather than being given equal copies.
            ("rigid", _TUNNEL_MUJOCO),
            (
                "rod",
                backend(
                    "vbd",
                    iterations=40,
                    rigid_avbd_beta=1.0e2,
                    rigid_contact_k_start=1.0e3,
                    rigid_contact_history=True,
                    rigid_body_contact_buffer_size=512,
                ),
            ),
        ),
        coupling=coupler(
            "proxy",
            mode="staggered",
            # 2, not the reference's 4. Measured on cable_route: the pass count
            # multiplies both solvers and their restart scaffolding, 4 passes
            # replayed a step in ~404 ms against 125 ms at 2, and the
            # physics-sensitive member of that suite got better rather than worse.
            iterations=2,
            mass_scale=20.0,
            collide_interval=1,
            contact_matching="latest",
        ),
        collision=CollisionSpec(
            rigid_gap=0.002,
            contact_matching="latest",
        ),
    ),
)

# ---------------------------------------------------------------------------
# The one rig.
# ---------------------------------------------------------------------------

T_OVERHEAD_BOARD = np.array(
    [
        [0.9999606199280351, 0.005088280314025779, -0.007271038205479379, 0.13614925374585718],
        [-0.008087350016361418, 0.8598311973225057, -0.5105143552150704, -0.09495805109200249],
        [0.003654225342326612, 0.5105530545539729, 0.8598384878119585, 0.513003668527475],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
"""The real rig's overhead camera, in the calibration board's frame [m].

Solved 2026-08-04 against a static table board by co-observation -- 100 frames,
primary solver DANIILIDIS, residual spread 1.179 mm and 0.1524 deg. Read as
roll-pitch-yaw it is 30.7009 deg about the board's x and 0.2094 / 0.4634 about
the other two: a camera hung 513 mm over the table, leaning thirty degrees off
vertical and otherwise square to the table's own axes.

**It is stated in the graphics basis** -- +x right, +y UP, -z forward -- and not
in the +z-forward, +y-down vision basis :class:`CameraSpec` stores. Nothing in
the numbers says so; the scene does. Read as a vision pose, a camera 513 mm
above a table looks 59 deg *up*, at the ceiling. So the view direction below is
the third column negated and the up axis is the second column as it stands.

**This is the only measured camera in the suite**, which is why it is the one
that survived the merge -- see the module docstring.
"""

_BOARD_TO_SURFACE = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
"""The board frame's axes onto the work surface's: a quarter turn about z.

The board lay on the real table with its +y running away from the camera, along
the direction the camera faces. In this scene that direction is **+x** -- the
camera hangs over the arms and looks out across the bench, away from them --
and pinning it is the whole transplant.

**This was the other quarter turn until 2026-09-01, and that was wrong.** The
lean and the height were right and the azimuth was 180 degrees out, which puts
the camera on the far side of the bench looking back at the robot. Three things
say so, and none of them is a matter of taste:

* Every one of the reference clips insets the real overhead camera's own
  picture ("RGBS View"), and in all five the arms come in at the *bottom* of
  that frame with the work beyond them.
* Turned the old way, all 226 of the robot's shapes project into the upper half
  of the frame and most of them above it -- the picture is of the bench with a
  robot leaning in from the back. Turned this way the arms enter from the two
  bottom corners, which is the reference framing.
* The suite's own ``close`` camera, which was measured rather than derived,
  looks along **+x**. An overhead that disagreed with it about which side of
  the bench the robot is on could not both be right.
"""

_OVERHEAD_EYE_Z = float(T_OVERHEAD_BOARD[2, 3])
"""How high the calibrated eye sits above the work surface [m]: 0.513.

The board itself does not survive the transplant -- it sat wherever it sat on
the real table and no variant here has a counterpart for it -- so the
calibration's two in-plane offsets (136 mm and -95 mm from the board's origin)
are dropped and the eye is slid along the calibrated ray until the axis lands on
whatever the variant is working on. What is reproduced is the lean, the 513 mm
height and the direction the camera faces; what is not is the board.

The consequence worth stating: the eye is not in one place across the suite. It
is wherever the calibrated ray, aimed at *this* variant's work, crosses 513 mm.
For ``strip`` that is polish's own camera to the millimetre, because polish's
aim is what the ray was transplanted onto; for a variant whose work sits
elsewhere on the bench the camera slides with it rather than framing an empty
patch of table.
"""

CLOSE_OFFSET = (-0.145, -0.280, 0.065)
"""Where ``close`` sits, relative to the aim [m, surface frame].

``polish``'s ``bit`` camera, restated as an offset instead of a pose: it sat at
world (0.45, -0.28, 0.95) looking at (0.595, 0, 0.885), and this is the
difference. Near-side, below the overhead's lean and well off to one side, which
is what keeps the working hand from standing between the lens and the contact.
"""

TOP_HEIGHT = 0.50
"""How far above the aim ``top`` sits [m]. ``polish``'s plan view, unchanged.

At 50 degrees this frames 0.47 x 0.62 m about the aim, which is the number the
module docstring leans on when it says the plan view replaces ``tunnel_route``'s
dedicated ``exit`` camera.
"""

PHOTO_EYE = (-0.62, -0.04, 0.42)
"""Where the clips' handheld phone stood [m, surface frame]: behind and above the
arms, a little toward the working (right, -y) side, looking out across the work
at the slat wall. Read off ``sample_videos/`` -- the arm bases enter the frame
at its two lower corners in every clip -- and first stated by
``rgbs/scripts/surface_cleaning.py``, whose ``photo`` camera this is."""

PHOTO_LOOK_AHEAD = 0.05
PHOTO_LOOK_Z = 0.08
"""The phone looks a little past the work and a little above the mat, so the
work sits in the lower-middle of the frame with the ledge and the slats above
it, which is the clips' composition."""

PHOTO_SIZE = (960, 540)
PHOTO_FOV_Y = 38.0
"""16:9 at a phone's main-lens vertical field of view."""

# THE "RGBS VIEW" INSET IS THE OVERHEAD, and the rig carried a second camera
# standing in for it until 2026-09-11. That camera existed because the
# overhead's *intrinsics* were unknown: at an assumed 45-degree lens the
# calibrated pose frames 0.43 m of bench where the clips' inset plainly shows
# about a metre, so the two could not be the same camera and a separate one was
# fitted to the picture -- an eye 1.08 m up over x = +0.30, at 30 degrees.
# The real stream reports 89.00 x 57.94 degrees, which frames 1.17 x 0.66 m
# from the calibrated pose, and the stand-in's whole reason for existing went
# with it. See the overhead's own intrinsics below.

CHANNELS = ("rgb", "depth")
"""What every camera in this suite produces.

The union of what the five suites asked for, not the intersection: ``peel``,
``box_decant`` and ``tunnel_route`` recorded depth and ``polish`` and
``surface_prep`` did not. Taking the union is the choice that makes the merged
observation space a superset of all five, so nothing that used to be recorded
stops being recorded -- and a policy trained on the suite gets one answer to
"what does a camera here give me" rather than nine.
"""

RGBS_RIG_NAME = "rgbs_views@v1"
"""The name every ``rgbs`` document stages.

One name, nine aims. The name resolves to :data:`RGBS_RIG` for anything that
looks it up cold, and it is what ``rgbs.suite`` stages; a variant whose work
sits elsewhere stages the same rig aimed at its own layout -- ``(:cameras
assets.peel.layout:HANDLE_VIEWS)``, a constant of the module that owns the
layout -- because a rig *aimed* at a layout is the layout's arithmetic and
not something a versioned registration can hold.
"""


def rgbs_views(aim: tuple[float, float, float]) -> tuple[CameraSpec, ...]:
    """The suite's four cameras, aimed at ``aim`` [m, surface frame].

    Args:
        aim: What the rig looks at, in :data:`CELL`'s work-surface frame -- the
            place the variant's work actually happens, which is a press, a cut,
            a bore mouth or a bond line depending on which variant is asking.

    Returns:
        ``overhead``, ``close``, ``top``, ``photo``, in that order and always
        those four.
    """
    aim_v = np.asarray(aim, dtype=np.float64)

    # The calibrated ray, turned onto this scene's axes, then walked back from
    # the aim until it reaches the height the camera was solved at.
    forward = _BOARD_TO_SURFACE @ -T_OVERHEAD_BOARD[:3, 2]
    up = _BOARD_TO_SURFACE @ T_OVERHEAD_BOARD[:3, 1]
    eye = aim_v - forward * ((aim_v[2] - _OVERHEAD_EYE_Z) / forward[2])

    def world(point) -> tuple[float, float, float]:
        """A surface-frame point as world. A cell's surface frame is a pure
        translation, so this is an offset and nothing turns."""
        return tuple(float(v) for v in in_surface(CELL, (*point, 0.0, 0.0, 0.0, 1.0))[:3])

    return (
        CameraSpec(
            name="overhead",
            # THE REAL CAMERA'S OWN INTRINSICS, not a field of view somebody
            # chose. Its colour stream at 1280 x 720 reports fx 651.2488,
            # fy 650.1978, cx 635.3629, cy 358.0204 -- an 89.00 x 57.94 degree
            # lens with its principal point 4.6 px left of centre and 2.0 px
            # above it. This shipped as `fov_y=45` at 640 x 480, which is a
            # different camera: 45 degrees sees 0.43 m across the aim plane
            # where 57.94 sees 0.66, so every framing judgement made against
            # the old number was made against a lens the rig does not have.
            #
            # Stated as `fx`/`fy`/`cx`/`cy` rather than as a field of view
            # because `CameraSpec` offers the explicit form for exactly this
            # case and a principal point is not at the image centre on any real
            # camera. What is NOT reproduced is the distortion -- the stream
            # reports inverse_brown_conrady [-0.054017, 0.061381, 0.00056,
            # 0.000577, -0.020458] and every renderer here is a pinhole, so a
            # pixel near the corner of a real frame sits a little differently
            # from its simulated twin.
            width=1280,
            height=720,
            fx=651.2488,
            fy=650.1978,
            cx=635.3629,
            cy=358.0204,
            eye=world(eye),
            look_at=world(aim_v),
            # The calibration's own up axis rather than world +z.
            # `look_at_matrix` builds the rest of the basis from whatever it is
            # handed, so passing this reproduces the solved rotation exactly
            # instead of to the 0.41 deg of roll that levelling the horizon
            # would cost.
            up=tuple(float(v) for v in up),
            near=0.01,
            far=4.0,
            channels=CHANNELS,
        ),
        CameraSpec(
            name="close",
            width=640,
            height=480,
            fov_y=35.0,
            eye=world(aim_v + np.asarray(CLOSE_OFFSET, dtype=np.float64)),
            look_at=world(aim_v),
            near=0.01,
            far=4.0,
            channels=CHANNELS,
        ),
        CameraSpec(
            name="top",
            width=640,
            height=480,
            fov_y=50.0,
            eye=world(aim_v + np.asarray((0.0, 0.0, TOP_HEIGHT), dtype=np.float64)),
            look_at=world((aim_v[0], aim_v[1], 0.0)),
            # Straight down is degenerate against a +z up hint -- forward and up
            # would be parallel and the look-at basis undefined. +x puts the
            # direction the arms reach at the top of the frame, for every
            # variant, which is what makes nine plan views comparable.
            up=(1.0, 0.0, 0.0),
            near=0.01,
            far=4.0,
            channels=CHANNELS,
        ),
        CameraSpec(
            name="photo",
            width=PHOTO_SIZE[0],
            height=PHOTO_SIZE[1],
            fov_y=PHOTO_FOV_Y,
            eye=world(PHOTO_EYE),
            look_at=world((aim_v[0] + PHOTO_LOOK_AHEAD, aim_v[1], PHOTO_LOOK_Z)),
            up=(0.0, 0.0, 1.0),
            near=0.05,
            far=8.0,
            channels=CHANNELS,
        ),
    )


POLISH_AIM = (-0.005, 0.0, 0.140)
"""The rig's default aim [m, surface frame]: the polishing drum's contact.

``MACHINE_AT[0]`` less the machine's ``contact_reach``, at its
``contact_height``. It is the default because it is the aim the calibration was
transplanted onto, and it is what :data:`RGBS_RIG` registers -- so a document
that stages :data:`RGBS_RIG_NAME` with no loader behind it still resolves to a
rig pointed somewhere sensible instead of at the origin.

Restated rather than imported: ``polish/layout.py`` imports this module
and not the other way about, and it asserts its own ``STRIP_AIM`` against this
at import, which is what holds the two together.
"""

RGBS_RIG = register_camera_rig(RGBS_RIG_NAME, rgbs_views(POLISH_AIM))
