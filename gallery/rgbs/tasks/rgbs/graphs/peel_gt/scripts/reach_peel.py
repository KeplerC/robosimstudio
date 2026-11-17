"""Can this arm take this tab, and with which wrist -- decided before anything moves.

**This node plans nothing.** It produces four *landmarks* in the tab's own frame
and one wrist orientation, and it says whether the arm can hold that orientation
where it matters. The legs between the landmarks are planned by the nodes that
drive them, at the moment they drive them, from wherever the hand actually is.
That is the difference between this and the ``sim.plan_peel`` it replaces: that
tool solved the whole chain up front and every motion node after it was the same
runner replaying an index into the result.

**Two grasps, chosen by the tab and not by a mode.** A pinch needs something
stiff to close on, so a strip presenting a rigid pull-tab is taken from behind
with the jaws across its faces, and a strip whose tab droops down the box face is
taken from above. The perception body reports which it is looking at; both bodies
can answer it. Getting it wrong is not loud -- the peel happens either way -- and
a top-down grasp on a layout measured for a pinch drags the sheet off from 95 mm
away and reports success. Which is why the gate below exists.

**The gate is on the waypoints that have to be accurate, and it is measured.**
``pre``, ``grasp`` and ``lift`` -- not the clearance via, which solves 155 mm
short on one layout and 161 mm short on the other while the peel the second one
leads into solves to a millimetre. Gating on the via would refuse a motion that
works. And rotation is gated
alongside position, because a full-pose solve can score 0.0 mm with the roll
parked at a joint limit and the jaws 92 degrees across the wrong dimension of the
film -- which is precisely what shipped once, from gating position alone.

**Both wrist twins are tried.** A parallel jaw closing across +z and across -z is
the same physical jaw, but only one of the two rolls is inside this wrist's
range: measured on this arm, one converges to 1.3 degrees and the other parks at
its roll limit 109 degrees out with a perfect position score. So the pick is by
*rotation* error first, and the winner carries the whole chain -- which roll fits
is a property of the advance direction and the arm, not of any one waypoint.
"""

import math
from typing import TypedDict

from gap import NodeContext

PINCH_TILT_DEG = 35.0
"""How far below horizontal a pinch advances [deg].

Level, the wrist has to tuck back against the arm's own base and the solve leaves
20-80 mm of error; tilted, every waypoint solves to about a millimetre.
# portable: measured as the angle a horizontal pull-tab is approached at, and
# re-measurable on any arm by the same sweep -- it is not a constant of this one.

**It was 20.0, and the number moved because the arm did.** These variants were
authored on ``yam_bimanual`` and ran on it at 20 degrees; the ``rgbs`` merge put
them on ``yam_i2rt``, which refuses that pose -- 8.0 mm and 44.6 degrees at the
``handle`` tab, against a gate of 10 mm and 5. Re-measured with
``rgbs/scripts/rgbs_peel_bench.py tilt`` on the layout as staged, the new arm's band is:

     tilt   worst pos   worst rot
      0-20   68-8 mm    53-45 deg   refused, and refused by rotation throughout
        25    0.00 mm     0.92 deg
        35    0.00 mm     0.86 deg
        45    0.00 mm     1.02 deg
        50    1.06 mm     1.16 deg
      55-75  15-73 mm    51-59 deg   refused

So the arm does not want a slightly different angle, it wants a *different band*:
25 to 50 degrees, with the old value sitting five degrees outside its near edge.
35 is the middle of it, which is where a constant belongs when the basin is flat
-- every value in the band solves to 0.00 mm and about a degree, so what is being
bought here is margin against the layout moving, not accuracy.

The docstring above was already right that this is not a constant of any one arm.
This is the first time that has been cashed in.
"""

MAX_SOLVE_ERROR = 0.01
"""Worst position error a landmark may carry and still be driven [m]."""

MAX_SOLVE_ROT_DEG = 5.0
"""Worst orientation error a locked-wrist landmark may carry [deg].

Looser than the servo's own arrival test is tight, and about three times the
1-2 degrees a holdable pose actually costs on this arm.
"""


class Output(TypedDict):
    route: str
    rotation: dict
    orientation: str
    via: dict
    pre: dict
    grasp: dict
    lift: dict
    worst_error: float
    worst_rot_deg: float
    feature: str
    entity: str
    hold_radius_m: float
    grasp_radius_m: float
    anchored: bool


def _pose(point, rotation) -> dict:
    return {"position": {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}, "rotation": dict(rotation)}


def _offset(point, peel, along: float, rise: float) -> list[float]:
    """A landmark, in the tab's own frame: metres along the peel, metres up.

    ``along`` is signed along the direction the peel *runs* -- negative is behind
    the tab, on the side the hand comes in from.
    """
    return [float(point[i] + along * peel[i] + (rise if i == 2 else 0.0)) for i in range(3)]


def run(
    ctx: NodeContext,
    tab_point: dict,
    tab_peel_dir: dict,
    tab_feature: str = "",
    tab_rigid: bool = False,
    tab_grasp_radius_m: float = 0.02,
    tab_hold_radius_m: float = 0.03,
    tab_anchored: bool = False,
    tab_entity: str = "",
    arm_id: int = 0,
    via_along: float = -0.10,
    via_rise: float = 0.18,
    pre_along: float = -0.08,
    pre_rise: float = 0.02,
    hover_rise: float = 0.16,
    high_rise: float = 0.30,
    lift_along: float = 0.24,
    lift_rise: float = 0.26,
    tilt_deg: float = PINCH_TILT_DEG,
    max_error: float = MAX_SOLVE_ERROR,
    max_rot_deg: float = MAX_SOLVE_ROT_DEG,
) -> Output:
    point = [float(tab_point[k]) for k in ("x", "y", "z")]
    peel = [float(tab_peel_dir[k]) for k in ("x", "y", "z")]
    rigid = bool(tab_rigid)

    if rigid:
        # Advance along the strip with the pads closing on the tape's faces --
        # how a person takes a pull-tab. Tilted down, because level the wrist
        # cannot get there. The closing heading is the peel's own bearing:
        # orthogonalised against a tilted advance it lands near vertical, which
        # is across the film's faces, which is where a flat film must be pinched.
        tilt = math.radians(float(tilt_deg))
        advance = {"x": math.cos(tilt) * peel[0], "y": math.cos(tilt) * peel[1], "z": -math.sin(tilt)}
        heading = math.degrees(math.atan2(peel[1], peel[0]))
        orientation = "lock"
        via = _offset(point, peel, via_along, via_rise)
        pre = _offset(point, peel, pre_along, pre_rise)
    else:
        # A drooping flap has no stiff face to pinch, so it is taken from
        # straight above and the roll is left to the solver -- there is no jaw
        # axis to be wrong about when the approach is the surface normal.
        advance = {"x": 0.0, "y": 0.0, "z": -1.0}
        heading = math.degrees(math.atan2(peel[1], peel[0]))
        orientation = "axis"
        via = _offset(point, peel, 0.0, high_rise)
        pre = _offset(point, peel, 0.0, hover_rise)
    lift = _offset(point, peel, lift_along, lift_rise)

    frame = ctx.tool("robot.grasp_frame", approach=advance, close_heading_deg=heading, arm_id=int(arm_id))
    twins = [frame["rotation"], frame["rotation_flipped"]] if orientation == "lock" else [frame["rotation"]]

    best = None
    for rotation in twins:
        worst_pos = worst_rot = 0.0
        for landmark in (pre, point, lift):
            # A single-waypoint plan is a solve that reports instead of raising:
            # it is thrown away here, and the leg that flies it is planned by
            # the node that flies it, from the pose the hand is actually in.
            reach = ctx.tool(
                "motion.plan_joint",
                pose=_pose(landmark, rotation),
                arm_id=int(arm_id),
                num_waypoints=1,
                orientation=orientation,
            )
            worst_pos = max(worst_pos, float(reach["position_error_m"]))
            worst_rot = max(worst_rot, math.degrees(float(reach["rotation_error_rad"])))
        if best is None or (worst_rot, worst_pos) < (best[1], best[2]):
            best = (rotation, worst_rot, worst_pos)

    rotation, worst_rot, worst_pos = best
    reachable = worst_pos <= float(max_error) and worst_rot <= float(max_rot_deg)
    return {
        "route": "reachable" if reachable else "unreachable",
        "rotation": dict(rotation),
        "orientation": orientation,
        "via": _pose(via, rotation),
        "pre": _pose(pre, rotation),
        "grasp": _pose(point, rotation),
        "lift": _pose(lift, rotation),
        "worst_error": float(worst_pos),
        "worst_rot_deg": float(worst_rot),
        "feature": str(tab_feature),
        "entity": str(tab_entity),
        "hold_radius_m": float(tab_hold_radius_m),
        "grasp_radius_m": float(tab_grasp_radius_m),
        "anchored": bool(tab_anchored),
    }
