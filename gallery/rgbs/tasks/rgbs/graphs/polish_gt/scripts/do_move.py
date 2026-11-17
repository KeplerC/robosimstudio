"""Plan one named strip motion and execute it -- the suite's workhorse node.

The plan is made against the strip's pose AT THIS INSTANT (the tool reads it
inside the same bus call that solves the IK), so a node never executes a path
from where the strip used to be.

Execution:
* one holder -- ``robot.execute_trajectory``, the streaming single-arm path;
* two holders -- ``robot.stream_dual``, which advances both arms one waypoint
  per control step inside a single tool call.

Both are one tool call for the whole path, and that is the point. Streaming a
pair by alternating ``robot.move_to_joints`` calls costs a graph round-trip per
waypoint -- measured at roughly seventy control steps each, which spent a whole
12000-step episode on one two-second carry and never reached the first press.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    motion: str
    pair: bool
    waypoints: int


def run(
    ctx: NodeContext,
    motion: str,
    align_spot: int = -1,
    holders: list | None = None,
    stations: dict | None = None,
    spot_x: float = 0.0,
    settle_steps: int = 45,
) -> Output:
    correct = None
    if align_spot >= 0:
        # Close the loop on where the mark ACTUALLY is. The plan puts the spot
        # under the drum's contact band; the strip arrives a few millimetres
        # off, and the band is only a few millimetres wide across the spindle,
        # so an uncorrected press can read real force and still polish nothing
        # -- the cells sit outside the patch.
        #
        # The residual is projected into the PRESSED PLANE (perpendicular to the
        # working normal), not perpendicular to the spindle axis: the component
        # along the normal is press depth, which `OVERDRIVE` owns, and adding it
        # here would let an alignment error become a harder press.
        state = ctx.tool("sim.get_polish_state")
        normal = state["work_normal"]
        contact = state["bit_contact"]
        centre = state["spots"][int(align_spot)]["centre"]
        delta = [contact[i] - centre[i] for i in range(3)]
        along = sum(delta[i] * normal[i] for i in range(3))
        shift = [delta[i] - along * normal[i] for i in range(3)]
        correct = {role: shift for role in (holders or ["work", "support"])}

    plan = ctx.tool(
        "sim.plan_strip_move",
        motion=motion,
        holders=holders,
        stations=stations,
        spot_x=spot_x,
        correct=correct,
    )
    arms = plan["arms"]
    if plan["pair"]:
        ctx.tool(
            "robot.stream_dual",
            tracks={str(int(a["arm_id"])): a["waypoints"] for a in arms},
            tolerance=0.005,
            settle_steps=int(settle_steps),
        )
    else:
        a = arms[0]
        ctx.tool(
            "robot.execute_trajectory",
            trajectory={"waypoints": [{"positions": w} for w in a["waypoints"]]},
            arm_id=int(a["arm_id"]),
            max_steps_per_waypoint=30,
            tolerance=0.001,
        )
    return {"motion": motion, "pair": bool(plan["pair"]), "waypoints": max(len(a["waypoints"]) for a in arms)}
