"""A stroke: a straight line from where the hand is to where the work is.

The legs where the *path* is the point rather than the destination. Advancing
onto a tab is a straight advance because anything else sweeps the fingers across
the film first; the lift is a straight haul because the haul is the peel -- the
bonds break under the load the path applies, and a path that curves applies a
different load. Both are short, both are at the tab, and both solve to 0.00 mm on
this arm where the transit into them cannot be flown as a line at all.

**Planned from the live pose, not from a plan made earlier.** No start is passed,
which means *where the hand is* -- read by the same solver that flies the line,
so each leg begins where the previous one actually finished rather than where
something predicted it would. The chain seeds each waypoint's solve from the
previous solution, which keeps it in one joint-space branch: interpolating
between branches is what swings an elbow through the table between two perfectly
good endpoints.

**A refusal is a route, not an exception.** ``planned: false`` comes back when
some point on the line is out of reach, and the graph takes its ``unreachable``
edge -- which is the correct outcome for a tab this arm cannot work, and the one
that stops it dragging the strip off the box from centimetres away and calling
that a peel.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    waypoints: int


def run(
    ctx: NodeContext,
    pose: dict,
    arm_id: int = 0,
    waypoints: int = 40,
    orientation: str = "axis",
    max_steps_per_waypoint: int = 15,
) -> Output:
    plan = ctx.tool(
        "motion.plan_linear",
        end=pose,
        arm_id=int(arm_id),
        num_waypoints=int(waypoints),
        orientation=str(orientation),
    )
    if not plan["planned"]:
        return {"route": "unreachable", "waypoints": 0}
    ctx.tool(
        "robot.execute_trajectory",
        trajectory=plan["trajectory"],
        arm_id=int(arm_id),
        max_steps_per_waypoint=int(max_steps_per_waypoint),
    )
    return {"route": "moved", "waypoints": len(plan["trajectory"]["waypoints"])}
