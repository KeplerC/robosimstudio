"""Back both hands out of the workspace at once, after they have let go.

Two straight withdrawals along the axes the hands came in on, streamed
together, and both halves of that matter.

*Straight, not home.* Homing swings an arm across the workspace on a path the
servo chooses, which here sweeps over a strip that has just been set down --
and, when the other hand is still holding, was measured diverging the
simulation outright. A hand that has let go only needs to be out of the way.

*Together, not one then the other.* The goal's release clause is the distance
from the strip to the NEAREST hand, so it starts counting its two seconds when
the second hand clears -- and parking in series spends that whole window inside
the second park instead of inside the settle that follows. The episode still
scores either way; what it costs is the settle node, which the runtime then
cuts off mid-call and reports as a failed node in an episode that succeeded.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    roles: list
    parked: bool
    waypoints: int


def run(ctx: NodeContext, roles: list | None = None, distance: float = 0.13) -> Output:
    roles = roles or ["work", "support"]
    tracks = {}
    for role in roles:
        plan = ctx.tool("sim.plan_retreat", role=role, distance=distance)
        tracks[str(int(plan["arm_id"]))] = plan["waypoints"]
    ctx.tool("robot.stream_dual", tracks=tracks, tolerance=0.002, settle_steps=30)
    return {"roles": roles, "parked": True, "waypoints": max(len(w) for w in tracks.values())}
