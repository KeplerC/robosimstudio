"""Drive one planned pick stage on BOTH arms at once.

The pair matters physically, not just for wall-clock. The hands take the strip
end-on, in along its own axis from opposite ends, and whichever one arrives
first shoves it: measured moving the strip 14 mm and leaving both tool centres
at +-75.4 mm on a strip whose half-length is 75 mm -- gripping the literal
tips, with no engagement to lose. Entering together, the two axial pushes are
equal and opposite, the strip drifts 3 mm instead of 14, and both hands land
within about a millimetre of the +-58 mm they were asked for.

``robot.stream_dual`` advances both arms one waypoint per control step inside a
single tool call. Alternating ``robot.move_to_joints`` instead costs a graph
round-trip per waypoint -- measured at roughly seventy control steps each --
which is both far too slow and, here, not simultaneous at all.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    stage: str
    roles: list
    waypoints: int


def run(ctx: NodeContext, arms: list, stage: str, settle_steps: int = 45) -> Output:
    tracks = {}
    roles = []
    for entry in arms:
        target = next((s for s in entry["stages"] if s["stage"] == stage), None)
        if target is None:
            raise RuntimeError(f"plan for {entry['role']!r} has no stage {stage!r}")
        tracks[str(int(entry["arm_id"]))] = [w["positions"] for w in target["trajectory"]["waypoints"]]
        roles.append(entry["role"])
    ctx.tool("robot.stream_dual", tracks=tracks, tolerance=0.005, settle_steps=int(settle_steps))
    return {"stage": stage, "roles": roles, "waypoints": max(len(w) for w in tracks.values())}
