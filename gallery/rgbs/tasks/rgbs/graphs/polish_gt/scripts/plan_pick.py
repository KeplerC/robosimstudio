"""Both hands' pick plan: approach/descend onto the strip's end stations.

Independent single-arm stages -- the strip lies on the mat and nothing holds
anything yet -- solved with the arm's full-pose LM (the jaws must close across
the strip's 25 mm width, and the runtime's own go_to_pose trades the wrist
roll away).
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    arms: list
    grip_hold_m: float


def run(ctx: NodeContext, station_work: float = -0.058, station_support: float = 0.058) -> Output:
    plan = ctx.tool("sim.plan_pick", station_work=station_work, station_support=station_support)
    return {"route": "planned", "arms": plan["arms"], "grip_hold_m": float(plan["grip_hold_m"])}
