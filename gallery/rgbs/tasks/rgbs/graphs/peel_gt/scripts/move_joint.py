"""A transit: reach a posture, easing there in joint space.

**Not a straight line, and that is measured rather than preferred.** On this arm
the leg from home to the tab's pre-grasp cannot be flown as a Cartesian line at
all -- every intermediate point is 162 mm out of reach *with the wrist roll
free*, because the line passes through a region the arm cannot occupy with its
tool pointing along the grasp's advance. The two endpoints are both exact. A
joint-space ease pins only the endpoints, so it goes where the arm can go.

**Streamed, not stepped.** The whole path is handed to
``robot.execute_trajectory``, one waypoint per control step. Commanding the
endpoint alone is a step input, and a position-servoed arm answers a step input
with a lunge that whips the fingers through whatever it is reaching for.
``max_steps_per_waypoint`` is not per waypoint on the streaming path -- it sizes
the settle at the end as ``max(it * 4, 60)`` -- and the default of 120 is a
480-step settle *per stage*, which on a 900-step task is most of the episode
spent standing still.

``gate`` is what separates a clearance via from a waypoint, and it is **on by
default** so that turning it off is a sentence somebody wrote. A via is a
direction to swing through, not a pose to reach: this task's solves 155-161 mm
short on its two layouts and the peel it leads into is exact, so the node says
``gate: false`` in the workflow and means it. Every other leg must *arrive*, and
routes ``unreachable`` when the solve says it will not -- before moving, which is
the difference between refusing a peel and driving into the box on the way to
finding out.
"""

import math
from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    waypoints: int
    position_error_m: float
    rotation_error_deg: float


def run(
    ctx: NodeContext,
    pose: dict,
    arm_id: int = 0,
    waypoints: int = 40,
    orientation: str = "axis",
    gate: bool = True,
    max_error: float = 0.01,
    max_rot_deg: float = 5.0,
    max_steps_per_waypoint: int = 15,
) -> Output:
    plan = ctx.tool(
        "motion.plan_joint",
        pose=pose,
        arm_id=int(arm_id),
        num_waypoints=int(waypoints),
        orientation=str(orientation),
    )
    error = float(plan["position_error_m"])
    rotation = math.degrees(float(plan["rotation_error_rad"]))
    short = error > float(max_error) or rotation > float(max_rot_deg)
    if not plan["planned"] or (gate and short):
        return {"route": "unreachable", "waypoints": 0, "position_error_m": error, "rotation_error_deg": rotation}
    ctx.tool(
        "robot.execute_trajectory",
        trajectory=plan["trajectory"],
        arm_id=int(arm_id),
        max_steps_per_waypoint=int(max_steps_per_waypoint),
    )
    return {
        "route": "moved",
        "waypoints": len(plan["trajectory"]["waypoints"]),
        "position_error_m": error,
        "rotation_error_deg": rotation,
    }
