"""Take hold of the film, then let the task's own anchor go. In that order.

Closing two rigid fingers on a sub-millimetre film does not reliably grip it, so
the strip is carried by a kinematic pin on the particles at the grasp site,
re-aimed at the hand every control step. ``sim.attach`` installs that; a graph
cannot re-aim anything per step itself. It is privileged and always will be -- no
real hand seizes particles, and no amount of perception makes this transfer.

**The ordering is the content of this node, and it is measured.** Until a gripper
has it, a rigid pull-tab is held at its rest position by the task's own pin --
that is what makes it a pull-tab rather than a drooping flap. Release it before
the hold is proven and the lift pulls against nothing; leave it on after, and two
kinematic holds fight over the same particles. The pin wins, because it targets a
fixed rest position while the dragger only asks: the tab does not move, the hand
lifts away from it, and not a single bond breaks. That was the whole of this
graph's first working run -- every node green, 220 lift waypoints executed,
``broken_fraction`` 0.0.

So: attach, prove the hold, release the anchor. It used to be one welded tool
whose docstring explained why; it is three lines here, where the order is visible
and a reordering is a diff.

**Two radii, and both are the tab's.** The *hold* reaches the whole declared
column, because a wide film is peeled by a clamp bar across its full edge and a
sphere around the tool centre is that bar only while the tab is narrower than it
-- measured, a 2 cm sphere took 7 of 15 edge particles and the unheld wings swung
from the held centre, reading as a film dangling below a hand that had it. The
*miss* test stays the grasp radius, on the nearest particle, whatever is seized
after. A hold that reached further than the grasp did would be a grasp nobody
made, so a wide take with a distant nearest particle is undone rather than kept.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    n_grabbed: int
    nearest_m: float


def run(
    ctx: NodeContext,
    entity: str = "",
    feature: str = "",
    hold_radius_m: float = 0.03,
    grasp_radius_m: float = 0.02,
    anchored: bool = False,
    max_particles: int = 128,
) -> Output:
    held = ctx.tool(
        "sim.attach",
        entity=str(entity),
        feature=str(feature),
        radius=float(hold_radius_m),
        max_particles=int(max_particles),
    )
    nearest = float(held.get("nearest_m", 0.0))
    if not held["attached"] or nearest > float(grasp_radius_m):
        # Either nothing was inside the hold, or what was is further off than a
        # grasp is allowed to land. Both are the same answer -- the grasp did not
        # happen -- and carrying on would lift an empty hand and report a peel.
        ctx.tool("sim.detach")
        return {"route": "missed", "n_grabbed": 0, "nearest_m": nearest}
    if anchored:
        ctx.tool("sim.release_anchor", entity=str(entity))
    return {"route": "held", "n_grabbed": int(held["n_grabbed"]), "nearest_m": nearest}
