"""The feedback poll between rubs: read the mark, decide whether to rub again.

The open-loop rig cannot have this. The mark's cleared fraction and the
smoothed press force are read every few steps while the policy loop holds the
pressed pose; the node exits on target, on stall (no progress across the stall
window -- a sagged press, or a mark whose remaining cells are outside the
contact band), or on budget. Routing: ``done`` when the target fraction is
reached, ``more`` otherwise -- the workflow decides whether another rub follows.

It used to be where the work happened, against a drum presenting its end face:
press, then wait. Against a barrel it is only the decision. Removal happens
during the ``rub`` legs, because half a mark is never under the band at any one
press -- measured, a dwell of any length at any overdrive from 1 to 6 mm stops
at 0.68 of a mark. A long ``max_steps`` here would buy nothing but time.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    cleared: float
    gained: float
    press_mean: float
    waited: int


def run(ctx: NodeContext, spot: int, until: float = 0.95, max_steps: int = 600, stall_steps: int = 150) -> Output:
    result = ctx.tool(
        "sim.wait_polish", spot=int(spot), until=float(until), max_steps=int(max_steps), stall_steps=int(stall_steps)
    )
    return {
        "route": "done" if result["done"] else "more",
        "cleared": float(result["cleared"]),
        "gained": float(result["gained"]),
        "press_mean": float(result["press_mean"]),
        "waited": int(result["waited"]),
    }
