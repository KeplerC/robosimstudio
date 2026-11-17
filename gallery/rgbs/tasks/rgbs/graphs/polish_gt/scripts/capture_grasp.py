"""Record where each hand actually took the strip.

Run once the jaws have closed and settled. Every carry after this commands
each hand to where it IS on the strip rather than to the station the plan
asked for -- the two differ by this arm's standing servo error, and two hands
commanded to stations they are not at pull against each other through a rigid
strip for the length of every move (measured letting go two thirds of the way
into the first raise).
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    captured: dict


def run(ctx: NodeContext, arms: list | None = None) -> Output:
    return {"captured": ctx.tool("sim.capture_grasp", arms=arms)["captured"]}
