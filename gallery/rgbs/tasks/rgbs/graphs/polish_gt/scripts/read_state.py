"""The opening read: wait out the spindle's spin-up, then verify it runs.

The environment zeroes joint velocities at reset, so the drum re-spins from
rest over ~0.15 s -- and this node runs sooner than that. It polls briefly (up
to ~2 s of sim time) before declaring the machine dead: an episode that starts
with a genuinely dead spindle must still fail loudly at step one hundred, not
grind nothing for twelve thousand.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    disc_speed: float
    spots_remaining: int
    arms: dict
    waited: int


def run(ctx: NodeContext) -> Output:
    waited = 0
    state = ctx.tool("sim.get_polish_state")
    while not state["spinning"] and waited < 120:
        waited += int(ctx.tool("sim.wait_steps", steps=15)["waited"])
        state = ctx.tool("sim.get_polish_state")
    return {
        "route": "ready" if state["spinning"] else "not_spinning",
        "disc_speed": float(state["disc_speed"]),
        "spots_remaining": int(state["spots_remaining"]),
        "arms": state["arms"],
        "waited": waited,
    }
