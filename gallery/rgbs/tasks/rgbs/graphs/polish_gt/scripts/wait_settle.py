"""Dwell while the policy loop holds -- the end-of-episode settle that lets the
goal's hold-for clock run with the hands already parked."""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    waited: int


def run(ctx: NodeContext, steps: int = 150) -> Output:
    waited = ctx.tool("sim.wait_steps", steps=int(steps))
    return {"waited": int(waited["waited"])}
