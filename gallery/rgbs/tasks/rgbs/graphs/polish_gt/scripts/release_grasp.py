"""Forget a hand's captured grasp -- it has let go of the strip."""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    released: list


def run(ctx: NodeContext, arms: list) -> Output:
    return {"released": ctx.tool("sim.release_grasp", arms=arms)["released"]}
