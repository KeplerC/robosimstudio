"""The verdict, read from the evaluator's own gate -- plus the field, so a
short run's trace names the gap instead of just failing."""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    success: bool
    polished_fraction: float
    spots_remaining: int


def run(ctx: NodeContext) -> Output:
    verdict = ctx.tool("sim.check_success")
    state = ctx.tool("sim.get_polish_state")
    success = bool(verdict.get("success", False))
    return {
        "route": "succeeded" if success else "short",
        "success": success,
        "polished_fraction": float(state["polished_fraction"]),
        "spots_remaining": int(state["spots_remaining"]),
    }
