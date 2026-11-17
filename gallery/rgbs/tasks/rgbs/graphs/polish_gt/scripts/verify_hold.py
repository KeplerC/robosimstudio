"""Did the close actually take the strip -- in the hands that should have it?

Routed on **measured pad force**, not the jaw-stall test. That test compares
the jaw gap against the part's declared dimension, and on this hand the two
disagree by construction: the pads stall at ~7.7 mm on a 4 mm strip while
pressing it with 70-86 N. A force is what "the hand is on the part" means.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    on_part: dict
    pad_force: dict


def run(ctx: NodeContext, which: str = "both") -> Output:
    state = ctx.tool("sim.get_polish_state")
    on_part = state["on_part"]
    need = ("work", "support") if which == "both" else (which,)
    ok = all(bool(on_part[k]) for k in need)
    return {
        "route": "held" if ok else "missed",
        "on_part": on_part,
        "pad_force": {k: round(float(v), 2) for k, v in state["pad_force"].items()},
    }
