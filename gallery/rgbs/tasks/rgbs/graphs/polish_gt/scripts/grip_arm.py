"""One hand's grip move: ramp the jaws, then dwell.

``until_force`` makes it a FORCE-gated close rather than a width-gated one, and
the pick has to use it. Ramped to a commanded width, the jaws keep travelling
long after the pads are on the strip -- ``jaw_gap`` still reads 25 mm while the
pads press it at 66 N -- and the over-squeeze wedges both hands outboard, from
+-58 mm to +-83 mm, leaving each pad half off the end of the part. Gated on
measured pad force the hands stay at +-60 mm and the strip is carried.

It also arms the force HOLD at the same number, which is what keeps the grip
for the rest of the episode rather than only for the close -- see
``ToolsPolish.GripHold``. ``hold_force`` sets it apart from the gate, and the
pick uses that: it gates at 30 N and holds at 45. The two are bounded by
different things -- the gate is a ramp, and what limits it is what the close may
do to the strip's pose (past ~60 N it walks both hands outboard and drags the
part), while the hold only ever nudges 0.04 mm at a time against a part already
captured. Twenty episodes each: 30 N scores 19, 45 N scores 20. -1 follows the
gate; 0 opts out.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    arm: str
    command_m: float
    pad_force: dict
    ramp_used: int
    waited: int


def run(
    ctx: NodeContext,
    width: str,
    arm: str = "work",
    settle_steps: int = 36,
    ramp_steps: int = 24,
    until_force: float = 0.0,
    hold_force: float = -1.0,
) -> Output:
    result = ctx.tool(
        "sim.set_grip",
        width=width,
        arm=arm,
        ramp_steps=int(ramp_steps),
        until_force=float(until_force),
        hold_force=float(hold_force),
    )
    waited = ctx.tool("sim.wait_steps", steps=int(settle_steps))
    return {
        "arm": result["arm"],
        "command_m": float(result["command_m"]),
        "pad_force": result.get("pad_force", {}),
        "ramp_used": int(result.get("ramp_used", ramp_steps)),
        "waited": int(waited["waited"]),
    }
