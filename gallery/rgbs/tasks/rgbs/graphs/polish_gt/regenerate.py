"""Regenerate polish/strip's three per-mark subgraphs. Kept as a script because
the three differ only in a spot index and hand-editing 1500 lines of JSON three
times is how a graph acquires an asymmetry nobody meant."""

import json
import pathlib

P = pathlib.Path("rgbs/tasks/rgbs/graphs/polish_gt/workflow.json")
STATIONS = {"work": -0.058, "support": 0.058}
HOLDERS = ["work", "support"]
ROUNDS = 3
# ABOVE the document's own 0.95, deliberately. `polished-fraction` is measured
# over ALL the cells, so three marks each stopped the moment they crossed 0.95
# land the episode at ~0.957 -- seven thousandths above the gate it has to pass.
# Asking for every cell puts it at ~1.00, and costs only the rounds a mark
# actually needs, because a finished one routes `done` and skips the rest.
#
# 0.99 rather than 1.0, and ROUNDS stays at 3 rather than 4, because rubbing is
# not free: at 4 rounds and a target no mark could reach, one episode spent
# 10084 control steps instead of 6300, eroded the grip past `GripHold`'s budget,
# and set the strip down BESIDE its rest -- scoring reward 1.00 and failing.
#
# That was against 20 mm marks, whose far cells sat 10 mm out against an 11 mm
# reach. Re-run at 4 rounds against the 18 mm marks that ship now, the same
# config scores in 4944 steps and does not fail: no mark needs a fourth round,
# so it is never spent. The guard that matters is therefore the reach margin in
# `SpecPolishMarks.spot_half`, not this number -- 3 is kept because a round that
# can never be needed is a round that can only cost, and because a future change
# that narrows the margin again should fail loudly rather than quietly buy
# itself more rubbing.
UNTIL = 0.99
POLL_STEPS = 60


def move(motion, spot_x, align=None):
    inputs = {"motion": motion, "holders": list(HOLDERS), "stations": dict(STATIONS), "spot_x": spot_x}
    if align is not None:
        inputs["align_spot"] = align
    return {"type": "script", "script": "scripts/do_move.py", "inputs": inputs}


def poll(spot):
    return {
        "type": "script",
        "script": "scripts/press_dwell.py",
        "inputs": {"spot": spot, "until": UNTIL, "max_steps": POLL_STEPS, "stall_steps": POLL_STEPS},
    }


def spot_subgraph(tag, spot_x, spot_index, approach):
    """press -> rub -> poll, up to ROUNDS times, then lift off."""
    nodes = {
        "approach": move(approach, spot_x),
        "press": move("press", spot_x, align=spot_index),
        "ok": {"type": "noop"},
        "lift": move("liftoff", spot_x),
        f"{tag}_dwell_0": poll(spot_index),
    }
    edges = [["START", "approach"], ["approach", "press"], ["ok", "END"], ["lift", "ok"], ["press", f"{tag}_dwell_0"]]
    conditional = {}
    for r in range(1, ROUNDS + 1):
        rub, dwell = f"{tag}_rub_{r}", f"{tag}_dwell_{r}"
        nodes[rub] = move("rub", spot_x, align=spot_index)
        nodes[dwell] = poll(spot_index)
        edges.append([rub, dwell])
        conditional[f"{tag}_dwell_{r - 1}"] = {"router_field": "route", "mapping": {"done": "lift", "more": rub}}
    conditional[f"{tag}_dwell_{ROUNDS}"] = {"router_field": "route", "mapping": {"done": "lift", "more": "lift"}}
    return {
        "skill": "generic",
        "nodes": nodes,
        "edges": edges,
        "conditional_edges": conditional,
        "exit": {"router_field": None, "success_values": ["ok"]},
    }


w = json.loads(P.read_text())
w["subgraphs"]["spotc_sg"] = spot_subgraph("c", 0.0, 1, "raise")
w["subgraphs"]["spotl_sg"] = spot_subgraph("l", -0.033, 0, "shift")
w["subgraphs"]["spotr_sg"] = spot_subgraph("r", 0.033, 2, "shift")
P.write_text(json.dumps(w, indent=1) + "\n")
print("rounds", ROUNDS, "until", UNTIL, "-> nodes/spot", len(w["subgraphs"]["spotc_sg"]["nodes"]))
