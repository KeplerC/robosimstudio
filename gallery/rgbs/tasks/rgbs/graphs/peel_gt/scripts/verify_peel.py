"""Did the strip actually come off? Read the states the asset publishes.

The task is scored on the broken-bond fraction, so that is what a verify node
reads -- not how far the hand travelled.

``sim.get_states`` rather than a hand-written bond tool, and the difference is
not cosmetic: it returns every state the *entity's own record* declares, so a
graph peeling ``film`` reads the film's ten and a graph peeling ``tape`` reads
the tape's ten. The suite tool this replaces computed four by hand, and
``film_peel`` -- which registers this suite's tools unchanged -- read the tape's
four for a film.

``peel_front`` is reported alongside because a strip torn out of the middle has a
high broken fraction and a front of zero, and those are different outcomes.
"""

from typing import Any, TypedDict

from gap import NodeContext

PEEL_SUCCESS_FRACTION = 0.95
"""The evaluator's own gate, so the graph and the score agree on 'peeled'."""


class Output(TypedDict):
    route: str
    broken_fraction: float
    peel_front: float


def _value(states: dict, entity: str, key: str) -> float:
    """One published state, whichever shape the tier came back in.

    ``sim.get_states`` carries an observability tier alongside a declared value
    and hands an undeclared one through bare, so a reader takes both forms. Both
    of these are ``internal``: no camera reads a bond field, and a policy that
    learned from one has learned something no real robot can.

    The named entity first, then whichever entity publishes the key -- because
    the name reaches this node from *perception*, and the vision body cannot
    produce one. Taking ``next(iter(states))`` instead would read the first
    entity alphabetically and report a peel from whatever the scene happened to
    list first.
    """
    blocks: list[Any] = [states[entity]] if entity and entity in states else []
    blocks += [block for name, block in sorted(states.items()) if name != entity]
    for published in blocks:
        if isinstance(published, dict) and key in published:
            raw = published[key]
            return float(raw.get("value", 0.0) if isinstance(raw, dict) else raw)
    return 0.0


def run(ctx: NodeContext, entity: str = "", success_fraction: float = PEEL_SUCCESS_FRACTION) -> Output:
    states = ctx.tool("sim.get_states", entity=str(entity))
    broken = _value(states, entity, "broken_fraction")
    return {
        "route": "peeled" if broken >= float(success_fraction) else "partial",
        "broken_fraction": broken,
        "peel_front": _value(states, entity, "peel_front"),
    }
