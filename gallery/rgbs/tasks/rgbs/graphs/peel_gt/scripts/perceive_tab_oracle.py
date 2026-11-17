"""Where the strip's free end is -- read off the particles, which is the oracle.

The privileged body of ``perceive_tab``. It answers exactly the question the
vision body has to answer from a camera, and it answers it by looking at the
simulator, so a graph that runs this branch is being handed something a real rig
would have to earn. That is legitimate for building a graph and for generating
data; it is not a perception pipeline, and the router beside this node is what
makes the difference visible instead of implied.

**The feature is declared one key at a time, and that is the interface.** The
subgraph publishes seven named, typed outputs rather than one opaque blob,
because that declaration is the whole contract between the two bodies -- a
sibling that produced a differently-shaped bag under the same name would type-
check and then fail downstream. ``tab_feature`` is the one no camera can
produce: it names a particle set the *asset* declares, and the vision body
returns an empty one meaning "seize whatever is inside the radius", which is what
a real gripper does. The contract is the keys, never where the values came from.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    tab_point: dict
    tab_peel_dir: dict
    tab_feature: str
    tab_rigid: bool
    tab_grasp_radius_m: float
    tab_hold_radius_m: float
    tab_anchored: bool
    tab_entity: str


def run(ctx: NodeContext, entity: str = "") -> Output:
    tab = ctx.tool("sim.get_tab", entity=str(entity))
    peel = tab.get("peel_dir") or {}
    found = bool(tab.get("point")) and any(abs(float(v)) > 0.0 for v in peel.values())
    return {
        "route": "found" if found else "not_found",
        "tab_point": dict(tab.get("point", {})),
        "tab_peel_dir": dict(peel),
        "tab_feature": str(tab.get("feature", "")),
        "tab_rigid": bool(tab.get("rigid", False)),
        "tab_grasp_radius_m": float(tab.get("grasp_radius_m", 0.0)),
        "tab_hold_radius_m": float(tab.get("hold_radius_m", 0.0)),
        "tab_anchored": bool(tab.get("anchored", False)),
        "tab_entity": str(tab.get("entity", "")),
    }
