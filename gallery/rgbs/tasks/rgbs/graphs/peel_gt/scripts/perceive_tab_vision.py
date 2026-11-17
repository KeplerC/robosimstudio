"""The same tab, from a camera. **Declared, and skipped on this machine.**

This is the non-privileged body of ``perceive_tab``: the four numbers the oracle
reads off the particles, earned instead from an RGB-D frame through an
open-vocabulary detector, a segmenter and a depth back-projection.

**It cannot run here, and it says so before it does anything.** The detector and
back-projection tools come from ``open-robot-skills``, which is not installed,
not vendored, and not resolvable in this checkout -- no ``GAP_SKILLS_PATH``, no
sibling package, ``import gap_skills`` fails. So :func:`run` checks the registry
first and raises a message naming the missing tools and the reason.

That check is the point of the file. The alternative was measured elsewhere in
this corpus and recorded in prose: with the bundle absent, a graph reached its
detector node, took the ``not_found`` exit for every object in turn, and reported
**0.00 with a plausible-looking trace** -- "a zero with a plausible-looking trace
is the kind of number that gets quoted". A body that cannot run must fail as a
missing dependency, never as a perception result.

**What it would produce, and where the two bodies genuinely differ.** Three of
the oracle's keys have honest camera answers and two do not:

``point``, ``peel_dir``, ``hold_radius_m``
    The strip's mask, back-projected. The free end is the extremity of the
    strip's long axis that is *not* over the substrate; the peel runs from there
    back along that axis; the hold reaches half the tab's observed width.
``rigid``
    Observable, and this is the nice one: a stiff pull-tab lies within a few
    degrees of horizontal at substrate height, and a drooping flap hangs down the
    face. The oracle answers the same question by asking whether the layout
    declared a handle set. Same fact, two instruments.
``feature``
    **Empty, always.** It names a particle set the *asset* declares, and no
    camera produces the name of a simulator's declared set. Returning empty means
    "seize whatever is inside the radius", which is what a real gripper does.
``anchored``
    **False, always.** Whether the task is still pinning the tab at rest is a
    fact about the staging, not about the scene. A real cell has no such pin.

Those two are what the interface test is for: the pair must agree on the *keys*,
never on where the values came from.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

REQUIRED_TOOLS = ("grounding-dino.detect", "sam3.segment_box", "geometry.mask_to_world_points")
"""What this body needs and this checkout does not have.

All three are ``open-robot-skills`` bundle tools. ``robot.get_observation`` is
the runtime's own and is always present, which is why it is not listed.
"""

WHY_ABSENT = (
    "the open-robot-skills bundle is not installed in this checkout, so the vision body of "
    "perceive_tab is declared and skipped. It is not a task failure and it is not a perception "
    "result: run the oracle body (--perception oracle, the default), or install the bundle and "
    "put it on GAP_SKILLS_PATH."
)

PROMPT = "the free end of the adhesive strip"
"""What the detector is asked for. The tab, not the strip: the whole strip is a
line across a box and its free end is the only part a hand can take."""

FLAT_TOLERANCE_DEG = 25.0
"""How far off horizontal the tab's long axis may lie and still be a pull-tab.

The one decision this body makes that the oracle makes from a declaration. A
package pull-tab is held out stiff at box-top height; a flap folds over the edge
and hangs, which is tens of degrees, not tens of millimetres -- so the threshold
is loose on purpose and the two classes do not touch it.
"""

GRASP_RADIUS_M = 0.02
"""How near the tool centre a grasp has to land to count [m].

Stated rather than perceived: it is a tolerance the *task* sets on the grasp, and
a camera has no opinion about it. The oracle reads the same number off the tape
asset. # portable: a task tolerance on the grasp, not a property of any hand.
"""


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


NOTHING: Output = {
    "route": "not_found",
    "tab_point": {},
    "tab_peel_dir": {},
    "tab_feature": "",
    "tab_rigid": False,
    "tab_grasp_radius_m": 0.0,
    "tab_hold_radius_m": 0.0,
    "tab_anchored": False,
    "tab_entity": "",
}
"""Every declared key, empty. A body that returned a *subset* on its miss path
would leave the downstream binding reading a stale value from an earlier visit
rather than nothing at all."""


def _missing(ctx: NodeContext) -> list[str]:
    """Which required tools this registry does not carry."""
    registry = getattr(ctx, "_tool_registry", None)
    if registry is None:
        return []
    return [name for name in REQUIRED_TOOLS if name not in registry]


def run(ctx: NodeContext, camera: str = "", prompt: str = PROMPT) -> Output:
    missing = _missing(ctx)
    if missing:
        raise RuntimeError(f"perceive_tab_vision needs {missing}, which are not registered: {WHY_ABSENT}")

    observation: Any = ctx.tool("robot.get_observation")
    frames = observation["cameras"] if isinstance(observation, dict) else observation.cameras
    frame = frames[camera] if camera else next(iter(frames.values()))

    boxes = ctx.tool("grounding-dino.detect", image=frame["rgb"], text=prompt)["boxes"]
    if not boxes:
        return dict(NOTHING)  # type: ignore[return-value]
    mask = ctx.tool("sam3.segment_box", image=frame["rgb"], box=boxes[0])["mask"]
    cloud = ctx.tool(
        "geometry.mask_to_world_points",
        mask=mask,
        depth=frame["depth"],
        intrinsics=frame["intrinsics"],
        camera_pose=frame["pose"],
    )
    points = np.asarray(cloud["points"], dtype=np.float64)
    if len(points) < 3:
        return dict(NOTHING)  # type: ignore[return-value]

    centre = points.mean(axis=0)
    # The strip's long axis, and which end of it is free: the tab hangs off the
    # substrate, so the free extremity is the one whose neighbourhood sits
    # lowest against the surface the rest of the strip lies on.
    axis = np.linalg.svd(points - centre)[2][0]
    projection = (points - centre) @ axis
    ends = (points[projection.argmin()], points[projection.argmax()])
    free = min(ends, key=lambda p: float(p[2]))
    peel = axis if float((centre - free) @ axis) > 0.0 else -axis
    peel[2] = 0.0
    peel = peel / max(float(np.linalg.norm(peel)), 1.0e-9)
    across = float(np.linalg.norm(points - free, axis=1).max())
    tilt = float(np.degrees(np.arcsin(abs(float(axis[2])))))

    return {
        "route": "found",
        "tab_point": {"x": float(free[0]), "y": float(free[1]), "z": float(free[2])},
        "tab_peel_dir": {"x": float(peel[0]), "y": float(peel[1]), "z": float(peel[2])},
        "tab_feature": "",
        "tab_rigid": bool(tilt <= FLAT_TOLERANCE_DEG),
        "tab_grasp_radius_m": GRASP_RADIUS_M,
        "tab_hold_radius_m": float(across),
        "tab_anchored": False,
        "tab_entity": "",
    }
