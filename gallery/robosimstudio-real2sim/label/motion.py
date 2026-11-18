"""Local panel motion helpers, adapted from the original grasp fixture."""

import numpy as np

WORK = np.array([0.0, 0.2, 0.0])
AJAR, CLOSE, TIP, HOVER, CLEAR = 0.72, 0.008, 0.0165, 0.05, 0.011
FAR = WORK + np.array([0.0, -0.125, 0.146])
NEAR = WORK + np.array([0.0, -0.040, 0.069])


def tick(session, steps):
    for _ in range(steps):
        arm = session.arms["right"]
        held = getattr(arm, "label_press_orientation", None)
        if held is not None:
            # Keep the requested vertical attitude through contact. The page's
            # auto-seating must not adopt each loaded deflection as a new target.
            arm.aim(np.asarray(arm.cursor_pos).copy(), held.copy(), moving=True)
        yield


def go(session, arm: str, target, *, steps: int = 600, tolerance: float = 0.004, speed: float | None = None) -> float:
    """Put the arm's cursor at ``target`` (world [m]) and step until the tool centre is there.

    Args:
        speed: Walk the cursor there at this speed [m/s], held as a drag until it
            arrives, the way a mouse moves the gizmo. ``None`` puts it there in one
            jump. The difference matters on a scene whose arm has no device IK
            (a coupled rope): there the page commands each cursor update directly,
            with no setpoint law to limit the rate, so a jump is a flick.

    Returns the distance left [m] -- under ``tolerance`` when the hand arrived.
    """
    panel = session.arms[arm]
    target = np.asarray(target, dtype=np.float64)
    if speed:
        start = np.array(panel.cursor_pos, dtype=np.float64)
        stride = float(speed) / float(session.env.request.physics.control_hz)
        legs = max(1, int(np.ceil(np.linalg.norm(target - start) / stride)))
        for k in range(1, legs + 1):
            panel.aim(start + (target - start) * (k / legs), moving=k < legs)
            yield from tick(session, 1)
    panel.aim(target.copy(), moving=False)
    for _ in range(steps):
        yield from tick(session, 1)
        if _miss(panel, session, target) < tolerance:
            break
    return _miss(panel, session, target)


def _miss(panel, session, target) -> float:
    return float(np.linalg.norm(panel.ik.ee_pose(session.env.state)[0] - target))


def column(label, state, c: int, along: float) -> np.ndarray:
    """Where the label's grid column ``c`` is, ``along`` metres from its tab end, now [m, world].

    Past the last row it carries on along the last cell.
    """
    nx, ny = label.spec.cells
    cell = label.spec.cell[1]
    q = state.particle_q.numpy()[np.asarray(label.particles)]
    j = min(int(along // cell), ny - 1)
    t = along / cell - j
    return (1.0 - t) * q[j * (nx + 1) + c] + t * q[(j + 1) * (nx + 1) + c]


def tab_reach(label, state, away: np.ndarray) -> float:
    """How far along ``away`` the tab reaches, its corners counted, lean and all [m]."""
    from robosimstudio.core.math import quat_to_mat

    pose = label.tab_pose(state)
    half = 0.5 * np.asarray(label.spec.tab_size, dtype=np.float64)
    corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * half
    return float(np.max((corners @ quat_to_mat(pose[3:7]).T + pose[:3]) @ away))
