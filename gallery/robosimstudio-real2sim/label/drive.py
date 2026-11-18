"""The old carry, lay and nine lines of straight-down presses, driven through the page.

The stock hand uses a 17.1 mm approach opening and a pinch 8 mm over the tab
centre (motion.py), avoiding the box and bench. It closes gradually to a
0.93 grip request, measured at 7.49 mm on the
tab. The stock servo gains and limits are untouched. Carry height is 180 mm.
The carry and lay keep the original plan, with lay endpoints calibrated for
this solver. Only robot cursor targets are commanded. The 3.5 mm press depth,
nine columns are retained; each bondable 5 mm row is pressed by one fully open fingertip. A 10 mm inter-dab lift clears the
sheet before the next lateral move; the downward orientation stays commanded.
Rubbing is deliberately absent: the old drives slipped and peeled the sheet.
"""

from __future__ import annotations

import numpy as np
from motion import AJAR, CLOSE, FAR, HOVER, NEAR, TIP, WORK, go, tick
from motion import column as _column


def column(label, state, c: float, along: float) -> np.ndarray:
    """The test's ``column`` at a fractional grid column: the two grid columns either side, interpolated (and past
    the label's edge, extrapolated from the two at it)."""
    nx = int(label.spec.cells[0])
    lo = int(np.clip(np.floor(c), 0, nx - 1))
    hi = lo + 1
    t = float(c) - lo
    a = _column(label, state, lo, along)
    return a if t == 0.0 else (1.0 - t) * a + t * _column(label, state, hi, along)


LIFT_Z = 0.18
DESCEND = 0.02
FAR_SHIFT = np.array([0.0, 0.005, 0.002])
NEAR_SHIFT = np.array([0.0, 0.005, 0.0015])
IDLE = 0.5
PICK_OPEN = 0.82


def _far_near():
    return (FAR + FAR_SHIFT, NEAR + NEAR_SHIFT)


def _lay(session, label, rec=None) -> dict:
    """Take the tab off the bench, carry the label over the box and lay it down along it; let go."""
    mark = rec.mark if rec is not None else lambda *a, **k: None
    arm = session.arms["right"]

    def state():
        return session.env.state

    far, near = _far_near()
    tab0 = label.tab_pose(state())[:3]
    arm.grip_request = PICK_OPEN
    yield from tick(session, int(40 * IDLE))
    yield from go(session, "right", tab0 + np.array([0.0, 0.0, 0.05]), speed=0.08, steps=60)
    miss = yield from go(session, "right", tab0 + np.array([0.0, 0.0, CLOSE]), speed=0.04)
    print(
        "PRE_GRASP",
        arm.ik.ee_pose(state())[0].tolist(),
        "box",
        state().body_q.numpy()[label.body][:3].tolist(),
        flush=True,
    )
    for k in range(1, 31):
        arm.grip_request = PICK_OPEN + (0.93 - PICK_OPEN) * k / 30
        yield from tick(session, 1)
    yield from tick(session, 30)
    mark("grasped")
    out = {
        "tab_miss": miss,
        "closed_width": float(session.env.gripper("right").width(state())),
        "pickup_offset": (arm.ik.ee_pose(state())[0] - label.tab_pose(state())[:3]).tolist(),
    }
    yield from go(session, "right", np.array([tab0[0], tab0[1], WORK[2] + LIFT_Z]), speed=0.04, steps=60)
    out["carried"] = float(label.tab_pose(state())[2] - tab0[2])
    out["carry_offset"] = (arm.ik.ee_pose(state())[0] - label.tab_pose(state())[:3]).tolist()
    mark("lifted")
    if out["carried"] < 0.10:
        raise RuntimeError("The fingers did not carry the tab")
    arm.label_press_orientation = np.asarray(arm.cursor_quat).copy()
    above = np.array([far[0], far[1], WORK[2] + LIFT_Z])
    yield from go(session, "right", above, speed=0.03, steps=30)
    yield from tick(session, 60)
    far, touchdown = yield from guided_touchdown(session, label)
    out.update(touchdown)
    mark("over_box")
    yield from tick(session, 30)
    mark("touched")
    for k in range(1, 31):
        yield from go(session, "right", far + (near - far) * k / 30, speed=0.03, steps=5)
    yield from tick(session, 20)
    arm.grip_request = AJAR
    yield from tick(session, 40)
    mark("released")
    yield from go(session, "right", near + np.array([0.0, 0.0, 0.05]), speed=0.05, steps=30)
    yield from tick(session, int(30 * IDLE))
    q = state().particle_q.numpy()[label.bond_particles]
    box = state().body_q.numpy()[label.body][:3]
    on_top = (np.abs(q[:, 0] - box[0]) < 0.05) & (np.abs(q[:, 1] - box[1]) < 0.06)
    flat = on_top & (q[:, 2] - box[2] - 0.025 < 0.0015)
    out.update(laid=label.bonded(), flat=float(flat.mean()))
    return out


def guided_touchdown(session, label):
    """Damp and centre the free edge, then lower it, through finger contact."""
    arm = session.arms["right"]
    dt = 1.0 / float(session.env.request.physics.control_hz)
    target = np.asarray(session.env.state.body_q.numpy()[label.body][:3], dtype=float)
    target += np.array([0.0, -0.040, 0.0255])
    command = np.asarray(arm.cursor_pos).copy()

    def edge():
        return session.env.state.particle_q.numpy()[label.particles].reshape(20, 9, 3)[-1].copy()

    previous = edge().mean(0)
    velocity = np.zeros(2)
    history = []

    def guide(q):
        nonlocal previous, velocity
        centre = q.mean(0)
        velocity = 0.5 * velocity + 0.5 * (centre[:2] - previous[:2]) / dt
        previous = centre.copy()
        history.append(centre[:2].copy())
        if len(history) > 36:
            history.pop(0)
        mean = np.mean(history, axis=0)
        # Follow the swing to remove its energy; steer its mean onto the spot.
        v = 1.0 * (target[:2] - mean) + 0.15 * velocity
        v *= min(1.0, 0.05 / max(np.linalg.norm(v), 1e-9))
        command[:2] += v * dt
        arm.aim(command.copy(), arm.label_press_orientation.copy(), moving=True)
        spread = float(np.linalg.norm(np.asarray(history) - mean, axis=1).max())
        return float(np.linalg.norm(target[:2] - mean)), spread

    for k in range(480):
        error, spread = guide(edge())
        if k > 120 and error < 0.004 and spread < 0.030:
            break
        yield from tick(session, 1)
    else:
        raise RuntimeError(f"Free edge did not centre: mean error {error * 1000:.1f} mm, swing {spread * 1000:.1f} mm")
    hover = edge().mean(0)
    print("ALIGNED_TAIL", hover.tolist(), "mean_error_mm", error * 1000, "swing_mm", spread * 1000, flush=True)
    for _ in range(360):
        q = edge()
        error, spread = guide(q)
        touching = q[:, 2].min() <= target[2] + 0.0007
        if touching and error < 0.004:
            break
        if not touching:
            command[2] = max(0.075, command[2] - DESCEND * dt)
        arm.aim(command.copy(), arm.label_press_orientation.copy(), moving=True)
        yield from tick(session, 1)
    else:
        raise RuntimeError("The label's free edge did not reach the box top")
    touch = edge().mean(0)
    print("TOUCHDOWN_TAIL", touch.tolist(), "mean_error_mm", error * 1000, flush=True)
    return command.copy(), {
        "hover_tail": hover.tolist(),
        "touchdown_tail": touch.tolist(),
        "touchdown_error_mm": error * 1000,
    }


def lay(session, label, rec=None):
    """Track continuously with Newton IK; hold attitude after the pickup."""
    arm = session.arms["right"]
    # Select the toolkit's incremental tracking solver. Full-pose PyRoKi can
    # choose a different elbow branch near this low pickup. This changes no
    # actuator, contact, geometry, object state or force limit.
    arm.ik._pyroki_tried = True
    arm.ik._pyroki = None
    arm.ik.solver_name = "newton"
    try:
        return (yield from _lay(session, label, rec))
    finally:
        if hasattr(arm, "label_press_orientation"):
            del arm.label_press_orientation
        arm.dragging = False


LINES = tuple(c - 0.8 for c in range(9))
DAB_PUSH = 0.0035
PRESS_OFFSET = np.array([0.0, -0.0499, 0.0])  # Centre the positive-Y blade; the other is clear of the parcel.
DAB_HOLD = 8
DAB_UP = 0.010
DAB_CLEAR = 0.003


def _press_down(session, label, rec=None) -> list[float]:
    """Press the label down with one fully open fingertip, press after press along nine lines (``LINES``) from just past
    the tab to its free end, where it lies now: bonded after each line.

    Pressed straight down, nothing slides: rubbed, the fingertips' friction held the hand back until it slipped and
    dragged the label with it, tearing what was bonded (g1, g2, m1, n1, s1), or held it stalled against the box (u1)."""
    mark = rec.mark if rec is not None else lambda *a, **k: None
    arm = session.arms["right"]
    arm.grip_request = 0.0  # Full 95 mm opening: a single blade presses, with no narrow gap to trap the sheet.
    yield from tick(session, int(40 * IDLE))
    out = []
    lift = np.array([0.0, 0.0, TIP - DAB_PUSH])
    up = np.array([0.0, 0.0, DAB_PUSH + DAB_UP])

    nx, _ny = label.spec.cells
    rows = np.unique((np.asarray(label.bond_particles) - int(label.particles[0])) // (nx + 1))

    def points(c: float):
        return [column(label, session.env.state, c, float(j) * label.spec.cell[1]) + lift + PRESS_OFFSET for j in rows]

    for i, c in enumerate(LINES):
        ps = points(c)
        start, end = ps[0], ps[-1]
        yield from go(session, "right", start + np.array([0.0, 0.0, HOVER + DAB_PUSH]), speed=0.08, steps=60)
        ps = points(c)
        start, end = ps[0], ps[-1]
        yield from go(session, "right", start + up, speed=0.05, steps=10)
        mark(f"stroke{i + 1}")
        for point in ps:
            p = point.copy()
            yield from go(session, "right", p + up, speed=0.06, steps=12)
            yield from tick(session, 6)
            # The stock arm can still trail the cursor laterally. Press from
            # the actual unloaded XY so the descent is vertical, not a rub.
            p[:2] = arm.ik.ee_pose(session.env.state)[0][:2]
            yield from go(session, "right", p, speed=0.05, steps=6)
            yield from tick(session, DAB_HOLD)
            yield from go(session, "right", p + up, speed=0.05, steps=8)
        yield from go(session, "right", end + np.array([0.0, 0.0, HOVER + DAB_PUSH]), speed=0.05, steps=30)
        out.append(label.bonded())
        mark(f"stroke{i + 1}_done", bonded=float(out[-1]))
    return out


def press_down(session, label, rec=None) -> list[float]:
    """Hold the downward target, using the stock servo, for all nine lines."""
    arm = session.arms["right"]
    arm.label_press_orientation = np.array([0.0, 1.0, 0.0, 0.0])
    try:
        return (yield from _press_down(session, label, rec))
    finally:
        if hasattr(arm, "label_press_orientation"):
            del arm.label_press_orientation
        arm.dragging = False
