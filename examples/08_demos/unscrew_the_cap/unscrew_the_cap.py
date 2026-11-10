# SPDX-License-Identifier: Apache-2.0
"""Unscrew the cap

Two YAM arms open a bottle: the right hand holds it by its base and the left turns the cap off its thread, two turns in four strokes of the wrist, and lifts it away: press Run.

Nothing joins the cap to the bottle but the contact of their threads, and nothing
holds either but fingers: the bottle stands free on the bench. The parts and the
physics are the ``thread`` kind's (``rss.bottle``, ``rss.cap``, ``rss.THREAD_60HZ``).
"""

import math

import numpy as np

import robosimstudio as rss
from robosimstudio.core.math import mat_to_quat, quat_yaw
from robosimstudio_materials.thread import screwed

CELL = "bench_075"
AT = (-0.05, 0.10)
"""Where the bottle stands [m], in the bench's frame: 0.36 m from the left arm's base, 0.51 m from the right's."""

# -- the routine: every number measured on this scene ------------------------------------------
DOWN = np.array([0.0, 1.0, 0.0, 0.0])
"""The tool pointing down (xyzw)."""
OVER = (0.0, 1.2, 1.4, -1.0, 0.0, 0.0)
"""A left-arm posture over the bench [rad], the seed the tool-down solve starts from."""
STROKE = math.radians(200.0)
"""One turn of the wrist [rad]: joint 6 runs +-120 degrees, so two turns of thread are four strokes."""
STROKES = 4
"""Three were measured too: 1.67 turns, and the lift then stalls on the thread with the cap still on."""
TURN = 2.5
"""Seconds a stroke takes."""
CLEAR = 0.0025
"""How far off the cap the jaws stop before they close slowly [m]. A jaw that arrives at 85 mm/s
overshoots its soft pad by about a millimetre and knocks a 3 g cap up inside its thread's play."""
HOLD_Z = 0.028
"""Height of the right hand's pads on the bottle [m]: as low as its fingers clear the bench."""
HOLD_TILT = math.radians(5.0)
"""How far the right hand points down as it comes in from the side. Level, its wrist pitch sits on
its limit. At 10 degrees the closing jaws lifted the bottle 1 mm and leaned it 2.8 degrees, and it
then turned 10 degrees a stroke in the fingers; at 20, on a bottle 50 mm shorter, the hand's wrist
camera rose into the left hand's fingers (100 N between the arms, the bottle dragged 25 mm)."""
HOLD_FROM = math.radians(54.0)
"""The level direction the right hand reaches along, from +x: the line from its base to the bottle."""
BACK = 0.09
"""How far back from the bottle the right hand starts its approach [m]."""
LIFT = 0.04
"""How far the left hand carries the cap up at the end [m]; much higher is past its reach over a 249 mm bottle."""


def ease(f: float) -> float:
    return f * f * (3.0 - 2.0 * f)


def routine(sim):
    """From the home pose to the cap in the left hand over a bottle the right hand still holds.

    It drives the two arms' joints itself, one yield a control step. During a stroke the left
    arm's first five joints hold still and only its wrist turns, so nothing sends the hand
    upward: the thread pushes the cap, and the hand with it, up 1.4 to 1.6 mm a stroke.
    """
    hz = float(sim.request.physics.control_hz)
    targets = sim.control.joint_target_q
    arms = {name: sim.arm(name) for name in ("left", "right")}
    grips = {name: sim.gripper(name) for name in ("left", "right")}
    hands = sim.scene.embodiment.hands
    lid, flask = sim["cap"], sim["bottle"]
    now = {name: [np.asarray(sim.state.joint_q.numpy(), dtype=np.float64)[arms[name].arm_coords], 0.0] for name in arms}

    def move(name, q, grip, seconds):
        """Ease one arm's joints to ``q`` and its grip to ``grip`` (0 open, 1 shut)."""
        q0, g0 = now[name][0].copy(), now[name][1]
        steps = max(1, round(seconds * hz))
        for k in range(1, steps + 1):
            f = ease(k / steps)
            arms[name].set_joint_targets(q0 + (np.asarray(q) - q0) * f, targets)
            grips[name].command = g0 + (grip - g0) * f
            grips[name].apply(targets)
            yield
        now[name] = [np.array(q, dtype=np.float64), grip]

    def solve(name, position, quat, seed=None):
        full = np.asarray(sim.state.joint_q.numpy(), dtype=np.float64)
        if seed is not None:
            full[arms[name].arm_coords] = seed
        return arms[name].solve_q_full(full, position, quat)[arms[name].arm_coords]

    # 1. The right hand takes the bottle's body from the side, near its base.
    origin = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    foot = np.asarray(flask.position, dtype=np.float64)  # a privileged read: where the bottle stands
    level = np.array([math.cos(HOLD_FROM), math.sin(HOLD_FROM), 0.0])
    reach = math.cos(HOLD_TILT) * level + math.sin(HOLD_TILT) * np.array([0.0, 0.0, -1.0])
    across = np.cross(level, [0.0, 0.0, -1.0])  # the jaws close level; the pads hang under the wrist
    side = mat_to_quat(np.stack([np.cross(across, reach), across, reach], axis=1))
    grasp = np.array([foot[0], foot[1], origin[2] + HOLD_Z])
    behind = solve("right", grasp - reach * BACK, side)
    at = solve("right", grasp, side, behind)
    radius = float(flask.entity.spec.body[0])
    yield from move("right", behind, 0.0, 2.0)
    yield from move("right", at, 0.0, 1.5)
    yield from move("right", at, 1.0 - (radius + 0.004) / hands["right"].open_width, 0.5)
    yield from move("right", at, 1.0, 1.3)

    # 2. The left hand comes down over the cap, the wrist's roll axis on the cap's own.
    spec = lid.entity.spec
    near = 1.0 - (spec.thread.grip_radius + CLEAR) / hands["left"].open_width

    def over(height, roll):
        middle = np.asarray(lid.position, dtype=np.float64)  # a privileged read: where the cap is now
        middle[2] += spec.grasp_height + height
        q = solve("left", hands["left"].tcp_for("roll_axis", middle, DOWN), DOWN, OVER)
        q[-1] = roll  # the grip point is on joint 6's axis, so turning it moves nothing but the jaws
        return q

    # Seen from above, counter-clockwise unscrews; which way joint 6 turns the hand is read off the arm.
    a, b = (quat_yaw(arms["left"].fk_ee(_with(sim, arms["left"], over(LIFT, r)))[1]) for r in (-0.5, 0.5))
    ccw = 1.0 if math.remainder(b - a, 2.0 * math.pi) > 0.0 else -1.0
    start, end = -ccw * STROKE / 2.0, ccw * STROKE / 2.0
    yield from move("left", over(LIFT, start), 0.0, 1.5)
    yield from move("left", over(0.0, start), 0.0, 1.5)

    # 3. Close, turn, open, turn back; the hand is re-aimed at the cap each time, since it has risen.
    for stroke in range(STROKES):
        yield from move("left", now["left"][0], near, 0.5)
        yield from move("left", now["left"][0], 1.0, 1.1)
        turned = now["left"][0].copy()
        turned[-1] = end
        yield from move("left", turned, 1.0, TURN)
        yield from move("left", turned, 1.0, 0.2)
        if stroke < STROKES - 1:
            yield from move("left", turned, 0.0, 0.5)
            yield from move("left", over(0.0, start), 0.0, 1.2)

    # 4. Lift the cap away.
    yield from move("left", over(LIFT, end), 1.0, 1.5)
    yield from move("left", now["left"][0], 1.0, 0.5)
    print(report(sim), flush=True)


def _with(sim, arm, q) -> np.ndarray:
    """The scene's joints with one arm's replaced: what the arm's forward kinematics takes."""
    full = np.asarray(sim.state.joint_q.numpy(), dtype=np.float64)
    full[arm.arm_coords] = q
    return full


def measures(sim) -> dict:
    """Where the cap and the bottle are now: the cap over the rim, and how far the bottle moved and leaned."""
    spec = sim["bottle"].entity.spec
    origin = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    cap, bottle = (np.asarray(sim[name].pose, dtype=np.float64) for name in ("cap", "bottle"))
    lean = 2.0 * math.asin(min(1.0, float(np.linalg.norm(bottle[3:5]))))
    return {
        "cap_over_rim": float(cap[2] - bottle[2] - spec.rim),
        "cap_off_axis": float(np.linalg.norm(cap[:2] - bottle[:2])),
        "bottle_moved": float(np.linalg.norm(bottle[:2] - origin[:2] - np.array(AT))),
        "bottle_lean_deg": math.degrees(lean),
        "bottle_turned_deg": math.degrees(quat_yaw(bottle[3:7])),
    }


def report(sim) -> str:
    m = measures(sim)
    return (
        f"[cap] {1000 * m['cap_over_rim']:.1f} mm over the rim (>= 30), the bottle moved "
        f"{1000 * m['bottle_moved']:.1f} mm, turned {m['bottle_turned_deg']:.1f} and leans {m['bottle_lean_deg']:.1f} degrees"
    )


def build():
    """The bench, the arms at home with their hands open, and a capped bottle standing free."""
    scene = rss.Scene(cell=CELL, physics=rss.THREAD_60HZ, name="unscrew_the_cap")
    scene.add(rss.robot("yam_bimanual"))
    flask = rss.bottle(name="bottle", at=(*AT, 0.0))
    scene.add(flask)
    scene.add(rss.cap(name="cap", **screwed(flask)))
    return scene.build(control_mode="joint_position")


def main() -> None:
    sim = build()
    x, y, z = (float(v) for v in sim["cap"].position)

    def page(studio):
        studio.aim_camera((x + 0.40, y + 0.12, z + 0.10), (x, y, z - 0.06))  # from the bench's open side

    rss.view(sim, page=page, routines={"Unscrew the cap": routine})


if __name__ == "__main__":
    main()
