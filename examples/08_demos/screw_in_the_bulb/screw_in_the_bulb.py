# SPDX-License-Identifier: Apache-2.0
"""Screw in the bulb

A YAM screws a lamp into its holder: the left hand takes the glass, turns it clockwise two turns in four strokes of the wrist, and stops when the lamp's base contact presses on the holder's: press Run.

The lamp starts standing loose in the holder's mouth, and nothing holds it but
the fingers and then the thread. The holder is a fixture, screwed to its post.
The parts and the physics are the ``thread`` kind's (``rss.bulb``,
``rss.socket``, ``rss.THREAD_60HZ``).
"""

import math

import numpy as np

import robosimstudio as rss
from robosimstudio.core.math import quat_yaw
from robosimstudio_materials.thread import screwed

CELL = "bench_075"
AT = (-0.0475, 0.31)
"""Where the holder's post stands [m], in the bench's frame: 0.30 m in front of the left arm's base."""
POST = 0.0885
"""The post's height [m]: the holder's floor is 90 mm over the bench."""
OUT = 2.15
"""Turns the lamp starts backed off by: past the thread's two, so it stands on the ridge's end, not held."""

# -- the routine: every number measured on this scene ------------------------------------------
DOWN = np.array([0.0, 1.0, 0.0, 0.0])
"""The tool pointing down (xyzw)."""
OVER = (0.0, 1.2, 1.4, -1.0, 0.0, 0.0)
"""A left-arm posture over the bench [rad], the seed the tool-down solve starts from."""
STROKE = math.radians(200.0)
"""One turn of the wrist [rad]: joint 6 runs +-120 degrees."""
STROKES = 6
"""The most strokes it takes; the base contact stops it in the fourth."""
TURN = 3.5
"""Seconds a stroke takes: slow enough that the one control step the hand turns on after the contact
closes, a thirtieth of a millimetre of thread, does not drive the lamp hard into the pad."""
CLEAR = 0.0025
"""How far off the glass the jaws stop before they close slowly [m]."""
PRESS = 0.0015
"""How far under the lamp the hand is aimed once it holds it [m]: about 1.5 N down, which keeps the
loose lamp on the thread until the first ridge catches."""
SEATED = 2.0
"""The force on the base contact that ends the turning [N]. The hand's press and the lamp's weight
put about 1.7 N on the pad as the tip lands, so it is reached there or, once the thread has taken
up its play a sixth of a turn later, as the thread starts to clamp the lamp down."""
LIFT = 0.06
"""How far the hand backs off upward at the end [m]."""


def ease(f: float) -> float:
    return f * f * (3.0 - 2.0 * f)


def routine(sim):
    """From the home pose to a lamp screwed home and the hand away.

    It drives the left arm's joints itself, one yield a control step. During a stroke the arm's
    first five joints hold still and only its wrist turns: the thread draws the lamp, and the
    hand with it, down 1.7 mm a stroke. The turning stops on a force read off the contact probe.
    """
    hz = float(sim.request.physics.control_hz)
    targets = sim.control.joint_target_q
    arm, grip, hand = sim.arm("left"), sim.gripper("left"), sim.scene.embodiment.hands["left"]
    lamp, holder = sim["bulb"], sim["socket"]
    spec = lamp.entity.spec
    contact = ([holder.entity.contact_shape], [lamp.entity.contact_shape])
    now = [np.asarray(sim.state.joint_q.numpy(), dtype=np.float64)[arm.arm_coords], 0.0]

    def move(q, closed, seconds, until=None):
        """Ease the arm's joints to ``q`` and the grip to ``closed``; ``until()`` true ends it where it is."""
        q0, g0 = now[0].copy(), now[1]
        steps = max(1, round(seconds * hz))
        for k in range(1, steps + 1):
            f = ease(k / steps)
            arm.set_joint_targets(q0 + (np.asarray(q) - q0) * f, targets)
            grip.command = g0 + (closed - g0) * f
            grip.apply(targets)
            yield
            if until is not None and until():
                q = np.asarray(sim.state.joint_q.numpy(), dtype=np.float64)[arm.arm_coords]
                arm.set_joint_targets(q, targets)
                break
        now[0], now[1] = np.array(q, dtype=np.float64), closed

    def over(height, roll):
        middle = np.asarray(lamp.position, dtype=np.float64)  # a privileged read: where the lamp is now
        middle[2] += spec.grasp_height + height
        full = np.asarray(sim.state.joint_q.numpy(), dtype=np.float64)
        full[arm.arm_coords] = OVER
        q = arm.solve_q_full(full, hand.tcp_for("roll_axis", middle, DOWN), DOWN)[arm.arm_coords]
        q[-1] = roll  # the grip point is on joint 6's axis, so turning it moves nothing but the jaws
        return q

    def seated() -> bool:
        sim.contacts.update()
        return float(np.linalg.norm(sim.contacts.pairwise_force(*contact))) > SEATED

    # Seen from above, clockwise screws in; which way joint 6 turns the hand is read off the arm.
    a, b = (quat_yaw(arm.fk_ee(_with(sim, arm, over(LIFT, r)))[1]) for r in (-0.5, 0.5))
    ccw = 1.0 if math.remainder(b - a, 2.0 * math.pi) > 0.0 else -1.0
    start, end = ccw * STROKE / 2.0, -ccw * STROKE / 2.0
    near = 1.0 - (float(spec.body[0]) + CLEAR) / hand.open_width
    yield from move(over(LIFT, start), 0.0, 1.5)
    yield from move(over(0.0, start), 0.0, 1.5)
    for _ in range(STROKES):
        yield from move(now[0], near, 0.5)
        yield from move(now[0], 1.0, 1.1)
        yield from move(over(-PRESS, start), 1.0, 0.3)
        turned = now[0].copy()
        turned[-1] = end
        yield from move(turned, 1.0, TURN, until=seated)
        home = seated()
        yield from move(now[0], 1.0, 0.3)
        yield from move(now[0], 0.0, 0.5)
        if home:
            break
        yield from move(over(0.0, start), 0.0, 1.2)
    yield from move(over(LIFT, now[0][-1]), 0.0, 1.2)
    print(report(sim), flush=True)


def _with(sim, arm, q) -> np.ndarray:
    """The scene's joints with one arm's replaced: what the arm's forward kinematics takes."""
    full = np.asarray(sim.state.joint_q.numpy(), dtype=np.float64)
    full[arm.arm_coords] = q
    return full


def measures(sim) -> dict:
    """Where the lamp is now and what its base contact carries: seated is a tip on the pad and a force through it."""
    lamp, holder = sim["bulb"], sim["socket"]
    origin = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    pose = np.asarray(lamp.pose, dtype=np.float64)
    sim.contacts.update()
    force = sim.contacts.pairwise_force([holder.entity.contact_shape], [lamp.entity.contact_shape])
    return {
        "tip_over_floor": float(pose[2] - origin[2] - POST - holder.entity.spec.rim),
        "pad_height": float(holder.entity.spec.contact[1]),
        "contact_force": float(np.linalg.norm(force)),
        "lean_deg": math.degrees(2.0 * math.asin(min(1.0, float(np.linalg.norm(pose[3:5]))))),
    }


def report(sim) -> str:
    m = measures(sim)
    return (
        f"[bulb] tip {1000 * m['tip_over_floor']:.2f} mm over the holder's floor (the pad stands "
        f"{1000 * m['pad_height']:.2f}), {m['contact_force']:.2f} N on the base contact, leaning {m['lean_deg']:.1f} degrees"
    )


def build():
    """The bench, the arms at home, the holder on its post and the lamp standing loose in its mouth."""
    scene = rss.Scene(cell=CELL, physics=rss.THREAD_60HZ, name="screw_in_the_bulb")
    scene.add(rss.robot("yam_bimanual"))
    scene.add(rss.post(name="stand", radius=0.013, half_height=POST / 2, at=(*AT, POST / 2), color=(0.25, 0.25, 0.28)))
    holder = rss.socket(name="socket", at=(*AT, POST))
    scene.add(holder)
    scene.add(rss.bulb(name="bulb", **screwed(holder, turns_out=OUT)))
    return scene.build(control_mode="joint_position")


def main() -> None:
    sim = build()
    x, y, z = (float(v) for v in sim["bulb"].position)

    def page(studio):
        studio.aim_camera((x + 0.36, y - 0.14, z + 0.10), (x, y, z))  # from the bench's open side

    rss.view(sim, page=page, routines={"Screw in the bulb": routine})


if __name__ == "__main__":
    main()
