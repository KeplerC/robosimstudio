# SPDX-License-Identifier: Apache-2.0
"""Screw in the bulb

A YAM takes a household lamp from its tray, sets it in a lamp holder and screws it in, three and a third turns in six strokes of the wrist, until the lamp's eyelet presses on the holder's centre contact and the lamp comes on: press Run.

Nothing holds the lamp but the fingers and then the thread, a medium Edison
screw cut to its published numbers. The holder is a fixture, screwed to the
bench, and so is the tray the lamp starts in. The parts and the physics are the
``thread`` kind's (``rss.bulb``, ``rss.socket``, ``rss.THREAD_60HZ``).
"""

import math

import numpy as np

import robosimstudio as rss
from robosimstudio.core.math import quat_yaw

CELL = "bench_075"
AT = (-0.0475, 0.31)
"""Where the holder stands [m], in the bench's frame: 0.30 m in front of the left arm's base."""
TRAY = (-0.0475, 0.19)
"""Where the lamp starts, standing on its base in a tray [m]: 120 mm to the holder's side."""
WALL = (0.0135, 0.005, 0.012)
"""The tray's four walls: their distance from the lamp's axis, 0.4 mm off its thread, their thickness and
their height [m], which a hand closed on the glass 81 mm up clears by 20 mm at any turn of its wrist."""

# -- the routine: every number measured on this scene ------------------------------------------
DOWN, UP = np.array([0.0, 1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
"""The tool pointing down (xyzw), and the way up."""
OVER = (0.0, 1.2, 1.4, -1.0, 0.0, 0.0)
"""A left-arm posture over the bench [rad], the seed the tool-down solve starts from."""
STROKE = math.radians(210.0)
"""One turn of the wrist [rad]: joint 6 runs +-120 degrees, so the lamp's 3.3 turns are six strokes,
and home comes two thirds of the way through the sixth."""
STROKES, TURN = 8, 2.5
"""The most strokes it takes, and the seconds one takes."""
GRIP = 0.0018
"""How far above the glass's widest part the hand's grip point goes [m]: the jaws' faces are strips 4 mm
wide, so the glass still meets them flat, and the palm is 2.3 mm clear of the top of the bulb."""
JAW = 0.0036
"""How far inside its finger joint's reading a jaw's face is [m]."""
CLEAR = 0.0025
"""How far off the glass the jaws stop before they close slowly [m]. Jaws that met the glass at 40 mm/s
knocked a lamp half a turn into its thread 0.9 mm up, past the thread's play, for a tenth of a second."""
LOOSE = 0.0004
"""How far each jaw backs off the glass for the wrist's turn back [m]. Let go altogether, the lamp
leaned up to 0.7 degrees in its thread, and one stroke in ten that took hold of it so slipped 3 to 20."""
CARRY, LIFT = 0.05, 0.06
"""How high the lamp is lifted to go from the tray to the holder, its tip 16 mm over the holder's
mouth, and how far above the lamp the hand starts, and backs off to at the end [m]."""
OUT = 3.30
"""Turns from home the lamp is set into the holder at, at least: its thread first meets the
holder's 3.25 turns out, so the first stroke has it caught within 18 degrees."""
PRESS = 0.0003
"""How far under its thread's helix the lamp is set in [m], as a hand pressing lightly brings it. Set in
on the helix, one run in two rode the holder's first turn a millimetre high for a third of a turn."""
SEATED, LIT_AT = 2.0, 0.1
"""The forces on the centre contact that end the turning and that light the lamp [N]. The lamp's weight puts
0.3 N on it as the eyelet lands; a twentieth of a turn later the thread has taken up its play and clamps."""
SLOW, SLOW_FROM = 0.2, 0.15
"""The share of its speed the wrist turns at once the contact reads over ``LIT_AT`` or, by the hand's own
count, with under this many turns to go: the eyelet lands within 0.05 turn of home. At full speed one
control step of a clamping thread took the force on the contact from under 2 N to between 5 and 115 N."""


def ease(f: float) -> float:
    return f * f * (3.0 - 2.0 * f)


def routine(sim):
    """From the home pose to a lamp screwed home, lit, and the hand away.

    It drives the left arm's joints itself, one yield a control step. During a stroke the wrist
    turns and the hand goes down with the thread, a pitch a turn; for the turn back the jaws
    only loosen. The turning slows near home and stops on a force read off the contact probe. The lamp's pose is read from the simulation,
    marked where it is, to aim the hand and to set the lamp in on its thread's helix, which a
    person does by feel.
    """
    hz = float(sim.request.physics.control_hz)
    targets = sim.control.joint_target_q
    arm, grip, hand = sim.arm("left"), sim.gripper("left"), sim.scene.embodiment.hands["left"]
    lamp, holder = sim["bulb"], sim["socket"]
    m = lamp.entity.spec.lamp
    contact = ([holder.entity.contact_shape], [lamp.entity.contact_shape])
    now = [np.asarray(sim.state.joint_q.numpy(), dtype=np.float64)[arm.arm_coords], 0.0]

    def move(q, closed, seconds, pace=None):
        """Ease the arm's joints to ``q`` and the grip to ``closed``.

        ``pace(done)`` says each step how fast to go on, 1 the speed asked for, or ``None`` to
        end the move where it is. Returns whether it was ended that way.
        """
        q0, g0 = now[0].copy(), now[1]
        steps, done = max(1, round(seconds * hz)), 0.0
        while done < 1.0 - 1.0e-9:
            rate = pace(done) if pace is not None else 1.0
            if rate is None:
                now[0] = np.asarray(sim.state.joint_q.numpy(), dtype=np.float64)[arm.arm_coords]
                arm.set_joint_targets(now[0], targets)
                yield
                return True
            done = min(1.0, done + rate / steps)
            f = ease(done)
            arm.set_joint_targets(q0 + (np.asarray(q) - q0) * f, targets)
            grip.command = now[1] = g0 + (closed - g0) * f
            grip.apply(targets)
            yield
        now[0] = np.array(q, dtype=np.float64)
        return False

    def reach(point, roll):
        """The arm's joints that put the hand's roll-axis grip point at ``point``, tool down, the wrist at ``roll``."""
        full = np.asarray(sim.state.joint_q.numpy(), dtype=np.float64)
        full[arm.arm_coords] = OVER
        q = arm.solve_q_full(full, hand.tcp_for("roll_axis", point, DOWN), DOWN)[arm.arm_coords]
        q[-1] = roll  # the grip point is on joint 6's axis, so turning it moves nothing but the jaws
        return q

    def widest() -> np.ndarray:
        """Where the hand's grip point goes on the lamp as it stands now."""
        pose = np.asarray(lamp.pose, dtype=np.float64)  # a privileged read: where the lamp is now
        x, y, z, w = pose[3:7]
        axis = np.array([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)])
        return pose[:3] + (m.equator + GRIP) * axis

    def settle(point, roll):
        """Bring the grip point onto ``point`` by what the arm's own joints say it is off."""
        at, turned_to = arm.fk_ee(np.asarray(sim.state.joint_q.numpy(), dtype=np.float64))
        off = np.asarray(point) - (np.asarray(at) - np.asarray(hand.tcp_for("roll_axis", (0.0, 0.0, 0.0), turned_to)))
        yield from move(reach(np.asarray(point) + off, roll), now[1], 0.3)

    def facing(q) -> float:
        """Which way the hand faces, seen from above, with the arm at ``q``."""
        return quat_yaw(arm.fk_ee(_with(sim, arm, q))[1])

    def pressing() -> float:
        """The force on the holder's centre contact [N], off the contact probe."""
        sim.contacts.update()
        return float(np.linalg.norm(sim.contacts.pairwise_force(*contact)))

    def close():
        """To ``CLEAR`` off the glass, onto it at 6 mm/s, and then the squeeze."""
        touch = 1.0 - (0.5 * m.bulb_diameter + JAW) / hand.open_width
        yield from move(now[0], touch - CLEAR / hand.open_width, 0.5)
        yield from move(now[0], touch + CLEAR / hand.open_width, 0.8)
        yield from move(now[0], 1.0, 0.6)

    # Seen from above, clockwise screws in; which way joint 6 turns the hand is read off the arm.
    aim = widest()
    ccw = 1.0 if math.remainder(facing(reach(aim, 0.5)) - facing(reach(aim, -0.5)), 2.0 * math.pi) > 0.0 else -1.0
    start, end = ccw * STROKE / 2.0, -ccw * STROKE / 2.0
    home = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64) + np.array([*AT, holder.entity.spec.rim])

    # 1. Where on its wrist's turn the hand takes the lamp, so that carried over the holder with the
    #    wrist at the start of a stroke, the lamp's thread is on the holder's helix, OUT turns from home.
    turned = quat_yaw(np.asarray(lamp.pose, dtype=np.float64)[3:7])  # a privileged read: how the lamp stands
    over_holder = home + (m.pitch * OUT + m.equator + GRIP) * UP
    for more in (0.0, 0.1, 0.2, 0.3, 0.4):  # a wrist that cannot get there takes it a tenth of a turn on
        swing = facing(reach(over_holder, start)) - facing(reach(aim, 0.0))
        take = ccw * math.remainder(turned + swing - 2.0 * math.pi * (OUT + more), 2.0 * math.pi)
        if abs(take) <= math.radians(115.0):
            break

    # 2. Take the lamp from its tray by the glass at its widest.
    yield from move(reach(aim + LIFT * UP, take), 0.0, 2.0)
    yield from move(reach(aim, take), 0.0, 1.5)
    yield from settle(aim, take)
    yield from close()
    # Where its jaws stopped on the glass, by the hand's own reading, and a little short of it.
    loose = grip.closed_fraction(sim.state) - LOOSE / hand.open_width
    yield from move(reach(aim + CARRY * UP, take), 1.0, 1.2)

    # 3. Carry it over the holder and set it in: its eyelet as far above home as its turn says the
    #    thread's helix is there, so the two threads meet start to start.
    def set_eyelet(point, seconds):
        """Send the hand so the lamp's eyelet comes to ``point``, by what the lamp is off it now. A privileged read."""
        off = point - np.asarray(lamp.pose, dtype=np.float64)[:3]  # a privileged read: where its eyelet is now
        aim[:] = aim + off
        yield from move(reach(aim, start), 1.0, seconds)

    aim = over_holder + CARRY * UP
    yield from move(reach(aim, start), 1.0, 2.5)
    yield from set_eyelet(home + (m.pitch * OUT + CARRY) * UP, 0.5)
    turned = quat_yaw(np.asarray(lamp.pose, dtype=np.float64)[3:7])  # a privileged read: the lamp's turn in the hand
    out = OUT - 0.03 + (turned / (2.0 * math.pi) - OUT + 0.03) % 1.0
    yield from set_eyelet(home + (m.pitch * out - PRESS) * UP, 1.5)
    yield from move(now[0], 1.0, 0.3)  # let the arm come to rest before the last correction
    yield from set_eyelet(home + (m.pitch * out - PRESS) * UP, 0.4)

    # 4. Turn, loosen, turn back, tighten: the hand goes down with the thread as it turns, and its jaws only
    #    just leave the glass for the turn back. ``left`` is its own count of the turns still to go.
    left, drop = out, m.pitch * STROKE / (2.0 * math.pi) * UP
    for stroke in range(STROKES):
        if stroke:
            yield from move(reach(aim, start), loose, 1.0)
            yield from move(now[0], 1.0, 0.5)

        def pace(done, left=left):
            force = pressing()
            if force > SEATED:
                return None
            return SLOW if force > LIT_AT or left - ease(done) * STROKE / (2.0 * math.pi) < SLOW_FROM else 1.0

        home_now = yield from move(reach(aim - drop, end), 1.0, TURN, pace)
        aim, left = aim - drop, left - STROKE / (2.0 * math.pi)
        yield from move(now[0], 1.0, 0.2)
        if home_now:
            break
        yield from move(now[0], loose, 0.3)
    yield from move(now[0], 0.0, 0.4)
    yield from move(reach(widest() + LIFT * UP, now[0][-1]), 0.0, 1.2)
    print(report(sim), flush=True)


def circuit(sim):
    """What to call each control step: the lamp is lit while its eyelet presses on the holder's centre contact."""
    lamp, holder = sim["bulb"], sim["socket"]
    contact = ([holder.entity.contact_shape], [lamp.entity.contact_shape])

    def step() -> None:
        sim.contacts.update()
        lamp.entity.light(float(np.linalg.norm(sim.contacts.pairwise_force(*contact))) > LIT_AT)

    return step


def _with(sim, arm, q) -> np.ndarray:
    """The scene's joints with one arm's replaced: what the arm's forward kinematics takes."""
    full = np.asarray(sim.state.joint_q.numpy(), dtype=np.float64)
    full[arm.arm_coords] = q
    return full


def measures(sim) -> dict:
    """Where the lamp is now and what its eyelet carries: home is an eyelet on the centre contact, a force through it, and a lit lamp."""
    lamp, holder = sim["bulb"], sim["socket"]
    origin = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    pose = np.asarray(lamp.pose, dtype=np.float64)
    sim.contacts.update()
    force = sim.contacts.pairwise_force([holder.entity.contact_shape], [lamp.entity.contact_shape])
    return {
        "eyelet_over_contact": float(pose[2] - origin[2] - holder.entity.spec.rim),
        "off_axis": float(np.linalg.norm(pose[:2] - origin[:2] - np.array(AT))),
        "contact_force": float(np.linalg.norm(force)),
        "lean_deg": math.degrees(2.0 * math.asin(min(1.0, float(np.linalg.norm(pose[3:5]))))),
        "lit": bool(lamp.entity.lit),
    }


def report(sim) -> str:
    m = measures(sim)
    return (
        f"[bulb] eyelet {1000 * m['eyelet_over_contact']:.2f} mm over the centre contact, {m['contact_force']:.2f} N "
        f"on it, leaning {m['lean_deg']:.1f} degrees, {'lit' if m['lit'] else 'dark'}"
    )


def build():
    """The bench, the arms at home, the holder on the bench and the lamp standing in its tray."""
    scene = rss.Scene(cell=CELL, physics=rss.THREAD_60HZ, name="screw_in_the_bulb")
    scene.add(rss.robot("yam_bimanual"))
    scene.add(rss.socket(name="socket", at=(*AT, 0.0)))
    inside, thick, tall = WALL
    for k, (dx, dy) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):  # the tray: four walls round the lamp's base
        along = inside + thick
        half = (thick / 2, along, tall / 2) if dx else (along, thick / 2, tall / 2)
        at = (TRAY[0] + dx * (inside + thick / 2), TRAY[1] + dy * (inside + thick / 2), tall / 2)
        scene.add(rss.box(name=f"tray_{k}", half=half, at=at, static=True, color=(0.25, 0.25, 0.28)))
    scene.add(rss.bulb(name="bulb", at=(*TRAY, 1.0e-4)))
    return scene.build(control_mode="joint_position")


def main() -> None:
    sim = build()
    origin = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    x, y, z = origin + np.array([*AT, 0.0])
    lamp, light = sim["bulb"], circuit(sim)
    glass = (x, y, z + sim["socket"].entity.spec.rim + lamp.entity.spec.lamp.equator)

    def page(studio):
        studio.aim_camera((x + 0.38, y - 0.17, z + 0.20), (x, y - 0.05, z + 0.075))  # from the bench's open side
        # The page's own picture of a lit lamp: a warm light in the glass, and a fainter one before it.
        warm = {"color": (255, 196, 110), "cast_shadow": False, "visible": False}
        glows = [studio.scene.add_light_point("/lamp_glow", intensity=0.4, position=glass, **warm)]
        before = (glass[0] + 0.066, glass[1] - 0.03, glass[2] + 0.035)
        glows += [studio.scene.add_light_point("/lamp_face", intensity=0.12, position=before, **warm)]

        def step() -> None:
            light()
            for glow in glows:
                glow.visible = bool(lamp.entity.lit)

        studio.each_step(step)

    rss.view(sim, page=page, routines={"Screw in the bulb": routine})


if __name__ == "__main__":
    main()
