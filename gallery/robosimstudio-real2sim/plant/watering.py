"""Contact-only watering; commands go only to the catalog robot's arm and fingers."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

import numpy as np
from scipy.spatial.transform import Rotation

from robosimstudio_studio.routine import grip, hand, wait

ORIGIN = np.array((0.6, 0, 0.75))
AXIS = np.array((1.0, 0.0, 0.0))
LIP = 0.041 * np.cross((0, 0, 1), AXIS) + (0, 0, 0.090)


def measurements(sim):
    p = np.asarray(sim["water"].particle_q)
    cp = np.asarray(sim["cup"].pose)
    rot = Rotation.from_quat(cp[3:]).as_matrix()
    local = (p - cp[:3]) @ rot
    inside_cup = np.all((local >= (-0.041, -0.041, -0.002)) & (local <= (0.041, 0.041, 0.094)), axis=1)
    radial = np.linalg.norm(p[:, :2] - np.array((0.6, -0.014)), axis=1)
    in_pot = (radial < 0.058) & (p[:, 2] >= 0.75 + 0.120) & (p[:, 2] < 0.75 + 0.16) & ~inside_cup
    return {
        "particles": len(p),
        "in_cup": int(inside_cup.sum()),
        "in_pot": int(in_pot.sum()),
        "left_cup": int((~inside_cup).sum()),
        "capture_fraction": float(in_pot.sum() / max(1, (~inside_cup).sum())),
        "below_table": int((p[:, 2] < 0.72).sum()),
        "below_soil": int(((radial < 0.045) & (p[:, 2] < 0.769) & ~inside_cup).sum()),
        "below_table_top": int((p[:, 2] < 0.748).sum()),
        "below_soil_top": int(((radial < 0.055) & (p[:, 2] < 0.8745) & ~inside_cup).sum()),
        "cup_position": cp[:3].tolist(),
        "cup_up": float(rot[2, 2]),
    }


def routine(sim, session):
    from robosimstudio.robots.teleop.lerobot_controller import apply_arm_actions

    panel = session.arms["right"]
    ik = panel.ik
    hz = float(session.env.request.physics.control_hz)
    out = {"marks": []}
    path = []
    recording = False
    tag = "approach"
    park_angles = sim.state.joint_q.numpy()[ik.arm_coords].copy()

    def remember():
        if recording:
            path.append((tag, sim.state.joint_q.numpy()[ik.arm_coords].copy()))

    def mark(name):
        m = measurements(sim)
        m.update(phase=name, time=float(sim.time))
        out["marks"].append(m)
        print(name, json.dumps(m), flush=True)
        (ROOT / "run/watering_measurements.json").write_text(json.dumps(out, indent=2))

    def move(target, quat, seconds=2):
        p0 = np.array(panel.cursor_pos)
        q0 = Rotation.from_quat(panel.cursor_quat)
        q1 = Rotation.from_quat(quat)
        delta = (q1 * q0.inv()).as_rotvec()
        for k in range(1, round(seconds * hz) + 1):
            u = k / (seconds * hz)
            s = u * u * (3 - 2 * u)
            q = (Rotation.from_rotvec(delta * s) * q0).as_quat()
            panel.aim(p0 + (np.array(target) - p0) * s, q, moving=True)
            remember()
            yield
        panel.aim(target, q1.as_quat(), moving=False)
        for _ in range(round(0.35 * hz)):
            remember()
            yield

    def joint_target(angles, closure):
        apply_arm_actions(sim, {"right": [*angles, closure]})
        full = sim.state.joint_q.numpy().copy()
        full[ik.arm_coords] = angles
        position, quat = ik.fk_ee(full)
        panel.aim(position, quat, moving=True)

    def move_joints(target, seconds, closure=0.0):
        initial = sim.state.joint_q.numpy()[ik.arm_coords].copy()
        panel.suspend()
        try:
            for k in range(1, round(seconds * hz) + 1):
                u = k / (seconds * hz)
                s = u * u * (3 - 2 * u)
                joint_target(initial + (np.asarray(target) - initial) * s, closure)
                yield
            yield from wait(session, 0.3)
        finally:
            panel.resume()

    yield from wait(session, 0.5)
    yield from move_joints((-0.645441175, 1.66822743, 1.31104362, -1.21361220, 0, -0.121841945), 1.8)
    start = np.array(sim["cup"].position)
    pick_q = (Rotation.from_euler("z", -90, degrees=True) * Rotation.from_euler("y", 180, degrees=True)).as_quat()
    offset = np.array((-0.036, 0, 0.075))
    yield from move(start + offset, pick_q, 1.7)
    yield from grip(session, "right", 1.0, 1.0)
    mark("grasp")
    recording = True
    tag = "lift"
    yield from move(start + offset + (0, 0, 0.32), pick_q, 2.0)
    mark("lift")
    if sim["cup"].position[2] < start[2] + 0.08:
        raise RuntimeError("Cup did not lift in the fingers")
    cp = np.array(sim["cup"].pose)
    tcp, quat = hand(session, "right")
    hand_R = Rotation.from_quat(quat)
    cup_R = Rotation.from_quat(cp[3:])
    q0 = cup_R.inv() * hand_R
    lip_local = LIP.copy()
    v0 = lip_local - cup_R.inv().apply(tcp - cp[:3])
    initial_base = cp[:3].copy()
    # Remove the small pickup tilt using only the robot's stock wrist servo.
    tag = "level"
    initial_cup_rotvec = cup_R.as_rotvec()
    for k in range(1, round(hz) + 1):
        u = k / hz
        s = u * u * (3 - 2 * u)
        r = Rotation.from_rotvec(initial_cup_rotvec * (1 - s))
        L = initial_base + r.apply(LIP)
        panel.aim(L - r.apply(v0), (r * q0).as_quat(), moving=True)
        remember()
        yield
    lip = initial_base + LIP
    deg = 0.0
    bias = np.zeros(3)
    carry_lip = np.array((0.60, -0.075, 1.20))
    pre_lip = initial_base + np.array((0, 0, 0.08)) + Rotation.from_rotvec(-AXIS * np.deg2rad(45)).apply(LIP)
    plan = [
        ("pretilt", pre_lip, 45.0, 2.0),
        ("carry", carry_lip, 75.0, 2.5),
        ("tip", (0.585, -0.100, 1.085), 85.0, 2.5),
        ("pour", (0.585, -0.100, 1.085), 105.0, 4.0),
        ("drain", (0.585, -0.100, 1.085), 105.0, 1.0),
    ]
    for name, target_tuple, angle, seconds in plan:
        tag = name
        target = np.array(target_tuple)
        controlled_angle = deg
        initial_out = measurements(sim)["left_cup"]
        total_particles = len(sim["water"].particle_q)
        last_out = initial_out
        flow_rate = 0.0
        for k in range(1, round(seconds * hz) + 1):
            u = k / (seconds * hz)
            s = u * u * (3 - 2 * u)
            L = lip + (target - lip) * s
            d = deg + (angle - deg) * s
            if name == "pour":
                desired = initial_out + (total_particles - initial_out) * u
                actual = measurements(sim)["left_cup"]
                flow_rate = 0.7 * flow_rate + 0.3 * (actual - last_out) * hz / total_particles
                last_out = actual
                controlled_angle += np.clip(
                    0.4 * (desired - actual) / total_particles + 0.5 * (1 / seconds - flow_rate) + 0.005, -0.35, 0.12
                )
                d = float(np.clip(controlled_angle, 80, 105))
            # Keep the wrist body above the blade as the cup tips farther.
            if name in ("pour", "drain"):
                L[2] += max(0.0, 0.08 * (np.cos(np.deg2rad(85)) - np.cos(np.deg2rad(d))))
            r = Rotation.from_rotvec(-AXIS * np.deg2rad(d))
            if name == "pretilt":
                L = initial_base + np.array((0, 0, 0.08 * s)) + r.apply(LIP)
            cp = np.array(sim["cup"].pose)
            now = cp[:3] + Rotation.from_quat(cp[3:]).apply(lip_local)
            bias += 0.025 * (L - now)
            n = np.linalg.norm(bias)
            if n > 0.025:
                bias *= 0.025 / n
            panel.aim(L - r.apply(v0) + bias, (r * q0).as_quat(), moving=True)
            remember()
            yield
        lip = target
        deg = d
        panel.aim(panel.cursor_pos, panel.cursor_quat, moving=False)
        for _ in range(round(0.2 * hz)):
            remember()
            yield
        mark(name)
    recording = False
    # Replay the measured arm joint path backwards through the stock servos.
    # This holds the same IK branch; it never writes a pose onto the cup.
    panel.suspend()
    try:
        for k, (phase, angles) in enumerate(reversed(path)):
            if phase in ("pour", "drain") and k % 3:
                continue
            joint_target(angles, 1.0)
            yield
        yield from wait(session, 0.5)
    finally:
        panel.resume()
    mark("set_down")
    yield from grip(session, "right", 0.0, 0.8)
    panel.suspend()
    try:
        for phase, angles in path:
            if phase != "lift":
                continue
            joint_target(angles, 0.0)
            yield
    finally:
        panel.resume()
    yield from move_joints(park_angles, 1.8)
    yield from wait(session, 2.0)
    mark("done")
    final = out["marks"][-1]
    final["put_down_target"] = start.tolist()
    final["put_down_error_mm"] = float(1000 * np.linalg.norm(np.array(final["cup_position"])[:2] - start[:2]))
    (ROOT / "run/watering_measurements.json").write_text(json.dumps(out, indent=2))
    if final["cup_up"] < 0.98:
        raise RuntimeError("Cup did not finish upright")
    if final["put_down_error_mm"] > 20:
        raise RuntimeError("Cup missed its return position")
    if final["left_cup"] < 100:
        raise RuntimeError("Too little water left the cup")
    if final["capture_fraction"] < 0.8:
        raise RuntimeError("Less than 80% capture")
    if final["below_table"] or final["below_soil"]:
        raise RuntimeError("Water crossed a supporting surface")
