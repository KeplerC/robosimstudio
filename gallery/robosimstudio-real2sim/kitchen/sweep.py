"""Contact-only bimanual coffee sweep, ported from the October take."""

import json
import math
from pathlib import Path

import numpy as np

from robosimstudio.core.math import quat_from_axis_angle, quat_multiply, quat_to_mat
from robosimstudio_studio.routine import hand, wait

PARK_PATH = np.array(
    ((1.57, 3.15, 3.15, 0, 0, 0), (1.57, 2.2, 2, 0, 0, 0), (0.8, 1.4, 1.3, 0, 0, 0), (0, 1.047, 1.047, 0, 0, 0))
)
MIRROR = np.array((-1, 1, 1, -1, 1, -1))
DOWN = np.array((0.0, 1.0, 0.0, 0.0))


def measure(sim):
    pose = np.asarray(sim["dustpan"].pose)
    rot = quat_to_mat(pose[3:])

    def inside(name, half):
        p = (np.asarray(sim[name].particle_q) - pose[:3]) @ rot
        floor = 0.002 + 0.008 * np.clip(p[:, 1] / 0.035, 0, 1)
        return (
            (np.abs(p[:, 0]) < 0.092) & (p[:, 1] > 0) & (p[:, 1] < 0.152) & (p[:, 2] > floor - half) & (p[:, 2] < 0.065)
        )

    g = inside("grounds", 0.0015)
    b = inside("beans", 0.003)
    return {
        "in_pan_pct": 100 * float(g.mean()),
        "beans_in_pan": int(b.sum()),
        "beans": len(b),
        "grounds": len(g),
        "lifted_mm": 1000 * (pose[2] - 0.75),
        "pan_pose": pose.tolist(),
    }


def routine(sim, session):
    hz = float(session.env.request.physics.control_hz)
    L, R = session.arms["left"], session.arms["right"]
    events = []
    start_time = sim.time

    def tool_reading():
        hp, hq = hand(session, "right")
        pose = np.asarray(sim["pusher"].pose)
        return {
            "right_target": np.asarray(R.cursor_pos).tolist(),
            "pusher_pose": pose.tolist(),
            "hand_pose": np.r_[hp, hq].tolist(),
            "pusher_in_hand_mm": ((pose[:3] - hp) @ quat_to_mat(hq) * 1000).tolist(),
        }

    def mark(name):
        row = dict(phase=name, t=round(sim.time - start_time, 3), **measure(sim), **tool_reading())
        events.append(row)
        print("[sweep]", row, flush=True)

    def move(seconds, plan):
        n = max(1, round(seconds * hz))
        starts = {a: np.array(a.cursor_pos) for a in plan}
        for k in range(1, n + 1):
            u = k / n
            e = u * u * (3 - 2 * u)
            for a, (target, q) in plan.items():
                pos = starts[a] + (np.asarray(target) - starts[a]) * e
                a.aim(pos, q(e) if callable(q) else q, moving=k < n)
            yield

    yield from wait(session, 1.83)
    mark("moves")
    L.suspend()
    R.suspend()
    dist = np.linalg.norm(np.diff(PARK_PATH, axis=0), axis=1)
    for k in range(1, round(2.4 * hz) + 1):
        u = k / round(2.4 * hz)
        s = u * u * (3 - 2 * u) * dist.sum()
        for i, d in enumerate(dist):
            if s <= d or i == len(dist) - 1:
                q = PARK_PATH[i] + (PARK_PATH[i + 1] - PARK_PATH[i]) * min(1, s / d)
                break
            s -= d
        t = sim.control.joint_target_q.numpy()
        t[list(L.ik.arm_coords)] = q
        t[list(R.ik.arm_coords)] = q * MIRROR
        sim.control.joint_target_q.assign(t)
        yield
    L.resume()
    R.resume()
    yield from wait(session, 0.2)
    pose = np.asarray(sim["dustpan"].pose)
    rot = quat_to_mat(pose[3:])
    pg = pose[:3] + rot @ np.array((-0.05, 0.16, 0.037))
    pp = np.asarray(sim["pusher"].position) + np.array((0.03, 0, 0))
    pp[2] = 0.767
    up = np.array((0, 0, 0.08))
    mark("reach")
    yield from move(2, {L: (pg + up, DOWN), R: (pp + up, DOWN)})
    yield from wait(session, 0.4)
    yield from move(1, {L: (pg, DOWN), R: (pp, DOWN)})
    yield from wait(session, 0.4)
    L.grip_request = 1.0
    R.grip_request = 1.0
    yield from wait(session, 1.0)
    print("grip widths", {n: sim.gripper(n).width(sim.state) for n in ("left", "right")}, flush=True)
    print("hand positions", {n: hand(session, n)[0].tolist() for n in ("left", "right")}, flush=True)
    mark("sweep")
    lip_y = float(sim["dustpan"].position[1])
    start = np.array(R.cursor_pos)
    end_y = lip_y + 0.09
    n = max(1, math.ceil((end_y - start[1]) / (0.12 / hz)))
    for k in range(1, n + 1):
        y = start[1] + (end_y - start[1]) * k / n
        d = y + 0.01 - lip_y
        z = start[2] + (0.002 + 0.008 * min(d / 0.035, 1) + 0.003 if d > 0 else 0)
        R.aim((start[0], y, z), DOWN, moving=k < n)
        if k % 30 == 0:
            mark("sweep_probe")
        yield
    yield from wait(session, 0.5)
    print("pusher after sweep", sim["pusher"].pose, flush=True)
    mark("swept")
    p = np.array(R.cursor_pos)
    yield from move(0.72, {R: (p + np.array((0, 0, 0.08)), DOWN)})
    yield from move(0.48, {R: (p + np.array((0, -0.135, 0.08)), DOWN)})
    l = np.array(L.cursor_pos)
    q = np.array(L.cursor_quat)
    mark("lift")
    yield from move(
        1.6,
        {
            L: (
                l + np.array((0, 0, 0.09)),
                lambda u: quat_multiply(
                    quat_from_axis_angle(np.array((0.0, 1.0, 0.0)), -math.radians(12) * u),
                    quat_multiply(quat_from_axis_angle(np.array((1.0, 0.0, 0.0)), -math.radians(5) * u), q),
                ),
            )
        },
    )
    yield from wait(session, 4.0)
    mark("done")
    result = measure(sim)
    Path(__file__).with_name("run").joinpath("sweep_result.json").write_text(
        json.dumps({"result": result, "events": events}, indent=2)
    )
    if result["in_pan_pct"] < 75 or result["lifted_mm"] <= 25:
        raise RuntimeError(f"Sweep missed the 75% / 25 mm bar: {result}")
