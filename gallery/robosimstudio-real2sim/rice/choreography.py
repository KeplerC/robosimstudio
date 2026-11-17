"""Contact-only bimanual cleanup; all task measurements use the pan's live frame."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robosimstudio_studio.routine import wait

RUN = Path(__file__).resolve().parent / "run"

DOWN = np.array([0.0, 1.0, 0.0, 0.0])


def measure(sim):
    pose = np.asarray(sim["dustpan"].pose)
    q = np.asarray(sim["rice"].particle_q)
    p = (q - pose[:3]) @ Rotation.from_quat(pose[3:]).as_matrix()
    ps = sim["dustpan"].spec
    top = ps.lip + (ps.floor - ps.lip) * np.clip(p[:, 1] / ps.ramp, 0, 1)
    inside = (
        (np.abs(p[:, 0]) < 0.092) & (p[:, 1] > 0) & (p[:, 1] < 0.152) & (p[:, 2] > top - 0.0015) & (p[:, 2] < 0.065)
    )
    result = {
        "in_pan_pct": float(100 * inside.mean()),
        "grains": len(q),
        "lifted_mm": float(1000 * (pose[2] - 0.75)),
        "pan_pose": pose.tolist(),
        "under_n": int((q[:, 2] < 0.744).sum()),
    }
    result["pass"] = bool(result["in_pan_pct"] >= 75 and result["lifted_mm"] > 25)
    return result


def move(session, seconds, goals, quats=None, ease=True):
    starts = {n: np.array(session.arms[n].cursor_pos) for n in goals}
    for k in range(1, max(1, round(seconds * 60)) + 1):
        u = k / max(1, round(seconds * 60))
        f = u * u * (3 - 2 * u) if ease else u
        for n, end in goals.items():
            q = None if quats is None else quats.get(n)
            if callable(q):
                q = q(f)
            session.arms[n].aim(starts[n] + f * (np.asarray(end) - starts[n]), q, moving=True)
        yield
    for n in goals:
        session.arms[n].dragging = False


def routine(sim, session):
    RUN.mkdir(exist_ok=True)
    events = []
    ps = sim["dustpan"].spec
    t0 = sim.time
    gripped_z = None

    def mark(name):
        nonlocal gripped_z
        left_tcp = np.asarray(session.arms["left"].ik.ee_pose(session.env.state)[0])
        if name == "gripped":
            gripped_z = float(left_tcp[2])
        row = dict(phase=name, t=float(sim.time - t0), **measure(sim))
        row["objects"] = {n: list(map(float, sim[n].pose)) for n in ("cup", "pusher")}
        row["width_mm"] = {n: float(session.env.gripper(n).width(session.env.state) * 1000) for n in ("left", "right")}
        row["left_tcp"] = left_tcp.tolist()
        row["grip_rise_mm"] = None if gripped_z is None else float(1000 * (left_tcp[2] - gripped_z))
        events.append(row)
        (RUN / "progress.json").write_text(json.dumps(events, indent=2))
        print(name, row, flush=True)

    yield from wait(session, 0.3)
    mark("start")
    pan = np.asarray(sim["dustpan"].pose)
    pg = pan[:3] + Rotation.from_quat(pan[3:]).apply([-0.065, 0.16, 0.037])
    pu = np.asarray(sim["pusher"].position)
    rg = pu + np.array([0.03, 0, 0])
    yield from move(
        session,
        2.0,
        {"left": pg + np.array([0, 0, 0.08]), "right": rg + np.array([0, 0, 0.08])},
        {
            "left": (Rotation.from_euler("y", -20, degrees=True) * Rotation.from_quat(DOWN)).as_quat(),
            "right": (Rotation.from_euler("y", 35, degrees=True) * Rotation.from_quat([1, 0, 0, 0])).as_quat(),
        },
    )
    yield from wait(session, 0.4)
    yield from move(session, 1.0, {"left": pg, "right": rg})
    yield from wait(session, 0.4)
    for arm in session.arms.values():
        arm.grip_request = 1.0
    yield from wait(session, 1.0)
    mark("gripped")
    for name in ("left", "right"):
        width = float(session.env.gripper(name).width(session.env.state))
        if not 0.015 < width < 0.026:
            raise RuntimeError(f"{name} pinch missed the 20 mm face: {width * 1000:.1f} mm")
    lip = float(sim["dustpan"].position[1])
    start = np.array(session.arms["right"].cursor_pos)
    end = lip + 0.08
    steps = round((end - start[1]) / 0.045 * 60)
    for k in range(1, steps + 1):
        y = start[1] + (end - start[1]) * k / steps
        d = float(sim["pusher"].position[1]) + 0.010 - float(sim["dustpan"].position[1])
        rotation = Rotation.from_quat(np.asarray(sim["pusher"].pose)[3:])
        front_bottom = rotation.apply([[-0.085, 0.010, -0.020], [0.085, 0.010, -0.020]])[:, 2].min()
        tilt_drop = max(0.0, -0.020 - front_bottom)
        rise = (ps.lip + (ps.floor - ps.lip) * np.clip(d / ps.ramp, 0, 1) + 0.001 + tilt_drop) if d > 0 else 0
        session.arms["right"].aim([start[0], y, start[2] + rise], moving=True)
        yield
    session.arms["right"].dragging = False
    yield from wait(session, 0.5)
    mark("swept")
    if sim["pusher"].position[1] - sim["dustpan"].position[1] < 0.055:
        raise RuntimeError("Pusher stalled before the flat pan floor")
    p = np.array(session.arms["right"].cursor_pos)
    yield from move(session, 0.72, {"right": p + np.array([0, 0, 0.08])})
    # Return along the cleared stroke lane before the left hand lifts the pan.
    yield from move(session, 1.2, {"right": rg + np.array([0, 0, 0.09])})
    yield from move(session, 0.8, {"right": rg + np.array([0, 0, 0.003])})
    session.arms["right"].grip_request = 0.0
    yield from wait(session, 0.6)
    mark("released")
    yield from move(session, 0.7, {"right": rg + np.array([0, 0, 0.09])})
    home_p, home_q = session.arms["right"].ik.home_pose
    l = np.array(session.arms["left"].cursor_pos)
    q0 = np.array(session.arms["left"].cursor_quat)

    def tipped(u):
        return (Rotation.from_rotvec([-np.deg2rad(25) * u, 0, 0]) * Rotation.from_quat(q0)).as_quat()

    yield from move(
        session, 1.6, {"left": l + np.array([0, 0, 0.06]), "right": home_p}, {"left": tipped, "right": home_q}
    )
    yield from wait(session, 0.5)
    mark("lifted")
    l = np.array(session.arms["left"].cursor_pos)
    yield from move(session, 1.2, {"left": l + np.array([0, -0.07, 0])})
    mark("presented")
    yield from wait(session, 1.0)
    mark("done")
    result = measure(sim)
    result["grip_rise_mm"] = events[-1]["grip_rise_mm"]
    result["duration_seconds"] = events[-1]["t"]
    (RUN / "result.json").write_text(json.dumps(result, indent=2))
    if not result["pass"]:
        raise RuntimeError(
            f"Cleanup below threshold: {result['in_pan_pct']:.1f}% in pan; lift {result['lifted_mm']:.1f} mm"
        )
