"""Robot-only Cartesian trajectories. No object state or constraint is written."""

# ruff: noqa: RUF005
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class Drive:
    def __init__(self, sim):
        self.sim = sim
        self.seed = sim.state.joint_q.numpy().copy()
        self.action = np.array(sim.idle_action("joint_position"), float)
        self.grips = {"left": 0.0, "right": 0.0}
        self.start_carton = sim["carton"].position.copy()
        self.max_shift = 0.0
        self.failures = []
        self.start_time = sim.time

    @property
    def elapsed(self):
        return self.sim.time - self.start_time

    def tick(self):
        self.sim.step(self.action, mode="joint_position")
        self.max_shift = max(
            self.max_shift, float(np.linalg.norm(self.sim["carton"].position[:2] - self.start_carton[:2]))
        )

    def hold(self, seconds):
        for _ in range(round(seconds * 50)):
            self.tick()
            yield

    def grip(self, name, value, seconds=0.5):
        slot = 0 if name == "left" else 1
        self.grips[name] = value
        self.action[slot * 7 + 6] = value
        yield from self.hold(seconds)

    def both(self, targets, seconds):
        starts = {name: self.sim.arm(name).ee_pose(self.sim.state) for name in targets}
        rots = {name: Slerp([0, 1], Rotation.from_quat([starts[name][1], p[1]])) for name, p in targets.items()}
        n = round(seconds * 50)
        last = 0
        prev = self.seed.copy()
        for k in [*range(5, n, 5), n]:
            f = k / n
            f = f * f * (3 - 2 * f)
            sol = self.seed.copy()
            for name, (p, _q) in targets.items():
                ik = self.sim.arm(name)
                start = starts[name][0]
                sol = ik.solve_q_full(sol, start + (np.array(p) - start) * f, rots[name](f).as_quat())
            for j in range(last + 1, k + 1):
                t = (j - last) / (k - last)
                for name in targets:
                    slot = 0 if name == "left" else 1
                    self.action[slot * 7 : slot * 7 + 6] = ((1 - t) * prev + t * sol)[
                        list(self.sim.arm(name).arm_coords)
                    ]
                self.tick()
                yield
            self.seed = sol
            prev = sol
            last = k

    def move(self, name, target, quat, seconds, gate=True):
        ik = self.sim.arm(name)
        start, q0 = ik.ee_pose(self.sim.state)
        target = np.asarray(target)
        n = max(2, round(seconds * 50))
        slot = 0 if name == "left" else 1
        rots = Slerp([0, 1], Rotation.from_quat([q0, quat]))
        last = 0
        prev = self.seed.copy()
        worst = [0.0, 0.0]
        for k in [*range(5, n, 5), n]:
            f = k / n
            f = f * f * (3 - 2 * f)
            p = start + (target - start) * f
            q = rots(f).as_quat()
            sol = ik.solve_q_full(self.seed, p, q)
            err = ik.pose_errors(sol, p, q)
            worst = np.maximum(worst, err)
            if gate and err[0] > 0.012:
                raise RuntimeError(f"{name} unreachable: {err[0] * 1000:.1f} mm at {p}")
            for j in range(last + 1, k + 1):
                t = (j - last) / (k - last)
                self.action[slot * 7 : slot * 7 + 6] = ((1 - t) * prev + t * sol)[list(ik.arm_coords)]
                self.tick()
                yield
            self.seed = sol
            prev = sol
            last = k
        # Per-leg residuals are gated above; task readings are printed by the routine.


def hand(angle):
    return (
        Rotation.from_euler("x", angle, degrees=True)
        * Rotation.from_euler("z", 90, degrees=True)
        * Rotation.from_euler("y", 180, degrees=True)
    ).as_quat()


LEFT_SEED = [0.2686, 1.77127, 1.2411, -1.2597, 0.6684, -1.2258]
HOOK_SEED = [-0.9332892, 1.6969256, 1.5033574, -1.1638916, -0.27842447, 0.66751033]
RIGHT_HOOK_SEED = [0.5551, 2.4451, 2.2012, -1.4972, -0.3062, -0.5184]


def joint_stage(d, targets, seconds, seeds=None):
    """Clear-space changes of wrist posture, with a measured IK branch per arm."""
    seeds = seeds or {}
    full = d.sim.state.joint_q.numpy().copy()
    start = d.action.copy()
    for name, (p, q) in targets.items():
        ik = d.sim.arm(name)
        if name in seeds:
            full[list(ik.arm_coords)] = seeds[name]
        full = ik.solve_q_full(full, p, q)
        error = ik.pose_errors(full, p, q)
        if error[0] > 0.008:
            raise RuntimeError(f"{name} stage unreachable: {error[0] * 1000:.1f} mm")
    n = round(seconds * 50)
    for k in range(n):
        f = (k + 1) / n
        f = f * f * (3 - 2 * f)
        for name in targets:
            slot = 0 if name == "left" else 1
            coords = list(d.sim.arm(name).arm_coords)
            d.action[slot * 7 : slot * 7 + 6] = start[slot * 7 : slot * 7 + 6] * (1 - f) + full[coords] * f
        d.tick()
        yield
    d.seed = full


def flap_angle(sim, which):
    e = sim["carton"].entity
    return float(sim.state.joint_q.numpy()[int(sim.model.joint_q_start.numpy()[e.hinges[which]])])


def flap_edge(sim, which, angle=None):
    root = sim["carton"].pose
    rr = Rotation.from_quat(root[3:])
    sign = -1 if which == 0 else 1
    h = root[:3] + rr.apply([0, sign * 0.048, 0.0775])
    rot = rr * Rotation.from_euler("z", 180 if which else 0, degrees=True)
    rot *= Rotation.from_euler("x", flap_angle(sim, which) if angle is None else angle)
    return h + rot.apply([0.062, 0.042, 0])


LEVEL_LEFT = [-0.07258, 1.71125, 0.19534, 1.51597, 1.49822, 0.34901]
LEVEL_RIGHT = [0.377043, 2.32455, 2.14464, -1.563195, -0.859983, -1.447463]


def hook_pose(sim, which, theta=None, entry=False):
    a = flap_angle(sim, which) if theta is None else theta
    # Fingers point along the seam from its short end. Roll clears the other finger.
    roll = 70 + 20 * np.clip(np.degrees(a) / 70, 0, 1)
    if not which:
        roll += 25 * np.clip((np.degrees(a) - 70) / 35, 0, 1)
    yaw = 30 if which else 30 * (1 - np.clip((np.degrees(a) - 45) / 30, 0, 1))
    pitch = 40
    if which:
        rr = (
            Rotation.from_euler("z", yaw, degrees=True)
            * Rotation.from_euler("x", pitch, degrees=True)
            * Rotation.from_euler("y", roll, degrees=True)
            * Rotation.from_euler("x", 90, degrees=True)
        )
        corner = rr.apply([0, 0.0519, 0.0165])
    else:
        rr = (
            Rotation.from_euler("z", yaw, degrees=True)
            * Rotation.from_euler("x", -pitch, degrees=True)
            * Rotation.from_euler("y", -roll, degrees=True)
            * Rotation.from_euler("x", -90, degrees=True)
            * Rotation.from_euler("z", 180, degrees=True)
        )
        corner = rr.apply([0, 0.0519, 0.0165])
    sign = 1 if which else -1
    tip = flap_edge(sim, which, a)
    tip[1] += -0.010 if which else 0.010
    tip += np.array([sign * (0.005 * np.cos(a) - 0.014 * np.sin(a)), 0, -0.005 * np.sin(a) - 0.014 * np.cos(a)])
    return tip - corner, rr.as_quat()


def idle_pose(sim, which):
    """Level, open hand beside its short end, lower native fingertip at lid height."""
    rr = Rotation.from_euler("x", 90 if which else -90, degrees=True)
    if which:
        rr *= Rotation.from_euler("z", 180, degrees=True)
    tip = sim["carton"].position + np.array([0, 0.120 if which else -0.120, 0.079])
    return tip - rr.apply([0, 0.0479, 0.0165]), rr.as_quat()


def open_flap(d, which):
    name = "left" if which else "right"
    for k in range(40):
        a = flap_angle(d.sim, which)
        target = min(np.radians(110 if which else 115), np.radians(10 + (k + 1) * 5), a + np.radians(25))
        p, q = hook_pose(d.sim, which, target)
        yield from d.move(name, p, q, 0.1)
        if k % 4 == 0:
            print("FLAP", name, round(d.elapsed, 2), round(np.degrees(flap_angle(d.sim, which)), 2), flush=True)
        if (flap_angle(d.sim, which) if which else a) > np.radians(94):
            break
    if flap_angle(d.sim, which) < np.radians(90):
        raise RuntimeError(f"{name} flap did not pass upright")


def routine(sim, session=None):
    from robosimstudio_materials.cutting.entity import knife_edge

    d = Drive(sim)
    sim._drive = d
    events = {}
    rows = []
    oldtick = d.tick
    tool_contact_peaks = {}
    tool_shapes = set(sim["cutter"].entity.shapes)
    shape_labels = list(sim.model.shape_label)

    def tick():
        oldtick()
        pr = sim["carton"].entity.tape_probe
        left = [i for i, s in enumerate(sim.model.shape_label) if "/left_arm/" in s]
        ke = sim["cutter"].entity
        for contact, (sa, sb) in enumerate(zip(*pr._shape_pairs(), strict=True)):
            other = sb if sa in tool_shapes else (sa if sb in tool_shapes else None)
            if other is None:
                continue
            label = shape_labels[other]
            if "/left_arm/" not in label and "/right_arm/" not in label:
                continue
            force = float(np.linalg.norm(pr.impulses[contact]) / pr.dt)
            if force > 1e-5:
                short = label.rsplit("/", 1)[-1]
                tool_contact_peaks[short] = max(tool_contact_peaks.get(short, 0), force)
        rows.append(
            [
                d.elapsed,
                *sim["carton"].position.tolist(),
                sim["carton"].entity.press,
                *pr.pairwise_force(left, sim["carton"].entity.shapes).tolist(),
                *pr.pairwise_force([ke.handle_shape], sim["carton"].entity.shapes).tolist(),
                *[float(np.degrees(flap_angle(sim, i))) for i in range(2)],
            ]
        )

    d.tick = tick
    sim._pull_rows = rows

    def event(name):
        events[name] = round(d.elapsed, 3)
        print("PHASE", name, events[name], flush=True)

    def tip():
        return knife_edge(sim["cutter"].entity.spec, sim["cutter"].pose)[1]

    def target(point):
        return np.asarray(point) - (tip() - sim.arm("right").ee_pose(sim.state)[0])

    br = Rotation.from_euler("y", 70, degrees=True) * Rotation.from_euler("x", 90, degrees=True)
    bq = br.as_quat()
    bp = sim["carton"].position + [-0.03, 0.066, 0.079] - br.apply([0, 0.0479, 0.0165]) - [0, 0, 0.0025]
    p = np.array([0.57813, -0.17726, 0.775])
    qp = Rotation.from_euler("y", 135, degrees=True).as_quat()
    yield from joint_stage(
        d,
        {"right": (p + [0, 0, 0.055], qp)},
        0.8,
        {"right": [-0.16217, 1.83644, 1.07701, -1.00146, -0.85717, -1.82002]},
    )
    yield from d.grip("right", 0.55, 0.2)
    yield from d.move("right", p, qp, 0.8)
    yield from d.hold(0.1)
    yield from d.grip("right", 1, 0.4)
    yield from d.move("right", p + [0, 0, 0.12], qp, 0.7)
    if sim["cutter"].position[2] < 0.83:
        raise RuntimeError("Cutter pickup failed")
    rh = Rotation.from_quat(sim.arm("right").ee_pose(sim.state)[1])
    rt = Rotation.from_quat(sim["cutter"].pose[3:])
    goal = Rotation.from_euler("z", 90, degrees=True) * Rotation.from_euler("y", 35, degrees=True)
    qc = (goal * (rh.inv() * rt).inv()).as_quat()
    yield from joint_stage(
        d,
        {"right": ([0.547, -0.10, 0.90], qc), "left": (bp + [0, 0.02, 0.019], bq)},
        0.8,
        {"left": [-0.07258, 1.71125, 0.19534, 1.51597, 1.49822, 0.34901]},
    )
    yield from d.hold(0.2)
    event("picked")

    c = sim["carton"].position.copy()
    y0 = c[1] + 0.045
    z = c[2] + 0.079
    yield from d.both({"right": (target([c[0] + 0.001, y0, c[2] + 0.12]), qc), "left": (bp, bq)}, 1.0)
    c = sim["carton"].position.copy()
    y0 = c[1] + 0.0415
    yield from d.move("right", target([c[0] + 0.001, y0, z]), qc, 2.0)
    yield from d.hold(0.1)
    event("blade_seated")
    e = sim["carton"].entity
    for _ in range(round((16.5 - d.elapsed) / 0.1)):
        if e.press < 0.8:
            z -= 0.00015
        elif e.press > 1.5:
            z += 0.0002
        yield from d.move("right", target([c[0] + 0.001, y0, z]), qc, 0.1)
    # Seat the measured blade edge before drawing: compensate arm/tool deflection.
    c = sim["carton"].position.copy()
    y0 = c[1] + 0.0445
    point = np.array([c[0] - 0.024 * 0.0445, y0])
    correction = np.zeros(2)
    for _ in range(5):
        if e.press < 5:
            z -= 0.002
        correction = np.clip(correction + 0.5 * (point - tip()[:2]), [-0.015, -0.025], [0.015, 0.025])
        yield from d.move("right", target([*(point + correction), z]), qc, 0.1)
    event("pull_start")
    previous = point
    for y in np.r_[np.linspace(y0, y0 - 0.004, 5), np.linspace(y0 - 0.004, c[1] - 0.057, 20)]:
        if e.press < 5:
            z -= 0.0005
        elif e.press > 6:
            z += 0.00035
        z = float(np.clip(z, c[2] + 0.054, c[2] + 0.080))
        correction = np.clip(correction + 0.5 * (previous - tip()[:2]), [-0.015, -0.025], [0.015, 0.025])
        point = np.array([c[0] - 0.024 * (y - c[1]), y])
        yield from d.move("right", target([*(point + correction), z]), qc, 0.1)
        previous = point
    event("cut_complete")
    print("SEAM", e.cut_fraction(), d.max_shift * 1000, flush=True)
    if e.cut_fraction() < 0.95:
        raise RuntimeError("Blade did not sever 95% of the tape length")
    rp, _ = sim.arm("right").ee_pose(sim.state)
    lp, lq = sim.arm("left").ee_pose(sim.state)
    yield from d.both({"right": (rp + [0, 0, 0.06], qc), "left": (lp + [0, 0, 0.06], lq)}, 0.8)
    yield from d.move("right", [0.57813, -0.17726, 0.89], qp, 1.2)
    hp, hq = hook_pose(sim, 1, entry=True)
    approach = Rotation.from_quat(hq).apply([0, 0, -0.055])
    yield from joint_stage(d, {"left": (hp + approach, hq)}, 0.8, {"left": LEVEL_LEFT})
    yield from d.hold(max(0, 22.4 - d.elapsed))
    event("putdown_start")
    yield from d.move("right", [0.57813, -0.17726, 0.775], qp, 1.2)
    yield from d.grip("right", 0, 0.4)
    event("cutter_released")
    yield from d.both({"right": ([0.55, -0.19, 0.93], qp), "left": (hp, hq)}, 0.6)
    event("far_catch")
    yield from open_flap(d, 1)
    event("far_open")
    lp, lq = sim.arm("left").ee_pose(sim.state)
    yield from d.move("left", lp + [0, 0.065, 0.015], lq, 0.6)
    ip, iq = idle_pose(sim, 1)
    rp_idle, rq_idle = idle_pose(sim, 0)
    yield from joint_stage(
        d,
        {"left": (ip, iq), "right": (rp_idle, rq_idle)},
        1.0,
        {"left": [-0.2239, 1.5656, 0.3446, 1.221, 0.8233, -1.5708]},
    )
    np_, nq = hook_pose(sim, 0, entry=True)
    approach = Rotation.from_quat(nq).apply([0, 0, -0.055])
    yield from d.hold(max(0, 33.6 - d.elapsed))
    yield from joint_stage(
        d,
        {"right": ([0.55, -0.19, 0.93], qp)},
        0.8,
        {"right": [-0.16217, 1.83644, 1.07701, -1.00146, -0.85717, -1.82002]},
    )
    yield from joint_stage(d, {"right": (np_ + approach, nq)}, 0.8, {"right": LEVEL_RIGHT})
    yield from d.hold(max(0, 35.8 - d.elapsed))
    yield from d.move("right", np_, nq, 0.6)
    event("near_catch")
    yield from open_flap(d, 0)
    event("near_open")

    lp, lq = sim.arm("left").ee_pose(sim.state)
    rp, rq = sim.arm("right").ee_pose(sim.state)
    yield from d.move("right", rp + [0, -0.065, 0.04], rq, 0.6)
    yield from joint_stage(d, {"left": idle_pose(sim, 1), "right": idle_pose(sim, 0)}, 0.6)
    yield from d.grip("left", 0, 0.2)
    event("hands_clear")
    yield from d.hold(max(0, 45.4 - d.elapsed))
    angles = [float(np.degrees(flap_angle(sim, i))) for i in range(2)]
    cp = sim["cutter"].position.copy()
    sim.contacts.update(step=None)
    cutter_shapes = sim["cutter"].entity.shapes
    finger_shapes = [
        i
        for i, label in enumerate(sim.model.shape_label)
        if label.rsplit("/", 1)[-1].startswith(("right_lf_down_", "right_rf_down_"))
    ]
    grip_force = sum(float(np.linalg.norm(sim.contacts.pairwise_force([i], cutter_shapes))) for i in finger_shapes)
    result = {
        "time_s": round(d.elapsed, 3),
        "phases_s": events,
        "tape_length_cut_fraction": e.cut_fraction(),
        "tape_intact_by_station": e.intact.numpy().reshape(-1, 2).tolist(),
        "flap_angles_deg": angles,
        "carton_max_slide_mm": d.max_shift * 1000,
        "carton_end_slide_mm": float(np.linalg.norm(sim["carton"].position[:2] - d.start_carton[:2]) * 1000),
        "cutter_position_m": cp.tolist(),
        "cutter_final_finger_force_n": grip_force,
        "tool_robot_contact_peaks_n": tool_contact_peaks,
        "task_success": bool(
            e.cut_fraction() >= 0.95
            and min(angles) >= 60
            and d.max_shift < 0.02
            and cp[2] < 0.80
            and grip_force < 0.1
            and all(label.startswith(("right_lf_down_", "right_rf_down_")) for label in tool_contact_peaks)
            and d.elapsed <= 47.4
        ),
    }
    sim._decant_result = result
    sim._decant_trace = rows
    print("RESULT", result, flush=True)
    if session is not None:
        previous = getattr(sim, "_decant_widget", None)
        if previous is not None:
            previous.remove()
        sim._decant_widget = session.server.gui.add_markdown(
            f"**Measured decant result**\n\nTape cut: {e.cut_fraction() * 100:.1f}% · Flaps: {angles[0]:.1f}° / {angles[1]:.1f}° · Maximum slide: {d.max_shift * 1000:.2f} mm · Cutter released: {cp[2] < 0.80 and grip_force < 0.1}"
        )
    # Persist measured task outcomes for each normal or physics-audit replay.
    result["cutter_mass_kg"] = float(sim.model.body_mass.numpy()[sim["cutter"].entity.body])
    history = np.asarray(e.history)
    drawn = history[(history[:, 0] >= 17) & (history[:, 0] <= 19.5), 2]
    brace = np.array(rows)
    braced = brace[(brace[:, 0] >= 5.5) & (brace[:, 0] <= 19), 7]
    result["left_brace_downforce_p05_p50_p95_n"] = np.percentile(braced, [5, 50, 95]).tolist()
    result["blade_press_p50_p95_n"] = np.percentile(drawn, [50, 95]).tolist()
    output_dir = Path(__file__).with_name("run")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "task_results.jsonl").open("a") as out:
        out.write(json.dumps(result) + "\n")
    if not result["task_success"]:
        raise RuntimeError("Measured decant task criteria failed")


def pick(sim):
    yield from routine(sim)
