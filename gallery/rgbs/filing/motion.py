"""Paired end grasps and 99 seconds of contact-driven filing."""

import json
import math
import os

import numpy as np

from robosimstudio.solvers import grasp_quat

SIDES = (("left", 1), ("right", -1))


def hand_quat(sim, name, sign, close=None, pitch=None):
    h = sim.handles.arms[name]
    close = (sign, 0, 0) if close is None else close
    down = (0.02 if name == "left" else 0.18) if pitch is None else math.tan(math.radians(pitch))
    return grasp_quat(h.approach_axis, h.close_axis, h.ee_offset_rot, (0, -sign, -down), close)


class Driver:
    def __init__(self, sim):
        self.sim = sim
        self.seed = sim.state.joint_q.numpy().copy()
        self.targets = {n: sim.arm(n).ee_pose(sim.state)[0].copy() for n, _ in SIDES}
        self.pitch = np.degrees(np.arctan((0.02, 0.18)))
        self.quats = {n: hand_quat(sim, n, s) for n, s in SIDES}
        self.grip = np.array((0.65, 0.65))
        self.rows = []

    def step(self, targets, grip):
        sim = self.sim
        for name, _ in SIDES:
            self.seed = sim.arm(name).solve_q_full(self.seed, targets[name], self.quats[name])
        action = []
        grip = np.broadcast_to(grip, (2,)).copy()
        for i, (n, _) in enumerate(SIDES):
            action.extend([*self.seed[sim.arm(n).arm_coords], grip[i]])
        sim.step(action, mode="joint_position")
        self.targets = {n: np.array(p) for n, p in targets.items()}
        self.grip = grip

    def move(self, targets, seconds, grip=None, pitch=None):
        start = {n: p.copy() for n, p in self.targets.items()}
        p0 = self.pitch.copy()
        p1 = p0 if pitch is None else np.broadcast_to(pitch, (2,)).copy()
        g0 = self.grip
        g1 = g0 if grip is None else np.broadcast_to(grip, (2,)).copy()
        for k in range(round(seconds * 50)):
            t = (k + 1) / round(seconds * 50)
            t = t * t * (3 - 2 * t)
            self.pitch = p0 + (p1 - p0) * t
            self.quats = {n: hand_quat(self.sim, n, sign, pitch=self.pitch[i]) for i, (n, sign) in enumerate(SIDES)}
            self.step({n: start[n] * (1 - t) + np.asarray(targets[n]) * t for n, _ in SIDES}, g0 + (g1 - g0) * t)
            if k % 5 == 4:
                yield

    def pair(self, center, station=0.030):
        return {n: np.asarray(center) + np.array((0, s * station, 0)) for n, s in SIDES}

    def log(self, phase):
        sim = self.sim
        field = sim["apparatus"].entity
        field._probe.update()
        row = {"time": sim.time, "phase": phase, "strip": sim["strip"].pose.tolist(), **field.read()}
        for name, _ in SIDES:
            h = sim.handles.arms[name]
            row[name + "_forces"] = [
                float(
                    np.linalg.norm(
                        field._probe.pairwise_force(field.strip_shapes, np.concatenate([ss for _, ss in group]))
                    )
                )
                for group in h.finger_groups
            ]
        row["target_pitch_deg"] = self.pitch.tolist()
        row["ee_positions"] = {n: sim.arm(n).ee_pose(sim.state)[0].tolist() for n, _ in SIDES}
        self.rows.append(row)
        print(f"{phase}: t={sim.time:.2f}s cleared={row['cleared']:.1%} press={row['force']:.2f}N", flush=True)


def approach(d, start, pitch=(1.1458, 10.204)):
    """Right pinch closes first; both keep the front/back closing axis."""
    yield from d.move(d.pair(start + np.array((0, 0, 0.07)), 0.10), 1.5, 0.68, pitch=pitch)
    ready = d.pair(start, 0.10)
    yield from d.move(ready, 1.5)
    right_in = {"left": ready["left"], "right": d.pair(start)["right"]}
    yield from d.move(right_in, 0.6)
    yield from d.move(right_in, 0.6, (0.68, 0.88))
    d.log("right_grasp")
    yield from d.move(d.pair(start), 1.2)
    yield from d.move(d.pair(start), 0.6, 0.88)
    d.log("grasp")


def routine(sim):
    """99 s; moving targets belong only to the robot, never the strip."""
    d = Driver(sim)
    start = sim["strip"].position.copy() + np.array((0, 0, 0.001))
    yield from approach(d, start)
    under = np.array((0.680, 0, 0.870))
    yield from d.move(d.pair(under), 2.0, pitch=np.degrees(np.arctan((0.02, 0.18))))
    d.log("under_drum")
    if sim["strip"].position[2] < 0.81:
        raise RuntimeError("Strip did not lift in the fingers")
    pressed = np.array((0.680, 0, 0.891))
    field = sim["apparatus"].entity
    bias = 0.0
    pressure = 2.0
    hold = under - np.array((0, 0, 0.004))
    # Main-view samples show separate seated passes, not one continuous press.
    windows = (
        (15.0, 17.2),
        (19.0, 21.0),
        (27.5, 32.5),
        (39.0, 41.0),
        (47.5, 50.5),
        (57.5, 60.5),
        (67.5, 70.5),
        (77.5, 80.5),
    )
    for k in range(73 * 50):  # 8 to 81 s, including the last descent.
        t = 8.0 + (k + 1) / 50
        w = 0.0
        for begin, end in windows:
            if begin - 1.0 < t <= begin:
                u = t - (begin - 1.0)
                w = u * u * (3 - 2 * u)
            elif begin < t <= end:
                w = 1.0
            elif end < t < end + 0.5:
                u = (t - end) / 0.5
                w = 1 - u * u * (3 - 2 * u)
        if w > 0.99:
            pressure = 0.95 * pressure + 0.05 * field.force
            bias = float(np.clip(bias + 0.0005 * (2.0 - pressure) / 50, 0.0, 0.012))
        else:
            pressure = 2.0
        center = hold.copy()
        center[2] += w * (pressed[2] + bias - hold[2])
        center[1] = w * 0.006 * math.sin(2 * math.pi * t / 5)
        d.step(d.pair(center), 0.88)
        if (k + 1) % 50 == 0:
            d.log("filing" if w > 0.99 else "clear_or_transition")
        if sim["strip"].position[2] < 0.81:
            raise RuntimeError("Strip lost during filing")
        if k % 5 == 4:
            yield
    d.log("clear_of_drum")
    yield from d.move(d.pair(hold), 4.0)
    d.log("hold_below_drum")
    yield from d.move(d.pair(hold), 1.5)
    yield from d.move(d.pair(pressed + np.array((0, 0, bias))), 1.0)
    pressure = 2.0
    for k in range(100):
        pressure = 0.95 * pressure + 0.05 * field.force
        bias = float(np.clip(bias + 0.0005 * (2.0 - pressure) / 50, 0.0, 0.012))
        d.step(d.pair(pressed + np.array((0, 0, bias))), 0.88)
        if k + 1 in (25, 75, 100):
            d.log("second_press")
        if k % 5 == 4:
            yield
    yield from d.move(d.pair(hold), 0.5)
    d.log("hold_ends")
    above = d.pair(start + np.array((0, 0, 0.050)))
    yield from d.move(above, 2.0)
    d.log("return_above_mat")
    if field.both_grips < field.airborne:
        raise RuntimeError("Both hands did not hold throughout the two-hand carry")
    yield from d.move(above, 0.5, (0.68, 0.88))
    d.log("left_released")
    staged = {"left": above["left"] + np.array((0, 0.08, 0.07)), "right": above["right"]}
    yield from d.move(staged, 0.5, pitch=(1.1458, 25))
    landed = {"left": staged["left"], "right": d.pair(start)["right"]}
    yield from d.move(landed, 1.3, pitch=(1.1458, 40))
    d.log("touchdown")
    yield from d.move(landed, 0.5, (0.68, 0.68))
    d.log("released")
    yield from d.move(d.pair(start + np.array((0, 0, 0.07)), 0.14), 1.2)
    q0 = sim.state.joint_q.numpy().copy()
    for k in range(150):
        u = (k + 1) / 150
        u = u * u * (3 - 2 * u)
        action = []
        for n, _ in SIDES:
            h = sim.handles.arms[n]
            action.extend([*(q0[list(h.arm_coords)] * (1 - u) + np.array(h.home_q) * u), 0.68])
        sim.step(action, mode="joint_position")
        if k % 5 == 4:
            yield
    d.log("home")
    os.makedirs("run", exist_ok=True)
    with open("run/last_contact_trace.json", "w") as f:
        json.dump(d.rows, f, indent=2)
    with open("run/contact_samples.json", "w") as f:
        json.dump(
            {
                "columns": [
                    "t",
                    "strip_z",
                    "drum_normal_N",
                    "left_jaw1_N",
                    "left_jaw2_N",
                    "right_jaw1_N",
                    "right_jaw2_N",
                    "cleared",
                ],
                "samples": field.contact_trace,
                "result": field.read(),
            },
            f,
        )
    for second, touching in ((84, False), (88, True), (89, True), (90, False)):
        sample = min(field.contact_trace, key=lambda row: abs(row[0] - second))
        if (sample[2] > 0.1) != touching:
            raise RuntimeError(f"Drum contact at {second}s was {sample[2]:.3f} N")
    if field.unexpected:
        raise RuntimeError(f"Non-finger support touched airborne strip: {field.unexpected}")
    if field.supported < field.airborne:
        raise RuntimeError(f"Finger support verified in only {field.supported}/{field.airborne} airborne steps")
    if field.read()["cleared"] < 0.9:
        raise RuntimeError(f"Mark only {field.read()['cleared']:.1%} cleared")
    if abs(sim["strip"].position[2] - 0.756) > 0.003:
        raise RuntimeError("Strip is not back on mat")
