"""Smooth joint targets sampled from a continuous Cartesian path; only the robot is commanded."""

import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


class Driver:
    def __init__(self, sim, quat):
        self.sim = sim
        self.arm = sim.arm("right")
        self.quat = quat
        self.seed = sim.state.joint_q.numpy().copy()
        self.q = self.seed[self.arm.arm_coords].copy()
        self.park = self.q.copy()
        self.left = self.seed[sim.arm("left").arm_coords].copy()
        self.pos = np.asarray(self.arm.ee_pose(sim.state)[0])
        self.closed = 0.0
        self.offset = None
        self.slip = 0.0
        self.ticks = 0

    def tick(self, q):
        self.q = np.asarray(q)
        self.sim.step(np.r_[self.left, 0.0, self.q, self.closed], mode="joint_position")
        self.ticks += 1
        if self.offset is not None:
            p, r = self.arm.ee_pose(self.sim.state)
            off = Rotation.from_quat(r).inv().apply(np.asarray(self.sim["handle"].position) - p)
            self.slip = max(self.slip, float(np.linalg.norm(off - self.offset) * 1000))

    def hold(self, seconds):
        for _ in range(round(seconds * 60)):
            self.tick(self.q)
            if self.ticks % 4 == 0:
                yield

    def move(self, target, speed):
        target = np.asarray(target, float)
        start = self.pos.copy()
        n = max(4, int(math.ceil(np.linalg.norm(target - start) / speed * 60)))
        n = int(math.ceil(n / 4) * 4)
        q0 = self.q.copy()
        for k in range(4, n + 1, 4):
            t = k / n
            f = t * t * (3 - 2 * t)
            pos = start + (target - start) * f
            self.seed = self.arm.solve_q_full(self.seed, pos, self.quat)
            q1 = self.seed[self.arm.arm_coords].copy()
            for j in range(1, 5):
                self.tick(q0 + (q1 - q0) * j / 4)
            q0 = q1
            yield
        self.pos = target
        yield from self.hold(0.15)
        miss = float(np.linalg.norm(self.arm.ee_pose(self.sim.state)[0] - target))
        if miss > 0.006:
            raise RuntimeError(f"Hand target missed by {miss * 1000:.2f} mm")

    def unfold(self):
        target = np.array([0.50156039, 2.02788234, 1.98255920, -1.52547204, 0, -0.89918298])
        start = self.q.copy()
        for k in range(1, 121):
            t = k / 120
            self.tick(start + (target - start) * (t * t * (3 - 2 * t)))
            if self.ticks % 4 == 0:
                yield
        self.seed = self.sim.state.joint_q.numpy().copy()
        self.pos = np.asarray(self.arm.ee_pose(self.sim.state)[0])
        yield from self.hold(0.3)
        self.approach = self.pos.copy()

    def fold(self):
        start = self.q.copy()
        for k in range(1, 121):
            t = k / 120
            self.tick(start + (self.park - start) * (t * t * (3 - 2 * t)))
            if self.ticks % 4 == 0:
                yield
        yield from self.hold(1.2)

    def grasp(self):
        p, r = self.arm.ee_pose(self.sim.state)
        self.offset = Rotation.from_quat(r).inv().apply(np.asarray(self.sim["handle"].position) - p)


def direct_routine(sim, START, ORIGIN, DOWN, LO, HI):
    d = Driver(sim, DOWN)
    grasp = START + ORIGIN + [0, 0, 0.078]
    yield from d.hold(0.5)
    yield from d.unfold()
    yield from d.move(grasp, 0.035)
    d.closed = 1.0
    yield from d.hold(1.0)
    initial = np.asarray(sim["handle"].position).copy()
    d.grasp()
    yield from d.move(grasp + np.array([0, 0, 0.05]), 0.05)
    if sim["handle"].position[2] < initial[2] + 0.025:
        raise RuntimeError("Brush did not lift in the fingers")
    pressed = np.array([0.6, -0.06, grasp[2] - 0.004])
    yield from d.move(pressed + np.array([0, 0, 0.05]), 0.08)
    yield from d.move(pressed, 0.025)
    sim["brush"].entity.clear_scrub()
    for dx in [-0.05, 0.075, -0.05, 0.075, -0.05, 0.075]:
        target = pressed + np.array([dx, 0, 0])
        yield from d.move(target, 0.08)
    coverage = sim["brush"].entity.scrubbed(LO, HI)
    yield from d.move(target + np.array([0, 0, 0.05]), 0.05)
    yield from d.move(grasp + np.array([0, 0, 0.05]), 0.08)
    yield from d.move(grasp, 0.025)
    yield from d.hold(0.6)
    slip = d.slip
    d.offset = None
    d.closed = 0.0
    yield from d.hold(0.7)
    yield from d.move(grasp + np.array([0, 0, 0.06]), 0.06)
    yield from d.move(d.approach, 0.06)
    yield from d.fold()
    pose = np.asarray(sim["handle"].pose)
    result = {
        "scrubbed": coverage,
        "slip_mm": slip,
        "return_mm": float(np.linalg.norm(pose[:3] - (START + ORIGIN)) * 1000),
        "upright_deg": float(np.degrees(np.arccos(np.clip(Rotation.from_quat(pose[3:]).as_matrix()[2, 2], -1, 1)))),
        "steps": d.ticks,
    }
    print("WORKSHOP_RESULT", json.dumps(result), flush=True)
    path = Path(__file__).resolve().parent / "run/latest_result.json"
    path.write_text(json.dumps(result, indent=2))
    grid = sim["brush"].entity.scrub_grid()
    np.savez_compressed(
        path.with_name("coverage.npz"), cells=grid.cells, origin=grid.origin, cell=grid.cell, lo=LO, hi=HI
    )
    with path.with_name("results.jsonl").open("a") as f:
        f.write(json.dumps(result) + "\n")
    if coverage < 0.8 or slip > 5 or result["return_mm"] > 30 or result["upright_deg"] > 10:
        raise RuntimeError(f"Brush result outside target: {result}")
