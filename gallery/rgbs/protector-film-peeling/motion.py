"""Timed joint commands only; all prop motion comes from contact and tape adhesion."""

import json
from pathlib import Path

import numpy as np
from contact_readings import finger_contacts
from scipy.spatial.transform import Rotation, Slerp


class Drive:
    def __init__(self, sim):
        self.sim = sim
        self.action = np.array(sim.idle_action("joint_position"), dtype=float)
        self.seed = {a: sim.state.joint_q.numpy().copy() for a in ("left", "right")}
        self.peak_pull = 0.0
        self.phone0 = np.array(sim["phone"].position)
        self.max_phone_mm = 0.0
        self.grasp_local = None
        self.pinch_offset = None
        self.load_peak = None
        self.crossing = None
        self.samples = []
        self.previous_peel = self.sim["film"].entity.peeled()
        self.left_contacts = {}
        self.regrasp = None
        self.tab0 = sim["film"].entity.tab_pose(sim.state)[:3].copy()
        self.before_close = None
        self.close_state = None
        self.bond_events = []
        self.broken_before = sim["film"].entity.broken.numpy().astype(bool)

    def tick(self):
        self.sim.step(self.action, mode="joint_position")
        self.peak_pull = max(self.peak_pull, self.sim["film"].entity.pull(self.sim.state))
        self.max_phone_mm = max(
            self.max_phone_mm, float(np.linalg.norm(np.array(self.sim["phone"].position) - self.phone0) * 1000)
        )

        r = self.reading()
        if round(self.sim.time * 60) % 6 == 0:
            self.samples.append(r)
        if self.grasp_local is not None:
            if self.load_peak is None or r["pull_N"] > self.load_peak["pull_N"]:
                self.load_peak = r
        if self.previous_peel < 0.95 <= r["peeled"]:
            r["finger_contacts"] = self.finger_contacts()
            self.crossing = r
            print("CROSS_95", r, flush=True)
        if r["peeled"] > self.previous_peel:
            film = self.sim["film"].entity
            broken = film.broken.numpy().astype(bool)
            indices = np.flatnonzero(broken & ~self.broken_before)
            self.bond_events.append(
                {
                    "time": r["time"],
                    "indices": indices.tolist(),
                    "along_mm": (film.along[indices] * 1000).tolist(),
                    "anchors_mm": (film.anchors[indices] * 1000).tolist(),
                }
            )
            self.broken_before = broken
        self.previous_peel = r["peeled"]
        if 33.5 <= self.sim.time < 35 and self.regrasp is None:
            pads = finger_contacts(self.sim, "phone", "left")
            heights = [pt[2] for pad in pads.values() for pt in pad["points_local_mm"]]
            if len(pads) == 2 and heights and min(heights) < -3.5 and max(heights) > 3.5:
                self.regrasp = {**r, "finger_contacts": pads}
                print("LEFT_REGRASP", self.regrasp, flush=True)
        for stamp in (12, 33.5, 34.8):
            if abs(self.sim.time - stamp) < 1e-5:
                contact = finger_contacts(self.sim, "phone", "left")
                self.left_contacts[str(stamp)] = contact
                print("LEFT_CONTACT", stamp, contact, flush=True)

    def finger_contacts(self):
        return finger_contacts(self.sim)

    def held_in_pads(self, contacts=None):
        """Read opposing contacts on the handle's broad faces, allowing retained sliding."""
        contacts = self.finger_contacts() if contacts is None else contacts
        half = np.asarray(self.sim["film"].entity.spec.tab_size) * 500
        signs = []
        for name in ("right_link_left_finger", "right_link_right_finger"):
            pad = contacts.get(name, {})
            if pad.get("normal_N", 0.0) <= 0.5:
                return False
            points = np.asarray(pad.get("points_local_mm", []))
            if not len(points):
                return False
            face = points[
                (np.abs(np.abs(points[:, 1]) - half[1]) < 0.25)
                & (np.abs(points[:, 0]) <= half[0] + 0.5)
                & (np.abs(points[:, 2]) <= half[2] + 0.5)
            ]
            if not len(face):
                return False
            signs.append(float(np.mean(face[:, 1])))
        return signs[0] * signs[1] < 0

    def begin_close(self):
        self.before_close = self.reading()
        print("BEFORE_CLOSE", self.before_close, flush=True)

    def mark_pinch(self):
        t = self.sim["film"].entity.tab_pose(self.sim.state)
        p, _q = self.sim.arm("right").ee_pose(self.sim.state)
        self.grasp_local = Rotation.from_quat(t[3:]).inv().apply(np.asarray(p) - t[:3])
        self.pinch_offset = float(np.linalg.norm(t[:3] - p))
        self.close_state = self.reading()
        self.close_state["finger_contacts"] = self.finger_contacts()
        self.close_state["tab_close_travel_mm"] = (
            None
            if self.before_close is None
            else float(np.linalg.norm(t[:3] - np.array(self.before_close["tab"])) * 1000)
        )
        print("CLOSE_STATE", self.close_state, flush=True)
        print("PINCH_REFERENCE", self.sim.time, self.pinch_offset, self.grasp_local.tolist(), self.finger_contacts())

    def reading(self):
        t = self.sim["film"].entity.tab_pose(self.sim.state)
        p, _q = self.sim.arm("right").ee_pose(self.sim.state)
        drift = (
            None
            if self.grasp_local is None
            else float(np.linalg.norm(t[:3] + Rotation.from_quat(t[3:]).apply(self.grasp_local) - p))
        )
        return dict(  # noqa: C408
            time=float(self.sim.time),
            tab=t[:3].tolist(),
            tab_displacement_mm=float(np.linalg.norm(t[:3] - self.tab0) * 1000),
            peeled=float(self.sim["film"].entity.peeled()),
            pull_N=float(self.sim["film"].entity.pull(self.sim.state)),
            tab_tcp_mm=float(np.linalg.norm(t[:3] - p) * 1000),
            grip_drift_mm=None if drift is None else drift * 1000,
            phone_mm=float(np.linalg.norm(np.asarray(self.sim["phone"].position) - self.phone0) * 1000),
        )

    def wait(self, seconds):
        for _ in range(round(seconds * 60)):
            self.tick()
            yield

    def close(self, arm, value, seconds=0.5):
        idx = 6 if arm == "left" else 13
        a = self.action[idx]
        n = max(1, round(seconds * 60))
        for k in range(1, n + 1):
            self.action[idx] = a + (value - a) * k / n
            self.tick()
            yield

    def move(self, arm, position, quat, seconds):
        ik = self.sim.arm(arm)
        a = 0 if arm == "left" else 7
        p0, q0 = ik.ee_pose(self.sim.state)
        p0 = np.array(p0)
        p1 = np.array(position)
        sl = Slerp([0, 1], Rotation.from_quat([q0, quat]))
        n = max(2, round(seconds * 60))
        qprev = self.seed[arm]
        last = 0
        for k in [*range(6, n, 6), n]:
            u = k / n
            f = u * u * (3 - 2 * u)
            p = p0 + (p1 - p0) * f
            q = sl(f).as_quat()
            sol = ik.solve_q_full(qprev, p, q, iterations=32)
            pe, _re = ik.pose_errors(sol, p, q)
            if pe > 0.012:
                raise RuntimeError(f"{arm} IK miss {pe * 1000:.1f} mm")
            for j in range(last + 1, k + 1):
                self.action[a : a + 6] = (qprev + (sol - qprev) * (j - last) / (k - last))[ik.arm_coords]
                self.tick()
                yield
            last = k
            qprev = sol
        self.seed[arm] = qprev

    def home(self, arm, q, seconds=2):
        a = 0 if arm == "left" else 7
        old = self.action[a : a + 6].copy()
        for k in range(1, round(seconds * 60) + 1):
            u = k / round(seconds * 60)
            u = u * u * (3 - 2 * u)
            self.action[a : a + 6] = old + (np.array(q) - old) * u
            self.tick()
            yield

    def homes(self, targets, seconds):
        old = self.action.copy()
        for k in range(1, round(seconds * 60) + 1):
            u = k / round(seconds * 60)
            u = u * u * (3 - 2 * u)
            for arm, q in targets.items():
                a = 0 if arm == "left" else 7
                self.action[a : a + 6] = old[a : a + 6] + (np.array(q) - old[a : a + 6]) * u
            self.tick()
            yield

    def save(self):
        run_dir = Path(__file__).with_name("run")
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dir.joinpath("latest_metrics.json").write_text(
            json.dumps(
                {
                    "samples": self.samples,
                    "before_close": self.before_close,
                    "close_state": self.close_state,
                    "bond_events": self.bond_events,
                    "crossing": self.crossing,
                    "load_peak": self.load_peak,
                    "left_contacts": self.left_contacts,
                    "regrasp": self.regrasp,
                    "max_phone_mm": self.max_phone_mm,
                    "final_right_contacts": self.finger_contacts(),
                },
                indent=2,
            )
        )
