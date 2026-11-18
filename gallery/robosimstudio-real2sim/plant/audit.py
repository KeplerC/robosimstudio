"""Read-only trajectory measurements; no simulation state is written."""

import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

import numpy as np
from scipy.spatial.transform import Rotation


class Audit:
    def __init__(self, sim, layout, session=None):
        self.sim = sim
        self.session = session
        self.hand_trace = []
        self.pose_trace = []
        self.contacts = {}
        self.below_top = set()
        labels = [str(s).split("/")[-1] for s in sim.model.body_label]
        self.leaves = []
        for lf in layout.leaves:
            v = lf.V - lf.junction
            self.leaves.append((lf.name, labels.index(lf.name), v[np.argmax(np.linalg.norm(v, axis=1))]))
        self.rest = self.points()
        self.peak_tip = np.zeros(7)
        self.peak_tip_time = np.zeros(7)
        self.peak_stalk = np.zeros(7)
        self.n = len(sim["water"].particle_q)
        self.escaped = np.zeros(self.n, bool)
        self.below_table = np.zeros(self.n, bool)
        self.below_soil = np.zeros(self.n, bool)
        self.min_z = 10.0
        self.rows = []
        self.pot_low_inside = np.zeros(self.n, bool)
        self.pot_through = np.zeros(self.n, bool)
        self.max_soil_depth = 0.0
        self.trajectory = []
        self.times = []
        self.deep = np.zeros(self.n, bool)

    def points(self):
        b = self.sim.state.body_q.numpy()
        junction = np.array([b[i, :3] for _, i, _ in self.leaves])
        tip = np.array([b[i, :3] + Rotation.from_quat(b[i, 3:]).apply(v) for _, i, v in self.leaves])
        return tip, junction

    def sample(self):
        p = np.asarray(self.sim["water"].particle_q)
        self.trajectory.append(p.copy())
        self.times.append(float(self.sim.time))
        bodies = self.sim.state.body_q.numpy()
        self.pose_trace.append(np.array([bodies[i] for _, i, _ in self.leaves]))
        cp = np.asarray(self.sim["cup"].pose)
        r = Rotation.from_quat(cp[3:])
        local = r.inv().apply(p - cp[:3])
        in_cup = np.all((local >= (-0.041, -0.041, -0.002)) & (local <= (0.041, 0.041, 0.094)), axis=1)
        radial = np.linalg.norm(p[:, :2] - [0.6, -0.014], axis=1)
        # Track a wall crossing only after entering below the rim. Rising to
        # the rim clears the flag: ordinary overflow/splash is not tunneling.
        z = p[:, 2]
        self.pot_low_inside[z >= 0.894] = False
        self.pot_low_inside |= (radial < 0.056) & (z < 0.894) & (z > 0.759) & ~in_cup
        outer = 0.0534 + 0.011 * np.clip((z - 0.759) / 0.126, 0, 1)
        outer = np.where(z >= 0.887, 0.072, outer)
        self.pot_through |= self.pot_low_inside & (radial > outer) & (z < 0.893) & (z > 0.759) & ~in_cup
        interior = (radial < 0.055) & ~in_cup
        self.deep |= interior & (z < 0.873)
        if interior.any():
            self.max_soil_depth = max(self.max_soil_depth, 0.0, 0.8745 - float(z[interior].min()))
        self.rows.append({"time": float(self.sim.time), "in_cup": int(in_cup.sum()), "left_cup": int((~in_cup).sum())})
        self.below_top.update(np.flatnonzero(z < 0.751).tolist())
        if self.session is not None:
            from robosimstudio_studio.routine import hand

            tcp, quat = hand(self.session, "right")
            target = np.asarray(self.session.arms["right"].cursor_pos)
            self.hand_trace.append(
                {
                    "time": float(self.sim.time),
                    "target": target.tolist(),
                    "tcp": np.asarray(tcp).tolist(),
                    "cup": cp.tolist(),
                    "relative": Rotation.from_quat(quat).inv().apply(cp[:3] - tcp).tolist(),
                    "target_error_mm": float(1000 * np.linalg.norm(target - tcp)),
                }
            )
        self.sim.contacts.update()
        a, b = self.sim.contacts._shape_pairs()
        mag = np.linalg.norm(self.sim.contacts.impulses, axis=1)
        sb = self.sim.model.shape_body.numpy()
        for i in np.flatnonzero(mag > 1e-6):
            names = [
                str(self.sim.model.body_label[sb[j]]) if sb[j] >= 0 else str(self.sim.model.shape_label[j])
                for j in [a[i], b[i]]
            ]
            if any("yam" in n for n in names) and any(
                any(obj in n for obj in ["cup", "pot", "saucer", "table", "leaf"]) for n in names
            ):
                key = " | ".join(names)
                v = self.contacts.setdefault(
                    key,
                    {
                        "first": float(self.sim.time),
                        "last": float(self.sim.time),
                        "max_force": 0.0,
                        "max_depth_mm": 0.0,
                    },
                )
                v.update(
                    last=float(self.sim.time),
                    max_force=max(v["max_force"], float(mag[i] / self.sim.contacts.dt)),
                    max_depth_mm=max(v["max_depth_mm"], -1000 * float(self.sim.contacts.depths[i])),
                )
        self.escaped |= ~in_cup
        self.below_table |= p[:, 2] < 0.72
        self.below_soil |= (radial < 0.045) & (p[:, 2] < 0.769) & ~in_cup
        self.min_z = min(self.min_z, float(p[:, 2].min()))
        tip, stalk = self.points()
        td = np.linalg.norm(tip - self.rest[0], axis=1) * 1000
        sd = np.linalg.norm(stalk - self.rest[1], axis=1) * 1000
        self.peak_tip_time[td > self.peak_tip] = float(self.sim.time)
        self.peak_tip = np.maximum(self.peak_tip, td)
        self.peak_stalk = np.maximum(self.peak_stalk, sd)
        self.last_tip = td
        self.last_stalk = sd
        return {
            "time": float(self.sim.time),
            "escaped": int(self.escaped.sum()),
            "through_table": int(self.below_table.sum()),
            "through_soil": int(self.below_soil.sum()),
            "peak_tip_mm": float(self.peak_tip.max()),
            "tip_mm": float(td.max()),
        }

    def save(self, path="run/trajectory_audit.json"):
        from watering import measurements

        m = measurements(self.sim)
        m.update(
            ever_left_cup=int(self.escaped.sum()),
            capture_of_ever_left=float(m["in_pot"] / max(1, self.escaped.sum())),
            through_table_ever=int(self.below_table.sum()),
            through_soil_ever=int(self.below_soil.sum()),
            minimum_particle_z=self.min_z,
            through_pot_wall_ever=int(self.pot_through.sum()),
            maximum_soil_surface_depth_mm=1000 * self.max_soil_depth,
            leaf_tip_peak_mm=self.peak_tip.tolist(),
            leaf_tip_peak_time=self.peak_tip_time.tolist(),
            leaf_tip_final_mm=self.last_tip.tolist(),
            stalk_junction_peak_mm=self.peak_stalk.tolist(),
            stalk_junction_final_mm=self.last_stalk.tolist(),
            leaves=[l[0] for l in self.leaves],
        )
        suspects = self.deep | self.below_table | self.below_soil | self.pot_through
        np.savez_compressed(
            ROOT / "run/particle_tracks.npz",
            times=self.times,
            ids=np.flatnonzero(suspects),
            positions=np.asarray(self.trajectory)[:, suspects],
        )
        np.savez_compressed(
            ROOT / "run/leaf_pose_trace.npz", times=self.times, poses=self.pose_trace, names=[l[0] for l in self.leaves]
        )
        (ROOT / "run/hand_trace.json").write_text(json.dumps(self.hand_trace))
        (ROOT / "run/flow_trace.json").write_text(json.dumps(self.rows))
        (ROOT / "run/contact_trace.json").write_text(json.dumps(self.contacts, indent=2))
        m["below_drawn_table_top_ever"] = len(self.below_top)
        m["robot_contacts"] = self.contacts
        m["recorded_utc"] = datetime.now(UTC).isoformat()
        m["phases"] = json.loads((ROOT / "run/watering_measurements.json").read_text()).get("marks", [])
        (ROOT / path).write_text(json.dumps(m, indent=2))
        with (ROOT / "run/audit_history.jsonl").open("a") as f:
            f.write(json.dumps(m) + "\n")
