"""Read the shipped VBD solver's contact output; never add or solve contacts."""

import numpy as np
from scipy.spatial.transform import Rotation


def finger_contacts(sim, prop="film_tab", arm="right"):
    coupled = sim.pipeline.solver
    solver = sim.pipeline.program.solver_for("phone")
    contacts = coupled.get_proxy_contacts("mujoco", "vbd")
    b0, b1, p0, p1, forces, count = solver.collect_rigid_contact_forces(
        coupled.entry_state("vbd", "output").body_q,
        coupled.entry_state("vbd", "input").body_q,
        contacts,
        sim.pipeline.sim_dt,
    )
    n = int(count.numpy()[0])
    a, b, x, y, f = (v.numpy()[:n] for v in (b0, b1, p0, p1, forces))
    normals = contacts.rigid_contact_normal.numpy()[:n]
    labels = [v.rsplit("/", 1)[-1] for v in solver.model.body_label]
    target = labels.index(prop)
    pose = sim["phone"].pose if prop == "phone" else sim["film"].entity.tab_pose(sim.state)
    rot = Rotation.from_quat(pose[3:])
    out = {}
    for i in range(n):
        other = int(b[i]) if a[i] == target else int(a[i]) if b[i] == target else -1
        if other < 0:
            continue
        name = labels[other]
        if not name.startswith(arm + "_link_") or not name.endswith("finger"):
            continue
        force = float(abs(np.dot(f[i], normals[i])))
        if force <= 1e-5:
            continue
        point = x[i] if a[i] == target else y[i]
        rec = out.setdefault(name, {"normal_N": 0.0, "points_local_mm": []})
        rec["normal_N"] += force
        rec["points_local_mm"].append((rot.inv().apply(point - np.asarray(pose[:3])) * 1000).tolist())
    return out
