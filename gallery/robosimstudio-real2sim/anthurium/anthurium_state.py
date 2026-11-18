"""Live-state measurements and reproducible shape bake for this scene only."""

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def plant_state(sim, blades, origin, camera):
    origin = np.asarray(origin)
    axis = np.asarray(camera["look_at"]) - camera["eye"]
    axis /= np.linalg.norm(axis)
    rest = sim.model.particle_q.numpy()
    velocity = sim.state.particle_qd.numpy()
    result = {}
    for b in blades:
        name = b["name"]
        ref = sim[name]
        points = ref.particle_q.astype(float) - origin
        authored = np.asarray(b["vertices"])
        # Trimesh may reorder vertices; recover the authored 10x10 mesh order.
        ids = cKDTree(rest[ref.particles].astype(float) - origin).query(authored)[1]
        ordered = points[ids]
        _, _, vt = np.linalg.svd(points - points.mean(0), full_matrices=False)
        n = vt[-1]
        n *= 1 if np.dot(n, b["inferred_normal"]) >= 0 else -1
        angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(n, axis)), 0, 1))))
        stalk = sim[name + "_stalk"]
        nodes = stalk.entity.centerline(sim.pipeline.snapshot).astype(float) - origin
        direction = ordered[110] - ordered[5]
        direction /= np.linalg.norm(direction)
        side = np.cross(n, direction)
        side /= np.linalg.norm(side)
        direction = np.cross(side, n)
        quat = Rotation.from_matrix(np.column_stack((direction, side, n))).as_quat()
        item = {
            "centre": points.mean(0).tolist(),
            "normal": n.tolist(),
            "angle_to_photo_axis_deg": angle,
            "angle_to_photo_forward_deg": float(np.degrees(np.arccos(np.clip(np.dot(n, axis), -1, 1)))),
            "pose": [*points.mean(0), *quat],
            "particles": ordered.tolist(),
            "stalk_nodes": nodes.tolist(),
            "particle_count": len(points),
            "held_count": len(ref.entity.held),
            "node_count": len(nodes),
            "capsule_count": len(stalk.bodies),
            "speed_p95_m_s": float(np.quantile(np.linalg.norm(velocity[ref.particles], axis=1), 0.95)),
            "cue": b["cue"],
            "extent_bench_m": np.ptp(points, axis=0).tolist(),
        }
        if b.get("spathe"):
            pose = sim.state.body_q.numpy()[sim[name + "_spadix"].bodies[0]].astype(float)
            pose[:3] -= origin
            item["spadix_pose"] = pose.tolist()
        result[name] = item
    return result


def rest_measure(sim, blades, frames=60):
    names = [b["name"] for b in blades]
    start = {n: sim[n].particle_q.copy() for n in names}
    tips = {n: sim[n + "_stalk"].entity.centerline(sim.pipeline.snapshot)[-1].copy() for n in names}
    t = time.perf_counter()
    for _ in range(frames):
        sim._step_once()
    elapsed = time.perf_counter() - t
    readings = {}
    for n in names:
        delta = sim[n].particle_q - start[n]
        tip = sim[n + "_stalk"].entity.centerline(sim.pipeline.snapshot)[-1]
        readings[n] = {
            "centre_drift_mm": float(np.linalg.norm(delta.mean(0)) * 1000),
            "particle_drift_p95_mm": float(np.quantile(np.linalg.norm(delta, axis=1), 0.95) * 1000),
            "stalk_tip_drift_mm": float(np.linalg.norm(tip - tips[n]) * 1000),
        }
    return {
        "frames": frames,
        "elapsed_wall_s": elapsed,
        "ms_per_control_step": elapsed / frames * 1000,
        "readings": readings,
    }


def bake(sim, blades, origin, camera, path):
    result = plant_state(sim, blades, origin, camera)
    for b in blades:
        name = b["name"]
        item = result[name]
        item["rest_vertices"] = b["vertices"]
        item["rest_nodes"] = b["nodes"]
        poses = sim.state.body_q.numpy()[sim[name + "_stalk"].bodies].astype(float)
        poses[:, :3] -= origin
        item["stalk_body_poses"] = poses.tolist()
        item["stalk_joint_q"] = [
            sim.state.joint_q.numpy()[
                int(sim.model.joint_q_start.numpy()[j]) : int(sim.model.joint_q_start.numpy()[j + 1])
            ].tolist()
            for j in sim[name + "_stalk"].entity.joints
        ]
    Path(path).write_text(json.dumps(result, indent=2))
    return {
        "saved": str(path),
        "leaves_head_on_under_20": sum(
            x["angle_to_photo_axis_deg"] < 20 for n, x in result.items() if n.startswith("leaf")
        ),
        "normal_angles": {
            n: [*[round(v, 4) for v in x["normal"]], round(x["angle_to_photo_axis_deg"], 2)] for n, x in result.items()
        },
    }


def install_initial_state(sim, blades, origin, baked):
    """Restore baked coordinates, retaining the original elastic rest geometry.

    A custom initial state is not a public SceneSpec field. This local reset
    wrapper writes both Newton state buffers after the normal reset. Nothing
    is frozen during stepping; the standard pipeline and material hooks run.
    """
    origin = np.asarray(origin)
    records = []
    rest = sim.model.particle_q.numpy().astype(float)
    for b in blades:
        name = b["name"]
        data = baked.get(name)
        if not data or "rest_vertices" not in data:
            continue
        if not np.allclose(data["rest_vertices"], b["vertices"], atol=1e-7):
            continue
        ref = sim[name]
        ids = cKDTree(rest[ref.particles] - origin).query(np.asarray(b["vertices"]))[1]
        records.append((name, data, ref.particles[ids]))
    base_reset = sim.reset_scene
    water_path = Path(__file__).resolve().parent / 'assets' / 'water_rest_480.json'
    water_bake = json.loads(water_path.read_text()) if water_path.exists() else None

    def reset_baked():
        base_reset()
        # Solver construction restores some static flags after handles are
        # built. Hide only the cell's display block; its rigid contacts remain.
        import newton
        flags=sim.model.shape_flags.numpy()
        flags[sim['table'].handle.shapes] &= ~int(newton.ShapeFlags.VISIBLE)
        sim.model.shape_flags.assign(flags)
        pq = sim.state.particle_q.numpy()
        bq = sim.state.body_q.numpy()
        jq = sim.state.joint_q.numpy()
        if water_bake:
            pq[sim['water'].particles] = np.asarray(water_bake['particles_bench']) + origin
        for name, data, ids in records:
            pq[ids] = np.asarray(data["particles"]) + origin
            poses = np.asarray(data["stalk_body_poses"]).copy()
            poses[:, :3] += origin
            bq[sim[name + "_stalk"].bodies] = poses
            starts = sim.model.joint_q_start.numpy()
            for j, values in zip(sim[name + "_stalk"].entity.joints, data["stalk_joint_q"], strict=False):
                jq[int(starts[j]) : int(starts[j + 1])] = values
            if "spadix_pose" in data:
                spike = sim[name + "_spadix"].entity
                holder = bq[spike.holder]
                offset = np.array([spike.offset[i] for i in range(7)])
                rotation = Rotation.from_quat(holder[3:])
                bq[spike.body, :3] = holder[:3] + rotation.apply(offset[:3])
                bq[spike.body, 3:] = (rotation * Rotation.from_quat(offset[3:])).as_quat()
        for state in (sim.pipeline.state_0, sim.pipeline.state_1):
            state.particle_q.assign(pq)
            state.particle_qd.zero_()
            state.body_q.assign(bq)
            state.body_qd.zero_()
            state.joint_q.assign(jq)
            state.joint_qd.zero_()
            if water_bake:
                for field in ('particle_elastic_strain', 'particle_Jp'):
                    array = getattr(state.mpm, field)
                    values = array.numpy()
                    values[sim['water'].particles] = np.asarray(water_bake[field])
                    array.assign(values)
        sim.pipeline.mark_dirty()

    sim.reset_scene = reset_baked
    sim.reset()
    return [n for n, _, _ in records]
