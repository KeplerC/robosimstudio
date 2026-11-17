"""Do the plates penetrate anything? Measured on the *drawn* geometry, every recorded frame.

    python check.py run/hero_motion --jobs 6

Colliders are what the solver sees; what a viewer sees is the drawn meshes, and
"the plates do not penetrate" is a claim about those. So this ignores the
colliders entirely and asks, of each plate's drawn surface (sampled at ~1 mm)
against everything else as it is drawn:

* the **shelf** and the **drying racks** -- boards, cradle rails, divider
  dowels, end panels: primitives, drawn exactly as they collide;
* the **bench** -- how far below its top;
* the **fingers** -- inside the fingertip meshes as drawn (closed shells), and
  the other way round: the fingertips' own surfaces, sampled at ~0.45 mm,
  inside the plate (``fingers_reverse``), since a fingertip's edge could dip
  between two of the plate's samples;
* the rest of the **robot** -- every other drawn link of both arms: the gripper
  body, wrist and elbow meshes are closed shells; the two upper-arm links are
  drawn as open shells, so they are tested by their convex hulls, which can
  only over-report;
* the **other plates** -- the closest two plates' surfaces come, and whether
  either is ever inside the other (the plate mesh is watertight).

Positive depth is penetration [mm]; zero or negative is the closest the two
came. Writes ``penetration.json`` beside the take, and exits non-zero if any
depth is positive.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import newton
import numpy as np
import trimesh
from scipy.spatial import cKDTree

import primitives
import model as S
from demo import MODE
from robosimstudio.core.math import quat_to_mat


def _T(row) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(np.asarray(row[3:7], dtype=np.float64))
    T[:3, 3] = row[:3]
    return T


def measure(a, part: int, jobs: int) -> dict:
    """The worst of every measure over this part's frames: every ``jobs``-th of them from ``part``."""
    take = Path(a.take)
    meta = json.loads((take / "meta.json").read_text())
    data = np.load(take / "take.npz")
    body_q = data["body_q"]
    every = int(meta.get("record_every", 2))
    lay = meta["layout"]
    L = replace(S.LAYOUT, **{k: tuple(v) if isinstance(v, list) else v for k, v in lay.items()})
    import pickle
    with (take/'geometry.pkl').open('rb') as stream:
        model, plate_bodies = pickle.load(stream)

    # the plate as drawn, sampled, and as a solid (it is watertight)
    drawn = trimesh.load(S.ASSETS / "plate" / "plate.obj", force="mesh", process=False)
    samples, _ = trimesh.sample.sample_surface_even(drawn, a.samples, seed=3)
    samples = np.vstack([samples, drawn.vertices])

    shelf = [p for p in primitives.statics(model) if p.label.startswith("shelf_")]
    racks = [p for p in primitives.statics(model) if p.label.startswith("rack_")]

    labels = [str(x) for x in model.shape_label]
    shape_body = model.shape_body.numpy()
    shape_tf = model.shape_transform.numpy()
    shape_type = model.shape_type.numpy()
    shape_scale = model.shape_scale.numpy()
    shape_flags = model.shape_flags.numpy()
    visible = int(newton.ShapeFlags.VISIBLE)
    parts = []  # (category, label, body, shape-in-body, mesh, tested as)
    for s, lab in enumerate(labels):
        if not (lab.startswith("yam") and shape_type[s] == int(newton.GeoType.MESH) and shape_flags[s] & visible):
            continue
        src = model.shape_source[s]
        m = trimesh.Trimesh(np.asarray(src.vertices) * shape_scale[s][:3], np.asarray(src.indices).reshape(-1, 3), process=True)
        how = "mesh" if m.is_watertight else "hull"
        if how == "hull":
            m = m.convex_hull
        parts.append(("fingers" if ("finger" in lab or "_lf_" in lab or "_rf_" in lab) else "robot", lab.rsplit("/", 1)[-1], int(shape_body[s]), _T(shape_tf[s]), m, how))

    # ... and the reverse test's samples: each fingertip's drawn surface
    tips = []
    for cat, lab, fb, Tl, mesh, how in parts:
        if cat == "fingers":
            pts, _ = trimesh.sample.sample_surface_even(mesh, 80000, seed=5)
            tips.append((lab, fb, Tl, np.vstack([pts, mesh.vertices])))
    lo_p, hi_p = drawn.bounds

    worst = {k: {"depth_mm": -1e9, "t": None, "plate": None, "with": None}
             for k in ("shelf", "racks", "bench", "fingers", "fingers_reverse", "robot")}
    # the fingers again, by what the hand was doing: "<hand>:<leg>" -> worst
    labels_rec = data["labels"] if "labels" in data else None
    by_leg: dict[str, dict] = {}
    closest_plates = {"clearance_mm": 1e9, "t": None, "pair": None}
    frames = range(0, len(body_q), a.stride)[part::jobs]
    for f in frames:
        bq = body_q[f]
        t = f * every / 60.0
        world = []
        for pi, b in enumerate(plate_bodies):
            T = _T(bq[b])
            P = samples @ T[:3, :3].T + T[:3, 3]
            world.append((P, T))
            for key, prims in (("shelf", shelf), ("racks", racks)):
                d, which = primitives.nearest(P, prims)
                if np.isfinite(d) and -d * 1000 > worst[key]["depth_mm"]:
                    worst[key] = {"depth_mm": round(-d * 1000, 3), "t": round(t, 2), "plate": pi, "with": which}
            onb = (P[:, 0] > 0.3025) & (P[:, 0] < 0.8975) & (np.abs(P[:, 1]) < 0.65)
            if onb.any():
                depth = S.BENCH_Z - float(P[onb, 2].min())
                if depth * 1000 > worst["bench"]["depth_mm"]:
                    worst["bench"] = {"depth_mm": round(depth * 1000, 3), "t": round(t, 2), "plate": pi, "with": "bench"}
            for cat, lab, fb, Tl, mesh, how in parts:
                Tf = _T(bq[fb]) @ Tl
                Q = (P - Tf[:3, 3]) @ Tf[:3, :3]
                lo, hi = mesh.bounds
                near = np.all((Q > lo - 1e-3) & (Q < hi + 1e-3), axis=1)
                if not near.any():
                    continue
                q = Q[near]
                _, dist, _ = trimesh.proximity.closest_point(mesh, q)
                inside = mesh.contains(q)
                signed = np.where(inside, dist, -dist)
                k = int(np.argmax(signed))
                if labels_rec is not None and cat == "fingers":
                    hand = lab.split("_", 1)[0]
                    leg = dict(p.split(":", 1) for p in str(labels_rec[f]).split("|")).get(hand, "?")
                    key = f"{hand}:{leg}"
                    if key not in by_leg or signed[k] * 1000 > by_leg[key]["depth_mm"]:
                        by_leg[key] = {"depth_mm": round(float(signed[k]) * 1000, 3), "t": round(t, 2), "plate": pi, "with": lab}
                if signed[k] * 1000 > worst[cat]["depth_mm"]:
                    worst[cat] = {"depth_mm": round(float(signed[k]) * 1000, 3), "t": round(t, 2), "plate": pi,
                                  "with": lab, "tested_as": how}
            for lab, fb, Tl, pts in tips:
                Tf = _T(bq[fb]) @ Tl
                Lp = (pts @ Tf[:3, :3].T + Tf[:3, 3] - T[:3, 3]) @ T[:3, :3]
                near = np.all((Lp > lo_p - 1e-3) & (Lp < hi_p + 1e-3), axis=1)
                if not near.any():
                    continue
                sd = trimesh.proximity.signed_distance(drawn, Lp[near])  # + inside the plate
                k = int(np.argmax(sd))
                if sd[k] * 1000 > worst["fingers_reverse"]["depth_mm"]:
                    worst["fingers_reverse"] = {"depth_mm": round(float(sd[k]) * 1000, 3), "t": round(t, 2), "plate": pi, "with": lab}
        # plate against plate: closest approach of the two surfaces, and a solid
        # test on any point that came within a millimetre -- for pairs whose
        # bounding spheres (radius + 5 mm) touch at all
        reach = 2.0 * (S.PLATE_RADIUS + 0.005)
        centres = [T[:3, 3] + T[:3, :3] @ np.array([0.0, 0.0, 0.5 * S.PLATE_DEPTH]) for _, T in world]
        trees = {}
        for i in range(len(world)):
            for j in range(i + 1, len(world)):
                if np.linalg.norm(centres[i] - centres[j]) > reach:
                    continue
                if j not in trees:
                    trees[j] = cKDTree(world[j][0])
                d, _ = trees[j].query(world[i][0], k=1, distance_upper_bound=0.05)
                dmin = float(d.min())
                if dmin < 0.001:
                    Tj = world[j][1]
                    close = world[i][0][d < 0.001]
                    local = (close - Tj[:3, 3]) @ Tj[:3, :3]
                    if drawn.contains(local).any():
                        dmin = -1.0
                if dmin * 1000 < closest_plates["clearance_mm"]:
                    closest_plates = {"clearance_mm": round(dmin * 1000, 3), "t": round(t, 2), "pair": [i, j]}
        if (f // a.stride) % (200 * jobs) < jobs and part == 0:
            print(f"frame {f}/{len(body_q)}", flush=True)
    return {
        "frames_checked": len(frames),
        "samples_per_plate": int(len(samples)),
        "worst_penetration": worst,
        "plates_closest_approach": closest_plates,
        "fingers_by_leg": by_leg,
    }


def merge(parts: list[dict]) -> dict:
    out = dict(parts[0])
    out["frames_checked"] = sum(p["frames_checked"] for p in parts)
    out["worst_penetration"] = {k: max((p["worst_penetration"][k] for p in parts), key=lambda v: v["depth_mm"])
                                for k in parts[0]["worst_penetration"]}
    out["plates_closest_approach"] = min((p["plates_closest_approach"] for p in parts), key=lambda v: v["clearance_mm"])
    legs: dict[str, dict] = {}
    for p in parts:
        for k, v in p["fingers_by_leg"].items():
            if k not in legs or v["depth_mm"] > legs[k]["depth_mm"]:
                legs[k] = v
    out["fingers_by_leg"] = legs
    return out



class Array:
    def __init__(self,data): self.data=np.asarray(data)
    def numpy(self): return self.data

def export(sim,take):
    """Freeze only the drawn model for an offline check; never build a second sim."""
    import pickle
    from types import SimpleNamespace
    take=Path(take)
    m=sim.model
    model=SimpleNamespace(shape_label=list(m.shape_label))
    for key in ('shape_body','shape_transform','shape_type','shape_scale','shape_flags'):
        setattr(model,key,Array(getattr(m,key).numpy()))
    model.shape_source=[]
    for src in m.shape_source:
        if src is not None and hasattr(src,'vertices'):
            model.shape_source.append(SimpleNamespace(vertices=np.asarray(src.vertices),indices=np.asarray(src.indices)))
        else: model.shape_source.append(None)
    with (take/'geometry.pkl').open('wb') as stream:
        pickle.dump((model,[int(sim[f"plate_{i}"].bodies[0]) for i in range(6)]),stream)
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("take")
    ap.add_argument("--samples", type=int, default=40000)
    ap.add_argument("--stride", type=int, default=1, help="check every Nth recorded frame")
    ap.add_argument("--jobs", type=int, default=8, help="processes to split the frames over")
    ap.add_argument("--part", type=int, default=-1, help=argparse.SUPPRESS)
    a = ap.parse_args()
    take = Path(a.take)
    if a.part >= 0:  # one worker's share, for the parent to merge
        (take / f"penetration.part{a.part}.json").write_text(json.dumps(measure(a, a.part, a.jobs)))
        return
    if a.jobs > 1:
        cmd = [sys.executable, __file__, str(take), "--samples", str(a.samples), "--stride", str(a.stride), "--jobs", str(a.jobs)]
        # one thread each: left to themselves the workers' BLAS/OpenMP pools
        # size to the machine, and eight of them oversubscribe it many times over
        env = dict(os.environ, **{v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")})
        procs = [subprocess.Popen(cmd + ["--part", str(k)], stdout=None if k == 0 else subprocess.DEVNULL, env=env) for k in range(a.jobs)]
        if any([p.wait() for p in procs]):
            sys.exit("a worker failed")
        paths = [take / f"penetration.part{k}.json" for k in range(a.jobs)]
        report = merge([json.loads(p.read_text()) for p in paths])
        for p in paths:
            p.unlink()
    else:
        report = measure(a, 0, 1)
    worst, closest_plates = report["worst_penetration"], report["plates_closest_approach"]
    for v in worst.values():
        if v["depth_mm"] <= -1e8:  # nothing of it ever came within a millimetre of a plate's bounds
            v.update(depth_mm=None, note="never within 1 mm of a plate")
    report["fingers_by_leg"] = dict(sorted(report["fingers_by_leg"].items(), key=lambda kv: -kv[1]["depth_mm"]))
    report["note"] = ("depth_mm > 0 is a drawn plate point inside the other drawn geometry; <= 0 is the closest it came. "
                      "plates clearance < 0 means one plate's surface was found inside the other.")
    (take / "penetration.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    bad = [k for k, v in worst.items() if v["depth_mm"] is not None and v["depth_mm"] > 0.0] + (["plates"] if closest_plates["clearance_mm"] < 0 else [])
    if bad:
        print("PENETRATION:", ", ".join(bad))
        sys.exit(1)


if __name__ == "__main__":
    main()
