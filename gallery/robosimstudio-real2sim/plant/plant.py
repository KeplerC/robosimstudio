"""Seven settled Alocasia blades and their elastic, rooted petioles."""

import json
from pathlib import Path

import numpy as np
from geometry import Layout, Leaf, Stalk, shell
from plant_assets import LEAF_NORMALS
from rig import plant_spec

ROOT = Path(__file__).resolve().parent
ORDER = (
    "leaf_back",
    "leaf_left",
    "leaf_front_left",
    "leaf_middle_right",
    "leaf_right",
    "leaf_front_right",
    "leaf_edge",
)
LAW = {
    "k_root": 3.0,
    "k_node": 1.5,
    "k_leaf": 0.10,
    "armature": 1e-6,
    "damping": 0.025,
    "leaf_damping": 0.006,
    "leaf_areal_density": 0.15,
    "stalk_density": 550,
}


def add_plant(scene):
    data = json.loads((ROOT / "assets/plant_layout.json").read_text())
    shapes = data["initial_state_bake"]["shape"]
    props = {p["name"]: p for p in data["props"]}
    layout = Layout(params={}, pot={})
    for i, name in enumerate(ORDER):
        sh = shapes[name]
        p = props[name]
        f = p["finish"]
        V = np.asarray(sh["points"])
        F = np.asarray(sh["triangles"]).reshape(-1, 3)
        nodes = np.asarray(shapes[f"stalk_{i + 1}"]["nodes"])
        nodes[0, 2] = 0.1265  # capsule base meets the recessed soil surface
        uv = np.asarray(sh["authored_uvs"]) * np.asarray(f["uv_scale"])
        lf = Leaf(
            name=name,
            kind="blade",
            V=V,
            F=F,
            UV=uv,
            tex="",
            color=(1, 1, 1),
            junction=nodes[-1],
            tex_c=str(ROOT / "assets" / Path(f["texture"]).name),
        )
        LEAF_NORMALS[lf.tex_c] = str(ROOT / "assets" / Path(f["normal_texture"]).name)
        lf.shell_V, lf.shell_F = shell(V, F, 0.001)
        lf.shell_UV = np.concatenate([uv, uv])
        lf.area = float(np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum() / 2)
        layout.leaves.append(lf)
        layout.stalks.append(Stalk(name=f"stalk_{i + 1}", nodes=nodes, radius=0.002, leaf=name))
    scene.add(plant_spec("plant", layout, LAW))
    return layout
