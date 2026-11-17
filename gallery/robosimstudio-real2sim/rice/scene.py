# SPDX-License-Identifier: Apache-2.0
"""A live, settled red-rice spill on bench_075, matched to the supplied still.

Scale is anchored on the 70 mm cup mouth. The rice is rss.heap(material="rice")
on GRANULAR_60HZ with a 3.6 mm grid and 240 Hz material steps. A mesh seeds the
spill; settled.npz records a new ten-second settle under this solver. Rebuild and Reset
restore it, with zero initial velocity.
A public particle-surface callback draws long grains at the simulated points.
Residual creep is measured by the gate and a separate three-second probe.
"""

import json
from pathlib import Path

import choreography
import numpy as np
from dustpan import dustpan
from rice_helpers import (
    RiceSkin,
    cup_assets,
    dress_rice,
    install_bake,
    marble_texture,
    spill_asset,
)

import robosimstudio as rss
from robosimstudio.core.render import Finish, RenderSpec
from robosimstudio.core.render.scenery import register_scenery
from robosimstudio.scene.registry import register_cell
from robosimstudio.specs import AssetSpec

# fmt: off
BUILD = {}  # use the page's visible GPU
# fmt: on
HERE = Path(__file__).resolve().parent
BENCH = np.array([0.6, 0.0, 0.75])
BAKE = HERE / "assets/settled.npz"

# The layout table: all prop sizes [m], bench-frame poses [xyz, xyzw], colours
# and finishes. Rice size is the measured settled particle envelope; its pose is
# the material anchor. Changing that pose also transforms the baked particles.
# bench_075 represents the counter with a solid fixed block down to the floor;
# its marble is a visual finish flush with the existing work surface.
LAYOUT = {
    "rice": {
        "kind": "simulated material",
        "material_kind": "granular",
        "preset": "rice",
        "size": [0.185983, 0.167408, 0.011715],
        "pose": [-0.029947337384201744, -0.010919591680729465, 0.0003, 0, 0, 0, 1],
        "color": "#8d451f",
        "finish": {
            "roughness": 0.29,
            "metallic": 0.0,
            "grain_palette": [
                "#7c4118",
                "#844519",
                "#88491c",
                "#8e4d1f",
                "#935122",
                "#995727",
                "#9f5e2d",
                "#8a4a1d",
            ],
            "bran_highlight_scale": 1.38,
        },
    },
    "cup": {
        "kind": "rigid",
        "size": [0.070, 0.070, 0.060],
        "pose": [
            0.09525937557220461,
            0.027417883276939392,
            0.032511889934539795,
            0.42194321751594543,
            -0.5351617932319641,
            0.00961634237319231,
            0.7317605018615723,
        ],
        "base_diameter": 0.062,
        "wall_thickness": 0.002,
        "mass_kg": 0.09,
        "friction": 0.8,
        "color": "#e3e9eb",
        "finish": {"roughness": 0.75, "metallic": 0.0},
    },
    "table": {
        "kind": "fixed",
        "size": [0.595, 1.30, 0.75],
        "pose": [0, 0, -0.375, 0, 0, 0, 1],
        "color": "#cbd0d1",
        "finish": {"roughness": 0.65, "texture": "assets/marble_final.png"},
    },
}
RICE_PARAMETERS = {
    "particle_spacing": 0.0024,
    "jitter": 0.45,
    "containers": ("cup",),
    "container_margin": 0.004,
    "settle_frames": 0,
}
PHOTO = rss.CameraSpec(
    name="photo",
    width=1920,
    height=1280,
    fov_y=21.4,
    eye=(0.405, 0.041, 1.530),
    look_at=(0.600, 0.041, 0.750),
    up=(0, 0, 1),
    near=0.01,
    far=5,
)
OUTLINE_PX = json.loads((HERE / "assets/outline_pixels.json").read_text())["outline"]
HOLES_PX = [
    (1150, 798, 0.0035),
    (1414, 767, 0.0040),
    (1194, 652, 0.0025),
    (958, 866, 0.0030),
]

cup_path, cup_colliders = cup_assets(LAYOUT["cup"]["size"])
if BAKE.exists():
    spill_path = str(HERE / "assets/spill.obj")
else:
    spill_path, _, _, _ = spill_asset(PHOTO, OUTLINE_PX, HOLES_PX)
marble = marble_texture(PHOTO)


def rgb(value):
    return tuple(int(value[i : i + 2], 16) / 255 for i in (1, 3, 5))


def prop_finish(name):
    values = {k: v for k, v in LAYOUT[name]["finish"].items() if k not in {"grain_palette", "bran_highlight_scale"}}
    if values.get("texture"):
        values["texture"] = str(HERE / values["texture"])
    return Finish(**values)


@register_scenery("rice_marble_finished")
def dress(kit):
    # The finish lies on the bench, without a new collider in the task area.
    vertices = np.array(
        [
            [0.3025, -0.65, 0.75005],
            [0.8975, -0.65, 0.75005],
            [0.8975, 0.65, 0.75005],
            [0.3025, 0.65, 0.75005],
        ]
    )
    kit.mesh(
        "marble_counter",
        vertices,
        [[0, 1, 2], [0, 2, 3]],
        color=(1, 1, 1),
        finish=prop_finish("table"),
        uvs=[[0, 0], [1, 0], [1, 1], [0, 1]],
    )
    if hasattr(kit, "stage"):
        from pxr import Gf, UsdGeom

        # Broad daylight adjusted to reproduce the soft leftward cup shadow.
        panel = UsdGeom.Xform.Get(kit.stage, "/root/_LabPanel")
        if panel:
            panel.ClearXformOpOrder()
            panel.AddTranslateOp().Set(Gf.Vec3d(0.80, -0.65, 1.95))
        dress_rice(kit, sim.handles.deformables["rice"].surface, LAYOUT["rice"]["finish"])


base_cell = rss.get_cell("bench_075")
register_cell(
    rss.replace(
        base_cell,
        name="rice_counter",
        statics=tuple(rss.replace(st, friction=0.4) if st.name == "table" else st for st in base_cell.statics),
    )
)
scene = rss.Scene(
    cell="rice_counter",
    name="rice",
    physics=rss.GRANULAR_60HZ.with_backend(
        "granular", rss.GRANULAR_60HZ.backend_for("granular").with_(voxel_size=0.0036, step_hz=240.0)
    )
    .with_backend("articulated", rss.backend("mujoco", nconmax=16384, njmax=32768, ls_iterations=50))
    .with_backend("rigid", rss.backend("mujoco", nconmax=16384, njmax=32768, ls_iterations=50)),
    render=RenderSpec(
        environment="lab",
        scenery="rice_marble_finished",
        quality="final",
        key_intensity=1.6,
        ambient=(0.65, 0.65, 0.65),
    ),
)
scene.add(
    rss.robot("yam", arms_at={"right": (0.27, -0.12, 0.022, 0, 0, 0.70710678, 0.70710678)})
)  # stock servos and fingers; far-side mount clears the photo view
scene.add(PHOTO)
c = LAYOUT["cup"]
scene.add(
    rss.body(
        name="cup",
        asset=AssetSpec(
            path=cup_path,
            collision_path=cup_colliders[0],
            collision_paths=cup_colliders[1:],
        ),
        at=tuple(c["pose"][:3]),
        quat=tuple(c["pose"][3:]),
        mass=c["mass_kg"],
        friction=c["friction"],
        color=rgb(c["color"]),
        finish=prop_finish("cup"),
        obb_grasp=False,
    )
)
r = LAYOUT["rice"]
scene.add(
    rss.heap(
        name="rice",
        material="rice",
        shape="mesh",
        asset=AssetSpec(path=spill_path),
        at=tuple(r["pose"][:3]),
        quat=tuple(r["pose"][3:]),
        color=rgb(r["color"]),
        finish=prop_finish("rice"),
        **RICE_PARAMETERS,
    )
)
scene.add(
    dustpan(
        lip=0.0002, at=(-0.037, 0.095, 0.0), color=(0.30, 0.45, 0.38), finish=Finish(roughness=0.28), obb_grasp=False
    )
)
scene.add(
    rss.box(
        name="pusher",
        entity="rice_pusher",
        half=(0.085, 0.010, 0.020),
        at=(-0.037, -0.13, 0.020),
        mass=0.10,
        friction=0.3,
        color=(0.55, 0.38, 0.22),
        finish=Finish(roughness=0.45),
    )
)
scene.add(
    rss.CameraSpec(name="wide", width=1280, height=960, fov_y=42, eye=(0.18, -0.80, 1.65), look_at=(0.55, 0.03, 0.98))
)
scene.add(rss.slab(name="kitchen_floor", half=(2, 2, 0.04), at=(0, 0, -0.79), color=(0.45, 0.40, 0.33)))
scene.add(rss.slab(name="back_wall", half=(0.04, 2, 1.3), at=(-0.8, 0, 0.5), color=(0.79, 0.82, 0.80)))
scene.add(rss.slab(name="end_wall", half=(2.0, 0.04, 1.3), at=(0.4, 1.15, 0.5), color=(0.79, 0.82, 0.80)))
scene.add(rss.slab(name="front_wall", half=(0.04, 2, 1.3), at=(1.2, 0, 0.5), color=(0.82, 0.80, 0.75)))
for index, y in enumerate((-0.48, -0.16, 0.16, 0.48)):
    scene.add(
        rss.slab(
            name=f"cabinet_door_{index}",
            half=(0.008, 0.155, 0.30),
            at=(0.307, y, -0.37),
            color=(0.40, 0.46, 0.43),
            finish=Finish(roughness=0.5),
        )
    )
    scene.add(
        rss.slab(
            name=f"cabinet_handle_{index}",
            half=(0.008, 0.004, 0.045),
            at=(0.323, y + 0.11, -0.22),
            color=(0.52, 0.44, 0.27),
            finish=Finish(roughness=0.3, metallic=0.7),
        )
    )
# A steel support under the catalog mounts, which stand 22 mm above the marble.
scene.add(
    rss.slab(
        name="mounting_rail",
        half=(0.080, 0.36, 0.031),
        at=(-0.3475, 0.095, -0.009),
        color=(0.16, 0.18, 0.18),
        finish=Finish(roughness=0.4, metallic=0.6),
    )
)
for i, y in enumerate((-0.20, 0.39)):
    scene.add(
        rss.slab(
            name=f"mounting_leg_{i}",
            half=(0.025, 0.025, 0.355),
            at=(-0.3475, y, -0.395),
            color=(0.16, 0.18, 0.18),
            finish=Finish(roughness=0.4, metallic=0.6),
        )
    )
scene.add(rss.slab(name="right_mount", half=(0.06, 0.07, 0.011), at=(0.27, -0.12, 0.011), color=(0.16, 0.18, 0.18)))
scene.add(
    rss.slab(
        name="backsplash",
        half=(0.30, 0.012, 0.30),
        at=(0, 0.66, 0.30),
        color=(0.75, 0.79, 0.76),
        finish=Finish(roughness=0.35),
    )
)
scene.add(rss.slab(name="oak_shelf", half=(0.26, 0.075, 0.015), at=(0, 0.59, 0.45), color=(0.48, 0.32, 0.19)))
for i in range(5):
    scene.add(
        rss.slab(
            name=f"tile_seam_{i}",
            half=(0.0008, 0.001, 0.30),
            at=(-0.24 + 0.12 * i, 0.647, 0.30),
            color=(0.66, 0.70, 0.67),
        )
    )
for i in range(3):
    scene.add(
        rss.slab(
            name=f"tile_row_{i}", half=(0.30, 0.001, 0.0008), at=(0, 0.647, 0.15 * (i + 1)), color=(0.66, 0.70, 0.67)
        )
    )
sim = scene.build(control_mode="joint_position", **BUILD)

sim.handles.deformables["rice"].surface = RiceSkin(len(sim["rice"].particle_q))
if BAKE.exists():
    install_bake(sim, BAKE)

if __name__ == "__main__":
    rss.view(sim)

routine = choreography.routine
measure = choreography.measure
