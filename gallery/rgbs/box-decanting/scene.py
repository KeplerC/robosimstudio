"""Box decanting: calibrated gallery reconstruction; see REPORT.md for limits."""

import math
from dataclasses import replace

from props import Carton, SafetyCutter
from room_look import decant_room
from scipy.spatial.transform import Rotation

import robosimstudio as rss
from robosimstudio.core.render.room import LAB, LAB_GRID_MAT, LAB_MAT_GREY
from robosimstudio.robots.registry import get_embodiment

PHOTO = rss.CameraSpec(
    name="photo",
    width=1920,
    height=1080,
    fx=1799.45,
    fy=1799.45,
    cx=960,
    cy=540,
    eye=(-0.5489, -0.028, 1.3284),
    look_at=(0.3988, 0.0322, 1.0152),
    up=(0.3131, 0.0117, 0.9496),
)
WIDE = rss.CameraSpec(name="wide", width=1280, height=720, eye=(-0.28, -0.08, 1.32), look_at=(0.51, 0, 0.99), fov_y=48)
physics = replace(
    rss.MUJOCO_500HZ,
    collision=rss.CollisionSpec(contact_ke=6e4, contact_kd=500, rigid_gap=0.0002, rigid_contact_max=8192),
    backends=(
        (
            "*",
            rss.backend(
                "mujoco", nconmax=2048, njmax=4096, enable_multiccd=True, iterations=80, ls_iterations=40, impratio=10
            ),
        ),
    ),
)
scene = rss.Scene(
    cell="bench_075",
    name="box-decanting",
    physics=physics,
    render=replace(LAB, scenery=decant_room.__name__, key_intensity=1.8, ambient=(1.6, 1.6, 1.6)),
)
scene.add(PHOTO)
scene.add(WIDE)
# All finishes are procedural library textures; no reference pixels are used.
scene.add(
    rss.slab(
        name="mat",
        half=(0.2975, 0.65, 0.002),
        at=(0, 0, 0.002),
        color=LAB_MAT_GREY,
        finish=replace(LAB_GRID_MAT, uv_offset=(0.033333, 0)),
        friction=0.75,
    )
)
yam = get_embodiment("yam")


def white_shells(builder, shapes):
    for s in range(*shapes):
        label = builder.shape_label[s].lower().rsplit("/", 1)[-1]
        if label.endswith("_visual"):
            builder.shape_color[s] = (
                (0.85, 0.85, 0.87) if ("link_2_geom_2" in label or "link_3_geom_2" in label) else (0.025, 0.025, 0.029)
            )
    return 0


yam = replace(
    yam,
    appearance=white_shells,
    arms=(
        replace(
            yam.arms[0],
            home_q=(
                0.16803716776334085,
                0.4497044838933234,
                0.00010156378032410399,
                0.38512024372268683,
                0.296977233715606,
                -0.5280321938030506,
            ),
        ),
        replace(
            yam.arms[1],
            home_q=(
                -0.11830375559974,
                0.48070837061443367,
                0.00010259351179980727,
                0.27574731997591595,
                -0.4034328030323448,
                0.39214480466759905,
            ),
        ),
    ),
)
scene.add(rss.robot(yam, arms_at={"left": (-0.3475, 0.300824, 0), "right": (-0.330456, -0.300824, 0)}))
CARTON = Carton(
    pose=(-0.05367, 0.01821, 0.0051), quat=tuple(Rotation.from_euler("z", -math.pi / 2 + 0.02404).as_quat())
)
carton = scene.add(CARTON)
CUTTER = SafetyCutter(
    name="cutter",
    pose=(-0.00937, -0.17726, 0.0167),
    quat=(0, 0, 0, 1),
)
cutter = scene.add(CUTTER)
from motion import Drive, routine

__all__ = ["Drive", "routine", "scene"]
