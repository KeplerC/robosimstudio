"""Workshop gallery port: a named D grip and the shipped brush's contact lattice."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from workshop_assets import paver_edges, tex, workshop_light
from workshop_brush import oriented_brush
from workshop_layout import CAMERA, LAYOUT

import robosimstudio as rss
from robosimstudio.core.render import Finish, RenderSpec
from robosimstudio.robots.registry import get_embodiment

HERE = Path(__file__).resolve().parent
ORIGIN = np.array([0.6, 0, 0.75])
START = np.array(LAYOUT["brush"]["settled_pose"][:3])
YAW = -0.17005
QUAT = Rotation.from_euler("z", YAW).as_quat()
DOWN = (Rotation.from_euler("z", YAW + np.pi / 2) * Rotation.from_quat([0, 1, 0, 0])).as_quat()
PATCH = (-0.10, -0.12, 0.10, 0.0)
LO = ORIGIN[:2] + PATCH[:2]
HI = ORIGIN[:2] + PATCH[2:]
scene = rss.Scene(
    cell="bench_075",
    name="workshop",
    physics=rss.BRUSH_60HZ,
    render=RenderSpec(environment="none", scenery=workshop_light, quality="final", background=(0.8, 0.8, 0.8)),
)
yam = get_embodiment("yam")
yam = replace(
    yam,
    arms=tuple(
        replace(
            a,
            home_q=(0, 0.8, 2.3, -1.4, 0, 0),
            home_gripper_q=(0.0475,),
        )
        for a in yam.arms
    ),
)
scene.add(rss.robot(yam, at=(-0.32, 0, 0.025), arms_at={"right": (-0.15, -0.20, 0.025), "left": (-0.30, 0.35, 0.025)}))


def rgb(c):
    return tuple(int(c[i : i + 2], 16) / 255 for i in (1, 3, 5))


for name, p in LAYOUT.items():
    if name in ("brush", "brush_plastic"):
        continue
    finish = Finish(roughness=0.95)
    if name.startswith("paver_"):
        finish = Finish(
            roughness=0.95, texture=tex("aggregate_coarse.png"), uv_scale=(2.5, 2.5), uv_offset=p["uv_offset"]
        )
    if name == "skirting":
        finish = Finish(roughness=0.4, texture=tex("wood_straight.png"), uv_scale=(0.2, 10))
    if name in ("wall", "wall_band"):
        finish = Finish(
            roughness=0.98, texture=tex("plaster_scan.png" if name == "wall" else "band_scan.png"), uv_scale=(3, 3)
        )
    scene.add(
        rss.slab(
            name=name,
            half=tuple(np.array(p["size"]) / 2),
            at=(*p["at"], *p["quat"]),
            color=rgb("#9a9485" if name == "paving_bed" else p["color"]),
            finish=finish,
            collide=p["kind"] != "visual",
        )
    )
# Additive terrace: continuation at the existing paving plane hides the bench sides.
for ix in range(-6, 3):
    for iy in range(-10, 11):
        if ix >= 0 and 0 <= iy < 7:
            continue
        x, y = -0.2975 + (ix + 0.5) * 0.2, -0.65 + (iy + 0.5) * 0.2
        if x > 0.285:
            continue
        scene.add(
            rss.slab(
                name=f"terrace_{ix}_{iy}",
                half=(0.0975, 0.0975, 0.003),
                at=(x, y, 0.005),
                color=rgb("#a69e8c"),
                finish=Finish(
                    roughness=0.95,
                    texture=tex("aggregate_coarse.png"),
                    uv_scale=(2.5, 2.5),
                    uv_offset=(0, 0),
                ),
            )
        )
for ix in range(3):
    scene.add(
        rss.slab(
            name=f"terrace_seam_{ix}",
            half=(0.0975, 0.0475, 0.003),
            at=(-0.1975 + ix * 0.2, 0.6975, 0.005),
            color=rgb("#a69e8c"),
            finish=Finish(roughness=0.95, texture=tex("aggregate_coarse.png"), uv_scale=(2.5, 2.5)),
        )
    )
scene.add(
    rss.slab(
        name="terrace_foundation",
        half=(1.0, 2.7, 0.3),
        at=(-0.7, 0, -0.30),
        color=rgb("#9a9485"),
        finish=Finish(roughness=1),
    )
)
for side in [-1, 1]:
    y = side * 1.675
    scene.add(
        rss.slab(
            name=f"wall_extension_{side}",
            half=(0.015, 1.025, 1.3),
            at=(0.306, y, 1.308),
            color=rgb("#e1dfdc"),
            finish=Finish(roughness=0.98, texture=tex("plaster_scan.png"), uv_scale=(3, 3)),
        )
    )
    scene.add(
        rss.slab(
            name=f"wall_band_extension_{side}",
            half=(0.0005, 1.025, 0.0125),
            at=(0.2905, y, 0.1205),
            color=rgb("#c8c9c6"),
            finish=Finish(roughness=0.98, texture=tex("band_scan.png"), uv_scale=(3, 3)),
        )
    )
    # Clone the original board dimensions so its texture axes and scale agree.
    for segment in (1, 2):
        scene.add(
            rss.slab(
                name=f"skirting_extension_{side}_{segment}",
                half=(0.006, 0.65, 0.05),
                at=(0.285, side * segment * 1.30, 0.058),
                color=rgb("#a06535"),
                finish=Finish(roughness=0.4, texture=tex("wood_straight.png"), uv_scale=(0.2, 10)),
            )
        )

scene.add(
    rss.slab(
        name="wall_upper",
        half=(0.015, 0.65, 0.85),
        at=(0.306, 0, 1.758),
        color=rgb("#e1dfdc"),
        finish=Finish(roughness=0.98, texture=tex("plaster_scan.png"), uv_scale=(3, 3)),
    )
)
handle = scene.add(
    rss.body(
        name="handle",
        asset=rss.AssetSpec(
            path=str(HERE / "assets/handle.obj"),
            collision_paths=tuple(str(p) for p in sorted((HERE / "assets").glob("*collision*.obj"))),
        ),
        at=tuple(START),
        quat=tuple(QUAT),
        mass=0.0541022,
        friction=1.0,
        color=rgb("#363a42"),
        finish=Finish(roughness=0.48, texture=tex("plastic_dusty.png"), uv_scale=(13, 6)),
    )
)
p = LAYOUT["brush"]
brush = scene.add(
    rss.brush(
        name="brush",
        at=(*START[:2], 0.008),
        handle="handle",
        material="nylon",
        rows=16,
        columns=20,
        sole=p["sole"],
        bristle_length=0.03,
        bristle_radius=0.0006,
        bristle_segments=4,
        flare=0.25,
        color=rgb(p["color"]),
        handle_color=rgb(p["plastic"]),
    )
)
scene.add(rss.RegionSpec(name="patch", frame="", xy_ranges=PATCH, z_range=(0.008, 0.012)))
scene.add(rss.CameraSpec(name="photo", **CAMERA))
scene.add(
    rss.CameraSpec(
        name="wide",
        width=1280,
        height=960,
        eye=(-0.35, 0.10, 1.60),
        look_at=(0.66, -0.065, 0.91),
        fov_y=25,
        near=0.01,
        far=8,
    )
)
from workshop_grime import GrimeSpec

grime = scene.add(GrimeSpec())
with paver_edges(), oriented_brush(YAW, START[2]):
    sim = scene.build(control_mode="joint_position")
grime.entity.source = brush.entity
from workshop_rest import restore_brush

restore_brush(sim, brush, HERE / "assets/settled.npz")

from workshop_motion import direct_routine


def routine(sim):
    yield from direct_routine(sim, START, ORIGIN, DOWN, LO, HI)
