"""Alocasia watering: MPM water, catalog YAM, elastic articulated plant."""

import os
import sys

sys.dont_write_bytecode = True
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["ROBOSIMSTUDIO_CACHE"] = str(ROOT / "run/cache")

import numpy as np
from cup import cup
from plant import add_plant
from plant_assets import EYE, FOV, LOOK, lights
from room import add_room

import robosimstudio as rss
from robosimstudio.core.render import Finish
from robosimstudio.core.render.spec import RenderSpec
from robosimstudio.robots.registry import get_embodiment

ORIGIN = np.array((0.6, 0, 0.75))
CUP_AT = np.array((-0.22, 0.30, 0.002))
PARK_Q = {
    "left": (0.0, 1.1, 1.5, 0, 0, 0),
    "right": (-1.5, 0.5, 1.5, 0, 0, 0),
}
yam = get_embodiment("yam")
yam = rss.replace(yam, arms=tuple(rss.replace(a, home_q=PARK_Q[a.name]) for a in yam.arms))
from robosimstudio.scene.registry import register_cell

register_cell(rss.replace(rss.get_cell("bench_075"), name="plant_dining", statics=()))
scene = rss.Scene(
    cell="plant_dining",
    name="plant",
    physics=rss.LIQUID_120HZ,
    render=RenderSpec(environment="lab", scenery=lights, key_intensity=0.8, ambient=(0.4, 0.44, 0.5), quality="final"),
)
# Both bases lie beyond the photo frame; the right arm uses a tabletop riser.
scene.add(
    rss.robot(
        yam,
        at=(-0.48, -0.60, 0),
        arms_at={"left": (-0.48, -0.80, 0.022), "right": (-0.22, 0.55, 0.302, 0, 0, -0.5, 0.866025403784)},
    )
)
scene.add(
    rss.CameraSpec(
        name="photo", width=1920, height=1280, fov_y=FOV, eye=tuple(EYE), look_at=tuple(LOOK), near=0.01, far=6
    )
)
scene.add(
    rss.CameraSpec(
        name="wide",
        width=1920,
        height=1080,
        fov_y=44,
        eye=(-1.15, -0.10, 1.55),
        look_at=(0.43, -0.15, 1.23),
        near=0.01,
        far=8,
    )
)
add_room(scene)
scene.add(rss.slab(name="room_floor", half=(3, 3, 0.015), at=(0, 0, -0.755), color=(0.38, 0.35, 0.31)))
scene.add(
    rss.slab(name="wall_below_photo", half=(0.008, 1.25, 0.2875), at=(0.310, 0, -0.4625), color=(0.72, 0.75, 0.79))
)
scene.add(rss.slab(name="wall_return", half=(1.0, 0.008, 1.1125), at=(-0.698, 1.25, 0.3625), color=(0.72, 0.75, 0.79)))
from plant_layout import LAYOUT

scene.add(
    rss.slab(name="wall_opposite", half=(1.0, 0.008, 1.1125), at=(-0.698, -1.25, 0.3625), color=(0.72, 0.75, 0.79))
)
for row in LAYOUT:
    if row["name"] in ("lamp_left", "lamp_right"):
        x, y, z = row["at"]
        bottom = z + 0.105
        top = 1.47
        scene.add(
            rss.post(
                name=row["name"] + "_cord",
                radius=0.0012,
                half_height=(top - bottom) / 2,
                at=(x, y, (top + bottom) / 2),
                color=(0.15, 0.15, 0.14),
            )
        )
from containment import filled_pot

scene.add(filled_pot())
plant_layout = add_plant(scene)
scene.add(cup(CUP_AT))
water = scene.add(
    rss.liquid(
        name="water",
        material="water",
        shape="cylinder",
        size=(0.038, 0.038, 0.045),
        particle_spacing=0.0015,
        surface_voxel=0.002,
        at=(*CUP_AT[:2], CUP_AT[2] + 0.026),
        containers=("cup_surface", "pot_liquid_boundary"),
        container_margin=0.010,
        settle_frames=600,
        render="surface",
        color=(0.50, 0.70, 0.82),
        finish=Finish(roughness=0.06, opacity=0.78, ior=1.333),
    )
)

scene.add(rss.slab(name="left_pedestal", half=(0.09, 0.09, 0.375), at=(-0.48, -0.80, -0.375), color=(0.18, 0.18, 0.19)))
scene.add(
    rss.slab(name="right_mounting_plate", half=(0.052, 0.025, 0.140), at=(-0.22, 0.57, 0.141), color=(0.18, 0.18, 0.19))
)
from audit import Audit
from watering import routine as water_routine


def routine(sim, session):
    audit = Audit(sim, plant_layout, session)
    try:
        for step in water_routine(sim, session):
            audit.sample()
            yield step
        audit.sample()
        if audit.pot_through.any():
            raise RuntimeError("Water crossed the pot wall below its rim")
    finally:
        audit.sample()
        audit.save()
