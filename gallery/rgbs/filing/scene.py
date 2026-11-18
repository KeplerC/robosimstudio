"""Task 3: Filing. All robot and workpiece motion is contact dynamics."""

import json
from dataclasses import replace

import look  # noqa: F401 -- registers the room
from filing_model import WOOD, WOOD_FINISH, FilingSpec

import robosimstudio as rss
from robosimstudio.core.render.finish import Finish
from robosimstudio.core.render.room import LAB
from robosimstudio.robots.embodiments.yam import YAM

physics = replace(rss.MUJOCO_500HZ, substeps=20, collision=replace(rss.MUJOCO_500HZ.collision, rigid_gap=0.0002))
scene = rss.Scene(
    cell="bench_075",
    name="filing",
    physics=physics,
    render=replace(LAB, scenery="filing_room", key_intensity=2.2, ambient=(1.1, 1.1, 1.1)),
)


def robot_look(builder, shape_range):
    for i in range(*shape_range):
        label = builder.shape_label[i].split("/")[-1]
        if "link_2" in label or "link_3" in label:
            builder.shape_color[i] = (0.83, 0.84, 0.82)
    return 1


home = json.load(open("assets/home_q.json"))
robot_spec = replace(
    YAM,
    arms=tuple(replace(a, home_q=tuple(home[a.name]), home_gripper_q=(0.0152,)) for a in YAM.arms),
    color=(0.055, 0.06, 0.065),
    finish=Finish(roughness=0.65, metallic=0.0),
    appearance=robot_look,
)
scene.add(rss.robot(robot_spec, arms_at={"left": (-0.35, 0.3008, 0.0), "right": (-0.333, -0.3008, 0.029)}))
strip = scene.add(
    rss.box(
        name="strip",
        half=(0.0075, 0.040, 0.006),
        at=(-0.16, 0, 0.0063),
        mass=0.0072,
        friction=0.65,
        ke=60000.0,
        kd=500.0,
        color=WOOD,
        finish=WOOD_FINISH,
    )
)
scene.add(
    rss.slab(
        name="right_base_riser", half=(0.085, 0.07, 0.0145), at=(-0.333, -0.3008, 0.0145), color=(0.13, 0.14, 0.15)
    )
)
# Stand, clamped to the workbench; geometry remains collidable.
scene.add(rss.slab(name="mount_table", half=(0.15, 0.65, 0.025), at=(-0.45, 0, -0.025), color=(0.18, 0.19, 0.19)))
scene.add(
    rss.slab(
        name="stand_base", half=(0.05, 0.044, 0.003), at=(0.15, 0.04, 0.003), color=(0.025, 0.027, 0.028), friction=0.65
    )
)
scene.add(rss.slab(name="stand_post", half=(0.014, 0.014, 0.15), at=(0.18, 0.04, 0.156), color=(0.018, 0.02, 0.023)))
scene.add(rss.slab(name="tool_clamp", half=(0.008, 0.027, 0.012), at=(0.100, 0, 0.219), color=(0.67, 0.68, 0.68)))
scene.add(rss.slab(name="tool_bracket", half=(0.065, 0.025, 0.006), at=(0.13, 0.015, 0.240), color=(0.72, 0.73, 0.72)))
# The grey rotary tool hangs nose down; its own motor powers only the drum.
scene.add(
    rss.post(name="motor_housing", radius=0.022, half_height=0.034, at=(0.080, 0, 0.219), color=(0.25, 0.27, 0.28))
)
scene.add(rss.post(name="tool_nose", radius=0.014, half_height=0.016, at=(0.080, 0, 0.176), color=(0.12, 0.135, 0.14)))
scene.add(rss.post(name="collet", radius=0.007, half_height=0.006, at=(0.080, 0, 0.163), color=(0.1, 0.11, 0.12)))
scene.add(rss.post(name="mandrel", radius=0.002, half_height=0.004, at=(0.080, 0, 0.156), color=(0.55, 0.56, 0.58)))
for i in range(5):
    scene.add(
        rss.slab(
            name=f"vent_{i}",
            half=(0.0005, 0.0015, 0.008),
            at=(0.058, -0.012 + i * 0.006, 0.208),
            color=(0.025, 0.03, 0.035),
        )
    )
scene.add(rss.slab(name="tool_label", half=(0.0005, 0.013, 0.012), at=(0.057, 0, 0.237), color=(0.72, 0.73, 0.70)))
scene.add(rss.slab(name="label_band", half=(0.0005, 0.013, 0.002), at=(0.0564, 0, 0.245), color=(0.75, 0.20, 0.04)))
apparatus = scene.add(FilingSpec())
scene.add(
    rss.CameraSpec(
        name="photo",
        width=960,
        height=540,
        fov_y=30.75,
        fx=981.805,
        fy=981.805,
        cx=480,
        cy=270,
        eye=(-0.4836, 0.0052, 1.3706),
        look_at=(0.457, 0.0026, 1.0311),
        up=(0.3395, 0.0136, 0.9405),
    )
)
scene.add(
    rss.CameraSpec(name="wide", width=1920, height=1440, fov_y=55, eye=(-0.05, -0.12, 1.32), look_at=(0.57, 0, 0.85))
)

from motion import routine

routines = {"routine": routine}
