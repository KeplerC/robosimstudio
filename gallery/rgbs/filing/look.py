"""Drawn mat and measured slat spacing; no reference-image textures."""

from dataclasses import replace

import numpy as np

from robosimstudio.core.render.finish import MATTE
from robosimstudio.core.render.room import (
    LAB_FLOOR,
    LAB_GLASS_FINISH,
    LAB_GRID_MAT,
    LAB_MAT_GREY,
    LAB_OAK,
    LAB_OAK_COLOR,
    LAB_WALL,
    _cylinder_mesh,
    house_room,
)
from robosimstudio.core.render.scenery import register_scenery


@register_scenery("filing_room")
def filing_room(kit):
    house_room(
        kit,
        walls=("+y", "-y"),
        floor_half=3.0,
        setback=2.25,
        wall_span=3.0,
        floor_lift=0.0005,
        floor_color=LAB_FLOOR,
        wall_color=LAB_WALL,
        floor_finish=MATTE,
        wall_finish=MATTE,
        prefix="filing",
    )
    kit.box("mat", (0.6, 0, 0.7505), (0.2975, 0.65, 0.00025), color=LAB_MAT_GREY, finish=LAB_GRID_MAT)
    kit.box("ledge", (0.9175, 0, 0.405), (0.02, 0.75, 0.405), color=LAB_OAK_COLOR, finish=LAB_OAK)
    kit.box(
        "glass",
        (0.9875, 0, 0.95),
        (0.004, 1.55, 0.95),
        color=(0.70, 0.74, 0.76),
        finish=replace(LAB_GLASS_FINISH, opacity=0.12),
    )
    for i, y in enumerate(np.arange(-1.47, 1.48, 0.14)):
        kit.box(f"oak_slat_{i}", (0.9675, float(y), 1.0), (0.01, 0.030, 1.0), color=LAB_OAK_COLOR, finish=LAB_OAK)
    kit.box("lounge_floor", (2.5, 0, 0.005), (1.5, 2.5, 0.005), color=(0.59, 0.57, 0.53), finish=MATTE)
    kit.box("lounge_wall", (3.8, 0, 1.4), (0.02, 2.5, 1.4), color=(0.56, 0.55, 0.51), finish=MATTE)
    pts, faces = _cylinder_mesh((1.7, -0.48, 0.735), 0.34, 0.018, segments=64)
    kit.mesh("pale_round_table", pts, faces, color=(0.90, 0.87, 0.79), finish=MATTE)
    kit.box("table_pedestal", (1.7, -0.48, 0.36), (0.06, 0.06, 0.36), color=(0.24, 0.23, 0.21), finish=MATTE)
    for i, y in enumerate((0.25, -1.45)):
        kit.box(f"chair_back_{i}", (2.5, y, 0.68), (0.05, 0.24, 0.28), color=(0.14, 0.13, 0.11), finish=MATTE)
        kit.box(f"chair_seat_{i}", (2.3, y, 0.43), (0.23, 0.24, 0.03), color=(0.14, 0.13, 0.11), finish=MATTE)
    if hasattr(kit, "stage"):
        from pxr import Gf, UsdGeom, UsdLux

        xf = UsdGeom.Xform.Define(kit.stage, "/root/FilingLoungeLight")
        xf.AddTranslateOp().Set(Gf.Vec3d(2.1, -0.3, 2.4))
        lamp = UsdLux.RectLight.Define(kit.stage, "/root/FilingLoungeLight/light")
        lamp.GetWidthAttr().Set(1.6)
        lamp.GetHeightAttr().Set(1.4)
        lamp.GetColorAttr().Set(Gf.Vec3f(1.0, 0.97, 0.92))
        lamp.GetIntensityAttr().Set(1800.0)
    # Simplified drawn overhead camera cage, visible only near photo's top edge.
    for y in (-0.13, 0.13):
        kit.box(f"cage_rail_{y}", (0.91, y, 1.25), (0.10, 0.004, 0.004), color=(0.55, 0.57, 0.59), finish=MATTE)
    for x in (0.81, 1.01):
        kit.box(f"cage_cross_{x}", (x, 0, 1.25), (0.004, 0.13, 0.004), color=(0.55, 0.57, 0.59), finish=MATTE)
    kit.box("overhead_camera", (0.91, 0.05, 1.24), (0.035, 0.025, 0.025), color=(0.09, 0.10, 0.11), finish=MATTE)
    kit.box("camera_front", (0.874, 0.05, 1.24), (0.002, 0.02, 0.018), color=(0.82, 0.84, 0.84), finish=MATTE)
    for i, y in enumerate((0.43, -0.48)):
        kit.box(f"rig_post_{i}", (0.920, y, 1.45), (0.012, 0.012, 0.64), color=(0.84, 0.85, 0.86), finish=MATTE)
    kit.box("apron", (0.2965, 0, 0.725), (0.006, 0.65, 0.025), color=(0.035, 0.035, 0.038), finish=MATTE)
