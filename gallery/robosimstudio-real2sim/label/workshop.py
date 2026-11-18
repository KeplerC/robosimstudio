"""Geometry and local procedural materials for a small packing workshop."""

import math
from pathlib import Path

import robosimstudio as rss
from robosimstudio.core.render.finish import Finish

A = Path(__file__).resolve().parent / "assets"
WOOD = Finish(
    roughness=0.9,
    texture=str(A / "wood_worn_r3.png"),
    roughness_texture=str(A / "wood_roughness_r3.png"),
    uv_scale=(1.0, 1.0),
)
STEEL = Finish(roughness=0.32, metallic=0.85)


def add_workshop(scene):
    def slab(n, half, at, color, finish=None):
        scene.add(rss.slab(name=n, half=half, at=at, color=color, finish=finish))

    def rod(n, r, h, at, c, quat=(0, 0, 0, 1), finish=None):
        scene.add(rss.post(name=n, radius=r, half_height=h, at=(*at, *quat), color=c, finish=finish))

    # Continuous extension behind the original 600 x 800 mm task surface.
    slab("bench_rear", (0.25, 0.4, 0.03), (-0.55, 0, -0.03), (1, 1, 1), WOOD)
    for side in (-1, 1):
        slab(f"bench_wing_{side}", (0.55, 0.32, 0.03), (-0.25, side * 0.72, -0.03), (1, 1, 1), WOOD)
    slab("bench_front_apron", (0.025, 1.04, 0.07), (0.27, 0, -0.13), (0.7, 0.5, 0.3), WOOD)
    for x in (-0.73, 0.25):
        for y in (-0.94, 0.94):
            slab(f"outer_leg_{x}_{y}", (0.04, 0.04, 0.334), (x, y, -0.394), (0.62, 0.44, 0.26), WOOD)
    slab("floor", (3, 3, 0.015), (0, 0, -0.746), (0.36, 0.38, 0.36))
    slab("back_wall", (0.04, 2, 1.5), (-1.0, 0, 0.772), (0.65, 0.70, 0.65))
    slab("left_wall", (2, 0.04, 1.5), (0.5, 1.65, 0.772), (0.74, 0.73, 0.66))
    slab("skirting", (0.02, 2, 0.055), (-0.95, 0, -0.66), (0.23, 0.28, 0.26))
    slab(
        "pegboard",
        (0.015, 0.69, 0.32),
        (-0.92, 0, 0.50),
        (1, 1, 1),
        Finish(roughness=0.85, texture=str(A / "pegboard_r3.png"), uv_scale=(2.5, 2.5)),
    )
    for yy in (-0.71, 0.71):
        slab(f"board_edge_{yy}", (0.025, 0.015, 0.34), (-0.91, yy, 0.5), (0.25, 0.17, 0.09), WOOD)
    for zz in (0.16, 0.84):
        slab(f"board_rail_{zz}", (0.025, 0.71, 0.012), (-0.91, 0, zz), (0.25, 0.17, 0.09), WOOD)
    for k, y in enumerate((-0.54, -0.44, -0.34, -0.24)):
        l = 0.18 - k * 0.018
        slab(f"wrench_shaft{k}", (0.006, 0.009, l / 2), (-0.885, y, 0.58 - l / 2), (0.58, 0.61, 0.62), STEEL)
        # Open jaw, assembled in steel on the board.
        slab(f"wrench_neck{k}", (0.006, 0.021, 0.008), (-0.885, y, 0.59), (0.58, 0.61, 0.62), STEEL)
        for s in (-1, 1):
            slab(f"wrench_jaw{k}_{s}", (0.006, 0.006, 0.018), (-0.885, y + s * 0.018, 0.61), (0.58, 0.61, 0.62), STEEL)
        rod(
            f"wrench_end{k}",
            0.013,
            0.006,
            (-0.885, y, 0.58 - l),
            (0.58, 0.61, 0.62),
            (0, 0.7071068, 0, 0.7071068),
            STEEL,
        )
    for k, y in enumerate((-0.08, 0.015, 0.11)):
        col = [(0.46, 0.065, 0.035), (0.65, 0.37, 0.035), (0.07, 0.17, 0.20)][k]
        rod(f"driver_handle{k}", 0.015, 0.047, (-0.88, y, 0.64), col)
        rod(f"driver_shaft{k}", 0.0035, 0.075, (-0.88, y, 0.518), (0.6, 0.63, 0.65), finish=STEEL)
    slab("hammer_handle", (0.012, 0.016, 0.15), (-0.88, 0.28, 0.49), (0.7, 0.49, 0.25), WOOD)
    slab("hammer_head", (0.022, 0.065, 0.022), (-0.88, 0.28, 0.65), (0.20, 0.21, 0.22), STEEL)
    # A small handsaw: handle and serrated steel blade.
    slab("saw_handle", (0.018, 0.026, 0.055), (-0.88, 0.51, 0.62), (0.55, 0.16, 0.035))
    slab("saw_blade", (0.003, 0.037, 0.14), (-0.875, 0.51, 0.43), (0.55, 0.57, 0.59), STEEL)
    for k in range(11):
        slab(f"saw_tooth{k}", (0.004, 0.006, 0.006), (-0.875, 0.552, 0.31 + k * 0.024), (0.55, 0.57, 0.59), STEEL)
    # Shelves set to the sides, safely outside the arm workspace.
    for k, z in enumerate((0.42, 0.94)):
        slab(f"shelf{k}", (0.19, 0.36, 0.023), (-0.78, 1.15, z), (0.65, 0.47, 0.28), WOOD)
        for j, y in enumerate((0.92, 1.17, 1.41)):
            h = 0.12 + (0.03 if j == 1 else 0)
            slab(
                f"stored_box{k}_{j}",
                (0.12, 0.095, h / 2),
                (-0.79, y, z + 0.023 + h / 2),
                (1.0, 1.0, 1.0),
                Finish(roughness=0.9, texture=str(A / "kraft_r3.png"), uv_scale=(5, 5)),
            )
            slab(f"carton_seal{k}_{j}", (0.121, 0.012, 0.001), (-0.79, y, z + 0.024 + h), (0.69, 0.55, 0.35))
    slab("tape_shelf", (0.17, 0.27, 0.02), (-0.79, -1.08, 0.48), (0.65, 0.47, 0.28), WOOD)
    for j in range(3):
        # Rings made of segments, with a visible cardboard core.
        for k in range(20):
            theta = 2 * math.pi * k / 20
            q = (math.sin(theta / 2), 0, 0, math.cos(theta / 2))
            scene.add(
                rss.slab(
                    name=f"tape_roll{j}_{k}",
                    half=(0.025, 0.0075, 0.009),
                    at=(-0.75, -1.23 + j * 0.15 + 0.044 * math.cos(theta), 0.544 + 0.044 * math.sin(theta), *q),
                    color=(0.66, 0.46, 0.20),
                )
            )
    # Window with frame and pale glazing, no image behind it.
    slab("window_recess", (0.015, 0.37, 0.45), (-0.949, -1.16, 1.12), (0.16, 0.21, 0.22))
    slab("window_glass", (0.012, 0.34, 0.42), (-0.928, -1.16, 1.12), (0.68, 0.82, 0.88), Finish(roughness=0.22))
    for y in (-1.52, -1.16, -0.80):
        slab(f"window_mullion{y}", (0.03, 0.018, 0.45), (-0.90, y, 1.12), (0.83, 0.82, 0.73))
    for z in (0.67, 1.12, 1.57):
        slab(f"window_cross{z}", (0.03, 0.37, 0.018), (-0.90, -1.16, z), (0.83, 0.82, 0.73))
    # A long workshop light, aligned with the renderer's overhead panel.
    slab("light_housing", (0.12, 0.65, 0.035), (-0.18, 0, 1.65), (0.19, 0.23, 0.23), STEEL)
    slab("light_diffuser", (0.11, 0.62, 0.005), (-0.18, 0, 1.61), (0.96, 0.94, 0.85))
