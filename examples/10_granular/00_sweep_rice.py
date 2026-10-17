# SPDX-License-Identifier: Apache-2.0
"""Scoop rice into the tray

A heap of rice under the right hand, a paddle beside it and a tray of four slabs: drag the hand down onto the paddle, close on it, and push the rice into the tray.

The rice is 27,648 particles on Newton's implicit MPM, stepped once per 60 Hz
control step beside the YAM (``rss.GRANULAR_60HZ``: 16.6 ms per control step on
an RTX PRO 6000, graph-captured, measured with this script). The paddle is
12 mm thick because a tool thinner than two 4 mm voxels keeps rice inside it as
it pushes. The panel reads the share of the rice inside the tray's box.
"""

import numpy as np

import robosimstudio as rss
from robosimstudio_materials.mpm.measures import fill_fraction


def main():
    cell = "bench_075"
    heap_at = (0.0, -0.31, 0.0)  # under the right hand's home tool centre, in the bench's frame [m]
    tray_at = (0.0, -0.09, 0.0)  # 220 mm toward the middle of the bench
    tray_inside = 0.16  # [m], square; walls 8 mm thick (two voxels) and 30 mm high
    paddle_at = (-0.13, -0.31, 0.006)  # lying flat beside the heap, 12 mm thick
    wall, height = 0.004, 0.015  # half-thickness and half-height of a tray wall [m]

    scene = rss.Scene(cell=cell, name="scoop_rice", physics=rss.GRANULAR_60HZ)
    scene.add(rss.robot("yam_bimanual"))
    rice = scene.add(rss.heap(name="rice", size=(0.12, 0.12, 0.03), at=heap_at, material="rice"))
    scene.add(rss.box(name="paddle", half=(0.04, 0.06, 0.006), at=paddle_at, color=(0.25, 0.25, 0.28)))
    half = tray_inside / 2 + wall
    for name, (dx, dy, hx, hy) in {
        "wall_near": (-half, 0.0, wall, half),
        "wall_far": (half, 0.0, wall, half),
        "wall_right": (0.0, -half, half, wall),
        "wall_left": (0.0, half, half, wall),
    }.items():
        scene.add(rss.slab(name=name, half=(hx, hy, height), at=(tray_at[0] + dx, tray_at[1] + dy, height)))
    sim = scene.build(control_mode="joint_position")

    # The tray's inside as a world box, from the bench's surface frame.
    origin = np.asarray(rss.get_cell(cell).surface_frame[:3])
    lo = origin + np.array((tray_at[0] - tray_inside / 2, tray_at[1] - tray_inside / 2, 0.0))
    hi = origin + np.array((tray_at[0] + tray_inside / 2, tray_at[1] + tray_inside / 2, 2 * height))

    def page(studio):
        fill = studio.gui.add_number("rice in the tray (%)", 0.0, disabled=True)
        studio.each_step(lambda: setattr(fill, "value", round(100.0 * fill_fraction(rice.particle_q, lo, hi), 1)))

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
