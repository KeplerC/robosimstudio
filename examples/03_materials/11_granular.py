# SPDX-License-Identifier: Apache-2.0
"""Granular: sweep rice

A heap of rice under the right hand, a pusher standing on its edge beyond it and a three-walled tray on the near side: turn the hand to point down, drag it onto the pusher, close on it, and sweep the heap into the tray at a walking pace.

The rice is 16,000 particles on Newton's implicit MPM, stepped once per 60 Hz
control step beside the YAM (``rss.GRANULAR_60HZ``: 16.5 ms per control step on
an RTX PRO 6000, GPU 3, graph-captured, measured by ``tests/test_sweep_rice``
on this scene). The pusher is 20 mm thick and 40 mm tall so the jaws close on
it from above, and the heap names it in ``containers`` so the grid sees it one
voxel fatter than the hand does; without that the bottom voxel of rice passes
under it. The tray's walls are five voxels thick and its open side faces the
heap, so the sweep ends inside it. Swept at 0.15 m/s as the test sweeps it,
93% of the rice is in the tray and the rest is on the bench, but for a grain
the squeeze shot off the bench's edge in two sweeps of thirty (the others lost
none); a tool faster than ~0.36 m/s outruns the grid. What reaches the tray
is packed, not heaped: 88% of the grains end in a strip 32 mm deep along the
pusher's face, half of them within 1.3 mm of the bench, and a tenth runs
round the pusher's ends (``sweep_where.py``, GPU 0). The panel reads the
share of the rice inside the tray.
"""

import numpy as np

import robosimstudio as rss
from robosimstudio_materials.mpm.measures import fill_fraction


def main():
    cell = "bench_075"
    heap_at = (0.0, -0.31, 0.0)  # under the right hand's home tool centre, in the bench's frame [m]
    heap_size = (0.10, 0.10, 0.025)  # [m]; 16,000 particles at the default 2.5 mm spacing
    pusher_at = (0.0, -0.45, 0.02)  # standing on its edge beyond the heap: 160 mm wide, 20 mm thick, 40 mm tall
    tray_at = (0.0, -0.12)  # the centre of the tray's inside, 190 mm toward the middle of the bench
    tray_inside = (0.28, 0.20)  # [m], x by y; the side toward the heap is open
    wall, height = 0.01, 0.015  # half-thickness (20 mm, five voxels) and half-height (30 mm) of a wall [m]

    scene = rss.Scene(cell=cell, name="sweep_rice", physics=rss.GRANULAR_60HZ)
    scene.add(rss.robot("yam_bimanual"))
    rice = scene.add(
        rss.heap(
            name="rice", size=heap_size, at=heap_at, material="rice", containers=("pusher",), container_margin=0.004
        )
    )
    scene.add(rss.box(name="pusher", half=(0.08, 0.01, 0.02), at=pusher_at, friction=0.3, color=(0.15, 0.15, 0.17)))
    hx, hy = tray_inside[0] / 2, tray_inside[1] / 2
    for name, (dx, dy, half_x, half_y) in {
        "wall_back": (0.0, hy + wall, hx + 2 * wall, wall),
        "wall_left": (-(hx + wall), wall, wall, hy + wall),
        "wall_right": (hx + wall, wall, wall, hy + wall),
    }.items():
        scene.add(rss.slab(name=name, half=(half_x, half_y, height), at=(tray_at[0] + dx, tray_at[1] + dy, height)))
    sim = scene.build(control_mode="joint_position")

    # The tray's inside as a world box, from the bench's surface frame; 5 mm under the top, where the bottom layer rests.
    origin = np.asarray(rss.get_cell(cell).surface_frame[:3])
    lo = origin + np.array((tray_at[0] - hx, tray_at[1] - hy, -0.005))
    hi = origin + np.array((tray_at[0] + hx, tray_at[1] + hy, 2 * height))

    def page(studio):
        fill = studio.gui.add_number("rice in the tray (%)", 0.0, disabled=True)
        studio.each_step(lambda: setattr(fill, "value", round(100.0 * fill_fraction(rice.particle_q, lo, hi), 1)))

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
