# SPDX-License-Identifier: Apache-2.0
"""Sweep rice, scored

The same rice, pusher and tray as a task document, with a goal on the share of rice in the tray: sweep, and the goal turns true at half.

The heap is a ``Heap.from_spec`` record the document places under its own name,
the tray walls are ``Body.from_spec`` fixtures, and the physics is the granular
arrangement under a task-language name. The region is the tray's inside;
``(:z-range -0.005 0.03)`` is the walls' height, starting 5 mm under the bench
top because the bottom layer of a pushed heap rests at the top to within a
rounding, and a region's default window is 150 mm, which would count rice
carried over the tray. ``tests/test_sweep_rice`` drives the hand through the
sweep and the goal turns true, at 93%; 16.5 ms per control step on an RTX PRO
6000, GPU 3, graph-captured.
"""

import robosimstudio as rss
import robosimstudio_bench as bench
from robosimstudio.assetlib import register_asset
from robosimstudio.domains.rigid import Body
from robosimstudio_materials.granular import GRANULAR_60HZ, Heap

TASK = """
(define (problem sweep_rice)
  (:language sweep the rice into the tray)
  (:staging (:scene bench_075) (:embodiment yam_bimanual) (:physics granular_60hz@v1))
  (:fixtures wall_back - wall_back wall_left - wall_left wall_right - wall_right)
  (:objects rice - rice_heap pusher - pusher)
  (:regions (tray (:ranges ((-0.14 -0.22 0.14 -0.02))) (:z-range -0.005 0.03)))
  (:goal (>= (fill_fraction rice tray) 0.5)))
"""


def main() -> None:
    tray_at = (0.0, -0.12)  # the centre of the tray's inside, in the bench's frame [m]; the region above is that inside
    hx, hy, wall, height = 0.14, 0.10, 0.01, 0.015  # inside half-extents, wall half-thickness and half-height [m]

    bench.register_arrangement("granular_60hz@v1", GRANULAR_60HZ)
    heap = rss.heap(
        name="rice", size=(0.10, 0.10, 0.025), at=(0.0, -0.31, 0.0), containers=("pusher",), container_margin=0.004
    )
    register_asset(Heap.from_spec(heap, name="rice_heap"))
    pusher = rss.box(
        name="pusher", half=(0.08, 0.01, 0.02), at=(0.0, -0.45, 0.02), friction=0.3, color=(0.15, 0.15, 0.17)
    )
    register_asset(Body.from_spec(pusher, name="pusher"))
    for name, (dx, dy, half_x, half_y) in {
        "wall_back": (0.0, hy + wall, hx + 2 * wall, wall),
        "wall_left": (-(hx + wall), wall, wall, hy + wall),
        "wall_right": (hx + wall, wall, wall, hy + wall),
    }.items():
        slab = rss.slab(name=name, half=(half_x, half_y, height), at=(tray_at[0] + dx, tray_at[1] + dy, height))
        register_asset(Body.from_spec(slab, name=name))

    env = bench.Env(bench.task(TASK))
    env.reset()
    rss.view(env)


if __name__ == "__main__":
    main()
