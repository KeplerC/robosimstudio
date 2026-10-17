# SPDX-License-Identifier: Apache-2.0
"""Scoop rice, scored

The same rice, paddle and tray as a task document, with a goal on the share of rice in the tray: scoop, and the goal turns true at half.

The heap is a ``Heap.from_spec`` record the document places under its own name,
the tray walls are ``Body.from_spec`` fixtures, and the physics is the granular
arrangement under a task-language name. ``(:z-range 0 0.03)`` is the tray's
depth; a region's default window is 150 mm, which would count rice carried
over the tray in the paddle. 16.7 ms per control step on an RTX PRO 6000,
graph-captured.
"""

import robosimstudio as rss
import robosimstudio_bench as bench
from robosimstudio.assetlib import register_asset
from robosimstudio.domains.rigid import Body
from robosimstudio_materials.granular import GRANULAR_60HZ, Heap

TASK = """
(define (problem scoop_rice)
  (:language scoop the rice into the tray)
  (:staging (:scene bench_075) (:embodiment yam_bimanual) (:physics granular_60hz@v1))
  (:fixtures wall_near - wall_near wall_far - wall_far wall_right - wall_right wall_left - wall_left)
  (:objects rice - rice_heap paddle - paddle)
  (:regions (tray (:ranges ((-0.08 -0.17 0.08 -0.01))) (:z-range 0.0 0.03)))
  (:goal (>= (fill_fraction rice tray) 0.5)))
"""


def main() -> None:
    tray_at = (0.0, -0.09)  # the tray's centre, in the bench's frame [m]; the region above is its inside
    half, wall, height = 0.084, 0.004, 0.015  # wall centre offset, half-thickness, half-height [m]

    bench.register_arrangement("granular_60hz@v1", GRANULAR_60HZ)
    register_asset(
        Heap.from_spec(rss.heap(name="rice", size=(0.12, 0.12, 0.03), at=(0.0, -0.31, 0.0)), name="rice_heap")
    )
    register_asset(
        Body.from_spec(rss.box(name="paddle", half=(0.04, 0.06, 0.006), at=(-0.13, -0.31, 0.006)), name="paddle")
    )
    for name, (dx, dy, hx, hy) in {
        "wall_near": (-half, 0.0, wall, half),
        "wall_far": (half, 0.0, wall, half),
        "wall_right": (0.0, -half, half, wall),
        "wall_left": (0.0, half, half, wall),
    }.items():
        slab = rss.slab(name=name, half=(hx, hy, height), at=(tray_at[0] + dx, tray_at[1] + dy, height))
        register_asset(Body.from_spec(slab, name=name))

    env = bench.Env(bench.task(TASK))
    env.reset()
    rss.view(env)


if __name__ == "__main__":
    main()
