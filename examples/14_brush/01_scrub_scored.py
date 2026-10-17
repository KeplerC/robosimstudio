# SPDX-License-Identifier: Apache-2.0
"""Scrub a patch, scored

The same brush, handle and square as a task document, with a goal on the share of the square the bristle tips have scrubbed: turn the hand to point down, close on the handle, scrub, and the goal turns true at half.

The brush is a ``Brush.from_spec`` record and its handle a ``Body.from_spec``
one, placed by the document under the names the brush's ``handle`` and the
goal use, and the physics is the brush arrangement under a task-language name.
The region is the square, and ``scrubbed`` reads its 5 mm cells off the
lattice the brush keeps on the device; a reset clears it. ``tests/test_scrub_bench``
drives the hand through the same three strokes from the robot's own home and
the goal turns true, at 1.0 (22.3 ms per control step, GPU 3).
"""

import robosimstudio as rss
import robosimstudio_bench as bench
from robosimstudio.assetlib import register_asset
from robosimstudio.domains.rigid import Body
from robosimstudio_materials.brush import BRUSH_60HZ, Brush

TASK = """
(define (problem scrub_patch)
  (:language scrub the marked square with the brush)
  (:staging (:scene bench_075) (:embodiment yam_bimanual) (:physics brush_60hz@v1))
  (:objects handle - brush_handle brush - dish_brush)
  (:regions (patch (:ranges ((-0.025 -0.27 0.075 -0.17))) (:z-range -0.005 0.03)))
  (:goal (>= (scrubbed brush patch) 0.5)))
"""


def main() -> None:
    brush_at = (0.025, -0.31, 0.0)  # the bristle tips' footprint centre, in the bench's frame [m]
    handle_half = (0.04, 0.011, 0.015)  # the block the hand closes on: 22 mm across the jaws [m]

    bench.register_arrangement("brush_60hz@v1", BRUSH_60HZ)
    brush = rss.brush(name="brush", at=brush_at, material="nylon", handle="handle")
    register_asset(Brush.from_spec(brush, name="dish_brush"))
    handle_at = (brush_at[0], brush_at[1], brush.sole_top + handle_half[2])  # standing on the sole
    handle = rss.box(name="handle", half=handle_half, at=handle_at, friction=1.0, color=(0.15, 0.15, 0.17))
    register_asset(Body.from_spec(handle, name="brush_handle"))

    env = bench.Env(bench.task(TASK))
    env.reset()
    rss.view(env)


if __name__ == "__main__":
    main()
