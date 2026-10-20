# SPDX-License-Identifier: Apache-2.0
"""Pour water, scored

The same cup, water and glass as a task document, with a goal on the water's level in the glass: pour as in ``03_materials/14_water``, and the goal turns true at the 25 mm line.

The water is a ``Water.from_spec`` record the document places under its own name, the cup a
``Body.from_spec`` object, the glass's walls and its painted line ``Body.from_spec`` fixtures, and
the physics ``rss.WATER_60HZ`` under a task-language name. The region is the glass's inside, from
the bench top to its 55 mm rim, and ``fill_level`` reads the height the water stands there. The
hands start at the YAM's home, pointing ahead. ``tests/test_pour_water`` drives the same pour
through the right arm's panel and ``evaluate`` says success, the water 29.5 mm deep in both of two
drives, 99.3 and 99.7% of it in the glass; 25 ms per control step on an RTX PRO 6000, GPUs 1 and 3.
The ``[genesis]`` extra lives in a virtual environment of its own (``examples/03_materials/README.md``).
"""

import robosimstudio as rss
import robosimstudio_bench as bench
from robosimstudio.assetlib import register_asset
from robosimstudio.domains.rigid import Body
from robosimstudio_materials.liquid import cup
from robosimstudio_materials.water import Water

TASK = """
(define (problem pour_water)
  (:language pour the water into the glass up to the line)
  (:staging (:scene bench_075) (:embodiment yam_bimanual) (:physics water_60hz@v1))
  (:fixtures glass_near - glass_near glass_far - glass_far glass_right - glass_right glass_left - glass_left
             line - fill_line)
  (:objects cup - water_cup water - cup_of_water)
  (:regions (glass (:ranges ((0.09 -0.16 0.15 -0.10))) (:z-range 0.0 0.055)))
  (:goal (>= (fill_level water glass) 0.025)))
"""


def main() -> None:
    cup_at, x, y = (0.12, -0.31, 0.0), 0.12, -0.13  # the cup, and the glass's centre, in the bench's frame [m]
    side, thick, tall, line = 0.06, 0.006, 0.055, 0.025  # the glass: inside, walls, height, fill line [m]

    bench.register_arrangement("water_60hz@v1", rss.WATER_60HZ)
    mug = cup(name="cup", inner=(0.05, 0.05, 0.07), wall=0.008, at=cup_at, color=(0.85, 0.85, 0.88))
    register_asset(Body.from_spec(mug, name="water_cup"))
    register_asset(Water.from_spec(rss.water(name="water", at=(*cup_at[:2], 0.009)), name="cup_of_water"))
    h = (side + thick) / 2
    for name, at, half in (
        ("glass_near", (x - h, y), (thick / 2, h + thick / 2)),
        ("glass_far", (x + h, y), (thick / 2, h + thick / 2)),
        ("glass_right", (x, y - h), (side / 2, thick / 2)),
        ("glass_left", (x, y + h), (side / 2, thick / 2)),
    ):
        wall = rss.slab(name=name, half=(*half, tall / 2), at=(*at, tall / 2), color=(0.72, 0.86, 0.92))
        register_asset(Body.from_spec(wall, name=name))
    paint = rss.slab(
        name="line",
        half=(0.0005, h, 0.001),
        at=(x + h + thick / 2 + 0.0005, y, line),
        collide=False,
        color=(0.85, 0.15, 0.1),
    )
    register_asset(Body.from_spec(paint, name="fill_line"))

    env = bench.Env(bench.task(TASK))
    env.reset()
    rss.view(env)


if __name__ == "__main__":
    main()
