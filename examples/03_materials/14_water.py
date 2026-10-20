# SPDX-License-Identifier: Apache-2.0
"""Water: pour into a glass

A cup of water under the right hand and a glass beside it: drag the hand down until its open jaws straddle the cup, close on it, drag it up to 170 mm over the bench and 125 mm across, then turn the gizmo 120 degrees about the axis pointing away from the robot, and the water runs into the glass past its red line.

The water is Genesis's SPH (``rss.water`` under ``rss.WATER_60HZ``), which keeps its volume where Newton's MPM
water loses it; it runs where the ``[genesis]`` extra is installed, in a virtual environment of its own (this
folder's README). The jaws close on the cup's 66 mm walls, and the water's reaction comes back onto the cup:
carried still, it reads 1.09 N where the water weighs 1.085 N. The wrist stops at 115 degrees: in three drives
99.2 to 99.9% of the 1,728 particles ended in the glass, standing 29.4 to 29.7 mm where their count fills 30.5 to
30.7 mm, 0.1 to 0.8% on the bench (``tests/test_pour_water``'s gesture, GPU 1, RTX PRO 6000). Turned 60 degrees
it keeps the water (at most 0.1% in the glass); carried 100 mm aside, 99.2% lands on the bench. 24 ms per step.
"""

import numpy as np

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment
from robosimstudio_materials.liquid import cup
from robosimstudio_materials.mpm.measures import spill_fraction
from robosimstudio_materials.water import fill_level


def main():
    cell = "bench_075"
    cup_at, glass_at = (0.12, -0.31, 0.0), (0.12, -0.13, 0.0)  # in the bench's frame [m]
    inner, wall = (0.05, 0.05, 0.07), 0.008  # the cup's cavity and its walls [m]
    side, thick, tall, line = 0.06, 0.006, 0.055, 0.025  # the glass: inside, walls, height, fill line [m]
    over_cup = (0.0, 1.723, 0.642, 1.081, 0.0, 0.0)  # right arm [rad]: tool centre 155 mm over the cup, jaws across y

    yam = get_embodiment("yam_bimanual")
    yam = rss.replace(yam, arms=(yam.arms[0], rss.replace(yam.arms[1], home_q=over_cup)))
    scene = rss.Scene(cell=cell, name="pour_water", physics=rss.WATER_60HZ)
    scene.add(rss.robot(yam))
    scene.add(cup(name="cup", inner=inner, wall=wall, at=cup_at, color=(0.85, 0.85, 0.88)))
    water = scene.add(rss.water(name="water", at=(*cup_at[:2], wall + 0.001)))
    h, x, y = (side + thick) / 2, glass_at[0], glass_at[1]
    for name, at, half in (
        ("near", (x - h, y), (thick / 2, h + thick / 2)),
        ("far", (x + h, y), (thick / 2, h + thick / 2)),
        ("right", (x, y - h), (side / 2, thick / 2)),
        ("left", (x, y + h), (side / 2, thick / 2)),
    ):
        scene.add(rss.slab(name=f"glass_{name}", half=(*half, tall / 2), at=(*at, tall / 2), color=(0.72, 0.86, 0.92)))
    paint = (x + h + thick / 2 + 0.0005, y, line)  # the line, painted on the face toward you: drawn, not collided
    scene.add(rss.slab(name="line", half=(0.0005, h, 0.001), at=paint, collide=False, color=(0.85, 0.15, 0.1)))
    sim = scene.build(control_mode="joint_position")

    origin = np.asarray(rss.get_cell(cell).surface_frame[:3])  # the glass's inside as a world box
    lo, hi = origin + np.array((x - side / 2, y - side / 2, 0.0)), origin + np.array((x + side / 2, y + side / 2, tall))

    def page(studio):
        poured = studio.gui.add_number("water in the glass (%)", 0.0, step=0.1, disabled=True)
        level = studio.gui.add_number("level in the glass (mm)", 0.0, step=0.1, disabled=True)
        studio.gui.add_number("fill line (mm)", round(1e3 * line, 1), step=0.1, disabled=True)
        spilled = studio.gui.add_number("spilled on the bench (%)", 0.0, step=0.1, disabled=True)

        def show():
            q = np.asarray(water.particle_q, dtype=np.float64)
            filled = fill_level(q, lo, hi, 0.004)
            poured.value, level.value = round(100.0 * filled["fraction"], 1), round(1e3 * filled["height"], 1)
            spilled.value = round(100.0 * spill_fraction(q, lo, hi, floor=origin[2], band=0.008), 1)

        studio.each_step(show)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
