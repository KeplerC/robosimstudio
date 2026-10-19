# SPDX-License-Identifier: Apache-2.0
"""Liquid: pour honey

A cup of honey under the right hand and a bowl beside it: drag the hand down until its open jaws straddle the cup, close on it, drag it up and across over the bowl, then turn the gizmo 40 degrees about the vertical and 140 degrees about the axis pointing away from the robot.

The jaws open 94 mm and the cup is 66 mm across, so they close on its outside walls; it is a rigid body on the
arm's island (``rss.LIQUID_120HZ``), held by friction, its 8 mm walls named in the honey's ``containers`` for
the margin a liquid needs. The arm sags about 30 mm under 0.48 kg of cup and honey: lift the gizmo 150 mm over
the bench to clear the bowl's 40 mm walls. The wrist's last joint stops at 120 degrees: turned 140 about the
away axis alone, the hand turns 122 and the honey creeps along the wall, about 56% in the bowl after 12 s held
(three drives, ``fixprobe.py``, GPUs 1 and 6). The turn about the vertical first lets the arm reach 140, where the
honey leaves the cup in one mass within a second: 99% or more in the bowl, 0 to 8 of the 8,000 particles through
the bench where it lands, the cup 7 mm along the pads (``tests/test_pour_honey``, GPU 1, RTX PRO 6000). Honey,
because Newton 1.6's water preset loses volume (``robosimstudio_materials.liquid``). The panel counts the honey
in the bowl. 29.8 ms per control step, graph-captured.
"""

import numpy as np

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment
from robosimstudio_materials.liquid import cup
from robosimstudio_materials.mpm.measures import fill_fraction


def main():
    cell = "bench_075"
    cup_at = (0.12, -0.31, 0.0)  # under the right hand, in the bench's frame [m]
    bowl_at = (0.12, -0.16, 0.0)  # 150 mm toward the middle of the bench
    inner, wall = (0.05, 0.05, 0.07), 0.008  # the cup's cavity and its walls [m]
    fill = 0.05  # depth of honey in the cup [m]: 8,000 particles
    bowl_inside, bowl_wall, bowl_height = 0.14, 0.004, 0.02  # [m], the bowl's slabs
    # The right arm's joints [rad]: tool centre 155 mm over the bench above the cup, pointing forward, the
    # jaws open across y. Solved with ``sim.arm("right").solve_q_full``.
    over_cup = (0.0, 1.723, 0.642, 1.081, 0.0, 0.0)

    yam = get_embodiment("yam_bimanual")
    left, right = yam.arms
    yam = rss.replace(yam, arms=(left, rss.replace(right, home_q=over_cup)))

    scene = rss.Scene(cell=cell, name="pour_honey", physics=rss.LIQUID_120HZ)
    scene.add(rss.robot(yam))
    scene.add(cup(name="cup", inner=inner, wall=wall, at=cup_at, color=(0.85, 0.85, 0.88)))
    honey = scene.add(
        rss.liquid(name="honey", size=(*inner[:2], fill), at=(*cup_at[:2], wall + 0.002), containers=("cup",))
    )
    h = bowl_inside / 2 + bowl_wall
    for name, (dx, dy, hx, hy) in {
        "bowl_near": (-h, 0.0, bowl_wall, h),
        "bowl_far": (h, 0.0, bowl_wall, h),
        "bowl_right": (0.0, -h, h, bowl_wall),
        "bowl_left": (0.0, h, h, bowl_wall),
    }.items():
        scene.add(rss.slab(name=name, half=(hx, hy, bowl_height), at=(bowl_at[0] + dx, bowl_at[1] + dy, bowl_height)))
    sim = scene.build(control_mode="joint_position")

    # The bowl's inside as a world box, 5 mm under the bench top, where the bottom layer rests.
    origin = np.asarray(rss.get_cell(cell).surface_frame[:3])
    lo = origin + np.array((bowl_at[0] - bowl_inside / 2, bowl_at[1] - bowl_inside / 2, -0.005))
    hi = origin + np.array((bowl_at[0] + bowl_inside / 2, bowl_at[1] + bowl_inside / 2, 2 * bowl_height))

    def page(studio):
        poured = studio.gui.add_number("honey in the bowl (%)", 0.0, disabled=True)
        studio.each_step(lambda: setattr(poured, "value", round(100.0 * fill_fraction(honey.particle_q, lo, hi), 1)))

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
