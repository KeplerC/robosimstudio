# SPDX-License-Identifier: Apache-2.0
"""Pour honey into a bowl

A cup of honey standing beside a bowl: click the cup, drag it over the bowl and tip it with the gizmo's rotation ring.

Honey rather than water, because Newton 1.6's water preset loses volume at
tabletop scale (see ``robosimstudio_materials.liquid``). The cup is one rigid
body with 8 mm walls -- two voxels -- and is named in the honey's ``containers``,
which gives its shapes the half-voxel margin a liquid needs to stay inside under
graph capture; ``rss.LIQUID_120HZ`` keeps the cup on the robot's island, turns
the granular projection off and steps the MPM twice per control step, because
honey falling from the lip to the bowl is faster after 7 mm than a 60 Hz step
can catch at a floor (poured from 100 mm, 60 Hz lost 85 of these 8,000
particles through the bowl and the bench and put 4% in the bowl; 120 Hz lost
none and put 42%). The panel counts the honey in the bowl. 30.0 ms per
control step for 8,000 particles on an RTX PRO 6000, graph-captured.
"""

import numpy as np

import robosimstudio as rss
from robosimstudio_materials.liquid import cup
from robosimstudio_materials.mpm.measures import fill_fraction


def main():
    cell = "bench_075"
    cup_at = (0.0, -0.31, 0.0)  # standing under the right hand, in the bench's frame [m]
    bowl_at = (0.0, -0.09, 0.0)  # 220 mm toward the middle of the bench
    inner, wall = (0.05, 0.05, 0.07), 0.008  # the cup's cavity and its wall [m]
    fill = 0.05  # depth of honey in the cup [m]: 8,000 particles
    bowl_inside, bowl_wall, bowl_height = 0.14, 0.004, 0.02  # [m], the bowl's slabs

    scene = rss.Scene(cell=cell, name="pour_honey", physics=rss.LIQUID_120HZ)
    scene.add(rss.robot("yam_bimanual"))
    scene.add(cup(name="cup", inner=inner, wall=wall, at=cup_at, color=(0.85, 0.85, 0.88)))
    honey = scene.add(
        rss.liquid(
            name="honey",
            size=(inner[0], inner[1], fill),
            at=(cup_at[0], cup_at[1], wall + 0.002),
            material="honey",
            containers=("cup",),
        )
    )
    half = bowl_inside / 2 + bowl_wall
    for name, (dx, dy, hx, hy) in {
        "bowl_near": (-half, 0.0, bowl_wall, half),
        "bowl_far": (half, 0.0, bowl_wall, half),
        "bowl_right": (0.0, -half, half, bowl_wall),
        "bowl_left": (0.0, half, half, bowl_wall),
    }.items():
        scene.add(rss.slab(name=name, half=(hx, hy, bowl_height), at=(bowl_at[0] + dx, bowl_at[1] + dy, bowl_height)))
    sim = scene.build(control_mode="joint_position")

    origin = np.asarray(rss.get_cell(cell).surface_frame[:3])
    lo = origin + np.array((bowl_at[0] - bowl_inside / 2, bowl_at[1] - bowl_inside / 2, 0.0))
    hi = origin + np.array((bowl_at[0] + bowl_inside / 2, bowl_at[1] + bowl_inside / 2, 2 * bowl_height))

    def page(studio):
        poured = studio.gui.add_number("honey in the bowl (%)", 0.0, disabled=True)
        studio.each_step(lambda: setattr(poured, "value", round(100.0 * fill_fraction(honey.particle_q, lo, hi), 1)))

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
