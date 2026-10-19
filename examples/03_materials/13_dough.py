# SPDX-License-Identifier: Apache-2.0
"""Dough: press a ball

A puck of dough under the right hand: close the gripper, drag the hand down into the dough and lift it away again.

What a plastic solid keeps is the measure: the press probe behind the ``dough``
preset kept 83% of a 15 mm compression after the plate lifted (plasticine 75%,
clay 66%). The panel reads the puck's height as a share of the height it
settled at, and its footprint against the settled one: the closed hand driven
15 mm under the top through the arm panel and lifted away leaves 76% under
the hand and 76% after (``tests/test_press_dough``, GPU 0). Change ``material`` to
``"plasticine"`` or ``"clay"`` for a stiffer lump. 23.4 ms per control step on
an RTX PRO 6000, graph-captured.
"""

import robosimstudio as rss
from robosimstudio_materials.mpm.measures import heap_height, spread


def main():
    cell = "bench_075"
    puck_at = (0.0, -0.31, 0.0)  # under the right hand, in the bench's frame [m]
    puck = (0.06, 0.06, 0.04)  # diameter, diameter, height [m]: a cylinder of 7,168 particles
    material = "dough"

    scene = rss.Scene(cell=cell, name="press_dough", physics=rss.DOUGH_60HZ)
    scene.add(rss.robot("yam_bimanual"))
    lump = scene.add(rss.dough(name="dough", shape="cylinder", size=puck, at=puck_at, material=material))
    sim = scene.build(control_mode="joint_position")

    rest = lump.entity.rest["particle_q"][lump.particles]
    floor = lump.entity.floor_z
    settled = heap_height(rest, floor)

    def page(studio):
        kept = studio.gui.add_number("height, vs settled (%)", 100.0, step=0.1, disabled=True)
        area = studio.gui.add_number("footprint, vs settled", 1.0, step=0.01, disabled=True)

        def each_step() -> None:
            points = lump.particle_q
            kept.value = round(100.0 * heap_height(points, floor) / settled, 1)
            area.value = round(spread(points, rest), 2)

        studio.each_step(each_step)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
