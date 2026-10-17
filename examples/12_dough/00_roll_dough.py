# SPDX-License-Identifier: Apache-2.0
"""Roll the dough flat

A block of dough under the right hand and a rolling pin beside it: drag the hand down onto the pin, close on it, and roll it across the dough.

Dough here is a plastic solid with a 1 kPa yield stress: it keeps what the pin
does to it. The pin is a capsule on the robot's island (``rss.DOUGH_60HZ``),
so the hand can hold it. The panel reads the dough's footprint against the one
it settled into and its height at the 95th percentile of its particles.
24.1 ms per control step on an RTX PRO 6000, graph-captured: the dough table
runs 150 solver iterations, three times the heap's, so the lump holds its
height while nothing presses it.
"""

import robosimstudio as rss
from robosimstudio_materials.mpm.measures import heap_height, spread


def main():
    cell = "bench_075"
    dough_at = (0.0, -0.31, 0.0)  # under the right hand, in the bench's frame [m]
    block = (0.08, 0.08, 0.03)  # [m]: 12,288 particles
    pin_at = (0.13, -0.31, 0.015)  # lying beside the block, its axis along the bench's y
    pin = (0.015, 0.06)  # radius and half-length [m]
    along_y = (0.7071068, 0.0, 0.0, 0.7071068)  # a capsule stands along its own z; this lays it along y

    scene = rss.Scene(cell=cell, name="roll_dough", physics=rss.DOUGH_60HZ)
    scene.add(rss.robot("yam_bimanual"))
    lump = scene.add(rss.dough(name="dough", size=block, at=dough_at, material="dough"))
    scene.add(
        rss.capsule(name="pin", radius=pin[0], half_height=pin[1], at=pin_at, quat=along_y, color=(0.55, 0.38, 0.22))
    )
    sim = scene.build(control_mode="joint_position")

    rest = lump.entity.rest["particle_q"][lump.particles]
    floor = lump.entity.floor_z

    def page(studio):
        area = studio.gui.add_number("footprint, vs settled", 1.0, disabled=True)
        height = studio.gui.add_number("height (mm)", 0.0, disabled=True)

        def each_step() -> None:
            points = lump.particle_q
            area.value = round(spread(points, rest), 2)
            height.value = round(1000.0 * heap_height(points, floor), 1)

        studio.each_step(each_step)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
