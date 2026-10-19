# SPDX-License-Identifier: Apache-2.0
"""Roll the dough flat

A block of dough and a roller standing in its rest beside it, the right hand pointing down over the roller's handle: drag the hand down onto the handle, close on it, lift the roller out of its rest, lower it onto the near edge of the block and roll it across, back and across again.

Dough here is a plastic solid with a 1 kPa yield stress: it keeps what the
roller does to it. The roller (``rss.roller``) is a 40 mm drum on an axle under
a handle the jaws close across, a small articulation on the robot's island
(``rss.DOUGH_60HZ``). The grip alone would not hold it down (a plain handle slid
30 mm up through the jaws in one pass), so the handle's shape carries the load:
a shoulder under the fingertips takes the dough's push, a wall either side of
each finger takes the drag. Three passes at 0.1 m/s, the drum's underside about
11 mm over the bench, take the block from 28.6 to 12.6 mm and its footprint to
about twice the settled one (2.01 to 2.11 in three drives on Newton 1.7, where
1.6 read 1.75), the drum turning 4.2 to 4.6 rad a pass; welded to its handle it
slid and left 18.6 mm on 1.6 (``probe_roll2``). The panel reads the footprint
against the settled one and the height at the 95th percentile of the particles.
21.8 ms per control step on an RTX PRO 6000, graph-captured
(``tests/test_roll_dough``, GPU 3).
"""

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment
from robosimstudio_materials.mpm.measures import heap_height, spread


def main():
    cell = "bench_075"
    dough_at = (0.0, -0.31, 0.0)  # in the bench's frame [m]
    block = (0.08, 0.08, 0.03)  # [m]: 12,288 particles
    roller_at = (-0.14, -0.31, 0.02)  # the axle of the 40 mm drum resting on the bench, along y
    # The right arm's joints [rad]: tool centre 150 mm over the bench above the roller, pointing down,
    # jaws closing along y, across the handle. Solved with ``sim.arm("right").solve_q_full``.
    over_roller = (0.0, 1.4922, 1.2524, -1.3311, 0.0, 0.0)

    yam = get_embodiment("yam_bimanual")
    left, right = yam.arms
    yam = rss.replace(yam, arms=(left, rss.replace(right, home_q=over_roller)))

    scene = rss.Scene(cell=cell, name="roll_dough", physics=rss.DOUGH_60HZ)
    scene.add(rss.robot(yam))
    lump = scene.add(rss.dough(name="dough", size=block, at=dough_at, material="dough"))
    scene.add(rss.roller(name="roller", at=roller_at, color=(0.55, 0.38, 0.22)))
    for name, dx in (("rest_near", -0.026), ("rest_far", 0.026)):  # chocks: the drum stays put until it is taken
        scene.add(rss.slab(name=name, half=(0.005, 0.07, 0.006), at=(roller_at[0] + dx, roller_at[1], 0.006)))
    sim = scene.build(control_mode="joint_position")

    rest = lump.entity.rest["particle_q"][lump.particles]
    floor = lump.entity.floor_z

    def page(studio):
        area = studio.gui.add_number("footprint, vs settled", 1.0, step=0.01, disabled=True)
        height = studio.gui.add_number("height (mm)", 0.0, step=0.1, disabled=True)

        def each_step() -> None:
            points = lump.particle_q
            area.value = round(spread(points, rest), 2)
            height.value = round(1000.0 * heap_height(points, floor), 1)

        studio.each_step(each_step)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
