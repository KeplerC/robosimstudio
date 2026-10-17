# SPDX-License-Identifier: Apache-2.0
"""Wipe a spill

A ketchup puddle under the right hand and a rubber-faced wiper beside it: drag the hand down onto the wiper, close on it, and push the puddle off its patch.

Ketchup is a Bingham fluid here (a 300 Pa yield stress, viscosity 100), so the
puddle holds its edge until the wiper moves it. The panel reads how much of it
has been pushed off the patch it started on and lies on the bench
(``spill_fraction``), and how far it has spread. A deformable sponge in place
of the rigid wiper is a later phase. 16.2 ms per control step on an RTX PRO
6000, graph-captured.
"""

import numpy as np

import robosimstudio as rss
from robosimstudio_materials.mpm.measures import spill_fraction, spread


def main():
    cell = "bench_075"
    puddle_at = (0.0, -0.31, 0.0)  # under the right hand, in the bench's frame [m]
    puddle = (0.10, 0.10, 0.006)  # a 100 mm square, 6 mm deep: 1.5 voxels, 2,400 particles
    wiper_at = (0.12, -0.31, 0.015)  # standing on its edge beside the puddle, 20 mm thick

    scene = rss.Scene(cell=cell, name="wipe_spill", physics=rss.LIQUID_60HZ)
    scene.add(rss.robot("yam_bimanual"))
    sauce = scene.add(rss.liquid(name="ketchup", size=puddle, at=puddle_at, material="ketchup"))
    scene.add(rss.box(name="wiper", half=(0.01, 0.06, 0.015), at=wiper_at, friction=1.5, color=(0.15, 0.15, 0.17)))
    sim = scene.build(control_mode="joint_position")

    # The patch the puddle started on, as a world box: what counts as wiped is what lies outside it.
    origin = np.asarray(rss.get_cell(cell).surface_frame[:3])
    lo = origin + np.array((puddle_at[0] - puddle[0] / 2, puddle_at[1] - puddle[1] / 2, 0.0))
    hi = origin + np.array((puddle_at[0] + puddle[0] / 2, puddle_at[1] + puddle[1] / 2, 0.03))
    rest = sauce.entity.rest["particle_q"][sauce.particles]

    def page(studio):
        wiped = studio.gui.add_number("pushed off the patch (%)", 0.0, disabled=True)
        area = studio.gui.add_number("footprint, vs settled", 1.0, disabled=True)

        def each_step() -> None:
            points = sauce.particle_q
            wiped.value = round(100.0 * spill_fraction(points, lo, hi, floor=origin[2], band=0.005), 1)
            area.value = round(spread(points, rest), 2)

        studio.each_step(each_step)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
