# SPDX-License-Identifier: Apache-2.0
"""Wipe a spill

A ketchup puddle in front of the right hand and a rubber-faced wiper standing under it: drag the hand down onto the wiper, close on it, and push the puddle toward the middle of the bench.

Ketchup is a Bingham fluid here (a 300 Pa yield stress, viscosity 100), so the
puddle holds its edge until the wiper moves it. The hand starts over the wiper
pointing down, its tool centre 100 mm over the bench and 85 mm over the wiper's
centre, its jaws across the wiper's 20 mm faces, so the pads cover the wiper's
whole height and the push cannot tip it: the same wiper on the
page's drag handle, which holds a body's tilt with a soft spring, pitched over
under the puddle's resistance and rode across it, moving 22-51% of a 12 mm
puddle off its patch in three identical runs; held by the hand it moved 96-98%
of that puddle in three, and 98% of this 6 mm one (``wipe_probe`` and
``session_wipe``, RTX PRO 6000, Newton 1.6.0). The panel counts the
ketchup that has left the patch it settled on, at any height, because the
wiper piles it into a ridge, and reads the ridge's height. The puddle is only
pushed, never dropped, so it would hold at 60 Hz; it runs on the one liquid
table, ``rss.LIQUID_120HZ``, at 25.9 ms per control step, graph-captured.
"""

import numpy as np

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment
from robosimstudio_materials.mpm.measures import heap_height, spill_fraction


def main():
    cell = "bench_075"
    puddle_at = (0.025, -0.19, 0.0)  # 120 mm from the wiper toward the middle of the bench, in its frame [m]
    puddle = (0.10, 0.10, 0.006)  # a 100 mm square, 6 mm deep: two layers of 40 x 40, 3,200 particles
    wiper_at = (0.025, -0.31, 0.015)  # under the right hand, standing on its 20 mm edge, long along x
    # The right arm's joints [rad]: tool centre 100 mm over the bench above the wiper, pointing down,
    # jaws closing along y, across the wiper's faces. Solved with ``sim.arm("right").solve_q_full``.
    over_wiper = (0.0, 2.149, 1.879, -1.3, 0.0, 0.0)

    yam = get_embodiment("yam_bimanual")
    left, right = yam.arms
    yam = rss.replace(yam, arms=(left, rss.replace(right, home_q=over_wiper)))

    scene = rss.Scene(cell=cell, name="wipe_spill", physics=rss.LIQUID_120HZ)
    scene.add(rss.robot(yam))
    sauce = scene.add(rss.liquid(name="ketchup", size=puddle, at=puddle_at, material="ketchup"))
    scene.add(rss.box(name="wiper", half=(0.06, 0.01, 0.015), at=wiper_at, friction=1.5, color=(0.15, 0.15, 0.17)))
    sim = scene.build(control_mode="joint_position")

    # The patch the puddle settled on: its footprint as built, half a spacing wider, 30 mm tall.
    rest = sauce.entity.rest["particle_q"][sauce.particles]
    pad = 0.5 * sauce.spec.particle_spacing
    lo, hi = rest.min(axis=0) - np.array((pad, pad, 0.0)), rest.max(axis=0) + np.array((pad, pad, 0.03))
    floor = sauce.entity.floor_z

    def page(studio):
        wiped = studio.gui.add_number("pushed off the patch (%)", 0.0, disabled=True)
        ridge = studio.gui.add_number("height (mm)", 0.0, disabled=True)

        def each_step() -> None:
            points = sauce.particle_q
            wiped.value = round(100.0 * spill_fraction(points, lo, hi), 1)
            ridge.value = round(1000.0 * heap_height(points, floor), 1)

        studio.each_step(each_step)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
