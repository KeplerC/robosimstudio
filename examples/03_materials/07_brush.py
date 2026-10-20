# SPDX-License-Identifier: Apache-2.0
"""Brush: scrub a patch

A nylon dish brush stands on the bench under the right hand, which points down at its handle, beside a marked square: drag the hand down onto the handle, close, press the bristles to the bench and scrub the square.

The brush is 400 bristles of four capsules each on VBD, rooted in a sole of
twenty strips, under a block on the arm's solver that the hand closes on
(``rss.BRUSH_60HZ``); the field cannot hold that block up across the two
solvers, so it is gravity-compensated, and the arm does not feel the bristles
push back. The panel reads the share of the square's 5 mm cells a bristle tip
has touched while down on the bench. ``tests/test_scrub_bench`` closes the hand,
presses 4 mm and makes three strokes across the square: all of its cells
scrubbed, every tip down throughout, the handle slipping 0.5 mm in the
fingers, at 22.4 ms per control step on an RTX PRO 6000, GPU 3,
graph-captured.
"""

import numpy as np

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment


def main():
    cell = "bench_075"
    brush_at = (0.025, -0.31, 0.0)  # the bristle tips' footprint centre, under the right hand [m]
    patch = (-0.025, -0.27, 0.075, -0.17)  # the 100 mm square to scrub, toward the middle of the bench [m]
    handle_half = (0.04, 0.011, 0.015)  # the block the hand closes on: 22 mm across the jaws, 30 mm tall [m]
    # The right arm's joints [rad]: tool centre 100 mm over the bench above the brush, pointing down,
    # jaws closing along y, across the handle. Solved with ``sim.arm("right").solve_q_full``.
    over_brush = (0.0, 2.149, 1.879, -1.3, 0.0, 0.0)

    yam = get_embodiment("yam_bimanual")
    left, right = yam.arms
    yam = rss.replace(yam, arms=(left, rss.replace(right, home_q=over_brush)))

    scene = rss.Scene(cell=cell, name="scrub_the_bench", physics=rss.BRUSH_60HZ)
    scene.add(rss.robot(yam))
    spec = rss.brush(name="brush", at=brush_at, material="nylon", handle="handle")
    handle_at = (brush_at[0], brush_at[1], spec.sole_top + handle_half[2])  # standing on the sole
    scene.add(rss.box(name="handle", half=handle_half, at=handle_at, friction=1.0, color=(0.15, 0.15, 0.17)))
    brush = scene.add(spec)
    scene.add(rss.RegionSpec(name="patch", frame="", xy_ranges=patch, z_range=(0.0, 0.004)))
    sim = scene.build(control_mode="joint_position")

    # The square as world x, y bounds, from the bench's surface frame.
    origin = np.asarray(rss.get_cell(cell).surface_frame[:2])
    lo, hi = origin + np.asarray(patch[:2]), origin + np.asarray(patch[2:])

    def page(studio):
        scrubbed = studio.gui.add_number("patch scrubbed (%)", 0.0, disabled=True)
        studio.each_step(lambda: setattr(scrubbed, "value", round(100.0 * brush.entity.scrubbed(lo, hi), 1)))

    # The drive panels and ``regions``, which draws the square on the bench: a bare scene's page leaves it out.
    rss.view(sim, page=page, panels=("inspector", "overlay", "arm", "object", "regions"))


if __name__ == "__main__":
    main()
