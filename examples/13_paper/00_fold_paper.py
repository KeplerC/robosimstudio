# SPDX-License-Identifier: Apache-2.0
"""Fold a sheet of paper

An A4 sheet hangs 35 mm over the bench's right edge: close the left hand and press the far end down, drag the right hand down and in so its jaws straddle the overhanging corner, press **Close**, carry the edge up and over while turning the hand over, lay it on the middle and draw it on and up 3 cm, stamp the fold with the left hand, set the right hand's **Grip amount** to 0.5 and draw it out past the edge: the sheet stays folded in thirds.

The pads hold the paper by friction alone, and the fold keeps 2.53 to 2.67 rad over 18 drives
(``tests/test_fold_paper_robot``, its take, Newton 1.7). The overhang is what makes the pinch possible:
a sheet lying flat gives the lower pad nothing to get under (seven top-down pinches raised it 0 mm), and
the palm stops the jaws at the corner. The three stamps also shove the folded sheet 46 to 47 mm along
the bench, away from the edge, and the fold rides with it (``tests/test_fold_paper_robot``, two drives).
The stamps need the edge still held and lifted clear: let go first and the stamped fold kept 1.0 rad;
held 3 mm over the sheet, the stamps pushed it into the lower finger and the leaving hand dragged it 94
to 108 mm (``fold_contact.py``, GPUs 3 and 5). The **Crease** folder reads ``crease_angle`` and
``fold_extent``; set ``yield_angle`` to 0 under **Material** and the same fold springs open.
"""

import robosimstudio as rss
from robosimstudio.assetlib.semantics.snapshot import Snapshot
from robosimstudio.assetlib.vocabulary.measures import cloth_fold
from robosimstudio.robots.registry import get_embodiment
from robosimstudio_materials.paper import crease_angle


def main():
    size = (0.21, 0.297)  # A4 [m]
    cells = (20, 28)  # a 10.5 mm cell, the one PAPER's drape was calibrated at
    corner = (-0.055, -0.435, 0.005)  # the near corner, 35 mm past the bench's right edge (bench frame)
    crease_y = corner[1] + size[1] / 3  # the fold line, a third of the way along the sheet
    # The right arm's joints [rad]: the tool along +x tipped 10 degrees down, the jaws closing
    # vertically, the pads 30 mm over the overhang and 20 mm short of its corner. Solved once with
    # ``sim.arm("right").solve_q_full``; move the sheet, solve again.
    beside = (-1.0458, 2.1677, 0.1161, 1.5708, -1.0282, 1.1579)

    # One-way coupling and full-surface contact are what let the pads pinch a sheet.
    physics = rss.replace(
        rss.GENERIC, coupling=rss.coupler("one_way"), collision=rss.CollisionSpec(full_surface_contact=True)
    )
    yam = get_embodiment("yam_bimanual")
    left, right = yam.arms
    yam = rss.replace(yam, arms=(left, rss.replace(right, home_q=beside)))

    scene = rss.Scene(cell="bench_0728", physics=physics, name="paper")
    scene.add(rss.robot(yam))
    scene.add(rss.paper(name="paper", size=size, cells=cells, at=corner))
    sim = scene.build(control_mode="joint_position")
    sheet = sim["paper"].entity
    line = {"crease": {"axis": 1, "at": crease_y}}

    def page(studio):
        with studio.gui.add_folder("Crease", order=-1.0):  # first on the panel: the readouts are the evidence
            angle = studio.gui.add_number("crease_angle (rad)", 0.0, step=0.01, disabled=True)
            extent = studio.gui.add_number("fold_extent", 1.0, step=0.01, disabled=True)

        def each_step() -> None:
            sheet.update_probe(sim.state, sim.step_count)
            snapshot = Snapshot(sim.make_context())
            angle.value = round(crease_angle(snapshot, "paper", line)["angle"], 2)
            extent.value = round(cloth_fold(snapshot.ctx, "paper", axis=1)[0], 2)

        studio.each_step(each_step)

    # "tune" adds the Material folder, where yield_angle is a slider, to the driving panels.
    rss.view(sim, page=page, panels=("inspector", "overlay", "arm", "object", "cloth", "tune"))


if __name__ == "__main__":
    main()
