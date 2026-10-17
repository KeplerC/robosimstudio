# SPDX-License-Identifier: Apache-2.0
"""Fold a sheet of paper

An A4 sheet of copy paper between two YAM arms: close both hands, press the left one down on the far end, put the right one on the near edge and tick **Edge on the right hand**, carry the edge up, across and down onto the middle, stamp the fold flat with the left hand, untick, and the sheet stays folded in thirds.

The carry is an attachment grasp and the page says so: ticked, the row along the near edge is
pinned at the right hand's frame, because the YAM's pads cannot pinch a sheet lying flat, nor
one hanging over the bench's edge, where the inner finger cannot get under the drape. The rest
is contact. Unheld, the sheet slides 0.29 m along the bench ahead of its own fold; the left hand
pressing the far end keeps it within 2 mm. The crease is the left hand's: the flat underside of
its closed fingers stamps the loop at three places along the fold while the right hand still holds
the edge down, and the hinges bent past the yield keep 2.3 to 3.1 rad over three drives
(``foldletter.py``, GPU 7); pressed after the edge is let go, the loop rolls open instead (1.4). The
**Crease** folder reads ``crease_angle`` through the fold line and ``fold_extent`` (the sheet's span
over its flat span). Set ``yield_angle`` to 0 under **Material** and the same fold springs open.
"""

import robosimstudio as rss
from robosimstudio.assetlib.semantics.snapshot import Snapshot
from robosimstudio.assetlib.vocabulary.measures import cloth_fold
from robosimstudio.robots.registry import get_embodiment
from robosimstudio_materials.paper import crease_angle


def main():
    size = (0.21, 0.297)  # A4 [m]
    cells = (20, 28)  # a 10.5 mm cell, the one PAPER's drape was calibrated at
    corner = (-size[0] / 2, -0.20, 0.005)  # the sheet's near corner, in the bench's work-surface frame
    crease_y = corner[1] + size[1] / 3  # the fold line, a third of the way along the sheet
    edge = ("edge", lambda rest: rest[:, 1] <= corner[1] + 0.005)  # the row along the near edge

    # One-way coupling and full-surface contact are what let a hand's colliders press a sheet.
    physics = rss.replace(
        rss.GENERIC, coupling=rss.coupler("one_way"), collision=rss.CollisionSpec(full_surface_contact=True)
    )

    scene = rss.Scene(cell="bench_0728", physics=physics, name="paper")
    scene.add(rss.robot(get_embodiment("yam_bimanual")))
    scene.add(rss.paper(name="paper", size=size, cells=cells, at=corner, particle_sets=(edge,)))
    sim = scene.build(control_mode="joint_position")
    sheet = sim["paper"].entity
    line = {"crease": {"axis": 1, "at": crease_y}}

    def page(studio):
        with studio.gui.add_folder("Crease"):
            hold = studio.gui.add_checkbox("Edge on the right hand", initial_value=False)
            angle = studio.gui.add_number("crease_angle (rad)", 0.0, disabled=True)
            extent = studio.gui.add_number("fold_extent", 1.0, disabled=True)

        def attach(_) -> None:  # the page's set grab, pinned to the right hand's frame: an attachment grasp
            grips = studio.deformables
            grips.selected, grips.selected_set, grips.follow_arm[0] = "paper", "edge", "right" if hold.value else ""
            grips.grab_set_requested, grips.release_all_requested = hold.value, not hold.value

        def each_step() -> None:
            sheet.update_probe(sim.state, sim.step_count)
            snapshot = Snapshot(sim.make_context())
            angle.value = round(crease_angle(snapshot, "paper", line)["angle"], 2)
            extent.value = round(cloth_fold(snapshot.ctx, "paper", axis=1)[0], 2)

        hold.on_update(attach)
        studio.each_step(each_step)

    # "tune" adds the Material folder, where yield_angle is a slider, to the driving panels.
    rss.view(sim, page=page, panels=("inspector", "overlay", "arm", "object", "cloth", "tune"))


if __name__ == "__main__":
    main()
