# SPDX-License-Identifier: Apache-2.0
"""Fold a sheet of paper

An A4 sheet of copy paper on the bench between two YAM arms: carry a corner over, press the fold flat, let go, and it stays folded.

Click the sheet to pin a patch to a handle and carry it over the far half; click the standing
loop for a second handle and press it flat, or push it down with a hand. Press **Release all**:
the hinges bent past the yield keep their angle, so the flap stays where it was pressed instead
of springing open. The **Crease** folder reads ``crease_angle`` through the sheet's centreline
(how far the sheet stays turned there, in radians) and ``fold_extent`` (its span across that
line over its flat span). Set ``yield_angle`` to 0 under **Material** and the same fold springs open.
"""

import robosimstudio as rss
from robosimstudio.assetlib.semantics.snapshot import Snapshot
from robosimstudio.assetlib.vocabulary.measures import cloth_fold
from robosimstudio.robots.registry import get_embodiment
from robosimstudio_materials.paper import crease_angle


def main():
    cell = "bench_0728"
    size = (0.21, 0.297)  # A4 [m]
    cells = (20, 28)  # a 10.5 mm cell, the one PAPER's drape was calibrated at
    corner = (-size[0] / 2, -0.20, 0.005)  # the sheet's near corner, in the bench's work-surface frame
    crease_y = corner[1] + size[1] / 2  # the fold line: the sheet's own centreline across y

    # A hand holds paper by friction alone, which needs one-way coupling and full-surface
    # contact, as it does for the towel.
    physics = rss.replace(
        rss.GENERIC,
        coupling=rss.coupler("one_way"),
        collision=rss.CollisionSpec(full_surface_contact=True),
    )

    scene = rss.Scene(cell=cell, physics=physics, name="paper")
    scene.add(rss.robot(get_embodiment("yam_bimanual")))
    scene.add(rss.paper(name="paper", size=size, cells=cells, at=corner))
    sim = scene.build(control_mode="joint_position")
    sheet = sim["paper"].entity
    line = {"crease": {"axis": 1, "at": crease_y}}

    def page(studio):
        with studio.gui.add_folder("Crease"):
            angle = studio.gui.add_number("crease_angle (rad)", 0.0, disabled=True)
            extent = studio.gui.add_number("fold_extent", 1.0, disabled=True)

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
