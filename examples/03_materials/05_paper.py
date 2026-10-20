# SPDX-License-Identifier: Apache-2.0
"""Paper: fold a sheet

Bring each hand in along the edge of the A4 sheet that runs past the box, from its own corner, with **Grip amount** 0.79, set 0.986 on both so the pads stop on the sheet, carry the edge up and over together, each hand turning with the flap to vertical and no further, set 0.79 30 mm over the far edge, then **Close** each hand in turn and run its fingertip along the fold from the middle out: the sheet stays folded in half.

The two arms collide with each other here and keep 79 mm apart; each hand turns 88 degrees while it holds the edge, so the flap bends by the grip instead of riding in it, and both have let go and left before a fingertip touches the fold. Over five drives of the take's own path (``tests/test_fold_paper_robot``, GPU 5) the edges ended 1.3 to 1.5 mm apart, 5.7 mm at worst at a corner, and the fold kept 2.31 to 2.34 rad three seconds after the hands left, ``fold_extent`` 0.51 to 0.53. A sheet flat on the bench gives a pad nothing to get under, so it lies on a box with its near edge 35 mm past it. **Close** takes the pads to 0.8 mm apart and crushes the sheet; 0.986 stops them 2.1 mm apart. The **Crease** folder reads ``crease_angle`` and ``fold_extent``.
"""

import numpy as np

import robosimstudio as rss
from robosimstudio.assetlib.semantics.snapshot import Snapshot
from robosimstudio.assetlib.vocabulary.measures import cloth_fold
from robosimstudio.robots.registry import get_embodiment
from robosimstudio_materials.paper import crease_angle


def main():
    size, cells = (0.297, 0.21), (28, 20)  # A4, its long side away from the robot; a 10.6 x 10.5 mm cell
    box = (0.295, 0.28, 0.08)  # what the sheet lies on [m]: flat on the bench, a pad has nothing to get under
    corner = (-0.05, -0.105, 0.085)  # the sheet's near corner (bench frame), across the middle, on the box
    overhang = 0.035  # how far the near edge runs past the box: the pads close on it from either corner
    nubs = 3.0  # the grip nubs' friction: the asset's 4.0 holds paper's 0.3 at 1.10, 3.0 at 0.95
    fold_x = corner[0] + size[0] / 2  # the fold line, half way along the sheet

    physics = rss.replace(
        rss.GENERIC, coupling=rss.coupler("one_way"), collision=rss.CollisionSpec(full_surface_contact=True)
    )
    # The two arms collide with each other; each hand's pads servo at 2 kN/m, their force capped at 30 N.
    yam = rss.replace(get_embodiment("yam_bimanual"), cross_arm_collisions=True)
    hands = rss.ActuationSpec(target_ke=2000.0, target_kd=40.0, effort_limit=30.0)

    scene = rss.Scene(cell="bench_0728", physics=physics, name="paper")
    scene.add(rss.robot(yam, gripper=hands))
    at = (corner[0] + overhang + box[0] / 2, 0.0, box[2] / 2)
    scene.add(rss.slab(name="box", half=tuple(v / 2 for v in box), at=at, color=(0.62, 0.47, 0.33)))
    scene.add(rss.paper(name="paper", size=size, cells=cells, at=corner))
    sim = scene.build(control_mode="joint_position")
    mu = sim.model.shape_material_mu  # VBD mixes a shape's friction with the sheet's on every contact
    mu.assign(np.minimum(mu.numpy(), nubs))
    sheet = sim["paper"].entity
    line = {"crease": {"axis": 0, "at": fold_x}}

    def page(studio):
        with studio.gui.add_folder("Crease", order=-1.0):  # first on the panel: the readouts are the evidence
            angle = studio.gui.add_number("crease_angle (rad)", 0.0, step=0.01, disabled=True)
            extent = studio.gui.add_number("fold_extent", 1.0, step=0.01, disabled=True)

        def each_step() -> None:
            sheet.update_probe(sim.state, sim.step_count)
            snapshot = Snapshot(sim.make_context())
            angle.value = round(crease_angle(snapshot, "paper", line)["angle"], 2)
            extent.value = round(cloth_fold(snapshot.ctx, "paper", axis=0)[0], 2)

        studio.each_step(each_step)

    # "tune" adds the Material folder, where yield_angle is a slider, to the driving panels.
    rss.view(sim, page=page, panels=("inspector", "overlay", "arm", "object", "cloth", "tune"))


if __name__ == "__main__":
    main()
