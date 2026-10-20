# SPDX-License-Identifier: Apache-2.0
"""Paper: fold a sheet

Bring each hand in along the edge of the A4 sheet that runs past the box, from its own corner, with **Grip amount** 0.79, and press **Close**: the fingers stall on the sheet at their 20 N cap. Carry the edge up to upright, each hand turning with the flap to 88 degrees and no further, lay it on the far half and open 40 mm over it. Then close each hand and run its fingertip along its half of the fold, out and back, pressing about 13 N: the sheet stays folded in half.

In eight drives of the take's path (``tests/test_fold_paper_robot``, GPUs 3 and 5) the arms, which collide,
came no closer than 79.7 mm and kept 2.2 mm on their own sides of the middle; each hand turned 88.3 degrees
holding the edge, its grip servo at the cap, pad friction 0.78 and nubs' 0.95, and only the box and one
fingertip touched the sheet as it was creased. The edges ended 0.5 to 1.1 mm apart and the fold kept 2.80
to 2.83 rad once the hands had gone, unchanged 3 s later, ``fold_extent`` 0.55 to 0.59; the far half moved
9.3 mm at most. The cap bounds the squeeze, not every push: at the 95th percentile one finger of a pair read
17.7 N holding and the creasing fingertip 24 to 33 N. Laid but not creased, the fold keeps 2.23 to 2.30
rad. At PAPER's own ``yield_angle`` 0.35 the sheet's bend over the pads pried the capped jaws open in the lay
and it ended flat, ``fold_extent`` 1.01 (2 of 2 drives). The **Crease** folder reads ``crease_angle``.
"""

import numpy as np

import robosimstudio as rss
from robosimstudio.assetlib.semantics.snapshot import Snapshot
from robosimstudio.assetlib.vocabulary.measures import cloth_fold
from robosimstudio.robots.registry import get_embodiment
from robosimstudio_materials.paper import PAPER, crease_angle


def main():
    size, cells = (0.297, 0.21), (28, 20)  # A4, its long side away from the robot; a 10.6 x 10.5 mm cell
    box = (0.295, 0.28, 0.08)  # what the sheet lies on [m]: flat on the bench, a pad has nothing to get under
    corner = (-0.05, -0.105, 0.085)  # the sheet's near corner (bench frame), across the middle, on the box
    overhang = 0.035  # how far the near edge runs past the box: the pads close on it from either corner
    nubs = 3.0  # the grip nubs' friction: the asset's 4.0 holds paper's 0.3 at 1.10, 3.0 at 0.95
    sheet = rss.replace(PAPER, yield_angle=0.2)  # a hinge bent 0.2 rad past its rest keeps the rest

    # The sheet's push fed back along each finger's slide: a hand closes until it pushes back with the
    # grip cap. VBD friction that slips at 1 mm/s, not 10, keeps a 5 g sheet where it lies.
    coupling = rss.coupler("one_way", contact_forces=True, slide_feedback=True)
    physics = rss.replace(rss.GENERIC, coupling=coupling, collision=rss.CollisionSpec(full_surface_contact=True))
    physics = physics.with_backend("*", rss.backend("vbd", friction_epsilon=1.0e-3, rigid_body_contact_buffer_size=256))
    # The arms collide. Each hand's servo stalls on a sheet at its 20 N cap (10 N a finger), closing at 0.1 m/s.
    yam = rss.replace(get_embodiment("yam_bimanual"), cross_arm_collisions=True)
    hands = rss.ActuationSpec(target_ke=1.0e5, target_kd=20.0, effort_limit=20.0, joint_damping=200.0)

    scene = rss.Scene(cell="bench_0728", physics=physics, name="paper")
    scene.add(rss.robot(yam, gripper=hands))
    at = (corner[0] + overhang + box[0] / 2, 0.0, box[2] / 2)  # a rubber-topped box: paper's 0.3 mixes to 0.95
    scene.add(rss.slab(name="box", half=tuple(v / 2 for v in box), at=at, color=(0.62, 0.47, 0.33), friction=3.0))
    scene.add(rss.paper(name="paper", size=size, cells=cells, at=corner, material=sheet))
    sim = scene.build(control_mode="joint_position")
    sim.model.shape_material_mu.assign(np.minimum(sim.model.shape_material_mu.numpy(), nubs))  # mixed per contact
    folded = sim["paper"].entity
    line = {"crease": {"axis": 0, "at": corner[0] + size[0] / 2}}  # the fold line, half way along the sheet

    def page(studio):
        with studio.gui.add_folder("Crease", order=-1.0):  # first on the panel: the readouts are the evidence
            angle = studio.gui.add_number("crease_angle (rad)", 0.0, step=0.01, disabled=True)
            extent = studio.gui.add_number("fold_extent", 1.0, step=0.01, disabled=True)

        def each_step() -> None:
            folded.update_probe(sim.state, sim.step_count)
            snapshot = Snapshot(sim.make_context())
            angle.value = round(crease_angle(snapshot, "paper", line)["angle"], 2)
            extent.value = round(cloth_fold(snapshot.ctx, "paper", axis=0)[0], 2)

        studio.each_step(each_step)

    rss.view(sim, page=page, panels=("inspector", "overlay", "arm", "object", "cloth", "tune"))


if __name__ == "__main__":
    main()
