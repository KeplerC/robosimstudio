# SPDX-License-Identifier: Apache-2.0
"""Paper: fold a sheet

Bring each hand in along the edge of the A4 sheet that runs past the box, from its own corner, with **Grip amount** 0.79, and press **Close**: the fingers stall on the sheet at the grip's 10 N cap. Carry the edge up and over, each hand turning with the flap to 88 degrees and no further, and press **Open** 100 mm over the far half: the flap falls onto it. Then, at **Grip amount** 0.6, run each hand's lower finger along its half of the fold, the sheet pushing it up its slide against the same 10 N cap: the sheet stays folded in half.

In 8 drives of the take's path (``probe2.py``, GPUs 3 and 5) the arms, which collide, came no closer than 79.3 mm
and kept 9.0 mm on their own sides; each hand turned 88.2 degrees holding the edge, its pads 5.0 mm apart at most,
and a finger pushed 7.4 N holding and 13.1 N pressing (95th percentile), but over 15 N for a step in 3: 15.6 to 17.2
at a touch, 67.9 once as an open hand slid off the fallen edge. The edges ended 2.4 mm apart (6.0 at the worst
corner) and the fold kept 2.89 to 2.90 rad once the hands had gone, unchanged 3 s later, ``fold_extent`` 0.50.
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
    # Copy paper at 20 VBD iterations: 6.1 cm bending length on the drape bench (PAPER's 30 reads 9.7 here), and
    # a hinge bent 0.2 rad past its rest keeps the rest, so a fold pressed flat keeps up to 2.94 rad.
    sheet = rss.replace(PAPER, edge_ke=6.0, yield_angle=0.2)
    # The sheet's push fed back along each finger's slide: a hand closes until it pushes back with its cap.
    # 20 iterations, not 5: at 5 a resting sheet pressed the box with 11 N, friction on that held it like glue
    # and lifting it pried the capped jaws; at 20, 2.6 N. The two halves of a fold lie 1 mm apart, not 2.
    coupling = rss.coupler("one_way", contact_forces=True, slide_feedback=True)
    physics = rss.replace(rss.GENERIC, coupling=coupling, collision=rss.CollisionSpec(full_surface_contact=True))
    vbd = {"iterations": 20, "friction_epsilon": 1.0e-3, "rigid_body_contact_buffer_size": 256}
    layers = {"particle_self_contact_radius": 1.0e-3, "particle_self_contact_margin": 1.0e-3}
    physics = physics.with_backend("*", rss.backend("vbd", **vbd, **layers))
    # The arms collide. One motor drives a hand's two fingers, capped at 10 N: the most either finger pushes.
    yam = rss.replace(get_embodiment("yam_bimanual"), cross_arm_collisions=True)
    hands = rss.ActuationSpec(target_ke=1.0e5, target_kd=20.0, effort_limit=10.0, joint_damping=200.0)

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
