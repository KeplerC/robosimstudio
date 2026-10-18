# SPDX-License-Identifier: Apache-2.0
"""Open a bag

A canvas bag lies with its mouth toward the right hand and its top layer's lip turned up: drag the left hand down onto the bag's closed end, drag the right hand down until its open jaws straddle the lip, press **Close**, and drag it up and back over the bag, and the mouth opens.

Nothing is pinned: the lip comes up because the pads hold it, and the left hand's closed fingertips
keep the bag from sliding after it (its closed end slid 15-22 mm under them, 36 mm without). Over
eight drives the mouth opened 54-91 mm, the pads closed 20-30 mm over the bench and up to 4 mm
either side of the lip (``p9_drive.py``, GPUs 3 and 5), and 33-74 mm in four filmed takes; closed
15 mm over it the pads take both layers and drag the bag. The bag is one SuperDex shell that collides with itself: its layers rest 3.9 mm
apart, a contact radius, and stayed at least 2.6 mm apart while the hand opened it. Set
``self_contact`` to False and the same gesture shows them through each other: -0.24 mm at rest,
-5.2 mm held open. The grip is stiffer than the YAM's own servo, 8 N a finger: at its own gains the
pads pressed the lip with 0.1 N and it slid out. The **Bag** folder reads ``mouth_opening``, the
widest span across the rim and the area it encloses, and ``layer_gap``. About 9 ms per control
step at rest and 40-60 ms while the pads press the cloth (RTX PRO 6000). Needs the ``[superdex]``
extra (Python 3.12).
"""

import numpy as np

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment
from robosimstudio_materials.bag import layer_gap, mouth_opening


def main():
    cell = "bench_0728"
    self_contact = True  # False: the same bag whose two layers pass through each other
    size = (0.24, 0.20)  # closed end to mouth, and across [m]; 10 mm cells
    turn = (0.0, 0.0, -np.sqrt(0.5), np.sqrt(0.5))  # the mouth toward -y, the right hand's side
    # The arms' joints [rad], both pointing down, jaws closing along y, tool centres 100 mm over the
    # bench: the right over the lip, the left over the closed end. ``sim.arm(...).solve_q_full``.
    over_lip = (0.437, 2.222, 1.994, -1.344, 0.0, 0.437)
    over_end = (-0.531, 2.336, 2.187, -1.421, 0.0, -0.531)
    grip = rss.ActuationSpec(target_ke=1500.0, target_kd=60.0, effort_limit=8.0)  # [N] per finger

    yam = get_embodiment("yam_bimanual")
    left, right = yam.arms
    yam = rss.replace(yam, arms=(rss.replace(left, home_q=over_end), rss.replace(right, home_q=over_lip)))

    scene = rss.Scene(cell=cell, physics=rss.BAG_60HZ, name="open_a_bag")
    scene.add(rss.robot(yam, gripper=grip))
    scene.add(rss.bag(name="bag", size=size, quat=turn, self_contact=self_contact))
    sim = scene.build(control_mode="joint_position")
    pouch = sim["bag"]

    def page(studio):
        with studio.gui.add_folder("Bag"):
            height = studio.gui.add_number("mouth height (mm)", 0.0, step=0.1, disabled=True)
            area = studio.gui.add_number("mouth area (cm²)", 0.0, step=0.1, disabled=True)
            gap = studio.gui.add_number("least layer gap (mm)", 0.0, step=0.01, disabled=True)

        def each_step() -> None:
            points = pouch.particle_q
            mouth = mouth_opening(points, pouch.entity)
            height.value = round(1000.0 * mouth["height"], 1)
            area.value = round(1.0e4 * mouth["area"], 1)
            gap.value = round(1000.0 * min(layer_gap(points, pouch.entity), 1.0), 2)

        studio.each_step(each_step)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
