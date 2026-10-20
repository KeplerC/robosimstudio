# SPDX-License-Identifier: Apache-2.0
"""Tape: peel a label

A paper label stuck on a box, its end in a blue pull tab under the right hand: drag the hand down until its open jaws straddle the tab, press **Close**, drag it up 30 mm, then back toward the arm 20 mm and up 10 mm, seven times, and the label peels off the box row by row.

The grasp is contact: the pads squeeze the tab and nothing pins it to the hand. The label is a cloth
held to the box by Newton 1.7 particle-body attachments, one per particle, and a bond lets go once the
label has lifted 1 mm off the box, so the peel front travels with the hand. In 24 drives all 144 bonds
let go, the tab stayed within 3.4 mm of where the pads took it and the free box moved 1.7 mm at most
(``peel_probe.py``, two drives to a GPU). The jaws are damped to reach the tab in 0.4 s: shutting at
8 mm a step they knocked it aside and closed on nothing in 3 of 116 closes (``peel_probe.py``,
``peel_close.py``, the round-3 review). Tick **Never let go** and no bond lets go: the pull drags the
box until the tab slides out of the pads (10 drives of 24), or, where the pads hold, drags the whole
box 46 mm by its label (14 of 24). The tab is rigid, because the pads cannot hold a cloth tab under
the coupling the attachments need, and weighs 30 g (``robosimstudio_materials/tape.py``). The pads
hold it with 5.8 N; only Close's first push (71 N) meets the 12 N finger cap. The **Label** folder
reads ``peeled_fraction`` and the peel front. 38 ms a control step through the page (RTX PRO 6000).
"""

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment


def main():
    cell = "bench_0728"
    box_half = (0.05, 0.06, 0.025)  # [m]; free on the bench, 0.5 kg
    box_at = (0.0, -0.08, box_half[2])  # its centre, in the bench's work-surface frame [m]
    top = 2 * box_half[2]
    # The label's tab end, a particle radius over the box top; turned half a turn so it runs toward -y,
    # back toward the right arm, which is the way the hand pulls it.
    corner = (0.02, -0.025, top + 0.0005)
    half_turn = (0.0, 0.0, 1.0, 0.0)
    # The right arm's joints [rad]: pointing down, jaws closing along y, the tool centre 100 mm over
    # the bench above the tab (``sim.arm("right").solve_q_full``).
    over_tab = (0.604, 2.485, 2.452, -1.539, 0.0, 0.604)
    # [N] per finger. Damped so the jaws reach the tab in 0.4 s: at kd 60 they hit it at 8 mm a step.
    grip = rss.ActuationSpec(target_ke=1500.0, target_kd=240.0, effort_limit=12.0)

    yam = get_embodiment("yam_bimanual")
    left, right = yam.arms
    yam = rss.replace(yam, arms=(left, rss.replace(right, home_q=over_tab)))

    scene = rss.Scene(cell=cell, physics=rss.TAPE_60HZ, name="peel_a_label")
    scene.add(rss.robot(yam, gripper=grip))
    scene.add(rss.box(name="box", half=box_half, at=box_at, mass=0.5, color=(0.72, 0.56, 0.38)))
    scene.add(rss.tape(name="label", on="box", at=corner, quat=half_turn, material=rss.tapes.PAPER_LABEL))
    sim = scene.build(control_mode="joint_position")
    label = sim["label"].entity

    def page(studio):
        with studio.gui.add_folder("Label", order=-1.0):  # first in the sidebar, above the drive panels
            never = studio.gui.add_checkbox("Never let go", initial_value=False)
            held = studio.gui.add_number("bonds holding", label.held(), step=1, disabled=True)
            front = studio.gui.add_number("peel front (mm)", 0.0, step=0.1, disabled=True)
            peeled = studio.gui.add_number("peeled_fraction", 0.0, step=0.001, disabled=True)

        def each_step() -> None:
            peeled.value = round(label.peeled(), 3)
            front.value = round(1000.0 * label.peel_front(), 1)
            held.value = label.held()

        opening = label.spec.material.break_opening  # [m]; 0 is a label that never lets go
        never.on_update(lambda _: label.write_break_opening(0.0 if never.value else opening))
        studio.each_step(each_step)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
