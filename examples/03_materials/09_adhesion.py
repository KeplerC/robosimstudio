# SPDX-License-Identifier: Apache-2.0
"""Adhesion: stick a label

A loose paper label lies on the bench beside a box, its end in a blue pull tab under the right hand: set **Grip amount** to 0.6, drag the hand down onto the tab and press **Close**; drag it up, over the box's far end and down toward its near end, and the label lies down; set the grip back to 0.6 and lift the hand. Then press **Close** and rub the shut fingertips along the label, four strokes 10 mm apart: it sticks where they pressed, and **bonded_fraction** passes 0.8.

The label carries ipc-toolkit's adhesion law (``rss.tapes.LABEL``, ``robosimstudio_materials/tape.py``):
a bond forms where the sheet is pressed onto the box harder than 5 kPa, pulls hardest, 4 kPa, at a
0.5 mm lift and lets go at 1 mm. The grasp is contact, the pads on the rigid tab. Laid, the label held
1-6 of its 126 bonds; in twelve drives the four strokes made 85-93% of them, and in eight the tab drawn
straight up 120 mm peeled all but 0-3 at 2.15-2.38 N on **pull on the tab**; the box moved 0.7 mm at most, as much as
it does with no bond. With **Adhesive off** the strokes made no bond (``tests/test_stick_a_label``). Rub lightly,
the gizmo 3 mm under the label: at 5 mm the fingertips dragged it along in two drives of three. 35 ms a control
step (RTX PRO 6000).
"""

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment


def main():
    cell = "bench_0728"
    box_half = (0.05, 0.06, 0.025)  # [m]; free on the bench, 0.5 kg
    box_at = (0.0, -0.08, box_half[2])  # its centre, in the bench's work-surface frame [m]
    # The label on the bench beside the box, its tab end nearest it, turned half a turn so it runs on
    # toward the right arm. Its first 12 mm past the fold carry no adhesive, like a real lift tab.
    corner = (0.02, -0.16, 0.0005)
    half_turn = (0.0, 0.0, 1.0, 0.0)
    # The right arm's joints [rad]: pointing down, jaws closing along y, the tool centre 100 mm over
    # the bench above the tab (``sim.arm("right").ik.solve_q_full``).
    over_tab = (0.3322, 2.1402, 1.8639, -1.2957, 0.0003, 0.3322)
    grip = rss.ActuationSpec(target_ke=1500.0, target_kd=60.0, effort_limit=12.0)  # [N] per finger

    yam = get_embodiment("yam_bimanual")
    left, right = yam.arms
    yam = rss.replace(yam, arms=(left, rss.replace(right, home_q=over_tab)))

    scene = rss.Scene(cell=cell, physics=rss.TAPE_60HZ, name="stick_a_label")
    scene.add(rss.robot(yam, gripper=grip))
    scene.add(rss.box(name="box", half=box_half, at=box_at, mass=0.5, color=(0.72, 0.56, 0.38)))
    label = rss.tape(
        name="label", on="box", at=corner, quat=half_turn, material=rss.tapes.LABEL, stuck=False, dry=0.012
    )
    scene.add(label)
    sim = scene.build(control_mode="joint_position")
    label = sim["label"].entity

    def page(studio):
        with studio.gui.add_folder("Label", order=-1.0):  # first in the sidebar, above the drive panels
            off = studio.gui.add_checkbox("Adhesive off", initial_value=False)
            bonded = studio.gui.add_number("bonded_fraction", 0.0, step=0.001, disabled=True)
            held = studio.gui.add_number("bonds holding", 0, step=1, disabled=True)
            pull = studio.gui.add_number("pull on the tab (N)", 0.0, step=0.01, disabled=True)

        def each_step() -> None:
            held.value = label.held()  # read once: each read copies the bonds' flags off the device
            bonded.value = round(held.value / len(label.bonds), 3)
            pull.value = round(label.pull(sim.state), 2)

        strength = label.spec.material.strength  # [Pa]; 0 makes no bond and lets every bond go
        off.on_update(lambda _: label.write_adhesive("strength", 0.0 if off.value else strength))
        studio.each_step(each_step)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
