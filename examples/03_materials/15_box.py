# SPDX-License-Identifier: Apache-2.0
"""Box: cut the tape and open a flap

A taped corrugated box under the right hand: press **Cut tape** and the two big flaps spring up a little; with the jaws open, drag the hand down into the seam, hook a finger under one flap, swing it up and over, and back off: it stays open.

The box is ``rss.carton``'s regular slotted container in C-flute board, 200 x 150 x 80 mm with 1 kg inside,
on the panels model: rigid plates on crease hinges that keep a bend past 0.25 rad, on MuJoCo with the arm
(``rss.CARTON_60HZ``). Cut, the big flaps spring to 14.8 degrees, about their crease's 14.3 degree yield, and the
small flaps under them stay at 2.6. The hand's open jaws reach into the 6 mm seam, its +y finger lifts the -y flap
56 mm out from the crease to 141 degrees, and two seconds after the hand has gone the flap keeps 127, the other
at 15.9 and the box 0.4 mm from where it stood (``tests/test_open_a_box``). The **keeps** readout is the angle a
crease holds when nothing pushes it.
"""

import math

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment


def main():
    at = (-0.14, -0.06, 0.0)  # the box's underside, middle (bench frame) [m]: every flap is in the right hand's reach
    # The right arm's joints [rad]: pointing down, the tool centre 160 mm over the middle of the box's underside.
    # Solved with ``sim.arm("right").solve_q_full`` from a pointing-down seed.
    over_box = (0.782, 1.893, 1.744, -1.423, 0.0, 0.782)
    left, right = get_embodiment("yam_bimanual").arms
    yam = rss.replace(get_embodiment("yam_bimanual"), arms=(left, rss.replace(right, home_q=over_box)))
    # A grip that stalls on the board rather than pinching through it.
    hands = rss.ActuationSpec(target_ke=1500.0, target_kd=240.0, effort_limit=12.0)

    scene = rss.Scene(cell="bench_0728", physics=rss.CARTON_60HZ, name="box")
    scene.add(rss.robot(yam, gripper=hands))
    scene.add(rss.carton(name="box", at=at, contents=1.0))
    sim = scene.build(control_mode="joint_position")
    box = sim["box"].entity
    shown = ("major_neg", "major_pos")

    def page(studio):
        cut = {"asked": False}
        with studio.gui.add_folder("Box", order=-1.0):
            angles = {n: studio.gui.add_number(f"{n} (deg)", 0.0, step=0.1, disabled=True) for n in shown}
            kept = {n: studio.gui.add_number(f"{n} keeps (deg)", 0.0, step=0.1, disabled=True) for n in shown}
            studio.gui.add_button("Cut tape").on_click(lambda _: cut.update(asked=True))

        def each_step() -> None:
            if cut["asked"]:
                cut["asked"] = False
                box.cut_tape()
            now, keeps = box.angles(), box.references()
            for n in shown:
                angles[n].value = round(math.degrees(now[box.index(n)]), 1)
                kept[n].value = round(math.degrees(keeps[box.index(n)]), 1)

        studio.each_step(each_step)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
