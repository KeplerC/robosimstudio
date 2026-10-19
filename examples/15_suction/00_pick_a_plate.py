# SPDX-License-Identifier: Apache-2.0
"""Pick up a plate

A plate lying flat under the right hand's suction cup and a square 200 mm toward the middle of the bench: drag the hand down onto the plate, tick **Vacuum** in the right arm's panel, lift, carry it over the square and untick.

The plate is 150 by 150 by 6 mm and 0.3 kg, flat on the bench, so a parallel
jaw has nothing to close across; ``yam_bimanual_suction`` carries a vacuum cup
on its right arm instead, its face 100 mm over the plate and pointing down at
home, and keeps the left hand's jaws. The cup seals when its face is within
2 mm of the plate and square to it within 15 degrees, and the arm's **Seal**
readout says sealed or not, the gap and the mass held. A hand dragged down
trails the gizmo by a few millimetres, and the panel puts the gizmo back on the
hand when a drag ends that close to it, so put the gizmo a little into the
plate. Tick **Vacuum** once the face is down: ticked on the way down, the cup
seals as soon as the face is within 2 mm and carries the plate about that far
under it.
The arm and the plate share one MuJoCo solver (``rss.MUJOCO_500HZ``), so the
seal is a weld switched on inside the captured graph and the arm carries the
plate's weight. Driven through the panel, the plate rides 198 mm with the cup
less than 1 mm off where it sealed and rests in the square with its top 6 mm
over the bench, at 3.5 ms per control step (``tests/test_suction_pick``, RTX
PRO 6000, GPU 5).
"""

import robosimstudio as rss


def main():
    cell = "bench_075"
    plate_at = (0.0, -0.31, 0.003)  # flat on the bench under the cup's home, in the bench's frame [m]
    plate_half = (0.075, 0.075, 0.003)  # 150 x 150 x 6 mm
    plate_mass = 0.3  # [kg]
    square = (-0.11, -0.22, 0.11, 0.0)  # x0, y0, x1, y1 [m]: 220 mm across, its centre 200 mm from the plate's

    scene = rss.Scene(cell=cell, name="pick_a_plate", physics=rss.MUJOCO_500HZ)
    scene.add(rss.robot("yam_bimanual_suction"))
    scene.add(rss.box(name="plate", half=plate_half, at=plate_at, mass=plate_mass, color=(0.88, 0.88, 0.85)))
    # The square, with a 10 mm height window: the plate counts as in it resting, not carried over it.
    scene.add(rss.RegionSpec(name="square", frame="", xy_ranges=square, z_range=(0.0, 0.01)))
    sim = scene.build(control_mode="joint_position")
    # The drive panels and ``regions``, which draws the square on the bench: a bare scene's page leaves it out.
    rss.view(sim, panels=("inspector", "overlay", "arm", "object", "regions"))


if __name__ == "__main__":
    main()
