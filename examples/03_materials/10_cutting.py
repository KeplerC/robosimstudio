# SPDX-License-Identifier: Apache-2.0
"""Cutting: cut the tofu

A block of firm tofu on a cutting board, and a kitchen knife standing on its edge in a rest under the right hand: drag the hand down onto the knife's handle, press **Close**, drag it up 70 mm, across to the tofu and down until the edge meets the board, and the tofu parts in two.

The grasp is contact: the pads close on the 20 mm handle at 20.4 mm and nothing pins it to the hand.
The tofu is a cage on the vendored walk-on-spheres cutter, its base held by the board; the cut starts
in the blade's plane, runs where the edge pushes past the blade's threshold (0.18 N across 60 mm of
tofu), and the knife feels that push. In three drives through the arm panel (``tests/test_cut_tofu``)
the tofu read 2 ``pieces`` with the edge on the board, and 1 piece, cut 88% through, with the hand
stopped 15 mm higher. Tick **Dull blade**, a blade a thousand times duller, and push the same way: the
push swings from step to step, up to 18-21 N on the way down and 3-19 N with the hand held low; the
knife tips 14-17 degrees in the pads and the arm stalls with the edge's lowest point 7-11 mm into the
tofu, which stays whole (13 drives, the test's and one without its pause). 13 ms a control
step cutting through the page, 9 ms idle (RTX PRO 6000).
"""

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment


def main():
    knife_y, tofu_y, blade_x = -0.31, -0.16, 0.04  # the rest's and the tofu's middles, and the blade's [m]
    # The right arm's joints [rad]: pointing down, jaws closing along y, the tool centre 100 mm over
    # the bench above the knife's handle (``sim.arm("right").solve_q_full``).
    over_knife = (0.0, 1.792, 1.376, -1.154, 0.0, 0.0)
    grip = rss.ActuationSpec(target_ke=1500.0, target_kd=60.0, effort_limit=12.0)  # [N] per finger

    yam = get_embodiment("yam_bimanual")
    yam = rss.replace(yam, arms=(yam.arms[0], rss.replace(yam.arms[1], home_q=over_knife)))
    # Standing on its edge, the handle's middle 40 mm up; the blade runs 120 mm along x from the handle.
    knife = rss.knife(name="knife", at=(blade_x - 0.105, knife_y, 0.04))
    walls, board = (0.05, 0.004, 0.012), (0.07, 0.08, 0.006)  # half-extents [m]; the walls 3 mm off the blade

    scene = rss.Scene(cell="bench_075", physics=rss.CUTTING_50HZ, name="cut_tofu")
    scene.add(rss.robot(yam, gripper=grip))
    scene.add(knife)
    for name, y in (("rest_a", knife_y - 0.00625), ("rest_b", knife_y + 0.00625)):
        scene.add(rss.slab(name=name, half=walls, at=(blade_x, y, walls[2]), color=(0.35, 0.25, 0.18)))
    scene.add(rss.slab(name="board", half=board, at=(blade_x, tofu_y, board[2]), color=(0.62, 0.45, 0.28)))
    at = (blade_x, tofu_y, 2 * board[2])  # the middle of its base, on the board
    scene.add(rss.cuttable(name="tofu", knife=knife, size=(0.06, 0.10, 0.04), at=at, material=rss.cuttables.TOFU))
    sim = scene.build(control_mode="joint_position")
    tofu = sim["tofu"].entity

    def page(studio):
        with studio.gui.add_folder("Tofu", order=-1.0):  # first in the sidebar, above the drive panels
            dull = studio.gui.add_checkbox("Dull blade", initial_value=False)
            pieces = studio.gui.add_number("pieces", 1, step=1, disabled=True)
            fraction = studio.gui.add_number("cut_fraction", 0.0, step=0.001, disabled=True)
            push = studio.gui.add_number("push on the edge (N)", 0.0, step=0.01, disabled=True)

        def each_step() -> None:
            pieces.value, fraction.value, push.value = tofu.pieces(), round(tofu.cut_fraction(), 3), round(tofu.push, 2)

        dull.on_update(lambda _: tofu.set_sharpness(0.001 if dull.value else 1.0))
        studio.each_step(each_step)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
