# SPDX-License-Identifier: Apache-2.0
"""Carton: fold a tray wall

A folding-carton tray blank on a block, its base held: press **Close**, drag the hand down onto a wall that runs past the block, and push it down and round the block's edge: it stays folded.

The blank is ``rss.carton``'s tray in folding carton board, 0.5 mm: a VBD sheet meshed from its die-line,
each crease a row of mesh edges that bends ten times easier than the board between and keeps a bend past
0.15 rad. Its base is pinned on the block, as a jig would hold it. The hand presses the -y wall 30 mm out
from the crease and sweeps it down to vertical: it turns 86 degrees, and two seconds after the hand has gone
it stands at 79, its crease keeping 78, and the other walls at 3 (``tests/test_fold_a_tray``). The **keeps**
readout is that crease.
"""

import math

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment


def main():
    at = (-0.14, -0.04)  # the blank's middle on the bench [m]: the -y wall is in the right hand's reach
    block = 0.07  # the block's height [m], so a wall folded down clears the bench
    # The right arm's joints [rad]: pointing down, the tool centre 50 mm over the -y wall's middle.
    # Solved with ``sim.arm("right").solve_q_full`` from a pointing-down seed.
    over_wall = (0.647, 1.741, 1.394, -1.225, 0.0, 0.647)
    left, right = get_embodiment("yam_bimanual").arms
    yam = rss.replace(get_embodiment("yam_bimanual"), arms=(left, rss.replace(right, home_q=over_wall)))
    hands = rss.ActuationSpec(target_ke=1500.0, target_kd=240.0, effort_limit=20.0)
    # Board this stiff needs VBD at 20 iterations, as the paper flap does; the hand pushes it one way.
    physics = rss.replace(
        rss.GENERIC, coupling=rss.coupler("one_way"), collision=rss.CollisionSpec(full_surface_contact=True)
    ).with_backend("*", rss.backend("vbd", iterations=20, rigid_body_contact_buffer_size=256))

    tray = rss.carton(name="tray", style="tray", board=rss.cartons.FOLDING_CARTON, at=(*at, block), fixed=True)
    scene = rss.Scene(cell="bench_0728", physics=physics, name="carton")
    scene.add(rss.robot(yam, gripper=hands))
    x, y, _ = tray.size  # the base's footprint: the walls run past the block's edges
    scene.add(rss.slab(name="block", half=(x / 2, y / 2, block / 2), at=(*at, block / 2), color=(0.35, 0.37, 0.4)))
    scene.add(tray)
    sim = scene.build(control_mode="joint_position")
    blank = sim["tray"].entity

    def page(studio):
        with studio.gui.add_folder("Tray", order=-1.0):
            angle = studio.gui.add_number("wall_yneg (deg)", 0.0, step=0.1, disabled=True)
            kept = studio.gui.add_number("wall_yneg keeps (deg)", 0.0, step=0.1, disabled=True)

        def each_step() -> None:
            angle.value = round(math.degrees(blank.fold_angle("wall_yneg")), 1)
            kept.value = round(math.degrees(blank.kept_angle("wall_yneg")), 1)

        studio.each_step(each_step)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
