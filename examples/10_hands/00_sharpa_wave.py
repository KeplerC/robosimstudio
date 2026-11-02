# SPDX-License-Identifier: Apache-2.0
"""Sharpa Wave hands

Two UR5e arms wearing Sharpa Wave hands, 22 joints each: the studio's **Close** makes a thumb-index pinch, and a routine plays the other postures.

``ur5e_sharpa_bimanual`` is a catalog robot like any other: drag a hand's gizmo
down onto the cube, press **Close** and lift, and the pinch holds it by friction
(``rss.MUJOCO_500HZ`` puts the hands and the cube in one engine). What a jaw
cannot say goes through the hand's named postures: **Run** cycles the left hand
through a fist, a pointing finger and the pinch while the right hand waits, each
posture written joint by joint with ``gripper.set_joint_targets``. The hand's
joints, gains and postures: ``robosimstudio/robots/grippers/sharpa_wave.py``.
"""

import robosimstudio as rss


def routine(sim):
    """The left hand through its postures, one every two seconds, then open."""
    hz = sim.request.physics.control_hz
    gripper = sim.gripper("left")
    postures = sim.scene.embodiment.hands["left"].postures
    for name in ("fist", "open", "point", "open", "pinch", "open"):
        gripper.set_joint_targets(postures[name])
        for _ in range(int(2 * hz)):
            yield
    gripper.set_joint_targets(None)  # back to the Grip slider


def main() -> None:
    scene = rss.Scene(cell="bench_075", name="sharpa_wave", physics=rss.MUJOCO_500HZ)
    # Under the right hand's pinch point at home: on the middle line of the
    # bench, 200 mm to the robot's right.
    scene.add(rss.box(name="cube", half=(0.0125, 0.0125, 0.0125), at=(0.0, -0.2, 0.0125), mass=0.05))
    scene.add(rss.robot("ur5e_sharpa_bimanual"))

    sim = scene.build(control_mode="joint_position")
    sim.run(100).sync()
    for side in ("left", "right"):
        tool = sim.arm(side).ee_pose(sim.state)[0]
        print(f"{side} pinch point at ({tool[0]:.3f}, {tool[1]:.3f}, {tool[2]:.3f}) m, 22 finger joints")

    rss.view(sim, routines={"postures": routine})


if __name__ == "__main__":
    main()
