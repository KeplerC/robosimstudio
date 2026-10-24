# SPDX-License-Identifier: Apache-2.0
"""Penalty kick

A ball, a spot and a goal on the lab floor: put the ball down in front of a foot with the mouse, and the duck kicks it in.

The kick is one of Pollen's own one-shot policies, trained blind to the ball:
it expects the ball 90 mm ahead of the trunk and 42 mm out to the kicking
side, swings, and stands back up on its own. So the aim is yours: click the
ball, drag its handle onto the spot in front of the left foot, **Release
all**, and press **Kick left**. **Forward roll** is another of its tricks.
Needs ``[microduck]``.
"""

import numpy as np
from duck import DUCK_200HZ, DuckSpec, frame
from props import goal

import robosimstudio as rss


def main() -> None:
    scene = rss.Scene(cell="void", physics=DUCK_200HZ, name="kick")
    duck = scene.add(DuckSpec(at=(0.0, 0.0)))
    ball = scene.add(
        rss.sphere(
            name="ball", radius=0.035, at=(0.25, -0.25, 0.035), mass=0.015, friction=0.8, color=(0.98, 0.45, 0.1)
        )
    )
    scene.add_all(goal(at=(1.0, 0.0), width=0.8, height=0.3))
    sim = scene.build()

    def page(studio):
        frame(studio, eye=(-0.4, -0.55, 0.32), look_at=(0.45, 0.05, 0.08))
        gui = studio.gui
        with gui.add_folder("Tricks"):
            gui.add_button("Kick left").on_click(lambda _: duck.trick("kick_left"))
            gui.add_button("Kick right").on_click(lambda _: duck.trick("kick_right"))
            gui.add_button("Forward roll").on_click(lambda _: duck.trick("roulade"))
            where = gui.add_text("left spot", "", disabled=True)
        with gui.add_folder("Drive"):
            for label, twist in {
                "Forward": (0.35, 0, 0),
                "Turn left": (0, 0, 1.0),
                "Turn right": (0, 0, -1.0),
                "Stop": (0, 0, 0),
            }.items():
                gui.add_button(label).on_click(lambda _, t=twist: duck.walk(*t))

        def each():
            p = duck.kick_spot("left")
            where.value = (
                f"({p[0]:.2f}, {p[1]:.2f}); ball {1000 * np.linalg.norm(ball.position[:2] - p[:2]):.0f} mm off"
            )

        studio.each_step(each)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
