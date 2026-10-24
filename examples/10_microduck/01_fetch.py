# SPDX-License-Identifier: Apache-2.0
"""Fetch a ball

The duck on the lab bench, a ball and a bowl: feed it the ball with the mouse, and it carries it over and drops it in.

Press **Open beak**, click the ball and drag its handle up into the open
beak, press **Bite** and **Release all**: the ball is held by the beak's
two pads and nothing else. Then drive it to the bowl with the buttons and
press **Drop it**, or press **Carry it to the bowl** and it steers there
itself, stopping short of the rim. Dropping is what a duck does: beak open,
head tipped down, and the ball rolls out over the lip. Needs ``[microduck]``.
"""

from duck import DUCK_200HZ, DuckSpec, frame
from props import ball, bowl

import robosimstudio as rss

FORWARD, TURN = 0.35, 1.0  # what the drive buttons command [m/s, rad/s]


def main() -> None:
    scene = rss.Scene(cell="bench_075", physics=DUCK_200HZ, name="fetch")
    duck = scene.add(DuckSpec(at=(0.0, -0.40), yaw=1.5708, on="surface"))  # on the bench, facing down its length
    scene.add(ball(at=(0.12, -0.15)))
    target = scene.add(bowl(at=(0.0, 0.25)))
    sim = scene.build()

    def page(studio):
        frame(studio, eye=(-0.12, -0.08, 1.22), look_at=(0.62, -0.08, 0.80))
        gui = studio.gui
        with gui.add_folder("Beak"):
            gui.add_button("Open beak").on_click(lambda _: duck.mouth(1.0))
            gui.add_button("Bite").on_click(lambda _: duck.mouth(0.0))
            gui.add_button("Drop it").on_click(lambda _: duck.drop())
            gui.add_button("Carry it to the bowl").on_click(lambda _: duck.carry_to(target.position, short=0.06))
        with gui.add_folder("Drive"):
            for label, twist in {
                "Forward": (FORWARD, 0, 0),
                "Back": (-FORWARD, 0, 0),
                "Turn left": (0, 0, TURN),
                "Turn right": (0, 0, -TURN),
                "Stop": (0, 0, 0),
            }.items():
                gui.add_button(label).on_click(lambda _, t=twist: duck.walk(*t))

    rss.view(sim, page=page, scenery=True)


if __name__ == "__main__":
    main()
