# SPDX-License-Identifier: Apache-2.0
"""Walk the duck

The Microduck on the lab floor, on its own walking policy: drive it from the page, turn its head, pick it up with the mouse.

The robot is Pollen Robotics' open-source biped on a free joint, its fifteen
servos position-controlled with the gains fitted to its XL330s, and once per
control step it runs the policy it walks with on the real robot -- the official
``velstand`` network, trained in MuJoCo Warp at 50 Hz. Nothing holds it up but
its feet. Click any part of it and drag: it hangs from the handle, kicks, and
finds its feet again when you put it down. Needs the ``[microduck]`` extra, for
its ONNX policies; the robot and the policies are downloaded on first run.
"""

from duck import DUCK_200HZ, DuckSpec, frame

import robosimstudio as rss

CELL = "void"
FORWARD, SIDEWAYS, TURN = 0.35, 0.25, 1.0  # what the buttons command [m/s, m/s, rad/s]


def main() -> None:
    scene = rss.Scene(cell=CELL, physics=DUCK_200HZ, name="microduck")
    duck = scene.add(DuckSpec(at=(0.0, 0.0)))
    scene.add(rss.sphere(name="ball", radius=0.035, at=(0.35, 0.10, 0.035), mass=0.015, color=(0.95, 0.35, 0.2)))
    sim = scene.build()

    def page(studio):
        frame(studio, eye=(0.3, 0.75, 0.32), look_at=(0.3, -0.05, 0.1))
        gui = studio.gui
        with gui.add_folder("Drive"):
            moves = {
                "Forward": (FORWARD, 0, 0),
                "Back": (-FORWARD, 0, 0),
                "Turn left": (0, 0, TURN),
                "Turn right": (0, 0, -TURN),
                "Step left": (0, SIDEWAYS, 0),
                "Step right": (0, -SIDEWAYS, 0),
                "Stop": (0, 0, 0),
            }
            for label, twist in moves.items():
                gui.add_button(label).on_click(lambda _, t=twist: duck.walk(*t))
            mode = gui.add_text("policy", "walk", disabled=True)
        with gui.add_folder("Head and beak"):
            yaw = gui.add_slider("head yaw (rad)", min=-1.2, max=1.2, step=0.05, initial_value=0.0)
            pitch = gui.add_slider("head pitch (rad)", min=-0.8, max=0.8, step=0.05, initial_value=0.0)
            beak = gui.add_slider("beak open", min=0.0, max=1.0, step=0.05, initial_value=0.0)
            for s in (yaw, pitch):
                s.on_update(lambda _: duck.look(head_pitch=pitch.value, head_yaw=yaw.value))
            beak.on_update(lambda _: duck.mouth(beak.value))
            gui.add_button("Peck the floor").on_click(lambda _: duck.peck())
            sit = gui.add_checkbox("Sit", False)
            sit.on_update(lambda _: duck.sit(sit.value))
        studio.each_step(lambda: setattr(mode, "value", duck.mode))

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
