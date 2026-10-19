# SPDX-License-Identifier: Apache-2.0
"""A cable carries a twist

Two cables hanging from a clamp, one twisting ten times more easily than it bends and one ten times less: drag **tip roll** to a turn, untick **hold the tips**, and the stiff one springs the roll back out while the soft one keeps it.

``twist_ke`` splits Newton's rod joint from its bend: left at ``None`` a cable
twists as stiffly as it bends; a data cable's jacket rolls far more easily
than it bends, a steel cable the other way round. Each free end is held in a
hard latch, as a hand holds it (``cable.entity.roll``), and the slider rolls
it about its own axis at 0.1 rad a step (a click's handle cannot: its wrench
is bounded by the capsule's own inertia). The page reads each cable's
``twist_turns`` over its middle third, the roll between consecutive frames
with the bend taken out. A clamped cable twists uniformly: one turn at the tip
puts 0.348 turns (8 of 23 joints) in the middle of both; let go, the stiff one
reads 0.000 within 60 steps, the soft one 0.33 then and 0.27 after four seconds.
"""

import numpy as np

import robosimstudio as rss
from robosimstudio_materials.rod.measures import twist_turns


def main():
    length = 0.4  # [m]
    radius = 0.006  # [m]
    segments = 24
    bend_ke = 0.04  # [N*m/rad], the shipped cable's
    top = 0.45  # the clamp, over the bench [m]: the cables hang 50 mm clear of it

    scene = rss.Scene(cell="bench_075", name="cable_twist")
    scene.add(rss.robot("yam_bimanual"))
    cables = {}
    for name, x, twist_ke in (("soft_twist", -0.10, bend_ke / 10), ("stiff_twist", 0.10, bend_ke * 10)):
        cables[name] = scene.add(
            rss.rod(
                name=name,
                length=length,
                radius=radius,
                segments=segments,
                at=(x, 0.0, top),
                along=(0.0, 0.0, -1.0),
                anchors=("start",),  # clamped: position, tangent and twist held at the top
                bend_ke=bend_ke,
                twist_ke=twist_ke,
            )
        )
    sim = scene.build(control_mode="joint_position")

    def page(studio):
        roll = studio.gui.add_slider("tip roll (turns)", min=0.0, max=2.0, step=0.05, initial_value=0.0)
        held = studio.gui.add_checkbox("hold the tips", initial_value=True)
        readouts = {n: studio.gui.add_number(f"{n}: mid-cable twist (turns)", 0.0, disabled=True) for n in cables}

        def drive():
            for name, cable in cables.items():
                if held.value:  # the latch: roll the held tip towards the slider, or let it go
                    cable.entity.roll(sim.pipeline, 2 * np.pi * roll.value)
                else:
                    cable.entity.pin(sim.pipeline, on=False)
                middle = cable.bodies[len(cable.bodies) // 3 : 2 * len(cable.bodies) // 3 + 1]
                readouts[name].value = round(twist_turns(np.asarray(sim.pipeline.snapshot.body_q)[middle, 3:7]), 3)

        studio.each_step(drive)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
