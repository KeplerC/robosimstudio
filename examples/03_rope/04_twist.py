# SPDX-License-Identifier: Apache-2.0
"""A cable that carries a twist

Two cables hanging from a clamp, one twisting ten times more easily than it bends and one ten times less: roll a free end a turn and let go, and the stiff one springs the roll back out while the soft one keeps it.

``twist_ke`` splits Newton's rod joint from its bend: left at ``None`` a cable
twists as stiffly as it bends, which is the isotropic pairing the builder
makes. A data cable's jacket rolls far more easily than it bends; a stranded
steel cable the other way round. The page reads each cable's ``twist_turns``
over its middle third -- the roll between consecutive capsule frames with the
bend taken out, so a cable bent into a hook with no roll reads 0. Rolled a
full turn at the tip, both carry the same twist in the middle (a clamped cable
twists uniformly); let go, the stiff one unwinds in a fraction of a second and
the soft one over seconds, the time being ``twist_kd / twist_ke``. They hang
because a level cantilever at this bend stiffness droops to the bench. Click a
capsule near a free end and drag its handle round to roll it.
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
        readouts = {
            name: studio.gui.add_number(f"{name}: mid-cable twist (turns)", 0.0, disabled=True) for name in cables
        }

        def read():
            body_q = np.asarray(sim.pipeline.snapshot.body_q)
            for name, cable in cables.items():
                bodies = cable.bodies
                middle = bodies[len(bodies) // 3 : 2 * len(bodies) // 3 + 1]
                readouts[name].value = round(twist_turns(body_q[middle, 3:7]), 3)

        studio.each_step(read)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
