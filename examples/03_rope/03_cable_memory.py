# SPDX-License-Identifier: Apache-2.0
"""A cable that keeps its bend

The between-posts cable, clamped at one end only and given plastic memory: drag its free end round the far post and let go, and it stays wrapped.

``dahl_eps_max`` turns on Newton's Dahl bending hysteresis for the capsule
chain: each cable joint carries a friction moment of ``bend_ke * dahl_eps_max``
that builds up over ``dahl_tau`` radians of bending, so a bend put in past that
is held rather than sprung back. The same cable at 0 is elastic and springs
straight. The solver decides at construction whether to run the friction at
all, so the two plastic sliders in the cable's Parameters are live here and
inert on a cable built at 0. The page reads the cable's straightness: 1.0 is a
straight line, and what it reads after the hand lets go is what the cable kept.
"""

import numpy as np

import robosimstudio as rss
from robosimstudio_materials.rod.measures import straightness


def main():
    span = 0.5  # from post to post [m]
    radius = 0.008  # of the cable [m]
    segments = 24
    post_height = 0.16  # [m]
    eps_max = 2.0  # the Dahl envelope [rad]; Newton's cable-bundle example uses this and the 0.1 below
    tau = 0.1  # the Dahl transition length [rad]

    scene = rss.Scene(cell="bench_075", name="cable_memory")
    scene.add(rss.robot("yam_bimanual"))
    for name, y in (("post_start", -span / 2), ("post_end", span / 2)):
        scene.add(rss.post(name=name, radius=0.02, half_height=post_height / 2, at=(0.0, y, post_height / 2)))
    cable = scene.add(
        rss.rod(
            name="cable",
            length=span,
            radius=radius,
            segments=segments,
            at=(0.0, -span / 2, post_height + radius + 0.002),  # 2 mm clear of the post's top
            along=(0.0, 1.0, 0.0),
            anchors=("start",),  # clamped on the near post; the far end is free to wrap the other
            dahl_eps_max=eps_max,
            dahl_tau=tau,
        )
    )
    sim = scene.build(control_mode="joint_position")

    def page(studio):
        out = studio.gui.add_number("straightness (1 = straight)", 1.0, disabled=True)

        def read():
            points = np.asarray(sim.pipeline.snapshot.body_q)[cable.bodies, :3]
            out.value = round(straightness(points), 3)

        studio.each_step(read)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
