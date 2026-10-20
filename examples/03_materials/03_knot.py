# SPDX-License-Identifier: Apache-2.0
"""Knot: pull it tight

A loose overhand knot in a rope lies on the bench between the two hands: drag each hand down onto its tail, slide its **Grip amount** to 0.90, drag both hands up 120 mm, then each out 85 mm, and the knot draws up and stays a knot.

A knot is a rope crossing itself, held apart where it crosses by its own contact
(``self_contact``: SuperDex tests the rope's skin against itself). The pads hold the
tails by friction, nothing pinned. Pulled to 490 mm across, the knot gives back 59
of its 361 mm of rope and keeps its three crossings, the closest strands 0.29 mm
into each other on any step (``k_knot.py``); with self-contact off one strand goes
straight through another, a whole 12 mm, and one crossing is left
(``tests/test_knot_pull``, GPU 1). BDF2 drew it to 239 mm but let a tail slide 88 mm
through a hand (``probe_c3``). 20 to 45 ms a control step. Needs ``[superdex]``.
"""

import numpy as np

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment
from robosimstudio_materials.rod.knot import crossings, knot_length, overhand, self_clearance


def main():
    radius = 0.006  # a 12 mm rope [m]
    nodes = overhand(at=(0.03, 0.0), size=0.10, radius=radius, tail=0.12, spacing=0.008)  # 100 mm across, 88 nodes
    # Each arm [rad]: tool centre 100 mm over its tail 40 mm in from the end, pointing down, jaws across (solve_q_full).
    right = (0.5459, 2.2689, 2.0725, -1.3745, 0.0, -1.0249)
    over_tail = {"right": right, "left": (-right[0], *right[1:5], -right[5])}  # the left is its mirror

    engine = rss.backend("superdex", substeps=2, feedback="off")  # the fingers drive the rope by pose
    physics = rss.replace(rss.GENERIC, backends=(("rod", engine), *rss.GENERIC.backends))
    yam = get_embodiment("yam_bimanual")
    yam = rss.replace(yam, arms=tuple(rss.replace(a, home_q=over_tail[a.name]) for a in yam.arms))

    scene = rss.Scene(cell="bench_075", physics=physics, name="knot")
    scene.add(rss.robot(yam))
    rope = scene.add(
        rss.rod(
            name="rope",
            radius=radius,
            nodes=tuple(map(tuple, nodes)),
            self_contact=True,  # off, the pulled rope passes through itself and comes out straight
            young_modulus=1.0e5,  # a rope bends easily: 2e6 Pa was a hose the hands could not draw up
            axial_stiffness=300.0,  # [N] and barely stretches
            mu_cable=0.25,  # friction sqrt(0.25) = 0.5, between strands and at the pads alike
            friction_falloff=1.0e-3,  # [m/s]; nothing is static above it
            contact_penalty=4.0e8,  # five times the default, so the pads hold five times as hard
            skin_samples="P1Q6",  # at the engine's default a pad fell between samples and pressed alone
        )
    )
    sim = scene.build(control_mode="joint_position")

    def page(studio):
        cross = studio.gui.add_number("crossings (3: still a knot)", 3, step=1, disabled=True)
        tied = studio.gui.add_number("rope in the knot (mm)", 0.0, step=0.1, disabled=True)
        clear = studio.gui.add_number("closest strands, surface to surface (mm)", 0.0, step=0.1, disabled=True)

        def read() -> None:
            points = np.asarray(rope.particle_q, dtype=np.float64)
            cross.value = crossings(points)
            tied.value = round(1e3 * knot_length(points), 1)
            clear.value = round(1e3 * self_clearance(points, radius), 1)

        studio.each_step(read)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
