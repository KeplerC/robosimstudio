# SPDX-License-Identifier: Apache-2.0
"""Scrub a stain

A towel with a stain along its middle under the right hand: press **Close**, drag the hand down onto the stain until **push** reads between 20 and 60 N, and drag it along the stain: only what it rubbed inside that band is wiped.

The push is what the towel puts on the hand's colliders: Newton 1.7's VBD contact force, read through
``coupler("one_way", contact_forces=True)``, while the arm still goes where it is dragged. It reads zero
until the fingers touch, then, stroking, 32 N with the fingertips 6 mm over the bench and 116 N at 4 mm,
which the bench's own reading takes up to within a newton (``tests/test_scrub_a_stain.py``, GPU 5). A
stroke that only brushes the stain wipes nothing, one pressed past 60 N marks what it rubs **too hard**
until the hand comes off, and pressing or lifting without sliding counts for nothing. Ease the hand up as
it leaves the stain: stopped dead, the towel it pushed ahead presses back. The stain is drawn brown, red
where pressed too hard, gone where wiped. Contact forces resolve a press, not a resting weight.
"""

import numpy as np

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment
from robosimstudio_materials.cloth import ForceGatedWipe


def main():
    size, cells = 0.35, 30  # a 0.35 m terry towel, 30 quads a side
    edge = -0.20  # the towel's near edge, in the bench's work-surface frame [m]
    band = (20.0, 60.0)  # the push that wipes [N]
    # The stain: 100 mm of the towel's top face along y = -0.08, where a hand pressing down reaches it.
    stain = ("stain", lambda r: (abs(r[:, 1] + 0.08) < 0.008) & (abs(r[:, 0]) < 0.05) & (r[:, 2] > np.median(r[:, 2])))
    # The right arm's joints [rad], closed fingers 90 mm over the stain's near end (``solve_q_full``).
    over_stain = (0.5928, 2.1268, 1.87, -1.3133, 0.0003, 0.5928)

    forces = rss.coupler("one_way", contact_forces=True)  # one-way, and reading what each contact pushes
    physics = rss.replace(rss.GENERIC, coupling=forces, collision=rss.CollisionSpec(full_surface_contact=True))
    yam = get_embodiment("yam_bimanual")
    left, right = yam.arms
    yam = rss.replace(yam, arms=(left, rss.replace(right, home_q=over_stain)))

    scene = rss.Scene(cell="bench_0728", physics=physics, name="cloth")
    scene.add(rss.robot(yam))
    towel = rss.sheet(
        name="towel",
        size=(size, size),
        cells=(cells, cells),
        at=(-size / 2, edge, 0.02),
        material=rss.materials.TERRY_TOWELLING,
        particle_sets=(stain,),
    )
    scene.add(towel)
    sim = scene.build(control_mode="joint_position")

    def page(studio):
        wipe = ForceGatedWipe(sim, "towel", "stain", "right", band)
        with studio.gui.add_folder("Scrub", order=-1.0):  # first on the panel: the readouts are the evidence
            push = studio.gui.add_number("push (N)", 0.0, step=0.1, disabled=True)
            wiped = studio.gui.add_number(f"wiped ({band[0]:.0f}-{band[1]:.0f} N)", 0.0, step=0.01, disabled=True)
            hard = studio.gui.add_number("too hard", 0.0, step=0.01, disabled=True)
            studio.gui.add_button("Clean stain").on_click(lambda _: wipe.reset())

        def each_step() -> None:
            reading = wipe.update()
            push.value, wiped.value, hard.value = (round(reading[k], 2) for k in ("push", "wiped", "crushed"))
            wipe.draw(studio.scene)  # the stain on the towel: brown, red where too hard, gone where wiped

        studio.each_step(each_step)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
