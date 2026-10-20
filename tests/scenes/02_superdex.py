# SPDX-License-Identifier: Apache-2.0
"""Cable on SuperDex

The same cable stepped by another engine, a Cosserat rod on SuperDex: drag the right hand down onto it, slide **Grip amount** to 0.80 so the jaws close on it, and drag the hand up.

One entry in the arrangement routes the ``rod`` kind to SuperDex; the arms stay
on MuJoCo. The cable is one centerline with a twist at every node, stiffness from
``young_modulus``, contact on a tubular skin (its axis rests 11.2 mm up for an
11 mm radius). SuperDex drives the fingers as colliders by pose and its push on
them is not fed back: fed back, they closed through the cable at the default cap
and, uncapped, swung from -27 to +103 mm and squirted it out (``probe_a2``). So
**Close** shuts the jaws through it, to 0.0 mm; the cable still comes up 111 mm,
but riding on the shut fingertips inside its skin, not held between the pads
(``c_cable.py``). **Grip amount** 0.80 stops the blade tips at 19.2 mm on its 22,
the pads 0.1 mm into its skin. Nothing is pinned: the middle rises 113 mm and
slides 0.4 mm in the hand over 2 s, at 11 ms a control step
(``tests/test_superdex_cable_robot``, GPU 1). Needs ``[superdex]``.
"""

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment


def main():
    length, radius, segments = 0.62, 0.011, 28  # [m]; elements of 22 mm
    x = -0.05  # the line it lies on, in the bench's frame [m]: 50 mm toward the arms, inside their reach
    young_modulus = 5.0e6  # [Pa]; a rubber-jacketed cord is of this order
    # Below this slip speed friction grows with slip, so nothing is static: at the engine's 10 mm/s the
    # held cable crept 1.6 mm in 5 s, at 1 mm/s 0.5 mm (``probe_a1``, GPU 1).
    friction_falloff = 1.0e-3  # [m/s]
    # The right arm's joints [rad]: tool centre 100 mm over the bench above the cable's middle, pointing
    # down, jaws across it, as in ``03_materials/00_rope`` (``sim.arm("right").solve_q_full``).
    over_cable = (0.857, 2.337, 2.189, -1.422, 0.0, -0.714)

    # One entry: an exact one wins over GENERIC's fallback, so the rest stays as it was.
    engine = rss.backend("superdex", substeps=2, feedback="off")
    physics = rss.replace(rss.GENERIC, backends=(("rod", engine), *rss.GENERIC.backends))
    yam = get_embodiment("yam_bimanual")
    yam = rss.replace(yam, arms=(yam.arms[0], rss.replace(yam.arms[1], home_q=over_cable)))

    scene = rss.Scene(cell="bench_075", physics=physics, name="cable_superdex")
    scene.add(rss.robot(yam))
    cable = scene.add(
        rss.rod(
            name="cable",
            length=length,
            radius=radius,
            segments=segments,
            at=(x, -length / 2, radius + 0.01),
            along=(0.0, 1.0, 0.0),
            young_modulus=young_modulus,
            friction_falloff=friction_falloff,
        )
    )
    sim = scene.build(control_mode="joint_position")
    top = rss.get_cell("bench_075").surface_z

    def page(studio):
        out = studio.gui.add_number("cable's middle over the bench (mm)", 0.0, step=0.1, disabled=True)

        def read() -> None:
            out.value = round(1e3 * (float(cable.particle_q[segments // 2, 2]) - top), 1)

        studio.each_step(read)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
