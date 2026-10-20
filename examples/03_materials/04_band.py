# SPDX-License-Identifier: Apache-2.0
"""Band: hook it on pegs

A rubber band hangs on the upper peg of a pegboard: press **Close** and drag the right hand's shut jaws into the loop, down past the lower peg, in toward the board under it, up, and back out, and the band stays stretched round both pegs.

A band is a closed rod (``RodSpec.closed``): SuperDex closes its polyline with an
element from the last node back to the first, so it has no ends (the capsule chain
refuses one). The shut jaws carry it by contact alone; nothing is pinned. It ends
round both pegs about 19% longer than at rest, held by the pegs; the same cord cut,
driven the same way, is left on the upper peg at its rest length, so what keeps it
is the loop (``tests/test_band_on_pegs``, GPU 1). Backing out, the jaws drag the
band along the lower peg: in one filmed take of two it went off the end, and the
peg's upturned tip is what keeps it. 18 ms a control step. Needs ``[superdex]``.
"""

import numpy as np

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment
from robosimstudio_materials.rod.band import RUBBER, band, encircles


def main():
    face, y = 0.10, -0.20  # the board's face and the pegs' line, in the bench's frame [m]
    pegs = {"upper": (0.26, 0.08), "lower": (0.135, 0.05)}  # height over the bench, length out of the board [m]
    radius = 0.006  # of each peg [m]
    diameter = 0.08  # the band at rest [m]: 251 mm round, so 19% stretched round both pegs
    along_x = (0.0, 0.70710678, 0.0, 0.70710678)  # a post's axis turned from z onto x
    # The right arm's joints [rad]: tool centre 50 mm in front of the band, level, pointing at the board,
    # jaws across y (``sim.arm("right").solve_q_full``).
    in_front = (0.5819, 1.1118, 0.3863, 0.7255, 0.5819, 0.0)

    engine = rss.backend("superdex", substeps=2, feedback="off")  # the fingers drive it by pose
    physics = rss.replace(rss.GENERIC, backends=(("rod", engine), *rss.GENERIC.backends))
    yam = get_embodiment("yam_bimanual")
    yam = rss.replace(yam, arms=(yam.arms[0], rss.replace(yam.arms[1], home_q=in_front)))

    scene = rss.Scene(cell="bench_075", physics=physics, name="band_on_pegs")
    scene.add(rss.robot(yam))
    scene.add(rss.slab(name="board", half=(0.005, 0.012, 0.15), at=(face + 0.005, y, 0.15)))
    for name, (z, length) in pegs.items():
        scene.add(rss.post(name=name, radius=radius, half_height=length / 2, at=(face - length / 2, y, z, *along_x)))
    z, length = pegs["lower"]  # its tip turns up 18 mm, as a pegboard hook's does
    scene.add(rss.post(name="tip", radius=0.004, half_height=0.009, at=(face - length + 0.004, y, z + 0.009)))
    # Hung 65 mm out, in front of the lower peg; skin samples "P1Q6", or the jaws slip between them.
    hang = (0.035, y, pegs["upper"][0] + radius + RUBBER.cord_radius + 0.001 - diameter / 2)
    loop = scene.add(band(name="band", diameter=diameter, at=hang, normal=(1, 0, 0), segments=64, skin_samples="P1Q6"))
    sim = scene.build(control_mode="joint_position")
    origin, rest = np.array([0.6, 0.0, 0.75]), loop.entity.spec.length  # the bench's frame in the world [m]

    def page(studio):
        on = {n: studio.gui.add_checkbox(f"round the {n} peg", False, disabled=True) for n in pegs}
        stretch = studio.gui.add_number("stretch (length / rest)", 1.0, step=1e-3, disabled=True)

        def read() -> None:
            points = np.asarray(loop.particle_q, dtype=np.float64)
            for name, (z, length) in pegs.items():
                on[name].value = encircles(points, origin + np.array([face, y, z]), (-1.0, 0.0, 0.0), length)
            stretch.value = round(float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) / rest, 3)

        studio.each_step(read)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
