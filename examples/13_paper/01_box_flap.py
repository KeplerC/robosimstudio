# SPDX-License-Identifier: Apache-2.0
"""Close a cardboard flap

A carton-board flap hinged on the rim of an open box: push it over with a hand and it creases at the hinge and stays shut.

The box is four slabs; the flap is a strip of CARDBOARD whose bound edge is ``pinned`` along
the far rim, standing out level over the bench. Drag a hand onto its free end and push it down
over the opening: the hinge row bends past the yield and keeps it, so the flap stays down when
the hand leaves. Board this stiff needs a better-converged solve than the default: VBD runs 20
iterations here instead of 5 (on the drape bench a 100 mm flap sags 19 % of its length under
its own weight at 5 iterations and 5 % at 20). The **Hinge** readout is ``crease_angle`` there.
"""

import robosimstudio as rss
from robosimstudio.assetlib.semantics.snapshot import Snapshot
from robosimstudio.robots.registry import get_embodiment
from robosimstudio_materials.paper import crease_angle


def main():
    box = (-0.05, -0.02)  # the box's centre on the bench [m]
    inner = 0.12  # inside span of the box [m]
    wall = 0.01  # wall thickness [m]
    height = 0.10
    flap = (0.12, 0.13)  # across the box, and out from the hinge [m]
    cells = (12, 13)  # a 10 mm cell, the one CARDBOARD was measured at
    rim = box[1] + inner / 2  # the inner face of the far wall; two pinned rows sit on the wall's top

    physics = rss.replace(
        rss.GENERIC,
        coupling=rss.coupler("one_way"),
        collision=rss.CollisionSpec(full_surface_contact=True),
    ).with_backend("*", rss.backend("vbd", iterations=20, rigid_body_contact_buffer_size=256))

    scene = rss.Scene(cell="bench_0728", physics=physics, name="box_flap")
    scene.add(rss.robot(get_embodiment("yam_bimanual")))
    half = inner / 2 + wall / 2  # the walls' centres from the box's, and their half-span along it
    for name, dx, dy in (("near", 0, -half), ("far", 0, half), ("left", -half, 0), ("right", half, 0)):
        size = (half, wall / 2, height / 2) if dx == 0 else (wall / 2, half, height / 2)
        scene.add(rss.slab(name=f"wall_{name}", half=size, at=(box[0] + dx, box[1] + dy, height / 2)))
    scene.add(
        rss.paper(
            name="flap",
            size=flap,
            cells=cells,
            at=(box[0] - flap[0] / 2, rim, height + 0.003),
            material=rss.papers.CARDBOARD,
            particle_sets=(("bound", lambda rest: rest[:, 1] <= rim + 0.015),),
            pinned="bound",
        )
    )
    sim = scene.build(control_mode="joint_position")
    sheet = sim["flap"].entity
    line = {"crease": {"axis": 1, "at": rim + 0.02}}

    def page(studio):
        with studio.gui.add_folder("Hinge"):
            angle = studio.gui.add_number("crease_angle (rad)", 0.0, disabled=True)

        def each_step() -> None:
            sheet.update_probe(sim.state, sim.step_count)
            angle.value = round(crease_angle(Snapshot(sim.make_context()), "flap", line)["angle"], 2)

        studio.each_step(each_step)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
