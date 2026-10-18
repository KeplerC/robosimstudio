# SPDX-License-Identifier: Apache-2.0
"""Close a cardboard flap

A carton-board flap standing open on the rim of a box and the left hand pointing down beside it: press **Close**, drag the hand toward the box to push the flap over, down to press it into the box, and up again, and it stays shut.

The box is four slabs; the flap is a strip of CARDBOARD ``pinned`` on the far rim, standing up and leaning out
15 degrees as an opened carton's does, because a hand cannot lift one lying out level: the YAM's closed fingers
stand 100 mm over their tips, the height of the box. Pushed over and pressed 55 mm out from the hinge until its
free edge is 74 mm under the rim, the hinge rows bend past the yield and keep 1.9 rad, so the flap comes back
level when the hand lifts (5 mm under, ``flapdrive.py``, GPU 7, RTX PRO 6000). Pressed to the same depth 75 mm
out, the board bends along its length instead: the hinge keeps 1.6 rad and the flap springs back to 13 degrees
over level, its edge 30 mm over the rim (two drives, ``fixprobe.py``, GPUs 1 and 6). Board this stiff needs VBD
at 20 iterations instead of 5 (on the drape bench a 100 mm flap sags 19 % of its length at 5 and 5 % at 20),
40 ms per control step. The **Hinge** readout is ``crease_angle``.
"""

import math

import robosimstudio as rss
from robosimstudio.assetlib.semantics.snapshot import Snapshot
from robosimstudio.robots.registry import get_embodiment
from robosimstudio_materials.paper import crease_angle


def main():
    box = (-0.05, -0.02)  # the box's centre on the bench [m]
    inner, wall, height = 0.12, 0.01, 0.10  # inside span, wall thickness and height of the box [m]
    flap = (0.12, 0.13)  # along the rim, and up from it [m]
    cells = (12, 13)  # a 10 mm cell, the one CARDBOARD was measured at
    lean = math.radians(15.0)  # how far the open flap leans out past upright
    # The left arm's joints [rad]: tool centre 30 mm outside the flap's top edge, pointing down,
    # jaws closing across the flap. Solved with ``sim.arm("left").solve_q_full``.
    beside_flap = (-0.521, 1.859, 1.827, -1.539, 0.0, -0.521)

    physics = rss.replace(
        rss.GENERIC, coupling=rss.coupler("one_way"), collision=rss.CollisionSpec(full_surface_contact=True)
    ).with_backend("*", rss.backend("vbd", iterations=20, rigid_body_contact_buffer_size=256))
    left, right = get_embodiment("yam_bimanual").arms
    yam = rss.replace(get_embodiment("yam_bimanual"), arms=(rss.replace(left, home_q=beside_flap), right))

    scene = rss.Scene(cell="bench_0728", physics=physics, name="box_flap")
    scene.add(rss.robot(yam))
    half = inner / 2 + wall / 2  # the walls' centres from the box's, and their half-span along it
    for name, dx, dy in (("near", 0, -half), ("far", 0, half), ("left", -half, 0), ("right", half, 0)):
        size = (half, wall / 2, height / 2) if dx == 0 else (wall / 2, half, height / 2)
        scene.add(rss.slab(name=f"wall_{name}", half=size, at=(box[0] + dx, box[1] + dy, height / 2)))
    turn = (math.pi / 2 - lean) / 2  # about x: the sheet's +y turned up, leaning out over +y
    quat, at = (math.sin(turn), 0.0, 0.0, math.cos(turn)), (box[0] - flap[0] / 2, box[1] + half, height + 0.003)
    bound = ("bound", lambda rest: rest[:, 2] <= height + 0.018)  # its two lowest rows, on the far wall's top
    paper = rss.paper(name="flap", size=flap, cells=cells, at=at, quat=quat, material=rss.papers.CARDBOARD)
    scene.add(rss.replace(paper, particle_sets=(bound,), pinned="bound"))
    sim = scene.build(control_mode="joint_position")
    sheet = sim["flap"].entity
    line = {"crease": {"axis": 2, "at": height + 0.003 + 0.02 * math.cos(lean)}}  # the hinge rows, by height

    def page(studio):
        with studio.gui.add_folder("Hinge"):
            angle = studio.gui.add_number("crease_angle (rad)", 0.0, step=0.01, disabled=True)

        def each_step() -> None:
            sheet.update_probe(sim.state, sim.step_count)
            angle.value = round(crease_angle(Snapshot(sim.make_context()), "flap", line)["angle"], 2)

        studio.each_step(each_step)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
