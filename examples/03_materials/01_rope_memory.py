# SPDX-License-Identifier: Apache-2.0
"""Rope memory: wrap a post

A cable with plastic memory lies on the bench, clamped at its far end: drag the right hand down onto it, press **Close**, lift it a little, carry its end round the post while turning the gizmo's ring a third of a turn, lay it down, press **Open** and lift away, and it stays wrapped.

``dahl_eps_max`` turns on Newton's Dahl bending hysteresis: a bend put into a
joint past ``dahl_tau`` radians is kept. The pads alone hold the cable (sticky
contacts, as in ``00_rope``). Carried 120 degrees round the post and let go,
it reads straightness 0.69; built at 0, the same drive springs back to 0.99
(``tests/test_cable_robot``, GPU 2). The bench and posts have no friction, so
the bend is the cable's: with 0.24 or 0.024 left, the elastic cable stayed
wrapped too, 0.74 and 0.78 (``mem_drive`` probe, GPU 2).
"""

import numpy as np

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment
from robosimstudio_materials.rod.measures import straightness


def main():
    length, radius, segments = 0.4, 0.011, 18  # [m]; spacing 22.2 mm, above twice the radius
    start = (0.15, -0.05, radius + 0.002)  # the clamped end, in the bench's frame [m]; the cable runs toward the arms
    post = (0.0, -0.086)  # the post it is wrapped round, 5 mm clear of the cable's side [m]
    eps_max, tau = 2.0, 0.1  # the Dahl envelope and transition [rad], Newton's cable-bundle example's
    # The right arm [rad]: 100 mm over capsule 13, pointing down, jaws across (``solve_q_full``).
    over_cable = (0.822, 1.900, 1.514, -1.185, 0.0, 0.822)

    vbd = rss.backend("vbd", rigid_body_contact_buffer_size=256, rigid_contact_history=True)
    physics = rss.replace(rss.GENERIC.with_backend("*", vbd), collision=rss.CollisionSpec(contact_matching="sticky"))
    yam = get_embodiment("yam_bimanual")
    yam = rss.replace(yam, arms=(yam.arms[0], rss.replace(yam.arms[1], home_q=over_cable)))

    scene = rss.Scene(cell="bench_075", physics=physics, name="cable_memory")
    scene.add(rss.robot(yam))
    scene.add(rss.post(name="clamp", radius=0.02, half_height=0.03, at=(start[0] + 0.033, start[1], 0.03)))
    scene.add(rss.post(name="post", radius=0.02, half_height=0.05, at=(*post, 0.05)))
    cable = scene.add(
        rss.rod(
            name="cable",
            length=length,
            radius=radius,
            segments=segments,
            at=start,
            along=(-1.0, 0.0, 0.0),
            anchors=("start",),
            dahl_eps_max=eps_max,
            dahl_tau=tau,
            allow_grasp_latch=False,  # nothing holds it but the pads
            scene_friction=True,  # pads at mu 60; the bench and posts at the two zeros below
            mu_surface=0.0,
            mu_base=0.0,
        )
    )
    sim = scene.build(control_mode="joint_position", device="cuda")  # VBD: a CPU's rope has no Dahl memory

    def page(studio):
        out = studio.gui.add_number("straightness (1 = straight)", 1.0, step=1e-3, disabled=True)

        def read() -> None:
            out.value = round(straightness(np.asarray(sim.pipeline.snapshot.body_q)[cable.bodies, :3]), 3)

        studio.each_step(read)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
