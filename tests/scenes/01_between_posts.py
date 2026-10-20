# SPDX-License-Identifier: Apache-2.0
"""Cable between two posts

A cable clamped by both ends on top of two posts: drag the right hand down its arrow onto the cable's middle, press **Close**, drag it back up, press **Open**, and the cable springs back.

``anchors`` holds each end's position, direction and twist where it was built:
one end clamped makes a cantilever, both make a span that sags under its own
weight (22 mm here). The cable is built 2 mm above the post tops so no contact
fights the clamps. The hand leans 40 degrees off straight down: pointing
straight down, a YAM's wrist cannot rise 30 mm off this span (``hook_reach``
probe, GPU 2). The jaws close on the 16 mm cable by contact (no latch); pulled
back 30 mm along the hand's axis, the middle rises 14 mm before the span, which
does not stretch, stops the hand; opened, it is back within 0.1 mm of where it
hung (``tests/test_cable_hook``, GPU 2).
"""

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment


def main():
    span = 0.5  # from post to post [m]
    radius = 0.008  # of the cable [m]
    segments = 24
    post_height = 0.16  # [m]
    x = -0.10  # the span's line, 100 mm toward the arms, where the right hand reaches its middle [m]
    # The right arm [rad]: tool centre 80 mm back from the span's middle along its own axis, leaning 40 degrees
    # out from straight down, jaws across the cable. Solved with ``sim.arm("right").solve_q_full``.
    over_middle = (1.3223, 1.8782, 1.7594, -1.2485, 0.6726, -0.3199)

    scene = rss.Scene(cell="bench_075", name="cable_between_posts")
    yam = get_embodiment("yam_bimanual")
    scene.add(rss.robot(rss.replace(yam, arms=(yam.arms[0], rss.replace(yam.arms[1], home_q=over_middle)))))
    for name, y in (("post_start", -span / 2), ("post_end", span / 2)):
        scene.add(rss.post(name=name, radius=0.02, half_height=post_height / 2, at=(x, y, post_height / 2)))
    scene.add(
        rss.rod(
            name="cable",
            length=span,
            radius=radius,
            segments=segments,
            at=(x, -span / 2, post_height + radius + 0.002),  # 2 mm clear of the post's top
            along=(0.0, 1.0, 0.0),
            anchors=("start", "end"),
            allow_grasp_latch=False,  # nothing holds it but the pads
        )
    )
    sim = scene.build(control_mode="joint_position")

    rss.view(sim)


if __name__ == "__main__":
    main()
