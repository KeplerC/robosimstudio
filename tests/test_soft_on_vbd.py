# SPDX-License-Identifier: Apache-2.0
"""A soft body routed to VBD rather than SuperDex is active to Newton, and falls."""

from __future__ import annotations

import unittest

import numpy as np

TIER = "medium"
"""Builds a scene with a robot and steps it: a GPU."""


class TestASoftBodyOnVbdFalls(unittest.TestCase):
    """EntitySoft cleared ACTIVE whatever the backend, so a body routed to VBD never moved."""

    def test_it_falls(self):
        import newton

        import robosimstudio as rss

        physics = rss.replace(rss.GENERIC, backends=tuple(e for e in rss.GENERIC.backends if e[0] != "soft"))
        scene = rss.Scene(cell="void", physics=physics, name="vbd_soft")
        # With a robot the arrangement couples, and the solvers copy the flags they are built with.
        scene.add(rss.robot("yam_bimanual"))
        body = scene.add(
            rss.SoftSpec(name="block", size=(0.04, 0.04, 0.02), cells=(4, 4, 2), pose=(0.0, 0.0, 0.3), settle_frames=0)
        )
        sim = scene.build(control_mode="joint_position")
        flags = sim.pipeline.model.particle_flags.numpy()[np.asarray(body.particles)]
        self.assertTrue(np.all(flags & int(newton.ParticleFlags.ACTIVE)))
        before = float(sim.pipeline.state_0.particle_q.numpy()[body.particles, 2].mean())
        for _ in range(30):
            sim.pipeline.step()
        after = float(sim.pipeline.state_0.particle_q.numpy()[body.particles, 2].mean())
        self.assertLess(after, before - 0.01)


if __name__ == "__main__":
    unittest.main()
