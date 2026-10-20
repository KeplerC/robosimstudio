# SPDX-License-Identifier: Apache-2.0
"""Regressions for the soft and cloth examples on the GPU: the VBD fallback and the shared settle."""

from __future__ import annotations

import unittest

import numpy as np

TIER = "medium"


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


def _two_sheets():
    import robosimstudio as rss

    scene = rss.Scene(cell="bench_0728", name="two_sheets")
    towel = rss.materials.TERRY_TOWELLING
    scene.add(rss.sheet(name="a", size=(0.2, 0.2), cells=(8, 8), at=(-0.3, 0.0, 0.5), material=towel))
    scene.add(rss.sheet(name="b", size=(0.2, 0.2), cells=(8, 8), at=(0.1, 0.0, 0.5), material=towel))
    return scene.build()


class TestSheetsShareOneSettle(unittest.TestCase):
    """Each sheet settled the whole scene and snapshotted, so earlier sheets' baselines were stale."""

    def test_every_sheet_baselines_the_state_reset_restores(self):
        sim = _two_sheets()
        things = {e.name: e for e in sim.scene.entities}
        a, b = things["a"], things["b"]
        self.assertEqual(len(sim.pipeline.particle_settles["runs"]), 1, "two like sheets settle once")
        np.testing.assert_array_equal(a.rest["particle_q"], b.rest["particle_q"])
        sim.reset()
        state = sim.pipeline.state_0.particle_q.numpy()
        for sheet in (a, b):
            baseline = np.asarray(sheet.probe.baseline).reshape(-1, 3)
            np.testing.assert_allclose(baseline, state[: len(baseline)], atol=1e-7)

    def test_the_studio_resettle_steps_the_one_run_once(self):
        """The ``Readings`` folder's button ran each sheet's settle: 120 steps on ``(60, 10)``, 2.53 s of blocked
        page loop, where the build ran the one run's 60 (``tests.test_fix_followup_settle`` on stand-ins)."""
        from types import SimpleNamespace
        from unittest import mock

        from robosimstudio_studio.tuning.panels import PanelReadings

        sim = _two_sheets()
        (run,) = sim.pipeline.particle_settles["runs"]
        panel = PanelReadings(SimpleNamespace(env=sim, driver=None, record=None))
        panel._note, panel.refresh, panel.resettle_requested = SimpleNamespace(value=""), lambda: None, True
        step = type(sim.pipeline).step
        with mock.patch.object(type(sim.pipeline), "step", autospec=True, side_effect=step) as stepped:
            panel.drain()
        self.assertEqual(run[0], stepped.call_count, f"the run {run} stepped once per sheet")
        self.assertEqual("re-settled a, b", panel._note.value)


if __name__ == "__main__":
    unittest.main()
