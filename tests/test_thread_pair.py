# SPDX-License-Identifier: Apache-2.0
"""A cap on a bottle's neck, with no robot: thread contact alone turns a turn into a pitch and holds a pull.

The bottle is clamped (``static=True``) and the cap is a free body under
``rss.THREAD_60HZ``; nothing joins them. A wrench on the cap, added each
substep the way a studio handle's is, pulls it straight up and then turns it:
pulled with 20 N it rises by the thread's own play and stays on, and turned it
rises one pitch a turn for the two turns its ridge is wound.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

TIER = "medium"
"""One GPU: two meshes cooked into signed-distance fields (cached on disk) and 6 s of scene at 1200 Hz."""

PULL = 20.0
"""The straight pull [N], 680 times the cap's weight, ramped over half a second."""
RATE = 2.0 * math.pi
"""The speed the turning servo asks for [rad/s]; it is limited to 2 mN m, and the cap turns at about 0.8 turn/s."""


class TestTheThreadAlone(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import warp as wp

        import robosimstudio as rss
        from robosimstudio.core.math import quat_yaw
        from robosimstudio_materials.thread import screwed

        @wp.kernel
        def drive(
            body_qd: wp.array[wp.spatial_vector], body: int, p: wp.array[float], body_f: wp.array[wp.spatial_vector]
        ):
            # p: the speed asked for about z, the servo's gain, its torque limit, and a force along z
            spin = wp.spatial_bottom(body_qd[body])
            torque = wp.clamp(p[1] * (p[0] - spin[2]), -p[2], p[2])
            body_f[body] = body_f[body] + wp.spatial_vector(0.0, 0.0, p[3], 0.0, 0.0, torque)

        scene = rss.Scene(cell="bench_075", physics=rss.THREAD_60HZ, name="thread_pair")
        flask = rss.bottle(name="bottle", static=True)
        scene.add(flask)
        lid = scene.add(rss.cap(name="cap", **screwed(flask)))
        sim = cls.sim = scene.build()
        cls.thread = flask.thread
        body = int(lid.entity.body)
        model = sim.model
        params = wp.zeros(4, dtype=float, device=model.device)

        def wrench(state, dt=None) -> None:
            wp.launch(drive, dim=1, inputs=[state.body_qd, body, params], outputs=[state.body_f], device=model.device)

        sim.pipeline.hooks.rigid_force.append(wrench)
        sim.pipeline.invalidate_graph()
        hz = float(sim.request.physics.control_hz)
        r = cls.readings = {"contacts": 0}

        def pose() -> np.ndarray:
            return sim.state.body_q.numpy()[body].astype(np.float64)

        def run(steps: int) -> None:
            for _ in range(steps):
                sim.run(1)
                r["contacts"] = max(r["contacts"], int(sim.pipeline.contacts.rigid_contact_count.numpy()[0]))

        run(round(0.5 * hz))
        seat = pose()
        r["mass"] = float(model.body_mass.numpy()[body])
        r["seat"] = float(seat[2] - flask.rim - rss.get_cell("bench_075").surface_z)
        # 1. A straight pull, ramped: a force put on a 3 g part all at once is a tunnelling test, not a pull.
        ramp = round(0.5 * hz)
        for k in range(ramp):
            params.assign([0.0, 0.0, 0.0, PULL * (k + 1) / ramp])
            run(1)
        run(round(0.5 * hz))
        pulled = pose()
        r["pull_rise"] = float(pulled[2] - seat[2])
        r["pull_turn"] = abs(math.remainder(quat_yaw(pulled[3:7]) - quat_yaw(seat[3:7]), 2.0 * math.pi))
        params.assign([0.0, 0.0, 0.0, 0.0])
        run(round(0.3 * hz))
        # 2. Turned counter-clockwise by a weak servo, with nothing but its weight along the axis.
        inertia = float(model.body_inertia.numpy()[body][2, 2])
        params.assign([RATE, 0.4 * inertia / sim.pipeline.sim_dt, 2.0e-3, 0.0])
        start = pose()
        turns, last, rows = 0.0, quat_yaw(start[3:7]), []
        for _ in range(round(3.0 * hz)):
            run(1)
            now = pose()
            yaw = quat_yaw(now[3:7])
            turns += math.remainder(yaw - last, 2.0 * math.pi) / (2.0 * math.pi)
            last = yaw
            rows.append((turns, now[2] - start[2], float(np.linalg.norm(now[3:5]))))
        cls.rows = np.asarray(rows)
        r["finite"] = sim.state_is_finite()
        print(f"\n[thread pair] {r} turned {cls.rows[-1, 0]:.2f}", flush=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def test_the_cap_is_a_few_grams_seated_on_the_rim(self) -> None:
        r = self.readings
        self.assertAlmostEqual(0.0029, r["mass"], delta=0.0003)
        self.assertAlmostEqual(0.0, r["seat"], delta=5.0e-5, msg="its seat rests on the bottle's rim")
        self.assertTrue(r["finite"])

    def test_a_20_newton_pull_does_not_take_it_off(self) -> None:
        """It rises by the thread's play, 0.26 mm, and a little the flanks give; half a pitch would be through the ridge."""
        r = self.readings
        play = 0.5 * self.thread.axial_play
        self.assertGreater(r["pull_rise"], 0.5 * play, "it was pulled: the flanks met")
        self.assertLess(r["pull_rise"], play + 0.15e-3, "and held, within 0.15 mm of where they meet")
        self.assertLess(r["pull_turn"], math.radians(2.0), "without unscrewing itself")

    def test_it_rises_one_pitch_a_turn_for_its_two_turns(self) -> None:
        turns, rise = self.rows[:, 0], self.rows[:, 1]
        self.assertGreater(turns[-1], 2.05, "the servo turned it past its two turns")
        on = (turns > 0.05) & (turns < 2.05)
        slope, offset = np.polyfit(turns[on], rise[on], 1)
        self.assertAlmostEqual(self.thread.pitch, slope, delta=0.02 * self.thread.pitch)
        off = rise[on] - (slope * turns[on] + offset)
        self.assertLess(float(np.sqrt((off**2).mean())), 50.0e-6, "on the helix, not hopping along it")
        for low in (0.05, 0.55, 1.05, 1.55):
            with self.subTest(turns_from=low):
                part = (turns > low) & (turns < low + 0.5)
                self.assertAlmostEqual(
                    self.thread.pitch, float(np.polyfit(turns[part], rise[part], 1)[0]), delta=0.06 * self.thread.pitch
                )

    def test_it_stays_upright_on_the_thread(self) -> None:
        on = (self.rows[:, 0] > 0.05) & (self.rows[:, 0] < 1.8)
        lean = 2.0 * np.degrees(np.arcsin(np.clip(self.rows[on, 2], 0.0, 1.0)))
        self.assertLess(float(lean.max()), 1.5)

    def test_the_contacts_fit_the_buffer_the_table_allows(self) -> None:
        """A thread is carried by a hundred-odd points at once; an overflow drops some without a word."""
        self.assertGreater(self.readings["contacts"], 60)
        self.assertLess(self.readings["contacts"], self.sim.request.physics.collision.rigid_contact_max // 2)


if __name__ == "__main__":
    unittest.main()
