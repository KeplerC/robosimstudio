# SPDX-License-Identifier: Apache-2.0
"""A lamp in its holder, with no robot: the Edison thread alone turns a turn into a pitch, and the eyelet closes on the centre contact.

The holder is a fixture and the lamp (``rss.bulb``, an A19 on a medium Edison
screw) is a free body under ``rss.THREAD_60HZ``, started three turns out, where
its thread has a quarter turn in the holder's. A torque on the lamp, added each
substep the way a studio handle's is, turns it clockwise: it goes down a
seventh of an inch a turn, its eyelet lands on the holder's centre contact, and
with the torque gone the contact stays closed, under the lamp's weight and
whatever of the clamp the thread's friction keeps.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

TIER = "medium"
"""One GPU: five turned parts cooked into signed-distance fields (cached on disk) and 5 s of scene at 1200 Hz."""

OUT = 3.0
"""Turns from home the lamp starts at; its thread first meets the holder's at 3.25."""
RATE = 2.0 * math.pi
"""The speed the turning servo asks for [rad/s]; it is limited to 10 mN m, and the lamp turns at about a turn a second."""
CLOSED = 2.0
"""The force on the centre contact that ends the turning [N]."""


class TestTheLampAlone(unittest.TestCase):
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
            # p: the speed asked for about z, the servo's gain, and its torque limit
            spin = wp.spatial_bottom(body_qd[body])
            torque = wp.clamp(p[1] * (p[0] - spin[2]), -p[2], p[2])
            body_f[body] = body_f[body] + wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, torque)

        scene = rss.Scene(cell="bench_075", physics=rss.THREAD_60HZ, name="lamp_pair")
        holder = rss.socket(name="socket")
        scene.add(holder)
        lamp = scene.add(rss.bulb(name="bulb", **screwed(holder, turns_out=OUT)))
        sim = cls.sim = scene.build()
        cls.lamp, cls.entity = holder.lamp, lamp.entity
        body = int(lamp.entity.body)
        model = sim.model
        params = wp.zeros(3, dtype=float, device=model.device)

        def wrench(state, dt=None) -> None:
            wp.launch(drive, dim=1, inputs=[state.body_qd, body, params], outputs=[state.body_f], device=model.device)

        sim.pipeline.hooks.rigid_force.append(wrench)
        sim.pipeline.invalidate_graph()
        hz = float(sim.request.physics.control_hz)
        home = float(rss.get_cell("bench_075").surface_z) + holder.rim
        pair = ([lamp.entity.contact_shape], [sim["socket"].entity.contact_shape])
        r = cls.readings = {"contacts": 0, "mass": float(model.body_mass.numpy()[body])}
        r["centre"] = float(model.body_com.numpy()[body][2])
        last = [None]
        turned = [0.0]
        rows = []

        def run(steps: int) -> None:
            for _ in range(steps):
                sim.run(1)
                pose = sim.state.body_q.numpy()[body].astype(np.float64)
                yaw = quat_yaw(pose[3:7])
                if last[0] is not None:
                    turned[0] -= math.remainder(yaw - last[0], 2.0 * math.pi) / (2.0 * math.pi)
                last[0] = yaw
                sim.contacts.update()
                force = float(np.linalg.norm(sim.contacts.pairwise_force(*pair)))
                lean = math.degrees(2.0 * math.asin(min(1.0, float(np.linalg.norm(pose[3:5])))))
                rows.append((OUT - turned[0], pose[2] - home, force, lean))
                r["contacts"] = max(r["contacts"], int(sim.pipeline.contacts.rigid_contact_count.numpy()[0]))

        run(round(0.5 * hz))
        cls.resting = rows[-1]
        # Clockwise seen from above screws a right-handed lamp in.
        params.assign([-RATE, 0.02, 0.01])
        cls.first = len(rows)
        for _ in range(round(8.0 * hz)):
            run(1)
            if rows[-1][2] > CLOSED:
                break
        cls.last = len(rows)
        params.assign([0.0, 0.0, 0.0])
        run(round(1.0 * hz))
        cls.rows = np.asarray(rows, dtype=np.float64)
        r["seconds"] = (cls.last - cls.first) / hz
        print(
            f"\n[lamp pair] {r} resting {cls.resting} closed {tuple(cls.rows[cls.last - 1])} end {tuple(cls.rows[-1])}"
        )

    def test_the_lamp_is_a_lamp_s_weight_hung_low(self) -> None:
        self.assertAlmostEqual(0.032, self.readings["mass"], delta=1e-6)
        self.assertLess(
            self.readings["centre"], self.lamp.equator - 0.02, "its centre of mass is well under its widest part"
        )

    def test_a_quarter_turn_in_it_stands_on_the_thread(self) -> None:
        out, height, force, lean = self.resting
        self.assertAlmostEqual(OUT, out, delta=0.01)
        self.assertAlmostEqual(self.lamp.pitch * OUT, height, delta=0.5 * self.lamp.axial_play + 0.03e-3)
        self.assertLess(lean, 1.0)
        self.assertEqual(0.0, force)

    def test_it_goes_down_one_pitch_a_turn(self) -> None:
        """Within 2% over two and a half turns, and never farther off the helix than the thread's play."""
        turning = self.rows[self.first : self.last]
        on = turning[(turning[:, 0] < OUT - 0.25) & (turning[:, 0] > 0.25)]
        self.assertGreater(float(np.ptp(on[:, 0])), 2.4)
        slope = float(np.polyfit(on[:, 0], on[:, 1], 1)[0])
        self.assertAlmostEqual(self.lamp.pitch, slope, delta=0.02 * self.lamp.pitch)
        off = on[:, 1] - self.lamp.pitch * on[:, 0]
        print(
            f"\n[lamp pair] {1e3 * slope:.4f} mm a turn against {1e3 * self.lamp.pitch:.4f}; off the helix "
            f"{1e3 * off.min():+.3f} to {1e3 * off.max():+.3f} mm; leaning at most {on[:, 3].max():.2f} degrees"
        )
        self.assertLess(float(np.abs(off).max()), 0.5 * self.lamp.axial_play + 0.03e-3)
        self.assertLess(float(on[:, 3].max()), 1.5)

    def test_the_eyelet_closes_on_the_centre_contact(self) -> None:
        out, height, force, _ = self.rows[self.last - 1]
        self.assertGreater(force, CLOSED)
        self.assertAlmostEqual(0.0, height, delta=0.05e-3)
        self.assertLess(abs(out), 0.1, "home, give or take the play a clamping thread takes up")

    def test_it_stays_closed_with_the_torque_gone(self) -> None:
        out, height, force, lean = self.rows[-1]
        # The lamp's weight, and whatever of the clamp the thread's friction kept: a screw does not back off.
        self.assertGreater(force, 0.9 * 9.81 * self.readings["mass"])
        self.assertLess(force, self.rows[self.last - 1, 2] + 0.5)
        self.assertAlmostEqual(0.0, height, delta=0.02e-3)
        self.assertLess(lean, 0.2)
        self.assertAlmostEqual(self.rows[self.last - 1, 0], out, delta=0.01, msg="it did not back off")

    def test_lighting_it_repaints_its_glass_and_nothing_else(self) -> None:
        from robosimstudio_materials.thread import GLASS, LIT

        entity = self.entity
        before = self.sim.model.shape_color.numpy().copy()
        np.testing.assert_allclose(before[entity.glass_shape][:3], GLASS, atol=1e-6)
        entity.light(True)
        after = self.sim.model.shape_color.numpy()
        np.testing.assert_allclose(after[entity.glass_shape][:3], LIT, atol=1e-6)
        others = np.arange(len(before)) != entity.glass_shape
        np.testing.assert_array_equal(before[others], after[others])
        self.assertIn(entity.glass_shape, self.sim.handles.live_shapes.tolist(), "a viewer reads it again each frame")
        entity.light(False)
        np.testing.assert_allclose(self.sim.model.shape_color.numpy()[entity.glass_shape][:3], GLASS, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "only a bulb lights"):
            self.sim["socket"].entity.light(True)

    def test_the_contacts_fit_the_buffer_the_table_allows(self) -> None:
        self.assertGreater(self.readings["contacts"], 100)
        self.assertLess(self.readings["contacts"], self.sim.request.physics.collision.rigid_contact_max // 2)


if __name__ == "__main__":
    unittest.main()
