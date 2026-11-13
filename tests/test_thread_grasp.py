# SPDX-License-Identifier: Apache-2.0
"""A YAM lifting a loose cap off its bottle: why ``rss.THREAD_60HZ`` states a finger priority and a part an impedance.

The cap stands unscrewed on a clamped bottle, held by nothing. The left hand
closes on it at its roll-axis grip point with the hand's own 50 N a jaw, lifts
it 30 mm and holds for two seconds, under Newton's collision pipeline each time.

As Newton builds the two, with MuJoCo mixing the fingers' contact class with
the cap's softer one, the cap slides out of the jaws and is lost. With the
fingers' class given priority (the table's setting) it comes up with the hand
and stays; with the cap's own impedance stated instead (the part's default) it
does too. So either holds, both are shipped, and this pins the reason.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "medium"
"""One GPU: the bimanual YAM and two threaded parts built three times, 6 s of scene each."""

DOWN = np.array([0.0, 1.0, 0.0, 0.0])
"""The tool pointing down (xyzw)."""
AT = (-0.0475, 0.31)
"""Where the bottle stands [m]: 0.30 m in front of the left arm's base."""
LIFT = 0.03
"""How far the hand carries the cap up [m]."""


def _lift_and_hold(*, finger_priority: int, impedance: float | None) -> dict:
    """Close on the loose cap, lift, hold: where the cap ended up in the hand and what the jaws put on it."""
    import robosimstudio as rss
    from robosimstudio_materials.thread import THREAD_60HZ, screwed

    physics = rss.replace(THREAD_60HZ, collision=rss.replace(THREAD_60HZ.collision, finger_priority=finger_priority))
    scene = rss.Scene(cell="bench_075", physics=physics, name="thread_grasp")
    scene.add(rss.robot("yam_bimanual"))
    flask = rss.bottle(name="bottle", at=(*AT, 0.0), body=(0.030, 0.085), static=True)
    scene.add(flask)
    lid = scene.add(rss.cap(name="cap", impedance=impedance, **screwed(flask, turns_out=2.2)))
    sim = scene.build(control_mode="joint_position")
    try:
        sim.run(30).sync()
        hz = float(sim.request.physics.control_hz)
        arm, grip, hand = sim.arm("left"), sim.gripper("left"), sim.scene.embodiment.hands["left"]
        targets, spec = sim.control.joint_target_q, lid.entity.spec
        start = np.asarray(lid.position, dtype=np.float64)
        now = [np.asarray(sim.state.joint_q.numpy(), dtype=np.float64)[arm.arm_coords], 0.0]

        def over(height: float) -> np.ndarray:
            middle = start + np.array([0.0, 0.0, spec.grasp_height + height])
            full = np.asarray(sim.state.joint_q.numpy(), dtype=np.float64)
            full[arm.arm_coords] = (0.0, 1.2, 1.4, -1.0, 0.0, 0.0)
            return arm.solve_q_full(full, hand.tcp_for("roll_axis", middle, DOWN), DOWN)[arm.arm_coords]

        def move(q: np.ndarray, closed: float, seconds: float) -> None:
            q0, g0 = now[0].copy(), now[1]
            steps = max(1, round(seconds * hz))
            for k in range(1, steps + 1):
                f = (k / steps) ** 2 * (3.0 - 2.0 * k / steps)
                arm.set_joint_targets(q0 + (q - q0) * f, targets)
                grip.command = g0 + (closed - g0) * f
                grip.apply(targets)
                sim.run(1)
            now[0], now[1] = np.array(q, dtype=np.float64), closed

        def off_the_grip_point() -> np.ndarray:
            """The roll-axis grip point's miss of the cap's ribbed band [m], from the tool centre's pose."""
            tcp, quat = arm.ee_pose(sim.state)
            middle = np.asarray(lid.position, dtype=np.float64) + np.array([0.0, 0.0, spec.grasp_height])
            return np.asarray(tcp, dtype=np.float64) - np.asarray(hand.tcp_for("roll_axis", middle, quat))

        near = 1.0 - (spec.thread.grip_radius + 0.0025) / hand.open_width
        move(over(0.05), 0.0, 1.2)
        move(over(0.0), 0.0, 1.2)
        move(over(0.0), near, 0.5)
        move(over(0.0), 1.0, 1.1)
        move(over(LIFT), 1.0, 1.0)
        lifted = off_the_grip_point()
        model = sim.model
        labels = [str(label).rsplit("/", 1)[-1] for label in model.body_label]
        bodies = model.shape_body.numpy()
        jaws = [
            [s for s in range(model.shape_count) if bodies[s] >= 0 and labels[bodies[s]] == f"left_link_{side}_finger"]
            for side in ("left", "right")
        ]
        sim.contacts.update()
        squeeze = [
            float(np.linalg.norm(sim.contacts.pairwise_force(jaw, [int(s) for s in lid.entity.shapes]))) for jaw in jaws
        ]
        move(over(LIFT), 1.0, 2.0)
        held = off_the_grip_point()
        return {
            "rose": float(lid.position[2] - start[2]),
            "miss": float(np.linalg.norm(held)),
            "creep": float(np.linalg.norm(held - lifted)) / 2.0,
            "squeeze": squeeze,
            "finite": sim.state_is_finite(),
        }
    finally:
        sim.close()


class TestWhatHoldsASmallPartOnNewtonsContacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.as_built = _lift_and_hold(finger_priority=0, impedance=None)
        cls.priority = _lift_and_hold(finger_priority=1, impedance=None)
        cls.impedance = _lift_and_hold(finger_priority=0, impedance=0.998)
        print(
            f"\n[thread grasp] as built {cls.as_built}\n  priority {cls.priority}\n  impedance {cls.impedance}",
            flush=True,
        )

    def test_as_built_the_cap_is_lost(self) -> None:
        """Mixed contact classes: the cap does not come up with the hand, and the jaws end on nothing."""
        r = self.as_built
        self.assertTrue(r["finite"])
        self.assertLess(
            r["rose"], 0.5 * LIFT, "the cap came up with the hand, so the mix held it: the table may not need this"
        )
        self.assertGreater(r["miss"], 0.010)

    def test_with_the_fingers_class_given_priority_it_is_held(self) -> None:
        self._held(self.priority)

    def test_with_the_parts_own_impedance_it_is_held(self) -> None:
        self._held(self.impedance)

    def _held(self, r: dict) -> None:
        self.assertTrue(r["finite"])
        self.assertAlmostEqual(LIFT, r["rose"], delta=0.0015, msg="the cap came up with the hand")
        self.assertLess(r["miss"], 0.0006, "still on the grip point after two seconds")
        self.assertLess(r["creep"], 1.0e-4, "creeping under 0.1 mm/s (0.013 measured)")

    def test_the_contact_probe_reads_the_squeeze_on_this_path(self) -> None:
        """50 N a jaw, the hand's own clamp, read off Newton's contacts by the scene's own probe."""
        for r in (self.priority, self.impedance):
            for jaw in r["squeeze"]:
                self.assertAlmostEqual(50.0, jaw, delta=4.0)
        self.assertLess(max(self.as_built["squeeze"]), 5.0, "nothing in the jaws")


if __name__ == "__main__":
    unittest.main()
