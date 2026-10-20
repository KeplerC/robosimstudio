# SPDX-License-Identifier: Apache-2.0
"""The SuperDex cable lifted by the robot's own hand, through the arm panel.

``tests/scenes/02_superdex`` says the right hand can be dragged down onto a
Cosserat cable on SuperDex, closed on it with **Grip amount** and lifted. Its old
clip lifted it with a click handle, which is graphics, not a robot. This drives
the arm panel the way the gizmo and the slider write it
(:mod:`tests.fixtures.grasp`): nothing pins a node, so what comes up came up on
the pads' friction.

SuperDex drives the fingers as colliders by pose, and with ``feedback="off"`` its
push on them does not reach MuJoCo, so the jaws go wherever they are commanded.
The control is **Close**: commanded shut, the jaws close through the 22 mm cable
(about 0 mm), which is why the example asks for **Grip amount** 0.80. The cable
still comes up behind a shut hand, 111 mm, riding on the fingertips inside its
skin (``c_cable.py``): a hook, not a grasp, so the control reads the jaws and not
the rise. The slider
is dragged over 0.8 s, as on the page, not jumped: at 0.78 a jump overshot to
20.0 mm and held while a drag settled at 21.9 mm and lifted nothing (``probe_a3``).
"""

from __future__ import annotations

import importlib.util
import time
import unittest

import numpy as np

TIER = "medium"
"""Two builds of the example on one GPU, each driven through its page for about 6 s of scene time."""

HAS_SUPERDEX = importlib.util.find_spec("superdex") is not None
PORTS = (9781, 9782)
GRIP_AMOUNT = 0.80
"""What the example tells a person to slide **Grip amount** to: 0.20 of the jaws' 94 mm is 19 mm."""
CLOSE_AT = (0.550, -0.011, 0.771)
"""Where the tool centre closes [m, world]: 10 mm over the axis of the cable's middle, as in ``03_materials/00_rope``;
the fingertips then stop about 4 mm over the bench beside an 11 mm-radius cable."""
LIFT = 0.12
"""How far the hand is dragged up once the jaws are on it [m]."""


def _drive(grip: float, port: int, *, ramp: float = 0.8) -> dict:
    """The page's gesture: the gizmo down, the slider to ``grip`` over ``ramp`` s, the gizmo up, held 2 s."""
    from tests.fixtures.examples import _run
    from tests.fixtures.grasp import go, open_session, tick

    sim, kwargs = _run("tests/scenes/02_superdex.py")
    cable = sim["cable"]
    session = open_session(sim, port, kwargs.get("page"))
    arm = session.arms["right"]
    bridge = cable.entity.bridge
    fingers = {sim.model.body_label.index(n) for n in sim.model.body_label if "right_link_" in n and "finger" in n}

    def nodes() -> np.ndarray:
        return np.asarray(cable.particle_q, dtype=np.float64)

    def hand() -> np.ndarray:
        return arm.ik.ee_pose(sim.state)[0]

    try:
        tick(session, 60)  # the cable lands on the bench
        lying = nodes()
        home = arm.ik.home_pose[0]
        under = int(np.argmin(np.linalg.norm(lying[:, :2] - home[:2], axis=1)))
        t = time.perf_counter()
        tick(session, 60)
        ms = 1e3 * (time.perf_counter() - t) / 60
        go(session, "right", np.array(CLOSE_AT), speed=0.08)
        steps = max(1, round(ramp * 60))
        for k in range(1, steps + 1):  # the slider dragged, not jumped
            arm.grip_request = grip * k / steps
            tick(session, 1)
        tick(session, 60)
        contacts = bridge.contact_points("cable")
        on_pads = np.isin(contacts["body"], list(fingers))
        offset = nodes()[under] - hand()
        go(session, "right", np.array(CLOSE_AT) + np.array([0.0, 0.0, LIFT]), speed=0.08)
        drift = 0.0
        for _ in range(120):
            tick(session, 1)
            drift = max(drift, float(np.linalg.norm(nodes()[under] - hand() - offset)))
        held = nodes()
        middle = slice(under - 1, under + 2)
        return {
            "under": under,
            "count": len(lying),
            "home": home,
            "lying": lying,
            "rise": float(held[middle, 2].mean() - lying[middle, 2].mean()),
            "drift": drift,
            "width": float(sim.gripper("right").width(sim.state)),
            "pad_gap": float(contacts["gap"][on_pads].min()) if on_pads.any() else float("nan"),
            "reached": float(np.linalg.norm(hand() - (np.array(CLOSE_AT) + np.array([0.0, 0.0, LIFT])))),
            "finite": bool(sim.state_is_finite()),
            "ms": ms,
            "top": float(lying[:, 2].min()),
            "sim": sim,
        }
    finally:
        session.teardown()


@unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed; nothing steps the cable")
class TestTheHandLiftsTheSuperdexCable(unittest.TestCase):
    """Down onto the middle, **Grip amount** 0.80, up 120 mm, held 2 s."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.run_ = _drive(GRIP_AMOUNT, PORTS[0])

    def test_the_cable_is_on_superdex_and_the_fingers_drive_it_by_pose(self) -> None:
        physics = self.run_["sim"].request.physics
        self.assertEqual("superdex", physics.backend_for("rod").solver)
        self.assertEqual("off", dict(physics.backend_for("rod").params).get("feedback"))

    def test_the_hand_starts_over_the_cables_middle(self) -> None:
        run = self.run_
        self.assertLess(float(np.linalg.norm(run["home"][:2] - run["lying"][run["under"], :2])), 0.012)
        self.assertLessEqual(abs(run["under"] - run["count"] // 2), 1, "the hand is not over the middle")

    def test_it_rests_on_its_skin(self) -> None:
        """An 11 mm cable's axis 11 mm over the bench, not on it (the skin; +0.9 mm without one)."""
        clearance = self.run_["top"] - 0.75
        self.assertAlmostEqual(0.011, clearance, delta=0.0015, msg=f"axis {clearance * 1e3:.2f} mm up")

    def test_the_cable_comes_up_with_the_hand(self) -> None:
        self.assertTrue(self.run_["finite"])
        self.assertLess(self.run_["reached"], 0.010, "the hand did not get to the top of the lift")
        self.assertGreater(self.run_["rise"], 0.10, f"the middle rose {self.run_['rise'] * 1e3:.1f} mm")

    def test_it_does_not_slide_in_the_fingers(self) -> None:
        self.assertLess(self.run_["drift"], 0.003, f"it slid {self.run_['drift'] * 1e3:.1f} mm")

    def test_the_jaws_are_on_the_cable_and_not_through_it(self) -> None:
        """The blade tips 2-4 mm inside the 22 mm cable's width, the pads' contact samples under 2 mm into it."""
        width = self.run_["width"]
        self.assertGreater(width, 0.017, f"the jaws are {width * 1e3:.1f} mm apart")
        self.assertLess(width, 0.021, "the jaws are not on the 22 mm cable")
        self.assertGreater(self.run_["pad_gap"], -0.002, f"a pad is {-self.run_['pad_gap'] * 1e3:.2f} mm inside")

    def test_a_control_step_fits_a_live_page(self) -> None:
        self.assertLess(self.run_["ms"], 60.0, f"{self.run_['ms']:.1f} ms a control step")


@unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed; nothing steps the cable")
class TestCloseCutsThroughIt(unittest.TestCase):
    """The control: **Close** commands the jaws shut, and with nothing fed back they go there.

    The cable still rises with the hand, on the shut fingertips inside its skin (111 mm, ``c_cable.py``).
    """

    def test_close_shuts_the_jaws_through_the_cable(self) -> None:
        run = _drive(1.0, PORTS[1])
        self.assertLess(run["width"], 0.003, f"the jaws stopped at {run['width'] * 1e3:.1f} mm")


if __name__ == "__main__":
    unittest.main()
