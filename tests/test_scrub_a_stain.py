# SPDX-License-Identifier: Apache-2.0
"""The scrub example's force readout, and the force-gated wipe it scores, driven the way its page asks.

``examples/01_cloth/03_scrub_a_stain`` reads the push of the towel on the right
hand's colliders -- Newton 1.7's VBD contact force (``CONTACT_F``), through
``coupler("one_way", contact_forces=True)`` -- and wipes the stain only where the
hand rubbed it while the push sat inside a band. This drives the example's own
arm panel through :mod:`tests.fixtures.grasp`, nothing pinned or written to a
joint, and holds the readout to physics first: nothing until the fingers touch,
more the deeper they press, and what the hand puts in coming out of the bench's
own reading. Then the three strokes the page promises: one that only
brushes the stain wipes none of it, one inside the band wipes it, and one pressed
past the band marks it too hard and wipes none.
"""

from __future__ import annotations

import itertools
import time
import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""One GPU: the example built once, pressed at three depths and stroked three times through its arm panel."""

SCRUB_PORT = 9741
SURFACE = 0.728
"""``bench_0728``'s top [world z, m]."""
TIP = 0.0167
"""The closed fingertips hang this far under the tool centre [m], pointing down."""
STAIN = (0.55, 0.65, -0.08)
"""The stain's two ends along x and its line in y [world, m]."""
DEPTHS = (0.0055, 0.0045, 0.0035)
"""Fingertip heights over the bench for the static presses [m]. Held still the hand first reads
the towel at 5.5 mm (2.3 N; nothing at 6.5); stroking, at 7.2 (1.9 N), the stroke heaping the
towel in front of the fingers."""
STROKES = {"brush": 0.5, "band": 20.0, "hard": 90.0}
"""What **push** reads [N] when the hand stops going down for each stroke, the way the page asks for
it: a height is a poor stand-in, because the push climbs 30 to 80 N a millimetre and a towel settled
a little less stands a little taller (the same 6.0 mm read 32 N here and 116 N in a take). Lowered
0.05 mm a control step, the hand stops a little past the reading it was asked for, and a stroke
pushes harder than the press it started from (11, 29 and 132 N for these three), so each target
sits low in its range."""
CLEAR = 0.04
"""How far before the stain each stroke starts along x [m]: the hand goes down away from it, so the
push's climb through the band touches none of it."""
PAST = 0.01
"""How far past the stain each stroke ends [m]. Stroked on to 40 mm past, the push climbed 20 to 50 N
a control step over the last eight, to 208 N in a light stroke whose median was 4.2 N and 940 N in
the hard one, there and nowhere else along the stroke; a stroke drags the towel 7 to 20 mm along
with it (``scrub2.py``), which is the likeliest thing for the fingers to meet there."""


def _settled_push(session, sim, read, steps: int = 30, window: int = 10) -> np.ndarray:
    tick(session, steps)
    samples = []
    for _ in range(window):
        tick(session, 1)
        samples.append(read())
    return np.mean(samples, axis=0)


class TestTheScrubReadsAPhysicalForce(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from robosimstudio_materials.cloth import ForceGatedWipe, contact_force

        cls.sim, kwargs = _run("01_cloth/03_scrub_a_stain.py")
        sim = cls.sim
        sim.run(10).sync()
        t = time.perf_counter()
        sim.run(60).sync()
        cls.ms_per_step = 1e3 * (time.perf_counter() - t) / 60
        cls.session = session = open_session(sim, SCRUB_PORT, kwargs.get("page"))
        inputs = list(session.server.gui._gui_input_handle_from_uuid.values())
        cls.shown = {
            h.label: h for h in inputs if getattr(h, "label", "") in ("push (N)", "wiped (20-60 N)", "too hard")
        }
        hand = lambda: contact_force(sim, "right")  # noqa: E731
        bench = lambda: contact_force(sim, "table")  # noqa: E731
        both = lambda: np.concatenate([hand(), bench()])  # noqa: E731
        gate = ForceGatedWipe(sim, "towel", "stain", "right", (20.0, 60.0))
        session.arms["right"].grip_request = 1.0
        x0, x1, y = STAIN
        start = np.array([x0 - CLEAR, y, SURFACE + TIP + 0.03])
        go(session, "right", start, speed=0.1)
        cls.off = _settled_push(session, sim, both)
        cls.presses = []
        for depth in DEPTHS:
            go(session, "right", (x0 - CLEAR, y, SURFACE + TIP + depth), speed=0.02)
            cls.presses.append((depth, _settled_push(session, sim, both)))
        go(session, "right", start, speed=0.05)
        cls.strokes = {}
        panel = session.arms["right"]
        for name, reads in STROKES.items():
            # What the page asks: down until **push** reads it, watching the readout, then along.
            gate.reset()
            z = SURFACE + TIP + 0.010
            go(session, "right", (x0 - CLEAR, y, z), speed=0.02)
            while gate.update()["push"] < reads and z > SURFACE + TIP:
                z -= 0.00005
                panel.cursor_pos, panel.dragging = np.array([x0 - CLEAR, y, z]), True
                tick(session, 1)
            panel.dragging = False
            tick(session, 20)
            pushes, n = [], 200
            for k in range(1, n + 1):
                along = x0 - CLEAR + (x1 - x0 + CLEAR + PAST) * k / n
                panel.cursor_pos, panel.dragging = np.array([along, y, z]), k < n
                tick(session, 1)
                pushes.append(gate.update()["push"])
            reading = gate.readout()
            cls.strokes[name] = {
                "over": round((z - SURFACE - TIP) * 1e3, 2),
                "push": float(np.median(pushes)),
                "most": float(np.max(pushes)),
                "wiped": reading["wiped"],
                "too hard": reading["crushed"],
            }
            go(session, "right", (x1 + PAST, y, SURFACE + TIP + 0.03), speed=0.05)
            go(session, "right", start, speed=0.1)
        tick(session, 2)
        cls.page = {label: float(handle.value) for label, handle in cls.shown.items()}
        cls.page_push = float(cls.shown["push (N)"].value)
        print(
            f"\n[scrub] off {np.round(cls.off, 3)} presses {[(d, np.round(f, 2).tolist()) for d, f in cls.presses]}",
            flush=True,
        )
        print(f"[scrub] strokes {cls.strokes} page {cls.page} {cls.ms_per_step:.1f} ms", flush=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_scene_steps_captured_inside_its_budget(self) -> None:
        self.assertTrue(self.sim.pipeline.captured, self.sim.pipeline.capture_error)
        self.assertLess(self.ms_per_step, 45.0, f"{self.ms_per_step:.1f} ms per control step; 23.6 measured on GPU 5")

    def test_nothing_until_the_fingers_touch(self) -> None:
        np.testing.assert_array_equal(self.off[:3], np.zeros(3), "30 mm over the towel the hand reads exactly nothing")

    def test_the_push_rises_with_the_press(self) -> None:
        pushes = [float(f[2]) for _, f in self.presses]
        self.assertGreater(pushes[0], 0.0, "touching, the towel pushes the hand up")
        self.assertTrue(all(b > a for a, b in itertools.pairwise(pushes)), f"deeper presses push harder: {pushes}")

    def test_the_bench_takes_what_the_hand_puts_in(self) -> None:
        # The bench's reading moves by the hand's push to within a newton: 17.5 for 18.4 N, 201.4 for
        # 201.3 (force_towel.py). Not its own resting load, which reads 72 N for a 0.6 N towel.
        for depth, force in self.presses:
            put_in, came_out = float(force[2]), float(self.off[5] - force[5])
            self.assertLess(
                abs(came_out - put_in),
                1.0 + 0.02 * put_in,
                f"at {depth * 1e3:.1f} mm: {put_in:.2f} N in, {came_out:.2f} N out",
            )

    def test_a_brushing_stroke_wipes_nothing(self) -> None:
        stroke = self.strokes["brush"]
        self.assertLess(stroke["most"], 20.0, stroke)
        self.assertEqual(stroke["wiped"], 0.0, stroke)

    def test_a_stroke_inside_the_band_wipes_the_stain(self) -> None:
        stroke = self.strokes["band"]
        self.assertGreater(stroke["push"], 20.0, stroke)
        self.assertLess(stroke["push"], 60.0, stroke)
        self.assertGreaterEqual(stroke["wiped"], 0.8, stroke)
        self.assertEqual(stroke["too hard"], 0.0, stroke)

    def test_a_stroke_past_the_band_is_too_hard(self) -> None:
        stroke = self.strokes["hard"]
        self.assertGreater(stroke["push"], 60.0, stroke)
        self.assertGreaterEqual(stroke["too hard"], 0.5, stroke)
        self.assertLess(stroke["wiped"], 0.2, stroke)

    def test_the_page_shows_the_readout(self) -> None:
        self.assertEqual(set(self.shown), {"push (N)", "wiped (20-60 N)", "too hard"})
        self.assertEqual(self.page_push, 0.0, "the hand is off the towel at the end, and the page says so")


if __name__ == "__main__":
    unittest.main()
