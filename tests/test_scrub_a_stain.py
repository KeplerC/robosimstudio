# SPDX-License-Identifier: Apache-2.0
"""The scrub example's force readout, and the force-gated wipe it scores, driven the way its page asks.

``examples/01_cloth/03_scrub_a_stain`` reads the push of the towel on the right
hand's colliders -- Newton 1.7's VBD contact force (``CONTACT_F``), through
``coupler("one_way", contact_forces=True)`` -- and wipes the stain only where the
hand rubbed it while the push sat inside a band. This drives the example's own
arm panel through :mod:`tests.fixtures.grasp`, nothing pinned or written to a
joint, and holds the readout to physics first: nothing until the fingers touch,
more the deeper they press, and what the hand puts in coming out of the bench's
own reading. Then the three strokes the page promises, in the order its clip
shows them: one that only brushes the stain wipes none of it, one pressed past the
band marks it too hard and wipes none, and one inside the band, on the stain cleaned
back, wipes it; lifting the hand straight up off any of them wipes nothing more,
because only a hand sliding along the towel rubs.
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
"""Fingertip heights over the bench for the static presses [m]. Held still, the fingers press the
towel 6.0 N at 5.5 mm, 28 N at 4.5 and 53 N at 3.5 (6.04/28.28/53.04 and 6.05/28.31/53.06 in two runs
of this test, GPUs 6 and 1)."""
STROKES = {"brush": 0.5, "hard": 90.0, "band": 20.0}
"""What **push** reads [N] when the hand stops going down for each stroke, the way the page asks for
it: a height is a poor stand-in, because the push climbs 30 to 80 N a millimetre and a towel settled
a little less stands a little taller (the same 6.0 mm read 32 N here and 116 N in a take). Lowered
0.05 mm a control step, the hand stops a little past the reading it was asked for, and a stroke
pushes harder than the press it started from (11, 114 and 31 N for these three, GPU 3), so each
target sits low in its range. They run in the clip's order, the stroke inside the band last, on the
stain cleaned back after the hard one."""
CLEAR = 0.04
"""How far before the stain each stroke starts along x [m]: the hand goes down away from it, so the
push's climb through the band touches none of it."""
PAST = 0.01
"""How far past the stain each stroke stays at its height [m]. Stroked on to 40 mm past, the push
climbed 20 to 50 N a control step over the last eight, to 208 N in a light stroke whose median was
4.2 N and 940 N in the hard one, there and nowhere else along the stroke; a stroke drags the towel 7
to 20 mm along with it (``scrub2.py``), which is the likeliest thing for the fingers to meet there."""
EASE = (0.015, 0.008)
"""Then on another 15 mm while rising 8 mm [m], off the towel the stroke pushed ahead of it. Stopped
dead 10 mm past the stain, that towel bunched under the fingers and pressed back into the band as the
hand came to rest, and a light stroke wiped the stain's last particle (1 of 9) in the take and in
two of three replays of its drive (``take_path.py``); eased off, three light strokes of three wiped
nothing, and the band and hard strokes read as before (``take_ramp.py``, GPU 3)."""
STEP = 0.00075
"""How far the cursor moves along a stroke each control step [m]: 45 mm/s."""


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
            begin, flat, end = x0 - CLEAR, x1 + PAST, x1 + PAST + EASE[0]
            pushes, n = [], round((end - begin) / STEP)
            for k in range(1, n + 1):
                along = begin + (end - begin) * k / n
                rise = EASE[1] * max(0.0, along - flat) / EASE[0]
                panel.cursor_pos, panel.dragging = np.array([along, y, z + rise]), k < n
                tick(session, 1)
                pushes.append(gate.update()["push"])
            reading = gate.readout()
            # Up 30 mm more at 0.05 m/s, the gate still reading: lifting a hard press takes the push
            # back down through the band while the fingers touch.
            for k in range(1, 37):
                panel.cursor_pos, panel.dragging = np.array([end, y, z + EASE[1] + 0.03 * k / 36]), k < 36
                tick(session, 1)
                gate.update()
            tick(session, 20)
            lifted = gate.update()
            cls.strokes[name] = {
                "over": round((z - SURFACE - TIP) * 1e3, 2),
                "push": float(np.median(pushes)),
                "most": float(np.max(pushes)),
                "wiped": reading["wiped"],
                "too hard": reading["crushed"],
                "wiped lifted": lifted["wiped"],
                "too hard lifted": lifted["crushed"],
            }
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
        self.assertEqual(stroke["wiped lifted"], 0.0, f"lifting straight up off it rubs nothing: {stroke}")

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

    def test_lifting_the_hard_press_off_wipes_nothing(self) -> None:
        # Gated on the band alone, the lift took the push back through it with the fingers still on the
        # towel and marked 3 of the stain's 9 particles wiped after a stroke that wiped none, in one drive
        # of two (scrub_lift.py).
        stroke = self.strokes["hard"]
        self.assertEqual(stroke["wiped lifted"], stroke["wiped"], stroke)
        self.assertGreaterEqual(stroke["too hard lifted"], stroke["too hard"], stroke)

    def test_the_page_shows_the_readout(self) -> None:
        self.assertEqual(set(self.shown), {"push (N)", "wiped (20-60 N)", "too hard"})
        self.assertEqual(self.page_push, 0.0, "the hand is off the towel at the end, and the page says so")


if __name__ == "__main__":
    unittest.main()
