# SPDX-License-Identifier: Apache-2.0
"""The twist example's page does what its docstring says: a turn at the tip reaches the middle, and only the stiff cable lets it go.

``examples/03_materials/02_twist.py`` holds each cable's free end in a hard latch
and rolls it with a slider; this runs the example's own ``main()`` and drives
its page the way a hand does -- the slider dragged across in steps of at most
15 degrees of roll, the readouts read, the **hold** box unticked -- with the
page's widgets replaced by plain values and no browser. The scene is the
example's, robot included, which is what makes the drive a real check: with
two islands a written pose reaches the cable only through the kinematic path
``EntityRod.pin`` takes (``robosimstudio.core.tuning.set_kinematic`` says
why), and the drive the verifier used, a massless tip teleported each step,
read 0.000 turns here while passing on the one-island scene.

The page's **hold the tips** box starts unticked, because the tips are the
hands' until someone ticks it, so this ticks it first. Measured on this scene,
re-measured after the cables moved to one per hand (``slider`` probe, GPU 2) to
the same digits as before (``probe6.py``, RTX PRO 6000, newton 1.6.0):
both cables straight and untwisted at rest; after the turn 0.348 turns in the
middle third of each (8 of the 23 joints' share of one turn, 0.3478) with
straightness 1.000 throughout the roll; released, the stiff cable reads 0.244
at 5 steps, 0.055 at 15, 0.002 at 30 and 0.000 at 60, the soft one 0.346,
0.343, 0.337 and 0.326 over the same steps and 0.267 at 240. A reset puts
both back at 0.
"""

from __future__ import annotations

import contextlib
import unittest

import numpy as np

from tests.fixtures.examples import _run

TIER = "medium"
"""Builds the example, robot and all, and rolls its cables: a GPU."""

SLIDER_STEPS = 24
"""The slider dragged to one turn in 24 steps: 15 degrees of roll a step, the fastest hand the page should see.
The page turns the tip at most 0.1 rad a step (``EntityRod.ROLL_STEP``), so the tip lags the slider and
finishes the turn on the 63rd step: a yanked slider is a hand's turn, not a jump."""
HOLD_STEPS = 50
"""Steps the slider rests at one turn before the hand lets go; with the drag, 74 roll steps, past the 63rd."""
UNIFORM_TURNS = 8.0 / 23.0
"""One tip turn, spread over 23 joints, seen through the 8 between capsules 8 and 16 of 24."""
RELEASE_STEPS = 60


class _Value:
    """A widget as the page sees it: a thing with a ``value``."""

    def __init__(self, value) -> None:
        self.value = value


class _Gui:
    def __init__(self) -> None:
        self.widgets: dict[str, _Value] = {}

    def add_slider(self, label, *, initial_value=0.0, **_) -> _Value:
        return self.widgets.setdefault(label, _Value(initial_value))

    def add_checkbox(self, label, *, initial_value=False, **_) -> _Value:
        return self.widgets.setdefault(label, _Value(initial_value))

    def add_number(self, label, value, **_) -> _Value:
        return self.widgets.setdefault(label, _Value(value))

    def add_folder(self, *_, **__):
        return contextlib.nullcontext()


class _Page:
    """What ``rss.view(sim, page=...)`` hands the example: widgets, and a per-step hook."""

    def __init__(self) -> None:
        self.gui = _Gui()
        self.steps: list = []

    def each_step(self, fn) -> None:
        self.steps.append(fn)


class TestTheTwistPage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim, kwargs = _run("03_materials/02_twist.py")
        cls.page = _Page()
        kwargs["page"](cls.page)
        cls.widgets = cls.page.gui.widgets
        cls.roll = cls.widgets["tip roll (turns)"]
        cls.hold = cls.widgets["hold the tips"]
        cls.names = ("soft_twist", "stiff_twist")
        cls.readouts = {n: cls.widgets[f"{n}: mid-cable twist (turns)"] for n in cls.names}
        cls.rods = [cls.sim.scene.by_name[n] for n in cls.names]
        cls.sim.run(60)
        cls.at_rest = cls.read()
        cls.hold.value = True  # the tips start free, for the hands; the slider rolls them only in the hold
        cls.straight_rest = cls.straight()
        cls.straight_rolling = 1.0
        for k in range(1, SLIDER_STEPS + 1):
            cls.roll.value = k / SLIDER_STEPS
            cls.tick(1)
            cls.straight_rolling = min(cls.straight_rolling, *cls.straight())
        cls.at_slider_end = cls.read()
        cls.tick(HOLD_STEPS)
        cls.rolled = cls.read()
        cls.straight_rolled = cls.straight()
        cls.hold.value = False
        cls.released = {}
        done = 0
        for at in (5, 15, 30, RELEASE_STEPS):
            cls.tick(at - done)
            done = at
            cls.released[at] = cls.read()
        cls.straight_released = cls.straight()
        cls.sim.reset()
        cls.sim.run(60)
        cls.after_reset = cls.read()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    @classmethod
    def tick(cls, steps: int) -> None:
        """The page's step: its hook, then the physics -- what the studio does per control step."""
        for _ in range(steps):
            for fn in cls.page.steps:
                fn()
            cls.sim.run(1)
        cls.sim.sync()

    @classmethod
    def read(cls) -> dict[str, float]:
        """The readouts, which the page's hook writes; running it here also advances the roll by one step's share."""
        for fn in cls.page.steps:
            fn()
        return {n: float(cls.readouts[n].value) for n in cls.names}

    @classmethod
    def straight(cls) -> list[float]:
        from robosimstudio_materials.rod.measures import straightness

        body_q = cls.sim.pipeline.state_0.body_q.numpy()
        return [straightness(rod.positions(body_q)) for rod in cls.rods]

    def test_both_hang_straight_and_untwisted(self) -> None:
        for name, turns in self.at_rest.items():
            self.assertLess(abs(turns), 0.02, name)
        for value in self.straight_rest:
            self.assertGreater(value, 0.99)

    def test_the_tip_lags_a_yanked_slider(self) -> None:
        """When the slider reaches one turn the tip has rolled under half of it: the page's rate limit, not a jump."""
        for name, turns in self.at_slider_end.items():
            self.assertGreater(abs(turns), 0.05, f"{name}: {self.at_slider_end}")
            self.assertLess(abs(turns), 0.2, f"{name}: {self.at_slider_end}")

    def test_the_turn_reaches_the_middle_of_both(self) -> None:
        for name, turns in self.rolled.items():
            self.assertGreaterEqual(abs(turns), 0.3, f"{name}: {self.rolled}")
            self.assertAlmostEqual(abs(turns), UNIFORM_TURNS, delta=0.03, msg=f"{name}: {self.rolled}")

    def test_the_cables_stay_straight_while_rolled(self) -> None:
        self.assertGreater(self.straight_rolling, 0.98, "the cable bent while the tip turned")
        for value in (*self.straight_rolled, *self.straight_released):
            self.assertGreater(value, 0.98)

    def test_the_stiff_cable_unwinds_first_and_the_soft_one_keeps_it(self) -> None:
        soft, stiff = (abs(self.released[RELEASE_STEPS][n]) for n in self.names)
        self.assertLess(stiff, 0.05, f"stiff after {RELEASE_STEPS} steps: {self.released}")
        self.assertGreater(soft, 0.25, f"soft after {RELEASE_STEPS} steps: {self.released}")
        for at in (5, 15, 30):
            self.assertLess(abs(self.released[at]["stiff_twist"]), abs(self.released[at]["soft_twist"]), at)
        self.assertLess(abs(self.released[15]["stiff_twist"]), 0.1, self.released)

    def test_a_reset_forgets_the_roll(self) -> None:
        for name, turns in self.after_reset.items():
            self.assertLess(abs(turns), 0.02, name)


# ---------------------------------------------------------------- the roll, on a scene of its own
# The free ends of two hanging cables, one twisting ten times more easily than it
# bends and one ten times less, are rolled a full turn and let go. Rolled, both
# carry the same twist in their middle third -- a clamped cable driven at its tip
# twists uniformly, 7 of 19 joints' worth of the turn, whatever its stiffness --
# and both stay straight. Released, the stiff cable springs its roll out within
# half a second and the soft one keeps most of it for seconds: the relaxation time
# is ``twist_kd / twist_ke`` (0.06 / 0.4 = 0.15 s against 0.06 / 0.004 = 15 s),
# which is where the twist stiffness shows and the thing a plug on the end of a
# cable cares about.
LENGTH = 0.30
RADIUS = 0.006
SEGMENTS = 20
TOP = 0.45
"""The cables hang from a clamp at this height [m], straight down, so gravity
tensions them instead of bending them. Clamped level, a 0.3 m cable at
``bend_ke`` 0.04 droops 156 mm at the tip (straightness 0.974), and rolling the
tip of a drooped cantilever kinks it rather than twisting it: measured
straightness 0.83 for both cables and whole-cable turn counts flipping sign
step to step. Hanging, the roll is a roll."""
BEND_KE = 0.04
ROLL_STEPS = 120
"""A full turn put into the tip over two seconds of scene, as a hand rolling a cable end would."""
ROLL_UNIFORM_TURNS = 7.0 / (SEGMENTS - 1)
"""What a full turn at the tip leaves in the middle third of a uniformly
twisted chain: the 7 joints between capsules 6 and 13 of 19. Measured 0.368 and
0.369 for the soft and the stiff cable, against 0.3684."""
ROLL_RELEASE_STEPS = 60
"""Measured after the release, middle-third turns soft / stiff: 0.367 / 0.243
at 5 steps, 0.365 / 0.125 at 10, 0.357 / 0.007 at 30, 0.345 / 0.000 at 60,
0.283 / 0.000 at 240; straightness 1.000 throughout."""


def _roll_scene():
    import robosimstudio as rss

    scene = rss.Scene(cell="void", name="twist")
    for name, x, twist_ke in (("soft_twist", -0.10, BEND_KE / 10), ("stiff_twist", 0.10, BEND_KE * 10)):
        scene.add(
            rss.rod(
                name=name,
                length=LENGTH,
                radius=RADIUS,
                segments=SEGMENTS,
                at=(x, 0.0, TOP),
                along=(0.0, 0.0, -1.0),
                anchors=("start",),
                bend_ke=BEND_KE,
                twist_ke=twist_ke,
            )
        )
    return scene.build()


def _roll_tips(sim, rods, angle: float, *, max_step: float) -> None:
    """Turn each cable's tip capsule about its own axis to ``angle``, through the entity's kinematic hold.

    :meth:`EntityRod.roll` pins the tip -- inverse mass to zero and the kinematic
    flag in the model and every island -- and writes its pose before each step.
    A dynamic tip teleported each step instead injects a twist impulse at a
    capsule whose polar inertia is 1e-8 kg*m^2, and both cables whipped into
    loops within a quarter turn (measured straightness 0.78 at 30 steps).
    """
    for rod in rods:
        rod.roll(sim.pipeline, angle, max_step=max_step)


def _middle_turns(sim, rod) -> float:
    from robosimstudio_materials.rod.measures import twist_turns

    bodies = rod.bodies
    middle = bodies[len(bodies) // 3 : 2 * len(bodies) // 3 + 1]
    return twist_turns(sim.pipeline.state_0.body_q.numpy()[middle, 3:7])


class TestARollTravelsAndAStiffTwistSpringsItOut(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from robosimstudio_materials.rod.measures import straightness

        cls.sim = _roll_scene()
        cls.rods = [cls.sim.scene.by_name[n] for n in ("soft_twist", "stiff_twist")]
        cls.sim.run(60).sync()
        cls.before = [_middle_turns(cls.sim, r) for r in cls.rods]
        for _ in range(ROLL_STEPS):
            _roll_tips(cls.sim, cls.rods, 2.0 * np.pi, max_step=2.0 * np.pi / ROLL_STEPS)
            cls.sim.run(1)
        cls.sim.sync()
        body_q = cls.sim.pipeline.state_0.body_q.numpy()
        cls.rolled = [_middle_turns(cls.sim, r) for r in cls.rods]
        cls.straight_rolled = [straightness(r.positions(body_q)) for r in cls.rods]
        for rod in cls.rods:
            rod.pin(cls.sim.pipeline, on=False)
        cls.sim.run(ROLL_RELEASE_STEPS).sync()
        body_q = cls.sim.pipeline.state_0.body_q.numpy()
        cls.released = [_middle_turns(cls.sim, r) for r in cls.rods]
        cls.straight_released = [straightness(r.positions(body_q)) for r in cls.rods]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def test_both_start_untwisted(self) -> None:
        for name, turns in zip(("soft", "stiff"), self.before, strict=True):
            self.assertLess(abs(turns), 0.02, name)

    def test_the_roll_reaches_the_middle_of_both(self) -> None:
        """A clamped cable driven at its tip twists uniformly: 7 of 19 joints' worth, whatever its stiffness."""
        for name, turns in zip(("soft", "stiff"), self.rolled, strict=True):
            self.assertAlmostEqual(abs(turns), ROLL_UNIFORM_TURNS, delta=0.05, msg=f"{name}: {self.rolled}")

    def test_the_stiff_cable_springs_the_roll_out_and_the_soft_one_keeps_it(self) -> None:
        soft, stiff = (abs(t) for t in self.released)
        self.assertLess(stiff, 0.05, f"stiff twist after {ROLL_RELEASE_STEPS} released steps: {stiff:.3f}")
        self.assertGreater(soft, 0.2, f"soft twist after {ROLL_RELEASE_STEPS} released steps: {soft:.3f}")

    def test_both_stay_straight(self) -> None:
        for name, value in zip(("soft", "stiff"), [*self.straight_rolled, *self.straight_released], strict=False):
            self.assertGreater(value, 0.98, f"{name}: rolled {self.straight_rolled}, released {self.straight_released}")


if __name__ == "__main__":
    unittest.main()
