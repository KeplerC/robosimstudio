# SPDX-License-Identifier: Apache-2.0
"""A page's set request names a hand the scene has, and one asked before a reset is not served after it.

``request_set`` with a hand the robot does not have raised a ``KeyError`` out of
the deformable panel's ``drain``, which is the page loop, after a slot had been
reserved for the grab and was never placed or cleared. And a reset cleared the
clicks, the pins and the slots but not the queue of requests, so a grab asked of
the episode the reset ended was made in the next one. No device: the panel over
stand-ins for the scene, its draggers and the browser.
"""

from __future__ import annotations

import types
import unittest
import unittest.mock

import numpy as np

TIER = "fast"
"""Stand-ins only; no model, no viser server."""


class _Dragger:
    """What the panel touches of a :class:`~robosimstudio.core.draggers.ParticleDragger`, recording grabs."""

    def __init__(self, model, *, mode: str, radius: float) -> None:
        del model, mode
        self.radius = radius
        self.active = False
        self.n_grabbed = 0
        self.grabs: list[tuple] = []

    def grab(self, state, position, quat, *, candidates, fallback_nearest):
        self.grabs.append((position, candidates))
        return 0

    def release(self) -> None:
        self.active = False

    def set_target(self, position, quat) -> None:
        pass

    def apply(self, state, dt) -> None:
        pass


def _panel():
    from robosimstudio_studio.panels import PanelCloth
    from tests.fixtures.viser_stubs import _StubGui, _StubSession

    hand = types.SimpleNamespace(ee_pose=lambda state: ((0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0)))
    iks = {"left": hand, "right": hand}
    q = np.zeros((4, 3))
    towel = types.SimpleNamespace(carries_particles=True, particles=[0, 1, 2, 3], probe=object())
    env = types.SimpleNamespace(
        scene=types.SimpleNamespace(by_name={"towel": towel}, embodiment=types.SimpleNamespace(iks=iks)),
        deformables=lambda: {"towel": towel},  # what Sim.deformables answers: the things carrying a probe
        handles=types.SimpleNamespace(deformables={"towel": types.SimpleNamespace(sets={"towel/edge": [0, 1]})}),
        pipeline=types.SimpleNamespace(hooks=types.SimpleNamespace(post_solve=[]), invalidate_graph=lambda: None),
        model=None,
        state=types.SimpleNamespace(particle_q=types.SimpleNamespace(numpy=lambda: q)),
        arm=iks.__getitem__,  # what Sim.arm does with a name it does not have
    )
    gui = _StubGui()
    with unittest.mock.patch("robosimstudio.core.draggers.ParticleDragger", _Dragger):
        panel = PanelCloth(_StubSession(env, gui), handles=2, chrome=False)
    panel.build(gui)
    return panel


class TestARequestForAHandTheSceneLacks(unittest.TestCase):
    def test_it_is_refused_with_the_hands_and_reserves_no_slot(self) -> None:
        panel = _panel()
        panel.request_set("towel", "edge", "middle")
        panel.drain()  # raised KeyError: 'middle', out of the page loop
        self.assertIn("no arm 'middle'", panel.session.status)
        self.assertIn("['left', 'right']", panel.session.status, "the hands it does have")
        self.assertEqual([], [h.index for h in panel.handles if h.live], "a slot reserved and never placed")
        self.assertEqual([], [d.grabs for d in panel.draggers if d.grabs])


class TestAResetDropsTheRequestsItEnded(unittest.TestCase):
    def test_a_request_queued_before_a_reset_is_not_served_after_it(self) -> None:
        panel = _panel()
        served: list[tuple] = []
        panel.grab_set = lambda *args, **kwargs: served.append((args, kwargs)) or 0
        panel.request_set("towel", "edge", "right")
        panel.on_reset({}, {})
        panel.drain()
        self.assertEqual([], served, "the grab was asked of the episode the reset ended")
        panel.request_set("towel", "edge", "right")
        panel.drain()
        self.assertEqual(1, len(served), "and one asked after it is served")


class TestAPinChosenMidStepIsTakenOnItsOwnHand(unittest.TestCase):
    """The dropdown runs on viser's threads: a step can read a slot's old hand after the mark for the new one."""

    def _held(self):
        panel = _panel()
        taken: list[tuple[int, str]] = []
        panel._retake_on_hand = lambda index, arm: taken.append((index, arm))
        panel.draggers[0].active = True
        panel.follow_arm[0] = "left"
        return panel, taken

    def test_a_mark_ahead_of_its_pin_waits_for_the_step_that_reads_the_pin(self) -> None:
        panel, taken = self._held()
        panel._retake[0] = "right"  # the dropdown marked the slot and has not yet written its pin
        panel.drain()
        self.assertEqual([], taken, "taken again on the hand the slot is leaving")
        self.assertEqual({0: "right"}, panel._retake, "and the mark for the new hand was lost")
        panel.follow_arm[0] = "right"
        panel.drain()
        self.assertEqual([(0, "right")], taken)
        self.assertEqual({}, panel._retake)

    def test_a_mark_for_the_pin_it_reads_is_taken_at_once(self) -> None:
        panel, taken = self._held()
        panel._pick_arm(0, "right")
        panel.drain()
        self.assertEqual([(0, "right")], taken)


if __name__ == "__main__":
    unittest.main()
