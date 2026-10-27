# SPDX-License-Identifier: Apache-2.0
"""``on`` and ``contact`` for two rigid bodies: touching, held up, and settled.

``contact`` asks the solver's contact set whether the pair is in it and how hard
they press; ``on`` takes that answer and adds the two things a lean on a tote's
outer wall does not have -- the support's push pointing up, and the item having
stopped. The case that made ``on`` exist is the milk carton nudged against the
outside of the bin: in contact, at rest, and read as placed, reward 1.00 at step
240. Each test here scripts the contact set and the item's velocity and asks
the atoms through the one evaluation path a goal uses, ``Atom.measure``.
"""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio.assetlib.semantics.snapshot import Snapshot
from robosimstudio.domains.rigid.measures.placement import SETTLED_SPEED, UPRIGHT_SHARE, Contact, On
from robosimstudio.scene.handles import EntityHandle, SceneHandles
from tests.fixtures.snapshots import contact_view, ctx

TIER = "fast"
"""No GPU: a scripted contact view and velocity read through the real atoms."""

ITEM, BIN, HAND = 0, 1, 2
RIGID = ("rigid", "rigid")
DT = 0.01


def _snapshot(rows, *, linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0), contacts: bool = True) -> Snapshot:
    """The item (body 0, shape 0) and a static bin (shape 1), the hand's shape 2 in the scene too."""
    handles = SceneHandles(
        objects={"item": EntityHandle(name="item", body=0, collision_shapes=np.array([ITEM]))},
        statics={"bin": EntityHandle(name="bin", static=True, collision_shapes=np.array([BIN]))},
    )
    body_qd = np.array([[*linear, *angular]], dtype=np.float64)
    view = contact_view(rows, dt=DT) if contacts else None
    return Snapshot(ctx=ctx(handles=handles, body_qd=body_qd, contacts=view))


def _on(snapshot: Snapshot) -> dict:
    return On.measure(snapshot, "item", "bin", substrates=RIGID)


def _contact(snapshot: Snapshot) -> dict:
    return Contact.measure(snapshot, "item", "bin", substrates=RIGID)


class TestContact(unittest.TestCase):
    def test_a_pair_in_the_set_touches_with_the_force_it_presses(self) -> None:
        reading = _contact(_snapshot([(BIN, ITEM, (0.0, 0.003, 0.004))]))
        self.assertEqual(1.0, reading["touching"])
        self.assertAlmostEqual(0.005 / DT, reading["force"], places=9)

    def test_the_hand_on_the_item_is_not_the_bin_on_it(self) -> None:
        reading = _contact(_snapshot([(HAND, ITEM, (0.0, 0.0, 0.5))]))
        self.assertEqual({"touching": 0.0, "force": 0.0}, reading)

    def test_no_contact_set_reads_nothing_rather_than_raising(self) -> None:
        self.assertEqual({"touching": 0.0, "force": 0.0}, _contact(_snapshot([], contacts=False)))
        self.assertEqual(
            {"on": 0.0, "touching": 0.0, "up_frac": 0.0, "force": 0.0, "speed": 0.0}, _on(_snapshot([], contacts=False))
        )


class TestOn(unittest.TestCase):
    def test_an_item_held_up_and_still_is_on(self) -> None:
        reading = _on(_snapshot([(BIN, ITEM, (0.0, 0.0, 0.01))]))
        self.assertEqual((1.0, 1.0, 1.0), (reading["on"], reading["touching"], reading["up_frac"]))
        self.assertAlmostEqual(1.0, reading["force"], places=9)

    def test_a_lean_on_the_outer_wall_touches_and_is_not_on(self) -> None:
        """The carton against the outside of the bin: in contact and at rest, pushed sideways only."""
        reading = _on(_snapshot([(BIN, ITEM, (0.0, -0.01, 0.0))]))
        self.assertEqual(1.0, reading["touching"])
        self.assertEqual(0.0, reading["up_frac"])
        self.assertEqual(0.0, reading["on"])

    def test_up_frac_is_the_vertical_share_of_the_push(self) -> None:
        for angle in (0.2, 0.7, 1.2):
            with self.subTest(degrees_from_vertical=np.degrees(angle)):
                push = (0.01 * np.sin(angle), 0.0, 0.01 * np.cos(angle))
                self.assertAlmostEqual(np.cos(angle), _on(_snapshot([(BIN, ITEM, push)]))["up_frac"], places=9)

    def test_more_sideways_than_upward_is_not_held_up(self) -> None:
        """Either side of the gate by three degrees: a push whose upward share is under the threshold is a lean."""
        gate = np.arccos(UPRIGHT_SHARE)
        for tilt, on in ((gate + 0.05, 0.0), (gate - 0.05, 1.0)):
            with self.subTest(degrees_from_vertical=np.degrees(tilt)):
                push = (0.01 * np.sin(tilt), 0.0, 0.01 * np.cos(tilt))
                self.assertEqual(on, _on(_snapshot([(BIN, ITEM, push)]))["on"])

    def test_a_push_from_above_is_not_support(self) -> None:
        """The bin's lid pressing the item down: touching, with the support share negative."""
        reading = _on(_snapshot([(BIN, ITEM, (0.0, 0.0, -0.01))]))
        self.assertEqual((1.0, -1.0, 0.0), (reading["touching"], reading["up_frac"], reading["on"]))

    def test_the_support_reads_the_same_whichever_way_round_the_solver_listed_the_pair(self) -> None:
        first = _on(_snapshot([(ITEM, BIN, (0.0, 0.0, -0.01))]))  # stored on the bin: the item pushes it down
        second = _on(_snapshot([(BIN, ITEM, (0.0, 0.0, 0.01))]))
        self.assertEqual(first, second)

    def test_a_moving_item_is_not_on_yet(self) -> None:
        supported = [(BIN, ITEM, (0.0, 0.0, 0.01))]
        sliding = _on(_snapshot(supported, linear=(1.5 * SETTLED_SPEED, 0.0, 0.0)))
        self.assertEqual((1.0, 0.0), (sliding["up_frac"], sliding["on"]))
        self.assertAlmostEqual(1.5 * SETTLED_SPEED, sliding["speed"], places=12)

    def test_spinning_in_place_is_still_settled(self) -> None:
        """The speed is the linear half of ``body_qd``; the angular half is a different quantity."""
        reading = _on(_snapshot([(BIN, ITEM, (0.0, 0.0, 0.01))], angular=(0.0, 0.0, 5.0)))
        self.assertEqual((0.0, 1.0), (reading["speed"], reading["on"]))


if __name__ == "__main__":
    unittest.main()
