# SPDX-License-Identifier: Apache-2.0
"""Reading a contact set as pairwise quantities between shape sets: ``ContactQueries``.

The solver stores each contact's impulse as the one acting on the pair's
*second* geom, and every query here turns that around for whichever set it was
asked about. That sign is the part that is easy to get backwards and impossible
to notice -- an inverted impulse flips ``on``'s support share and a grasp check's
outward push -- so these are stated as properties of the answer: swapping the
arguments negates it, a force is the impulse over the substep, contacts outside
the two sets do not count, and a grasp is two sides pushed the way they open.
"""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio.core.math import quat_from_axis_angle
from robosimstudio.core.probes.contact import MIN_IMPULSE
from tests.fixtures.snapshots import contact_view

TIER = "fast"
"""No GPU: a ``ContactView`` over scripted rows is the real query surface on plain arrays."""

FINGER_A, FINGER_B, OBJECT, TABLE = 0, 1, 2, 3
"""Shape indices; each shape is its own body, so a shape index is also its body's row in ``body_q``."""


def _poses(*quats) -> np.ndarray:
    """``body_q`` for bodies at the origin with these xyzw rotations (identity where none is given)."""
    rows = [[0.0, 0.0, 0.0, *q] for q in quats]
    return np.array(rows + [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * (4 - len(rows)))


class TestThePairwiseImpulse(unittest.TestCase):
    def test_swapping_the_two_sets_negates_it(self) -> None:
        view = contact_view([(OBJECT, TABLE, (0.1, -0.2, 0.3)), (TABLE, OBJECT, (0.0, 0.05, 0.4))])
        np.testing.assert_allclose(
            view.pairwise_impulse([OBJECT], [TABLE]), -view.pairwise_impulse([TABLE], [OBJECT]), atol=1e-15
        )

    def test_the_sign_follows_the_first_argument_whichever_column_it_is_in(self) -> None:
        """A table under an object pushes it up, whether the solver listed the pair (table, object) or the reverse."""
        object_second = contact_view([(TABLE, OBJECT, (0.0, 0.0, 0.2))])
        object_first = contact_view([(OBJECT, TABLE, (0.0, 0.0, -0.2))])  # the impulse on the table: down
        for view in (object_second, object_first):
            np.testing.assert_allclose([0.0, 0.0, 0.2], view.pairwise_impulse([OBJECT], [TABLE]))

    def test_contacts_add_and_a_third_shapes_do_not_count(self) -> None:
        view = contact_view(
            [
                (TABLE, OBJECT, (0.0, 0.0, 0.2)),
                (TABLE, OBJECT, (0.0, 0.1, 0.1)),
                (FINGER_A, OBJECT, (5.0, 5.0, 5.0)),  # the object's, but not against the table
                (FINGER_A, TABLE, (7.0, 7.0, 7.0)),  # the table's, but not against the object
            ]
        )
        np.testing.assert_allclose([0.0, 0.1, 0.3], view.pairwise_impulse([OBJECT], [TABLE]))

    def test_the_force_is_the_impulse_over_the_substep(self) -> None:
        view = contact_view([(TABLE, OBJECT, (0.01, 0.0, 0.03))], dt=0.002)
        np.testing.assert_allclose(
            view.pairwise_impulse([OBJECT], [TABLE]) / 0.002, view.pairwise_force([OBJECT], [TABLE])
        )
        np.testing.assert_allclose([5.0, 0.0, 15.0], view.pairwise_force([OBJECT], [TABLE]))

    def test_no_contacts_is_a_zero_vector_and_no_touch(self) -> None:
        view = contact_view([])
        np.testing.assert_array_equal(np.zeros(3), view.pairwise_impulse([OBJECT], [TABLE]))
        self.assertFalse(view.in_contact([OBJECT], [TABLE]))
        self.assertFalse(view.touching([OBJECT]))
        self.assertFalse(view.check_grasp([[(FINGER_A, [FINGER_A])], [(FINGER_B, [FINGER_B])]], [OBJECT], _poses()))


class TestTouching(unittest.TestCase):
    def test_in_contact_holds_from_the_threshold_up(self) -> None:
        at = contact_view([(TABLE, OBJECT, (0.0, 0.0, MIN_IMPULSE))])
        under = contact_view([(TABLE, OBJECT, (0.0, 0.0, 0.5 * MIN_IMPULSE))])
        self.assertTrue(at.in_contact([OBJECT], [TABLE]))
        self.assertFalse(under.in_contact([OBJECT], [TABLE]), "the numerical zero is not a touch")

    def test_pushes_that_cancel_are_not_in_contact(self) -> None:
        """``in_contact`` reads the net impulse between the sets, so equal and opposite rows sum to nothing."""
        view = contact_view([(TABLE, OBJECT, (0.0, 0.0, 0.2)), (TABLE, OBJECT, (0.0, 0.0, -0.2))])
        self.assertFalse(view.in_contact([OBJECT], [TABLE]))

    def test_touching_ignores_the_excluded_shapes_and_its_own(self) -> None:
        """The "completely on target" test: resting on the target and held by the hand is still nothing else."""
        view = contact_view([(TABLE, OBJECT, (0.0, 0.0, 0.2)), (OBJECT, FINGER_A, (0.1, 0.0, 0.0))])
        self.assertTrue(view.touching([OBJECT]))
        self.assertTrue(view.touching([OBJECT], exclude_shapes=[TABLE]), "the finger still touches it")
        self.assertFalse(view.touching([OBJECT], exclude_shapes=[TABLE, FINGER_A]))
        self.assertFalse(view.touching([OBJECT, TABLE, FINGER_A]), "a set touching only itself touches nothing")

    def test_touching_needs_more_than_the_numerical_zero(self) -> None:
        self.assertFalse(contact_view([(TABLE, OBJECT, (0.0, 0.0, 0.5 * MIN_IMPULSE))]).touching([OBJECT]))


class TestCheckGrasp(unittest.TestCase):
    """Two fingers either side of an object along world y; each opens along its own local +y times its side's sign."""

    SIDES = ([(FINGER_A, [FINGER_A])], [(FINGER_B, [FINGER_B])])

    def _view(self, push_a, push_b):
        """``push_*`` is the impulse the object puts on that finger (stored on the finger, the pair's second geom)."""
        return contact_view([(OBJECT, FINGER_A, push_a), (OBJECT, FINGER_B, push_b)])

    def test_both_sides_pushed_the_way_they_open_is_a_grasp(self) -> None:
        view = self._view((0.0, 0.3, 0.0), (0.0, -0.3, 0.0))
        self.assertTrue(view.check_grasp(self.SIDES, [OBJECT], _poses()))

    def test_one_side_alone_is_not(self) -> None:
        view = contact_view([(OBJECT, FINGER_A, (0.0, 0.3, 0.0))])
        self.assertFalse(view.check_grasp(self.SIDES, [OBJECT], _poses()))

    def test_an_object_resting_on_closed_fingers_is_not_a_grasp(self) -> None:
        """Both fingers pushed down by the object's weight: touching, at 90 degrees to how either opens."""
        view = self._view((0.0, 0.0, -0.3), (0.0, 0.0, -0.3))
        self.assertFalse(view.check_grasp(self.SIDES, [OBJECT], _poses()))
        self.assertTrue(view.check_grasp(self.SIDES, [OBJECT], _poses(), max_angle=95.0), "the angle is what refused")

    def test_the_opening_direction_turns_with_the_finger(self) -> None:
        """Turned a quarter about z, finger A opens along -x: a push along -x is outward, one along +y is not."""
        turned = _poses(quat_from_axis_angle((0.0, 0.0, 1.0), np.pi / 2))
        self.assertTrue(self._view((-0.3, 0.0, 0.0), (0.0, -0.3, 0.0)).check_grasp(self.SIDES, [OBJECT], turned))
        self.assertFalse(self._view((0.0, 0.3, 0.0), (0.0, -0.3, 0.0)).check_grasp(self.SIDES, [OBJECT], turned))

    def test_a_mirrored_hand_states_its_signs(self) -> None:
        """A hand whose finger frames are already mirrored tests ``(+1, +1)``: B then opens along its own +y."""
        view = self._view((0.0, 0.3, 0.0), (0.0, 0.3, 0.0))
        self.assertFalse(view.check_grasp(self.SIDES, [OBJECT], _poses()))
        self.assertTrue(view.check_grasp(self.SIDES, [OBJECT], _poses(), open_axis_signs=(1.0, 1.0)))

    def test_a_side_of_two_bodies_passes_on_either(self) -> None:
        """A fingertip on its own body: the side counts its better body, so a pad pushed outward carries a link that is not."""
        tip = 3  # a fourth body, standing in for finger A's separate tip
        sides = ([(FINGER_A, [FINGER_A]), (tip, [tip])], [(FINGER_B, [FINGER_B])])
        view = contact_view([(OBJECT, tip, (0.0, 0.3, 0.0)), (OBJECT, FINGER_B, (0.0, -0.3, 0.0))])
        self.assertTrue(view.check_grasp(sides, [OBJECT], _poses()))

    def test_no_sides_is_no_grasp(self) -> None:
        self.assertFalse(self._view((0.0, 0.3, 0.0), (0.0, -0.3, 0.0)).check_grasp([], [OBJECT], _poses()))


if __name__ == "__main__":
    unittest.main()
