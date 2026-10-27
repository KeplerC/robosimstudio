# SPDX-License-Identifier: Apache-2.0
"""``robosimstudio.arms``: which arm a runtime ``arm_id`` means, and which action slot that arm owns.

Two numbers that coincide whenever the work arm is declared first, and that a
pairing declaring it second pulls apart: ``arm_id`` counts in **role** order
(work, support, then the rest in declaration order), the slot in **declaration**
order, which is the action vector's layout. Conflating them drove the idle hand
on the cable pairing; these hold the two orders apart, each the inverse of
itself, and say what a scene without arms answers.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

TIER = "fast"
"""No GPU: a stand-in sim."""


def _env(*declared: str, work: str = "", support: str = "", handles: bool = True) -> types.SimpleNamespace:
    """An env whose robot declares ``declared`` in action-vector order, its binding naming ``work`` and ``support``."""
    embodiment = types.SimpleNamespace(iks=dict.fromkeys(declared))
    env = types.SimpleNamespace(scene=types.SimpleNamespace(embodiment=embodiment))
    if handles:
        env.handles = types.SimpleNamespace(work_arm=work, support_arm=support)
    return env


class TestArmNames(unittest.TestCase):
    def test_arm_name_on_a_scene_without_a_task(self) -> None:
        from robosimstudio import arms

        sim = types.SimpleNamespace(request=types.SimpleNamespace(full_name="bare"))
        with mock.patch.object(arms, "arm_order", return_value=()):
            with self.assertRaisesRegex(RuntimeError, "'bare' has no arms"):
                arms.arm_name(sim, 0)


class TestRoleOrderAndSlotOrderAreTwoOrders(unittest.TestCase):
    def test_a_work_arm_declared_second_is_arm_zero_but_keeps_its_slot(self) -> None:
        """The idle-hand bug: ``(left, right)`` declared, ``right`` doing the task."""
        from robosimstudio import arms

        env = _env("left", "right", work="right", support="left")
        self.assertEqual(arms.arm_name(env, 0), "right")
        self.assertEqual(arms.arm_slot(env, "right"), 1, "its numbers still live where the layout put them")
        self.assertEqual(arms.arm_slots(env), ("left", "right"))
        self.assertEqual(arms.arm_order(env), ("right", "left"))

    def test_arm_id_of_inverts_arm_name_for_every_arm(self) -> None:
        from robosimstudio import arms

        for env in (
            _env("left", "right", work="right", support="left"),
            _env("a", "b", "c", work="c"),
            _env("solo"),
        ):
            for arm_id, name in enumerate(arms.arm_order(env)):
                with self.subTest(order=arms.arm_order(env), name=name):
                    self.assertEqual(arms.arm_name(env, arm_id), name)
                    self.assertEqual(arms.arm_id_of(env, name), arm_id)

    def test_the_rest_follow_in_declaration_order(self) -> None:
        """A third arm's place cannot depend on a set's iteration order."""
        from robosimstudio import arms

        env = _env("a", "b", "c", "d", work="c", support="a")
        self.assertEqual(arms.arm_order(env), ("c", "a", "b", "d"))
        self.assertEqual([arms.arm_slot(env, name) for name in ("a", "b", "c", "d")], [0, 1, 2, 3])

    def test_a_role_naming_an_arm_the_robot_lacks_is_ignored(self) -> None:
        from robosimstudio import arms

        self.assertEqual(arms.arm_order(_env("left", "right", work="middle", support="right")), ("right", "left"))

    def test_one_arm_or_no_roles_is_declaration_order(self) -> None:
        from robosimstudio import arms

        self.assertEqual(arms.arm_order(_env("solo", work="solo")), ("solo",))
        self.assertEqual(arms.arm_order(_env("left", "right", handles=False)), ("left", "right"))

    def test_arm_name_clamps_an_out_of_range_id(self) -> None:
        """A bimanual graph that over-counts drives the last arm rather than ending the episode."""
        from robosimstudio import arms

        env = _env("left", "right", work="right")
        self.assertEqual(arms.arm_name(env, 7), "left")
        self.assertEqual(arms.arm_name(env, -3), "right")

    def test_an_arm_the_robot_does_not_have_is_refused_naming_the_ones_it_does(self) -> None:
        from robosimstudio import arms

        env = _env("left", "right", work="right")
        with self.assertRaisesRegex(KeyError, r"no arm 'middle'; have \['left', 'right'\]"):
            arms.arm_slot(env, "middle")
        with self.assertRaisesRegex(KeyError, r"no arm 'middle'; have \['right', 'left'\]"):
            arms.arm_id_of(env, "middle")

    def test_a_scene_without_a_robot_has_no_slots_and_names_its_task(self) -> None:
        from robosimstudio import arms

        env = types.SimpleNamespace(
            scene=types.SimpleNamespace(embodiment=None), task=types.SimpleNamespace(full_name="s/v")
        )
        self.assertEqual(arms.arm_slots(env), ())
        with self.assertRaisesRegex(RuntimeError, "'s/v' has no arms"):
            arms.arm_name(env, 0)


class TestArmRoles(unittest.TestCase):
    def test_the_binding_names_work_and_support_and_nothing_else(self) -> None:
        from robosimstudio import arms

        env = _env("a", "b", "c", work="b", support="c")
        self.assertEqual([arms.arm_role(env, name) for name in ("a", "b", "c", "")], ["", "work", "support", ""])
        self.assertEqual(arms.arm_role(_env("a", handles=False), "a"), "", "no handles, no roles")
        self.assertEqual(arms.arm_role(_env("a", work=""), ""), "", "an unnamed role matches no unnamed arm")


if __name__ == "__main__":
    unittest.main()
