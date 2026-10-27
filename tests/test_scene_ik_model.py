# SPDX-License-Identifier: Apache-2.0
"""The fleet IK's index bookkeeping: a reduced model's coordinates and tool-centre body, lined up with the scene's.

When the scene's own model is too hostile for the batched solver, the IK solves
on a reduced rebuild of the robot in its place, and every answer has to be
written back into the scene's coordinates. :func:`_pair_coords` pairs each arm's
coordinates position by position and refuses two builds that disagree about how
many there are; :func:`_ee_body_in` checks, rather than assumes, that the
tool-centre body is the same link in both. Either check missing aims every
solve at the wrong joint or the wrong link, silently.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.scene.ik_model import IKPlan, _ee_body_in, _pair_coords, plan_ik_model

TIER = "fast"
"""Stand-in scenes and models: the functions read index lists and labels, nothing built."""


def _arm(coords, grip=(), followers=(), ee_body: int = 0):
    return SimpleNamespace(
        arm_coords=list(coords), gripper_coords=list(grip), gripper_follower_coords=tuple(followers), ee_body=ee_body
    )


def _scene(arms: dict, labels=()) -> SimpleNamespace:
    return SimpleNamespace(embodiment=SimpleNamespace(arms=arms), index_model=SimpleNamespace(body_label=list(labels)))


class TestPairingTheCoordinates(unittest.TestCase):
    def test_each_reduced_coordinate_maps_to_the_scene_coordinate_it_is(self) -> None:
        """The scene's arms sit after a cable's 14 coordinates; the reduced build dropped the cable."""
        scene = _scene({"left": _arm(range(14, 20), grip=[20], followers=[21])})
        reduced = _scene({"left": _arm(range(0, 6), grip=[6], followers=[7])})
        mapping = _pair_coords(scene, reduced, SimpleNamespace(joint_coord_count=10))
        np.testing.assert_array_equal(mapping, [14, 15, 16, 17, 18, 19, 20, 21, -1, -1])
        self.assertEqual(mapping.dtype, np.int32)

    def test_two_arms_are_paired_each_with_its_own(self) -> None:
        scene = _scene({"left": _arm([10, 11]), "right": _arm([12, 13])})
        reduced = _scene({"right": _arm([0, 1]), "left": _arm([2, 3])})
        mapping = _pair_coords(scene, reduced, SimpleNamespace(joint_coord_count=4))
        np.testing.assert_array_equal(mapping, [12, 13, 10, 11], "by name, not by the order the arms are listed in")

    def test_an_arm_the_reduced_build_lacks_is_left_unpaired(self) -> None:
        scene = _scene({"left": _arm([4, 5]), "right": _arm([6, 7])})
        mapping = _pair_coords(scene, _scene({"left": _arm([0, 1])}), SimpleNamespace(joint_coord_count=3))
        np.testing.assert_array_equal(mapping, [4, 5, -1])

    def test_builds_that_disagree_about_an_arm_are_refused_by_its_name(self) -> None:
        for reduced in (_arm([0, 1, 2]), _arm([0, 1], grip=[2, 3]), _arm([0, 1], followers=[2])):
            with self.subTest(reduced=reduced):
                with self.assertRaisesRegex(ValueError, r"arm 'left': reduced build has \d+ coordinates where"):
                    _pair_coords(
                        _scene({"left": _arm([8, 9], grip=[10])}),
                        _scene({"left": reduced}),
                        SimpleNamespace(joint_coord_count=6),
                    )


class TestTheToolCentreBody(unittest.TestCase):
    LABELS = ("world_base", "link1", "link2", "tcp")

    def test_on_the_template_it_is_the_scenes_own_index(self) -> None:
        plan = IKPlan(tier="template", model=SimpleNamespace(body_count=0, body_label=[]))
        self.assertEqual(_ee_body_in(plan, _scene({}, self.LABELS), "left", _arm([], ee_body=3)), 3)

    def test_on_a_reduced_model_holding_the_same_link_it_is_the_same_index(self) -> None:
        plan = IKPlan(tier="reduced", model=SimpleNamespace(body_count=4, body_label=list(self.LABELS)))
        self.assertEqual(_ee_body_in(plan, _scene({}, self.LABELS), "left", _arm([], ee_body=3)), 3)

    def test_an_index_past_the_reduced_model_is_refused(self) -> None:
        plan = IKPlan(tier="reduced", model=SimpleNamespace(body_count=3, body_label=list(self.LABELS[:3])))
        with self.assertRaisesRegex(ValueError, r"arm 'left': tool-centre body 3 is outside the reduced IK model \(3"):
            _ee_body_in(plan, _scene({}, self.LABELS), "left", _arm([], ee_body=3))

    def test_another_link_at_that_index_is_refused_naming_both(self) -> None:
        """The silent failure: an index that exists in both but is a different link in each."""
        plan = IKPlan(tier="reduced", model=SimpleNamespace(body_count=4, body_label=["a", "b", "c", "cable_7"]))
        with self.assertRaisesRegex(ValueError, r"body 3 is 'cable_7' in the reduced IK model but 'tcp' in the scene"):
            _ee_body_in(plan, _scene({}, self.LABELS), "left", _arm([], ee_body=3))


class TestThePlan(unittest.TestCase):
    def test_only_the_host_tier_has_no_batched_solve(self) -> None:
        self.assertEqual([IKPlan(tier=t).batched for t in ("template", "reduced", "host")], [True, True, False])

    def test_a_scene_without_a_robot_plans_the_host_tier_and_says_why(self) -> None:
        for embodiment in (None, SimpleNamespace(arms={})):
            with self.subTest(embodiment=embodiment):
                scene = SimpleNamespace(embodiment=embodiment, index_model=None)
                plan = plan_ik_model(None, None, scene)
                self.assertEqual((plan.tier, plan.reason), ("host", "the scene has no robot"))


if __name__ == "__main__":
    unittest.main()
