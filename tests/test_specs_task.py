# SPDX-License-Identifier: Apache-2.0
"""The task-side specs: an arrangement's rates, a suite's variants, where parked objects go, and an episode's overrides.

A suite stamps its name on every variant and refuses, where it is written, a
variant table or a binding table that would only fail later on whichever
pairing somebody selected first. Parking is a function of the document alone:
stated poses first, then one lattice, every pose distinct. An episode's
``active`` set is ``None`` for "every entity".
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "fast"
"""No GPU: frozen dataclasses."""


def _task(name: str = "v", *objects: str, **fields):
    from robosimstudio.domains.rigid import box
    from robosimstudio.specs import TaskSpec

    return TaskSpec(
        name=name, scene="bench_075", embodiment="panda", entities=tuple(box(name=o) for o in objects), **fields
    )


class TestAnArrangementStatesItsRates(unittest.TestCase):
    def test_substeps_must_be_a_positive_whole_number(self) -> None:
        from robosimstudio.specs import ArrangementSpec

        for bad in (0, -2, 3.5):
            with self.assertRaises(ValueError):
                ArrangementSpec(substeps=bad)
        with self.assertRaises(ValueError):
            ArrangementSpec(control_hz=0.0)


class TestASuiteStampsAndRefuses(unittest.TestCase):
    def test_every_variant_carries_the_suites_name(self) -> None:
        from robosimstudio.specs import TaskSuite

        suite = TaskSuite(name="tools", variants=(_task("pick"), _task("place")))
        self.assertEqual([v.full_name for v in suite.variants], ["tools/pick", "tools/place"])
        self.assertEqual(suite.variant_names, ("pick", "place"))
        self.assertIs(suite.variant("place"), suite.variants[1])

    def test_an_unknown_variant_is_refused_naming_the_ones_there_are(self) -> None:
        from robosimstudio.specs import TaskSuite

        suite = TaskSuite(name="tools", variants=(_task("pick"),))
        with self.assertRaisesRegex(KeyError, r"suite 'tools' has no variant 'drop'; has \['pick'\]"):
            suite.variant("drop")

    def test_a_suite_of_nothing_is_refused(self) -> None:
        from robosimstudio.specs import TaskSuite

        with self.assertRaisesRegex(ValueError, "suite 'empty' declares no variants"):
            TaskSuite(name="empty", variants=())

    def test_two_variants_of_one_name_are_refused_by_that_name(self) -> None:
        from robosimstudio.specs import TaskSuite

        with self.assertRaisesRegex(ValueError, r"duplicate variants \['pick'\]"):
            TaskSuite(name="tools", variants=(_task("pick"), _task("place"), _task("pick")))

    def test_the_first_binding_must_be_the_native_pairing(self) -> None:
        from robosimstudio.specs import Binding, TaskSuite

        with self.assertRaisesRegex(ValueError, "bindings\\[0\\] must be the native pairing"):
            TaskSuite(name="s", variants=(_task(bindings=(Binding(embodiment="ur5e"),)),))
        TaskSuite(name="s", variants=(_task(bindings=(Binding(), Binding(embodiment="ur5e"))),))

    def test_the_native_binding_may_not_restate_the_tasks_gains(self) -> None:
        from robosimstudio.specs import ActuationSpec, Binding, TaskSuite

        gains = ActuationSpec(target_ke=100.0, target_kd=10.0)
        for field in ("arm_actuation", "gripper_actuation"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "must not override actuation"):
                TaskSuite(name="s", variants=(_task(bindings=(Binding(**{field: gains}),)),))

    def test_a_pairing_bound_twice_is_refused(self) -> None:
        from robosimstudio.specs import Binding, TaskSuite

        with self.assertRaisesRegex(ValueError, "duplicate binding for scene='bench_075' embodiment='ur5e'"):
            TaskSuite(
                name="s",
                variants=(_task(bindings=(Binding(), Binding(embodiment="ur5e"), Binding(embodiment="ur5e"))),),
            )

    def test_a_binding_may_claim_only_graphs_the_task_ships(self) -> None:
        from robosimstudio.specs import Binding, TaskSuite

        shipped = (("gt", "graphs/gt"),)
        TaskSuite(name="s", variants=(_task(graphs=shipped, bindings=(Binding(graphs=("gt",)),)),))
        with self.assertRaisesRegex(ValueError, r"claims graphs \['learned'\] the task does not ship"):
            TaskSuite(name="s", variants=(_task(graphs=shipped, bindings=(Binding(graphs=("gt", "learned")),)),))


class TestParking(unittest.TestCase):
    def test_a_stated_pose_beats_the_lattice_and_the_rest_keep_their_slots(self) -> None:
        from robosimstudio.specs import park_lattice, park_poses

        stated = (0.0, -3.0, 2.0, 0.0, 0.0, 0.0, 1.0)
        poses = park_poses(_task("v", "a", "b", "c", parking=(("b", stated),)))
        self.assertEqual(poses["b"], stated)
        self.assertEqual(poses["a"], park_lattice(0))
        self.assertEqual(poses["c"], park_lattice(2), "c keeps its own index; b's stated pose frees no slot")

    def test_only_objects_are_parked(self) -> None:
        from robosimstudio.specs import StaticSpec, park_poses

        task = _task("v", "a")
        task = task.with_entities(statics=(StaticSpec(name="post"),))
        self.assertEqual(set(park_poses(task)), {"a"})

    def test_every_lattice_slot_is_distinct_level_and_far_overhead(self) -> None:
        from robosimstudio.specs import PARK_HEIGHT, PARK_PITCH, park_lattice

        slots = [park_lattice(i) for i in range(20)]
        self.assertEqual(len(set(slots)), 20)
        for slot in slots:
            self.assertEqual(slot[2:], (PARK_HEIGHT, 0.0, 0.0, 0.0, 1.0))
        gaps = [np.hypot(a[0] - b[0], a[1] - b[1]) for i, a in enumerate(slots) for b in slots[i + 1 :]]
        self.assertGreaterEqual(min(gaps), PARK_PITCH - 1e-12, "no two parked bodies closer than the pitch")

    def test_a_row_is_centred_on_the_world_origin(self) -> None:
        from robosimstudio.specs import park_lattice
        from robosimstudio.specs import task as task_module

        row = [park_lattice(i)[0] for i in range(task_module._PARK_ROW)]
        self.assertAlmostEqual(sum(row), 0.0, places=12)
        self.assertEqual(park_lattice(task_module._PARK_ROW)[:2], (row[0], task_module.PARK_PITCH))


class TestAnEpisodesOverrides(unittest.TestCase):
    def test_no_active_override_means_every_entity(self) -> None:
        from robosimstudio.specs import Episode

        self.assertIsNone(Episode(episode_id=0, seed=0).active)
        self.assertIsNone(Episode(episode_id=0, seed=0, overrides=(("other", 1),)).active)

    def test_the_active_set_is_the_names_as_strings(self) -> None:
        from robosimstudio.specs import Episode

        episode = Episode(episode_id=3, seed=1, overrides=(("active", ("hammer", "drill", "hammer")),))
        self.assertEqual(episode.active, frozenset({"hammer", "drill"}))
        self.assertEqual(Episode(episode_id=0, seed=0, overrides=(("active", ()),)).active, frozenset(), "none run")

    def test_get_and_as_dict_read_the_same_overrides(self) -> None:
        from robosimstudio.specs import Episode

        episode = Episode(episode_id=0, seed=0, overrides=(("mass", 0.2), ("color", "red")))
        self.assertEqual((episode.get("mass"), episode.get("absent", "fallback")), (0.2, "fallback"))
        self.assertEqual(episode.as_dict, {"mass": 0.2, "color": "red"})


if __name__ == "__main__":
    unittest.main()
