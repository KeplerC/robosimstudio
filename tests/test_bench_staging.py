# SPDX-License-Identifier: Apache-2.0
"""A document's staging section: one-clause sub-sections, gains lowered to actuation,
and a variant's staging merged into its suite's clause by clause."""

from __future__ import annotations

import unittest

from robosimstudio_bench.taskbuild.staging import Staging, parse_staging
from robosimstudio_bench.taskdoc import merge_suite, parse_problem, parse_suite
from robosimstudio_bench.taskdoc.reader import parse_sexp

TIER = "fast"
"""Parsing and lowering only: no scene, no GPU."""


def _suite(text):
    return parse_suite(parse_sexp(text))


def _problem(text):
    return parse_problem(parse_sexp(text))


class TestStagingMergesClauseByClause(unittest.TestCase):
    def test_variant_count_keeps_suite_sampler(self):
        suite = _suite(
            "(define (suite s) (:staging (:scene bench_0728)"
            " (:episodes (:count 50) (:sampler foo) (:model-key-fields layout))))"
        )
        doc = merge_suite(_problem("(define (problem v1) (:staging (:episodes (:count 5))))"), suite)
        self.assertEqual(
            doc.staging["episodes"],
            [[":sampler", "foo"], [":model-key-fields", "layout"], [":count", 5]],
        )
        self.assertEqual(doc.staging["scene"], "bench_0728")

    def test_a_value_with_a_head_is_replaced_whole(self):
        suite = _suite("(define (suite s) (:staging (:embodiment yam_bimanual (:work-arm left))))")
        doc = merge_suite(_problem("(define (problem v1) (:staging (:embodiment panda)))"), suite)
        self.assertEqual(doc.staging["embodiment"], "panda")


class TestStagingSubSections(unittest.TestCase):
    """``_plain`` unwraps a single value; a one-clause sub-section is still one clause."""

    def test_one_clause(self):
        self.assertEqual(parse_staging({"episodes": [":count", 50]}).episodes, (("count", 50),))
        self.assertEqual(parse_staging({"episodes": [":sampler", "fixed"]}).episodes, (("sampler", "fixed"),))
        self.assertEqual(parse_staging({"arm-actuation": [":ke", 100]}).arm_actuation, (("ke", 100),))

    def test_two_clauses(self):
        staging = parse_staging({"episodes": [[":count", 50], [":seed", 3]]})
        self.assertEqual(staging.episodes, (("count", 50), ("seed", 3)))
        gains = parse_staging({"arm-actuation": [[":ke", 100], [":kd", 7]]}).arm_actuation
        self.assertEqual(gains, (("ke", 100), ("kd", 7)))

    def test_through_the_reader(self):
        """The same sections as written in a document."""
        from robosimstudio_bench.taskdoc.parse import _plain

        (form,) = parse_sexp("(:staging (:episodes (:count 50)) (:arm-actuation (:ke 100)))")
        staging = parse_staging(_plain(form[1:]))
        self.assertEqual((staging.episodes, staging.arm_actuation), ((("count", 50),), (("ke", 100),)))

    def test_an_effort_limit_lowers_to_the_servo_ceiling(self):
        """``(:effort-limit 12)`` is ``ActuationSpec.effort_limit``, and the scored tape page states its material page's 12 N.

        Before the clause existed the scored document could not say it: ``_pairs`` refused the key, and
        its fingers kept the YAM import's 1e6 N where ``03_materials/08_tape`` caps them at 12 N. Close's
        first push reaches the cap: 1500 times the 47.5 mm to go is about 71 N uncapped against 12 N
        capped, and the finger travels 4.1 mm against 1.9 mm in that first step; the hold after it is
        about 5.8 N on both pages (``g2_close.py``). So the clause is what makes the scored page close
        the way its material page does.
        """
        import importlib.util
        from pathlib import Path

        from robosimstudio.specs import ActuationSpec
        from robosimstudio_bench.taskbuild.lower import _actuation
        from robosimstudio_bench.taskbuild.staging import stage
        from robosimstudio_bench.taskdoc.parse import parse_problem

        gains = parse_staging({"gripper-actuation": [[":ke", 1500], [":kd", 240], [":effort-limit", 12]]})
        self.assertEqual((("ke", 1500), ("kd", 240), ("effort_limit", 12)), gains.gripper_actuation)
        lowered = _actuation(gains.gripper_actuation)
        self.assertEqual(ActuationSpec(target_ke=1500.0, target_kd=240.0, effort_limit=12.0), lowered)
        with self.assertRaisesRegex(ValueError, "effort_limit"):
            parse_staging({"gripper-actuation": [":stiffness", 1]})

        path = Path(__file__).resolve().parents[1] / "examples" / "06_tasks" / "08_tape_scored.py"
        spec = importlib.util.spec_from_file_location("_rss_tape_scored_staging", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # the module body only imports and states TASK
        doc = parse_problem(parse_sexp(module.TASK))
        scored = _actuation(stage(doc.staging).staging.gripper_actuation)
        self.assertEqual(ActuationSpec(target_ke=1500.0, target_kd=240.0, effort_limit=12.0), scored)
        damped = parse_staging({"gripper-actuation": [[":ke", 1500], [":kd", 60], [":joint-damping", 0.5]]})
        self.assertEqual(0.5, _actuation(damped.gripper_actuation).joint_damping)
        rubbed = parse_staging({"gripper-actuation": [[":ke", 1500], [":kd", 60], [":joint-friction", 0.1]]})
        self.assertEqual(0.1, _actuation(rubbed.gripper_actuation).joint_friction)
        with self.assertRaisesRegex(ValueError, "states its gains"):
            _actuation(parse_staging({"gripper-actuation": [":joint-damping", 0.5]}).gripper_actuation)

    def test_an_explicit_false_survives_a_profile_that_says_true(self):
        document = parse_staging({"scene-resident": "false"})
        merged = document.merged_over(Staging(scene_resident=True))
        self.assertIs(merged.scene_resident, False)
        self.assertIs(Staging().merged_over(Staging(scene_resident=True)).scene_resident, True)


if __name__ == "__main__":
    unittest.main()
