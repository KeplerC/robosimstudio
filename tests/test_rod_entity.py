# SPDX-License-Identifier: Apache-2.0
"""The capsule-chain rod held to what it says, against stand-ins built without a scene.

The four spawn ranges are independent, a rod whose centreline cannot be read
does not publish 0 m, the grip shapes are found off one copy of
``shape_body``, and the anchor latches with the hand it posed, and only when
the rod permits a latch.

No GPU: the spawn is numpy and Warp's host quaternions, and the entity is a
stand-in for the arm, its solver and the pipeline.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

TIER = "fast"
"""numpy, Warp's host maths and stand-ins: no GPU."""


def _straight(n: int = 8):
    import warp as wp

    points = [wp.vec3(0.0, 0.1 * i, 0.01) for i in range(n)]
    quats = [wp.quat_identity() for _ in range(n)]
    return points, quats


def _spec(**ranges):
    return SimpleNamespace(direction=(0.0, 1.0, 0.0), **ranges)


class TestTheSpawnRangesAreIndependent(unittest.TestCase):
    """``rod/entity.py::_bow_cable``: a shift or a yaw needs no bow to happen."""

    def test_a_shift_without_a_bow_moves_the_rod(self) -> None:
        from robosimstudio_materials.rod.entity import _bow_cable

        points, quats = _straight()
        spec = _spec(shift_range=(-0.02, 0.02))
        got, _ = _bow_cable(points, quats, spec, SimpleNamespace(overrides={"cable_shift": 1.0}))
        moved = np.asarray([list(p) for p in got]) - np.asarray([list(p) for p in points])
        # Lateral to a rod along +y is -x: up x along.
        np.testing.assert_allclose(moved, np.tile([-0.02, 0.0, 0.0], (len(points), 1)), atol=1e-9)

    def test_a_yaw_without_a_bow_turns_the_rod(self) -> None:
        from robosimstudio_materials.rod.entity import _bow_cable

        points, quats = _straight()
        spec = _spec(yaw_range=(0.0, 20.0))
        got, _ = _bow_cable(points, quats, spec, SimpleNamespace(overrides={"cable_yaw": 1.0}))
        pts = np.asarray([list(p) for p in got])
        chord = pts[-1] - pts[0]
        self.assertAlmostEqual(math.degrees(math.atan2(-chord[0], chord[1])), 20.0, places=4)

    def test_a_bow_drawn_at_zero_keeps_the_shift(self) -> None:
        from robosimstudio_materials.rod.entity import _bow_cable

        points, quats = _straight()
        spec = _spec(bow_range=(-0.02, 0.02), shift_range=(0.0, 0.03))
        episode = SimpleNamespace(overrides={"cable_bow": 0.0, "cable_shift": 1.0})
        got, _ = _bow_cable(points, quats, spec, episode)
        self.assertAlmostEqual(float(got[3][0]) - float(points[3][0]), -0.03, places=6)

    def test_nothing_drawn_is_the_same_objects(self) -> None:
        from robosimstudio_materials.rod.entity import _bow_cable

        points, quats = _straight()
        spec = _spec(shift_range=(-0.02, 0.02), yaw_range=(-5.0, 5.0))
        got_p, got_q = _bow_cable(points, quats, spec, SimpleNamespace(overrides={}))
        self.assertIs(got_p, points)
        self.assertIs(got_q, quats)


class _Rod:
    """An ``EntityRod`` with only what ``published_states`` and ``centerline`` read."""

    @staticmethod
    def make(n_bodies: int, n_half: int):
        from robosimstudio_materials.rod.entity import EntityRod

        class Rod(EntityRod):
            name = "cable"

        rod = object.__new__(Rod)
        rod.bodies = np.arange(n_bodies)
        rod._half_heights = np.full(n_half, 0.006)
        rod._half_heights_per_world = {}
        rod._hold_flag = None
        return rod


class TestTheLengthIsNotInvented(unittest.TestCase):
    """``rod/entity.py::published_states``: a rod it cannot read is an error, not a 0 m rod."""

    def test_a_mismatched_rod_raises(self) -> None:
        rod = _Rod.make(4, 3)
        snap = SimpleNamespace(body_q=np.tile([0, 0, 0, 0, 0, 0, 1.0], (4, 1)), world=0)
        with self.assertRaises(RuntimeError):
            rod.published_states(snap)

    def test_a_built_rod_reports_its_length(self) -> None:
        rod = _Rod.make(4, 4)
        body_q = np.tile([0, 0, 0, 0, 0, 0, 1.0], (4, 1))
        body_q[:, 2] = 0.012 * np.arange(4) + 0.006  # capsules end to end up +z
        got = rod.published_states(SimpleNamespace(body_q=body_q, world=0))
        self.assertAlmostEqual(got["length_m"], 0.048, places=9)
        self.assertEqual(got["n_segments"], 4.0)

    def test_no_poses_leaves_the_length_out(self) -> None:
        got = _Rod.make(4, 4).published_states(None)
        self.assertNotIn("length_m", got)
        self.assertEqual(got["held"], 0.0)


class TestTheBladesAreReadOnce(unittest.TestCase):
    """``rod/entity.py::_blade_shapes``: one host copy of ``shape_body``, not one per shape."""

    def test_shape_body_is_copied_once(self) -> None:
        from robosimstudio_materials.rod.entity import EntityRod

        calls = []

        class Body:
            def numpy(self):
                calls.append(1)
                return np.array([0, 1, 1, 2, -1])

        model = SimpleNamespace(
            body_label=["a/base", "a/left_tip", "a/palm"], body_count=3, shape_count=5, shape_body=Body()
        )
        fake = SimpleNamespace(_hand_body_labels=lambda: ({"left_tip"}, set()))
        self.assertEqual(EntityRod._blade_shapes(fake, model), {1, 2})
        self.assertEqual(len(calls), 1)


class _Array:
    """A device array as far as ``_pose_anchor`` reads and writes one."""

    def __init__(self, n: int) -> None:
        self.values = np.zeros(n)

    def numpy(self) -> np.ndarray:
        return self.values.copy()

    def assign(self, values) -> None:
        self.values = np.asarray(values, dtype=np.float64)


def _pose(*, allow_grasp_latch: bool, arm: str = "right"):
    """``EntityRod._pose_anchor`` against a stand-in rod; returns the ``hold_segment`` mock."""
    from robosimstudio_materials.rod import entity as rod

    ik = SimpleNamespace(
        solve_q=lambda seed, target, approach: np.zeros(4),
        fk_ee=lambda q: (np.zeros(3), None),
    )
    index = SimpleNamespace(arm_coords=(0, 1, 2), gripper_coords=(3,))
    embodiment = SimpleNamespace(iks={arm: ik}, arms={arm: index})
    hold = mock.Mock()
    fake = SimpleNamespace(
        spec=SimpleNamespace(anchor_grasp=(0.0, 0.0, 0.0), anchor_grip=0.0, allow_grasp_latch=allow_grasp_latch),
        embodiment=embodiment,
        worlds=None,
        _fleet_model=None,
        _hold_flag=object(),
        _anchor_arm=lambda: arm,
        placed=lambda pose: np.asarray(pose, dtype=np.float64),
        _coord=lambda c, world: int(c),
        hold_segment=hold,
    )
    state = SimpleNamespace(joint_q=_Array(4), joint_qd=None, body_q=None)
    pipeline = SimpleNamespace(
        state_0=state,
        state_1=SimpleNamespace(joint_q=_Array(4), body_q=None),
        control=SimpleNamespace(joint_target_q=_Array(4)),
    )
    with mock.patch.object(rod.newton, "eval_fk"), mock.patch.object(rod.wp, "copy"):
        rod.EntityRod._pose_anchor(fake, pipeline, world=0)
    return hold, pipeline


class TestTheAnchorLatchesTheHandItPosed(unittest.TestCase):
    """``rod/entity.py``: the anchor's latch follows ``_anchor_arm`` and ``allow_grasp_latch``."""

    def test_the_latch_is_on_the_resolved_arm(self) -> None:
        hold, pipeline = _pose(allow_grasp_latch=True, arm="right")
        hold.assert_called_once_with(True, pipeline.state_0, arm="right", slot=1, world=0)

    def test_a_contact_only_cable_is_not_latched(self) -> None:
        hold, _ = _pose(allow_grasp_latch=False)
        hold.assert_not_called()


if __name__ == "__main__":
    unittest.main()
