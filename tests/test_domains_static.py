# SPDX-License-Identifier: Apache-2.0
"""Where a static stands for one episode, and which statics a replicated build can move.

A fixture board is a task's variable rather than its setting, so an episode may
place it: ``static_poses[name] = (x, y, z, yaw)`` or ``static_xforms[name]`` for
a full pose, in the spec's own frame. :func:`episode_xform` reads that, and a
yaw has to come back as the quaternion that turns the board by that yaw about
the vertical -- read the other way round, a weave board turns the wrong way and
every seat on it is somewhere nobody looked.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from robosimstudio.core.math import quat_to_mat
from robosimstudio.domains.static.entity import MESH_SUFFIXES, episode_xform, placeable_static
from robosimstudio.specs import AssetSpec, StaticSpec

TIER = "fast"
"""A frozen spec and a dict per episode: no model, no GPU."""

DECLARED = (0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 1.0)


def _board(**fields) -> StaticSpec:
    return StaticSpec(name="board", xform=DECLARED, **fields)


class TestAnEpisodePlacesAStatic(unittest.TestCase):
    def test_no_episode_is_the_declared_pose(self) -> None:
        self.assertEqual(DECLARED, episode_xform(_board(), None))

    def test_an_episode_that_does_not_name_it_is_the_declared_pose(self) -> None:
        for episode in ({}, {"static_poses": {"other": (1.0, 1.0, 0.0, 0.5)}}, {"static_xforms": None}):
            with self.subTest(episode=episode):
                self.assertEqual(DECLARED, episode_xform(_board(), episode))

    def test_a_yaw_turns_the_board_about_the_vertical(self) -> None:
        for yaw in (0.0, 0.4, math.pi / 2, -2.5):
            with self.subTest(yaw=yaw):
                pose = episode_xform(_board(), {"static_poses": {"board": (0.3, 0.1, 0.02, yaw)}})
                self.assertEqual((0.3, 0.1, 0.02), pose[:3])
                np.testing.assert_allclose(1.0, np.linalg.norm(pose[3:]), atol=1e-12)
                turned = quat_to_mat(np.asarray(pose[3:])) @ np.array([1.0, 0.0, 0.0])
                np.testing.assert_allclose([math.cos(yaw), math.sin(yaw), 0.0], turned, atol=1e-12)

    def test_a_full_xform_is_taken_as_given_and_beats_a_pose(self) -> None:
        full = (0.5, 0.0, 0.1, 0.0, 0.0, math.sin(0.3), math.cos(0.3))
        episode = {"static_xforms": {"board": (*full, 99.0)}, "static_poses": {"board": (9.0, 9.0, 9.0, 1.0)}}
        self.assertEqual(full, episode_xform(_board(), episode), "seven numbers, and the xform wins")

    def test_the_answer_is_floats(self) -> None:
        pose = episode_xform(_board(), {"static_poses": {"board": (1, 2, 3, 0)}})
        self.assertTrue(all(type(v) is float for v in pose), pose)


class TestOnlyAStaticWithItsOwnGeometryCanBePlaced(unittest.TestCase):
    def test_a_procedural_box_and_a_bare_mesh_can(self) -> None:
        self.assertTrue(placeable_static(_board()))
        for suffix in MESH_SUFFIXES:
            with self.subTest(suffix=suffix):
                self.assertTrue(placeable_static(_board(asset=AssetSpec(path=f"meshes/board{suffix.upper()}"))))

    def test_an_imported_arena_cannot(self) -> None:
        for path in ("arena/bench.xml", "arena/bench.usd", "arena/bench.usda"):
            with self.subTest(path=path):
                self.assertFalse(placeable_static(_board(asset=AssetSpec(path=path))))


if __name__ == "__main__":
    unittest.main()
