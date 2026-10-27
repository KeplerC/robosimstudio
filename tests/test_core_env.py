# SPDX-License-Identifier: Apache-2.0
"""``EnvBase``'s verdict on an episode: diverged, asked to end, or out of steps -- and in that order.

The base environment scores nothing; what it decides is whether the episode is
over and how it closed. A diverged solver keeps stepping happily on NaN, so
divergence is checked rather than assumed and outranks everything; a policy may
ask to end, and the first reason it gave is the one kept; ``max_steps``
truncates, and is only the close kind when nothing terminated first.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from robosimstudio.core.env import EnvBase
from tests.fixtures.arrays import host

TIER = "fast"
"""No GPU: an environment over a stand-in pipeline whose arrays are numpy."""


def _env(*, max_steps: int = 0, body_count: int = 1, particle_count: int = 0, **state) -> EnvBase:
    arrays = {"joint_q": [0.0], "joint_qd": [0.0], "body_q": [[0.0] * 7], "particle_q": [[0.0] * 3], **state}
    pipeline = SimpleNamespace(
        model=SimpleNamespace(body_count=body_count, particle_count=particle_count),
        state=SimpleNamespace(**{name: None if value is None else host(value) for name, value in arrays.items()}),
    )
    return EnvBase(pipeline, max_steps=max_steps)


class TestTheVerdict(unittest.TestCase):
    def test_a_healthy_step_mid_episode_is_neither(self) -> None:
        env = _env(max_steps=10)
        env.step_count = 3
        self.assertEqual((False, False, {"success": False, "reward": 0.0}), env.evaluate())

    def test_a_diverged_state_terminates_and_outranks_a_request(self) -> None:
        env = _env(max_steps=10, joint_q=[np.nan])
        env.request_termination("done early")
        env.step_count = 10
        terminated, truncated, info = env.evaluate()
        self.assertEqual((True, True, "diverged"), (terminated, truncated, info["close_kind"]))
        self.assertNotIn("termination_reason", info)

    def test_a_request_terminates_with_the_first_reason_given(self) -> None:
        env = _env()
        env.request_termination("the cup is full")
        env.request_termination("a later thought")
        terminated, _, info = env.evaluate()
        self.assertTrue(terminated)
        self.assertEqual(("policy_request", "the cup is full"), (info["close_kind"], info["termination_reason"]))
        self.assertFalse(info["success"], "a policy may end an episode; declare success it may not")

    def test_the_step_limit_truncates_and_is_the_close_only_when_nothing_terminated(self) -> None:
        env = _env(max_steps=5)
        env.step_count = 5
        terminated, truncated, info = env.evaluate()
        self.assertEqual((False, True, "truncated"), (terminated, truncated, info["close_kind"]))
        env.request_termination("also asked")
        terminated, truncated, info = env.evaluate()
        self.assertEqual((True, True, "policy_request"), (terminated, truncated, info["close_kind"]))

    def test_no_step_limit_never_truncates(self) -> None:
        env = _env(max_steps=0)
        env.step_count = 10**6
        self.assertFalse(env.evaluate()[1])


class TestFiniteness(unittest.TestCase):
    def test_a_nan_anywhere_in_the_rigid_state_is_diverged(self) -> None:
        for name, value in (("joint_q", [np.nan]), ("joint_qd", [np.inf]), ("body_q", [[0.0, np.nan, 0, 0, 0, 0, 1]])):
            with self.subTest(array=name):
                self.assertFalse(_env(**{name: value}).state_is_finite())
        self.assertTrue(_env().state_is_finite())

    def test_particles_count_only_where_the_model_has_them(self) -> None:
        self.assertTrue(_env(particle_q=[[np.nan] * 3]).state_is_finite(), "no particles: the array is not read")
        self.assertFalse(_env(particle_count=1, particle_q=[[np.nan] * 3]).state_is_finite())

    def test_a_scene_of_particles_alone_has_no_rigid_arrays_to_read(self) -> None:
        env = _env(body_count=0, particle_count=1, joint_q=None, joint_qd=None, body_q=None)
        self.assertTrue(env.state_is_finite())

    def test_with_a_readback_the_devices_flag_is_the_answer(self) -> None:
        """The flag the step's own batch carried: no array is read back to ask."""
        env = _env()
        env.pipeline.readback = object()
        env.pipeline.snapshot = SimpleNamespace(is_finite=mock.Mock(return_value=False))
        with mock.patch.object(env.pipeline.state.joint_q, "numpy", side_effect=AssertionError("read the state")):
            self.assertFalse(env.state_is_finite())
        env.pipeline.snapshot.is_finite.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
