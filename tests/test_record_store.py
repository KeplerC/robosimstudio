# SPDX-License-Identifier: Apache-2.0
"""Reading an episode back: what an offline re-render replays."""

from __future__ import annotations

import unittest

import numpy as np

TIER = "fast"
"""No GPU: an ``EpisodeData`` over numpy arrays, nothing on disk."""


class TestReplaySteps(unittest.TestCase):
    """What an offline re-render replays, and what it must not take a length from.

    The bug this guards: ``privileged.npz`` carries an episode's *constants*
    beside its per-step state -- ``<cloth>/bond_index``, ``<cloth>/bond_anchor``,
    ``<cloth>/bond_row``, ``particle_subset`` -- and the slicer behind
    :meth:`EpisodeData.replay_steps` truncates to its shortest array. A 340-step
    tape peel with 45 bonds replayed as 45 steps: no error and no warning, just
    the first eighth of the episode and then the end of the clip.
    """

    def _episode(self, **privileged):
        from pathlib import Path

        from robosimstudio.record.store import EpisodeData

        return EpisodeData(path=Path("."), meta={}, steps={}, privileged=privileged)

    def test_an_episode_constant_does_not_shorten_the_replay(self):
        """Twelve steps stay twelve steps next to four bonds and a subset."""
        episode = self._episode(
            body_q=np.zeros((12, 3, 7), dtype=np.float32),
            particle_q=np.zeros((12, 5, 3), dtype=np.float32),
            tape_bond_active=np.zeros((12, 4), dtype=np.uint8),
            particle_subset=np.arange(5, dtype=np.int32),
            **{
                "tape/bond_index": np.zeros(4, dtype=np.int32),
                "tape/bond_anchor": np.zeros((4, 3), dtype=np.float32),
                "tape/bond_row": np.zeros(4, dtype=np.int32),
            },
        )
        steps = list(episode.replay_steps())
        self.assertEqual(len(steps), 12)
        # And narrowed to what a state can be written from: a per-step channel
        # no replay reads is still a length nothing should be measured against.
        self.assertEqual(sorted(steps[0]), ["body_q", "particle_q"])

    def test_a_recording_cut_mid_write_still_truncates(self):
        """The shortest *replayable* channel is still the honest length.

        Truncation is the point when a run was interrupted: the last step's
        particles may never have been written, and rendering a body pose against
        a cloth from another step would be a frame that never existed.
        """
        episode = self._episode(
            body_q=np.zeros((12, 3, 7), dtype=np.float32),
            particle_q=np.zeros((7, 5, 3), dtype=np.float32),
        )
        self.assertEqual(len(list(episode.replay_steps())), 7)


if __name__ == "__main__":
    unittest.main()
