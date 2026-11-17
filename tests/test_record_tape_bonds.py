# SPDX-License-Identifier: Apache-2.0
"""A tape's bonds are an index array, and recording an episode with one in it records.

``EpisodeRecorder`` writes a cloth's bond anchors once per episode, read off the bond object
the old peel tape carried (``bond_idx``, ``bond_anchor``, ``rows``). ``robosimstudio_materials.tape``
keeps its bonds as a plain integer array, which has none of them, and the recorder raised on
it at the episode's first step and turned itself off: a tape task ran and scored and wrote no
episode.
"""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.record.recorder import EpisodeRecorder
from robosimstudio.specs import RecordSpec

TIER = "fast"


class TestATapesBondsDoNotStopTheRecording(unittest.TestCase):
    def test_anchors_skip_a_bond_index_array(self) -> None:
        tape = SimpleNamespace(carries_particles=True, bonds=np.arange(12, dtype=np.int32))
        env = SimpleNamespace(scene=SimpleNamespace(by_name={"label": tape}))
        with tempfile.TemporaryDirectory() as root:
            recorder = EpisodeRecorder(root, RecordSpec(enabled=True, record_video=False))
            recorder._record_anchors(env)
            self.assertNotIn("label/bond_index", recorder._anchors)


if __name__ == "__main__":
    unittest.main()
