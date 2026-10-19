# SPDX-License-Identifier: Apache-2.0
"""A take's checks read what its example's page reads, off the scene the take filmed.

``scripts/recording/takes.py`` ships a clip only when its checks pass, so a check
that reads a different number from the one the example's page shows passes or
fails a clip for the wrong reason. Each check here is run on a stand-in film
holding just the arrays it reads: no GPU, no browser and no scene.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TIER = "fast"
"""Calls the take table's check functions on plain arrays."""

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from recording import takes  # noqa: E402


def _twisted_cables(count: int, per_joint: float) -> SimpleNamespace:
    """Two cables of ``count`` capsules along z, each frame turned ``per_joint`` [rad] past the last."""
    angles = per_joint * np.arange(count)
    frames = np.zeros((count, 7))
    frames[:, 5], frames[:, 6] = np.sin(angles / 2), np.cos(angles / 2)
    body_q = np.concatenate([frames, frames])
    refs = {"soft_twist": np.arange(count), "stiff_twist": np.arange(count, 2 * count)}
    sim = _Sim(body_q, {name: SimpleNamespace(bodies=bodies) for name, bodies in refs.items()})
    return SimpleNamespace(sim=sim, twist_held={"soft_twist": 1.0, "stiff_twist": 1.0})


class _Sim:
    """What a check reads off ``film.sim``: the state's arrays, and things by name."""

    def __init__(self, body_q=None, refs=None) -> None:
        self.state = SimpleNamespace(body_q=SimpleNamespace(numpy=lambda: body_q))
        self._refs = refs or {}

    def __getitem__(self, name: str):
        return self._refs[name]


class TestTheTwistTakeReadsTheMiddleThird(unittest.TestCase):
    """``03_materials/02_twist`` reads each cable through the middle third of its capsules, and so does its take's check."""

    def test_the_third_follows_the_cable_s_length(self) -> None:
        # 30 capsules, a tenth of a radian a joint: the middle third is capsules 10 to 20, ten joints, a radian.
        _, said = takes._twisted(_twisted_cables(30, 0.1))
        self.assertIn(f"let go soft {1.0 / (2 * np.pi):.3f} stiff {1.0 / (2 * np.pi):.3f}", said)

    def test_the_example_s_cable_reads_capsules_8_to_16(self) -> None:
        _, said = takes._twisted(_twisted_cables(24, 0.1))
        self.assertIn(f"let go soft {0.8 / (2 * np.pi):.3f}", said)


if __name__ == "__main__":
    unittest.main()
