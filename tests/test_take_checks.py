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
    """``04_twist`` reads each cable through the middle third of its capsules, and so does its take's check."""

    def test_the_third_follows_the_cable_s_length(self) -> None:
        # 30 capsules, a tenth of a radian a joint: the middle third is capsules 10 to 20, ten joints, a radian.
        _, said = takes._twisted(_twisted_cables(30, 0.1))
        self.assertIn(f"let go soft {1.0 / (2 * np.pi):.3f} stiff {1.0 / (2 * np.pi):.3f}", said)

    def test_the_example_s_cable_reads_capsules_8_to_16(self) -> None:
        _, said = takes._twisted(_twisted_cables(24, 0.1))
        self.assertIn(f"let go soft {0.8 / (2 * np.pi):.3f}", said)


class TestTheFlapTakeMeasuresFromItsBench(unittest.TestCase):
    """``01_box_flap``'s check reads the rim off the bench the scene stands on, as the takes' other checks do."""

    def test_an_edge_on_the_rim_reads_zero_on_a_taller_bench(self) -> None:
        from unittest import mock

        rest = np.array([[0.0, 0.0, 0.10], [0.0, 0.0, 0.20], [0.01, 0.0, 0.20]])
        now = rest + np.array([0.6, 0.0, 0.75])  # the flap shut, its free edge on the rim, bench_075 under it
        flap = SimpleNamespace(
            entity=SimpleNamespace(update_probe=lambda *_: None, rest={"particle_q": rest}),
            particles=np.arange(3),
            particle_q=np.concatenate([now[:1], now[1:] - np.array([0.0, 0.0, 0.10])]),
        )
        sim = _Sim(refs={"flap": flap})
        sim.step_count, sim.make_context = 0, lambda: None
        film = SimpleNamespace(sim=sim, bench=lambda p: (p[0] + 0.6, p[1], p[2] + 0.75))
        with (
            mock.patch("robosimstudio.assetlib.semantics.snapshot.Snapshot"),
            mock.patch("robosimstudio_materials.paper.crease_angle", return_value={"angle": 2.0}),
        ):
            ok, said = takes._flap_shut(film)
        self.assertTrue(ok, said)
        self.assertIn("+0 mm off the rim", said)


if __name__ == "__main__":
    unittest.main()
