# SPDX-License-Identifier: Apache-2.0
"""A whole cut on the CPU, through the one door into the vendored cage cutter.

``cutting/solver.make_body`` builds the uncut block as a ``cagecut`` body; the
entity then begins a cut in the blade's plane and advances it. Here that is
done by hand, at a quarter of the shipped walks and a third of the quadrature
-- the cut's topology does not depend on either -- and what is held is what the
kind relies on the vendored code for: an uncut block is one piece of its own
volume, a blade through it parts it in two and the two together keep that
volume, and a plane the block does not lie across is refused with
``CutError``, which the entity reads as "no cut here".
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "fast"
"""Warp's CPU device: a few seconds, most of them compiling the cutter's kernels the first time."""

EDGE = (np.array([0.03, -0.01, 0.05]), np.array([0.03, 0.11, 0.05]))
"""A knife's edge across the middle of the 60 x 100 x 40 mm block, 10 mm over its top [m, block frame]."""
DOWN = np.array([0.0, 0.0, -1.0])


def _block():
    """The shipped tofu as a ``Body`` in its scaled frame: ``(body, scale)``."""
    from robosimstudio_materials.cutting.law import BlockFrame
    from robosimstudio_materials.cutting.solver import make_body
    from robosimstudio_materials.cutting.spec import CuttableSpec

    spec = CuttableSpec(name="tofu", walks=64, quad_points=100)
    frame = BlockFrame.from_base((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), spec.size)
    return make_body(spec, frame, dt=1.0 / 60.0), float(frame.scale), float(np.prod(spec.size))


def _blade(body, scale: float, start, line, sweep) -> None:
    """Hand the cutter one unbounded blade, as ``EntityCuttable._begin`` does, and begin it."""
    body.specs = [
        {
            "kind": "blade",
            "p1": np.asarray(start) * scale,
            "p2": (np.asarray(start) + np.asarray(line)) * scale,
            "direction": np.asarray(sweep),
            "step_size": 0.001 * scale,
            "angle": 0.0,
            "bounded": False,
            "plunge": False,
        }
    ]
    body.begin_cut(0)


class TestACutThroughTheBlock(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from robosimstudio_materials.cutting import entity
        from robosimstudio_materials.cutting.cagecut.embed.quadrature import cage_volume
        from robosimstudio_materials.cutting.law import blade_normal, cut_line
        from robosimstudio_materials.cutting.solver import piece_labels

        body, scale, cls.volume = _block()
        snap = body.cutter.snapshot
        cls.before = (np.asarray(piece_labels(snap)), cage_volume(snap) / scale**3)
        alive = np.flatnonzero(snap.face_alive)
        normal = blade_normal(*EDGE, DOWN)
        crossed = [np.ptp(np.sign((snap.verts[snap.face(f)] / scale - EDGE[0]) @ normal)) > 1 for f in alive]
        line, sweep, _, _ = cut_line(*EDGE, DOWN, snap.face_normal[alive], np.asarray(crossed, dtype=bool))
        _blade(body, scale, EDGE[0] - sweep * entity.CLEARANCE, line, sweep)
        cls.steps, done = 0, False
        while not done and cls.steps < 200:
            done = body.advance_cut(0.002 * scale).completed
            cls.steps += 1
        cls.completed = done
        snap = body.cutter.snapshot
        cls.after = (np.asarray(piece_labels(snap)), cage_volume(snap) / scale**3)

    def test_an_uncut_block_is_one_piece_of_its_own_volume(self) -> None:
        labels, volume = self.before
        self.assertEqual([0], np.unique(labels).tolist())
        self.assertAlmostEqual(self.volume, volume, delta=1e-9)

    def test_a_blade_through_it_parts_it_in_two(self) -> None:
        self.assertTrue(self.completed, f"not through after {self.steps} advances")
        labels, _ = self.after
        self.assertEqual([0, 1], np.unique(labels[labels >= 0]).tolist())

    def test_the_two_pieces_keep_the_blocks_volume(self) -> None:
        self.assertAlmostEqual(self.volume, self.after[1], delta=1e-9)


class TestAPlaneTheBlockIsNotAcross(unittest.TestCase):
    def test_is_refused_with_the_cutters_reason(self) -> None:
        from robosimstudio_materials.cutting.solver import CutError

        body, scale, _ = _block()
        along = (0.0, 0.12, 0.0)
        for what, start, message in (
            ("beside it", (0.5, -0.01, 0.05), "does not intersect the cage"),
            ("on its face", (0.0, -0.01, 0.05), "does not intersect the cage"),
            ("under it, going down", (0.03, -0.01, -0.05), "every intersected part lies behind it"),
        ):
            with self.subTest(what), self.assertRaisesRegex(CutError, message):
                _blade(body, scale, start, along, DOWN)


if __name__ == "__main__":
    unittest.main()
