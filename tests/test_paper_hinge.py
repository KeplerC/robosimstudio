# SPDX-License-Identifier: Apache-2.0
"""``robosimstudio_materials/paper.py``: the hinge that yields, in VBD's own angle convention.

One hinge of four particles bent to 90 degrees with a yield of 0.2 rad and rate 1
leaves a rest angle of ``pi/2 - 0.2`` after one launch; with a yield of 0 it
leaves the rest angle alone. The angle it reads is the one Newton computes: the
builder's ``add_edge`` rest angle and a line-for-line port of the VBD bending
kernel agree with it on several poses, sign included. The sheet built on the CPU
and the fold on a GPU are ``tests/test_paper_crease``.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

TIER = "fast"
"""The crease kernel launched on Warp's CPU device over one hinge: no GPU."""


def _hinge(angle: float) -> np.ndarray:
    """Four points: an edge along y, one opposite vertex flat at -x, the other folded up by ``angle``."""
    return np.array(
        [
            [0.0, 0.0, 0.0],  # v0, edge start
            [0.0, 1.0, 0.0],  # v1, edge end
            [-1.0, 0.5, 0.0],  # opposite 0
            [math.cos(angle), 0.5, math.sin(angle)],  # opposite 1
        ],
        dtype=np.float32,
    )


ROW = np.array([[2, 3, 0, 1]], dtype=np.int32)
"""The edge row, ``(opposite0, opposite1, v0, v1)`` as the builder lays it out."""


def _vbd_angle(pos: np.ndarray, row: np.ndarray) -> float:
    """The first lines of Newton's ``evaluate_dihedral_angle_based_bending_force_hessian``, ported."""
    x0, x1, x2, x3 = (pos[int(i)].astype(np.float64) for i in row)
    x02, x03, x13, x12, e = x2 - x0, x3 - x0, x3 - x1, x2 - x1, x3 - x2
    n1, n2 = np.cross(x02, x03), np.cross(x13, x12)
    n1_hat, n2_hat, e_hat = n1 / np.linalg.norm(n1), n2 / np.linalg.norm(n2), e / np.linalg.norm(e)
    return math.atan2(float(np.dot(np.cross(n1_hat, n2_hat), e_hat)), float(np.dot(n1_hat, n2_hat)))


def _launch(pos: np.ndarray, *, rest: float, params: tuple[float, float, float], dt: float = 0.0) -> float:
    """One launch of the crease kernel over one hinge, on the CPU. Returns the rest angle after."""
    import warp as wp

    from robosimstudio_materials.paper import _crease_kernel

    device = "cpu"
    rows = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device)
    edges = wp.array(ROW, dtype=wp.int32, device=device)
    prm = wp.array(np.array(params, dtype=np.float32), dtype=wp.float32, device=device)
    q = wp.array(pos, dtype=wp.vec3, device=device)
    rest_wp = wp.array(np.array([rest], dtype=np.float32), dtype=wp.float32, device=device)
    wp.launch(_crease_kernel, dim=1, inputs=[rows, edges, prm, float(dt), q], outputs=[rest_wp], device=device)
    return float(rest_wp.numpy()[0])


class TestTheHingeKernel(unittest.TestCase):
    """One hinge, on the CPU."""

    def test_a_right_angle_past_a_yield_of_a_fifth_keeps_all_but_the_yield(self) -> None:
        theta = _vbd_angle(_hinge(math.pi / 2), ROW[0])
        self.assertAlmostEqual(abs(theta), math.pi / 2, places=5)
        got = _launch(_hinge(math.pi / 2), rest=0.0, params=(0.2, 1.0, 0.0))
        self.assertAlmostEqual(abs(got), math.pi / 2 - 0.2, places=5)
        self.assertEqual(np.sign(got), np.sign(theta), "the rest angle moves toward the bend, on the bend's side")

    def test_a_yield_of_zero_is_the_off_switch(self) -> None:
        self.assertEqual(0.0, _launch(_hinge(math.pi / 2), rest=0.0, params=(0.0, 1.0, 0.0)))
        self.assertAlmostEqual(0.3, _launch(_hinge(math.pi / 2), rest=0.3, params=(0.0, 1.0, 0.0)), places=6)

    def test_below_the_yield_nothing_is_kept(self) -> None:
        self.assertEqual(0.0, _launch(_hinge(0.15), rest=0.0, params=(0.2, 1.0, 0.0)))

    def test_the_rate_scales_the_step(self) -> None:
        got = _launch(_hinge(math.pi / 2), rest=0.0, params=(0.2, 0.25, 0.0))
        self.assertAlmostEqual(abs(got), 0.25 * (math.pi / 2 - 0.2), places=5)

    def test_recovery_relaxes_the_rest_angle_toward_flat(self) -> None:
        # A yield wide enough that nothing plastic happens, so only the recovery term acts.
        got = _launch(_hinge(0.0), rest=1.0, params=(3.0, 1.0, 2.0), dt=0.1)
        self.assertAlmostEqual(0.8, got, places=5)

    def test_a_fold_past_flat_on_itself_continues_the_same_crease(self) -> None:
        """At a dihedral near +-pi the sign flips with contact noise; the excess is wrapped, so a
        crease at 2.8 rad reading -3.0 rad moves on toward pi rather than to a mirrored -2.8."""
        theta = _vbd_angle(_hinge(-3.0), ROW[0])
        self.assertGreater(theta, 2.9, "folding opposite 1 by -3 rad reads +3 rad in this convention")
        # The crease was written at -2.8, the same fold read before the sign flipped across pi.
        got = _launch(_hinge(-3.0), rest=-2.8, params=(0.2, 1.0, 0.0))
        self.assertLess(got, -2.8, "it continues toward -pi")
        self.assertGreater(got, -math.pi - 0.1)
        self.assertAlmostEqual(-2.8 + ((3.0 + 2.8) - 2.0 * math.pi) + 0.2, got, places=4)

    def test_the_angle_is_newtons_on_four_poses(self) -> None:
        """The kernel's angle against the builder's ``add_edge`` rest angle and the VBD port."""
        import newton
        import warp as wp

        from robosimstudio_materials.paper import dihedral_angles

        for angle in (0.0, math.radians(60.0), math.radians(-135.0), math.pi / 2):
            with self.subTest(angle_deg=round(math.degrees(angle))):
                pos = _hinge(angle)
                builder = newton.ModelBuilder()
                for p in pos:
                    builder.add_particle(wp.vec3(*[float(c) for c in p]), wp.vec3(0.0), 1.0)
                i, j, k, m = (int(v) for v in ROW[0])
                builder.add_triangle(i, k, m)
                builder.add_triangle(j, m, k)
                builder.add_edge(i, j, k, m)
                newtons = float(builder.edge_rest_angle[0])
                ported = _vbd_angle(pos, ROW[0])
                mine = dihedral_angles(
                    wp.array(pos, dtype=wp.vec3, device="cpu"), wp.array(ROW, dtype=wp.int32, device="cpu")
                )
                self.assertAlmostEqual(ported, newtons, places=5)
                self.assertAlmostEqual(float(mine[0]), newtons, places=5)
                if angle:
                    self.assertAlmostEqual(abs(float(mine[0])), abs(angle), places=5)


if __name__ == "__main__":
    unittest.main()
