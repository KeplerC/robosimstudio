# SPDX-License-Identifier: Apache-2.0
"""The tuning page's readings: a cantilever's droop and bending length, a sheet's spread, its settle speed.

A reading is a number in a unit cloth is specified in, or a hole that says why
it is not one. The cantilever reads its droop at the mid-width of the tip row
against the clamp, as a fraction of the overhang, and only inside the band the
elastica inverts well; a sheet's spread is its footprint against the layout it
was laid out in; the settle speed is a high percentile, so one grain squeezed
between layers does not hold it up; ``relax`` says whether it got there. The
strips and sheets here are point grids laid out by hand, so every answer is a
ruler's: a flat strip has no droop, a sheet drawn in to half its width has half
its spread.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from robosimstudio_materials.cloth import DROOP_BAND
from robosimstudio_studio.tuning import metrics
from robosimstudio_studio.tuning.metrics import Reading, measure_cantilever, measure_sheet, relax, settle_speed
from tests.fixtures.arrays import host

TIER = "fast"
"""No GPU: point grids laid out by hand stand in for a built cloth and its state."""

OVERHANG = 0.10
CLAMP_X = 0.05


def _strip(columns: int = 7, rows: int = 3) -> np.ndarray:
    """A strip's rest layout: x from 0 to 0.15 (clamped to 0.05, so 0.10 overhangs), y across 20 mm, z = 0."""
    x, y = np.meshgrid(np.linspace(0.0, CLAMP_X + OVERHANG, columns), np.linspace(-0.01, 0.01, rows), indexing="ij")
    return np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]


def _cloth(rest: np.ndarray | None, now: np.ndarray, qd: np.ndarray | None = None, **extra):
    """A cloth over every particle, laid out at ``rest``, now at ``now``; and the state it is read from."""
    entity = SimpleNamespace(particles=np.arange(len(now)), **extra)
    if rest is not None:
        entity.rest = {"particle_q": rest}
    state = SimpleNamespace(particle_q=host(now), particle_qd=host(np.zeros_like(now) if qd is None else qd))
    return entity, state


def _drooped(rest: np.ndarray, ratio: float) -> np.ndarray:
    """The strip bent down linearly past the clamp, its tip ``ratio`` of the overhang below the clamp."""
    now = rest.copy()
    past = np.clip(rest[:, 0] - CLAMP_X, 0.0, None)
    now[:, 2] = -ratio * past
    return now


def _named(readings: list[Reading]) -> dict[str, Reading]:
    return {r.name: r for r in readings}


class TestTheCantilever(unittest.TestCase):
    def setUp(self) -> None:
        # The elastica's table takes two seconds to solve, once a process; what is held here is what it is fed.
        self.enterContext(mock.patch.object(metrics, "bending_length_from_droop", return_value=0.05))

    def test_a_strip_lying_flat_has_no_droop_and_is_out_of_band(self) -> None:
        rest = _strip()
        got = _named(measure_cantilever(*_cloth(rest, rest.copy()), overhang=OVERHANG, clamp_x=CLAMP_X))
        self.assertEqual(0.0, got["tip droop"].value)
        self.assertEqual(0.0, got["droop / overhang"].value)
        self.assertFalse(got["bending length"].known)
        self.assertIn("barely bent", got["bending length"].note)

    def test_a_strip_hanging_past_the_band_is_out_of_it_the_other_way(self) -> None:
        rest = _strip()
        ratio = DROOP_BAND[1] + 0.1
        got = _named(measure_cantilever(*_cloth(rest, _drooped(rest, ratio)), overhang=OVERHANG, clamp_x=CLAMP_X))
        self.assertAlmostEqual(ratio, got["droop / overhang"].value)
        self.assertIn(f"droop {ratio:.2f} out of band -- hung straight down", got["bending length"].note)

    def test_the_droop_is_read_at_the_middle_of_the_tip_row_not_its_mean(self) -> None:
        rest = _strip()
        now = _drooped(rest, 0.3)
        tip = np.flatnonzero(rest[:, 0] == rest[:, 0].max())
        now[tip[[0, -1]], 2] += 0.02  # the free corners curl up 20 mm
        got = _named(measure_cantilever(*_cloth(rest, now), overhang=OVERHANG, clamp_x=CLAMP_X))
        self.assertAlmostEqual(0.03, got["tip droop"].value, places=12)

    def test_the_tip_is_the_rest_layouts_however_far_it_has_swung_back(self) -> None:
        rest = _strip()
        now = _drooped(rest, 0.5)
        tip = rest[:, 0] == rest[:, 0].max()
        now[tip, 0] -= 0.04  # hanging, the tip is no longer the strip's furthest x
        got = _named(measure_cantilever(*_cloth(rest, now), overhang=OVERHANG, clamp_x=CLAMP_X))
        self.assertAlmostEqual(0.5, got["droop / overhang"].value)

    def test_a_droop_in_band_is_inverted_into_centimetres_and_named_after_its_fabrics(self) -> None:
        rest = _strip()
        with (
            mock.patch.object(metrics, "bending_length_from_droop", return_value=0.042) as invert,
            mock.patch.object(metrics, "resembles", return_value=("denim", "canvas")),
        ):
            got = _named(measure_cantilever(*_cloth(rest, _drooped(rest, 0.3)), overhang=OVERHANG, clamp_x=CLAMP_X))
        (ratio, overhang), _ = invert.call_args
        self.assertAlmostEqual(0.3, ratio)
        self.assertEqual(OVERHANG, overhang)
        self.assertAlmostEqual(4.2, got["bending length"].value)
        self.assertEqual(("cm", "like denim, canvas"), (got["bending length"].unit, got["bending length"].note))

    def test_a_length_no_reference_fabric_covers_says_so(self) -> None:
        rest = _strip()
        with (
            mock.patch.object(metrics, "bending_length_from_droop", return_value=0.042),
            mock.patch.object(metrics, "resembles", return_value=()),
        ):
            got = _named(measure_cantilever(*_cloth(rest, _drooped(rest, 0.3)), overhang=OVERHANG, clamp_x=CLAMP_X))
        self.assertEqual("no reference fabric in this band", got["bending length"].note)

    def test_a_rig_without_a_rest_layout_or_an_overhang_is_not_measurable(self) -> None:
        rest = _strip()
        for entity_state, overhang in (
            (_cloth(None, rest.copy()), OVERHANG),
            (_cloth(rest, rest.copy()), 0.0),
        ):
            (only,) = measure_cantilever(*entity_state, overhang=overhang, clamp_x=CLAMP_X)
            self.assertEqual(("bending length", "rig not measurable"), (only.name, only.note))
            self.assertFalse(only.known)

    def test_a_clamp_short_of_every_particle_has_no_root_row(self) -> None:
        rest = _strip()
        (only,) = measure_cantilever(*_cloth(rest, rest.copy()), overhang=OVERHANG, clamp_x=-1.0)
        self.assertEqual("no tip or clamp row", only.note)


class TestASheet(unittest.TestCase):
    def _grid(self, width: float = 0.2, depth: float = 0.1) -> np.ndarray:
        x, y = np.meshgrid(np.linspace(0.0, width, 5), np.linspace(0.0, depth, 4), indexing="ij")
        return np.c_[x.ravel(), y.ravel(), np.full(x.size, 0.3)]

    def test_as_laid_out_it_has_spread_one_its_own_footprint_and_no_height(self) -> None:
        rest = self._grid()
        got = _named(measure_sheet(*_cloth(rest, rest.copy())))
        self.assertAlmostEqual(1.0, got["spread"].value)
        self.assertAlmostEqual(0.2 * 0.1, got["footprint"].value)
        self.assertEqual(0.0, got["height"].value)
        self.assertEqual("spread", measure_sheet(*_cloth(rest, rest.copy()))[0].name, "spread reads first")

    def test_drawn_in_to_half_its_width_it_has_half_its_spread(self) -> None:
        rest = self._grid()
        now = rest.copy()
        now[:, 0] *= 0.5
        now[:, 2] += 0.05 * (now[:, 0] > 0.05)  # and it bunched up 50 mm
        got = _named(measure_sheet(*_cloth(rest, now)))
        self.assertAlmostEqual(0.5, got["spread"].value)
        self.assertAlmostEqual(0.05, got["height"].value)

    def test_without_a_rest_layout_there_is_no_spread(self) -> None:
        rest = self._grid()
        self.assertNotIn("spread", _named(measure_sheet(*_cloth(None, rest.copy()))))

    def test_a_cloth_with_no_particles_reads_nothing(self) -> None:
        self.assertEqual([], measure_sheet(*_cloth(None, np.zeros((0, 3)))))

    def test_a_slab_is_read_on_its_mid_surface_not_a_face(self) -> None:
        rest = self._grid()
        up = np.array([0.0, 0.0, 0.001])
        faces = np.concatenate([rest, rest + 4.0 * up])  # a 4 mm slab, a particle on each face
        entity, state = _cloth(None, faces, mid_surface_positions=lambda state: rest + 2.0 * up)
        self.assertEqual(0.0, _named(measure_sheet(entity, state))["height"].value)


def _mid_of_two_faces(state) -> np.ndarray:
    """A slab's mid-surface, as a slab cloth reads it: the average of the particle on each face."""
    q = np.asarray(state.particle_q.numpy(), dtype=np.float64)
    half = len(q) // 2
    return 0.5 * (q[:half] + q[half:])


def _slab(rest: np.ndarray, now: np.ndarray) -> tuple:
    """A 4 mm slab over a mid-surface laid out at ``rest`` and now at ``now``: a particle on each face."""
    up = np.array([0.0, 0.0, 0.002])
    entity, state = _cloth(np.concatenate([rest - up, rest + up]), np.concatenate([now - up, now + up]))
    entity.mid_surface_positions = _mid_of_two_faces
    return entity, state


class TestASlab(unittest.TestCase):
    """A slab is read on its mid-surface, and so is the rest layout it is held against."""

    def test_a_slab_as_laid_out_has_spread_one(self) -> None:
        x, y = np.meshgrid(np.linspace(0.0, 0.2, 5), np.linspace(0.0, 0.1, 4), indexing="ij")
        rest = np.c_[x.ravel(), y.ravel(), np.full(x.size, 0.3)]
        got = _named(measure_sheet(*_slab(rest, rest.copy())))
        self.assertIn("spread", got, "the rest layout was held against the mid-surface face for face")
        self.assertAlmostEqual(1.0, got["spread"].value)

    def test_a_slab_on_the_cantilever_rig_is_measured(self) -> None:
        rest = _strip()
        with mock.patch.object(metrics, "bending_length_from_droop", return_value=0.05):
            got = _named(measure_cantilever(*_slab(rest, _drooped(rest, 0.5)), overhang=OVERHANG, clamp_x=CLAMP_X))
        self.assertIn("droop / overhang", got, got)
        self.assertAlmostEqual(0.5, got["droop / overhang"].value, places=9)


class TestTheSettleSpeed(unittest.TestCase):
    def test_it_is_the_95th_percentile_so_one_squeezed_particle_does_not_hold_it_up(self) -> None:
        qd = np.zeros((100, 3))
        qd[:, 0] = np.arange(100) * 0.001
        qd[99, 0] = 50.0  # one particle shot out from between two layers
        entity, state = _cloth(None, np.zeros((100, 3)), qd)
        speed = settle_speed(entity, state)
        self.assertAlmostEqual(np.percentile(np.arange(100) * 0.001, 95), speed.value, places=6)
        self.assertEqual("m/s", speed.unit)

    def test_with_no_particles_it_is_not_known(self) -> None:
        speed = settle_speed(*_cloth(None, np.zeros((0, 3))))
        self.assertFalse(speed.known)
        self.assertEqual("-- no particles", speed.text())


class _Velocities:
    def __init__(self, n: int) -> None:
        self.zeroed = 0
        self.values = np.ones((n, 3))

    def numpy(self) -> np.ndarray:
        return self.values

    def zero_(self) -> None:
        self.zeroed += 1
        self.values[:] = 0.0


def _pipeline(drift: float, n: int = 4) -> SimpleNamespace:
    """A cloth that moves ``drift`` metres along x every step."""
    pipeline = SimpleNamespace(state=SimpleNamespace(particle_q=host(np.zeros((n, 3))), particle_qd=_Velocities(n)))
    pipeline.steps = 0

    def step() -> None:
        pipeline.steps += 1
        pipeline.state.particle_q.data[:, 0] += drift

    pipeline.step = step
    return pipeline


class TestRelax(unittest.TestCase):
    def test_a_cloth_that_never_stops_runs_out_of_steps_and_says_so(self) -> None:
        pipeline = _pipeline(drift=1e-3)
        self.assertEqual((False, 30), relax(pipeline, max_steps=30))
        self.assertEqual(30, pipeline.steps)
        self.assertEqual(10, pipeline.state.particle_qd.zeroed, "its velocities zeroed every third step")

    def test_a_still_cloth_has_settled_at_the_first_damping(self) -> None:
        pipeline = _pipeline(drift=0.0)
        self.assertEqual((True, 3), relax(pipeline, damp_every=3))

    def test_settled_means_moved_less_than_the_tolerance_between_dampings(self) -> None:
        self.assertEqual((True, 2), relax(_pipeline(drift=1e-7), damp_every=2, tol=1e-6))
        self.assertFalse(relax(_pipeline(drift=1e-6), damp_every=2, tol=1e-6, max_steps=8)[0])


class TestAReading(unittest.TestCase):
    def test_a_known_value_reads_with_its_unit_and_note(self) -> None:
        self.assertEqual("4.2 cm  (like denim)", Reading("bending length", 4.2, "cm", "like denim").text())
        self.assertEqual("0.5", Reading("spread", 0.5).text())

    def test_an_unknown_one_reads_as_why(self) -> None:
        unknown = Reading("bending length", math.nan, "cm", "rig not measurable")
        self.assertFalse(unknown.known)
        self.assertEqual("-- rig not measurable", unknown.text())


if __name__ == "__main__":
    unittest.main()
