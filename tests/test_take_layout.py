# SPDX-License-Identifier: Apache-2.0
"""Each take reads its example's layout off the built scene, and reads the numbers it used to restate.

``scripts/recording/takes.py`` once copied the examples' layouts into constants: the tape box's top,
the label's tab, the suction square, the A4 sheet's corner, the knife's handle and the board's top,
the bowl, the glass, the tray, the pegs, the post, the tied knot. A take now reads each off the spec
its example's scene was built from (:func:`recording.beats.laid`), so an example that moves a box
moves the take with it (``tests/test_take_checks`` moves them). This builds each example and checks
every number a take now derives is the constant it replaced, to a micrometre: the takes do exactly
what they did, and their clips stand.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from tests.fixtures.examples import _run

TIER = "medium"
"""Builds twelve example scenes on one GPU, then seven scored scripts beside their materials, and steps none of them."""

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from recording import takes  # noqa: E402
from recording.beats import At, Drag, DragTo, Top, resolve  # noqa: E402

TOLERANCE = 1e-6
"""[m]: the float32 a built body's pose is kept in, and nothing a hand would feel."""


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


HAS_SUPERDEX, HAS_GENESIS = _importable("superdex"), _importable("genesis")


def _film(rel: str, key: str = ""):
    """The example built, and a film over it with the recorder's own ``bench``: what a take reads at the page."""
    from recording.lockstep import Film

    sim, _ = _run(rel)
    take = takes.find(key) if key else None
    film = Film(take, port=0, scratch=None, software=False, probe=False, log=print)
    film.sim = sim
    return film


def _bench(film, point) -> tuple[float, float, float]:
    """A world point in the bench frame: the bench's origin taken off, as the takes' checks do."""
    origin = film.bench((0.0, 0.0, 0.0))
    return tuple(float(p) - float(o) for p, o in zip(point, origin, strict=True))


def _drags(film, beats=None) -> list[tuple[str, float]]:
    """Each drag of the film's take (or of ``beats``) to a point of its layout: its axis, and the bench coordinate along it."""
    out = []
    for beat in film.take.beats if beats is None else beats:
        if isinstance(beat, Drag) and beat.to is not None and not isinstance(beat.to, At | Top):
            i = "xyz".index(beat.axis)
            out.append((beat.axis, _bench(film, resolve(beat.to, film))[i]))
        elif isinstance(beat, DragTo):
            out += list(zip("xyz", _bench(film, resolve(beat.to, film)), strict=True))
    return out


class TestTheTakesReadTheNumbersTheyRestated(unittest.TestCase):
    def assertNear(self, expected, got, what: str = "") -> None:
        expected, got = [float(v) for v in _flat(expected)], [float(v) for v in _flat(got)]
        self.assertEqual(len(expected), len(got), what)
        worst = max(abs(a - b) for a, b in zip(expected, got, strict=True))
        self.assertLess(worst, TOLERANCE, f"{what}: {got} where the take said {expected}")

    def assertDrags(self, expected, film) -> None:
        got = _drags(film)
        self.assertEqual([a for a, _ in expected], [a for a, _ in got], "the drags' axes")
        self.assertNear([v for _, v in expected], [v for _, v in got], film.take.key)

    def test_the_peel_closes_4_mm_over_the_tab_and_climbs_from_the_50_mm_box_top(self) -> None:
        film = _film("03_materials/08_tape.py", "03_materials/08_tape")
        top = 0.05  # the TAPE_TOP it restated
        stairs = [leg for k in range(1, 8) for leg in (("y", -0.04 - 0.02 * k), ("z", top + 0.03 + 0.01 * k))]
        self.assertDrags([("z", top + 0.0125 + 0.004), ("z", top + 0.03), *stairs], film)

    def test_the_stick_lays_the_label_over_the_same_box(self) -> None:
        film = _film("03_materials/09_adhesion.py")
        self.assertNear((0.0, -0.125, 0.146), _bench(film, takes._stick_far(film)), "STICK_FAR")
        self.assertNear((0.0, -0.040, 0.05 + 0.019), _bench(film, takes._stick_near(film)), "STICK_NEAR")

    def test_the_suction_square_and_plate(self) -> None:
        film = _film("04_extending/02_suction.py")
        self.assertNear((-0.11, -0.22, 0.11, 0.0), takes._square(film), "SQUARE")
        self.assertNear(0.003, takes._plate_half(film), "PLATE_HALF")

    def test_the_cut_handle_tofu_and_board(self) -> None:
        film = _film("03_materials/10_cutting.py", "03_materials/10_cutting")
        handle, edge, board = 0.04, 0.04, 0.012  # KNIFE_HANDLE, KNIFE_EDGE and BOARD_TOP
        self.assertDrags([("z", handle), ("z", handle + 0.07), ("y", -0.16), ("z", board + edge - 0.005)], film)

    def test_the_scrubbed_square_is_the_example_s_patch(self) -> None:
        from unittest import mock

        film = _film("03_materials/07_brush.py")
        brush = film.sim["brush"].entity
        with mock.patch.object(brush, "scrubbed", return_value=0.0) as scrubbed:
            takes._scrubbed(film)
        lo, hi = scrubbed.call_args[0]
        self.assertNear(((-0.025, -0.27), (0.075, -0.17)), (_bench(film, (*lo, 0))[:2], _bench(film, (*hi, 0))[:2]))

    def test_the_fold_s_sheet_and_box(self) -> None:
        film = _film("03_materials/05_paper.py")
        lay = takes._fold_layout(film)
        edge = -0.05  # the sheet's near short edge, 35 mm over the box's near edge
        self.assertNear(edge, _bench(film, (lay["edge"], 0, 0))[0], "the sheet's near edge")
        self.assertNear(edge + 0.297, _bench(film, (lay["far"], 0, 0))[0], "its far edge")
        self.assertNear(edge + 0.297 / 2, _bench(film, (lay["fold"], 0, 0))[0], "the fold line")
        self.assertNear(edge + 0.297 / 2, lay["line"], "the line crease_angle reads, in the bench frame")
        self.assertNear(edge + 0.013, _bench(film, (lay["pinch"], 0, 0))[0], "where the pads close")
        self.assertNear(
            (-0.08, 0.08), [lay["hands"]["right"], lay["hands"]["left"]], "each hand 25 mm in from its corner"
        )
        self.assertNear(edge + 0.035, _bench(film, (lay["box_edge"], 0, 0))[0], "the box's near face")
        self.assertNear(0.08, _bench(film, (0, 0, lay["top"]))[2], "the box's top")

    def test_the_rice_tray_and_pusher(self) -> None:
        film = _film("03_materials/11_granular.py", "03_materials/11_granular")
        self.assertNear(((-0.14, -0.22, -0.005), (0.14, -0.02, 0.03)), takes._tray(film), "the tray")
        self.assertDrags([("x", 0.0), ("y", -0.45), ("z", 0.12), ("y", -0.16)], film)

    def test_the_honey_bowl(self) -> None:
        film = _film("03_materials/12_liquid.py")
        self.assertNear(((0.05, -0.23, -0.005), (0.19, -0.09, 0.04)), takes._walled(film, "bowl", -0.005))

    @unittest.skipUnless(HAS_GENESIS, "the [genesis] extra is not installed; nothing builds the water")
    def test_the_water_glass(self) -> None:
        film = _film("03_materials/14_water.py")
        self.assertNear(((0.09, -0.16, 0.0), (0.15, -0.10, 0.055)), takes._walled(film, "glass", 0.0))

    @unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed; nothing builds the band")
    def test_the_pegs_and_the_path_round_them(self) -> None:
        film = _film("03_materials/04_band.py", "03_materials/04_band")
        self.assertDrags([("x", 0.030), ("z", 0.105), ("x", 0.082), ("z", 0.120), ("x", -0.025), ("x", -0.13)], film)
        for name, (z, length) in {"upper": (0.26, 0.08), "lower": (0.135, 0.05)}.items():
            root, out, long = takes._peg(film, name)
            self.assertNear(((0.7, -0.2, 0.75 + z), (-1.0, 0.0, 0.0), length), (root, out, long), name)

    @unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed; nothing builds the rope")
    def test_the_knot_as_tied(self) -> None:
        import numpy as np

        from robosimstudio_materials.rod.knot import knot_length, overhand

        film = _film("03_materials/03_knot.py")
        built = knot_length(overhand(at=(0.03, 0.0), size=0.10, radius=0.006, tail=0.12, spacing=0.008))
        self.assertNear(built, knot_length(np.asarray(film.sim["rope"].spec.nodes, dtype=np.float64)), "the knot")

    def test_the_post_the_cable_wraps(self) -> None:
        film = _film("03_materials/01_rope_memory.py")
        post, lays = takes._wrap_post(film)
        self.assertNear(((0.0, -0.086, 0.0), 0.031), (_bench(film, post), lays), "the post")


SCORED = {
    "06_tasks/03_granular_scored": "03_materials/11_granular",
    "06_tasks/04_brush_scored": "03_materials/07_brush",
    "06_tasks/05_suction_scored": "04_extending/02_suction",
    "06_tasks/07_band_scored": "03_materials/04_band",
    "06_tasks/08_tape_scored": "03_materials/08_tape",
    "06_tasks/09_cutting_scored": "03_materials/10_cutting",
    "06_tasks/10_water_scored": "03_materials/14_water",
}
"""Each scored script whose take aims at its layout, and the material script whose drive the take plays.

``06_tasks/06_bag_scored``'s drive aims at bench heights and at the bag's lip where it lies, nothing laid."""


class TestTheScoredTakesAimWhereTheirMaterialTakesDo(unittest.TestCase):
    """A scored script lays its material's scene out as a task document, and its take plays that scene's drive.

    So the drive, aimed on the scored script's scene, goes where it goes on the material's, and the
    layout its checks read (the tray, the square, the plate, the glass) is the same one: built both
    ways, each number to a micrometre.
    """

    def assertNear(self, expected, got, what: str = "") -> None:
        expected, got = [float(v) for v in _flat(expected)], [float(v) for v in _flat(got)]
        self.assertEqual(len(expected), len(got), what)
        worst = max((abs(a - b) for a, b in zip(expected, got, strict=True)), default=0.0)
        self.assertLess(worst, TOLERANCE, f"{what}: {got} on the scored scene, {expected} on the material's")

    def _both(self, scored: str):
        material = SCORED[scored]
        return _film(f"{scored}.py", scored), _film(f"{material}.py", material)

    def assertSameDrive(self, scored, material) -> None:
        here, there = _drags(scored, material.take.beats), _drags(material)
        self.assertTrue(there, "the material take aims at nothing laid")
        self.assertEqual([a for a, _ in there], [a for a, _ in here], "the drags' axes")
        self.assertNear([v for _, v in there], [v for _, v in here], scored.take.key)

    def test_the_rice_sweep(self) -> None:
        scored, material = self._both("06_tasks/03_granular_scored")
        self.assertSameDrive(scored, material)
        self.assertNear(takes._tray(material), takes._tray(scored), "the tray")

    def test_the_scrub(self) -> None:
        scored, material = self._both("06_tasks/04_brush_scored")
        self.assertSameDrive(scored, material)
        square = lambda film: film.sim.handles.regions["patch"].xy_ranges  # noqa: E731
        self.assertNear(square(material), square(scored), "the square")

    def test_the_suction_square_and_plate(self) -> None:
        scored, material = self._both("06_tasks/05_suction_scored")
        self.assertNear(takes._square(material), takes._square(scored), "the square")
        self.assertNear(takes._plate_half(material), takes._plate_half(scored), "the plate")
        self.assertNear(material.position("plate"), scored.position("plate"), "where it lies")

    @unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed; nothing builds the band")
    def test_the_band_path(self) -> None:
        self.assertSameDrive(*self._both("06_tasks/07_band_scored"))

    def test_the_peel(self) -> None:
        self.assertSameDrive(*self._both("06_tasks/08_tape_scored"))

    def test_the_cut(self) -> None:
        self.assertSameDrive(*self._both("06_tasks/09_cutting_scored"))

    @unittest.skipUnless(HAS_GENESIS, "the [genesis] extra is not installed; nothing builds the water")
    def test_the_glass(self) -> None:
        scored, material = self._both("06_tasks/10_water_scored")
        self.assertNear(takes._walled(material, "glass", 0.0), takes._walled(scored, "glass", 0.0), "the glass")
        self.assertNear(material.position("cup"), scored.position("cup"), "the cup")


def _flat(value):
    if isinstance(value, int | float):
        yield value
        return
    for item in value:
        yield from _flat(item)


if __name__ == "__main__":
    unittest.main()
