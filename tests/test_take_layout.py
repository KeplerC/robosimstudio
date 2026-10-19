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
import math
import sys
import unittest
from pathlib import Path

from tests.fixtures.examples import _run

TIER = "medium"
"""Builds fourteen example scenes on one GPU and steps none of them."""

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from recording import takes  # noqa: E402
from recording.beats import At, Drag, DragTo, Top, laid, resolve  # noqa: E402

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


def _drags(film) -> list[tuple[str, float]]:
    """Each drag of the film's take to a point of its layout: its axis, and the bench coordinate along it."""
    out = []
    for beat in film.take.beats:
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
        film = _film("19_tape/00_peel_a_label.py", "19_tape/00_peel_a_label")
        top = 0.05  # the TAPE_TOP it restated
        stairs = [leg for k in range(1, 8) for leg in (("y", -0.04 - 0.02 * k), ("z", top + 0.03 + 0.01 * k))]
        self.assertDrags([("z", top + 0.0125 + 0.004), ("z", top + 0.03), *stairs], film)

    def test_the_stick_lays_the_label_over_the_same_box(self) -> None:
        film = _film("19_tape/02_stick_a_label.py")
        self.assertNear((0.0, -0.125, 0.146), _bench(film, takes._stick_far(film)), "STICK_FAR")
        self.assertNear((0.0, -0.040, 0.05 + 0.019), _bench(film, takes._stick_near(film)), "STICK_NEAR")

    def test_the_suction_square_and_plate(self) -> None:
        film = _film("15_suction/00_pick_a_plate.py")
        self.assertNear((-0.11, -0.22, 0.11, 0.0), takes._square(film), "SQUARE")
        self.assertNear(0.003, takes._plate_half(film), "PLATE_HALF")

    def test_the_cut_handle_tofu_and_board(self) -> None:
        film = _film("20_cutting/00_cut_tofu.py", "20_cutting/00_cut_tofu")
        handle, edge, board = 0.04, 0.04, 0.012  # KNIFE_HANDLE, KNIFE_EDGE and BOARD_TOP
        self.assertDrags([("z", handle), ("z", handle + 0.07), ("y", -0.16), ("z", board + edge - 0.005)], film)

    def test_the_scrubbed_square_is_the_example_s_patch(self) -> None:
        from unittest import mock

        film = _film("14_brush/00_scrub_the_bench.py")
        brush = film.sim["brush"].entity
        with mock.patch.object(brush, "scrubbed", return_value=0.0) as scrubbed:
            takes._scrubbed(film)
        lo, hi = scrubbed.call_args[0]
        self.assertNear(((-0.025, -0.27), (0.075, -0.17)), (_bench(film, (*lo, 0))[:2], _bench(film, (*hi, 0))[:2]))

    def test_the_fold_s_sheet(self) -> None:
        film = _film("13_paper/00_fold_paper.py")
        edge, sheet = -0.435, takes._sheet(film)  # FOLD_EDGE
        self.assertNear(edge, sheet["edge"], "FOLD_EDGE")
        self.assertNear(edge + 0.297 / 3, sheet["crease"], "FOLD_CREASE")
        self.assertNear((-0.02, edge + 0.0135), sheet["pinch"], "FOLD_PINCH")
        self.assertNear(-0.055 - 0.06, sheet["short"], "the corner the jaws come in from")
        self.assertNear((0.05, edge + 0.262), sheet["hold"], "the left hand's hold")
        self.assertNear((0.0, 0.055, 0.11), sheet["stamps"], "the stamps")
        self.assertNear(-0.40, _bench(film, (0.0, takes._bench_edge(film), 0.0))[1], "the bench's edge")

    def test_the_box_flap_s_hinge_rim_and_push(self) -> None:
        film = _film("13_paper/01_box_flap.py", "13_paper/01_box_flap")
        self.assertNear(0.103 + 0.02 * math.cos(math.radians(15.0)), takes._flap_hinge(film), "the hinge rows")
        self.assertNear(0.10, laid(film, "wall_far", (0, 0, 1))[2], "the rim")
        self.assertDrags([("z", 0.162), ("y", -0.01), ("z", 0.062), ("z", 0.162)], film)

    def test_the_rice_tray_and_pusher(self) -> None:
        film = _film("10_granular/00_sweep_rice.py", "10_granular/00_sweep_rice")
        self.assertNear(((-0.14, -0.22, -0.005), (0.14, -0.02, 0.03)), takes._tray(film), "the tray")
        self.assertDrags([("x", 0.0), ("y", -0.45), ("z", 0.12), ("y", -0.16)], film)

    def test_the_honey_bowl(self) -> None:
        film = _film("11_liquid/00_pour_honey.py")
        self.assertNear(((0.05, -0.23, -0.005), (0.19, -0.09, 0.04)), takes._walled(film, "bowl", -0.005))

    @unittest.skipUnless(HAS_GENESIS, "the [genesis] extra is not installed; nothing builds the water")
    def test_the_water_glass(self) -> None:
        film = _film("21_water/00_pour_water.py")
        self.assertNear(((0.09, -0.16, 0.0), (0.15, -0.10, 0.055)), takes._walled(film, "glass", 0.0))

    @unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed; nothing builds the band")
    def test_the_pegs_and_the_path_round_them(self) -> None:
        film = _film("18_bands/00_band_on_pegs.py", "18_bands/00_band_on_pegs")
        self.assertDrags([("x", 0.030), ("z", 0.105), ("x", 0.082), ("z", 0.120), ("x", -0.025), ("x", -0.13)], film)
        for name, (z, length) in {"upper": (0.26, 0.08), "lower": (0.135, 0.05)}.items():
            root, out, long = takes._peg(film, name)
            self.assertNear(((0.7, -0.2, 0.75 + z), (-1.0, 0.0, 0.0), length), (root, out, long), name)

    @unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed; nothing builds the rope")
    def test_the_knot_as_tied(self) -> None:
        import numpy as np

        from robosimstudio_materials.rod.knot import knot_length, overhand

        film = _film("17_knots/00_pull_knot_tight.py")
        built = knot_length(overhand(at=(0.03, 0.0), size=0.10, radius=0.006, tail=0.12, spacing=0.008))
        self.assertNear(built, knot_length(np.asarray(film.sim["rope"].spec.nodes, dtype=np.float64)), "the knot")

    def test_the_post_the_cable_wraps(self) -> None:
        film = _film("03_rope/03_cable_memory.py")
        post, lays = takes._wrap_post(film)
        self.assertNear(((0.0, -0.086, 0.0), 0.031), (_bench(film, post), lays), "the post")

    def test_the_roller_and_the_block(self) -> None:
        film = _film("12_dough/00_roll_dough.py", "12_dough/00_roll_dough")
        z = [0.820, 0.850, None, 0.838, None, None, None, 0.890]  # the world heights it said, over bench_075
        x = [None, None, 0.515, None, 0.675, 0.515, 0.675, None]
        world = [("z", v - 0.75) if v else ("x", u - 0.6) for v, u in zip(z, x, strict=True)]
        self.assertDrags(world, film)


def _flat(value):
    if isinstance(value, int | float):
        yield value
        return
    for item in value:
        yield from _flat(item)


if __name__ == "__main__":
    unittest.main()
