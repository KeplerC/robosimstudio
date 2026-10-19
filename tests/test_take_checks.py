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


class TestTheTopOfABoxIsItsTop(unittest.TestCase):
    """``Top`` on an ``rss.box`` reads the box's size off ``spec.shape``: a plate's top, not its middle.

    A box's spec carries its halves only in ``shape.half_extents``, so the film once took its top
    for its origin, and the suction take's ``Top("plate", dz=-0.003)`` aimed 6 mm under the top.
    """

    @staticmethod
    def _film(quat) -> object:
        from recording.lockstep import Film

        plate = SimpleNamespace(
            spec=_box((0.0, -0.25, 0.003), (0.075, 0.075, 0.003)),  # 04_extending/02_suction's plate
            particles=np.zeros(0, dtype=np.int64),
            bodies=np.array([4]),
            position=np.array([0.6, -0.25, 0.753]),
            pose=np.array([0.6, -0.25, 0.753, *quat]),
        )
        film = Film(None, port=0, scratch=None, software=False, probe=False, log=print)
        film.sim = _Sim(refs={"plate": plate})
        return film

    def test_a_flat_plate_s_top_is_3_mm_over_its_middle(self) -> None:
        from recording.beats import Top

        film = self._film((0.0, 0.0, 0.0, 1.0))
        self.assertAlmostEqual(0.756, film.top("plate"), 12)
        self.assertAlmostEqual(0.006, film.height("plate"), 12)
        self.assertAlmostEqual(0.753, Top("plate", dz=-0.003).resolve(film)[2], 12)

    def test_a_plate_stood_on_its_edge_tops_out_at_its_half_width(self) -> None:
        film = self._film((np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)))  # turned 90 degrees about x
        self.assertAlmostEqual(0.753 + 0.075, film.top("plate"), 12)


# =============================================================================
# The layout a take aims at is the one its example builds
# =============================================================================

ORIGIN = (0.6, 0.0, 0.75)
"""Where the stand-in bench's frame is in the world [m]: ``bench_075``'s."""


def _box(at, half) -> SimpleNamespace:
    """An unturned body's spec, as ``rss.box`` gives it."""
    return SimpleNamespace(pose=at, quat=(0.0, 0.0, 0.0, 1.0), shape=SimpleNamespace(kind="box", half_extents=half))


def _slab(at, half) -> SimpleNamespace:
    """An unturned static's spec, as ``rss.slab`` gives it."""
    return SimpleNamespace(xform=(*at, 0.0, 0.0, 0.0, 1.0), shape=SimpleNamespace(kind="box", half_extents=half))


def _laid_film(things: dict, regions: dict | None = None) -> SimpleNamespace:
    """A stand-in film holding only the specs a take's layout reads, its bench at :data:`ORIGIN`."""
    sim = _Sim(
        refs={name: thing if hasattr(thing, "spec") else SimpleNamespace(spec=thing) for name, thing in things.items()}
    )
    sim.handles = SimpleNamespace(regions={k: SimpleNamespace(xy_ranges=v) for k, v in (regions or {}).items()})
    return SimpleNamespace(sim=sim, bench=lambda p: tuple(float(a) + b for a, b in zip(p, ORIGIN, strict=True)))


def _drags(key: str, film) -> list[tuple[str, float]]:
    """Each drag of a take to a point of its layout: its axis, and the bench-frame coordinate it drags to along it.

    A drag to where a thing is now (:class:`~recording.beats.At`, :class:`~recording.beats.Top`) reads
    the state, not the layout, and is left out.
    """
    from recording.beats import At, Drag, DragTo, Top, resolve

    out = []
    for beat in takes.find(key).beats:
        if isinstance(beat, Drag) and beat.to is not None and not isinstance(beat.to, At | Top):
            i = "xyz".index(beat.axis)
            out.append((beat.axis, round(resolve(beat.to, film)[i] - ORIGIN[i], 9)))
        elif isinstance(beat, DragTo):
            out += [(a, round(v - o, 9)) for a, v, o in zip("xyz", resolve(beat.to, film), ORIGIN, strict=True)]
    return out


class TestTheTakesAimWhereTheirExampleLaysThings(unittest.TestCase):
    """Each layout is moved off the example's own (to numbers no example uses) and the take moves with it.

    Before, a take restated its example's box top, tab, knife, board, square, bowl and sheet as
    constants, and an example that moved one moved the take off it without a word. The equality
    with each example's own numbers, on the built scene, is ``tests/test_take_layout``.
    """

    def test_the_peel_closes_over_the_tab_and_climbs_from_the_box_top(self) -> None:
        tab = np.array([*(np.array([0.03, -0.06, 0.0825]) + ORIGIN), 0.0, 0.0, 1.0, 0.0])
        film = _laid_film(
            {
                "box": _box((0.03, -0.10, 0.035), (0.05, 0.06, 0.035)),  # its top 70 mm up
                "label": SimpleNamespace(spec=None, entity=SimpleNamespace(tab_q0=tab)),
            }
        )
        stairs = [leg for k in range(1, 8) for leg in (("y", -0.06 - 0.02 * k), ("z", 0.10 + 0.01 * k))]
        expected = [("z", 0.0865), ("z", 0.10), *stairs]
        self.assertEqual([(a, round(v, 9)) for a, v in expected], _drags("03_materials/08_tape", film))

    def test_the_cut_takes_the_knife_where_it_stands_and_cuts_on_the_board(self) -> None:
        edge = ((0.045, 0.0, -0.045), (1.0, 0.0, 0.0), 0.12)  # a 55 mm blade's edge, 45 mm under the handle
        knife = SimpleNamespace(**vars(_box((-0.01, -0.25, 0.05), (0.045, 0.01, 0.01))), edge=edge)
        film = _laid_film(
            {
                "knife": knife,
                "board": _slab((0.06, -0.12, 0.008), (0.07, 0.08, 0.008)),
                "tofu": SimpleNamespace(at=(0.06, -0.12, 0.016)),
            }
        )
        expected = [("z", 0.05), ("z", 0.12), ("y", -0.12), ("z", 0.056)]
        self.assertEqual([(a, round(v, 9)) for a, v in expected], _drags("03_materials/10_cutting", film))

    def test_the_rice_sweep_goes_over_the_pusher_and_into_the_tray(self) -> None:
        film = _laid_film(
            {
                "pusher": _box((0.02, -0.40, 0.02), (0.08, 0.01, 0.02)),
                "wall_back": _slab((0.02, 0.03, 0.02), (0.16, 0.01, 0.02)),
            }
        )
        drags = _drags("03_materials/11_granular", film)
        self.assertEqual([("x", 0.02), ("y", -0.40), ("z", 0.12)], drags[:3])  # over the pusher, 100 mm up
        self.assertIn(("y", round(0.02 - 0.14, 9)), drags)  # 140 mm short of the back wall's face

    def test_the_pours_count_what_is_inside_the_bowl_and_the_glass_where_they_stand(self) -> None:
        walls = {}
        for prefix, (x, y), side, thick, tall in (
            ("bowl", (0.20, -0.05), 0.14, 0.004, 0.02),
            ("glass", (0.20, -0.05), 0.06, 0.003, 0.0275),
        ):
            h = side / 2 + thick
            for name, (dx, dy, hx, hy) in {
                "near": (-h, 0.0, thick, h),
                "far": (h, 0.0, thick, h),
                "right": (0.0, -h, h, thick),
                "left": (0.0, h, h, thick),
            }.items():
                walls[f"{prefix}_{name}"] = _slab((x + dx, y + dy, tall), (hx, hy, tall))
        inside = np.array([[0.20, -0.05, 0.005], [0.21, -0.04, 0.01]]) + ORIGIN
        liquid = SimpleNamespace(spec=None, particle_q=inside)
        honey = _laid_film({**walls, "honey": liquid})
        ok, said = takes._poured(honey)
        self.assertTrue(ok, said)
        water = _laid_film({**walls, "water": liquid})
        with self.subTest("glass"):
            from unittest import mock

            with mock.patch(
                "robosimstudio_materials.water.fill_level", return_value={"fraction": 1.0, "height": 0.03}
            ) as level:
                takes._water_in_glass(water)
            lo, hi = level.call_args[0][1:3]
            np.testing.assert_allclose(lo, np.array([0.17, -0.08, 0.0]) + ORIGIN, atol=1e-12)
            np.testing.assert_allclose(hi, np.array([0.23, -0.02, 0.055]) + ORIGIN, atol=1e-12)

    def test_the_tray_and_the_drawn_squares_are_the_scene_s(self) -> None:
        film = _laid_film(
            {
                "wall_left": _slab((-0.12, -0.05, 0.02), (0.01, 0.11, 0.02)),
                "wall_right": _slab((0.18, -0.05, 0.02), (0.01, 0.11, 0.02)),
                "wall_back": _slab((0.03, 0.05, 0.02), (0.16, 0.01, 0.02)),
                "plate": _box((0.0, -0.25, 0.004), (0.075, 0.075, 0.004)),
            },
            regions={"square": (-0.05, -0.15, 0.15, 0.05)},
        )
        np.testing.assert_allclose(takes._tray(film), ((-0.11, -0.16, -0.005), (0.17, 0.04, 0.04)), atol=1e-12)
        self.assertEqual((-0.05, -0.15, 0.15, 0.05), takes._square(film))
        self.assertEqual(0.004, takes._plate_half(film))

    def test_the_stick_and_the_fold_work_on_the_box_and_the_sheet_as_laid(self) -> None:
        film = _laid_film(
            {
                "box": _box((0.02, -0.10, 0.03), (0.05, 0.06, 0.03)),
                "paper": SimpleNamespace(pose=(-0.05, -0.43, 0.005), size=(0.20, 0.30)),
            }
        )
        np.testing.assert_allclose(takes._stick_far(film), np.array([0.02, -0.145, 0.156]) + ORIGIN, atol=1e-12)
        np.testing.assert_allclose(takes._stick_near(film), np.array([0.02, -0.06, 0.079]) + ORIGIN, atol=1e-12)
        sheet = takes._sheet(film)
        self.assertAlmostEqual(-0.33, sheet["crease"], 12)
        np.testing.assert_allclose(sheet["pinch"], (-0.015, -0.4165), atol=1e-12)
        np.testing.assert_allclose(sheet["hold"], (0.05, -0.165), atol=1e-12)
        np.testing.assert_allclose(sheet["stamps"], (0.005, 0.06, 0.115), atol=1e-12)

    def test_the_pegs_and_the_post_are_where_the_example_stands_them(self) -> None:
        along_x = (0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5))
        post = SimpleNamespace(
            xform=(0.02, -0.15, 0.20, *along_x), shape=SimpleNamespace(radius=0.006, half_height=0.03)
        )
        cable_post = SimpleNamespace(xform=(0.05, -0.10, 0.05, 0.0, 0.0, 0.0, 1.0), shape=SimpleNamespace(radius=0.025))
        film = _laid_film({"upper": post, "post": cable_post, "cable": SimpleNamespace(radius=0.01)})
        root, out, length = takes._peg(film, "upper")
        np.testing.assert_allclose(root, np.array([0.05, -0.15, 0.20]) + ORIGIN, atol=1e-12)
        np.testing.assert_allclose(out, (-1.0, 0.0, 0.0), atol=1e-12)
        self.assertAlmostEqual(0.06, length, 12)
        at, lays = takes._wrap_post(film)
        np.testing.assert_allclose(at, np.array([0.05, -0.10, 0.0]) + ORIGIN, atol=1e-12)
        self.assertAlmostEqual(0.035, lays, 12)

    def test_the_band_path_follows_its_pegs(self) -> None:
        along_x = (0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5))
        pegs = _laid_film(
            {
                "band": SimpleNamespace(nodes=((0.05, -0.24, 0.20), (0.05, -0.16, 0.20))),
                "lower": SimpleNamespace(xform=(0.09, -0.2, 0.15, *along_x)),
                "board": _slab((0.12, -0.2, 0.15), (0.005, 0.012, 0.15)),
            }
        )
        path = [("x", 0.045), ("z", 0.12), ("x", 0.097), ("z", 0.135), ("x", -0.01), ("x", -0.13)]
        self.assertEqual(path, _drags("03_materials/04_band", pegs))


if __name__ == "__main__":
    unittest.main()
