# SPDX-License-Identifier: Apache-2.0
"""The selector language: recipes compiled into material-frame selections, and refused when wrong.

A deformable feature is a selection stated against the sheet's rest layout, and
:func:`~robosimstudio.assetlib.semantics.selectors.compile_selector` is what
turns the data a sidecar carries into the callable a cloth entity evaluates
once at scene-ready. The claims here are the algebra's -- a list is ``all``,
``invert`` is the complement, ``any`` and ``all`` obey De Morgan, two halves
partition a sheet -- the selections that are derived from the rest extent and
so survive a translation, the boundary-tracing ``rim`` that a plain edge band
is not, and the refusals: a typo'd parameter, an unknown op or axis, an index
list composed as if it were a mask. Each refusal names what it refused.
"""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio.assetlib.semantics.selectors import (
    compile_selector,
    selector_to_dict,
    validate_selector,
)
from tests.fixtures.grids import ragged_sheet, sheet

TIER = "fast"
"""Numpy masks over synthetic sheets: nothing is built, so no GPU."""


def _select(recipe, rest: np.ndarray) -> np.ndarray:
    return np.asarray(compile_selector(recipe)(rest))


RIGHT = {"op": "beyond", "axis": "x", "threshold": 0.0}
FAR = {"op": "beyond", "axis": "y", "threshold": 0.05}


class TestTheAlgebra(unittest.TestCase):
    def setUp(self) -> None:
        self.rest = sheet(cells=21)

    def test_a_list_means_all_of_its_entries(self) -> None:
        listed = _select([RIGHT, FAR], self.rest)
        spelled = _select({"op": "all", "of": [RIGHT, FAR]}, self.rest)
        np.testing.assert_array_equal(listed, spelled)
        np.testing.assert_array_equal(listed, _select(RIGHT, self.rest) & _select(FAR, self.rest))
        self.assertTrue(listed.any(), "an empty intersection would make the comparison vacuous")

    def test_invert_is_the_complement_and_undoes_itself(self) -> None:
        right = _select(RIGHT, self.rest)
        np.testing.assert_array_equal(_select({"op": "invert", "of": RIGHT}, self.rest), ~right)
        twice = {"op": "invert", "of": {"op": "invert", "of": RIGHT}}
        np.testing.assert_array_equal(_select(twice, self.rest), right)

    def test_any_and_all_obey_de_morgan(self) -> None:
        union = _select({"op": "any", "of": [RIGHT, FAR]}, self.rest)
        complements = [{"op": "invert", "of": RIGHT}, {"op": "invert", "of": FAR}]
        np.testing.assert_array_equal(union, ~_select({"op": "all", "of": complements}, self.rest))
        np.testing.assert_array_equal(union, _select(RIGHT, self.rest) | _select(FAR, self.rest))

    def test_beyond_below_is_strictly_the_other_side(self) -> None:
        above = _select(RIGHT, self.rest)
        below = _select({**RIGHT, "above": False}, self.rest)
        on_the_plane = self.rest[:, 0] == 0.0
        self.assertTrue(on_the_plane.any(), "an odd grid has a column on the plane")
        np.testing.assert_array_equal(above | below, ~on_the_plane, "neither side takes the plane itself")
        self.assertFalse((above & below).any())

    def test_an_axis_by_name_and_by_index_select_alike(self) -> None:
        np.testing.assert_array_equal(_select(FAR, self.rest), _select({**FAR, "axis": 1}, self.rest))

    def test_indices_are_the_literal_list(self) -> None:
        np.testing.assert_array_equal(_select({"op": "indices", "of": [4, 1, 9]}, self.rest), [4, 1, 9])


class TestSelectionsFollowTheSheet(unittest.TestCase):
    def test_the_two_halves_partition_an_odd_grid_and_the_middle_row_goes_high(self) -> None:
        rest = sheet(cells=21)
        low = _select({"op": "half", "axis": "y", "side": -1}, rest)
        high = _select({"op": "half", "axis": "y", "side": 1}, rest)
        self.assertFalse((low & high).any(), "the halves overlap")
        self.assertTrue((low | high).all(), "a row fell between the halves")
        self.assertTrue(high[rest[:, 1] == 0.0].all(), "the midpoint row goes to side +1")
        self.assertEqual(10 * 21, int(low.sum()))

    def test_a_derived_selection_does_not_move_when_the_sheet_does(self) -> None:
        """``half``, ``extreme`` and ``rim`` read the rest extent, so a garment laid out elsewhere keeps its sets."""
        home, away = sheet(cells=15), sheet(cells=15, center=(0.7, -0.4))
        for recipe in (
            {"op": "half", "axis": "x", "side": 1},
            {"op": "extreme", "axis": "y", "band": 0.03, "high": False},
            {"op": "rim", "axis": "x", "along": "y", "band": 0.02, "bins": 5},
        ):
            with self.subTest(op=recipe["op"]):
                picked = _select(recipe, home)
                self.assertTrue(0 < picked.sum() < len(home), "all or nothing proves no invariance")
                np.testing.assert_array_equal(picked, _select(recipe, away))

    def test_rim_traces_a_ragged_edge_where_an_edge_band_takes_only_the_sleeve(self) -> None:
        rest = ragged_sheet(cells=21, reach=0.04)
        edge = _select({"op": "extreme", "axis": "x", "band": 0.012}, rest)
        rim = _select({"op": "rim", "axis": "x", "along": "y", "band": 0.012, "bins": 21}, rest)
        sleeve = rest[:, 0] > rest[:, 0].max() - 0.012
        np.testing.assert_array_equal(edge, sleeve, "the plain band selects the sleeve tip alone")
        rows = np.unique(np.round(rest[rim, 1], 9))
        self.assertEqual(21, len(rows), "the rim keeps a strip in every row along the fold line")
        self.assertTrue(rim[sleeve].all())
        # Each row's strip is that row's own outermost column, sleeve or not.
        for y in rows:
            row = np.isclose(rest[:, 1], y)
            self.assertEqual(rest[row, 0].max(), rest[rim & row, 0].max())


class TestABadRecipeIsRefusedByName(unittest.TestCase):
    def test_a_typoed_parameter_is_refused_rather_than_defaulted(self) -> None:
        with self.assertRaisesRegex(ValueError, r"bad parameters for op 'rim'.*bnis"):
            compile_selector({"op": "rim", "axis": "x", "along": "y", "bnis": 8})

    def test_an_unknown_op_names_the_ones_there_are(self) -> None:
        with self.assertRaisesRegex(ValueError, r"unknown selector op 'edge'.*'beyond'"):
            compile_selector({"op": "edge", "axis": "x"})
        with self.assertRaisesRegex(ValueError, "has no 'op'"):
            compile_selector({"axis": "x"})
        with self.assertRaisesRegex(ValueError, "expected a recipe mapping or a list, got str"):
            compile_selector("half")

    def test_an_axis_that_is_not_one_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown axis 'w'"):
            compile_selector({"op": "beyond", "axis": "w", "threshold": 0.0})
        with self.assertRaisesRegex(ValueError, "axis must be 0, 1 or 2, got 3"):
            compile_selector({"op": "beyond", "axis": 3, "threshold": 0.0})

    def test_a_band_upside_down_a_side_that_is_not_one_and_no_bins_are_refused(self) -> None:
        for recipe, said in (
            ({"op": "between", "axis": "x", "lo": 0.1, "hi": 0.0}, "is above hi"),
            ({"op": "half", "axis": "x", "side": 0}, "side must be -1 or \\+1"),
            ({"op": "rim", "axis": "x", "along": "y", "bins": 0}, "bins must be at least 1"),
            ({"op": "all", "of": []}, "'all' needs at least one"),
        ):
            with self.subTest(op=recipe["op"]), self.assertRaisesRegex(ValueError, said):
                compile_selector(recipe)

    def test_an_index_list_composed_as_a_mask_is_refused_where_it_is_read(self) -> None:
        composed = compile_selector({"op": "all", "of": [RIGHT, {"op": "indices", "of": [0, 1]}]})
        with self.assertRaisesRegex(ValueError, "an 'indices' recipe cannot be composed"):
            composed(sheet(cells=5))

    def test_the_path_in_the_message_names_the_bad_sub_recipe(self) -> None:
        with self.assertRaisesRegex(ValueError, r"^hem\.of\[1\]: unknown selector op"):
            compile_selector({"op": "any", "of": [RIGHT, {"op": "nope"}]}, where="hem")

    def test_validate_reports_what_compile_raises(self) -> None:
        self.assertEqual([], validate_selector([RIGHT, FAR]))
        (problem,) = validate_selector({"op": "nope"}, where="tab")
        self.assertIn("tab: unknown selector op 'nope'", problem)


class TestTheCanonicalForm(unittest.TestCase):
    def test_spelling_differences_vanish_and_the_selection_does_not_change(self) -> None:
        rest = sheet(cells=21)
        recipe = [RIGHT, {"op": "between", "axis": "y", "lo": -0.3 / 6, "hi": 0.3 / 6 * 2}]
        canonical = selector_to_dict(recipe)
        self.assertEqual("all", canonical["op"])
        self.assertEqual([0, 1], [part["axis"] for part in canonical["of"]])
        self.assertEqual(-0.05, canonical["of"][1]["lo"], "rounded, so a golden diff is about content")
        np.testing.assert_array_equal(_select(recipe, rest), _select(canonical, rest))
        self.assertEqual(canonical, selector_to_dict(canonical), "canonical form is a fixed point")

    def test_two_spellings_of_one_selection_compare_equal(self) -> None:
        by_name = {"op": "invert", "of": {"op": "half", "axis": "z", "side": 1}}
        by_index = {"op": "invert", "of": {"op": "half", "axis": 2, "side": 1}}
        self.assertEqual(selector_to_dict(by_name), selector_to_dict(by_index))


if __name__ == "__main__":
    unittest.main()
