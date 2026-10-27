# SPDX-License-Identifier: Apache-2.0
"""The mint gate: the purity lint on a measure's source, the dedup sweep across the
catalog, the anti-overfit lint, and one listing of the catalog per gate."""

import unittest
from unittest import mock

from robosimstudio.assetlib.semantics.atoms import ATOMS, ArgSpec, AtomSpec, get_atom, register_atom
from robosimstudio_bench.goals import mint
from tests.fixtures.registries import restored

TIER = "fast"
"""Atom declarations and their source text: no scene, no GPU."""


class TestTheMintGateListsTheCatalogOnce(unittest.TestCase):
    def test_once(self):
        """One report per registered atom, in catalog order, from one listing of the catalog."""
        with mock.patch.object(mint, "list_atoms", wraps=mint.list_atoms) as list_atoms:
            reports = mint.mint_gate(exempt={})
        self.assertEqual(list_atoms.call_count, 1)
        self.assertEqual([report.atom for report in reports], mint.list_atoms())
        self.assertGreater(len(reports), 10)


_COUNT = 0


def _pure(snapshot, obj):
    return float(obj.x) * 2.0


def _remembers(snapshot, obj):
    global _COUNT  # noqa: PLW0603 -- the shape under test
    _COUNT += 1
    return _COUNT


def _writes_its_operand(snapshot, obj):
    obj.x = 0.0
    obj = obj.y  # rebinding the name writes nothing into what was handed in
    return obj


def _reads_the_clock(snapshot, obj):
    import time

    return time.monotonic()


def _samples(snapshot, obj):
    import random

    return random.random()


def _yields(snapshot, obj):
    yield obj.x


def _atom(name: str, fn=None, **kwargs) -> AtomSpec:
    return AtomSpec(name=name, summary="a probe", fn=fn, args=(ArgSpec(name="obj"),), returns=("v",), **kwargs)


def _kinds(fn) -> list[str]:
    return [f.kind for f in mint.purity_problems(_atom("probe", fn))]


class TestThePurityLint(unittest.TestCase):
    """``purity_problems``: the shapes a generated measure takes when it is not one."""

    def test_a_reading_of_its_operands_is_pure(self):
        self.assertEqual(_kinds(_pure), [])

    def test_an_atom_with_no_implementation_is_not_impure(self):
        self.assertEqual(_kinds(None), [])

    def test_a_measure_that_remembers_is_impure(self):
        self.assertEqual(_kinds(_remembers), ["impure"])

    def test_writing_into_an_operand_is_impure_and_rebinding_its_name_is_not(self):
        (finding,) = mint.purity_problems(_atom("probe", _writes_its_operand))
        self.assertIn("writes into its operand 'obj'", finding.detail)

    def test_a_clock_or_an_entropy_source_is_impure(self):
        for fn, call in ((_reads_the_clock, "'monotonic'"), (_samples, "'random'")):
            with self.subTest(fn=fn.__name__):
                (finding,) = mint.purity_problems(_atom("probe", fn))
                self.assertIn(call, finding.detail)

    def test_a_generator_is_impure(self):
        self.assertEqual(_kinds(_yields), ["impure"])

    def test_a_source_it_cannot_read_is_a_note_not_a_verdict(self):
        unreadable = eval("lambda snapshot, obj: 1.0")
        (finding,) = mint.purity_problems(_atom("probe", unreadable))
        self.assertEqual((finding.kind, finding.severity), ("purity-unknown", "note"))


class TestTheDedupSweep(unittest.TestCase):
    def test_speed_and_settle_speed_are_not_one_question(self):
        self.assertEqual(mint.near_synonyms(get_atom("speed")), [])
        self.assertNotIn("speed", mint.near_synonyms(get_atom("settle_speed")))

    def test_an_unremarked_twin_is_an_error_and_a_remarked_one_a_note(self):
        with restored(ATOMS):
            register_atom(_atom("probe_twin_a", _pure))
            register_atom(_atom("probe_twin_b", _pure))
            self.assertEqual(mint.near_synonyms(get_atom("probe_twin_a")), ["probe_twin_b"])
            plain = mint.mint_problems(_atom("probe_twin_a", _pure), require_calibration=False)
            said = mint.mint_problems(_atom("probe_twin_a", _pure, provenance="stricter"), require_calibration=False)
        self.assertEqual([(f.kind, f.severity) for f in plain], [("duplicate", "error")])
        self.assertEqual([(f.kind, f.severity) for f in said], [("duplicate", "note")])

    def test_a_name_that_hard_codes_an_instance_is_overfit(self):
        findings = mint.mint_problems(
            _atom("hammer_in_bin", _pure), instances=("hammer_0", "hammer", "tote"), require_calibration=False
        )
        (overfit,) = (f for f in findings if f.kind == "overfit")
        self.assertIn("hammer", overfit.detail)
        self.assertNotIn("tote", overfit.detail)


if __name__ == "__main__":
    unittest.main()
