# SPDX-License-Identifier: Apache-2.0
"""``goals.catalog``: the live registries behind the checker's protocol, under both spellings.

BDDL writes ``bond-fraction`` and Python registers ``bond_fraction``; the
catalog is the one adapter between them, so every lookup is asked both ways. A
published state is a word an asset defines as much as a semantic state is --
the tape task's whole goal reads one -- and the grammar an agent is handed is
read off the parser's own tuples, so it cannot claim a form the parser refuses.
"""

from __future__ import annotations

import unittest

from robosimstudio.assetlib.record import AssetRecord
from robosimstudio.assetlib.registry import ASSETS
from robosimstudio.assetlib.semantics.atoms import get_atom
from robosimstudio.assetlib.semantics.features import FeatureSpec
from robosimstudio.assetlib.semantics.published import PublishedState
from robosimstudio.assetlib.semantics.states import StateSpec
from robosimstudio_bench.goals.catalog import AssetCatalog, catalog_json
from robosimstudio_bench.taskdoc.parse import COMPARISONS, CONNECTIVES, MODALS
from tests.fixtures.registries import registered

TIER = "fast"
"""Lookups into registries a test fills: no scene, no GPU."""

TAPE = AssetRecord(
    name="probe_tape",
    features=(FeatureSpec(name="free_end", kind="loop"),),
    states=(StateSpec(name="adhered", measure="bond_fraction", op="<=", value=0.05),),
    published=(PublishedState(name="bond_fraction", type="float", summary="the share of bonds broken"),),
)


class TestTheCatalogAnswersBothSpellings(unittest.TestCase):
    def setUp(self):
        context = registered(ASSETS, TAPE.name, TAPE)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        self.catalog = AssetCatalog()

    def test_an_asset_is_there_or_it_is_not(self):
        self.assertTrue(self.catalog.has_asset("probe_tape"))
        self.assertFalse(self.catalog.has_asset("probe_tape_missing"))

    def test_a_feature_is_found_under_either_spelling(self):
        self.assertEqual(self.catalog.features_of("probe_tape"), {"free_end": "loop", "free-end": "loop"})

    def test_published_states_are_words_beside_the_semantic_ones(self):
        self.assertEqual(set(self.catalog.states_of("probe_tape")), {"adhered", "bond_fraction", "bond-fraction"})

    def test_an_unknown_asset_has_no_features_and_no_states(self):
        self.assertEqual((self.catalog.features_of("nope"), self.catalog.states_of("nope")), ({}, ()))

    def test_an_atom_is_found_under_either_spelling_and_an_unknown_one_is_none(self):
        self.assertIs(self.catalog.atom("contact-force"), get_atom("contact_force"))
        self.assertIs(self.catalog.atom("contact_force"), get_atom("contact_force"))
        self.assertIsNone(self.catalog.atom("no-such-atom"))


class TestTheAgentsCatalog(unittest.TestCase):
    def test_the_grammar_is_the_parsers_own(self):
        grammar = catalog_json()["grammar"]
        self.assertEqual(grammar["connectives"], list(CONNECTIVES))
        self.assertEqual(grammar["comparisons"], list(COMPARISONS))
        self.assertEqual(grammar["modals"], list(MODALS))
        self.assertLessEqual(set(grammar["quantity_modals"]), set(MODALS))

    def test_every_registered_atom_and_asset_is_listed(self):
        from robosimstudio.assetlib.semantics.atoms import list_atoms

        with registered(ASSETS, TAPE.name, TAPE):
            listed = catalog_json()
        self.assertEqual(sorted(listed["atoms"]), sorted(list_atoms()))
        self.assertIn("probe_tape", listed["assets"])


if __name__ == "__main__":
    unittest.main()
