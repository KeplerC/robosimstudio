# SPDX-License-Identifier: Apache-2.0
"""``goals.query``: a query in the document's own spelling, dispatched under the task
scope bind uses, and refused with ``QueryRefused`` rather than a bare lookup error."""

from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

from robosimstudio.assetlib.semantics.atoms import ATOMS, ArgSpec, AtomSpec, register_atom
from robosimstudio.assetlib.semantics.handlers import HANDLERS, handler
from robosimstudio_bench.goals import query as query_module
from robosimstudio_bench.goals.bind import Operand
from robosimstudio_bench.taskdoc.ast import Ref
from tests.fixtures.registries import restored

TIER = "fast"
"""Probe atoms and handlers in the real registries, and stubbed reads: no scene, no GPU."""


PROBE = "probe_insertion"
"""An atom registered for one test at a time, through the real atom registry and dispatch."""


@contextlib.contextmanager
def _probe(*handlers: tuple[dict, object], returns=("depth", "lat"), asset: str = ""):
    """``PROBE`` registered with ``handlers`` (``(handler kwargs, fn)``), and operands that carry ``asset``.

    Only the operand resolution and the read are stubbed -- they are ``bind``'s,
    and tested there. The atom lookup and the dispatch are the real ones, so a
    query that drifts from how ``bind`` calls them fails here.
    """
    with (
        restored(ATOMS, HANDLERS),
        mock.patch.object(query_module, "_operand", side_effect=lambda r, d, c: Operand(ref=Ref(r), asset=asset)),
        mock.patch.object(query_module, "read", return_value=None),
    ):
        register_atom(AtomSpec(name=PROBE, summary="a probe", args=(ArgSpec(name="plug"),), returns=returns))
        for kwargs, fn in handlers:
            handler(PROBE, **kwargs)(fn)
        yield


class TestAQueryReadsTheDocumentsSpelling(unittest.TestCase):
    def _query(self, atom, reported, *references, asset=""):
        answered: list[str] = []

        def kind(*values):
            answered.append("kind")
            return reported

        def scoped(*values):
            answered.append("asset")
            return reported

        with _probe(({}, kind), ({"scope": "asset", "owner": "widget"}, scoped), asset=asset):
            out = query_module.query(SimpleNamespace(entities={}), atom, *references)
        return out, answered

    def test_a_hyphen_is_the_registered_underscore(self):
        out, answered = self._query("probe-insertion", {"depth": 1.0}, "plug")
        self.assertEqual(out, {"depth": 1.0})
        self.assertEqual(answered, ["kind"])

    def test_a_selector_narrows_the_answer(self):
        out, _ = self._query("probe-insertion:lat", {"depth": 1.0, "lat": 0.2}, "plug")
        self.assertEqual(out, {"lat": 0.2})

    def test_a_selector_the_atom_does_not_report_is_refused(self):
        with self.assertRaisesRegex(query_module.QueryRefused, "'mate' is not among them"):
            self._query("probe_insertion:mate", {"depth": 1.0}, "plug")

    def test_a_scalar_return_is_keyed_by_the_primary(self):
        out, _ = self._query("probe_insertion", 0.5, "plug")
        self.assertEqual(out, {"depth": 0.5})

    def test_the_asset_scope_is_passed_as_bind_passes_it(self):
        _, answered = self._query("probe_insertion", {"depth": 1.0}, "plug", asset="widget")
        self.assertEqual(answered, ["asset"])


class TestAQueryTakesTheTaskScope(unittest.TestCase):
    """bind dispatches under ``task or doc.name``; a query must, or a task-scoped
    override scores the goal with one handler and answers the query with another."""

    def _query(self, **kwargs):
        general = ({}, lambda *v: {"depth": 0.0})
        override = ({"scope": "task", "owner": "peg_task"}, lambda *v: {"depth": 1.0})
        with _probe(general, override, returns=("depth",)):
            return query_module.query(SimpleNamespace(entities={}), PROBE, "plug", **kwargs)

    def test_the_documents_name_is_the_task_scope(self):
        self.assertEqual(self._query(doc=SimpleNamespace(name="peg_task")), {"depth": 1.0})

    def test_an_explicit_task_wins(self):
        self.assertEqual(self._query(doc=SimpleNamespace(name="other"), task="peg_task"), {"depth": 1.0})

    def test_no_task_is_the_kind_scope(self):
        self.assertEqual(self._query(), {"depth": 0.0})


class TestQueryRefusals(unittest.TestCase):
    """Dispatch refuses with a plain ``LookupError``; the query API promises ``QueryRefused``."""

    def test_applicable_skips_an_unimplemented_atom(self):
        leaf = SimpleNamespace(name="threaded", args=(1, 2))
        with (
            mock.patch.object(query_module, "leaves_of", return_value=[leaf]),
            mock.patch.object(query_module, "dispatch", side_effect=LookupError("no handler")),
        ):
            operand = SimpleNamespace(kind="", substrate="rigid")
            self.assertEqual(query_module.applicable("family", operand, operand), [])

    def test_query_converts_it(self):
        # An atom no handler implements: the real dispatch refuses it with a plain LookupError.
        with _probe(), self.assertRaisesRegex(query_module.QueryRefused, "no handler implements"):
            query_module.query(SimpleNamespace(entities={}), PROBE, "plug", doc=None)

    def test_an_unknown_atom_is_refused(self):
        with self.assertRaises(query_module.QueryRefused):
            query_module.query(SimpleNamespace(entities={}), "no_such_atom_anywhere", doc=None)


if __name__ == "__main__":
    unittest.main()
