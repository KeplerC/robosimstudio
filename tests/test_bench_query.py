# SPDX-License-Identifier: Apache-2.0
"""``goals.query``: a query in the document's own spelling, dispatched under the task
scope bind uses, and refused with ``QueryRefused`` rather than a bare lookup error."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from robosimstudio_bench.goals import query as query_module
from robosimstudio_bench.goals.bind import Operand
from robosimstudio_bench.taskdoc.ast import Ref

TIER = "fast"
"""Stub operands and handlers: no scene, no GPU."""


class TestAQueryReadsTheDocumentsSpelling(unittest.TestCase):
    def _query(self, atom, reported, *references, asset=""):
        spec = SimpleNamespace(args=(), reads_scene=False, returns=("depth", "lat"), primary="depth")
        handler = SimpleNamespace(fn=lambda *values: reported)
        with (
            mock.patch.object(query_module, "leaves_of", return_value=[]),
            mock.patch.object(query_module, "get_atom", return_value=spec) as get_atom,
            mock.patch.object(query_module, "dispatch", return_value=handler) as dispatch,
            mock.patch.object(query_module, "_layouts", return_value=()),
            mock.patch.object(query_module, "_operand", side_effect=lambda r, d, c: Operand(ref=Ref(r), asset=asset)),
            mock.patch.object(query_module, "read", return_value=None),
        ):
            out = query_module.query(SimpleNamespace(), atom, *references)
        return out, get_atom, dispatch

    def test_a_hyphen_is_the_registered_underscore(self):
        _, get_atom, dispatch = self._query("cloth-coverage", {"depth": 1.0})
        get_atom.assert_called_once_with("cloth_coverage")
        self.assertEqual(dispatch.call_args.args[0], "cloth_coverage")

    def test_a_selector_narrows_the_answer(self):
        out, get_atom, _ = self._query("insertion:lat", {"depth": 1.0, "lat": 0.2})
        get_atom.assert_called_once_with("insertion")
        self.assertEqual(out, {"lat": 0.2})

    def test_a_selector_the_atom_does_not_report_is_refused(self):
        with self.assertRaises(query_module.QueryRefused):
            self._query("insertion:mate", {"depth": 1.0})

    def test_a_scalar_return_is_keyed_by_the_primary(self):
        out, _, _ = self._query("insertion", 0.5)
        self.assertEqual(out, {"depth": 0.5})

    def test_the_asset_scope_is_passed_as_bind_passes_it(self):
        _, _, dispatch = self._query("insertion", {"depth": 1.0}, "plug", asset="widget")
        self.assertEqual(dispatch.call_args.kwargs.get("asset"), "widget")


class TestAQueryTakesTheTaskScope(unittest.TestCase):
    """bind dispatches under ``task or doc.name``; a query must, or a task-scoped
    override scores the goal with one handler and answers the query with another."""

    def _query(self, **kwargs):
        from robosimstudio.assetlib.semantics import handlers

        spec = SimpleNamespace(args=(), reads_scene=False, returns=("depth",), primary="depth")
        candidates = [
            handlers.HandlerSpec(atom="insertion", fn=lambda *v: {"depth": 0.0}),
            handlers.HandlerSpec(atom="insertion", fn=lambda *v: {"depth": 1.0}, scope="task", owner="peg_task"),
        ]
        with (
            mock.patch.object(query_module, "leaves_of", return_value=[]),
            mock.patch.object(query_module, "get_atom", return_value=spec),
            mock.patch.object(handlers, "handlers_for", return_value=candidates),
            mock.patch.object(query_module, "_layouts", return_value=()),
            mock.patch.object(query_module, "_operand", side_effect=lambda r, d, c: Operand(ref=Ref(r))),
            mock.patch.object(query_module, "read", return_value=None),
        ):
            return query_module.query(SimpleNamespace(), "insertion", "plug", **kwargs)

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
        spec = SimpleNamespace(args=(), reads_scene=False)
        with (
            mock.patch.object(query_module, "leaves_of", return_value=[]),
            mock.patch.object(query_module, "get_atom", return_value=spec),
            mock.patch.object(query_module, "dispatch", side_effect=LookupError("no handler")),
            mock.patch.object(query_module, "_layouts", return_value={}),
            self.assertRaises(query_module.QueryRefused),
        ):
            query_module.query(SimpleNamespace(), "threaded", doc=None)

    def test_an_unknown_atom_is_refused(self):
        with (
            mock.patch.object(query_module, "leaves_of", return_value=[]),
            mock.patch.object(query_module, "get_atom", side_effect=LookupError("unknown atom")),
            self.assertRaises(query_module.QueryRefused),
        ):
            query_module.query(SimpleNamespace(), "nope", doc=None)


if __name__ == "__main__":
    unittest.main()
