# SPDX-License-Identifier: Apache-2.0
"""The inspector's text: what a clicked thing is, what it publishes, and each facet of its declaration.

Every line the inspector writes is built from the asset's record and the
entity's readings, and is held here to what it should say: a fact named once
however many of the record's fields repeat it, a reading with its unit and an
``internal`` one marked as no camera's, a point and a direction marked apart,
a recipe shown as its selector and a resolver as its key, and what a facet is
missing or pins said out loud. The records and facets are stand-ins with the
fields the inspector reads.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from robosimstudio_studio.inspector import TIER_MARK, PanelInspector
from tests.fixtures.studio_fakes import FakeEnv, FakeSession

TIER = "fast"
"""No GPU, no browser: records, facets and readings are namespaces; the gui is a stub."""


STEP_READ = SimpleNamespace(step=1)
"""A step's context, as ``env.last_context`` holds one once a step has been read."""


def _inspector(record=None, **env) -> PanelInspector:
    session = FakeSession(FakeEnv())
    for name, value in env.items():
        setattr(session.env, name, value)
    panel = PanelInspector(session)
    panel._record = lambda subject: record  # the asset library's answer, for this subject
    return panel


def _subject(name: str = "towel", kind: str = "cloth", entity=None) -> SimpleNamespace:
    return SimpleNamespace(key=f"{kind}:{name}", name=name, kind=kind, asset="test_towel", entity=entity)


def _record(**fields) -> SimpleNamespace:
    base = {"name": "towel", "substrate": "cloth", "tags": (), "summary": "", "published": (), "provenance": ""}
    return SimpleNamespace(**{**base, **fields})


def _facet(**fields) -> SimpleNamespace:
    base = {
        "name": "grasp",
        "kind": "grasp_points",
        "contracted": True,
        "summary": "",
        "tier": "literal",
        "members": (),
        "missing": (),
        "pins_mesh": False,
        "mesh_sha": "",
        "provenance": "",
    }
    return SimpleNamespace(**{**base, **fields})


def _member(name: str, frame: str = "", declared: bool = True, meaning: str = "") -> SimpleNamespace:
    return SimpleNamespace(name=name, frame=frame, declared=declared, value=(0, 0, 1), meaning=meaning)


class TestWhatItIs(unittest.TestCase):
    def test_a_subject_with_no_asset_says_it_is_not_typed(self) -> None:
        self.assertEqual("**cloth** `towel` -- *not typed against the catalog*", _inspector()._identity(_subject()))

    def test_each_fact_is_named_once_and_the_summary_follows(self) -> None:
        record = _record(substrate="cloth", tags=("towel", "terry", "terry", "cloth"), summary="A bath towel.")
        self.assertEqual("**cloth** `towel` `terry`\n\nA bath towel.", _inspector(record)._identity(_subject()))


class TestWhatItPublishes(unittest.TestCase):
    def _published(self, values, *states, context=STEP_READ) -> str:
        entity = SimpleNamespace(published_states=lambda ctx: values(ctx) if callable(values) else values)
        panel = _inspector(_record(published=states), last_context=context)
        return panel._published_text(_subject(entity=entity))

    def test_each_reading_has_its_unit_and_one_no_camera_sees_is_marked(self) -> None:
        text = self._published(
            {"opening": 0.0123456, "held_by": "left"},
            SimpleNamespace(name="opening", unit="m", tier="camera"),
            SimpleNamespace(name="held-by", unit="", tier="internal"),
            SimpleNamespace(name="folded", unit="", tier="camera"),  # not published this step
        )
        self.assertEqual("**publishes**\n- opening 0.01235 m\n- held-by left *(internal)*", text)

    def test_a_reading_that_raises_is_said_rather_than_raised(self) -> None:
        def broken(ctx):
            raise RuntimeError("no particles")

        text = self._published(broken, SimpleNamespace(name="opening", unit="m", tier="camera"))
        self.assertEqual("*unreadable: no particles*", text)

    def test_before_a_step_has_been_read_there_is_nothing_to_show(self) -> None:
        text = self._published({"opening": 1.0}, SimpleNamespace(name="opening", unit="", tier=""), context=None)
        self.assertEqual("", text)


class TestAFacet(unittest.TestCase):
    def test_a_literal_marks_points_directions_and_fields_its_kind_does_not_declare(self) -> None:
        facet = _facet(
            members=(
                _member("centre", "point", meaning="where the fingers meet"),
                _member("approach", "direction"),
                _member("width", declared=False),
            )
        )
        rows = _inspector()._facet_text(facet).splitlines()
        self.assertEqual("**grasp** `grasp_points`", rows[0])
        self.assertEqual(f"- `centre`{TIER_MARK['point']} = `(0, 0, 1)`", rows[1])
        self.assertTrue(rows[2].strip() == "where the fingers meet" and rows[2].startswith("  "))
        self.assertEqual(f"- `approach`{TIER_MARK['direction']} = `(0, 0, 1)`", rows[3])
        self.assertEqual("- `width` ⚠ undeclared = `(0, 0, 1)`", rows[4])

    def test_a_recipe_is_its_selector_and_a_resolver_its_key(self) -> None:
        panel = _inspector()
        recipe = panel._facet_text(_facet(tier="recipe", select="(near rim 0.01)"))
        self.assertIn("```\n(near rim 0.01)\n```", recipe)
        resolver = panel._facet_text(_facet(tier="resolver", resolver="grasps/towel@2"))
        self.assertIn("`resolver` → `grasps/towel@2`", resolver)

    def test_what_it_lacks_and_what_it_pins_are_said(self) -> None:
        panel = _inspector()
        text = panel._facet_text(
            _facet(contracted=False, missing=("centre", "width"), pins_mesh=True, mesh_sha="0123456789abcdef")
        )
        self.assertIn("*(uncontracted)*", text.splitlines()[0])
        self.assertIn("⚠ missing required: centre, width", text)
        self.assertIn("pins the discretisation to 0123456789ab", text)
        self.assertNotIn("0123456789abc", text, "twelve characters of the hash")
        unpinned = panel._facet_text(_facet(pins_mesh=True, provenance="measured\n   on the bench"))
        self.assertIn("pins the discretisation to **nothing**", unpinned)
        self.assertIn("> measured on the bench", unpinned)


class TestTheMaterialSheet(unittest.TestCase):
    def test_a_material_the_folder_tunes_goes_through_the_folder(self) -> None:
        cloth = object()
        panel = _inspector()
        panel.session.tune = SimpleNamespace(tuner=SimpleNamespace(entities={"shirt": object(), "towel": cloth}))
        subject = _subject(entity=cloth)
        self.assertEqual("towel", panel._sheet(subject, SimpleNamespace(name="material")))
        self.assertEqual("", panel._sheet(subject, SimpleNamespace(name="contact")), "only the material")
        self.assertEqual("", panel._sheet(_subject(entity=object()), SimpleNamespace(name="material")))

    def test_a_page_with_no_material_folder_writes_directly(self) -> None:
        self.assertEqual("", _inspector()._sheet(_subject(), SimpleNamespace(name="material")))


class TestASelection(unittest.TestCase):
    def test_it_is_rebuilt_on_the_drain_after_a_selection_and_gone_with_it(self) -> None:
        panel = _inspector()
        gui = panel.session.server.gui
        panel.build(gui)
        self.assertEqual([None], [f.label for f in gui.folders], "nothing but its anchor until something is clicked")
        panel.session.selected = _subject()
        panel.on_select(panel.session.selected)
        self.assertEqual(1, len(gui.folders), "selecting builds nothing off the main thread")
        panel.drain()
        folder = gui.folders[-1]
        self.assertEqual("Inspector: towel", folder.label)
        self.assertIs(gui.folders[0], folder.parent, "rebuilt inside the anchor, so it keeps its place")
        panel.session.selected = None
        panel.drain()
        self.assertTrue(folder.removed)


if __name__ == "__main__":
    unittest.main()
