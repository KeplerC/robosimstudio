# SPDX-License-Identifier: Apache-2.0
"""The studio session without a scene: which panels a page shows, what a click selects, resets and scenery.

The session is built without its constructor, over panels made with
``__new__`` -- their keys are their class names, so the real classes serve --
and a picker that answers whatever the test says the ray met. What a selection
keeps is a filter over the panels a scene warrants, and what a click selects is
whichever of a cloth and a shape the ray met first; both are bookkeeping, and
the page built over a real scene is ``test_studio_page``'s, in the medium tier.
"""

from __future__ import annotations

import types
import unittest
import unittest.mock
from unittest import mock

import numpy as np

from robosimstudio_studio import session as session_mod
from robosimstudio_studio.bench_panels import PanelSolution, PanelTask
from robosimstudio_studio.goal_panel import PanelGoal
from robosimstudio_studio.pick import ParticleHit, ShapeHit
from robosimstudio_studio.session import (
    DOCUMENT_PANELS,
    PANEL_SETS,
    PanelArm,
    PanelCamera,
    PanelCloth,
    PanelInspector,
    PanelLeaderArm,
    PanelObject,
    PanelOverlay,
    PanelRecord,
    PanelRegions,
    PanelRenderView,
    PanelScene,
    PanelVideo,
    StudioConfig,
    StudioSession,
    panel_key,
)
from robosimstudio_studio.tuning.panels import PanelReadings, PanelTune

TIER = "fast"
"""No GPU: sessions built without their constructors."""


DRIVE = ("inspector", "overlay", "arm", "object", "cloth")
"""What ``panels="drive"`` keeps: the things you take hold of, and what is behind the one you clicked."""


def _bare(cls):
    """A panel of the real class, never built: its key is all a selection reads."""
    return cls.__new__(cls)


def _page(panels="all", *, task: bool = False) -> tuple[StudioSession, list]:
    """A session over every panel a page with an arm, an object and a cloth warrants, before the selection."""
    session = StudioSession.__new__(StudioSession)
    session.config = StudioConfig(panels=panels)
    session.env = types.SimpleNamespace(task=object()) if task else types.SimpleNamespace()
    session.objects, session.deformables = _bare(PanelObject), _bare(PanelCloth)
    session.video, session.record, session.scene_picker = _bare(PanelVideo), _bare(PanelRecord), _bare(PanelScene)
    session.arms = {"left": _bare(PanelArm), "right": _bare(PanelArm)}
    session.select_request, session.click_ray, session.picker = None, None, None
    built = [
        _bare(PanelRenderView),
        _bare(PanelInspector),
        _bare(PanelCamera),
        session.scene_picker,
        *session.arms.values(),
        _bare(PanelLeaderArm),
        session.objects,
        session.deformables,
        _bare(PanelTune),
        _bare(PanelReadings),
        _bare(PanelRegions),
        _bare(PanelOverlay),
        session.record,
        session.video,
    ]
    if task:
        built[1:1] = [_bare(PanelTask), _bare(PanelGoal), _bare(PanelSolution)]
    return session, built


def _keys(panels) -> set[str]:
    return {panel_key(p) for p in panels}


class TestTheSelection(unittest.TestCase):
    """A selection filters the panels a scene warrants; it does not decide which exist."""

    def test_all_is_the_default_and_is_no_filter(self) -> None:
        self.assertEqual("all", StudioConfig().panels)
        session, built = _page()
        keys = _keys(session._selected(built))
        self.assertLessEqual(set(DRIVE), keys)
        # The half a selection exists to remove is present without one.
        for key in ("video", "record", "overlay", "regions"):
            with self.subTest(panel=key):
                self.assertIn(key, keys)

    def test_drive_keeps_exactly_the_drive_panels(self) -> None:
        session, built = _page("drive")
        self.assertEqual(set(DRIVE), _keys(session._selected(built)))
        self.assertEqual(DRIVE, tuple(PANEL_SETS["drive"]))

    def test_the_keys_may_be_given_directly(self) -> None:
        session, built = _page(("object", "cloth"))
        self.assertEqual({"object", "cloth"}, _keys(session._selected(built)))

    def test_drive_leaves_out_the_scene_list_and_the_look(self) -> None:
        """``Appearance`` is built inline rather than as a panel, so it is gated through ``_wants``."""
        session, built = _page("drive")
        self.assertNotIn("scene", _keys(session._selected(built)))
        self.assertFalse(session._wants("appearance"))
        whole, built = _page()
        self.assertIn("scene", _keys(whole._selected(built)))
        self.assertTrue(whole._wants("appearance"))

    def test_a_dropped_panel_is_forgotten_not_just_hidden(self) -> None:
        """``drain_click`` reaches ``self.objects`` by name; only its ``is not None`` guard stands between a
        click and a panel whose ``build`` never ran."""
        session, built = _page(("scene",))
        kept = session._selected(built)
        self.assertEqual(["scene"], [panel_key(p) for p in kept])
        self.assertIsNone(session.objects)
        self.assertIsNone(session.deformables)
        self.assertIsNone(session.video)
        self.assertIs(kept[0], session.scene_picker, "a kept panel keeps its name")
        self.assertEqual({}, session.arms)
        session.drain_click()  # no ray pending; must not raise

    def test_an_unknown_name_says_what_the_names_are(self) -> None:
        session, built = _page("minimal")
        with self.assertRaises(ValueError) as caught:
            session._selected(built)
        self.assertIn("drive", str(caught.exception))


class TestATaskPageKeepsItsDocumentPanels(unittest.TestCase):
    def test_a_named_selection_on_a_task_keeps_the_goal_the_regions_and_the_runner(self) -> None:
        session, built = _page("drive", task=True)
        self.assertEqual((*DRIVE, *DOCUMENT_PANELS), session._wanted_panels())
        kept = _keys(session._selected(built))
        self.assertEqual(set(DRIVE) | set(DOCUMENT_PANELS), kept)
        self.assertNotIn("task", kept, "the menu swaps the page onto another document")

    def test_keys_given_on_a_task_are_taken_as_they_are(self) -> None:
        session, _ = _page(("object",), task=True)
        self.assertEqual(("object",), session._wanted_panels())

    def test_all_on_a_task_is_still_no_filter(self) -> None:
        session, built = _page(task=True)
        self.assertEqual((), session._wanted_panels())
        self.assertEqual(len(built), len(session._selected(built)))


class TestAPanelsKey(unittest.TestCase):
    def test_it_is_the_class_name_without_panel_lowercased(self) -> None:
        self.assertEqual("renderview", panel_key(_bare(PanelRenderView)))

    def test_a_key_attribute_wins(self) -> None:
        panel = _bare(PanelVideo)
        panel.key = "film"
        self.assertEqual("film", panel_key(panel))

    def test_a_duck_typed_panel_is_keyed_the_same_way(self) -> None:
        self.assertEqual("lamp", panel_key(type("PanelLamp", (), {})()))


class _Picker:
    """Answers a ray with whatever the test says it met."""

    def __init__(self, shape=None, particle=None) -> None:
        self.shape, self.particle, self.asked = shape, particle, []

    def shape_at(self, state, origin, direction):
        self.asked.append("shape")
        return self.shape

    def particle_at(self, state, origin, direction):
        self.asked.append("particle")
        return self.particle


def _clicking(shape=None, particle=None, *, owner: str = "towel", labels=()) -> StudioSession:
    """A session with a ray pending, a cloth panel and an object panel that say what they did."""
    session = StudioSession.__new__(StudioSession)
    session.select_request, session.selected, session.panels = None, None, []
    session.click_ray = (np.zeros(3), np.array([1.0, 0.0, 0.0]))
    session.picker = _Picker(shape, particle)
    session.arms, session.active_arm, session.status, session.scene_picker = {}, "", "", None
    session.env = types.SimpleNamespace(state=None, model=types.SimpleNamespace(body_label=list(labels)))
    session.deformables = types.SimpleNamespace(
        owner_of=lambda index: owner, click_at=lambda name, point: f"took {name} at {point[0]:.1f}"
    )
    session.objects = types.SimpleNamespace(click_body=lambda body: f"took body {body}")
    session.index = types.SimpleNamespace(
        at_particle=lambda index: types.SimpleNamespace(key=f"particle {index}"),
        at_shape=lambda shape: types.SimpleNamespace(key=f"shape {shape}", label=f"shape {shape}"),
        at_body=lambda body: None,
        by_key=lambda key: types.SimpleNamespace(key=key),
    )
    return session


def _shape(distance: float, body: int = 3) -> ShapeHit:
    return ShapeHit(shape=7, body=body, point=np.array([distance, 0.0, 0.0]), distance=distance)


def _particle(distance: float) -> ParticleHit:
    return ParticleHit(index=11, point=np.array([distance, 0.0, 0.0]), distance=distance, miss=0.01)


class TestAClick(unittest.TestCase):
    """Whichever of a cloth and a shape the ray met first wins, with no mode to switch between them."""

    def test_a_cloth_in_front_of_a_box_is_the_cloth(self) -> None:
        session = _clicking(_shape(1.0), _particle(0.5))
        session.drain_click()
        self.assertEqual("took towel at 0.5", session.status)
        self.assertEqual("particle 11", session.selected.key)
        self.assertIsNone(session.click_ray, "a click is taken once")

    def test_a_box_in_front_of_a_cloth_is_the_box(self) -> None:
        session = _clicking(_shape(0.4), _particle(0.5))
        session.drain_click()
        self.assertEqual("took body 3", session.status)
        self.assertEqual("shape 7", session.selected.key)

    def test_a_particle_no_cloth_owns_falls_through_to_the_shape(self) -> None:
        session = _clicking(_shape(1.0), _particle(0.5), owner="")
        session.drain_click()
        self.assertEqual("took body 3", session.status)

    def test_a_page_with_no_cloth_panel_never_asks_for_a_particle(self) -> None:
        session = _clicking(_shape(1.0), _particle(0.5))
        session.deformables = None
        session.drain_click()
        self.assertEqual(["shape"], session.picker.asked)
        self.assertEqual("took body 3", session.status)

    def test_a_click_on_nothing_says_so(self) -> None:
        session = _clicking()
        session.drain_click()
        self.assertEqual("nothing there", session.status)
        self.assertIsNone(session.selected)

    def test_without_an_object_panel_a_shape_is_named_by_what_owns_it(self) -> None:
        session = _clicking(_shape(1.0))
        session.objects = None
        session.drain_click()
        self.assertEqual("shape 7", session.status)

    def test_a_body_nothing_names_is_named_by_its_label_or_its_index(self) -> None:
        session = _clicking(labels=("world", "robot/base", "robot/link3"))
        self.assertEqual("link3", session._body_name(2))
        self.assertEqual("body 7", session._body_name(7))
        self.assertEqual("body -1", session._body_name(-1))

    def test_a_scene_list_pick_is_a_selection_without_a_ray(self) -> None:
        session = _clicking()
        session.click_ray = None
        session.select_request = "cube"
        session.drain_click()
        self.assertEqual("cube", session.selected.key)
        self.assertIsNone(session.select_request)
        self.assertEqual([], session.picker.asked)

    def test_the_same_subject_twice_tells_the_panels_once(self) -> None:
        session = _clicking()
        told = []
        session.panels = [types.SimpleNamespace(on_select=told.append)]
        cube = types.SimpleNamespace(key="cube")
        session.select(cube)
        session.select(types.SimpleNamespace(key="cube"))
        self.assertEqual([cube], told)
        session.select(None)
        self.assertEqual([cube, None], told, "clearing it is news")


class TestAResetThatRecompilesRebindsThePage(unittest.TestCase):
    def _session(self, swap: bool):
        from robosimstudio_studio.session import StudioSession

        session = StudioSession.__new__(StudioSession)
        env = types.SimpleNamespace(model=object())

        def reset():
            if swap:
                env.model = object()
            return {}, {}

        env.reset = reset
        session.env = env
        session.panels = []
        session._drop_panels = unittest.mock.Mock()
        session._bind = unittest.mock.Mock()
        session.on_reset = unittest.mock.Mock()
        return session

    def test_a_new_model_rebuilds_the_page_around_it(self) -> None:
        session = self._session(swap=True)
        self.assertTrue(session.reset_now())
        session._drop_panels.assert_called_once_with()
        session._bind.assert_called_once_with(session.env)
        session.on_reset.assert_called_once()

    def test_the_same_model_leaves_the_page_alone(self) -> None:
        session = self._session(swap=False)
        self.assertTrue(session.reset_now())
        session._bind.assert_not_called()
        session._drop_panels.assert_not_called()


class SceneryInvalidatesPicker(unittest.TestCase):
    def test_apply_scenery_invalidates_the_picker(self):
        stub = mock.MagicMock()
        with (
            mock.patch.object(session_mod, "garment_visuals", return_value=()),
            mock.patch.object(session_mod, "scene_repaints_live", return_value=False),
        ):
            StudioSession.apply_scenery(stub)
        stub._set_scenery_visible.assert_called_once()
        stub.picker.invalidate.assert_called_once_with()

    def test_no_picker_is_fine(self):
        stub = mock.MagicMock()
        stub.picker = None
        with (
            mock.patch.object(session_mod, "garment_visuals", return_value=()),
            mock.patch.object(session_mod, "scene_repaints_live", return_value=False),
        ):
            StudioSession.apply_scenery(stub)
        stub._set_scenery_visible.assert_called_once_with(stub.show_scenery)
        stub.viewer.set_model.assert_called_once_with(stub.env.model)  # the re-batch still happens


if __name__ == "__main__":
    unittest.main()
