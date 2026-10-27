# SPDX-License-Identifier: Apache-2.0
"""The page shows what it was asked to show, and a script can add to it.

Two complaints, one about each direction. The studio builds about a dozen
folders over a bare scene -- cameras, video, takes, material presets, regions,
an event log -- and an example that is showing one rope has use for none of
them; that is *too much*. And the only way to put a number of your own on the
page was to not use the studio at all, which is what left every example driving
a bare ``rss.viewer`` and hand-writing a loop; that is *too little*.

So: a selection, and a seam.

The selection is a filter and nothing more. *Which panels a scene warrants* is
still decided by the scene -- a page over a cable has no cloth folder however it
is configured -- and :meth:`StudioSession._selected` only decides which of those
to show. Both halves are checked here, because a filter that quietly became the
thing deciding what exists would pass a test that only counted folders.

Transport is deliberately not selectable: Pause, Speed and Reset are built
inline rather than as a panel, so no selection can take away the ability to stop
the thing you are looking at.
"""

from __future__ import annotations

import importlib.util
import unittest

TIER = "medium"
"""Compiles scenes and starts a viser server on a spare port."""

HAS_GAP = importlib.util.find_spec("gap") is not None

DRIVE = ("inspector", "overlay", "arm", "object", "cloth")
"""What ``panels="drive"`` keeps: the things you take hold of, and what is behind
the one you clicked. Not the scene list -- clicking is the way in -- and not
``Appearance``, which is built inline and gated the same way."""


def _sim(cloth: bool = True):
    import robosimstudio as rss

    scene = rss.Scene(cell="void", name="panels")
    scene.add(rss.robot("panda"))
    scene.add(rss.box(name="cube", half=(0.025,) * 3, at=(0.0, 0.0, 0.20)))
    if cloth:
        scene.add(rss.sheet(name="towel", size=(0.2, 0.2), cells=(8, 8), at=(-0.1, -0.1, 0.4)))
    return scene.build()


def _session(sim, port: int, **config):
    from robosimstudio_studio.session import StudioConfig, StudioSession

    return StudioSession(sim, config=StudioConfig(port=port, **config))


def _keys(session) -> set[str]:
    from robosimstudio_studio.session import panel_key

    return {panel_key(p) for p in session.panels}


class TestTheSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = _sim()

    def test_all_is_the_default_and_is_no_filter(self) -> None:
        session = _session(self.sim, 8951)
        try:
            keys = _keys(session)
            self.assertLessEqual(set(DRIVE), keys)
            # The half a selection exists to remove is present without one.
            # No "events": the narration panel left with the description layer.
            for key in ("video", "record", "overlay", "regions"):
                with self.subTest(panel=key):
                    self.assertIn(key, keys)
        finally:
            session.teardown()

    def test_drive_keeps_exactly_the_drive_panels(self) -> None:
        session = _session(self.sim, 8952, panels="drive")
        try:
            self.assertEqual(set(DRIVE), _keys(session))
        finally:
            session.teardown()

    def test_the_keys_may_be_given_directly(self) -> None:
        session = _session(self.sim, 8953, panels=("object", "cloth"))
        try:
            self.assertEqual({"object", "cloth"}, _keys(session))
        finally:
            session.teardown()

    def test_drive_leaves_out_the_scene_list_and_the_look(self) -> None:
        """Both are session-wide furniture, and a page showing one rope wants
        neither. ``Appearance`` is built inline rather than as a panel, so it is
        gated by the same selection through ``_wants``."""
        session = _session(self.sim, 8959, panels="drive")
        try:
            self.assertNotIn("scene", _keys(session))
            self.assertFalse(session._wants("appearance"))
        finally:
            session.teardown()
        whole = _session(self.sim, 8960)
        try:
            self.assertIn("scene", _keys(whole))
            self.assertTrue(whole._wants("appearance"))
        finally:
            whole.teardown()

    def test_a_dropped_panel_is_forgotten_not_just_hidden(self) -> None:
        """``drain_click`` reaches ``self.objects`` by name, and its ``is not
        None`` guard is the only thing between a click and a panel whose
        ``build`` never ran."""
        session = _session(self.sim, 8954, panels=("scene",))
        try:
            self.assertIsNone(session.objects)
            self.assertIsNone(session.deformables)
            self.assertIsNone(session.video)
            self.assertEqual({}, session.arms)
            session.drain_click()  # no ray pending; must not raise
        finally:
            session.teardown()

    def test_an_unknown_name_says_what_the_names_are(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _session(self.sim, 8955, panels="minimal").teardown()
        self.assertIn("drive", str(caught.exception))


class TestTheSceneStillDecidesWhatExists(unittest.TestCase):
    """A selection filters; it does not conjure."""

    def test_asking_for_cloth_on_a_scene_with_none_gets_none(self) -> None:
        sim = _sim(cloth=False)
        session = _session(sim, 8956, panels=("object", "cloth"))
        try:
            self.assertEqual({"object"}, _keys(session))
            self.assertIsNone(session.deformables)
        finally:
            session.teardown()


class TestATaskPageKeepsItsDocument(unittest.TestCase):
    """``rss.view`` on a task: ``drive``, and still the goal, its square and Run.

    ``drive`` was written for hand-assembled scenes and filtered a task's page
    down to the same four keys, so ``examples/06_tasks`` opened with no goal
    beside the scene and no square to drag the cube into, and the policy pages
    in ``examples/07_policies`` had no Run. Opened through :func:`robosimstudio.view`
    itself, with only the blocking loop replaced, so the door's own default is
    what is under test.
    """

    @staticmethod
    def _view(env, port: int):
        from unittest import mock

        import robosimstudio as rss
        from robosimstudio_studio.session import StudioSession

        opened: list = []
        # A bound list.append would take no self here: the lambda is what makes it a method.
        with mock.patch.object(StudioSession, "run", lambda self: opened.append(self)):  # noqa: PLW0108
            rss.view(env, port=port, share=False, speed="max")
        (session,) = opened
        return session

    def test_the_goal_its_regions_and_the_runner_stay(self) -> None:
        from robosimstudio.sim.draw import _VIEW_DEFAULTS
        from tests.fixtures.examples import _run

        self.assertEqual("drive", _VIEW_DEFAULTS["panels"])
        env, _ = _run("06_tasks/00_your_first_task.py")
        session = self._view(env, 9220)
        try:
            keys = _keys(session)
            self.assertLessEqual({"goal", "regions", "solution"}, keys, "the document's panels")
            self.assertLessEqual(set(DRIVE) - {"cloth"}, keys, "and what drive keeps on any page")
            for key in ("task", "scene", "video", "record", "camera"):
                with self.subTest(dropped=key):
                    self.assertNotIn(key, keys)
            self.assertIn("containment", session.goal_panel._tree_text(), "the clause, beside the scene")
            self.assertIn("square", session.env.handles.regions)
            self.assertIsNotNone(session.driver, "the Policy folder, which holds Run")
        finally:
            session.teardown()

    def test_a_bare_scene_under_drive_is_unchanged(self) -> None:
        """The other half: no document, nothing added -- the same four keys as ever."""
        session = self._view(_sim(cloth=True), 9221)
        try:
            self.assertEqual(set(DRIVE), _keys(session))
        finally:
            session.teardown()

    def test_keys_given_explicitly_are_taken_as_they_are(self) -> None:
        from tests.fixtures.examples import _run

        env, _ = _run("06_tasks/00_your_first_task.py")
        session = _session(env, 9222, panels=("object",))
        try:
            self.assertEqual({"object"}, _keys(session))
            self.assertIsNone(session.goal_panel)
        finally:
            session.teardown()

    @unittest.skipUnless(HAS_GAP, "the graph runtime is not installed (the [gap] extra)")
    def test_run_places_the_cube_from_the_policy_page(self) -> None:
        """What the 08 clips press: Run, and the page's own graph picks and places."""
        from tests.fixtures.examples import _run

        env, _ = _run("07_policies/00_pick_and_place.py")
        session = self._view(env, 9223)
        try:
            self.assertIn("pick", session.driver.choices)
            session.driver.start_requested = True  # what the Run button's callback writes
            session.tick()
            self.assertTrue(session.driver.running, session.driver.message)
            for _ in range(1500):  # the document's max-steps; the graph takes about 600
                if not session.driver.running:
                    break
                session.tick()
            self.assertIn("SUCCESS", session.driver.message)
        finally:
            session.driver.stop("")
            session.teardown()


class TestABareScenePageIsUsable(unittest.TestCase):
    """Three things a stubbed page cannot see, each of which shipped broken.

    The examples are hand-assembled scenes, and every one of these is a place
    where the studio reached for something only a *task* has. A stub that
    records what a panel added never calls ``refresh``, so none of it showed up
    until a real session ran eight steps.
    """

    def test_a_robot_and_nothing_loose_still_gets_an_object_panel(self) -> None:
        """Gated on the objects table alone, a page whose only movable thing was
        the robot arrived with nothing clickable at all."""
        import robosimstudio as rss

        scene = rss.Scene(cell="void", name="justarobot")
        scene.add(rss.robot("panda"))
        session = _session(scene.build(control_mode="joint_position"), 8961, panels="drive")
        try:
            self.assertIsNotNone(session.objects, "a robot alone got no handles")
            self.assertIn("object", _keys(session))
            self.assertTrue(session.objects.arm_bodies, "the arm's links are not draggable")
        finally:
            session.teardown()

    def test_reset_works_on_a_scene_with_no_task(self) -> None:
        """``Env.reset`` takes an episode and ``EnvBase.reset`` does not. ``adopt``
        dispatched on that and the Reset button did not, so pressing Reset on any
        example paused the page on a ``TypeError``."""
        session = _session(_sim(cloth=False), 8963, panels="drive")
        try:
            session.reset_requested = True
            session.tick()
            self.assertNotIn("did not load", session.status)
            self.assertFalse(session.paused, f"Reset paused the page: {session.status}")
            session.tick()
            self.assertTrue(session.env.state_is_finite())
        finally:
            session.teardown()

    def test_a_cloth_page_refreshes_without_a_document(self) -> None:
        """``PanelCloth`` read ``env.doc`` and ``env.last_context`` outright, and
        ``PanelTune`` read ``env.task``: three attributes a ``Sim`` has none of.
        Stepping is what reaches them, so it is stepped."""
        session = _session(_sim(cloth=True), 8962, panels="all")
        try:
            for _ in range(4):
                session.tick()
            self.assertIsNotNone(session.deformables)
        finally:
            session.teardown()


class TestLettingGoIsVisible(unittest.TestCase):
    """Taking hold of a thing is one click on the thing. Letting go used to be a
    button two folds deep in a panel named after a *kind*, which made a handle
    read as something you could not get rid of."""

    def _widgets(self, session):
        return (session._holding_readout, session._linked_toggle, session._release_last, session._release_all)

    def test_the_controls_appear_when_you_are_holding_something(self) -> None:
        sim = _sim(cloth=False)
        session = _session(sim, 8964, panels="drive")
        try:
            session.refresh_holding()
            for widget in self._widgets(session):
                self.assertFalse(widget.visible, "a release control before anything was picked up")

            body = int(sorted(session.objects.by_body)[0])
            session.objects.click_body(body)
            session.objects.drain()
            session.refresh_holding()
            for widget in self._widgets(session):
                self.assertTrue(widget.visible)
            self.assertNotEqual("nothing", session._holding_readout.value)

            session.release_all_requested = True
            session.tick()
            self.assertEqual([], session.held())
            session.refresh_holding()
            self.assertEqual("nothing", session._holding_readout.value)
            for widget in self._widgets(session):
                self.assertFalse(widget.visible)
        finally:
            session.teardown()

    def test_release_last_lets_go_of_one_and_leaves_the_rest(self) -> None:
        sim = _sim(cloth=False)
        session = _session(sim, 8967, panels="drive")
        try:
            first, second = (int(b) for b in sorted(session.objects.by_body)[:2])
            for body in (first, second):
                session.objects.click_body(body)
                session.objects.drain()
            self.assertEqual(2, len(session.objects.handles.live))
            session.release_last_requested = True
            session.tick()
            live = session.objects.handles.live
            self.assertEqual(1, len(live), "Release last let go of more than one")
            self.assertEqual(first, int(live[0].key), "it let go of the older one")
        finally:
            session.teardown()

    def test_release_last_spans_kinds(self) -> None:
        """A cloth handle and an object handle are two sets, and ``Release last``
        has to mean the last thing *you* took hold of across both."""
        import numpy as np

        sim = _sim(cloth=True)
        session = _session(sim, 8968, panels="drive")
        try:
            session.objects.click_body(int(sorted(session.objects.by_body)[0]))
            session.objects.drain()
            session.deformables.click_at("towel", np.asarray(sim["towel"].particle_q[0], dtype=np.float64))
            session.deformables.drain()
            self.assertTrue(session.deformables.handles.live, "the cloth grab missed, so this is untested")

            session.release_last_requested = True
            session.tick()
            self.assertEqual([], session.deformables.handles.live, "the cloth was the last grabbed")
            self.assertTrue(session.objects.handles.live, "and the box was not")
        finally:
            session.teardown()

    def test_move_together_reaches_every_kind(self) -> None:
        sim = _sim(cloth=True)
        session = _session(sim, 8969, panels="drive")
        try:
            session.linked_request = True
            session.tick()
            self.assertTrue(session.objects.handles.linked)
            self.assertTrue(session.deformables.handles.linked)
        finally:
            session.teardown()

    def test_a_selected_page_builds_no_per_kind_folds(self) -> None:
        """``Objects`` and ``Deformable`` are the studio's furniture. Their shared
        controls moved to the session, and what is left -- reaching a body a click
        cannot, grabbing a declared strip by name -- is worth a fold on a task and
        is noise on a page about one towel."""
        from robosimstudio_studio.panels import PanelCloth, PanelObject
        from tests.fixtures.viser_stubs import _StubGui, _StubSession

        sim = _sim(cloth=True)
        for chrome, expected in ((False, set()), (True, {"Objects", "Deformable"})):
            with self.subTest(chrome=chrome):
                gui = _StubGui()
                session = _StubSession(sim, gui)
                for cls in (PanelObject, PanelCloth):
                    cls(session, handles=2, chrome=chrome).build(gui)
                labels = {f.label for f in gui.folders if f.label}
                self.assertEqual(expected, labels & {"Objects", "Deformable"})

    def test_the_missed_grab_still_says_so_without_a_fold(self) -> None:
        """The cloth fold carried "missed by 30 mm"; a page with no fold would
        otherwise answer a wide click with nothing, which reads as broken."""
        sim = _sim(cloth=True)
        session = _session(sim, 8970, panels="drive")
        try:
            session.deformables.click_at("towel", (5.0, 5.0, 5.0))
            session.deformables.drain()
            session.refresh_holding()
            self.assertIn("missed by", session._holding_readout.value)
            self.assertTrue(session._holding_readout.visible)
            self.assertFalse(session._release_all.visible, "nothing is held, so nothing to release")
        finally:
            session.teardown()


class TestAGripPinnedToAHandIsTakenWhereItIs(unittest.TestCase):
    """A click on the cloth with its slot pinned to a hand holds the patch where it lies, in the hand's frame.

    The slot is aimed at the hand's pose every step. Seized at the click in the
    identity frame, the patch was walked onto the hand from the first step (15 mm
    a step, the pin's rate limit) and turned by the hand's orientation; a declared
    set taken onto a hand already kept the hand's frame, and a click now takes the
    same one.
    """

    def test_the_patch_stays_put_and_rides_the_hand(self) -> None:
        import numpy as np

        from tests.fixtures.grasp import open_session, tick

        sim = _sim(cloth=True)
        session = open_session(sim, 8971)
        try:
            grips = session.deformables
            arm = next(iter(session.arms))
            grips.follow_arm[0] = arm
            towel = sim["towel"]
            before = np.asarray(sim.state.particle_q.numpy(), dtype=np.float64)
            grips.click_at("towel", before[int(towel.particles[0])])
            tick(session, 10)
            held = np.asarray(grips.draggers[0].grabbed_indices, dtype=np.int64)
            self.assertTrue(len(held), "the click missed, so this is untested")
            after = np.asarray(sim.state.particle_q.numpy(), dtype=np.float64)
            moved = float(np.linalg.norm(after[held] - before[held], axis=1).max())
            hand = np.asarray(session.arms[arm].ik.ee_pose(sim.state)[0], dtype=np.float64)
            self.assertGreater(float(np.linalg.norm(before[held] - hand, axis=1).min()), 0.1, "far from the hand")
            self.assertLess(
                moved, 0.005, f"the held patch moved {moved * 1000:.0f} mm toward the hand in 10 steps; 127 before"
            )
        finally:
            session.teardown()

    def test_a_pin_chosen_while_it_holds_keeps_the_patch_where_it_is(self) -> None:
        """Pinned in the slot's dropdown after the click, the grip stayed in the click's frame and rode onto the hand."""
        import numpy as np

        from tests.fixtures.grasp import tick

        sim = _sim(cloth=True)
        session = _session(sim, 8972)
        try:
            grips = session.deformables
            arm = next(iter(session.arms))
            inputs = session.server.gui._gui_input_handle_from_uuid.values()
            pins = sorted((h for h in inputs if str(getattr(h, "label", "")).startswith("Pin ")), key=lambda h: h.label)
            towel = sim["towel"]
            grips.click_at("towel", np.asarray(sim.state.particle_q.numpy(), dtype=np.float64)[int(towel.particles[0])])
            tick(session, 1)
            (slot,) = [h.index for h in grips.handles.live]
            held = np.asarray(grips.draggers[slot].grabbed_indices, dtype=np.int64)
            self.assertTrue(len(held), "the click missed, so this is untested")
            before = np.asarray(sim.state.particle_q.numpy(), dtype=np.float64)
            pins[slot].value = arm  # what the browser does when the user pins the slot to the hand
            self.assertEqual(arm, grips.follow_arm[slot])
            tick(session, 10)
            after = np.asarray(sim.state.particle_q.numpy(), dtype=np.float64)
            moved = float(np.linalg.norm(after[held] - before[held], axis=1).max())
            hand = np.asarray(session.arms[arm].ik.ee_pose(sim.state)[0], dtype=np.float64)
            self.assertGreater(float(np.linalg.norm(before[held] - hand, axis=1).min()), 0.1, "far from the hand")
            self.assertLess(moved, 0.005, f"the held patch moved {moved * 1000:.0f} mm toward the hand in 10 steps")
            self.assertEqual(sorted(held.tolist()), sorted(grips.draggers[slot].grabbed_indices.tolist()))
        finally:
            session.teardown()


def _sim_with_an_edge():
    """``_sim``'s towel with one declared set, its near edge row, for a set grab to take."""
    import robosimstudio as rss

    scene = rss.Scene(cell="void", name="panels")
    scene.add(rss.robot("panda"))
    edge = ("edge", lambda rest: rest[:, 1] <= rest[:, 1].min() + 1e-4)
    scene.add(rss.sheet(name="towel", size=(0.2, 0.2), cells=(8, 8), at=(-0.1, -0.1, 0.4), particle_sets=(edge,)))
    return scene.build()


class TestASetGrabsPinGoesWithTheGrab(unittest.TestCase):
    """A pin ``grab_set(arm=...)`` puts on a slot shows on the slot's dropdown and goes when the grab is let go.

    The pin was written to ``follow_arm`` and nowhere else: the ``Pin N to hand``
    dropdown went on showing no hand, so the page could not unpin it, and the pin
    outlived the grab -- the next click that took the slot rode the hand. A pin the
    user set in the dropdown is the user's and stays through a release.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = _sim_with_an_edge()

    def _open(self, port: int):
        session = _session(self.sim, port)
        inputs = session.server.gui._gui_input_handle_from_uuid.values()
        pins = sorted((h for h in inputs if str(getattr(h, "label", "")).startswith("Pin ")), key=lambda h: h.label)
        return session, session.deformables, next(iter(session.arms)), pins

    def test_the_dropdown_follows_the_pin_both_ways(self) -> None:
        session, grips, arm, pins = self._open(8977)
        try:
            self.assertTrue(grips.grab_set("edge", arm=arm), "the set grab took nothing, so this is untested")
            (slot,) = [h.index for h in grips.handles.live]
            self.assertEqual(arm, pins[slot].value, "the dropdown shows the hand the grab pinned the slot to")
            pins[slot].value = ""  # what the browser does when the user unpins it
            self.assertEqual("", grips.follow_arm[slot], "and unpinning it there unpins the slot")
        finally:
            session.teardown()

    def test_letting_go_unpins_what_the_grab_pinned(self) -> None:
        import numpy as np

        session, grips, arm, pins = self._open(8978)

        def reset() -> None:
            session.reset_requested = True
            session.tick()

        try:
            for release in (session.release_all, session.release_last, reset):
                self.assertTrue(grips.grab_set("edge", arm=arm), "the set grab took nothing, so this is untested")
                (slot,) = [h.index for h in grips.handles.live]
                release()
                self.assertEqual("", grips.follow_arm[slot], "the grab's pin went with it")
                self.assertEqual("", pins[slot].value, "and the dropdown says so")
            q = np.asarray(self.sim.state.particle_q.numpy(), dtype=np.float64)
            grips.click_at("towel", q[int(self.sim["towel"].particles[-1])])
            grips.drain()
            (click,) = [h.index for h in grips.handles.live]
            self.assertEqual("", grips.follow_arm[click], "a click on the freed slot stays on the mouse")
        finally:
            session.teardown()

    def test_a_pin_the_user_set_stays(self) -> None:
        session, grips, arm, pins = self._open(8979)
        try:
            pins[0].value = arm  # the user pins slot 1 to the hand
            self.assertTrue(grips.grab_set("edge"), "the set grab took nothing, so this is untested")
            self.assertEqual([0], [h.index for h in grips.handles.live])
            session.release_all()
            self.assertEqual(arm, grips.follow_arm[0], "a release leaves the user's pin")
            self.assertEqual(arm, pins[0].value)
        finally:
            session.teardown()


class TestAPagesSetGrabIsItsOwn(unittest.TestCase):
    """``request_set`` names its cloth, set and hand, and its release lets go of what it took and nothing else.

    The fold page used to write the panel's selections and its ``grab_set_arm`` to
    ask, leaving them changed for the panel's own **Grab set**, and let go by hand:
    every handle pinned to the right hand, a grip the user pinned there included.
    """

    def test_the_selections_stay_and_the_users_grip_stays(self) -> None:
        import numpy as np

        sim = _sim_with_an_edge()
        session = _session(sim, 8980)
        try:
            grips, arm = session.deformables, next(iter(session.arms))
            q = np.asarray(sim.state.particle_q.numpy(), dtype=np.float64)
            grips.click_at("towel", q[int(sim["towel"].particles[-1])])  # the far corner, nowhere near the edge
            grips.drain()
            (click,) = [h.index for h in grips.handles.live]
            grips.follow_arm[click] = arm  # the user pins it to the same hand
            grips.selected_set = ""
            grips.request_set("towel", "edge", arm)
            grips.drain()
            (edge,) = [h.index for h in grips.handles.live if h.index != click]
            self.assertEqual(arm, grips.follow_arm[edge], "the edge rides the hand")
            self.assertEqual("", grips.selected_set, "and the panel's Set selection is the user's still")
            grips.request_set("towel", "edge", arm, hold=False)
            grips.drain()
            self.assertEqual([click], [h.index for h in grips.handles.live], "only the request's grab is let go")
            self.assertEqual(arm, grips.follow_arm[click], "and the user's pin stays")
        finally:
            session.teardown()

    def test_a_grip_on_the_mouse_stays_on_the_mouse(self) -> None:
        """The fold page's box pinned slot 0 to the hand: a click already there rode it, and the edge rode none."""
        import numpy as np

        sim = _sim_with_an_edge()
        session = _session(sim, 8982)
        try:
            grips, arm = session.deformables, next(iter(session.arms))
            inputs = session.server.gui._gui_input_handle_from_uuid.values()
            pins = sorted((h for h in inputs if str(getattr(h, "label", "")).startswith("Pin ")), key=lambda h: h.label)
            q = np.asarray(sim.state.particle_q.numpy(), dtype=np.float64)
            grips.click_at("towel", q[int(sim["towel"].particles[-1])])  # the far corner, nowhere near the edge
            grips.drain()
            (click,) = [h.index for h in grips.handles.live]
            grips.request_set("towel", "edge", arm)
            grips.drain()
            (edge,) = [h.index for h in grips.handles.live if h.index != click]
            self.assertEqual(arm, grips.follow_arm[edge], "the edge rides the hand")
            self.assertEqual("", grips.follow_arm[click], "and the click stays on the mouse")
            grips.request_set("towel", "edge", arm, hold=False)
            grips.drain()
            self.assertEqual([click], [h.index for h in grips.handles.live], "only the request's grab is let go")
            self.assertEqual("", grips.follow_arm[edge], "and its slot is off the hand")
            self.assertEqual("", pins[edge].value, "on its dropdown too")
        finally:
            session.teardown()


class TestTheInspectorWaitsToBeAsked(unittest.TestCase):
    def test_nothing_is_selected_on_a_scene_with_no_goal(self) -> None:
        """A hand-assembled scene has no "first thing the goal is about", and the
        old fallback opened the page on whatever sorted first."""
        sim = _sim(cloth=False)
        session = _session(sim, 8965, panels="drive")
        try:
            self.assertIsNone(session.selected)
            self.assertIsNone(session.inspector._folder, "an Inspector heading before anybody clicked")
        finally:
            session.teardown()

    def test_clicking_something_brings_it_up(self) -> None:
        sim = _sim(cloth=False)
        session = _session(sim, 8966, panels="drive")
        try:
            subject = session.index.by_name("cube")
            self.assertIsNotNone(subject)
            session.select(subject)
            session.inspector.drain()
            self.assertIsNotNone(session.inspector._folder, "clicking built no Inspector")
            self.assertIn("cube", str(session.inspector._folder.label))
        finally:
            session.teardown()


class TestTheSeam(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = _sim(cloth=False)

    def test_each_step_runs_on_a_step_and_not_on_a_paused_pass(self) -> None:
        session = _session(self.sim, 8957, panels=("object",))
        try:
            calls = []
            session.each_step(lambda: calls.append(1))
            session.tick()
            self.assertEqual(1, len(calls), "a stepped tick did not run the callback")
            session.paused = True
            session.tick()
            self.assertEqual(1, len(calls), "a paused tick ran it anyway")
        finally:
            session.teardown()

    def test_the_viser_surface_is_visers_own(self) -> None:
        """So a component written against a bare ``ViserServer`` drops in."""
        session = _session(self.sim, 8958, panels=("object",))
        try:
            self.assertIs(session.gui, session.server.gui)
            self.assertIs(session.scene, session.server.scene)
            handle = session.gui.add_number("mine (mm)", 0.0, disabled=True)
            handle.value = 12.0
            self.assertEqual(12.0, handle.value)
        finally:
            session.teardown()


if __name__ == "__main__":
    unittest.main()
