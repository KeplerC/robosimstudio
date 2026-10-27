# SPDX-License-Identifier: Apache-2.0
"""The Cameras panel: which pixels it shows, which camera it draws, where it aims one, and what it gives back.

The panel's choices are bookkeeping around a renderer, and every one of them
has a claim a test can hold without rendering anything: the environment's own
frames are the default source when it has a rig, a paused scene is encoded
once, only the camera being rendered gets a frustum, ``Set from view`` takes
the browser's eye and look-at as the camera's pose and ``Reset pose`` puts the
declared one back, and closing gives the renderer, the images and the frustums
back. The cameras are real :class:`~robosimstudio.core.cameras.CameraSpec`\\ s;
the renderer, the env and the browser are stand-ins that record what they
were asked.
"""

from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.machine import Platform
from robosimstudio_studio.cameras import FRUSTUM_COLOR, FRUSTUM_COLOR_SELECTED, FRUSTUM_ROOT, SOURCE_OBS, PanelCamera
from tests.fixtures.studio_fakes import FakeSession

TIER = "fast"
"""No GPU: the renderer is a stand-in and the frames are arrays."""

FRONT = CameraSpec(name="front", width=64, height=48, fov_y=50.0, eye=(1.0, 0.0, 1.0), look_at=(0.0, 0.0, 0.0))
WRIST = CameraSpec(name="wrist", width=32, height=32, fov_y=70.0, eye=(0.0, 1.0, 0.5), look_at=(0.0, 0.0, 0.0))


class _Renderer:
    """A renderer that hands back one flat frame per camera and remembers being closed."""

    def __init__(self, cameras, channels) -> None:
        self.cameras, self.channels = list(cameras), tuple(channels)
        self.closed = False

    def render(self, state):
        return {c.name: {"rgb": np.full((c.height, c.width, 3), 7, np.uint8)} for c in self.cameras}

    def close(self) -> None:
        self.closed = True


class _Env:
    """An env with two cameras, an observation rig (or none), and a renderer factory that records."""

    def __init__(self, *, rig: bool = True, fails: str = "") -> None:
        self.cameras = (FRONT, WRIST)
        self.renderer = object() if rig else None
        self.renderer_name = "tiled"
        self.render_every = 4
        self.frames: dict = {}
        self.step_count = 0
        self.model = object()
        self.state = SimpleNamespace(body_q=None)
        self.platform = Platform(device="cpu", device_why="test", renderer="tiled", renderer_why="test")
        self.fails = fails
        self.made: list[_Renderer] = []

    def make_renderer(self, backend, *, cameras, channels, **_):
        if self.fails:
            raise RuntimeError(self.fails)
        self.made.append(_Renderer(cameras, channels))
        return self.made[-1]


def _panel(**env) -> tuple[PanelCamera, FakeSession]:
    session = FakeSession(_Env(**env))
    panel = PanelCamera(session)
    panel.build(session.server.gui)
    return panel, session


def _frustums(session) -> dict:
    return {n.name.rsplit("/", 1)[-1]: n for n in session.server.scene.nodes if n.name.startswith(FRUSTUM_ROOT)}


def _browser(eye=(2.0, -1.0, 1.5), look_at=(0.1, 0.2, 0.3)) -> SimpleNamespace:
    return SimpleNamespace(camera=SimpleNamespace(position=eye, look_at=look_at))


def _press(gui, label: str, client) -> None:
    """A button pressed in a browser: the event names the client that pressed it."""
    for fn in gui.find(label).callbacks:
        fn(SimpleNamespace(client=client))


class TestWhatItLooksThrough(unittest.TestCase):
    def test_the_environments_own_frames_are_the_default_when_it_has_a_rig(self) -> None:
        panel, _ = _panel()
        self.assertEqual((SOURCE_OBS, "tiled"), panel.backends)
        self.assertEqual(SOURCE_OBS, panel.backend)

    def test_without_a_rig_it_renders_for_itself_whatever_was_asked(self) -> None:
        session = FakeSession(_Env(rig=False))
        panel = PanelCamera(session, backend=SOURCE_OBS)
        self.assertEqual(("tiled",), panel.backends)
        self.assertEqual("tiled", panel.backend)

    def test_the_resolution_is_greyed_out_on_the_rigs_own_frames(self) -> None:
        _panel_obs, session = _panel()
        gui = session.server.gui
        self.assertTrue(gui.find("Width").disabled and gui.find("Height").disabled)
        _panel_own, session = _panel(rig=False)
        gui = session.server.gui
        self.assertFalse(gui.find("Width").disabled or gui.find("Height").disabled)

    def test_it_asks_for_the_rig_only_while_it_shows_it(self) -> None:
        session = FakeSession(_Env())
        panel = PanelCamera(session)
        self.assertFalse(panel.wants_obs, "nothing is displayed before the page is built")
        panel.build(session.server.gui)
        self.assertTrue(panel.wants_obs)
        session.server.gui.find("Live render").fire(False)
        self.assertFalse(panel.wants_obs)


class TestTheRigsOwnFrames(unittest.TestCase):
    def test_the_selected_cameras_selected_channels_are_shown_and_nothing_is_built(self) -> None:
        panel, session = _panel()
        rgb = np.zeros((48, 64, 3), np.uint8)
        session.env.frames = {"front": {"rgb": rgb, "depth": np.ones((48, 64))}, "wrist": {"rgb": rgb}}
        panel.drain()
        self.assertEqual(["front rgb"], [image.label for image in session.server.gui.images])
        self.assertEqual([], session.env.made, "the rig's frames are already paid for")

    def test_a_paused_scene_is_encoded_once(self) -> None:
        panel, session = _panel()
        first = np.zeros((48, 64, 3), np.uint8)
        session.env.frames = {"front": {"rgb": first}}
        panel.drain()
        (image,) = session.server.gui.images
        session.env.frames = {"front": {"rgb": np.ones((48, 64, 3), np.uint8)}}
        panel.budget.reset()
        panel.drain()
        self.assertIs(first, image.image, "the same state was encoded again")
        session.env.step_count += session.env.render_every  # the rig has rendered again
        panel.drain()
        self.assertIsNot(first, image.image)


class TestTheFrustums(unittest.TestCase):
    def test_only_the_camera_being_rendered_is_drawn(self) -> None:
        panel, session = _panel()
        panel.drain()
        self.assertEqual(["front"], list(_frustums(session)))
        self.assertEqual(FRUSTUM_COLOR_SELECTED, _frustums(session)["front"].color)
        session.server.gui.find("Camera").fire("wrist")
        panel.drain()
        drawn = _frustums(session)
        self.assertFalse(drawn["front"].visible, "switching moves the frustum rather than adding one")
        self.assertTrue(drawn["wrist"].visible)
        session.server.gui.find("Camera").fire("all")
        panel.drain()
        self.assertTrue(all(node.visible for node in _frustums(session).values()))

    def test_unticking_the_gizmo_takes_them_down_on_that_drain(self) -> None:
        panel, session = _panel()
        panel.drain()
        (frustum,) = _frustums(session).values()
        session.server.gui.find("Camera gizmo").fire(False)
        panel.drain()
        self.assertTrue(frustum.removed)
        self.assertEqual({}, panel._frusta)

    def test_a_click_on_one_picks_the_camera_the_pose_buttons_act_on(self) -> None:
        panel, session = _panel()
        session.server.gui.find("Camera").fire("all")
        panel.drain()
        self.assertEqual("front", panel._selected_name(), "on all, the first camera until one is clicked")
        browser = _browser()
        for fn in _frustums(session)["wrist"].clicked:
            fn(SimpleNamespace(client=browser))
        panel.drain()
        self.assertEqual("wrist", panel._selected_name())
        self.assertEqual(WRIST.eye, browser.camera.position, "and the browser flies to it")
        self.assertEqual(FRUSTUM_COLOR_SELECTED, _frustums(session)["wrist"].color)
        self.assertEqual(FRUSTUM_COLOR, _frustums(session)["front"].color)


class TestAimingByLooking(unittest.TestCase):
    def test_set_from_view_takes_the_browsers_eye_and_look_at(self) -> None:
        panel, session = _panel()
        _press(session.server.gui, "Set from view", _browser())
        panel.drain()
        (aimed,) = [c for c in panel.specs() if c.name == "front"]
        self.assertEqual(((2.0, -1.0, 1.5), (0.1, 0.2, 0.3)), (aimed.eye, aimed.look_at))
        self.assertEqual(WRIST, panel.specs()[1], "the other camera is as declared")
        self.assertTrue(session.server.gui.find("Pose").value.endswith("edited"))
        self.assertEqual("front aimed from the browser view", session.status)

    def test_reset_pose_puts_the_declared_camera_back(self) -> None:
        panel, session = _panel()
        _press(session.server.gui, "Set from view", _browser())
        panel.drain()
        session.server.gui.find("Reset pose").fire()
        panel.drain()
        self.assertEqual(list(session.env.cameras), panel.specs())
        self.assertTrue(session.server.gui.find("Pose").value.endswith("declared"))

    def test_look_through_flies_the_browser_to_the_camera(self) -> None:
        panel, session = _panel()
        browser = _browser()
        _press(session.server.gui, "Look through", browser)
        panel.drain()
        self.assertEqual((FRONT.eye, FRONT.look_at), (browser.camera.position, browser.camera.look_at))

    def test_a_press_with_no_browser_behind_it_changes_nothing(self) -> None:
        panel, session = _panel()
        _press(session.server.gui, "Set from view", None)
        panel.drain()
        self.assertEqual({}, panel.poses)


class TestItsOwnRenderer(unittest.TestCase):
    def test_close_gives_back_the_renderer_the_images_and_the_frustums(self) -> None:
        panel, session = _panel(rig=False)
        panel.drain()
        (renderer,) = session.env.made
        self.assertEqual(["front"], [c.name for c in renderer.cameras])
        images, frustums = list(session.server.gui.images), list(_frustums(session).values())
        self.assertTrue(images and frustums)
        panel.close()
        self.assertTrue(renderer.closed)
        self.assertTrue(all(i.removed for i in images) and all(f.removed for f in frustums))

    def test_a_renderer_that_will_not_build_turns_the_view_off_saying_why(self) -> None:
        panel, session = _panel(rig=False, fails="no EGL display")
        with contextlib.redirect_stdout(io.StringIO()):
            panel.drain()
        self.assertFalse(session.server.gui.find("Live render").value)
        self.assertEqual("tiled would not build -- RuntimeError: no EGL display", session.status)

    def test_a_reset_rebuilds_only_for_a_new_model(self) -> None:
        panel, session = _panel(rig=False)
        panel.drain()
        (renderer,) = session.env.made
        panel.on_reset({}, {})
        self.assertFalse(renderer.closed, "the same model keeps its renderer")
        session.env.model = object()
        panel.on_reset({}, {})
        self.assertTrue(renderer.closed)
        self.assertEqual({}, panel._frusta, "a frustum holding the old model's pose is a wrong picture")

    def test_an_aimed_camera_is_a_new_renderer(self) -> None:
        panel, session = _panel(rig=False)
        panel.drain()
        _press(session.server.gui, "Set from view", _browser())
        panel.budget.reset()
        panel.drain()
        first, second = session.env.made
        self.assertTrue(first.closed)
        self.assertEqual((2.0, -1.0, 1.5), second.cameras[0].eye)


if __name__ == "__main__":
    unittest.main()
