# SPDX-License-Identifier: Apache-2.0
"""The fleet grid: its spacing, its labels, the viewer it owns, and the boxes it places."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from robosimstudio_studio.fleet_view import _place
from tests.fixtures.viser_stubs import _StubScene

TIER = "fast"
"""No GPU: a stand-in viewer records what the grid asks of it."""


class _Viewer:
    def __init__(self) -> None:
        self.server = SimpleNamespace(scene=_StubScene())
        self.model = SimpleNamespace(up_axis=2)
        self.world_offsets = np.zeros((2, 3), np.float32)
        self.closed = False
        self.calls: list[str] = []

    def set_world_offsets(self, grid) -> None:
        self.calls.append("set_world_offsets")
        self.world_offsets = np.array([[0.0, 0.0, 0.0], list(grid)], np.float32)

    def begin_frame(self, t) -> None:
        self.calls.append("begin_frame")

    def log_state(self, state) -> None:
        self.calls.append("log_state")

    def end_frame(self) -> None:
        self.calls.append("end_frame")

    def close(self) -> None:
        self.closed = True


def _view(owns: bool = False):
    from robosimstudio_studio.fleet_view import FleetView

    view = object.__new__(FleetView)
    view.viewer = _Viewer()
    view.owns_viewer = owns
    view.shown = [0, 1]
    view.interval = 0.0
    view._last = 0.0
    view._labels = {}
    view._captions = {}
    view.pending_spacing = None
    return view


_SLOTS = [SimpleNamespace(parked=True), SimpleNamespace(parked=True)]


_FLEET = SimpleNamespace(control_dt=0.01, pipeline=SimpleNamespace(state_0=None))


class FleetViewSpacing(unittest.TestCase):
    def test_respacing_moves_existing_labels(self) -> None:
        view = _view()
        view._update_labels(_SLOTS)
        handle = view._labels[1]
        self.assertEqual(handle.position, (0.0, 0.0, 1.4))
        view.set_spacing(3.0)
        view._update_labels(_SLOTS)
        self.assertIs(view._labels[1], handle)
        self.assertEqual(handle.position, (3.0, 3.0, 1.4))

    def test_pending_spacing_is_applied_by_the_draw(self) -> None:
        view = _view()
        view.pending_spacing = 2.5
        view.on_tick(_FLEET, _SLOTS, 1)
        self.assertIsNone(view.pending_spacing)
        calls = view.viewer.calls
        self.assertIn("set_world_offsets", calls)
        self.assertLess(calls.index("set_world_offsets"), calls.index("log_state"))
        np.testing.assert_allclose(view.viewer.world_offsets[1], [2.5, 2.5, 0.0])

    def test_slider_callback_only_assigns(self) -> None:
        from robosimstudio_studio import bench_session

        callbacks = []
        slider = SimpleNamespace(value=1.75, on_update=callbacks.append)
        gui = mock.MagicMock()
        gui.add_slider.return_value = slider
        view = _view()
        session = SimpleNamespace(
            server=SimpleNamespace(gui=gui),
            config=SimpleNamespace(dark_mode=False),
            fleet=SimpleNamespace(view=view, build_gui=lambda gui: None),
            lighting=mock.MagicMock(),
        )
        with mock.patch.object(bench_session, "PanelTask", mock.MagicMock()):
            bench_session.build_fleet_gui(session)
        self.assertEqual(len(callbacks), 1)
        callbacks[0](None)
        self.assertEqual(view.pending_spacing, 1.75)
        self.assertNotIn("set_world_offsets", view.viewer.calls)


class FleetViewClose(unittest.TestCase):
    def test_owned_viewer_is_closed(self) -> None:
        view = _view(owns=True)
        view.close()
        self.assertTrue(view.viewer.closed)

    def test_borrowed_viewer_stays_open(self) -> None:
        view = _view(owns=False)
        view.close()
        self.assertFalse(view.viewer.closed)


class TestFleetViewPlace(unittest.TestCase):
    def test_a_drifted_quaternion_does_not_scale_the_box(self) -> None:
        # A quarter turn about z, its quaternion grown about 2%: not the
        # identity, which an unnormalised matrix also gets right.
        pose = np.array([0, 0, 0, 0, 0, 0.7212, 0.7212])
        _, half = _place(pose, np.zeros(3), np.array([0.1, 0.2, 0.3]))
        np.testing.assert_allclose(half, [0.2, 0.1, 0.3], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
