# SPDX-License-Identifier: Apache-2.0
"""The fleet grid: its spacing, its labels, the viewer it owns, and the boxes it measures a world by.

The grid's pitch comes from what one world's *shapes* occupy -- each grown to
its own half-extent, carried through its body, unioned -- and every box here is
one whose answer can be written down: a sphere is its radius each way, a
capsule's caps add to its half-height, a quarter turn swaps two half-extents,
a mesh off its own origin is measured where its vertices are.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import newton
import numpy as np

from robosimstudio_studio.fleet_view import FleetView, _place
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


GEO = newton.GeoType
QUARTER_Z = (0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))
"""A quarter turn about z, xyzw."""


def _shape(kind, scale=(0.0, 0.0, 0.0), *, at=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 0.0, 1.0), body=-1, world=0, **kw):
    """One shape row of a model: what ``_shape_boxes`` reads, with the radius and source a mesh falls back to."""
    return {
        "type": int(kind),
        "scale": tuple(scale),
        "xform": (*at, *quat),
        "body": body,
        "world": world,
        "radius": kw.get("radius", 0.0),
        "source": kw.get("source"),
    }


def _measured(*shapes, body_q=((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),), up_axis: int = 2) -> FleetView:
    """A grid over a fleet whose model holds ``shapes`` and whose world 0 has its bodies at ``body_q``."""
    rows = list(shapes)
    column = lambda key, dtype=np.float64: SimpleNamespace(  # noqa: E731
        numpy=lambda: np.array([r[key] for r in rows], dtype=dtype)
    )
    model = SimpleNamespace(
        shape_type=column("type", np.int32),
        shape_scale=column("scale"),
        shape_transform=column("xform"),
        shape_body=column("body", np.int32),
        shape_collision_radius=column("radius"),
        shape_world=column("world", np.int32),
        shape_source=[r["source"] for r in rows],
        shape_count=len(rows),
    )
    snapshot = SimpleNamespace(body=lambda world: np.asarray(body_q, dtype=np.float64))
    view = object.__new__(FleetView)
    view.fleet = SimpleNamespace(model=model, snapshot=lambda: snapshot)
    view.viewer = SimpleNamespace(model=SimpleNamespace(up_axis=up_axis))
    view._mesh_bounds = {}
    return view


class TestOneShapesOwnBox(unittest.TestCase):
    def _box(self, kind, scale, radius: float = 0.0, source=None):
        view = _measured(_shape(kind, scale, radius=radius, source=source))
        centre, half = view._local_box(view.fleet.model, 0, int(kind), scale, radius)
        return np.asarray(centre), np.asarray(half)

    def test_a_sphere_is_its_radius_every_way(self) -> None:
        centre, half = self._box(GEO.SPHERE, (0.05, 0.0, 0.0))
        np.testing.assert_allclose(centre, 0.0)
        np.testing.assert_allclose(half, [0.05, 0.05, 0.05])

    def test_a_capsules_caps_add_to_its_half_height_and_a_cylinders_do_not(self) -> None:
        _, capsule = self._box(GEO.CAPSULE, (0.02, 0.1, 0.0))
        _, cylinder = self._box(GEO.CYLINDER, (0.02, 0.1, 0.0))
        np.testing.assert_allclose(capsule, [0.02, 0.02, 0.12])
        np.testing.assert_allclose(cylinder, [0.02, 0.02, 0.1])

    def test_a_box_is_its_half_extents_whatever_their_sign(self) -> None:
        _, half = self._box(GEO.BOX, (0.1, -0.2, 0.3))
        np.testing.assert_allclose(half, [0.1, 0.2, 0.3])

    def test_a_mesh_off_its_own_origin_is_measured_where_its_vertices_are(self) -> None:
        panel = SimpleNamespace(vertices=[[0.0, 0.0, 0.0], [2.0, 1.0, 0.5], [1.0, 0.5, 0.0]])
        centre, half = self._box(GEO.MESH, (2.0, 1.0, 1.0), source=panel)
        np.testing.assert_allclose(centre, [2.0, 0.5, 0.25])
        np.testing.assert_allclose(half, [2.0, 0.5, 0.25])

    def test_a_source_with_no_vertices_falls_back_to_the_broad_phase_sphere(self) -> None:
        centre, half = self._box(GEO.GAUSSIAN, (1.0, 1.0, 1.0), radius=0.3, source=SimpleNamespace())
        np.testing.assert_allclose(centre, 0.0)
        np.testing.assert_allclose(half, [0.3, 0.3, 0.3])


class TestAMeshesBounds(unittest.TestCase):
    def test_they_are_read_once_per_mesh(self) -> None:
        view = _measured()
        mesh = SimpleNamespace(vertices=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))
        first = view._mesh_box(mesh)
        mesh.vertices = np.zeros((2, 3))  # a hull set walks one source many times; it is not walked again
        np.testing.assert_allclose(view._mesh_box(mesh), first)
        np.testing.assert_allclose(first, [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])

    def test_a_mesh_with_no_vertices_has_no_bounds(self) -> None:
        view = _measured()
        self.assertIsNone(view._mesh_box(SimpleNamespace(vertices=np.zeros((0, 3)))))
        self.assertIsNone(view._mesh_box(SimpleNamespace()))


class TestWhatAWorldOccupies(unittest.TestCase):
    def test_a_static_box_and_a_turned_box_on_a_body_union(self) -> None:
        view = _measured(
            _shape(GEO.BOX, (0.5, 0.5, 0.5), at=(1.0, 0.0, 0.5)),  # the bench, welded to the world
            _shape(GEO.BOX, (0.2, 0.4, 0.1), body=0),  # a tray, a quarter turn round on its body
            body_q=((-1.0, 0.0, 0.1, *QUARTER_Z),),
        )
        # bench: x 0.5..1.5, y -0.5..0.5, z 0..1; tray turned: x -1.4..-0.6, y -0.2..0.2, z 0..0.2
        np.testing.assert_allclose(view._occupied_extent(), [2.9, 1.0, 1.0], atol=1e-12)

    def test_the_ground_and_the_other_worlds_are_not_part_of_it(self) -> None:
        view = _measured(
            _shape(GEO.BOX, (0.1, 0.1, 0.1)),
            _shape(GEO.PLANE, (10.0, 10.0, 0.0)),  # the 20-metre slab this exists to avoid
            _shape(GEO.BOX, (5.0, 5.0, 5.0), world=1),  # the same cell, replicated
        )
        np.testing.assert_allclose(view._occupied_extent(), [0.2, 0.2, 0.2])

    def test_a_world_of_nothing_but_ground_has_no_extent_and_no_pitch(self) -> None:
        view = _measured(_shape(GEO.PLANE, (10.0, 10.0, 0.0)))
        self.assertIsNone(view._occupied_extent())
        self.assertIsNone(view._tight_spacing(1.15), "the viewer's own choice is kept")


class TestThePitch(unittest.TestCase):
    def test_it_is_the_extent_times_the_margin_and_flat_along_up(self) -> None:
        view = _measured(_shape(GEO.BOX, (0.5, 0.75, 0.4)))
        self.assertEqual((1.0 * 1.15, 1.5 * 1.15, 0.0), view._tight_spacing(1.15))

    def test_a_cell_smaller_than_half_a_metre_is_spaced_as_half_a_metre(self) -> None:
        view = _measured(_shape(GEO.SPHERE, (0.05, 0.0, 0.0)), up_axis=1)
        self.assertEqual((0.5 * 2.0, 0.0, 0.5 * 2.0), view._tight_spacing(2.0))


if __name__ == "__main__":
    unittest.main()
