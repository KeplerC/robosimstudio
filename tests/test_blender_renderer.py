# SPDX-License-Identifier: Apache-2.0
"""The blender renderer, held without Blender: which Blender it runs, how it talks to it, and what it sends.

Blender is a separate process (``bpy`` has no wheel for this package's
Python), so most of what can go wrong is on this side of the socket: finding
the executable, the command line, the framing, a camera's intrinsics as
Blender's lens and shift, and the scene and lights put into the message.
Rendering through a real Blender is ``test_blender_render.py``.
"""

from __future__ import annotations

import os
import socket
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

TIER = "fast"


class TestWhichBlender(unittest.TestCase):
    def test_a_name_in_code_wins_then_the_variable_then_the_path(self) -> None:
        from robosimstudio.solvers.render import blender as B

        with mock.patch.dict(os.environ, {B.ENV: sys.executable}):
            self.assertEqual(B.find_blender(), sys.executable)
            self.assertEqual(B.find_blender(sys.executable), sys.executable)
        with (
            mock.patch.dict(os.environ, {B.ENV: ""}),
            mock.patch.object(B.shutil, "which", return_value="/opt/b/blender"),
        ):
            self.assertEqual(B.find_blender(), "/opt/b/blender")

    def test_a_named_blender_that_is_not_there_is_refused_by_name(self) -> None:
        from robosimstudio.solvers.render import blender as B

        with mock.patch.dict(os.environ, {B.ENV: "/nowhere/blender"}):
            with self.assertRaisesRegex(FileNotFoundError, "/nowhere/blender.*ROBOSIMSTUDIO_BLENDER"):
                B.find_blender()

    def test_none_anywhere_says_how_to_get_one(self) -> None:
        from robosimstudio.solvers.render import blender as B

        with (
            mock.patch.dict(os.environ, {B.ENV: ""}),
            mock.patch.object(B.shutil, "which", return_value=None),
            mock.patch.object(B.sys, "platform", "linux"),
        ):
            with self.assertRaisesRegex(FileNotFoundError, "blender.org"):
                B.find_blender()

    def test_the_mac_app_is_found_by_itself(self) -> None:
        from robosimstudio.solvers.render import blender as B

        real = os.path.isfile
        with (
            mock.patch.dict(os.environ, {B.ENV: ""}),
            mock.patch.object(B.shutil, "which", return_value=None),
            mock.patch.object(B.sys, "platform", "darwin"),
            mock.patch.object(B.os.path, "isfile", lambda p: p == B.MAC_APP or real(p)),
        ):
            self.assertEqual(B.find_blender(), B.MAC_APP)

    def test_blender_runs_the_worker_in_the_background_and_a_python_runs_it_as_a_script(self) -> None:
        from robosimstudio.solvers.render.blender import blender_argv

        argv = blender_argv("/opt/blender/blender", 4242)
        self.assertEqual(argv[:3], ["/opt/blender/blender", "-b", "--factory-startup"])
        self.assertEqual(argv[-2:], ["--", "4242"])
        worker = Path(argv[argv.index("--python") + 1])
        self.assertTrue(worker.is_file() and worker.name == "blender_worker.py")
        self.assertEqual(blender_argv("/venv313/bin/python3.13", 7)[1:], [str(worker), "7"])


class TestTheWire(unittest.TestCase):
    def test_a_header_and_its_blobs_arrive_whole(self) -> None:
        from robosimstudio.solvers.render.blender_worker import receive, send

        a, b = socket.socketpair()
        try:
            image = np.arange(3 * 64 * 48, dtype=np.uint8).reshape(48, 64, 3)
            depth = np.linspace(0.0, 2.0, 64 * 48, dtype=np.float32)
            send(a, {"op": "frame", "cameras": [{"rgb": 0, "depth": 1}]}, [image, depth.tobytes()])
            header, blobs = receive(b)
        finally:
            a.close()
            b.close()
        self.assertEqual(header["op"], "frame")
        self.assertEqual(header["nblobs"], 2)
        np.testing.assert_array_equal(np.frombuffer(blobs[0], dtype=np.uint8).reshape(48, 64, 3), image)
        np.testing.assert_array_equal(np.frombuffer(blobs[1], dtype=np.float32), depth)

    def test_a_closed_connection_is_an_error_not_a_hang(self) -> None:
        from robosimstudio.solvers.render.blender_worker import receive

        a, b = socket.socketpair()
        a.close()
        try:
            with self.assertRaises(ConnectionError):
                receive(b)
        finally:
            b.close()


class TestTheCamera(unittest.TestCase):
    """``K`` as a lens on a horizontal sensor and a shift, which ``test_blender_render`` holds against ``tiled``."""

    def test_a_centred_camera_has_no_shift(self) -> None:
        from robosimstudio.solvers.render.blender_worker import camera_settings

        got = camera_settings(640, 360, 500.0, 500.0, 319.5, 179.5)
        self.assertAlmostEqual(got["lens"], 500.0 * 36.0 / 640.0)
        self.assertEqual(got["shift"], (0.0, 0.0))
        self.assertEqual(got["pixel_aspect"], (1.0, 1.0))

    def test_the_principal_point_moves_the_frame_the_other_way(self) -> None:
        """A principal point right of centre is a frame shifted left; one below centre, a frame shifted up."""
        from robosimstudio.solvers.render.blender_worker import camera_settings

        sx, sy = camera_settings(640, 360, 500.0, 500.0, 319.5 + 64.0, 179.5 + 32.0)["shift"]
        self.assertAlmostEqual(sx, -64.0 / 640.0)
        self.assertAlmostEqual(sy, 32.0 / 640.0, msg="in units of the larger side")
        sx, _ = camera_settings(360, 640, 500.0, 500.0, 179.5 + 64.0, 319.5)["shift"]
        self.assertAlmostEqual(sx, -64.0 / 640.0, msg="a portrait frame's larger side is its height")

    def test_unequal_focal_lengths_are_a_pixel_aspect(self) -> None:
        from robosimstudio.solvers.render.blender_worker import camera_settings

        self.assertEqual(camera_settings(320, 180, 210.0, 220.5, 159.5, 89.5)["pixel_aspect"], (220.5 / 210.0, 1.0))
        self.assertEqual(camera_settings(320, 180, 220.5, 210.0, 159.5, 89.5)["pixel_aspect"], (1.0, 220.5 / 210.0))


class TestWhatIsSent(unittest.TestCase):
    def test_the_lab_room_is_the_path_tracers_panel_and_dome(self) -> None:
        from robosimstudio.solvers.render import blender as B

        cell = types.SimpleNamespace(center=(0.6, 0.0, 0.0))
        (panel,) = B._lights("lab", (0, 0, -1), 2.0, (), True, cell, 0.75)
        self.assertEqual(panel["type"], "area")
        self.assertEqual(panel["position"], [0.6, 0.0, 0.75 + 1.7])
        self.assertAlmostEqual(panel["energy"], 2.0 * B.LAB_PANEL_WATTS)
        world = B._world("lab", (1.0, 1.2, 1.4), None)
        self.assertAlmostEqual(world["strength"], 1.2 * B.LAB_DOME_STRENGTH)
        self.assertNotIn("background", world)

    def test_elsewhere_it_is_the_tiled_tracers_rig(self) -> None:
        from robosimstudio.solvers.render import blender as B

        rig = B._lights("", (-1, 1, -1), 1.0, (((1, 0, -0.2), 0.5),), False, None, 0.0)
        self.assertEqual([r["type"] for r in rig], ["sun", "sun"])
        self.assertEqual([r["shadow"] for r in rig], [False, False], "the key as asked; a fill never")
        self.assertAlmostEqual(rig[1]["energy"], 0.5 * B.SUN_PER_KEY)
        world = B._world("", (0.2, 0.4, 0.6), (1.0, 1.0, 1.0))
        np.testing.assert_allclose(world["color"], [0.5, 1.0, 1.5])
        self.assertAlmostEqual(world["strength"], 0.4 * B.AMBIENT_STRENGTH)
        self.assertEqual(world["background"], [1.0, 1.0, 1.0])

    def test_the_palettes_display_colours_are_sent_as_linear_albedo(self) -> None:
        from robosimstudio.solvers.render.blender import _linear

        np.testing.assert_allclose(_linear((1.0, 0.5, 0.0)), [1.0, 0.21404, 0.0], atol=1e-5)

    def test_every_visible_shape_is_sent_once_hung_off_its_body(self) -> None:
        import newton

        from robosimstudio.solvers.render.blender import _describe

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0)))
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
        model = builder.finalize(device="cpu")
        header, blobs, bodies, drawables = _describe(
            model, finishes=None, deformables=None, deformable_colors=None, deformable_finishes=None
        )
        self.assertEqual(bodies, [body])
        self.assertEqual(sorted(m["body"] for m in header["meshes"]), [-1, body])
        box = next(m for m in header["meshes"] if m["body"] == body)
        verts = np.frombuffer(blobs[box["verts"]], dtype=np.float32).reshape(-1, 3)
        np.testing.assert_allclose(np.abs(verts).max(axis=0), 0.05, atol=1e-6)
        faces = np.frombuffer(blobs[box["faces"]], dtype=np.int32).reshape(-1, 3)
        self.assertLess(int(faces.max()), len(verts))
        self.assertFalse(drawables)


class TestTheRegistry(unittest.TestCase):
    def test_it_is_nameable_and_never_picked(self) -> None:
        from robosimstudio.core.render.spec import RenderSpec
        from robosimstudio.solvers.render import backend_class, choose_renderer, list_renderers, look_kwargs

        self.assertEqual(backend_class("blender").__name__, "RendererBlender")
        self.assertNotIn("blender", list_renderers())
        with mock.patch.dict(os.environ, {"ROBOSIMSTUDIO_RENDERER": "blender"}):
            self.assertEqual(choose_renderer("", "cpu")[0], "blender", "a CPU scene may name it")
        self.assertEqual(look_kwargs("blender", RenderSpec()), {})
        got = look_kwargs("blender", RenderSpec(environment="lab", key_intensity=1.3, background=(0.1, 0.1, 0.1)))
        self.assertEqual(got, {"environment": "lab", "key_intensity": 1.3, "background": (0.1, 0.1, 0.1)})


if __name__ == "__main__":
    unittest.main()
