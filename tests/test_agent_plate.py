# SPDX-License-Identifier: Apache-2.0
"""A fixed camera's background as the clip's own pixels: the clean plate, the composite over it, the camera fit.

No GPU and no page. The plate is cut from a clip drawn for the purpose (a
square sweeping over a still background), the composite is fed a segmentation
drawn by hand, and the camera fit is handed points a known camera projected.
What needs a scene -- the tiled pass that labels what moves, its shadow pass
-- is exercised on the page and documented in ``examples/live/REFERENCE.md``.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.core.render import plate
from robosimstudio_real2sim_agent import calib, frames

TIER = "fast"
"""No GPU: arrays stand in for renders, and a small mp4 for the clip."""


def _clip(path: Path, frames_n: int = 30, w: int = 96, h: int = 64) -> np.ndarray:
    """A grey-blue background with a red square sweeping across it once; returns the background."""
    import imageio.v2 as imageio

    background = np.zeros((h, w, 3), np.uint8)
    background[..., 0], background[..., 1], background[..., 2] = 60, 110, 170
    background[h // 2 :, :] = (200, 190, 170)  # a table under a wall
    writer = imageio.get_writer(str(path), fps=10, codec="libx264", quality=10, macro_block_size=16)
    for i in range(frames_n):
        frame = background.copy()
        x = int(i * (w - 16) / (frames_n - 1))
        frame[20:36, x : x + 16] = (220, 30, 30)
        writer.append_data(frame)
    writer.close()
    return background


class TestPlate(unittest.TestCase):
    def test_what_passes_through_is_gone(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="plate-"))
        background = _clip(tmp / "clip.mp4")
        reply = frames.plate(tmp / "clip.mp4", count=20, out_dir=tmp, name="plate.png")
        dotted = frames.plate(tmp / "clip.mp4", count=6, out_dir=tmp, name="plate_cam.v2")
        self.assertEqual(Path(dotted["plate"]).name, "plate_cam.v2.png", "only a trailing .png is dropped")
        self.assertTrue(reply["ok"], reply)
        self.assertEqual(Path(reply["plate"]).name, "plate.png", "a name given with .png keeps one")
        from PIL import Image

        got = np.asarray(Image.open(reply["plate"]).convert("RGB")).astype(int)
        self.assertLess(
            np.abs(got - background).mean(), 6.0, "the square swept past every pixel in under half the frames"
        )
        self.assertTrue(Path(reply["spread"]).exists())

    def test_windows_and_paint(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="plate-"))
        _clip(tmp / "clip.mp4")
        self.assertFalse(frames.plate(tmp / "clip.mp4", windows="2:1", out_dir=tmp)["ok"], "an empty window is refused")
        from PIL import Image

        mask = np.zeros((64, 96), np.uint8)
        mask[40:50, 40:50] = 255
        Image.fromarray(mask).save(tmp / "mask.png")
        reply = frames.plate(tmp / "clip.mp4", windows="0:1,2:2.9", count=12, out_dir=tmp, paint=tmp / "mask.png")
        self.assertTrue(reply["ok"], reply)
        self.assertIn("painted", reply["painted"])
        self.assertIn("0 to 1 s", reply["windows"])


class TestComposite(unittest.TestCase):
    def test_moving_pixels_are_the_render_and_the_rest_the_plate(self) -> None:
        rgb = np.full((6, 6, 3), 10, np.uint8)
        backdrop = np.full((6, 6, 3), 200, np.uint8)
        seg = np.zeros((6, 6), np.int32)  # a miss
        seg[0, :] = plate.LABEL_STATIC
        seg[1, :] = plate.LABEL_MOVING
        seg[2, :] = -2  # a particle
        seg[3, :] = -3  # a deformable's mesh
        out = plate.composite(rgb, seg, backdrop)
        self.assertTrue((out[0] == 200).all(), "a static row is the plate's")
        self.assertTrue((out[2, 1:5] == 10).all(), "moving, particle and mesh pixels inside are the render's")
        thin = np.zeros((6, 6), np.int32)
        thin[2, :] = -2  # a rope one pixel across is all edge, and keeps its own pixels
        self.assertTrue((plate.composite(rgb, thin, backdrop)[2] == 10).all())
        rgb, backdrop = np.full((9, 9, 3), 10, np.uint8), np.full((9, 9, 3), 200, np.uint8)
        seg = np.zeros((9, 9), np.int32)
        seg[2:7, 2:7] = plate.LABEL_MOVING
        out = plate.composite(rgb, seg, backdrop)
        self.assertEqual(int(out[4, 4, 0]), 10, "a broad thing's inside is the render's")
        self.assertGreater(int(out[2, 4, 0]), 10, "its outer ring fades into the plate")
        self.assertLess(int(out[2, 4, 0]), 200)
        self.assertEqual(int(out[0, 0, 0]), 200)

    def test_a_shadow_darkens_only_the_plate_and_leaves_no_halo(self) -> None:
        bare = np.full((20, 20, 3), 200, np.uint8)
        full = bare.copy()
        full[5:10, 5:15] = 100  # the moving thing's shadow on the table
        full[12:16, 5:15] = 30  # the moving thing itself, dark
        seg = np.zeros((20, 20), np.int32)
        seg[12:16, 5:15] = plate.LABEL_MOVING
        shadow = plate.shade(full, bare, seg)
        self.assertAlmostEqual(float(shadow[7, 10]), 101 / 201, places=3)
        self.assertEqual(float(shadow[14, 10]), 1.0, "the thing's own pixels are not a shadow")
        soft = plate.shade(full, bare, seg, soften=2)
        self.assertGreater(float(soft[17, 10]), 0.99, "no halo of the dark thing under the blur")

    def test_a_thin_thing_is_the_render_only_where_the_render_drew_it(self) -> None:
        seg = np.zeros((9, 9), np.int32)
        seg[2:7, 2:7] = -2  # a rope's collider spheres, wider than the rope drawn
        static_depth = np.full((9, 9), 1.0, np.float32)  # the bench, a metre away
        depth = static_depth.copy()
        depth[4, 2:7] = 0.996  # the rope drawn: 4 mm proud of the bench
        depth[0, 0] = 0.9993  # the path tracer's own mat, 0.7 mm proud, outside the colliders
        out = plate.nearer(seg, depth, static_depth, grow=1)
        moving = (out == plate.LABEL_MOVING) | (out < 0)
        self.assertEqual(np.argwhere(moving)[:, 0].tolist(), [4] * 5, "the rope's row alone")
        self.assertFalse(moving[0, 0], "the mat is the plate's")

    def test_a_rope_in_the_air_keeps_only_its_own_pixels(self) -> None:
        seg = np.zeros((9, 9), np.int32)
        seg[2:7, 2:7] = -2  # the rope's collider spheres, half a metre from the camera
        static_depth = np.zeros((9, 9), np.float32)  # nothing static behind it: the colliders hit nothing
        own = np.where(seg < 0, 0.5, 0.0).astype(np.float32)
        depth = np.full((9, 9), 3.0, np.float32)  # the render's own room, three metres off
        depth[4, 2:7] = 0.502  # the rope's tube
        out = plate.nearer(seg, depth, static_depth, own, grow=1)
        moving = (out == plate.LABEL_MOVING) | (out < 0)
        self.assertEqual(np.argwhere(moving)[:, 0].tolist(), [4] * 5, "the room behind the spheres is the plate's")

    def test_a_moving_thing_keeps_its_recesses_and_what_stands_proud_of_it(self) -> None:
        # a lockbox face (1 m), a key standing 5 cm proud of it, a seam 3 cm deep across it, nothing static behind
        seg = np.full((12, 12), -2, np.int32)
        own = np.full((12, 12), 1.0, np.float32)
        own[4:8, 5:7] = 0.95  # the key's collider
        depth = own.copy()
        depth[10, :] = 1.03  # the seam: the render looks into it, past the colliders' face
        out = plate.nearer(seg, depth, np.zeros((12, 12), np.float32), own, grow=4)
        self.assertTrue(
            ((out == plate.LABEL_MOVING) | (out < 0)).all(), "the face round the key and the seam are drawn"
        )

    def test_a_bulging_collider_over_another_moving_thing_shows_that_thing(self) -> None:
        # a gripper (0.7 m) in front of the box face (1 m); its collider is a pixel wider than its drawn mesh
        seg = np.full((9, 9), -2, np.int32)
        own = np.full((9, 9), 1.0, np.float32)
        own[2:7, 2:7] = 0.7
        depth = own.copy()
        depth[2, 2:7] = 1.0  # the drawn gripper stops a pixel short: the render shows the box there
        out = plate.nearer(seg, depth, np.zeros((9, 9), np.float32), own, grow=2)
        self.assertTrue(((out == plate.LABEL_MOVING) | (out < 0)).all(), "the box behind the bulge is the render's")

    def test_a_plate_takes_any_size_asked(self) -> None:
        from PIL import Image

        tmp = Path(tempfile.mkdtemp(prefix="plate-"))
        Image.fromarray(np.zeros((30, 40, 3), np.uint8)).save(tmp / "p.png")
        self.assertEqual(plate.load_plate(tmp / "p.png", 80, 60).shape, (60, 80, 3))
        self.assertEqual(plate.load_plate(tmp / "p.png", 64, 64).shape, (64, 64, 3), "a camera scaled to a square")
        with self.assertRaises(FileNotFoundError):
            plate.load_plate(tmp / "nothing.png", 80, 60)


class TestStatic(unittest.TestCase):
    def test_only_what_is_held_by_fixed_joints_is_static(self) -> None:
        import newton

        b = newton.ModelBuilder()
        fixed = b.add_link()
        b.add_shape_box(fixed, hx=0.1, hy=0.1, hz=0.1)
        j0 = b.add_joint_fixed(-1, fixed)
        hinged = b.add_link()
        b.add_shape_box(hinged, hx=0.1, hy=0.1, hz=0.1)
        j1 = b.add_joint_revolute(fixed, hinged, axis=(0, 0, 1))
        b.add_articulation([j0, j1])
        free = b.add_body()
        b.add_shape_sphere(free, radius=0.05)
        b.add_shape_box(-1, hx=1.0, hy=1.0, hz=0.01)  # on the world
        model = b.finalize(device="cpu")
        self.assertEqual(plate.static_bodies(model).tolist(), [True, False, False])
        self.assertEqual(plate.static_shapes(model).tolist(), [True, False, False, True])


class TestCameraFit(unittest.TestCase):
    CAM = CameraSpec(name="c", width=1280, height=720, fov_y=48, eye=(1.9, -0.7, 1.6), look_at=(0.6, 0.1, 0.75))

    def _points(self, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(3)
        world = np.c_[rng.uniform(0.2, 1.0, n), rng.uniform(-0.6, 0.6, n), rng.uniform(0.75, 1.1, n)]
        return world, self.CAM.project(world)[0]

    def test_the_camera_comes_back(self) -> None:
        world, pixels = self._points()
        fit = calib.fit_camera(world, pixels, 1280, 720)
        self.assertLess(fit["rms_px"], 0.05)
        self.assertAlmostEqual(fit["fov_y"], 48.0, places=1)
        spec = CameraSpec(name="fit", **fit["spec"])
        self.assertLess(np.abs(spec.project(world)[0] - pixels).max(), 0.5, "the printed keywords make that camera")

    def test_a_mirrored_set_is_refused(self) -> None:
        world, pixels = self._points()
        with self.assertRaisesRegex(ValueError, "mirror"):
            calib.fit_camera(world * np.array([1.0, -1.0, 1.0]), pixels, 1280, 720)

    def test_a_point_read_off_wrong_stands_out(self) -> None:
        world, pixels = self._points(12)
        pixels = pixels.copy()
        pixels[4] += (40.0, -25.0)
        errors = calib.fit_camera(world, pixels, 1280, 720)["errors_px"]
        self.assertEqual(int(np.argmax(errors)), 4)

    def test_a_point_behind_the_origin_parses_on_the_command_line(self) -> None:
        from robosimstudio_real2sim_agent import ctl

        argv = ctl._join_points(["camera", "fit", "--point", "-0.3,0.2,0.75=812,455", "--size", "640x480"])
        args = ctl.build_parser().parse_args(argv)
        self.assertEqual(args.point, ["-0.3,0.2,0.75=812,455"])
        kept = ctl._join_points(["camera", "fit", "--point", "--ref", "still.png"])
        self.assertEqual(kept[2:], ["--point", "--ref", "still.png"], "an option after --point is left an option")

    def test_parse_point(self) -> None:
        self.assertEqual(calib.parse_point("0.6,0.3,0.75=812,455"), ((0.6, 0.3, 0.75), (812.0, 455.0)))
        with self.assertRaises(ValueError):
            calib.parse_point("0.6,0.3=812,455")


if __name__ == "__main__":
    unittest.main()
