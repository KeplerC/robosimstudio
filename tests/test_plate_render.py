# SPDX-License-Identifier: Apache-2.0
"""A camera with a clean plate, rendered through ``Sim.render``: the plate behind, what moves in front.

A free box over a bench, a robot whose links move, and a flat blue plate: the
bench and everything past it come back the plate's blue, the box comes back
drawn, and the same camera without its plate draws the bench. Whatever
renderer the machine picks draws the frame; the composite's segmentation and
shadow passes are the tiled tracer's either way.
"""

from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

import numpy as np

TIER = "medium"
"""Builds and renders a scene on the GPU."""


class TestPlateRender(unittest.TestCase):
    def test_the_plate_behind_and_the_box_in_front(self) -> None:
        from PIL import Image

        import robosimstudio as rss

        tmp = Path(tempfile.mkdtemp(prefix="plate-render-"))
        Image.new("RGB", (320, 180), (20, 60, 230)).save(tmp / "plate.png")
        scene = rss.Scene(cell="bench_075", name="plate_render")
        scene.add(rss.robot("yam_bimanual"))
        scene.add(rss.box(name="cube", half=(0.05,) * 3, at=(0.0, 0.0, 0.05), color=(0.9, 0.15, 0.1)))
        cam = rss.CameraSpec(
            name="front",
            width=320,
            height=180,
            fov_y=40.0,
            eye=(1.5, 0.0, 1.1),
            look_at=(0.6, 0.0, 0.78),
            background=str(tmp / "plate.png"),
        )
        scene.add(cam)
        sim = scene.build(control_mode="joint_position")
        sim.run(10)
        rgb = sim.render("front")["rgb"].astype(int)
        u, v = cam.project(np.array([[0.6, 0.0, 0.8]]))[0][0]  # the cube's centre, in the world
        self.assertGreater(rgb[int(v), int(u), 0], 120, "the cube is drawn over the plate")
        corner = rgb[5, 5]
        self.assertLess(abs(corner[2] - 230) + abs(corner[0] - 20), 30, "past the bench is the plate's blue")
        bench = cam.project(np.array([[0.75, 0.25, 0.75]]))[0][0]
        b = rgb[int(bench[1]), int(bench[0])]
        self.assertGreater(b[2], b[0] + 80, "the bench top is the plate's, darkened at most by a shadow")
        bare = sim.render(dataclasses.replace(cam, background=""))["rgb"].astype(int)
        self.assertGreater(np.abs(bare - rgb).mean(), 10.0, "without its plate the camera draws the bench and room")


if __name__ == "__main__":
    unittest.main()
