# SPDX-License-Identifier: Apache-2.0
"""The YAM's wrist-camera meshes are not drawn, and nothing else on the robot goes with them.

``yam_bimanual_arms.xml`` states each D405 mesh ``rgba="0 0 0 0"``, which MuJoCo
does not render. The studio and the path tracer drew them anyway, as a black box
beside each wrist, until :meth:`Entity.apply_display` began clearing the visible
flag of a shape its asset made fully transparent. This builds the robot and holds
the change to exactly those two shapes: every other shape the file draws is still
drawn, and every collider still collides.
"""

from __future__ import annotations

import unittest

import newton
import numpy as np

TIER = "medium"

VISIBLE = int(newton.ShapeFlags.VISIBLE)
COLLIDES = int(newton.ShapeFlags.COLLIDE_SHAPES)


class TestYamWristCameraIsNotDrawn(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss
        from robosimstudio.paths import resolve_asset
        from robosimstudio.robots.registry import get_embodiment

        yam = get_embodiment("yam_bimanual")
        scene = rss.Scene(cell="bench_0728", physics=rss.GENERIC, name="yam_wrist_camera")
        scene.add(rss.robot(yam))
        cls.sim = scene.build(control_mode="joint_position")
        model = cls.sim.model
        cls.labels = list(model.shape_label)
        cls.flags = model.shape_flags.numpy()
        cls.opacity = model.shape_opacity.numpy()

        # The file as Newton's importer reads it, before anything here paints it.
        raw = newton.ModelBuilder()
        raw.add_mjcf(str(resolve_asset(yam.asset)), floating=False)
        cls.raw = {
            label: (int(raw.shape_flags[s]), float(raw.shape_opacity[s])) for s, label in enumerate(raw.shape_label)
        }

    def _shape(self, suffix: str) -> int:
        found = [s for s, label in enumerate(self.labels) if label.endswith(suffix)]
        self.assertEqual(len(found), 1, f"{suffix}: {found}")
        return found[0]

    def test_both_d405_meshes_are_hidden(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                s = self._shape(f"{side}_camera_d405_geom_0_visual")
                self.assertEqual(float(self.opacity[s]), 0.0)
                self.assertFalse(self.flags[s] & VISIBLE)

    def test_the_camera_colliders_still_collide(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                self.assertTrue(self.flags[self._shape(f"{side}_cam_col")] & COLLIDES)

    def test_every_other_shape_the_file_draws_is_still_drawn(self):
        matched, hidden = 0, []
        for s, label in enumerate(self.labels):
            if label not in self.raw:
                continue
            matched += 1
            raw_flags, raw_opacity = self.raw[label]
            if raw_flags & VISIBLE and not self.flags[s] & VISIBLE:
                hidden.append(label.rsplit("/", 1)[-1])
                self.assertEqual(raw_opacity, 0.0, label)
            self.assertEqual(bool(raw_flags & COLLIDES), bool(self.flags[s] & COLLIDES), label)
        self.assertGreater(matched, 50)
        self.assertEqual(sorted(hidden), ["left_camera_d405_geom_0_visual", "right_camera_d405_geom_0_visual"])

    def test_nothing_drawn_is_fully_transparent(self):
        drawn = (self.flags & VISIBLE) != 0
        self.assertFalse(np.any(self.opacity[drawn] <= 0.0))


if __name__ == "__main__":
    unittest.main()
