# SPDX-License-Identifier: Apache-2.0
"""A shape its asset stated fully transparent is not drawn.

MuJoCo does not render a geom whose ``rgba`` alpha is 0. Newton's importer keeps
that alpha as ``shape_opacity`` and leaves the shape visible, and the studio's
materials and the path tracer's finishes take the colour and not the alpha, so
the YAM's wrist-camera meshes (``rgba="0 0 0 0"``) drew as a black box beside
each wrist. :meth:`Entity.apply_display` now clears the visible flag on such a
shape and leaves everything else about it alone. No device: a builder, a fake
entity that imports a small MJCF, and the paint step.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import newton

from robosimstudio.scene.build.imports import is_transparent
from robosimstudio.scene.entity import Entity

TIER = "fast"

MJCF = """
<mujoco model="wrist">
  <worldbody>
    <body name="wrist" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 0 1"/>
      <geom name="shell" type="box" size="0.02 0.02 0.02" contype="0" conaffinity="0" rgba="0.3 0.3 0.3 1"/>
      <geom name="tinted" type="box" size="0.01 0.01 0.01" contype="0" conaffinity="0" rgba="0 0 1 0.5"/>
      <geom name="camera_mesh" type="box" size="0.01 0.02 0.01" contype="0" conaffinity="0" group="3"
            mass="0.001" rgba="0 0 0 0"/>
      <geom name="pad" type="sphere" size="0.02" group="3" rgba="0 0 1 0" mass="0.001"/>
    </body>
  </worldbody>
</mujoco>
"""


class _Wrist(Entity):
    """Imports :data:`MJCF`; one visual shell, one translucent, one hidden mesh, one hidden collider."""

    role = "robot"

    def __init__(self) -> None:
        super().__init__(SimpleNamespace(name="wrist", color=None, visible=True, collide=True, finish=None))

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        builder.add_mjcf(MJCF, floating=False)


def _by_name(builder: newton.ModelBuilder) -> dict[str, int]:
    return {label.rsplit("/", 1)[-1].removesuffix("_visual"): s for s, label in enumerate(builder.shape_label)}


VISIBLE = int(newton.ShapeFlags.VISIBLE)
COLLIDES = int(newton.ShapeFlags.COLLIDE_SHAPES)


class TestTransparentShapesAreNotDrawn(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = newton.ModelBuilder()
        self.entity = _Wrist()
        self.entity.build_into(self.builder, episode=None)
        self.shape = _by_name(self.builder)

    def test_the_importer_keeps_the_alpha_and_leaves_the_shape_visible(self):
        # The premise: without the paint step the alpha-0 mesh would be drawn.
        raw = newton.ModelBuilder()
        raw.add_mjcf(MJCF, floating=False)
        mesh = _by_name(raw)["camera_mesh"]
        self.assertEqual(float(raw.shape_opacity[mesh]), 0.0)
        self.assertTrue(int(raw.shape_flags[mesh]) & VISIBLE)
        self.assertTrue(is_transparent(raw, mesh))

    def test_an_alpha_0_visual_shape_is_hidden(self):
        self.assertFalse(int(self.builder.shape_flags[self.shape["camera_mesh"]]) & VISIBLE)

    def test_opaque_and_translucent_shapes_stay_drawn(self):
        for name in ("shell", "tinted"):
            with self.subTest(shape=name):
                self.assertFalse(is_transparent(self.builder, self.shape[name]))
                self.assertTrue(int(self.builder.shape_flags[self.shape[name]]) & VISIBLE)

    def test_an_alpha_0_collider_still_collides(self):
        pad = self.shape["pad"]
        self.assertTrue(is_transparent(self.builder, pad))
        self.assertTrue(int(self.builder.shape_flags[pad]) & COLLIDES)
        self.assertFalse(int(self.builder.shape_flags[pad]) & VISIBLE)

    def test_the_hidden_mesh_keeps_its_body_and_mass(self):
        raw = newton.ModelBuilder()
        raw.add_mjcf(MJCF, floating=False)
        self.assertEqual(self.builder.shape_body, raw.shape_body)
        self.assertEqual(self.builder.body_mass, raw.body_mass)
        mesh = self.shape["camera_mesh"]
        self.assertEqual(
            tuple(float(v) for v in self.builder.shape_transform[mesh]),
            tuple(float(v) for v in raw.shape_transform[_by_name(raw)["camera_mesh"]]),
        )


if __name__ == "__main__":
    unittest.main()
