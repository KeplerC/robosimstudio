"""The filled pot's union boundary for liquid; original solids for rigid contact."""

from pathlib import Path

import newton
import numpy as np
import trimesh

import robosimstudio as rss
from robosimstudio.domains.static.entity import EntityStatic
from robosimstudio.scene.registry import register_entity


class RigidScenery(EntityStatic):
    def build(self, builder, episode):
        start = builder.shape_count
        super().build(builder, episode)
        for i in range(start, builder.shape_count):
            builder.shape_flags[i] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)


class LiquidBoundary(EntityStatic):
    def build(self, builder, episode):
        start = builder.shape_count
        super().build(builder, episode)
        for i in range(start, builder.shape_count):
            builder.shape_flags[i] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
            builder.shape_margin[i] = max(builder.shape_margin[i], 0.026 if "table" in self.spec.name else 0.016)


register_entity("plant_rigid_scenery")(RigidScenery)
register_entity("plant_liquid_boundary")(LiquidBoundary)


def filled_pot():
    # Exterior of pot, soil and saucer. The lower plug lies inside the
    # wooden table; buried saucer/pot undersides cannot push liquid downward.
    profile = np.array(
        [
            (0, -0.035),
            (0.075, -0.035),
            (0.075, 0.011),
            (0.069, 0.011),
            (0.066, 0.006),
            (0.050, 0.006),
            (0.050, 0.013),
            (0.061, 0.135),
            (0.068, 0.137),
            (0.070, 0.141),
            (0.068, 0.145),
            (0.061, 0.145),
            (0.058, 0.125),
            (0.057, 0.1245),
            (0, 0.1245),
        ]
    )
    mesh = trimesh.creation.revolve(profile, sections=128)
    mesh.process(validate=True)
    mesh.fix_normals()
    assert mesh.is_watertight
    path = Path(__file__).resolve().parent / "assets" / "pot_liquid_boundary.obj"
    mesh.export(path)
    return rss.StaticSpec(
        name="pot_liquid_boundary",
        entity="plant_liquid_boundary",
        asset=rss.AssetSpec(path=str(path)),
        xform=(0, -0.014, 0, 0, 0, 0, 1),
        visible=False,
        collide=True,
    )
