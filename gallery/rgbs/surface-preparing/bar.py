# SPDX-License-Identifier: Apache-2.0
"""Grab bar geometry adapted from the shipped wipe demo; no grasp latch."""
import itertools
import math
from dataclasses import dataclass, replace
from types import SimpleNamespace
import newton
import numpy as np
import warp as wp
import robosimstudio as rss
from robosimstudio.domains.rigid import EntityRigid
from robosimstudio.core.render.finish import BRUSHED_METAL
LAYOUT = SimpleNamespace(tube_radius=.0125, straight_half=.170, bend_radius=.032, leg_drop=.048)
CHROME=(.8,.8,.8)
STEEL=replace(BRUSHED_METAL, roughness=.3, metallic=.95)
@dataclass(frozen=True, kw_only=True)
class BarSpec(rss.ObjectSpec):
    """A tube on two elbows whose flat legs meet the top plates. Its axis is local z, its crown local +x."""

    entity: str = "grab_bar_tube"
    name: str = "bar"
    static: bool = True
    radius: float = LAYOUT.tube_radius
    half_len: float = LAYOUT.straight_half
    bend_radius: float = LAYOUT.bend_radius
    leg_drop: float = LAYOUT.leg_drop
    color: tuple[float, float, float] = CHROME
    friction: float = 0.6

    def __post_init__(self) -> None:
        if self.shape is None:  # the straight run is the spec's own cylinder
            object.__setattr__(
                self, "shape", rss.ShapeSpec(kind="cylinder", radius=self.radius, half_height=self.half_len)
            )


@rss.register_entity("grab_bar_tube")
class GrabBar(EntityRigid):
    """The straight run, built by ``EntityRigid``, plus an elbow and a leg at each end, welded to the world."""

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        super().build(builder, episode)
        spec = self.spec
        cfg, r, bend = self._contact_cfg(builder), float(spec.radius), float(spec.bend_radius)
        for end, sign in (("far", 1.0), ("near", -1.0)):
            centre = np.array([-bend, 0.0, sign * spec.half_len])  # the bend's centre of curvature
            points = [
                centre + bend * np.array([math.sin(a), 0.0, sign * math.cos(a)]) for a in np.linspace(math.pi / 2, 0, 6)
            ]
            points.append(points[-1] + np.array([-(spec.leg_drop - bend), 0.0, 0.0]))  # the leg
            for k, (a, b) in enumerate(itertools.pairwise(points)):
                self._tube(builder, a, b, r, cfg, f"{spec.name}_{end}_{k}")
                if k == len(points)-2: continue  # flat leg ends at the top plate
                builder.add_shape_sphere(
                    self.body, xform=wp.transform(wp.vec3(*b), wp.quat_identity()), radius=r, cfg=cfg, color=CHROME
                )

    def _tube(self, builder, a, b, radius, cfg, label) -> None:
        d = b - a
        length = float(np.linalg.norm(d))
        axis = np.cross([0.0, 0.0, 1.0], d / length)  # a cylinder's axis is its local z: turn z onto d
        s = float(np.linalg.norm(axis))
        q = (
            wp.quat_from_axis_angle(wp.vec3(*(axis / s)), math.atan2(s, d[2] / length))
            if s > 1e-9
            else wp.quat_identity()
        )
        builder.add_shape_cylinder(
            self.body,
            xform=wp.transform(wp.vec3(*(0.5 * (a + b))), q),
            radius=radius,
            half_height=0.5 * length,
            cfg=cfg,
            color=CHROME,
            label=label,
        )

    def apply_display(self, builder: newton.ModelBuilder) -> None:
        super().apply_display(builder)
        for s in range(*self.range.shapes):
            self.finishes[s] = STEEL


