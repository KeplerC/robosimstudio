# SPDX-License-Identifier: Apache-2.0
"""The workpiece: the clip's chrome grab bar, welded to the world.

A 25 mm tube on two 90-degree elbows whose legs drop into the black blocks --
a bathroom grab bar, which is what the real rig wipes. It is a **fixture** and
not a free body: the reference bar is bolted into its blocks and the press is
reacted by the blocks, so the model is a ``static`` :class:`ObjectSpec` and the
whole wipe force goes into the foam's wrap round the tube. The record of the
free 22 mm bar with end cubes that this replaced -- and of the support hand
never getting it off its pads -- is in ``PROVENANCE.md``.

**Still an ``ObjectSpec`` rather than a ``StaticSpec``**, for the reason
``box_decant``'s knife records: only the ``objects`` bin gets a full
``EntityHandle``, and without one the goal cannot say ``(hands-clear bar)`` and
the routine cannot read the bar's pose off ``handles.entity("bar").body``. A
static object is a jointless link, which MuJoCo's converter refuses outright
("outside articulations and no standalone joint to world"); the entity welds it
with a FIXED joint from the world, which is the converter's static-root path.
Under GENERIC it builds without the weld, so a rigid-only check never sees the
failure -- the test that matters builds the shipped two-engine arrangement.

The straight run is the spec's own cylinder, so the taint field's radius check
reads it first; the elbows and legs are added here. Every contact the bar makes
is against the foam (SuperDex, against an SDF where a cylinder is exact) or the
gripper pads steadying it at the far elbow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import newton
import numpy as np
import warp as wp

from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs import SimRequirements, contacts_with
from robosimstudio.specs.asset import ObjectSpec, ShapeSpec

from ..look import STEEL
from .layout import BAR_QUAT, LAYOUT, Layout

__all__ = ["CHROME", "GRIP_SPAN", "EntityGrabBar", "SpecGrabBar", "bar_frame", "grab_bar", "taint_point_local"]

GRIP_SPAN = 0.09396
"""YAM ``linear_4310`` fingertip separation at full open [m]."""

CHROME = (0.80, 0.80, 0.80)
"""The tube's colour under the brushed-steel finish. Stated once here and read
by the taint field as what a cleared cell fades to, so the decal and the bar
under it cannot drift apart."""


@dataclass(frozen=True, kw_only=True)
class SpecGrabBar(ObjectSpec):
    """A tube on two elbows, fixed to the world.

    Its axis is the body's local **z** (where ``ShapeSpec`` puts a cylinder),
    the crown is local +x and the legs drop along local -x; the loader turns
    that onto the bench with :data:`~.layout.BAR_QUAT`. The dimensions default
    to the layout's so the record, the field and the routine all read one bar.
    """

    entity: str = "grab_bar"
    category: str = "grab_bar"
    name: str = "bar"
    static: bool = True

    radius: float = LAYOUT.tube_radius
    half_len: float = LAYOUT.straight_half
    bend_radius: float = LAYOUT.bend_radius
    leg_drop: float = LAYOUT.leg_drop
    buried: float = 0.015
    """How far each leg runs into its block [m]: the leg ends inside the block
    so no seam shows at the top face."""
    elbow_segments: int = 5
    """Cylinders per elbow. Five reads as a swept tube from every camera the
    suite uses; the joints carry spheres so nothing shows between them."""

    shape: ShapeSpec | None = None
    friction: float = 0.6
    color: tuple[float, float, float] = CHROME

    def __post_init__(self) -> None:
        # The straight run IS the spec's primitive; stating it twice is how the
        # two copies drift. `ObjectSpec` is frozen, so this is the one place it
        # can be derived.
        if self.shape is None:
            object.__setattr__(self, "shape", ShapeSpec(kind="cylinder", radius=self.radius, half_height=self.half_len))
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()


def bar_frame(spec: SpecGrabBar) -> dict[str, float]:
    """Everything measured of the bar, in its own frame. The single source."""
    leg_u = float(spec.half_len) + float(spec.bend_radius)
    return {
        "radius": float(spec.radius),
        "half_len": float(spec.half_len),
        "bend_radius": float(spec.bend_radius),
        "leg_drop": float(spec.leg_drop),
        # Axial distance from the centre to a leg's axis.
        "leg_u": leg_u,
        # Leg to leg, outer edge to outer edge.
        "overall": 2.0 * (leg_u + float(spec.radius)),
        # The straight cylinder each leg adds below its elbow.
        "leg_len": float(spec.leg_drop) - float(spec.bend_radius) + float(spec.buried),
    }


@register_entity("grab_bar")
class EntityGrabBar(EntityRigid):
    """The straight run (built by ``EntityRigid``) plus an elbow and a leg at each end.

    Each elbow is a fan of short cylinders round the bend's centreline with a
    sphere at every joint. In the bar's own frame the tube runs along local z,
    the crown is local +x and the legs drop along local -x.
    """

    spec: SpecGrabBar

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        super().build(builder, episode)
        spec = self.spec
        self._assert_geometry()
        # A static object is a jointless link, which MuJoCo's converter refuses
        # ("outside articulations and no standalone joint to world"): weld it.
        # A FIXED joint from the world is the converter's static-root path.
        world = builder.body_q[self.body]
        builder.add_joint_fixed(-1, self.body, parent_xform=world, label=f"{spec.name}_weld")
        cfg = self._contact_cfg(builder)
        r, R, n = float(spec.radius), float(spec.bend_radius), int(spec.elbow_segments)
        colour = tuple(float(v) for v in spec.color)
        for end, sign in (("far", 1.0), ("near", -1.0)):
            centre = np.array([-R, 0.0, sign * spec.half_len])  # the bend's centre of curvature
            last = None
            for k in range(n + 1):
                a = (math.pi / 2.0) * (1.0 - k / n)  # 90 deg (tube end) -> 0 (leg top)
                p = centre + np.array([R * math.sin(a), 0.0, sign * R * math.cos(a)])
                builder.add_shape_sphere(
                    self.body,
                    xform=wp.transform(wp.vec3(*p), wp.quat_identity()),
                    radius=r,
                    cfg=cfg,
                    color=colour,
                    label=f"{spec.name}_{end}_elbow_{k}",
                )
                if last is not None:
                    mid = 0.5 * (last + p)
                    d = p - last
                    length = float(np.linalg.norm(d))
                    d = d / length
                    # a cylinder's axis is its local z: rotate z onto d
                    axis = np.cross([0.0, 0.0, 1.0], d)
                    s = float(np.linalg.norm(axis))
                    if s < 1e-9:
                        q = wp.quat_identity()
                    else:
                        q = wp.quat_from_axis_angle(wp.vec3(*(axis / s)), float(math.atan2(s, d[2])))
                    builder.add_shape_cylinder(
                        self.body,
                        xform=wp.transform(wp.vec3(*mid), q),
                        radius=r,
                        half_height=0.5 * length,
                        cfg=cfg,
                        color=colour,
                        label=f"{spec.name}_{end}_bend_{k}",
                    )
                last = p
            # the leg: straight down from the elbow's end into the block
            leg_len = bar_frame(spec)["leg_len"]
            top = np.array([-R, 0.0, sign * (spec.half_len + R)])
            mid = top + np.array([-0.5 * leg_len, 0.0, 0.0])
            q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -math.pi / 2.0)  # z onto -x
            builder.add_shape_cylinder(
                self.body,
                xform=wp.transform(wp.vec3(*mid), q),
                radius=r,
                half_height=0.5 * leg_len,
                cfg=cfg,
                color=colour,
                label=f"{spec.name}_{end}_leg",
            )

    def _assert_geometry(self) -> None:
        spec = self.spec
        frame = bar_frame(spec)
        assert frame["leg_len"] > 0.0, (
            f"a {spec.bend_radius * 1e3:.0f} mm elbow already spans the {spec.leg_drop * 1e3:.0f} mm drop; "
            "the leg has no length"
        )
        assert 2.0 * float(spec.radius) + 0.010 <= GRIP_SPAN, (
            f"a {2 * spec.radius * 1e3:.0f} mm tube across a {GRIP_SPAN * 1e3:.0f} mm jaw span: the support "
            "hand cannot close on it"
        )
        assert float(spec.half_len) > 4.0 * float(spec.radius), "the straight run is shorter than two tube widths"

    def apply_display(self, builder: newton.ModelBuilder) -> None:
        """Chrome on every piece: one finish, so the elbows read as the tube."""
        super().apply_display(builder)
        for s in range(*self.range.shapes):
            self.finishes[s] = STEEL

    def as_declared(self) -> SpecGrabBar:
        return self.spec


def grab_bar(layout: Layout = LAYOUT) -> SpecGrabBar:
    """The bar, welded where the layout stands it: axis at ``layout.bar_z``."""
    return SpecGrabBar(
        pose=(layout.bar_x, layout.bar_y, layout.bar_z),
        quat=BAR_QUAT,
        radius=layout.tube_radius,
        half_len=layout.straight_half,
        bend_radius=layout.bend_radius,
        leg_drop=layout.leg_drop,
        # `deformables` is what puts the bar in reach of the foam. The sponge
        # declares the mirror side of the same rule (`objects`); either alone is
        # enough to make the edge, and stating both is how the pair survives
        # one of them being edited.
        sim=SimRequirements(domain="rigid", contacts=contacts_with("surface", "fixtures", "robot", "deformables")),
    )


def taint_point_local(spec: SpecGrabBar, u: float, theta: float) -> np.ndarray:
    """A point on the tube's surface in the bar's own frame, from (axial, angle).

    ``theta`` is measured from the bar's local +x, which the loader stands up as
    world +z -- so ``theta = 0`` is the crown of the bar as it lies on the
    bench, and the sign follows the right hand about the bar's own axis. The
    one conversion, used by anything aiming at a point on the stripe.
    """
    r = float(spec.radius)
    return np.array([r * np.cos(theta), r * np.sin(theta), float(u)], dtype=np.float64)
