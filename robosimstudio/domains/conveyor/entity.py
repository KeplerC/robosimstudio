# SPDX-License-Identifier: Apache-2.0
"""A slat conveyor: kinematic slats that carry what lies on them by contact, in every engine.

Why slats, and why kinematic. A belt has to carry two kinds of thing: a rigid parcel on
MuJoCo and a deformable one on SuperDex. MuJoCo reads a contact's slip from the joint
velocities, so a body that stays put while claiming a velocity would carry a box; SuperDex
reads a collider's surface velocity from its consecutive poses
(:mod:`~robosimstudio.engines.superdex.mirror`), so the same trick leaves a mailer where it
lies. The surface has to really move. Each slat is a kinematic link: Newton gives a kinematic
DOF a 1e10 armature on MuJoCo, so the velocity written to it is the velocity its contacts see,
and its pose, written beside it, is what SuperDex sweeps.

Why two DOFs and a fast return. A free joint per slat is six DOFs, and MuJoCo's solve grows
with the DOF count much faster than linearly: 58 free slats cost 110 ms per control step with
a box on them, filtering slat-against-slat contacts saved 5 ms of it, and 29 cost 40 ms
(``p0g_mjcost.py``). A slat that only slides along the belt and dips under it needs two, and a
return run ``return_ratio`` times faster than the belt needs few slats under it: 21 slats on
a 1.4 m belt at 80 mm pitch cost 6.0 ms per control step with a box on them, and 38 ms with a
SuperDex mailer beside it (each slat is a SuperDex collider too); the box rode at 0.2000 m/s and
the mailer at 0.1997 on a 0.2 m/s belt (``p1_smoke.py``, RTX PRO 6000).

Two belts can meet: an infeed that ends against a deck running across it, the classic
right-angle transfer, works because every belt's parts share one collision group
(:data:`CONVEYOR_GROUP`) and never touch each other.

The path. Every slat's pose is a function of the belt's *travel* -- the integral of its speed
-- so a stopped belt stops every slat, the returning ones too, and a restarted one picks up
where it was. Along the belt (``x'``, from the upstream end) and up (``z``)::

    top run   x' = u                z = 0           u in [0, L)
    dive      x' = L + u            z = -D u / d    u in [0, d)
    return    x' = L + d - k w      z = -D          k = return_ratio, under the top run
    rise      x' = -d + u           z = -D + D u / d

Both pose and velocity are written every substep, joints and bodies alike: written to the
joints alone, SuperDex read a pose a step stale and a mailer rode at 0.18 m/s on a 0.2 belt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import newton
import numpy as np
import warp as wp

from robosimstudio.scene.entity import Entity
from robosimstudio.scene.ref import Ref
from robosimstudio.specs import SimRequirements, Vec3, contacts_with

__all__ = ["CONVEYOR_GROUP", "ConveyorRef", "ConveyorSpec", "EntityConveyor", "conveyor", "slat_path"]

CONVEYOR_GROUP = -7
"""Newton's collision group for every conveyor's slats and frame. A negative group collides with
everything but itself, so a parcel, a hand and a bench still meet the belt, while slat against
slat, slat against its own frame and one belt against another -- an infeed's end poking into the
deck it feeds -- never make a contact. Two immovable things in contact are an infeasible
constraint: with an infeed's skirts reaching into a deck's slats, MuJoCo flung a box 8 m."""


@dataclass(frozen=True, kw_only=True)
class ConveyorSpec:
    """A straight slat conveyor.

    Attributes:
        at: The middle of the carrying surface's top, in the work-surface frame.
        yaw: The direction the top run travels, about ``z`` [rad]; 0 carries along ``+x``.
        length: The top run [m].
        width: Across the belt [m].
        pitch: Slat pitch on the top run [m]; rounded so a whole number of slats fills the loop.
        gap: Between neighbouring slats [m].
        thickness: Of a slat [m].
        drop: How far under the top run a slat returns [m].
        dive: Belt travel over which a slat dips at the downstream end and rises at the upstream one [m].
        return_ratio: How much faster than the belt the return run moves.
        speed: Belt speed at the start [m/s]; the ref's ``speed`` changes it while running.
        friction: Of the slats.
        rail_height: Side rails this far over the carrying surface [m]; 0 for none (the sides
            are then skirts that end at the surface).
        color: Of the slats.
        frame_color: Of the skirts, rails and end drums.
    """

    entity: str = "conveyor"
    name: str = "belt"
    at: Vec3 = (0.0, 0.0, 0.1)
    yaw: float = 0.0
    length: float = 1.0
    width: float = 0.45
    pitch: float = 0.08
    gap: float = 0.002
    thickness: float = 0.012
    drop: float = 0.04
    dive: float = 0.03
    return_ratio: float = 8.0
    speed: float = 0.2
    friction: float = 1.0
    rail_height: float = 0.0
    color: Vec3 = (0.09, 0.09, 0.10)
    frame_color: Vec3 = (0.62, 0.63, 0.65)
    sim: SimRequirements = field(
        default_factory=lambda: SimRequirements(
            domain="conveyor", contacts=contacts_with("objects", "deformables", "robot")
        )
    )

    def __post_init__(self) -> None:
        for key in ("length", "width", "pitch", "thickness", "drop", "dive", "return_ratio"):
            if not float(getattr(self, key)) > 0.0:
                raise ValueError(f"conveyor {self.name!r}: {key} must be positive, got {getattr(self, key)}")
        if self.gap < 0.0 or self.gap >= self.pitch:
            raise ValueError(f"conveyor {self.name!r}: gap must be in [0, pitch), got {self.gap}")
        if self.drop <= self.thickness:
            # The return run would rub the underside of the top run.
            raise ValueError(
                f"conveyor {self.name!r}: drop ({self.drop}) must exceed the slat thickness ({self.thickness})"
            )

    @property
    def period(self) -> float:
        """Belt travel [m] for one slat to go round once."""
        L, d = float(self.length), float(self.dive)
        return L + d + (L + 2.0 * d) / float(self.return_ratio) + d


def slat_path(s: float, length: float, dive: float, drop: float, ratio: float) -> tuple[float, float, float, float]:
    """``(x', z, dx'/ds, dz/ds)`` of a slat at belt travel ``s``, as the drive kernel computes it."""
    L, d, D, k = float(length), float(dive), float(drop), float(ratio)
    ret = (L + 2.0 * d) / k
    s = float(s) % (L + d + ret + d)
    if s < L:
        return s, 0.0, 1.0, 0.0
    s -= L
    if s < d:
        return L + s, -D * s / d, 1.0, -D / d
    s -= d
    if s < ret:
        return L + d - k * s, -D, -k, 0.0
    s -= ret
    return -d + s, -D + D * s / d, 1.0, D / d


@wp.func
def _path(s: float, L: float, d: float, D: float, k: float):
    ret = (L + 2.0 * d) / k
    period = L + d + ret + d
    s = s - period * wp.floor(s / period)
    if s < L:
        return wp.vec4(s, 0.0, 1.0, 0.0)
    s = s - L
    if s < d:
        return wp.vec4(L + s, -D * s / d, 1.0, -D / d)
    s = s - d
    if s < ret:
        return wp.vec4(L + d - k * s, -D, -k, 0.0)
    s = s - ret
    return wp.vec4(-d + s, -D + D * s / d, 1.0, D / d)


@wp.kernel
def _drive_slats(
    travel: wp.array(dtype=float),
    speed: wp.array(dtype=float),
    dt: float,
    s0: wp.array(dtype=float),
    q_start: wp.array(dtype=int),
    qd_start: wp.array(dtype=int),
    bodies: wp.array(dtype=int),
    start: wp.vec3,
    yaw: float,
    L: float,
    d: float,
    D: float,
    k: float,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()
    v = speed[0]
    p = _path(s0[i] + travel[0] + v * dt, L, d, D, k)
    joint_q[q_start[i] + 0] = p[0]
    joint_q[q_start[i] + 1] = p[1]
    joint_qd[qd_start[i] + 0] = p[2] * v
    joint_qd[qd_start[i] + 1] = p[3] * v
    cy = wp.cos(yaw)
    sy = wp.sin(yaw)
    b = bodies[i]
    body_q[b] = wp.transform(
        start + wp.vec3(cy * p[0], sy * p[0], p[1]), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
    )
    body_qd[b] = wp.spatial_vector(wp.vec3(cy * p[2] * v, sy * p[2] * v, p[3] * v), wp.vec3(0.0, 0.0, 0.0))


@wp.kernel
def _advance(travel: wp.array(dtype=float), speed: wp.array(dtype=float), dt: float):
    travel[0] = travel[0] + speed[0] * dt


class ConveyorRef(Ref):
    """The belt's speed, read and set while it runs."""

    @property
    def speed(self) -> float:
        """The belt speed now [m/s]."""
        return self.entity.speed

    @speed.setter
    def speed(self, value: float) -> None:
        self.entity.set_speed(value)

    @property
    def travel(self) -> float:
        """How far the belt has moved since the build [m]."""
        return self.entity.travel

    def stop(self) -> None:
        self.entity.set_speed(0.0)

    def run(self, speed: float | None = None) -> None:
        """Start the belt at ``speed``, or at the spec's speed."""
        self.entity.set_speed(self.entity.spec.speed if speed is None else speed)

    def to_belt(self, point: Vec3) -> np.ndarray:
        """A world point in the belt's own frame: ``(along, across, up)`` from the middle of the top."""
        e = self.entity
        rel = np.asarray(point, dtype=np.float64) - e.centre
        cy, sy = math.cos(e.spec.yaw), math.sin(e.spec.yaw)
        return np.array([cy * rel[0] + sy * rel[1], -sy * rel[0] + cy * rel[1], rel[2]])


class EntityConveyor(Entity):
    """Builds :class:`ConveyorSpec`: the slats, their drive, and a static frame round them."""

    domain = "conveyor"
    ref = ConveyorRef

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec = self.spec
        L, d, k = float(spec.length), float(spec.dive), float(spec.return_ratio)
        n = max(2, int(math.ceil(spec.period / float(spec.pitch))))
        spacing = spec.period / n  # the pitch, rounded to fit the loop
        hz = float(spec.thickness) / 2.0
        centre = self.placed(np.array([*spec.at, 0.0, 0.0, 0.0, 1.0], dtype=np.float64))
        cy, sy = math.cos(spec.yaw), math.sin(spec.yaw)
        along = np.array([cy, sy, 0.0])
        self.centre = np.asarray(centre[:3], dtype=np.float64)
        # The slats' origin: the upstream end of the top run, on the slats' mid-plane.
        self.start = self.centre - along * (L / 2.0) - np.array([0.0, 0.0, hz])
        qyaw = (0.0, 0.0, math.sin(spec.yaw / 2.0), math.cos(spec.yaw / 2.0))
        parent = wp.transform(tuple(float(v) for v in self.start), wp.quat(*qyaw))

        cfg = builder.default_shape_cfg.copy()
        cfg.density = 600.0
        cfg.mu = float(spec.friction)
        cfg.collision_group = CONVEYOR_GROUP
        dof = newton.ModelBuilder.JointDofConfig
        self.bodies: list[int] = []
        self.joints: list[int] = []
        slat_shapes: list[int] = []
        self.s0 = [i * spacing for i in range(n)]
        for i, s in enumerate(self.s0):
            x, z, _, _ = slat_path(s, L, d, spec.drop, k)
            pos = self.start + along * x + np.array([0.0, 0.0, z])
            body = builder.add_link(
                xform=wp.transform(tuple(float(v) for v in pos), wp.quat(*qyaw)),
                label=f"{spec.name}/slat{i:03d}",
                is_kinematic=True,
            )
            slat_shapes.append(
                builder.add_shape_box(
                    body,
                    hx=spacing / 2.0 - float(spec.gap) / 2.0,
                    hy=float(spec.width) / 2.0,
                    hz=hz,
                    cfg=cfg,
                    label=f"{spec.name}/slat{i:03d}",
                )
            )
            joint = builder.add_joint_d6(
                -1,
                body,
                linear_axes=[dof(axis=(1.0, 0.0, 0.0)), dof(axis=(0.0, 0.0, 1.0))],
                parent_xform=parent,
                label=f"{spec.name}/slat{i:03d}",
            )
            q0 = builder.joint_q_start[joint] if hasattr(builder, "joint_q_start") else None
            if q0 is not None:
                builder.joint_q[q0] = x
                builder.joint_q[q0 + 1] = z
            builder.add_articulation([joint], label=f"{spec.name}/slat{i:03d}")
            self.bodies.append(body)
            self.joints.append(joint)
        self.slat_shapes = slat_shapes
        self.n = n
        self.spacing = spacing
        self.frame_shapes = self._build_frame(builder, along, qyaw)
        self.speed = float(spec.speed)
        self._dev: dict[str, Any] | None = None
        self._model = None

    def _build_frame(self, builder: newton.ModelBuilder, along: np.ndarray, qyaw: tuple) -> list[int]:
        """Side skirts (rails when ``rail_height`` > 0) and an end drum at each end, welded to the world."""
        spec = self.spec
        cfg = builder.default_shape_cfg.copy()
        cfg.mu = 0.3  # a guide, not a grip
        cfg.collision_group = CONVEYOR_GROUP
        L, W = float(spec.length), float(spec.width)
        skirt = 0.01
        top = float(spec.rail_height)
        bottom = -(float(spec.drop) + float(spec.thickness) + 0.02)
        half_h = (top - bottom) / 2.0
        across = np.array([-along[1], along[0], 0.0])
        shapes = []
        for side in (-1.0, 1.0):
            centre = self.centre + across * side * (W / 2.0 + skirt + 0.003) + np.array([0.0, 0.0, bottom + half_h])
            shapes.append(
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(tuple(float(v) for v in centre), wp.quat(*qyaw)),
                    hx=L / 2.0,  # no further than the top run: past it is another belt's room
                    hy=skirt,
                    hz=half_h,
                    cfg=cfg,
                    label=f"{spec.name}/skirt{'LR'[int(side > 0)]}",
                )
            )
        radius = (float(spec.drop) - float(spec.thickness)) / 2.0
        if radius > 0.004:
            # A drum's axis lies across the belt: the box frame turned about x by 90 degrees.
            q_drum = _quat_mul(qyaw, (math.sin(math.pi / 4.0), 0.0, 0.0, math.cos(math.pi / 4.0)))
            for end, sign in (("up", -1.0), ("down", 1.0)):
                centre = self.centre + along * sign * (L / 2.0 + float(spec.dive) / 2.0)
                centre = centre - np.array([0.0, 0.0, float(spec.thickness) + radius])
                shapes.append(
                    builder.add_shape_cylinder(
                        -1,
                        xform=wp.transform(tuple(float(v) for v in centre), wp.quat(*q_drum)),
                        radius=radius,
                        half_height=W / 2.0,
                        cfg=cfg,
                        label=f"{spec.name}/drum_{end}",
                    )
                )
        return shapes

    def apply_display(self, builder: newton.ModelBuilder) -> None:
        for s in self.slat_shapes:
            builder.shape_color[s] = tuple(float(c) for c in self.spec.color)
        for s in self.frame_shapes:
            builder.shape_color[s] = tuple(float(c) for c in self.spec.frame_color)

    def handle(self, model: newton.Model) -> Any | None:
        """None: a belt is machinery, not a thing to click and drag."""
        del model
        return None

    def contribute_hooks(self, hooks) -> None:
        hooks.add("post_solve", self._drive, name=f"{self.spec.name}-drive", cadence=1)

    def on_scene_ready(self, pipeline) -> None:
        self._model = pipeline.model

    def reset(self, pipeline, episode, *, world: int = 0) -> None:
        del pipeline, episode, world
        self.speed = float(self.spec.speed)
        if self._dev is not None:
            self._dev["travel"].zero_()
            self._dev["speed"].assign([self.speed])

    @property
    def travel(self) -> float:
        return 0.0 if self._dev is None else float(self._dev["travel"].numpy()[0])

    def set_speed(self, value: float) -> None:
        self.speed = float(value)
        if self._dev is not None:
            self._dev["speed"].assign([self.speed])

    def _arrays(self, device) -> dict[str, Any]:
        if self._dev is not None and self._dev["device"] == device:
            return self._dev
        model = self._model
        jq = model.joint_q_start.numpy()
        jqd = model.joint_qd_start.numpy()
        self._dev = {
            "device": device,
            "travel": wp.zeros(1, dtype=float, device=device),
            "speed": wp.array([self.speed], dtype=float, device=device),
            "s0": wp.array(self.s0, dtype=float, device=device),
            "q": wp.array([int(jq[j]) for j in self.joints], dtype=int, device=device),
            "qd": wp.array([int(jqd[j]) for j in self.joints], dtype=int, device=device),
            "b": wp.array(self.bodies, dtype=int, device=device),
        }
        return self._dev

    def _drive(self, state, dt) -> None:
        if self._model is None:
            return
        a = self._arrays(state.joint_q.device)
        spec = self.spec
        wp.launch(
            _drive_slats,
            dim=self.n,
            inputs=[
                a["travel"], a["speed"], float(dt), a["s0"], a["q"], a["qd"], a["b"],
                wp.vec3(*(float(v) for v in self.start)), float(spec.yaw), float(spec.length), float(spec.dive),
                float(spec.drop), float(spec.return_ratio),
                state.joint_q, state.joint_qd, state.body_q, state.body_qd,
            ],
            device=a["travel"].device,
        )  # fmt: skip
        wp.launch(_advance, dim=1, inputs=[a["travel"], a["speed"], float(dt)], device=a["travel"].device)


def conveyor(*, name: str = "belt", at: Vec3 = (0.0, 0.0, 0.1), **kwargs: Any) -> ConveyorSpec:
    """A slat conveyor whose carrying surface's top is centred at ``at`` (work-surface frame).

    ``yaw`` points the direction the top run travels; ``speed`` starts it, and the ref the
    scene hands back changes it (``belt.stop()``, ``belt.run()``, ``belt.speed = 0.1``).
    Whatever lies on it is carried by contact -- a rigid body on MuJoCo and a SuperDex
    deformable alike -- so the arrangement only needs to state where those live.
    """
    return ConveyorSpec(name=name, at=tuple(float(v) for v in at), **kwargs)


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )
