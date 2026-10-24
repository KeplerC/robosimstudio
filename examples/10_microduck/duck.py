# SPDX-License-Identifier: Apache-2.0
"""The Microduck as a kind of thing: a free-standing biped with its own policies on board.

The robot is Pollen Robotics' MJCF imported on a free joint, so nothing holds it
up but its feet; :mod:`assets` adds the beak. Once per control step the entity's
``before`` hook does what the robot's daemon does: it reads the IMU and the joints
off the state, runs the active ONNX policy and writes fifteen servo targets. The
policies are the official ones, trained in MuJoCo Warp at 50 Hz over a 5 ms step,
which is the arrangement :data:`DUCK_200HZ` states.

``Ref`` accessors steer it: :meth:`DuckRef.walk` sets the twist,
:meth:`DuckRef.look` the head, :meth:`DuckRef.mouth` the beak, and
:meth:`DuckRef.peck` runs the ground-pick move (crouch, beak to the floor, stand).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import newton
import numpy as np
import warp as wp

import robosimstudio as rss
from assets import MOUTH_RANGE, fetch, patched_mjcf
from robosimstudio.specs.physics import DOMAINS, ArrangementSpec, backend

JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)  # fmt: skip
"""The fourteen joints every policy reads and drives, in its order. The mouth is not one."""

HOME = np.array(
    [0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0, 0.0, 0.0873, 0.4579, 0.0049, -0.4530],
    dtype=np.float32,
)
"""The pose the policies were trained around: an action is an offset from it."""

TRUNK_Z = 0.125  # where the runtime places the trunk before it lets go [m]
PECK_PERIOD, PECK_END = 4.0, 0.7  # the ground pick is phase-driven: 4 s period, done at 0.7 of it
NOD = 0.6  # head pitch that tips a held ball out over the beak's lip [rad]
TRICKS = {"kick_left": 3.0, "kick_right": 3.0, "roulade": 2.0}
KICK_SPOT = (0.09, 0.042)  # where a kick expects the ball: ahead of the trunk, out to the kicking foot [m]
"""One-shot policies and how long each runs before the gait takes back over [s]; each ends standing."""
HELD = (-0.025, 0.0, -0.060)  # where a thing sits in the open beak, in the head frame [m]

DUCK_200HZ = ArrangementSpec(
    control_hz=50.0,
    substeps=4,
    backends=(("*", backend("mujoco", nconmax=2048, njmax=8192)),),
)
"""What the policies were trained at: 50 Hz control over a 5 ms MuJoCo step."""

if "duck" not in DOMAINS:
    rss.register_domain(
        "duck",
        state=("bodies", "joints", "shapes"),
        reads="rigid",
        role="objects",
        contacts=("surface", "fixtures", "objects", "self"),
        summary="A free-standing biped that runs its own policy.",
    )


@dataclass(frozen=True, kw_only=True)
class DuckSpec:
    entity: str = "microduck"
    name: str = "duck"
    at: tuple[float, float] = (0.0, 0.0)  # x, y in the work-surface frame [m]
    yaw: float = 0.0  # heading [rad]
    on: str = "floor"  # "floor", or "surface" to stand on the cell's work surface: the lab bench
    sim: Any = field(
        default_factory=lambda: rss.SimRequirements(
            domain="duck", contacts=rss.contacts_with("surface", "fixtures", "objects", "self")
        )
    )


class Brain:
    """The daemon's half: three policies behind one 61-float observation, and the beak."""

    def __init__(self, policies) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        load = lambda f: ort.InferenceSession(str(policies / f), opts, providers=["CPUExecutionProvider"])  # noqa: E731
        self.sessions = {
            "walk": load("velstand.onnx"),
            "peck": load("alpha_ground_pick.onnx"),
            "sit": load("alpha_sitstand.onnx"),
            "kick_left": load("ball_kick_left.onnx"),
            "kick_right": load("ball_kick_right.onnx"),
            "roulade": load("roulade.onnx"),
        }
        self.reset()

    def reset(self) -> None:
        self.mode = "walk"
        self.twist = np.zeros(3, np.float32)  # vx, vy [m/s], yaw rate [rad/s]
        self.head = np.zeros(4, np.float32)  # neck_pitch, head_pitch, head_yaw, head_roll offsets [rad]
        self.body = np.zeros(6, np.float32)  # x, y, z, roll, pitch, yaw of the trunk while standing
        self.sitting = False
        self.opening = 0.0  # beak, 0 shut .. 1 open
        self.phase = 0.0
        self.last = np.zeros(14, np.float32)
        self.nod = 0.0  # seconds left of a drop: beak open, head tipped down

    def command(self) -> np.ndarray:
        cmd = np.zeros(13, np.float32)
        if self.mode in TRICKS:
            return cmd  # trained on an all-zero command, head and body slots included
        if self.mode == "walk":
            cmd[0:3] = self.twist
        elif self.mode == "peck":
            cmd[0:2] = math.cos(2 * math.pi * self.phase), math.sin(2 * math.pi * self.phase)
        elif self.mode == "sit":
            cmd[0] = 1.0 if self.sitting else 0.0
        cmd[3:7] = self.head
        cmd[7:13] = self.body
        return cmd

    def act(self, gyro, gravity, q, qd, dt) -> np.ndarray:
        """Fourteen joint targets from one observation; advances the ground pick's clock."""
        if self.mode == "peck":
            self.phase += dt / PECK_PERIOD
            if self.phase >= PECK_END:
                self.mode, self.phase = "walk", 0.0
        elif self.mode in TRICKS:
            self.phase += dt
            if self.phase >= TRICKS[self.mode]:
                self.mode, self.phase = "walk", 0.0
        command = self.command()
        if self.nod > 0.0:
            self.nod -= dt
            command[4] = NOD  # head_pitch, the head's own joint: the beak tips 35 deg down
            if self.nod <= 0.0:
                self.opening = 0.0
        obs = np.concatenate([gyro, gravity, q - HOME, qd, self.last, command]).astype(np.float32)[None]
        session = self.sessions[self.mode]
        self.last = session.run(None, {session.get_inputs()[0].name: obs})[0][0].astype(np.float32)
        return HOME + self.last

    def mouth_target(self) -> float:
        lo, hi = MOUTH_RANGE
        return lo + float(np.clip(self.opening, 0.0, 1.0)) * (hi - lo)


class DuckRef(rss.Ref):
    """Steer the duck. Each setter takes effect at the next control step."""

    @property
    def brain(self) -> Brain:
        return self.entity.brain

    def walk(self, vx: float = 0.0, vy: float = 0.0, yaw_rate: float = 0.0) -> None:
        """Forward and sideways speed [m/s] and turn rate [rad/s]. Zero stands still."""
        b = self.brain
        if b.mode == "sit" and not b.sitting and (vx or vy or yaw_rate):
            b.mode = "walk"  # stood back up: hand back to the gait
        b.twist[:] = (vx, vy, yaw_rate)

    def look(self, neck_pitch: float = 0.0, head_pitch: float = 0.0, head_yaw: float = 0.0, head_roll: float = 0.0):
        """Head offsets from the standing pose [rad]; the policy keeps its balance around them."""
        self.brain.head[:] = (neck_pitch, head_pitch, head_yaw, head_roll)

    def mouth(self, opening: float) -> None:
        """Beak opening, 0 shut (squeezing) to 1 open (30 deg)."""
        self.brain.opening = float(opening)

    def drop(self, seconds: float = 1.2) -> None:
        """Open the beak and tip the head down, so what it holds rolls out; then shut it again."""
        b = self.brain
        b.opening, b.nod = 1.0, float(seconds)

    def carry_to(self, xy, *, drop: bool = True, short: float = 0.03, speed: float = 0.4) -> None:
        """Walk until the beak is within ``short`` of ``xy`` (world, m), steering on the beak; then drop what it holds.

        The policy tracks a commanded speed loosely -- about half of it on this table -- so the walk
        closes on where the beak actually is, every control step, rather than on dead reckoning.
        """
        self.entity.errand = {
            "to": np.asarray(xy, dtype=np.float64)[:2],
            "drop": drop,
            "short": short,
            "speed": speed,
            "wait": 0.0,
        }

    @property
    def busy(self) -> bool:
        """Whether an errand or a drop is still under way."""
        return self.entity.errand is not None or self.brain.nod > 0.0

    def peck(self) -> None:
        """Crouch and put the beak to the floor in front of the feet, then stand: 2.8 s."""
        b = self.brain
        if b.mode == "walk":
            b.twist[:] = 0.0
            b.mode, b.phase = "peck", 0.0

    def trick(self, name: str) -> None:
        """``"kick_left"``, ``"kick_right"`` or ``"roulade"`` (a forward roll), from a standstill."""
        b = self.brain
        if b.mode == "walk":
            b.twist[:] = 0.0
            b.mode, b.phase = name, 0.0

    def sit(self, sitting: bool = True) -> None:
        b = self.brain
        b.mode, b.sitting = "sit", bool(sitting)

    @property
    def mode(self) -> str:
        return self.brain.mode

    def trunk(self) -> np.ndarray:
        """Trunk pose, ``(x, y, z, qx, qy, qz, qw)`` in world."""
        return np.asarray(self._scene.pipeline.snapshot.body_q[self.entity.trunk], dtype=np.float64)

    def mouth_tip(self) -> np.ndarray:
        """Where the beak's tip is, in world [m]."""
        e = self.entity
        X = np.asarray(self._scene.pipeline.snapshot.body_q[e.head], dtype=np.float64)
        return _apply(X, e.tip_local)

    def kick_spot(self, side: str = "left", radius: float = 0.035) -> np.ndarray:
        """Where a kick expects the ball's centre, in world [m], wherever the duck now stands."""
        T = self.trunk()
        fwd = _rot(T[3:7])[:, 0]
        yaw = math.atan2(fwd[1], fwd[0])
        ahead, out = KICK_SPOT[0], KICK_SPOT[1] if side == "left" else -KICK_SPOT[1]
        c, s = math.cos(yaw), math.sin(yaw)
        ground = T[2] - TRUNK_Z + 0.009  # the trunk stands 116 mm over its soles, not 125
        return np.array([T[0] + ahead * c - out * s, T[1] + ahead * s + out * c, ground + radius])

    def held_point(self, ahead: float = 0.0) -> np.ndarray:
        """Where a thing sits in the open beak, or ``ahead`` metres out along it, in world [m]: where to feed it."""
        X = np.asarray(self._scene.pipeline.snapshot.body_q[self.entity.head], dtype=np.float64)
        return _apply(X, (HELD[0], HELD[1], HELD[2] - ahead))


def frame(studio, eye, look_at) -> None:
    """Point each browser that connects at ``look_at`` from ``eye`` [m, world], over its first half second."""
    seen: dict[int, int] = {}

    def each() -> None:
        for cid, client in studio.server.get_clients().items():
            n = seen.get(cid, 0)
            if n < 25:
                seen[cid] = n + 1
                client.camera.up_direction = (0.0, 0.0, 1.0)
                client.camera.position = tuple(eye)
                client.camera.look_at = tuple(look_at)

    studio.each_step(each)


def _rot(q) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _apply(X, p) -> np.ndarray:
    return np.asarray(X[:3]) + _rot(X[3:7]) @ np.asarray(p)


@rss.register_entity("microduck")
class Duck(rss.Entity):
    domain = "duck"
    ref = DuckRef

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        from newton.solvers import SolverMuJoCo

        SolverMuJoCo.register_custom_attributes(builder)
        spec = self.spec
        # x and y in the work-surface frame; z on the ground, or on the work surface itself.
        pose = self.placed(np.array([spec.at[0], spec.at[1], 0.0, 0.0, 0.0, 0.0, 1.0]))
        ground = float(pose[2]) if spec.on == "surface" else 0.0
        half = 0.5 * spec.yaw
        xform = wp.transform(
            (float(pose[0]), float(pose[1]), ground + TRUNK_Z), wp.quat(0.0, 0.0, math.sin(half), math.cos(half))
        )
        body0, joint0 = builder.body_count, builder.joint_count
        builder.add_mjcf(
            str(patched_mjcf()),
            xform=xform,
            floating=True,
            enable_self_collisions=True,
            collapse_fixed_joints=False,
            verbose=False,
        )
        labels = builder.joint_label
        self.coords = {}
        for j in range(joint0, builder.joint_count):
            short = labels[j].rsplit("/", 1)[-1]
            if short in JOINTS or short == "mouth":
                self.coords[short] = int(builder.joint_q_start[j])
        self.bodies = list(range(body0, builder.body_count))
        bl = builder.body_label
        self.trunk = next(b for b in self.bodies if bl[b].rsplit("/", 1)[-1] == "trunk_base")
        self.head = next(b for b in self.bodies if bl[b].rsplit("/", 1)[-1] == "jaw_soft")
        self.tip_local = (-0.00809, 0.0, -0.0777)  # the MJCF's mouth_tip site, head frame
        # Stand it in its home pose, servo targets included, so the first policy step starts from rest.
        for name, value in zip(JOINTS, HOME, strict=True):
            builder.joint_q[self.coords[name]] = float(value)
            builder.joint_target_q[self.coords[name]] = float(value)
        self.home_q = {k: float(builder.joint_q[v]) for k, v in self.coords.items()}
        self.root_q = list(builder.joint_q[builder.joint_q_start[joint0] : builder.joint_q_start[joint0] + 7])
        self.root_coord = int(builder.joint_q_start[joint0])
        self.root_dof = int(builder.joint_qd_start[joint0])
        self.brain = Brain(fetch() / "policies")
        self.errand = None

    def bind(self, model: newton.Model) -> None:
        super().bind(model)
        self.qd_index = np.array([self._dof_of(model, n) for n in JOINTS])
        self.q_index = np.array([self.coords[n] for n in JOINTS])

    def _dof_of(self, model, name: str) -> int:
        starts = model.joint_q_start.numpy()
        j = int(np.searchsorted(starts, self.coords[name], side="right") - 1)
        return int(model.joint_qd_start.numpy()[j])

    def handle(self, model):
        del model
        from robosimstudio.scene.handles import EntityHandle

        return EntityHandle(name=self.name, body=self.trunk, bodies=tuple(self.bodies), coord_start=self.root_coord)

    def on_scene_ready(self, pipeline) -> None:
        self.pipeline = pipeline

    def contribute_hooks(self, hooks) -> None:
        hooks.add("before", self._think, name="duck-brain", cadence=1)

    def _think(self, state, dt) -> None:
        pipeline = getattr(self, "pipeline", None)
        if pipeline is None:
            return
        X = state.body_q.numpy()[self.trunk]
        w_world = state.body_qd.numpy()[self.trunk][3:6]
        R = _rot(X[3:7])
        if self.errand is not None:
            self._steer(R, state.body_q.numpy()[self.head])
        q = state.joint_q.numpy()
        qd = state.joint_qd.numpy()
        targets = self.brain.act(
            (R.T @ w_world).astype(np.float32),
            (R.T @ np.array([0.0, 0.0, -1.0])).astype(np.float32),
            q[self.q_index].astype(np.float32),
            qd[self.qd_index].astype(np.float32),
            1.0 / 50.0,
        )
        out = pipeline.control.joint_target_q.numpy()
        out[self.q_index] = targets
        out[self.coords["mouth"]] = self.brain.mouth_target()
        pipeline.control.joint_target_q.assign(out)

    def _steer(self, R, head) -> None:
        """One control step of an errand: turn the beak onto the target, walk, stop short, drop."""
        job, b = self.errand, self.brain
        if job["wait"] > 0.0:  # stopped: let it settle, then drop
            job["wait"] -= 1.0 / 50.0
            if job["wait"] <= 0.0:
                if job["drop"]:
                    b.opening, b.nod = 1.0, 1.2
                self.errand = None
            return
        beak = _apply(head, HELD)[:2]
        to = job["to"] - beak
        fwd = R[:2, 0] / max(np.linalg.norm(R[:2, 0]), 1e-9)
        along = float(fwd @ to)
        if np.linalg.norm(to) < job["short"] or (along < 0.0 and np.linalg.norm(to) < 0.1):  # there, or just past it
            b.twist[:] = 0.0
            job["wait"] = 0.6
            return
        bearing = math.atan2(fwd[0] * to[1] - fwd[1] * to[0], along)
        if abs(bearing) > 0.6:  # well off to one side: turn on the spot first
            b.twist[:] = (0.0, 0.0, math.copysign(0.8, bearing))
            return
        speed = job["speed"] if along > job["short"] + 0.15 else 0.2
        b.twist[:] = (speed, 0.0, float(np.clip(2.0 * bearing, -0.8, 0.8)))

    def reset(self, pipeline, episode, *, world: int = 0) -> None:
        self.brain.reset()
        self.errand = None
