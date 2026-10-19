# SPDX-License-Identifier: Apache-2.0
"""A vacuum cup: whether its face is on a part, and holding the part once it is.

A cup holds a flat part by sealing onto one of its faces, so there are two
questions here and they are kept apart. :func:`face_on_box` and
:meth:`Suction.seal` answer the first from geometry alone: how far the face's
centre stands off a flat face of the part (one it seals on, else the nearest),
how far the cup's normal is from that face's, and whether the whole cup is over
it. Geometry rather than the contact pipeline, because a cup seals across the
bellows' give before the rubber touches, and a contact list only says touching
or not. :meth:`Suction.hold` answers the second in one of two tiers:

* **weld**: a MuJoCo weld equality between the pad body and the part, one row
  per part the scene built, disabled at build and switched on at hold with the
  pose the part has in the pad's frame at that moment. Newton 1.6 keeps the
  rows in ``model.mujoco.equality_constraint_*``, and
  :meth:`~newton.solvers.SolverMuJoCo.notify_model_changed` with
  ``CONSTRAINT_PROPERTIES`` copies them into ``eq_data`` and ``eq_active`` by
  kernel, into arrays a captured graph already reads, so a hold and a release
  cost no recapture: the plate is held, carried 198 mm and let go with the same
  graph and no eager step. A weld is a constraint, so it carries whatever the
  arm can lift, and what the arm cannot shows as the arm sagging rather than
  the part falling: a 2 kg plate stays on the cup while the YAM's tool centre
  sags 38 mm below the cursor. It needs the part on the arm's MuJoCo solver
  (``rss.MUJOCO_500HZ``); :data:`~robosimstudio.specs.GENERIC` puts free
  bodies on VBD, where the row exists but would weld the pad to a mirror, so
  it is refused there.
* **wrench**: a spring-damper between the pad and the part, applied as equal
  and opposite wrenches every substep, so the arm carries what it lifts: the
  same 2 kg plate sags the YAM 35 mm and rides 5 mm under the cup. The spring
  pulls the part toward the pose it had under the face, damped on the two
  bodies' relative velocity, and lets go on the device when the pull a stiff
  seal would feel at its stretch passes :attr:`Suction.break_force`, the vacuum
  times the face's area (49.5 N for the shipped 30 mm cup, a 12.4 mm stretch on
  a part of any mass). The fallback where the weld is not available, and the
  tier whose hold has the ceiling a real cup has.

Every number here is ``tests/test_suction_pick`` on an RTX PRO 6000 (GPU 5,
Newton 1.6.0).

Both stay live in a captured graph: the wrench kernel is launched every substep
and returns early on an idle cup, and the weld toggle writes arrays the solver
already reads. One world: the rows and the wrench address world 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import newton
import numpy as np
import warp as wp

from .math import mat_to_quat, quat_to_mat

__all__ = [
    "BREAK_PRESSURE",
    "GAP_TOLERANCE",
    "TILT_TOLERANCE",
    "SealReport",
    "Suction",
    "face_on_box",
    "flat_faces",
]

GAP_TOLERANCE = 0.002
"""How far the face's centre may stand off the part's face and still seal [m]: the bellows' give."""
TILT_TOLERANCE = math.radians(15.0)
"""How far the cup's normal may be from the part's face normal and still seal [rad]."""
COMPRESSION_LIMIT = 0.010
"""How far into the part the face's centre may read before a seal is refused [m]. A cup pressed
onto a plate on MuJoCo rests within 0.2 mm of the plate's face (``probe_descend.py``, GPU 5); a
centimetre is the cup through the part."""
BREAK_PRESSURE = 70_000.0
"""The vacuum a seal holds against the air [Pa]: the 0.7 bar a shop cup is rated for.
Times the face's area it is the force the wrench tier lets go at."""
WRENCH_STIFFNESS = 4000.0
"""The wrench tier's spring [N/m]: the 49 N a 30 mm cup breaks at is a 12 mm stretch.

The break reads the stretch against this stiffness even where a light part's
spring is softened below it, so the seal lets go at the same stretch whatever
the part weighs."""
WRENCH_ANGULAR_STIFFNESS = 40.0
"""The wrench tier's angular spring [N.m/rad]."""
WRENCH_DAMPING_RATIO = 1.0
"""Critical damping, on the part's velocity relative to the pad's."""


@dataclass(frozen=True)
class SealReport:
    """What a seal test or a hold found, in the units the page shows."""

    sealed: bool
    """Whether the cup is on the part: gap and tilt inside tolerance, the whole face over it."""
    gap_m: float
    """The face centre's stand-off from the part's face it seals on, else its nearest [m], negative when pressed in."""
    tilt_rad: float
    """The angle between the cup's normal and that face's normal [rad]."""
    body: int
    """The part's model body index, or -1."""
    why: str = ""
    """Why it did not seal, in a sentence, or empty."""


def face_on_box(face_pose, pad_radius: float, box_pose, half) -> tuple[float, float, bool]:
    """Where a cup's face stands relative to one box, in world.

    Args:
        face_pose: The face's frame, ``(x, y, z, qx, qy, qz, qw)``: its origin
            is the face's centre and its +z points out of the cup, into the part.
        pad_radius: The cup's radius [m].
        box_pose: The box's frame, the same shape.
        half: The box's half-extents [m].

    Returns:
        ``(gap, tilt, covered)`` for the box face the cup points at: the face
        centre's stand-off from that face's plane (negative is inside the box),
        the angle between the cup's normal and the plane's inward normal, and
        whether the cup's whole footprint lies within that face. The face chosen
        is the one whose outward normal is most opposed to the cup's normal,
        among the faces the centre is not beyond the far side of: a cup over a
        plate pointing up faces the plate's sides, not its underside through
        it, and reports a right angle. Each axis keeps at least one of its two
        faces, since the centre cannot be beyond both, so there is always one.
    """
    face_pose = np.asarray(face_pose, dtype=np.float64).reshape(7)
    box_pose = np.asarray(box_pose, dtype=np.float64).reshape(7)
    half = np.asarray(half, dtype=np.float64).reshape(3)
    normal = quat_to_mat(face_pose[3:7])[:, 2]
    rot = quat_to_mat(box_pose[3:7])
    local = rot.T @ (face_pose[:3] - box_pose[:3])
    facing: list[tuple[float, float, bool]] = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            gap = float(sign * local[axis] - half[axis])
            if gap < -2.0 * half[axis]:
                continue
            outward = sign * rot[:, axis]
            tilt = math.acos(float(np.clip(-np.dot(normal, outward), -1.0, 1.0)))
            covered = all(abs(local[k]) + pad_radius <= half[k] + 1e-9 for k in range(3) if k != axis)
            facing.append((gap, tilt, covered))
    return min(facing, key=lambda found: found[1])


def flat_faces(model: newton.Model, body: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """The boxes a body offers a cup: ``(shape pose in the body, half-extents)`` per shape.

    A box is its half-extents; a mesh is its vertices' bounding box in the
    shape's frame, which is right for a mesh plate or lid and wrong for a bowl,
    whose bounding box has a top nothing can seal on. Other primitives offer no
    flat face and are left out.
    """
    bodies = model.shape_body.numpy()
    types = model.shape_type.numpy()
    scales = model.shape_scale.numpy()
    xforms = model.shape_transform.numpy()
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for s in np.flatnonzero(bodies == int(body)):
        kind = int(types[s])
        if kind == int(newton.GeoType.BOX):
            out.append((np.asarray(xforms[s], dtype=np.float64), np.asarray(scales[s], dtype=np.float64)))
        elif kind in (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH)):
            vertices = getattr(model.shape_source[s], "vertices", None)
            if vertices is None:
                continue
            points = np.asarray(vertices, dtype=np.float64) * np.asarray(scales[s], dtype=np.float64)
            lo, hi = points.min(axis=0), points.max(axis=0)
            pose = np.asarray(xforms[s], dtype=np.float64).copy()
            pose[:3] = pose[:3] + quat_to_mat(pose[3:7]) @ (0.5 * (lo + hi))
            out.append((pose, 0.5 * (hi - lo)))
    return out


def _refusal(gap: float, tilt: float, covered: bool) -> str:
    """Why a face with this gap, tilt and cover does not seal, in a sentence, or empty when it does."""
    if tilt > TILT_TOLERANCE:
        return f"the cup is {math.degrees(tilt):.1f} degrees off the face; {math.degrees(TILT_TOLERANCE):.0f} is the most it seals at"
    if gap > GAP_TOLERANCE:
        return f"the face is {gap * 1e3:.1f} mm off the part; {GAP_TOLERANCE * 1e3:.0f} mm is the most it seals across"
    if gap < -COMPRESSION_LIMIT:
        return f"the face reads {-gap * 1e3:.1f} mm inside the part"
    if not covered:
        return "the cup hangs over the part's edge"
    return ""


def _compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a * b`` on ``(x, y, z, qx, qy, qz, qw)`` poses."""
    ra, rb = quat_to_mat(a[3:7]), quat_to_mat(b[3:7])
    return np.concatenate([a[:3] + ra @ b[:3], mat_to_quat(ra @ rb)])


def _inverse(a: np.ndarray) -> np.ndarray:
    ra = quat_to_mat(a[3:7])
    return np.concatenate([-(ra.T @ a[:3]), mat_to_quat(ra.T)])


@wp.kernel
def _cup_wrench_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[wp.float32],
    body_inertia: wp.array[wp.mat33],
    pad: int,
    held: wp.array[wp.int32],
    offset: wp.array[wp.transform],
    lin_stiffness: float,
    ang_stiffness: float,
    damping_ratio: float,
    break_force: wp.array[wp.float32],
    dt: float,
    broke: wp.array[wp.int32],
    body_f: wp.array[wp.spatial_vector],
):
    """Spring the held part toward its pose under the pad, and push the pad back as hard.

    Newton's conventions: ``body_qd`` is (linear at the centre of mass,
    angular) and ``body_f`` is (force, torque about the centre of mass), both
    in world. The springs are softened to a natural frequency of half the
    substep rate, so a light part does not make the explicit force unstable.
    The seal breaks on the device, so a captured graph lets go on the substep
    it happens, when the stretch times the unsoftened ``lin_stiffness`` passes
    ``break_force``: the pull a stiff seal would feel. Read off the softened
    spring instead, the break would be a stretch that grows as the part gets
    lighter (43 mm for 5 g at 960 Hz).
    """
    part = held[0]
    if part < 0:
        return
    target = body_q[pad] * offset[0]
    tf = body_q[part]
    com_local = body_com[part]
    com = wp.transform_point(tf, com_local)
    com_target = wp.transform_point(target, com_local)
    pad_com = wp.transform_point(body_q[pad], body_com[pad])

    mass = body_mass[part]
    inertia = body_inertia[part]
    moment = (inertia[0, 0] + inertia[1, 1] + inertia[2, 2]) / 3.0
    omega = 0.5 / dt
    k = wp.min(lin_stiffness, mass * omega * omega)
    k_ang = wp.min(ang_stiffness, moment * omega * omega)

    stretch = com_target - com
    if wp.length(stretch) * lin_stiffness > break_force[0]:
        held[0] = -1
        broke[0] = 1
        return
    pull = stretch * k

    qd_pad = body_qd[pad]
    qd = body_qd[part]
    w_pad = wp.spatial_bottom(qd_pad)
    v_pad_here = wp.spatial_top(qd_pad) + wp.cross(w_pad, com - pad_com)
    v_rel = wp.spatial_top(qd) - v_pad_here
    w_rel = wp.spatial_bottom(qd) - w_pad
    force = pull - v_rel * (2.0 * damping_ratio * wp.sqrt(wp.max(k * mass, 1.0e-12)))

    q_err = wp.transform_get_rotation(target) * wp.quat_inverse(wp.transform_get_rotation(tf))
    if q_err[3] < 0.0:
        q_err = wp.quat(-q_err[0], -q_err[1], -q_err[2], -q_err[3])
    axis = wp.vec3(q_err[0], q_err[1], q_err[2])
    sin_half = wp.length(axis)
    rot_vec = wp.vec3(0.0, 0.0, 0.0)
    if sin_half > 1.0e-8:
        rot_vec = (axis / sin_half) * (2.0 * wp.atan2(sin_half, q_err[3]))
    torque = rot_vec * k_ang - w_rel * (2.0 * damping_ratio * wp.sqrt(wp.max(k_ang * moment, 1.0e-12)))

    wp.atomic_add(body_f, part, wp.spatial_vector(force, torque))
    wp.atomic_add(body_f, pad, wp.spatial_vector(-force, -torque - wp.cross(com - pad_com, force)))


class Suction:
    """One cup on one arm: seal tests, holds and releases against a built scene.

    Built by the embodiment at bind, one per arm whose hand declares a
    :attr:`~robosimstudio.specs.GripperSpec.suction_pad`, and reached as
    ``sim.suction("right")``. The weld rows were added before the model was
    finalized (the embodiment's ``after_scene_built``), one per part that offers
    a flat face, so :attr:`rows` is what this cup can hold.

    Args:
        model: The finalized model.
        arm: The arm the cup is on.
        pad_body: The pad body's index; its origin is the face and its +z the normal.
        pad_radius: The cup's radius [m].
        rows: Part body -> weld equality row.
        owners: Part body -> the entity that built it, to ask which solver steps it.
    """

    def __init__(
        self,
        model: newton.Model,
        *,
        arm: str,
        pad_body: int,
        pad_radius: float,
        rows: dict[int, int] | None = None,
        owners: dict[int, str] | None = None,
    ) -> None:
        self.model = model
        self.arm = str(arm)
        self.pad_body = int(pad_body)
        self.pad_radius = float(pad_radius)
        self.rows: dict[int, int] = {int(k): int(v) for k, v in (rows or {}).items()}
        self.owners: dict[int, str] = {int(k): str(v) for k, v in (owners or {}).items()}
        self.pipeline: Any = None
        self.mujoco: Any = None
        """The MuJoCo solver that steps the pad and carries the weld rows, or ``None``."""
        self.weldable: frozenset[int] = frozenset()
        """The parts the weld tier can hold: those the pad's MuJoCo solver steps too."""
        self.tier = ""
        """``"weld"`` or ``"wrench"`` while holding; empty otherwise."""
        self.last = SealReport(False, math.inf, math.nan, -1, "nothing tested yet")
        """The last seal test or hold; a look with ``record=False`` (the page's **Seal** readout) leaves it."""
        self._held = -1
        device = model.device
        self._break_dev = wp.array([BREAK_PRESSURE * math.pi * self.pad_radius**2], dtype=wp.float32, device=device)
        self._held_dev = wp.array([-1], dtype=wp.int32, device=device)
        self._broke_dev = wp.zeros(1, dtype=wp.int32, device=device)
        self._offset_dev = wp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=wp.transform, device=device)
        self._faces: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}

    @property
    def break_force(self) -> float:
        """The pull the wrench tier lets go at [N]: by default the vacuum times the face's area.

        Read as the stretch times :data:`WRENCH_STIFFNESS`, the pull a stiff
        seal would feel, so the seal lets go at one stretch whatever the part
        weighs. A device value, so a change reaches a graph that is already
        captured.
        """
        return float(self._break_dev.numpy()[0])

    @break_force.setter
    def break_force(self, newtons: float) -> None:
        self._break_dev.assign(np.array([float(newtons)], dtype=np.float32))

    # ---------------------------------------------------------------- wiring
    def contribute_hooks(self, hooks: Any) -> None:
        """The wrench tier's kernel, in every substep from the first capture, idle until a hold."""
        hooks.rigid_force.append(self.apply)

    def apply(self, state: newton.State, dt: float) -> None:
        """Add the wrench tier's pair of wrenches to ``state.body_f``. A :class:`SubstepHooks` hook."""
        wp.launch(
            _cup_wrench_kernel,
            dim=1,
            inputs=[
                state.body_q,
                state.body_qd,
                self.model.body_com,
                self.model.body_mass,
                self.model.body_inertia,
                self.pad_body,
                self._held_dev,
                self._offset_dev,
                WRENCH_STIFFNESS,
                WRENCH_ANGULAR_STIFFNESS,
                WRENCH_DAMPING_RATIO,
                self._break_dev,
                float(dt),
                self._broke_dev,
            ],
            outputs=[state.body_f],
            device=self.model.device,
        )

    def attach(self, pipeline: Any) -> None:
        """The pipeline can be stepped: find the MuJoCo solver that steps the pad, and the parts it steps too.

        A weld row is a constraint inside one solver, so it holds a part only
        when the solver stepping the pad steps the part as well. A program can
        have more than one MuJoCo island; a part on another, or a pad on none,
        is held by the wrench.
        """
        self.pipeline = pipeline
        self.mujoco, self.weldable = None, frozenset()
        program = getattr(pipeline, "program", None)
        if program is None:
            return
        cup = next((i for i in program.islands if self.pad_body in i.bodies), None)
        if cup is None or cup.backend.solver != "mujoco":
            return
        solver = program.solver_of_island(cup.name)
        if getattr(getattr(solver, "model", None), "mujoco", None) is None:
            return
        weldable = set()
        for body, owner in self.owners.items():
            try:
                island = program.island_of(owner)
            except (KeyError, LookupError):
                continue
            if island.name == cup.name and body in self.rows:
                weldable.add(body)
        self.mujoco, self.weldable = solver, frozenset(weldable)

    def weld_available(self, body: int) -> bool:
        """Whether the weld tier can hold ``body``: a row was built and the pad's MuJoCo solver steps the part."""
        return int(body) in self.weldable

    # -------------------------------------------------------------- geometry
    def _body_q(self, body_q: Any = None) -> np.ndarray:
        if body_q is None:
            body_q = self.pipeline.state_0.body_q.numpy()
        return np.asarray(body_q, dtype=np.float64)

    def face_pose(self, body_q: Any = None) -> np.ndarray:
        """The face's frame in world, ``(x, y, z, qx, qy, qz, qw)``."""
        return self._body_q(body_q)[self.pad_body].copy()

    def _flat(self, body: int) -> list[tuple[np.ndarray, np.ndarray]]:
        if body not in self._faces:
            self._faces[body] = flat_faces(self.model, body)
        return self._faces[body]

    def seal(self, body: int, body_q: Any = None, *, record: bool = True) -> SealReport:
        """Whether the cup is on ``body`` now: within :data:`GAP_TOLERANCE` and :data:`TILT_TOLERANCE`, fully over it.

        Args:
            body: The part's model body index.
            body_q: The bodies' poses to test against, ``(N, 7)``; the
                pipeline's current state when not given.
            record: Keep the report as :attr:`last`. The page's **Seal** readout
                passes False, so a look at the cup does not overwrite what a
                hold found.
        """
        report = self._seal(int(body), self._body_q(body_q))
        if record:
            self.last = report
        return report

    def _seal(self, body: int, poses: np.ndarray) -> SealReport:
        """The part's best face: one that seals, the nearest by gap among several, else the nearest refusal.

        A part of several shapes (convex hulls, a tray's base and rim) offers a
        box per shape, and the nearest face by gap alone can be one the cup is
        not over while another seals.
        """
        faces = self._flat(body)
        if not faces:
            return SealReport(False, math.inf, math.nan, body, "the part offers no flat face: no box or mesh shape")
        reports = []
        for pose, half in faces:
            gap, tilt, covered = face_on_box(poses[self.pad_body], self.pad_radius, _compose(poses[body], pose), half)
            why = _refusal(gap, tilt, covered)
            reports.append(SealReport(not why, gap, tilt, body, why))
        return min([r for r in reports if r.sealed] or reports, key=lambda r: abs(r.gap_m))

    def under(self, body_q: Any = None, *, record: bool = True) -> SealReport:
        """The seal test against whatever is under the cup: the part that seals, else the nearest refusal.

        What a vacuum switched on finds. Every part with a weld row is tested;
        a sealing part wins, the nearest by gap among several, and with none the
        report is the refusal of the part whose face is closest. ``record`` is
        :meth:`seal`'s.
        """
        if not self.rows:
            report = SealReport(False, math.inf, math.nan, -1, "nothing in the scene offers the cup a flat face")
        else:
            poses = self._body_q(body_q)
            reports = [self._seal(body, poses) for body in sorted(self.rows)]
            sealing = [r for r in reports if r.sealed]
            pool = sealing or [r for r in reports if math.isfinite(r.gap_m)] or reports
            report = min(pool, key=lambda r: abs(r.gap_m) if math.isfinite(r.gap_m) else math.inf)
        if record:
            self.last = report
        return report

    # ------------------------------------------------------------------ hold
    @property
    def held(self) -> int:
        """The part held, or -1. Read off the device on the wrench tier, which can let go by itself."""
        if self._held >= 0 and self.tier == "wrench" and int(self._held_dev.numpy()[0]) < 0:
            self._held, self.tier = -1, ""
        return self._held

    @property
    def holding(self) -> bool:
        return self.held >= 0

    @property
    def broke(self) -> bool:
        """Whether the wrench tier's seal broke since the last hold, of either tier, or the last reset."""
        return bool(int(self._broke_dev.numpy()[0]))

    @property
    def held_mass(self) -> float:
        """The held part's mass [kg], 0 when nothing is held.

        Read off the device, since a part's mass is tunable live, and only its
        own entry: the page's readout asks for it every refresh.
        """
        held = self.held
        return float(self.model.body_mass[held : held + 1].numpy()[0]) if held >= 0 else 0.0

    def hold(self, body: int | None = None, *, tier: str = "auto", check: bool = True) -> SealReport:
        """Seal onto a part and hold it: by weld where MuJoCo steps the part, else by wrench.

        Args:
            body: The part's model body index; ``None`` takes whatever is under
                the cup (:meth:`under`), which is what switching a vacuum on does.
            tier: ``"auto"``, ``"weld"`` or ``"wrench"``. Asking for a weld the
                scene cannot make raises, because holding by wrench instead would
                pass a test of the weld on the wrench.
            check: Test the seal first and refuse to hold when it fails.

        Returns:
            The seal report; ``sealed`` is whether the part asked for is now
            held. Asked again for the part it holds, in its tier or ``"auto"``,
            the hold stands and the report says where that part is now: a wrench
            hold rides a few millimetres under the face, past
            :data:`GAP_TOLERANCE`, and a fresh seal test would drop it. A part
            already held is let go only once another has sealed.
        """
        if self.pipeline is None:
            raise RuntimeError("Suction.hold before the scene is ready")
        if tier not in ("auto", "weld", "wrench"):
            raise ValueError(f"tier must be 'auto', 'weld' or 'wrench', not {tier!r}")
        held = self.held
        poses = self._body_q()
        if held >= 0 and (body is None or int(body) == held) and tier in ("auto", self.tier):
            found = self._seal(held, poses)
            self.last = SealReport(True, found.gap_m, found.tilt_rad, held)
            return self.last
        if body is None:
            report = self.under(poses)
            if not report.sealed:
                return report
            body = report.body
        body = int(body)
        report = self.seal(body, poses) if check else SealReport(True, 0.0, 0.0, body)
        if not report.sealed:
            return report
        if tier == "auto":
            tier = "weld" if self.weld_available(body) else "wrench"
        if tier == "weld" and not self.weld_available(body):
            raise RuntimeError(
                f"no weld from the cup to body {body}: it needs a row (built for {sorted(self.rows)}) and the part "
                f"on the arm's MuJoCo solver (it steps {sorted(self.weldable)})"
            )
        if held >= 0:
            self.release()
        relpose = _compose(_inverse(poses[self.pad_body]), poses[body]).astype(np.float32)
        self._broke_dev.zero_()
        if tier == "weld":
            self._weld(body, relpose)
        else:
            self._offset_dev.assign(relpose.reshape(1, 7))
            self._held_dev.assign(np.array([body], dtype=np.int32))
        self._held, self.tier = body, tier
        self.last = report
        return report

    def release(self) -> None:
        """Let go. Harmless when nothing is held."""
        if self._held >= 0 and self.tier == "weld":
            self._weld(self._held, None)
        self._held_dev.assign(np.array([-1], dtype=np.int32))
        self._held, self.tier = -1, ""

    def reset(self) -> None:
        """A new episode: nothing held, no broken seal remembered."""
        self.release()
        self._broke_dev.zero_()
        self.last = SealReport(False, math.inf, math.nan, -1, "nothing tested yet")

    def _weld(self, body: int, relpose: np.ndarray | None) -> None:
        """Switch the pad-to-body row on with ``relpose`` (the part in the pad's frame), or off with ``None``."""
        mujoco = self.mujoco.model.mujoco
        row = self.rows[body]
        enabled = mujoco.equality_constraint_enabled.numpy()
        enabled[row] = relpose is not None
        if relpose is not None:
            poses = mujoco.equality_constraint_relpose.numpy()
            poses[row] = relpose
            mujoco.equality_constraint_relpose.assign(poses)
        mujoco.equality_constraint_enabled.assign(enabled)
        self.mujoco.notify_model_changed(int(newton.ModelFlags.CONSTRAINT_PROPERTIES))

    def __repr__(self) -> str:
        return f"Suction({self.arm!r}, pad={self.pad_body}, held={self._held}, tier={self.tier or 'none'!r})"
