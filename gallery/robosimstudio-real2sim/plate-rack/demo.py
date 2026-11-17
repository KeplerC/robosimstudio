"""Two native YAM arms put six plates away by top-rim contact.

Run through scene.py and the live page. Hand.carry reads the measured grasp and computes robot joint targets; it never writes a plate pose. A 30 Hz state recording supports the drawn-surface check and a second path-traced camera view."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

import model as S
from robosimstudio.arms import arm_slot
from robosimstudio.core.math import mat_to_quat, quat_to_mat
from robosimstudio.solvers import grasp_quat

MODE = "joint_position"
HZ = 60.0

GRASP_DEPTH = 0.007
"""Distance down the top flange from the lip to the TCP [m], fitted for the native YAM fingertips."""
OPEN_EXTRA = 0.020
"""How much wider than the drawn rim the jaws open to approach [m]."""
SQUEEZE = 0.003
"""Nominal gap target below the drawn rim width [m]. Hand.action_into limits the closing target to 0.25 mm ahead of the measured gap; the unchanged catalog servo supplies the squeeze."""
PREGRASP_BACK = 0.05
GRASP_SPOKES = (0.0,)
"""Allowed rim spokes [degrees from the top]. This port uses only a top-rim pinch."""
MARGIN_ENOUGH = math.radians(10.0)
"""Planner early-acceptance joint margin [rad]; only the top spoke is enabled."""
LIFT = 0.025
"""Carry height above the seated plate height [m]."""
SET_BELOW = 0.004
"""How far below its resting height a set is aimed [m], so that it ends on the
cradle's push (``SET_STOP_FORCE``) rather than in the air."""
SET_STOP_FORCE = 1.0
"""The set stops the moment the shelf's cradle pushes back on the plate this hard [N]."""
SET_BACK = 0.002
"""Set-down offset behind the slot centre plane [m], chosen for a released lean below three degrees."""
SHELF_WIDTH = 0.17
"""Shelf and drying-rack depth across the slots [m]; model.rack_specs width."""
RELEASE_SECONDS = 0.2
"""How fast the jaws open on a set plate [s]: before it leans back into one of them."""
RETREAT_CLEAR = 0.03
"""Required clearance along the approach before returning home [m]. Included in the grasp planning waypoints."""
RETREAT_BACK = 0.05
RETREAT_UP = 0.015
"""The rest of the retreat: on back along the approach to this far, then up [m]
-- as far of it as the arm reaches (``Hand.line(tolerant=True)``), then a
joint-space move home."""
STAGE_OUT = 0.02
"""Where a carry stops: this far outside the shelf's side [m], plate edge to shelf edge."""

SPEED = {"move": 0.035, "fine": 0.025, "slide": 0.025, "lift": 0.025, "set": 0.012}
"""Cartesian speed by phase [m/s], replanned for the native fingertip grip."""

FEEDBACK = {"grasp": True, "integral": False}
"""Re-read the measured grasp every control step. Plate-error integration is disabled to avoid winding up against a divider; hand servo-lag compensation is separate."""

PLATE_I_GAIN = 0.03
PLATE_I_LIMIT = 0.02
"""The plate-position integral: gain per control step, and its ceiling [m]."""

TCP_I_GAIN = 0.05
TCP_I_LIMIT = (0.035, math.radians(8.0))
TCP_I_AXES = np.array([1.0, 1.0, 1.0])
"""Hand-pose integral axes: all three world translation axes. Its correction is bounded by TCP_I_LIMIT (35 mm and 8 degrees)."""

HANDOVER = 0.3
"""How long a carry leg takes to ease the last command into its own aim [s] (``Hand.carry``)."""

IK_ITERS = 80
IK_POS_TOL = 0.002
IK_ROT_TOL = math.radians(2.0)


# ------------------------------------------------------------------- maths
def _rotvec(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_rotvec()


def _rotvec_mat(v: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(v).as_matrix()


def _clamp(v: np.ndarray, limit: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v * (limit / n) if n > limit else v


def smooth(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def T_of(pos, quat) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = quat_to_mat(np.asarray(quat, dtype=np.float64))
    M[:3, 3] = pos
    return M


def slerp(R0: np.ndarray, R1: np.ndarray, t: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation, Slerp

    return Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([R0, R1])))([t]).as_matrix()[0]


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# -------------------------------------------------------------------- hand
class Hand:
    """One arm: its commanded joints, its grip command, and a queue of legs."""

    def __init__(self, sim, name: str):
        self.sim = sim
        self.name = name
        self.ik = sim.arm(name)
        self.gripper = sim.gripper(name)
        self.coords = list(self.ik.arm_coords)
        self.slot = int(arm_slot(sim, name))
        self.q = np.asarray(sim.state.joint_q.numpy(), dtype=np.float64).copy()
        self.home = self.q[self.coords].copy()
        self.grip = 0.0
        self.gentle_close = False
        self.close_sign = None
        self.worst_ik = 0.0
        self.bias = np.zeros(3)
        """The plate-position integral of ``carry``: where the arm has to be told to go
        for the plate to end up where it should [m]. Reset per plate."""
        self._last_target = np.zeros(3)
        self.track_err = 0.0
        self.stop = None
        """A callable the carry legs poll after each step; True ends the leg where it is."""
        self.tcp_bias_p = np.zeros(3)
        self.tcp_bias_r = np.zeros(3)
        self._last_cmd = None
        self.worst_ik_rot = 0.0
        self.last_ik = (0.0, 0.0)
        self.script = None
        self.label = "idle"
        hand = sim.scene.embodiment.hands[name]
        self.hand = hand

    # --- kinematics -------------------------------------------------------
    def fk(self, q=None) -> np.ndarray:
        p, quat = self.ik.fk_ee(np.asarray(self.q if q is None else q, dtype=np.float32))
        return T_of(np.asarray(p, float), np.asarray(quat, float))

    def measured(self) -> np.ndarray:
        p, quat = self.ik.ee_pose(self.sim.state)
        return T_of(np.asarray(p, float), np.asarray(quat, float))

    def solve(self, T: np.ndarray, seed=None, iters: int = IK_ITERS, restarts: int = 0) -> tuple[np.ndarray, float, float]:
        quat = mat_to_quat(T[:3, :3])
        base = np.asarray(self.q if seed is None else seed, dtype=np.float64)
        best = None
        rng = np.random.default_rng(len(self.coords))
        for attempt in range(restarts + 1):
            s = base.copy()
            if attempt:
                for c in self.coords:
                    s[c] += rng.uniform(-0.5, 0.5)
            q = np.asarray(self.ik.solve_q_full(s, T[:3, 3], quat, iterations=iters), dtype=np.float64)
            q = self.polish(q, T)
            ep, ea = self.ik.pose_errors(q, T[:3, 3], quat)
            score = ep / IK_POS_TOL + ea / IK_ROT_TOL
            if best is None or score < best[0]:
                best = (score, q, float(ep), float(ea))
            if ep < IK_POS_TOL and ea < IK_ROT_TOL:
                break
        return best[1], best[2], best[3]

    def polish(self, q: np.ndarray, T: np.ndarray, steps: int = 6) -> np.ndarray:
        """Close the full 6-D pose error with damped least squares on the arm's own FK.

        The library solve converges position and the tool axis by descent and
        then sets the roll exactly, and parks 1-1.7 degrees off the target
        orientation however many iterations it is given (measured, 40 to 300)
        -- a residual that changes from step to step, so a hand driven through
        it every control step twitches, and a plate held by its rim is shaken
        loose. A few Gauss-Newton steps on all six coordinates take it out.
        """
        lo, hi = self._limits()
        q = np.asarray(q, dtype=np.float64).copy()
        h = 1.0e-4
        for _ in range(steps):
            F = self.fk(q)
            err = np.r_[T[:3, 3] - F[:3, 3], _rotvec(T[:3, :3] @ F[:3, :3].T)]
            if np.linalg.norm(err[:3]) < 2.0e-5 and np.linalg.norm(err[3:]) < 2.0e-4:
                break
            J = np.empty((6, len(self.coords)))
            for j, c in enumerate(self.coords):
                qp = q.copy()
                qp[c] += h
                Fp = self.fk(qp)
                J[:3, j] = (Fp[:3, 3] - F[:3, 3]) / h
                J[3:, j] = _rotvec(Fp[:3, :3] @ F[:3, :3].T) / h
            W = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])  # metres against radians: 0.1 m per rad
            JW = W @ J
            dq = JW.T @ np.linalg.solve(JW @ JW.T + (1.0e-3) ** 2 * np.eye(6), W @ err)
            for j, c in enumerate(self.coords):
                q[c] = float(np.clip(q[c] + dq[j], lo[j], hi[j]))
        return q

    def _limits(self):
        if not hasattr(self, "_lim"):
            lower = self.sim.model.joint_limit_lower.numpy()
            upper = self.sim.model.joint_limit_upper.numpy()
            dofs = list(self.ik.arm_dofs)
            self._lim = (lower[dofs], upper[dofs])
        return self._lim

    def grasp_R(self, approach, close) -> np.ndarray:
        h = self.hand
        return quat_to_mat(grasp_quat(h.approach_axis, h.close_axis, h.tcp_offset_rot, approach, close))

    # --- legs (generators: one yield per control step) --------------------
    def line(self, T1: np.ndarray, speed: float, label: str, min_steps: int = 10, tolerant: bool = False,
             max_err: float = 0.01):
        """Straight line in the TCP frame, orientation slerped, IK every step.

        ``tolerant`` ends the leg where it is, instead of refusing, when the
        line leaves what the arm can reach -- for a retreat, where any
        distance gained is enough. Otherwise an IK error over ``max_err`` [m]
        refuses.
        """
        T0 = self.fk()
        dist = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
        ang = float(np.arccos(np.clip((np.trace(T0[:3, :3].T @ T1[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
        n = max(min_steps, int(math.ceil(max(dist / speed, ang / 1.6) * HZ)))
        self.label = label
        for k in range(1, n + 1):
            s = smooth(k / n)
            T = np.eye(4)
            T[:3, 3] = T0[:3, 3] + s * (T1[:3, 3] - T0[:3, 3])
            T[:3, :3] = slerp(T0[:3, :3], T1[:3, :3], s)
            q, ep, ea = self.solve(T)
            if ep > 0.004 and tolerant:
                return
            self.worst_ik = max(self.worst_ik, ep)
            if ep > max_err:
                raise RuntimeError(f"{self.name} {label}: IK {ep * 1000:.0f} mm off at step {k}/{n}")
            self.q = q
            yield

    def carry(self, plate_ref, T_plate0: np.ndarray, T_plate1: np.ndarray, speed: float, label: str, T_hand_plate: np.ndarray,
              gain: float = 0.08):
        """Move the *plate* along a straight line, re-reading where it sits in the hand every step.

        The hand's target is the plate's target composed with the plate's grasp
        as measured now (low-passed by ``gain`` per step), so a plate that
        creeps between the fingertips is carried into the slot where it actually is rather
        than where it was when the leg began. Returns the final estimate via
        ``self.hand_plate``.
        """
        from robosimstudio.core.math import mat_to_quat as _q

        dist = float(np.linalg.norm(T_plate1[:3, 3] - T_plate0[:3, 3]))
        ang = float(np.arccos(np.clip((np.trace(T_plate0[:3, :3].T @ T_plate1[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
        n = max(20, int(math.ceil(max(dist / speed, ang / 0.8) * HZ)))
        self.label = label
        est = T_hand_plate.copy()
        # The command must not jump where one leg hands over to the next. A leg
        # aims at the hand's *measured* pose composed with the grasp just re-read,
        # plus a lag correction; the arm had been told something else -- the last
        # leg's aim plus its correction -- and the difference (the arm's lag, or
        # after a wait a correction for a lag that has gone) would arrive as a
        # step along the jaws' axis, shoving one pad into the plate (1.2 mm into
        # its collider at the start of the legs after a wait, measured). So the
        # correction starts again from nothing and the old command is eased into
        # the new aim over ``HANDOVER`` seconds.
        self.tcp_bias_p = np.zeros(3)
        self.tcp_bias_r = np.zeros(3)
        self._last_cmd = None
        T_told = self.fk()
        handover = None
        for k in range(1, n + 61):
            s = smooth(k / n)
            Tp = np.eye(4)
            Tp[:3, 3] = T_plate0[:3, 3] + s * (T_plate1[:3, 3] - T_plate0[:3, 3])
            Tp[:3, :3] = slerp(T_plate0[:3, :3], T_plate1[:3, :3], s)
            T_meas = self.measured()
            T_plate_now = plate_T(plate_ref)
            now = np.linalg.inv(T_meas) @ T_plate_now
            if FEEDBACK["grasp"]:
                est[:3, 3] += gain * (now[:3, 3] - est[:3, 3])
                est[:3, :3] = slerp(est[:3, :3], now[:3, :3], gain)
            # What is left after the grasp is accounted for is the arm not
            # being where it was told: servo lag. A slow integral on the
            # plate's own position error takes it out.
            if k > 1 and FEEDBACK["integral"]:
                self.bias += PLATE_I_GAIN * (self._last_target - T_plate_now[:3, 3])
                norm = float(np.linalg.norm(self.bias))
                if norm > PLATE_I_LIMIT:
                    self.bias *= PLATE_I_LIMIT / norm
            self._last_target = Tp[:3, 3].copy()
            Tc = Tp.copy()
            Tc[:3, 3] += self.bias
            T_goal = Tc @ np.linalg.inv(est)
            # The arm's own lag, measured on the arm: where the hand is against
            # where it was last told to be. Integrated and added to the command,
            # so the servo's standing error (the plate's weight on a soft
            # position loop) is taken out without ever looking at the plate --
            # which is what keeps it from winding up when the plate touches a tine.
            if self._last_cmd is not None:
                self.tcp_bias_p += TCP_I_GAIN * TCP_I_AXES * (self._last_cmd[:3, 3] - T_meas[:3, 3])
                rv = _rotvec(self._last_cmd[:3, :3] @ T_meas[:3, :3].T)
                self.tcp_bias_r += TCP_I_GAIN * rv
                self.tcp_bias_p = _clamp(self.tcp_bias_p, TCP_I_LIMIT[0])
                self.tcp_bias_r = _clamp(self.tcp_bias_r, TCP_I_LIMIT[1])
            self._last_cmd = T_goal.copy()
            T_cmd = T_goal.copy()
            T_cmd[:3, 3] += self.tcp_bias_p
            T_cmd[:3, :3] = _rotvec_mat(self.tcp_bias_r) @ T_goal[:3, :3]
            if handover is None:
                handover = (T_told[:3, 3] - T_cmd[:3, 3], _rotvec(T_told[:3, :3] @ T_cmd[:3, :3].T))
            w = math.exp(-(k - 1) / (HANDOVER * HZ))
            T_cmd[:3, 3] += w * handover[0]
            T_cmd[:3, :3] = _rotvec_mat(w * handover[1]) @ T_cmd[:3, :3]
            self.track_err = max(self.track_err, float(np.linalg.norm(T_meas[:3, 3] - self.fk()[:3, 3])))
            q, ep, ea = self.solve(T_cmd)
            if ea > IK_ROT_TOL or ep > IK_POS_TOL:
                q, ep, ea = self.solve(T_cmd, iters=200, restarts=3)
            if ep > 0.01 or ea > math.radians(6.0):
                # the correction itself may be what left the arm's reach: drop it
                self.tcp_bias_p[:] = 0.0
                self.tcp_bias_r[:] = 0.0
                q, ep, ea = self.solve(T_goal, iters=200, restarts=6)
            self.worst_ik = max(self.worst_ik, ep)
            self.worst_ik_rot = max(self.worst_ik_rot, ea)
            self.last_ik = (ep, ea)
            if ep > 0.01 or ea > math.radians(6.0):
                raise RuntimeError(f"{self.name} {label}: IK {ep * 1000:.0f} mm, {math.degrees(ea):.1f} deg off at step {k}/{n}")
            self.q = q
            yield
            if self.stop is not None and self.stop():
                break
        self.hand_plate = est

    def path(self, Ts: list[np.ndarray], speed: float, label: str):
        """A polyline of TCP poses, each leg a line."""
        for T in Ts:
            yield from self.line(T, speed, label, min_steps=2)

    def joint(self, q_arm, seconds: float, label: str):
        q0 = self.q[self.coords].copy()
        n = max(2, int(seconds * HZ))
        self.label = label
        for k in range(1, n + 1):
            s = smooth(k / n)
            for i, c in enumerate(self.coords):
                self.q[c] = q0[i] + s * (q_arm[i] - q0[i])
            yield

    def set_grip(self, width: float, seconds: float, label: str):
        g0 = float(self.gripper.closed_fraction(self.sim.state))
        self.gentle_close = label == "close"
        g1 = float(self.gripper.command_for_width(max(width, 0.0)))
        n = max(2, int(seconds * HZ))
        self.label = label
        for k in range(1, n + 1):
            self.grip = g0 + smooth(k / n) * (g1 - g0)
            yield

    def wait(self, seconds: float, label: str):
        self.label = label
        for _ in range(max(1, int(seconds * HZ))):
            yield

    def action_into(self, action: np.ndarray) -> None:
        off = self.slot * (len(self.coords) + 1)
        action[off : off + len(self.coords)] = self.q[self.coords]
        command = self.grip
        if self.gentle_close:
            # A small position lead requests a gentle squeeze from the unchanged
            # catalog servo. It is refreshed from the measured jaw gap each step.
            gap = self.gripper.width(self.sim.state)
            command = min(command, float(self.gripper.command_for_width(max(0., gap - .00025))))
        action[off + len(self.coords)] = command


# ------------------------------------------------------------------ plates
def plate_T(ref) -> np.ndarray:
    pose = np.asarray(ref.pose, dtype=np.float64)
    return T_of(pose[:3], pose[3:7])


def grasp_in_plate(close_sign: float, spoke: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The pinch in the plate's own frame: TCP point, approach, closing axis.

    ``GRASP_DEPTH`` down the flange from the lip, approach along the flange going
    in, jaws closing across it -- on the rim ``spoke`` radians round from the
    plate's +y (its top, standing).
    """
    phi = S.RIM["flange_angle"]
    r = S.RIM["R"] - 0.004 - GRASP_DEPTH * math.cos(phi)
    point = S.flange_point(r)
    approach = np.array([0.0, -math.cos(phi), -math.sin(phi)])
    close = close_sign * np.array([0.0, -math.sin(phi), math.cos(phi)])
    c, s = math.cos(spoke), math.sin(spoke)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ point, Rz @ approach, Rz @ close


def rim_width() -> float:
    phi = S.RIM["flange_angle"]
    r = S.RIM["R"] - 0.004 - GRASP_DEPTH * math.cos(phi)
    ths = [S.flange_thickness(r, sp) for sp in np.linspace(-0.3, 0.3, 7)]
    return float(np.nanmax(ths))


class Director:
    """Both hands, the plates, the rack token, the recorder."""

    def __init__(self, sim, plates, layout: S.Layout, args):
        self.sim = sim
        self.plates = plates
        self.L = layout
        self.args = args
        self.hands = {name: Hand(sim, name) for name in ("right", "left")}
        self.idle = np.asarray(sim.idle_action(MODE), dtype=np.float32).copy()
        self.rack_owner: str | None = None
        self.n = 0
        self.events: list[dict] = []
        self.rim = rim_width()
        self.hand_offsets: dict[int, np.ndarray] = {}
        self.watch: dict[int, tuple] = {}
        self.stop_when = None
        self.shelf_staged = set()
        self.slot_of = {i: k for i, k in enumerate(layout.slots)}
        """Plates being inserted: their contacts are logged while they are, if anything but the fingers touches them."""
        print(f"[demo] rim {self.rim * 1000:.2f} mm across at the pinch; flange {math.degrees(S.RIM['flange_angle']):.1f} deg")

    def log(self, **kw):
        kw["step"] = self.n
        kw["t"] = round(self.n / HZ, 3)
        self.events.append(kw)
        if kw.get("leg") == "staged":
            self.shelf_staged.add(kw["plate"])
        print("[demo]", json.dumps(kw), flush=True)

    # --- the script for one plate --------------------------------------
    def plate_script(self, hand: Hand, i: int, slot_k: int):
        sim = self.sim
        ref = self.plates[i]
        side = -1.0 if hand.name == "right" else 1.0
        slot = S.slot_world(slot_k, self.L)
        # the shelf slot, and its twin at the same x in this arm's drying rack
        y_rack = side * self.L.rack_y
        rack = S.slot_world(slot_k, self.L, y_rack)
        y_stage = side * (0.5 * SHELF_WIDTH + S.PLATE_RADIUS + STAGE_OUT)
        lift_foot = S.rest_foot() + LIFT
        set_foot = S.rest_foot() - SET_BELOW

        # 1. pregrasp -- choose the jaw polarity once, by the whole path: the
        # pinch, and where the hand will have to be to put the plate at the lift,
        # the stage, the slot's mouth and the bottom of the slot. A parallel jaw
        # holds the rim either way round, and one way round can be reachable
        # where the other is not.
        Tp = plate_T(ref)
        checkpoints = [S.plate_in_slot(rack, y_rack, lift_foot), S.plate_in_slot(slot, y_stage, lift_foot),
                       S.plate_in_slot(slot, slot["y"], lift_foot), S.plate_in_slot(slot, slot["y"], set_foot)]
        best = None
        lo, hi = hand._limits()
        for spoke_deg in GRASP_SPOKES:
            for sign in (1.0, -1.0):
                point, approach, close = grasp_in_plate(sign, math.radians(spoke_deg))
                a_w, c_w = Tp[:3, :3] @ approach, Tp[:3, :3] @ close
                T = np.eye(4)
                T[:3, :3] = hand.grasp_R(a_w, c_w)
                T[:3, 3] = Tp[:3, :3] @ point + Tp[:3, 3]
                pre = T.copy()
                pre[:3, 3] -= PREGRASP_BACK * a_w
                q_pre, ep1, ea1 = hand.solve(pre, iters=200, restarts=4)
                q_g, ep2, ea2 = hand.solve(T, seed=q_pre, iters=200)
                # the same grasp, carried: the hand pose that puts the plate at
                # each checkpoint, and how close any joint comes to its limit
                T_hp = np.linalg.inv(T) @ Tp
                worst = max(ep1 / IK_POS_TOL + ea1 / IK_ROT_TOL, ep2 / IK_POS_TOL + ea2 / IK_ROT_TOL)
                margin = float(np.min(np.minimum(q_g[hand.coords] - lo, hi - q_g[hand.coords])))
                q = q_g
                for Tc in checkpoints:
                    q, ep, ea = hand.solve(Tc @ np.linalg.inv(T_hp), seed=q, iters=200, restarts=2)
                    worst = max(worst, ep / IK_POS_TOL + ea / IK_ROT_TOL)
                    margin = min(margin, float(np.min(np.minimum(q[hand.coords] - lo, hi - q[hand.coords]))))
                # ... and, having let go at the last of them, backed out clear of the rim
                T_clear = hand.fk(q)
                T_clear[:3, 3] -= RETREAT_CLEAR * T_clear[:3, :3][:, 2]
                q, ep, ea = hand.solve(T_clear, seed=q, iters=200, restarts=2)
                worst = max(worst, ep / IK_POS_TOL + ea / IK_ROT_TOL)
                margin = min(margin, float(np.min(np.minimum(q[hand.coords] - lo, hi - q[hand.coords]))))
                # reachable everywhere first, then enough margin from every joint
                # limit, then the pinch nearest the top
                rank = (0 if worst < 1.0 else 1, max(0.0, MARGIN_ENOUGH - margin) if worst < 1.0 else worst, abs(spoke_deg))
                if best is None or rank < best[0]:
                    best = (rank, sign, T, pre, q_pre, ep1, ep2, worst, margin, spoke_deg)
        _, sign, T_grasp, T_pre, q_pre, ep1, ep2, path_score, margin, spoke_deg = best
        best = (path_score, sign, T_grasp, T_pre, q_pre, ep1, ep2)
        path_score, sign, T_grasp, T_pre, q_pre, ep1, ep2 = best
        hand.close_sign = sign
        hand.bias = np.zeros(3)
        hand.track_err = 0.0
        hand.tcp_bias_p = np.zeros(3)
        hand.tcp_bias_r = np.zeros(3)
        hand._last_cmd = None
        self.log(hand=hand.name, plate=i, leg="plan", slot=slot_k, jaw=sign, spoke_deg=spoke_deg,
                 joint_margin_deg=round(math.degrees(margin), 1), path_ik_score=round(float(path_score), 2),
                 pregrasp_ik_mm=round(ep1 * 1000, 2), grasp_ik_mm=round(ep2 * 1000, 2),
                 approach_tilt_deg=round(math.degrees(math.acos(-T_grasp[:3, :3][:, 2][2])), 1))
        yield from hand.set_grip(self.rim + OPEN_EXTRA, 0.2, "open")
        yield from hand.joint(q_pre[hand.coords], 0.8, "pregrasp")
        yield from hand.line(T_pre, 2 * SPEED["fine"], "pregrasp-settle", min_steps=10)

        # 2. pinch
        yield from hand.line(T_grasp, 2 * SPEED["fine"], "pinch")
        yield from hand.wait(0.15, "pinch-hold")
        sim.sync()
        self.log(hand=hand.name, plate=i, leg="at_pinch", contacts=self.contacts(ref),
                 tcp_err_mm=round(float(np.linalg.norm(hand.measured()[:3, 3] - T_grasp[:3, 3])) * 1000, 2))
        yield from hand.set_grip(self.rim - SQUEEZE, 1.4, "close")
        yield from hand.wait(0.3, "grip")
        sim.sync()
        self.log(hand=hand.name, plate=i, leg="gripped", contacts=self.contacts(ref),
                 grip_closed=round(float(hand.gripper.closed_fraction(sim.state)), 3))

        # the plate as it sits in the hand, before it has been moved at all
        T_hand_plate = np.linalg.inv(hand.measured()) @ plate_T(ref)
        self.watch[i] = (hand, ref, T_hand_plate)

        # Every leg from here moves the *plate*, closed on where it actually is
        # (Hand.carry): the grasp re-read every step, the arm's lag taken out by
        # an integral on the hand's own pose.

        # 3. lift: from where the plate rests (leaning back on the rack's
        # dividers) to upright in the middle of its rack slot, 25 mm up -- off
        # the cradle and clear of both divider rows ...
        T_rest = plate_T(ref)
        T_up = S.plate_in_slot(rack, T_rest[1, 3], lift_foot)
        yield from hand.carry(ref, T_rest, T_up, SPEED["fine"], "lift", T_hand_plate)
        # ... where it hangs its whole weight on the fingertips for the first time and
        # settles in them: read the grasp again before it goes anywhere
        yield from hand.wait(0.25, "hold")
        sim.sync()
        T_hand_plate = np.linalg.inv(hand.measured()) @ plate_T(ref)
        self.log(hand=hand.name, plate=i, leg="lifted", grip_closed=round(float(hand.gripper.closed_fraction(sim.state)), 3),
                 off_target=self._off_target(ref, T_up))
        if np.linalg.norm(plate_T(ref)[:3, 3] - T_up[:3, 3]) > .008:
            raise RuntimeError(f"{ref.name}: plate did not lift in the fingertips")
        # ... and put it back in the middle of its slot as it now sits, before
        # it slides past the dividers
        yield from hand.carry(ref, plate_T(ref), T_up, SPEED["fine"], "centre", T_hand_plate)
        T_hand_plate = hand.hand_plate

        # 4. out of the rack along its slot and across to the shelf's side: one
        # straight slide, since the rack's slot stands at the shelf slot's x
        T_stage = S.plate_in_slot(slot, y_stage, lift_foot)
        # Stay on this arm's side until the opposing wrist has cleared the shelf.
        # The 320 mm outer waypoint permits pickup overlap without forearm contact.
        T_outer = S.plate_in_slot(slot, side * .320, lift_foot)
        yield from hand.carry(ref, plate_T(ref), T_outer, SPEED["move"], "outer-carry", T_hand_plate)
        T_hand_plate = hand.hand_plate
        while self.rack_owner not in (None, hand.name):
            hand.label = "wait-outer"
            yield
        self.rack_owner = hand.name
        yield from hand.carry(ref, plate_T(ref), T_stage, SPEED["move"], "carry", T_hand_plate)
        T_hand_plate = hand.hand_plate

        # wait for the shelf
        while self.rack_owner not in (None, hand.name):
            hand.label = "wait-shelf"
            yield
        self.rack_owner = hand.name
        yield from hand.wait(0.3, "stage")
        sim.sync()
        self.log(hand=hand.name, plate=i, leg="staged", contacts=self.contacts(ref), off_target=self._off_target(ref, T_stage))
        T_hand_plate = np.linalg.inv(hand.measured()) @ plate_T(ref)
        yield from hand.carry(ref, plate_T(ref), T_stage, SPEED["fine"], "stage", T_hand_plate)
        T_hand_plate = hand.hand_plate
        self.watch[i] = (hand, ref, T_hand_plate)

        # 5. into the slot, along it, between the divider rows
        T_mouth = S.plate_in_slot(slot, slot["y"], lift_foot)
        yield from hand.carry(ref, T_stage, T_mouth, SPEED["slide"], "slide", T_hand_plate)
        T_hand_plate = hand.hand_plate

        # 6. down onto the shelf's cradle -- stopping the moment the cradle
        # bears the plate -- then let go
        self.stop_when = (ref, "shelf_cradle", SET_STOP_FORCE)
        T_set = S.plate_in_slot(slot, slot["y"], set_foot)
        T_set[0, 3] += SET_BACK
        yield from hand.carry(ref, T_mouth, T_set, SPEED["set"], "set", T_hand_plate)
        self.stop_when = None
        T_hand_plate = hand.hand_plate
        # ... and let go at once. Held in the jaws with its rim on the cradle,
        # the plate is pinned between the two: the arm's lag goes on pushing it
        # down, the cradle's push has a component along the jaws' axis, and one
        # pad was pressed 2-7 mm into the plate's collider within a quarter of a
        # second (and threw the plate 4 mm when it let go). Opening at once, the
        # fingertips are off it in under a tenth of one.
        sim.sync()
        self.log(hand=hand.name, plate=i, leg="set", contacts=self.contacts(ref), **self._in_slot(ref, slot),
                 tcp_bias_mm=np.round(hand.tcp_bias_p * 1000, 2).tolist(),
                 tcp_bias_deg=round(math.degrees(float(np.linalg.norm(hand.tcp_bias_r))), 2),
                 worst_tracking_mm=round(hand.track_err * 1000, 2))
        self.watch.pop(i, None)
        yield from hand.set_grip(self.rim + OPEN_EXTRA, RELEASE_SECONDS, "release")
        yield from hand.wait(0.5, "settle")
        sim.sync()
        self.log(hand=hand.name, plate=i, leg="released", pose=np.round(plate_T(ref)[:3, 3], 4).tolist(),
                 contacts=self.contacts(ref), **self._in_slot(ref, slot))

        # The fingers are open and the plate has settled. The next arm starts
        # its slow inward carry from ±320 mm while this empty hand retreats.
        self.rack_owner = None

        # 7. retreat: back along the approach until the blades are clear of the
        # rim -- all of it, or the move home drags a fingertip through the plate
        # -- then on and up as far as the arm reaches, then home
        T_clear = hand.fk()
        a_w = T_clear[:3, :3][:, 2]
        T_clear[:3, 3] -= RETREAT_CLEAR * a_w
        yield from hand.line(T_clear, 2 * SPEED["fine"], "retreat", max_err=IK_POS_TOL)
        T_out = T_clear.copy()
        T_out[:3, 3] -= (RETREAT_BACK - RETREAT_CLEAR) * a_w
        T_out[2, 3] += RETREAT_UP
        yield from hand.line(T_out, 2 * SPEED["move"], "retreat", tolerant=True)
        yield from hand.joint(hand.home, 0.8, "home")

    def _in_slot(self, ref, slot: dict) -> dict:
        """Where the plate stands in its slot: how far its mid-depth plane is off the
        slot's centre plane at its foot and at the dividers' tops [mm, + away from the
        robot], its tilt from upright [deg, + top away from the robot], its yaw [deg]."""
        T = plate_T(ref)
        mid = T[:3, 3] + T[:3, :3] @ np.array([0.0, 0.0, 0.5 * S.PLATE_DEPTH])
        up = T[:3, :3][:, 1]
        out = {}
        for name, z in (("foot", T[2, 3] + S.lowest_offset(T[:3, :3]) + 0.01), ("top", S.board_top() + S.DIVIDER[1])):
            t = (z - mid[2]) / up[2]
            out[f"dx_{name}_mm"] = round(float((mid + t * up)[0] - slot["x"]) * 1000, 2)
        out["tilt_deg"] = round(math.degrees(math.atan2(T[0, 1], T[2, 1])), 2)
        out["yaw_deg"] = round(math.degrees(math.atan2(-T[0, 0], abs(T[1, 0]))), 2)
        out["dy_mm"] = round((T[1, 3] - slot["y"]) * 1000, 1)
        return out

    def _bearing(self, ref, other: str, force: float) -> bool:
        return any(c[0].startswith(other) and c[2] >= force for c in self.contacts(ref))

    def contacts(self, ref) -> list:
        """What the plate is touching right now: ``[other shape label, overlap mm, force N]``, strongest first."""
        probe = getattr(self.sim, "contacts", None)
        if probe is None:
            return []
        probe.update()
        mine = set(int(s) for s in ref.shapes)
        if not len(probe.geoms):
            return []
        a, b = probe._shape_pairs()
        labels = self.sim.model.shape_label
        rows = {}
        for k in range(len(a)):
            if int(a[k]) in mine and int(b[k]) not in mine:
                other = int(b[k])
            elif int(b[k]) in mine and int(a[k]) not in mine:
                other = int(a[k])
            else:
                continue
            name = str(labels[other]).split("/")[-1]
            force = float(np.linalg.norm(probe.impulses[k])) / probe.dt
            depth = -float(probe.depths[k]) * 1000.0
            prev = rows.get(name, [name, -1e9, 0.0])
            rows[name] = [name, round(max(prev[1], depth), 3), round(prev[2] + force, 3)]
        return sorted(rows.values(), key=lambda r: -r[2])[:8]

    def _off_target(self, ref, T_target: np.ndarray) -> list[float]:
        """How far the plate is from where it was meant to be: [mm, deg]."""
        T = plate_T(ref)
        d = T_target[:3, :3].T @ T[:3, :3]
        ang = math.degrees(math.acos(np.clip((np.trace(d) - 1.0) / 2.0, -1.0, 1.0)))
        return [round(float(np.linalg.norm(T[:3, 3] - T_target[:3, 3])) * 1000, 2), round(ang, 2)]

    def _slip(self, hand: Hand, ref, T_hand_plate: np.ndarray) -> list[float]:
        """How far the plate has moved in the hand since ``T_hand_plate`` was read: [mm, deg]."""
        now = np.linalg.inv(hand.measured()) @ plate_T(ref)
        d = np.linalg.inv(T_hand_plate) @ now
        ang = math.degrees(math.acos(np.clip((np.trace(d[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
        return [round(float(np.linalg.norm(now[:3, 3] - T_hand_plate[:3, 3])) * 1000, 2), round(ang, 2)]

    def hold_probe(self, hand: Hand, i: int, slot_k: int):
        """Grasp, lift 30 mm straight up, hold 2 s: how well does the pinch hold a still plate?"""
        script = self.plate_script(hand, i, slot_k)
        while True:
            next(script)
            yield
            if hand.label == "grip":
                break
        for _ in range(int(0.3 * HZ) - 1):
            next(script)
            yield
        sim = self.sim
        ref = self.plates[i]
        sim.sync()
        T_hp = np.linalg.inv(hand.measured()) @ plate_T(ref)
        T_up = hand.fk()
        T_up[2, 3] += 0.03
        yield from hand.line(T_up, SPEED["lift"], "probe-lift")
        prev = None
        for k in range(int(self.args.probe_seconds * HZ)):
            hand.label = "probe-hold"
            if k % 12 == 0:
                sim.sync()
                T_m = hand.measured()
                jitter = 0.0 if prev is None else float(np.linalg.norm(T_m[:3, 3] - prev))
                prev = T_m[:3, 3].copy()
                self.log(hand=hand.name, plate=i, leg="probe", k=k, slip=self._slip(hand, ref, T_hp),
                         grip_closed=round(float(hand.gripper.closed_fraction(sim.state)), 4), tcp_move_mm=round(jitter * 1000, 3),
                         contacts=[c for c in self.contacts(ref)][:4])
            yield
        raise RuntimeError("probe done")

    def hand_script(self, hand: Hand, jobs: list[tuple[int, int]]):
        if getattr(self.args, "probe_hold", False) and jobs:
            yield from self.hold_probe(hand, *jobs[0])
        for i, k in jobs:
            yield from self.plate_script(hand, i, k)
        hand.label = "done"
        while True:
            yield

    # --- the loop -------------------------------------------------------
    def run(self, recorder=None, max_steps: int = 20000):
        """Pipeline independent rack pickups; serialize the shelf's insertion corridor."""
        for h in self.hands.values():
            h.stop = lambda h=h: self.stop_when is not None and h.label == "set" and self._bearing(*self.stop_when)
        running = {}
        next_job = 0
        finished = 0
        while finished < len(self.L.slots):
            if next_job < len(self.L.slots):
                arm = self.L.arms[next_job]
                if arm not in running and (next_job == 0 or next_job - 1 in self.shelf_staged):
                    running[arm] = self.plate_script(self.hands[arm], next_job, self.L.slots[next_job])
                    next_job += 1
            for arm, script in list(running.items()):
                try:
                    next(script)
                except StopIteration:
                    del running[arm]
                    self.hands[arm].label = "idle"
                    finished += 1
            self.sim.step(self.action(), mode=MODE)
            self.n += 1
            if recorder: recorder(self)
            if any(float(p.position[2]) < .81 for p in self.plates):
                raise RuntimeError("A plate dropped below its supported height")
            if self.n > max_steps:
                raise RuntimeError("The transfer schedule exceeded its step budget")
            if self.n % 4 == 0: yield
        for _ in range(120):
            self.sim.step(self.action(), mode=MODE)
            self.n += 1
            if recorder: recorder(self)
            if self.n % 4 == 0: yield
        rows = [self._in_slot(self.plates[i], S.slot_world(k, self.L)) for i,k in enumerate(self.L.slots)]
        for i,row in enumerate(rows):
            T = plate_T(self.plates[i])
            tilt = math.degrees(math.asin(min(1.,abs(float(T[2,2])))))
            if tilt > 3 or abs(row["dy_mm"]) > 10 or abs(row["dx_foot_mm"]) > 5:
                raise RuntimeError(f"Plate {i} is not upright in its slot: {row}")
        self.log(leg="success", plates=rows)

    def action(self) -> np.ndarray:
        a = self.idle.copy()
        for h in self.hands.values():
            h.action_into(a)
        return a


# ----------------------------------------------------------------- record
class Recorder:
    def __init__(self, sim, cameras, backend: str | None, every: int = 2):
        self.sim = sim
        self.every = every
        self.body_q: list[np.ndarray] = []
        self.joint_q: list[np.ndarray] = []
        self.labels: list[str] = []
        self.frames: dict[str, list[np.ndarray]] = {c.name: [] for c in cameras}
        self.renderer = sim.make_renderer(backend, cameras=cameras) if backend else None

    def __call__(self, director: Director) -> None:
        if director.n % self.every:
            return
        st = self.sim.state
        self.sim.sync()
        self.body_q.append(st.body_q.numpy().copy())
        self.joint_q.append(st.joint_q.numpy().copy())
        self.labels.append("|".join(f"{n}:{h.label}" for n, h in director.hands.items()))
        if self.renderer is not None:
            fr = self.renderer.render(st)
            for name in self.frames:
                rgb = np.asarray(fr[name]["rgb"])[..., :3]
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb * (255.0 if rgb.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
                self.frames[name].append(rgb)

    def save(self, out: Path, fps: float) -> None:
        np.savez_compressed(out / "take.npz", body_q=np.stack(self.body_q), joint_q=np.stack(self.joint_q), labels=np.array(self.labels), every=self.every)
        if self.renderer is None:
            return
        import imageio.v2 as imageio

        for name, frames in self.frames.items():
            if not frames:
                continue
            w = imageio.get_writer(str(out / f"{name}.mp4"), fps=fps, codec="libx264", quality=8, macro_block_size=1, pixelformat="yuv420p")
            for f in frames:
                w.append_data(f)
            w.close()
            print(f"[demo] wrote {name}.mp4: {len(frames)} frames")
        self.renderer.close()


def cameras(scale: float = 1.0):
    from robosimstudio.core.cameras import CameraSpec

    w, h = int(1280 * scale), int(720 * scale)
    return [
        # the whole bench, from the far front corner: both arms, both stands, the rack
        CameraSpec(name="front", width=w, height=h, fov_y=55.0, eye=(1.40, -0.78, 1.38), look_at=(0.52, 0.0, 0.86), near=0.01, far=6.0),
        # low, square to the slots: each plate going in between the wires, edge-on
        CameraSpec(name="side", width=w, height=h, fov_y=36.0, eye=(0.56, -0.62, 1.02), look_at=(0.56, 0.0, 0.86), near=0.01, far=6.0),
        # the rack close, from beyond it
        CameraSpec(name="rack", width=w, height=h, fov_y=36.0, eye=(0.98, 0.40, 1.10), look_at=(0.55, -0.02, 0.84), near=0.01, far=6.0),
        # straight down over the robot: the plates' faces, once the arms are home
        CameraSpec(name="top", width=w, height=h, fov_y=38.0, eye=(0.22, -0.02, 1.40), look_at=(0.56, 0.0, 0.84), near=0.01, far=6.0),
        # the filled shelf square to its slots, one plate in each, upright
        CameraSpec(name="slots", width=w, height=h, fov_y=30.0, eye=(0.55, -0.62, 1.00), look_at=(0.55, 0.0, 0.85), near=0.01, far=6.0),
    ]


