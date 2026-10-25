# SPDX-License-Identifier: Apache-2.0
"""Whether a twin's physics is real: its own routine replayed with the robot changed, and what moved read off it.

A twin's picture can match its clip while the physics does not hold: a thing
that starts inside a finger and is pushed out, a "grasp" that is the hand
wedging the thing at 700 N, a lift that works with the fingers held open, a
grip that only holds at a friction no rubber has. None of that shows in a
still or a take. So the routine is played again, five ways (:data:`MODES`):

- ``written``: as it is, every step logged (what moved, the force each hand
  puts on it, overlaps);
- ``frozen``: every arm and finger target held where it started. Nothing may
  move: what does is pushed by something other than the robot, or started
  inside something;
- ``open``: the fingers held fully open, the arms as written. A pinch task
  fails; one that still succeeds is carried by something else (a support, a
  hook, two arms squeezing), and the report has to say what;
- ``built``: the robot as built -- every contact friction at most
  :data:`REAL_MU`, the arm's torque capped at the real arm's
  (:func:`real_arm_torque`). The routine still finishes;
- ``heavy``: as built, and every handled thing twice as heavy: the grasp's
  margin, a warning when it fails rather than a fault.

**One door.** Every routine moves the robot by writing ``joint_target_q``:
the arm panels' IK and the gripper write it just before the page steps, a
routine that steps the scene itself writes it through ``sim.step``. Both
step through ``EnvBase._step_once``, so :class:`Replay` holds the targets in
a pre-step hook there and logs in a post-step hook, and never needs to know
how a routine was written. Code that calls ``pipeline.step()`` itself goes
round that door and is not held: a routine is not written that way, and a
settle a material runs while the scene is built is not a routine's step.

**Free flight** is judged on nothing being within the contact margin, not on
the contact force: contacts are read after the last substep, and one that
comes and goes inside a step reads as none, while the step's velocity change
still carries it.
"""

from __future__ import annotations

import math
import re
import traceback
from collections.abc import Iterator
from typing import Any, ClassVar

import numpy as np
import warp as wp

from robosimstudio.engines.pose import compose_many

MODES = ("written", "frozen", "open", "built", "heavy")

REAL_MU = 1.0
"""The most friction a real contact here has: rubber pads on plastic, ceramic or metal run 0.5 to 1.0."""

OVERRUN = 0.1
"""How far past the written run's length a replay whose routine is still going may run, as a share of it."""

SAMPLE_EVERY = 5
"""Every how many steps a replay reads the jaws and the particles (contacts and poses it reads every step)."""

HEAVY = 2.0
"""How much heavier ``heavy`` makes every handled thing."""

STOPPED = "stopped before it ended (a reset, a save or Stop)"
"""The outcome of a replay the page stopped: not a reading, so ``./ctl physics`` plays it again."""

FAILED = "the replay itself failed"
"""The start of the outcome of a replay that failed in its own code (not the routine's): not a reading either."""

SETTLE_MM = 2.0
"""How far a thing may move with the robot frozen: a resting contact settles by a fraction of a millimetre."""

# The real arms' joint torque limits [Nm], from their makers' data sheets, by a word in the robot's name or its own
# joints' names: a robot file's own numbers are a claim (the bimanual YAM's are right; a UR5e built for a twin gave
# its wrists 150). A Universal Robots arm whose size the names do not give is left uncapped rather than guessed.
ARM_TORQUE = {
    "ur3": (56.0, 56.0, 28.0, 12.0, 12.0, 12.0),
    "ur5": (150.0, 150.0, 150.0, 28.0, 28.0, 28.0),
    "ur10": (330.0, 330.0, 150.0, 56.0, 56.0, 56.0),
    "ur16": (330.0, 330.0, 150.0, 56.0, 56.0, 56.0),
    "yam": (27.0, 27.0, 27.0, 10.0, 10.0, 10.0),
    "panda": (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
    "fr3": (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
}

_SHAPE_PROPERTIES = 1 << 4  # newton.ModelFlags, spelled as bits as robosimstudio.core.tuning does
_BODY_INERTIAL = 1 << 3
_JOINT_DOF = 1 << 1


# --------------------------------------------------------------------------- what the scene holds


def robot_name(sim: Any) -> str:
    spec = getattr(getattr(sim.scene, "embodiment", None), "spec", None)
    return str(getattr(spec, "name", "") or "")


def _robot_bodies(sim: Any) -> set[int]:
    """Every body of the robot, shapeless links (a flange, a tool frame) included."""
    m = sim.model
    owners = {
        int(b) for b in np.unique(m.shape_body.numpy()[np.asarray(sim.scene.handles.robot_shapes, dtype=np.int64)])
    }
    own = getattr(getattr(sim.scene, "embodiment", None), "bodies", None)
    return owners | ({int(b) for b in own} if own is not None else set())


def _robot_joint_labels(sim: Any) -> list[str]:
    """The labels of the robot's own joints: a thing's joint named after a robot must not name the arm."""
    m = sim.model
    labels = list(getattr(m, "joint_label", []) or [])
    if not labels:
        return []
    robot = _robot_bodies(sim)
    child = m.joint_child.numpy()
    return [str(labels[j]) for j in range(min(len(labels), len(child))) if int(child[j]) in robot]


def real_arm_torque(sim: Any) -> tuple[float, ...] | None:
    """The real arm's torque limits, per joint from the base, or ``None`` for a robot not in :data:`ARM_TORQUE`."""
    words = f"{robot_name(sim)} {' '.join(_robot_joint_labels(sim))}".lower()
    for key in sorted(ARM_TORQUE, key=len, reverse=True):
        if re.search(rf"{key}(?![0-9])", words):  # a UR3 is not a UR30
            return ARM_TORQUE[key]
    return None


def layout(sim: Any) -> dict[str, np.ndarray]:
    """Where each arm's and each finger's target sits in ``joint_target_q``, and the fingers' open targets.

    Newton sizes the vector by DOFs unless coordinate-layout targets are on, and
    the two differ for any scene with a free body (7 coordinates, 6 DOFs).
    """
    m, h = sim.model, sim.scene.handles
    n = int(sim.control.joint_target_q.shape[0])
    by_coord = n == m.joint_coord_count and n != m.joint_dof_count
    arm, grip, opened = [], [], []
    for a in h.arms.values():
        arm += list(a.arm_coords if by_coord else a.arm_dofs)
        g = list(a.gripper_coords if by_coord else a.gripper_dofs)
        grip += g
        opened += list(a._finger_limits(len(g))[0][: len(g)])  # a mirrored pair opens at (+w, -w), not at 0
    # every target the robot has, a torso, a rail or a head too: `frozen` holds them all
    robot = _robot_bodies(sim)
    child, q0, qd0 = m.joint_child.numpy(), m.joint_q_start.numpy(), m.joint_qd_start.numpy()
    starts = q0 if by_coord else qd0
    rows: list[int] = []
    for j in range(len(child)):
        if int(child[j]) in robot:
            end = int(starts[j + 1]) if j + 1 < len(starts) else n
            rows += list(range(int(starts[j]), min(end, n)))
    return {
        "robot": np.asarray(sorted(set(rows) | set(arm) | set(grip)), dtype=np.int64),
        "arm": np.asarray(arm, dtype=np.int64),
        "grip": np.asarray(grip, dtype=np.int64),
        "open": np.asarray(opened, dtype=np.float64),
    }


def things(sim: Any) -> dict[str, dict[str, Any]]:
    """Every dynamic thing that is not the robot: its bodies and its shapes."""
    out = {}
    for name, e in sim.scene.handles.objects.items():
        bodies = [int(b) for b in (e.bodies or (e.body,)) if int(b) >= 0]
        if bodies and not e.static:
            out[name] = {"bodies": bodies, "shapes": np.asarray(e.shapes, dtype=np.int64)}
    return out


def fingers(sim: Any) -> dict[str, np.ndarray]:
    return {
        a: np.asarray(sorted({int(s) for g in arm.finger_groups for _, sh in g for s in sh}), dtype=np.int64)
        for a, arm in sim.scene.handles.arms.items()
    }


def model_facts(sim: Any) -> dict[str, Any]:
    """What the model holds that the contact rule is about, and the numbers the robot runs on.

    A joint or equality from a thing's body to the robot is an attachment
    whatever it is called (``ties``); a thing welded to the world or to another
    thing is noted for the report to explain (``welds``), while a hinge or a
    slide to the world or to furniture is a mechanism (a lid, a drawer) and is
    not. A stiffness on a thing's own joint (``drives``) is its elasticity (a
    rod's bending) while its target stays put, and a motor once something moves
    the target: :class:`Replay` watches those targets (``drive_rows``) and the
    verdict reads the written run's. An infinite mass is a kinematic thing. The friction is the one
    MuJoCo mixes at a finger contact (the larger of the two surfaces'), which is
    not the number a twin wrote for its thing when the hand's own pads are
    grippier.
    """
    m = sim.model
    mu = m.shape_material_mu.numpy()
    mass = m.body_mass.numpy()
    th, fing = things(sim), fingers(sim)
    fixed_body = _fixed_bodies(m)
    owner = {b: name for name, t in th.items() for b in t["bodies"]}
    robot_bodies = {b for b in _robot_bodies(sim) if b >= 0}
    out: dict[str, Any] = {
        "robot": robot_name(sim),
        "real_arm_torque": real_arm_torque(sim),
        "things": {
            name: {
                "mass_kg": round(float(sum(mass[b] for b in t["bodies"])), 4),
                "bodies": len(t["bodies"]),
                "mu": sorted({round(float(v), 3) for v in mu[t["shapes"]]}) if len(t["shapes"]) else [],
                "kinematic": bool(any(fixed_body[b] for b in t["bodies"])),
            }
            for name, t in th.items()
        },
        "finger_mu": {a: sorted({round(float(v), 3) for v in mu[s]}) for a, s in fing.items()},
    }
    kinds = _joint_kinds()
    free, fixed_kind = kinds.get("free", -1), kinds.get("fixed", -2)
    kinds = {v: k for k, v in kinds.items()}
    jt, jp, jc = m.joint_type.numpy(), m.joint_parent.numpy(), m.joint_child.numpy()
    qd0 = m.joint_qd_start.numpy()
    ke = m.joint_target_ke.numpy()
    n_targets = int(sim.control.joint_target_q.shape[0])
    by_coord = n_targets == m.joint_coord_count and n_targets != m.joint_dof_count
    q0 = m.joint_q_start.numpy()
    ties, welds, drives, drive_rows = [], [], [], []
    for j in range(len(jt)):
        child, parent = int(jc[j]), int(jp[j])
        if parent in owner and child in robot_bodies:  # a robot link hung off a thing: the same tie the other way
            ties.append(f"{owner[parent]} by a {kinds.get(int(jt[j]), 'joint')} joint to the robot")
            continue
        if child not in owner or int(jt[j]) == free:
            continue
        thing, kind = owner[child], kinds.get(int(jt[j]), "joint")
        if parent in robot_bodies:
            ties.append(f"{thing} by a {kind} joint to the robot")
            continue
        if int(jt[j]) == fixed_kind and owner.get(parent) != thing:
            welds.append(f"{thing} welded to {'the world' if parent == -1 else owner.get(parent, f'body {parent}')}")
        end = int(qd0[j + 1]) if j + 1 < len(qd0) else int(m.joint_dof_count)
        stiff = [d for d in range(int(qd0[j]), end) if ke[d] > 0]
        if stiff:
            drives.append(f"{thing}'s {kind} joint {j} (ke {float(max(ke[d] for d in stiff)):g})")
            # where the target of each stiff dof sits in joint_target_q (a hinge's or a slide's one coordinate is its
            # dof's; a ball joint's four coordinates do not line up with its three dofs, and are read as they fall)
            drive_rows += [int(q0[j]) + (d - int(qd0[j])) if by_coord else d for d in stiff]
    if getattr(m, "equality_constraint_count", 0):
        b1, b2 = m.equality_constraint_body1.numpy(), m.equality_constraint_body2.numpy()
        for k in range(int(m.equality_constraint_count)):
            pair = {int(b1[k]), int(b2[k])}
            hit = next((owner[b] for b in pair if b in owner), None)
            if hit and pair & robot_bodies:
                ties.append(f"{hit} by equality constraint {k} to the robot")
            elif hit:
                welds.append(f"{hit} by equality constraint {k}")
    out["ties"] = ties
    out["welds"] = welds
    out["drives"] = drives
    out["drive_rows"] = drive_rows
    out["kinematic"] = [name for name, t in out["things"].items() if t["kinematic"]]
    return out


# --------------------------------------------------------------------------- the replay


def _fixed_bodies(model: Any) -> np.ndarray:
    """Per body, whether nothing but a script moves it: an infinite mass, or Newton's KINEMATIC flag."""
    fixed = model.body_inv_mass.numpy() == 0.0
    flags = getattr(model, "body_flags", None)
    if flags is not None:
        try:
            import newton

            fixed |= (flags.numpy() & int(newton.BodyFlags.KINEMATIC)) != 0
        except Exception:  # pragma: no cover - a Newton without the flag
            pass
    return fixed


def _joint_kinds() -> dict[str, int]:
    """Newton's joint types by lower-case name (``free``, ``fixed``, ``revolute``, ...)."""
    try:
        import newton

        return {k.lower(): int(v) for k, v in newton.JointType.__members__.items()}
    except Exception:  # pragma: no cover - newton is always there on a page
        return {}


def _turn_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    d = abs(float(np.dot(q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1))))
    return math.degrees(2 * math.acos(min(1.0, d)))


class Broken(BaseException):
    """The replay's own hold failed inside a step: a broken replay, not a reading of the routine."""


@wp.kernel
def _write_rows(rows: wp.array(dtype=wp.int32), values: wp.array(dtype=wp.float32), target: wp.array(dtype=wp.float32)):
    i = wp.tid()
    target[rows[i]] = values[i]


class Overrun(BaseException):
    """A replay's routine was still going well past the written run's length: raised from each step past it.

    A ``BaseException``, and raised again on every step until the replay
    clears it, so a routine that catches ``Exception`` around its own stepping
    cannot swallow it and loop on.
    """


class Replay:
    """Holds the robot's targets for ``frozen`` and ``open`` before each step (a pre-step hook), logs after (a post).

    Logged per step: each thing's centre of mass and turn (every body of a
    jointed thing), the force each hand's fingers and the robot as a whole put
    on it, whether anything is within the contact margin of it, the deepest
    overlap, each deformable's particles, the jaw widths.
    """

    def __init__(self, sim: Any, mode: str) -> None:
        self.sim, self.mode = sim, mode
        self.lay, self.th, self.fing = layout(sim), things(sim), fingers(sim)
        self.dt = 1.0 / float(sim.request.physics.control_hz)
        n = int(sim.model.shape_count)
        self.names = list(self.th)
        self.owner = np.full(n, -1, dtype=np.int64)
        for i, name in enumerate(self.names):
            self.owner[self.th[name]["shapes"]] = i
        free_kind = _joint_kinds().get("free", -1)
        jt, jc = sim.model.joint_type.numpy(), sim.model.joint_child.numpy()
        free_bodies = {int(c) for t, c in zip(jt, jc, strict=False) if int(t) == free_kind}
        # a thing on a hinge, or one a cloth or a soft body can push (their contacts are not MuJoCo's), does not fall
        # at g when nothing in MuJoCo's list touches it: free flight is read only where it means something
        soft_scene = bool(sim.particle_probes())
        self.free = {
            name
            for name in self.names
            if not soft_scene and len(self.th[name]["bodies"]) == 1 and self.th[name]["bodies"][0] in free_bodies
        }
        self.parts = [(f"{name}#{k}", b) for name in self.names for k, b in enumerate(self.th[name]["bodies"][1:], 1)]
        self.arms = list(self.fing)
        self.finger_of = np.full(n, -1, dtype=np.int64)
        for i, a in enumerate(self.arms):
            self.finger_of[self.fing[a]] = i
        self.robot = np.zeros(n, dtype=bool)
        self.robot[np.asarray(sim.scene.handles.robot_shapes, dtype=np.int64)] = True
        com = sim.model.body_com.numpy()
        self.pose_names = self.names + [p for p, _ in self.parts]
        self.pose_bodies = np.asarray(
            [self.th[n]["bodies"][0] for n in self.names] + [b for _, b in self.parts], dtype=np.int64
        )
        self.com_local = np.zeros((len(self.pose_bodies), 7))
        self.com_local[:, :3] = com[self.pose_bodies]
        self.com_local[:, 6] = 1.0
        self.rows: list[dict] = []
        self.soft0: dict[str, np.ndarray] = {}
        self.mu_seen: list[float] = []
        self.steps = 0
        self.error = ""
        self.start: np.ndarray | None = None
        self.held: np.ndarray | None = None
        """The target rows ``frozen`` or ``open`` hold, and what they are held at (``held_to``)."""
        self.held_to: np.ndarray | None = None
        self.hooks0: set[int] = set()
        self.lengths0: tuple[int, ...] = ()
        self.hooks_added = 0
        """Post-solve hooks found on the pipeline beyond those it had at the start: a dragger or a pin attached."""
        self.limit = 0
        """Steps after which a routine that steps the scene itself is stopped, raised from the step; 0 never."""
        self.drive_rows = np.asarray(model_facts(sim)["drive_rows"] if mode == "written" else [], dtype=np.int64)
        """The ``joint_target_q`` rows of the stiff joints on things (:func:`model_facts`'s ``drive_rows``), watched
        on the written replay, the one the verdict reads them off."""
        self.drive0: np.ndarray | None = None
        self.drive_moved = 0.0
        """The most any of those targets moved from where it started: a motor on a thing, if above zero."""
        self.broken = ""
        """Why the replay's own hold failed, if it did: the replay is then not a reading."""
        self.inside = False
        """Whether the step now running was taken inside the routine's own ``next()`` (so may raise into it)."""

    def install(self) -> None:
        """Read the start (the first row: what a push-out in the first step moves is measured from it), then hook in."""
        self.start = self.sim.control.joint_target_q.numpy().copy()
        if self.mode == "frozen":
            self.held = self.lay["robot"]
            self.held_to = self.start[self.held]
        elif self.mode == "open":
            self.held, self.held_to = self.lay["grip"], self.lay["open"]
        self._hold_hook, self._after_hook = self._hold, self._after  # one object each, to find and remove
        self.hooks0 = self._hooks()
        if len(self.drive_rows):
            self.drive0 = self.start[self.drive_rows].copy()
        self.log(contacts=False)  # the reset stepped nothing: the probe still holds the last replay's contacts
        if self.held is not None:
            self._held_rows = wp.array(self.held.astype(np.int32), dtype=wp.int32, device=self.sim.model.device)
            self._held_values = wp.array(
                np.asarray(self.held_to, dtype=np.float32), dtype=wp.float32, device=self.sim.model.device
            )
            self.sim.pre_step_hooks.append(self._hold_hook)
        self.sim.post_step_hooks.append(self._after_hook)
        self.lengths0 = self._hook_lengths()

    def _hook_lists(self) -> list:
        phases = getattr(self.sim.pipeline, "hooks", None)
        lists = [getattr(phases, phase, ()) for phase in getattr(phases, "PHASES", ())]
        return [*lists, self.sim.pre_step_hooks, self.sim.post_step_hooks]

    def _hook_lengths(self) -> tuple[int, ...]:
        return tuple(len(hooks) for hooks in self._hook_lists())

    def _hooks(self) -> set[int]:
        """Every hook on the scene now: each substep phase's, and the env's pre- and post-step lists."""
        return {id(fn) for hooks in self._hook_lists() for fn in hooks}

    def remove(self) -> None:
        self.limit = 0
        for hooks, hook in (
            (self.sim.pre_step_hooks, getattr(self, "_hold_hook", None)),
            (self.sim.post_step_hooks, getattr(self, "_after_hook", None)),
        ):
            if hook is not None and any(h is hook for h in hooks):  # a reset may have retired it already
                hooks.remove(hook)

    def _hold(self, _env: Any = None) -> None:
        try:
            wp.launch(
                _write_rows,
                dim=len(self.held),
                inputs=[self._held_rows, self._held_values],
                outputs=[self.sim.control.joint_target_q],
                device=self.sim.model.device,
            )
        except Exception as exc:  # the replay's own fault: not a reading the routine's checks made
            self.broken = self.broken or f"holding the robot's targets: {type(exc).__name__}: {exc}"
            if self.inside:  # a step the routine took itself; one the page takes must not raise into the page
                raise Broken(self.broken) from exc

    def _after(self, _env: Any = None) -> None:
        self.steps += 1
        if self.limit and self.inside and self.steps >= self.limit:
            raise Overrun(self.steps)
        try:
            self.log()
        except Exception:
            self.error = self.error or traceback.format_exc()[-800:]

    def log(self, contacts: bool = True) -> None:
        sim = self.sim
        bq, bqd = sim.state.body_q.numpy(), sim.state.body_qd.numpy()
        n = len(self.names)
        hand = np.zeros((n, len(self.arms)))
        robot, near, depth_robot = np.zeros(n), np.zeros(n), np.zeros(n)
        probe = getattr(sim, "contacts", None)
        if contacts and self._hook_lengths() != self.lengths0:
            mine = {id(getattr(self, "_hold_hook", None)), id(getattr(self, "_after_hook", None))}
            self.hooks_added = max(self.hooks_added, len(self._hooks() - self.hooks0 - mine))
        if probe is not None and n and contacts:
            probe.update(sim.step_count)  # the step's read, shared with a panel reading the same frame
            if len(probe.geoms):
                a, b = probe._shape_pairs()
                force = np.linalg.norm(probe.impulses, axis=1) / probe.dt
                depth = np.maximum(-np.asarray(probe.depths, dtype=np.float64), 0.0)
                mapped = (a >= 0) & (b >= 0)  # a geom with no Newton shape reads -1, which numpy would wrap
                a, b, force, depth = a[mapped], b[mapped], force[mapped], depth[mapped]
                for x, y in ((a, b), (b, a)):
                    ox = self.owner[x]
                    real = (ox >= 0) & (self.owner[y] != ox)  # a thing's own shapes touching are not a contact
                    np.add.at(near, ox[real], 1)
                    on_robot = real & self.robot[y]
                    np.add.at(robot, ox[on_robot], force[on_robot])
                    np.maximum.at(depth_robot, ox[on_robot], depth[on_robot])
                    on_finger = real & (self.finger_of[y] >= 0)
                    np.add.at(hand, (ox[on_finger], self.finger_of[y][on_finger]), force[on_finger])
                if self.steps % 10 == 0:
                    self._mixed_friction(probe)
        rows = self.pose_bodies
        com = compose_many(bq[rows], self.com_local)[:, :3]  # each body's centre of mass, not its origin
        pose = {name: [*com[i], *bq[b, 3:]] for i, (name, b) in enumerate(zip(self.pose_names, rows, strict=True))}
        vel = {name: bqd[self.th[name]["bodies"][0], :3].tolist() for name in self.names}
        soft, jaw = {}, {}
        sampled = self.steps % SAMPLE_EVERY == 0
        if sampled and self.drive0 is not None:
            moved = np.abs(sim.control.joint_target_q.numpy()[self.drive_rows] - self.drive0)
            self.drive_moved = max(self.drive_moved, float(moved.max()))
        for name, p in (sim.particle_probes() or {}).items() if sampled else ():
            try:
                p.update(sim.state, None)
                pts = np.asarray(p.points())
            except Exception:
                continue
            if len(pts):
                base = self.soft0.setdefault(name, pts.copy())
                away = np.linalg.norm(pts - base, axis=1)
                soft[name] = [*pts.mean(0).tolist(), float(away.max()), float(np.median(away))]
        for a in self.arms if sampled else ():
            try:
                jaw[a] = float(sim.gripper(a).width(sim.state))
            except Exception:
                pass
        self.rows.append(
            {
                "pose": pose,
                "vel": vel,
                "hand": hand.tolist(),
                "robot": robot.tolist(),
                "near": near.tolist(),
                "depth": depth_robot.tolist(),
                "soft": soft,
                "jaw": jaw,
            }
        )

    def _mixed_friction(self, probe: Any) -> None:
        """The friction MuJoCo used at this step's finger contacts: a sample every ten steps is plenty."""
        try:
            data = probe.solver.mjw_data
            count = int(np.ravel(data.nacon.numpy())[0])
            if not count:
                return
            g = data.contact.geom.numpy()[:count]
            mu = data.contact.friction.numpy()[:count, 0]
            sa, sb = probe.geom_to_shape[g[:, 0]], probe.geom_to_shape[g[:, 1]]
            mapped = (sa >= 0) & (sb >= 0)
            sa, sb, mu = sa[mapped], sb[mapped], mu[mapped]
            hit = ((self.finger_of[sa] >= 0) & (self.owner[sb] >= 0)) | (
                (self.finger_of[sb] >= 0) & (self.owner[sa] >= 0)
            )
            if hit.any():
                self.mu_seen += [float(mu[hit].min()), float(mu[hit].max())]
        except Exception:
            pass

    def summary(self, outcome: str) -> dict[str, Any]:
        rows = self.rows
        out: dict[str, Any] = {
            "outcome": outcome,
            "steps": self.steps,
            "log_error": self.error,
            "hooks_added": self.hooks_added,
            "thing_targets_moved": round(self.drive_moved, 6),
            "things": {},
        }
        if not rows:
            return out
        for name in self.names + [p for p, _ in self.parts]:
            P = np.array([r["pose"][name] for r in rows])
            d = np.linalg.norm(P[:, :3] - P[0, :3], axis=1)
            t = {
                "moved_max_mm": round(float(d.max()) * 1000, 1),
                "end_mm": round(float(d[-1]) * 1000, 1),
                "end": [round(float(v), 4) for v in P[-1, :3]],
                "rise_max_mm": round(float((P[:, 2] - P[0, 2]).max()) * 1000, 1),
                "turn_end_deg": round(_turn_deg(P[0, 3:], P[-1, 3:]), 1),
                "end_quat": [round(float(v), 5) for v in P[-1, 3:]],
            }
            if name in self.names:
                i = self.names.index(name)
                hand = np.array([r["hand"][i] for r in rows])
                t["hand_force_p50_p95_N"] = [
                    [round(float(np.percentile(c[c > 0.5], q)), 1) for q in (50, 95)] if (c > 0.5).any() else [0.0, 0.0]
                    for c in hand.T
                ]
                t["overlap_with_robot_max_mm"] = round(float(max(r["depth"][i] for r in rows)) * 1000, 2)
                if name in self.free and t["moved_max_mm"] > 5.0:
                    stepped = rows[1:]  # row 0 is the start, read without contacts
                    near = np.array([r["near"][i] for r in stepped]) == 0
                    v = np.array([r["vel"][name] for r in stepped])
                    free = near[1:-1] & near[:-2] & near[2:] & (np.linalg.norm(v[1:-1], axis=1) > 0.01)
                    acc = (v[2:] - v[1:-1]) / self.dt
                    g = np.ravel(self.sim.model.gravity.numpy())[:3]
                    off = np.linalg.norm(acc - g, axis=1)
                    t["free_flight_steps"] = int(free.sum())
                    t["free_flight_not_falling"] = int((free & (off > 3.0)).sum())
            out["things"][name] = t
        soft = {}
        for name in {k for r in rows for k in r["soft"]}:
            S = np.array([r["soft"][name] for r in rows if name in r["soft"]])
            c = S[:, :3]
            soft[name] = {
                "moved_max_mm": round(float(np.linalg.norm(c - c[0], axis=1).max()) * 1000, 1),
                "end_mm": round(float(np.linalg.norm(c[-1] - c[0])) * 1000, 1),
                "end": [round(float(v), 4) for v in c[-1]],
                "particle_moved_max_mm": round(float(S[:, 3].max()) * 1000, 1),
                # how far its particles end from where they started: a loop seated beside its start has its centre
                # near where it began, and its particles moved
                "particle_end_mm": round(float(S[-1, 4]) * 1000, 1),
            }
        out["deformables"] = soft
        out["jaw_mm"] = {
            a: [round(min(w) * 1000, 1), round(max(w) * 1000, 1)]
            for a in self.arms
            if (w := [r["jaw"][a] for r in rows[50:] if a in r["jaw"]] or [r["jaw"][a] for r in rows if a in r["jaw"]])
        }
        if self.mu_seen:
            out["finger_mu"] = [round(min(self.mu_seen), 3), round(max(self.mu_seen), 3)]
        return out


class AsBuilt:
    """The robot as built, for the length of a replay: friction at most :data:`REAL_MU`, the real arm's torque.

    ``heavier`` also scales every handled thing's mass and inertia; a thing with
    an infinite mass (kinematic, or welded) is left so. Only the entries changed
    are saved, and :meth:`restore` writes those back into the arrays as they are
    then, so a slider moved during the replay keeps what it set elsewhere; the
    solvers are told once each way, as :mod:`robosimstudio.core.tuning` tells
    them for a slider. :meth:`restore` puts back whatever :meth:`apply` got as
    far as changing.

    Not :class:`~robosimstudio.core.tuning.BodyWriter`: that sets one entity's
    mass to a value and holds only a massless body fixed, where this scales
    every handled thing at once, must also leave a body flagged ``KINEMATIC``
    (which keeps its mass) alone, and must put back exactly the rows it wrote
    rather than divide its way back.
    """

    MASS: ClassVar[tuple[str, ...]] = ("body_mass", "body_inv_mass", "body_inertia", "body_inv_inertia")
    FLAGS: ClassVar[dict[str, int]] = {"shape_material_mu": _SHAPE_PROPERTIES, "joint_effort_limit": _JOINT_DOF}

    def __init__(self, sim: Any, heavier: float = 1.0) -> None:
        self.sim, self.heavier = sim, float(heavier)
        self.saved: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        """Per array: the rows changed, and what they held."""

    def _change(self, name: str, rows: np.ndarray, new: np.ndarray) -> None:
        from robosimstudio.core import tuning

        array = getattr(self.sim.model, name)
        values = array.numpy()
        self.saved[name] = (rows, values[rows].copy())
        values[rows] = new
        array.assign(values)
        tuning.reach_island_copies(self.sim.pipeline, name)  # an island that keeps its own copy sees it too

    def apply(self) -> None:
        from robosimstudio.core import tuning

        m = self.sim.model
        mu = m.shape_material_mu.numpy()
        grippy = np.flatnonzero(mu > REAL_MU)
        flags = _SHAPE_PROPERTIES if len(grippy) else 0
        if len(grippy):
            self._change("shape_material_mu", grippy, np.full(len(grippy), REAL_MU, dtype=mu.dtype))
        limits = real_arm_torque(self.sim)
        if limits is not None:
            effort = m.joint_effort_limit.numpy()
            dofs = [d for arm in self.sim.scene.handles.arms.values() for d in arm.arm_dofs[: len(limits)]]
            caps = [
                limits[i]
                for arm in self.sim.scene.handles.arms.values()
                for i in range(len(arm.arm_dofs[: len(limits)]))
            ]
            rows = np.asarray(dofs, dtype=np.int64)
            self._change("joint_effort_limit", rows, np.minimum(effort[rows], np.asarray(caps, dtype=effort.dtype)))
            flags |= _JOINT_DOF
        if self.heavier != 1.0:
            fixed = _fixed_bodies(m)
            rows = np.asarray(
                [b for t in things(self.sim).values() for b in t["bodies"] if not fixed[b]], dtype=np.int64
            )
            if len(rows):  # a body of infinite mass stays so: giving it an inverse mass would set it free
                for name in self.MASS:
                    scale = self.heavier if name in ("body_mass", "body_inertia") else 1.0 / self.heavier
                    self._change(name, rows, getattr(m, name).numpy()[rows] * scale)
                flags |= _BODY_INERTIAL
        if flags:
            tuning.notify(self.sim.pipeline, flags)

    def restore(self) -> None:
        from robosimstudio.core import tuning

        flags = 0
        for name, (rows, old) in self.saved.items():
            array = getattr(self.sim.model, name)
            values = array.numpy()
            values[rows] = old
            array.assign(values)
            tuning.reach_island_copies(self.sim.pipeline, name)
            flags |= self.FLAGS.get(name, _BODY_INERTIAL)
        self.saved = {}
        if flags:
            tuning.notify(self.sim.pipeline, flags)


def replay(
    sim: Any, routine: Iterator, mode: str, store: dict, *, steps: int = 0, steps_itself: bool = False
) -> Iterator[None]:
    """The routine (the generator it returned) played under ``mode``, its summary in ``store[mode]`` however it ends.

    A replay other than ``written`` runs as long as the written run,
    ``steps``: on past the routine's end when it stops early (a thing that
    drifts late still shows), and when it is still going, a little past
    (:data:`OVERRUN`, so a routine whose last lines run on the step after the
    written one's did is not cut off there) and no further: a routine waiting
    for a pose the frozen arm never reaches is stopped and read as not done. One that ``steps_itself`` is stepped here once it has
    stopped; one driving the arm panels has the page step it, as it always
    does. A replay the page stops (a reset, a save, Stop) is stored as
    :data:`STOPPED`, one that fails itself as :data:`FAILED`, and the model is
    put back whatever ended it.
    """
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {', '.join(MODES)}")
    capped = mode != "written" and steps > 0
    limit = steps + max(25, int(OVERRUN * steps))
    tap: Replay | None = None
    built = AsBuilt(sim, HEAVY if mode == "heavy" else 1.0) if mode in ("built", "heavy") else None
    outcome = "finished"
    try:
        tap = Replay(sim, mode)
        if capped:
            # a routine may loop on `sim.run` until a pose is reached, never yielding: the cap is enforced at a step
            # the routine takes itself, raised into it; a step the page takes between yields is cut at the yield
            tap.limit = limit
        if built is not None:
            built.apply()
        tap.install()
        try:
            while True:
                tap.inside = True  # a step taken now is the routine's own: the cap and a broken hold raise into it
                try:
                    next(routine)
                except StopIteration:
                    break
                finally:
                    tap.inside = False
                yield
                if tap.broken:
                    raise Broken(tap.broken)
                if capped and tap.steps >= limit:
                    outcome = f"still going when the written replay had long ended ({steps} steps)"
                    routine.close()
                    break
        except Overrun:
            outcome = f"still going when the written replay had long ended ({steps} steps)"
        except Exception as exc:  # the routine's own checks stopping it is the reading, not a failure of this
            outcome = f"{type(exc).__name__}: {str(exc)[:300]}"
        tap.limit = 0
        while capped and tap.steps < steps:
            if steps_itself:
                sim.run(1)
            yield
            if tap.broken:
                raise Broken(tap.broken)
    except GeneratorExit:
        outcome = STOPPED
        if tap is not None:
            tap.remove()  # first: a cleanup that steps must not meet the hold or the cap
        routine.close()  # its own cleanup runs now, on this scene, not at collection on the next one
        raise
    except Broken as exc:
        outcome = f"{FAILED}: {exc}"
        tap.remove()
        routine.close()
    except Exception as exc:
        outcome = f"{FAILED}: {type(exc).__name__}: {str(exc)[:300]}"
        raise
    finally:
        if tap is not None:
            tap.remove()
        if built is not None:
            built.restore()
        store[mode] = tap.summary(outcome) if tap is not None else {"outcome": outcome, "steps": 0, "things": {}}


# --------------------------------------------------------------------------- the verdict


def _moved(run: dict, floor: float = 20.0) -> list[str]:
    """What the run moved: things, a jointed thing's parts (a lid, a door, a key), deformables."""
    hit = [k for k, t in run.get("things", {}).items() if t["moved_max_mm"] > floor]
    return hit + [k for k, t in run.get("deformables", {}).items() if t["moved_max_mm"] > floor]


def _ended_elsewhere(run: dict, written: dict, names: list[str], mm: float = 50.0, deg: float = 30.0) -> str:
    """What did not end where the written run left it, said in a few words; ``""`` when everything did.

    Each thing has to end within ``mm`` and ``deg`` of its written end, and to have moved on the way at least a
    third as far as the written run moved it: a mug lifted and set back near its start ends there too.
    """
    for k in names:
        a = run.get("things", {}).get(k) or run.get("deformables", {}).get(k)
        b = written.get("things", {}).get(k) or written.get("deformables", {}).get(k)
        if not a or not b:
            continue
        if a.get("moved_max_mm", 0.0) < b.get("moved_max_mm", 0.0) / 3:
            return (
                f"{k} moved {a.get('moved_max_mm', 0.0):.0f} mm where the written run moved it {b['moved_max_mm']:.0f}"
            )
        off = float(np.linalg.norm(np.subtract(a["end"], b["end"])) * 1000)
        if off > mm:
            return f"{k} ended {off:.0f} mm from where the written run left it"
        if abs(a.get("particle_end_mm", 0.0) - b.get("particle_end_mm", 0.0)) > mm / 2:
            return f"{k}'s shape ended unlike the written run's"
        if "end_quat" in a and "end_quat" in b:  # the turn between the two ends, not between the sizes of two turns
            turn = _turn_deg(np.asarray(a["end_quat"]), np.asarray(b["end_quat"]))
        else:
            turn = abs(a.get("turn_end_deg", 0.0) - b.get("turn_end_deg", 0.0))
        if turn > deg:
            return f"{k} ended turned {turn:.0f}° from how the written run left it"
    return ""


def _same_end(run: dict, written: dict, names: list[str], mm: float = 50.0, deg: float = 30.0) -> bool:
    return not _ended_elsewhere(run, written, names, mm, deg)


def verdict(model: dict, runs: dict[str, dict], settle_mm: float = SETTLE_MM) -> dict[str, Any]:
    """The lines ``./ctl physics`` prints: PASS and FAIL as ``check`` says them, NOTE for what the report must explain."""
    lines: list[dict[str, Any]] = []

    def line(ok: bool | None, name: str, detail: str) -> None:
        lines.append({"ok": ok, "name": name, "detail": detail})

    driven = (runs.get("written") or {}).get("thing_targets_moved", 0.0) > 1e-6
    found = model.get("ties", []) + (model.get("drives", []) if driven else [])
    found += [f"{k} is kinematic" for k in model.get("kinematic", [])]
    if driven:
        found.append("a target on a thing's own joint moved while the routine played: a motor")
    added = (runs.get("written") or {}).get("hooks_added", 0)
    if added:
        found.append(f"{added} post-solve hook(s) attached while the routine played (a dragger or a pin)")
    welds = model.get("welds", [])
    if found:
        line(False, "contact", "; ".join(found + welds))
    elif welds:
        line(None, "contact", "; ".join(welds) + ": a welded thing cannot move, so say in the report why it is welded")
    else:
        springs = (
            f"; {len(model['drives'])} stiff joint(s) on things held at their targets (their elasticity)"
            if model.get("drives")
            else ""
        )
        line(
            True, "contact", "nothing ties a thing to the robot, no motor on a thing's joints, nothing welded" + springs
        )
    written = runs.get("written") or {}
    names = _moved(written)
    if written:
        flew = {
            k: t.get("free_flight_not_falling", 0)
            for k, t in written["things"].items()
            if t.get("free_flight_not_falling")
        }
        line(
            written["outcome"] == "finished",
            "written",
            f"finished; moved {', '.join(names) or 'nothing'}"
            if written["outcome"] == "finished"
            else f"stopped: {written['outcome']}",
        )
        if flew:
            line(
                None,
                "flight",
                f"free of every contact but not falling at g, steps: {flew}: say in the report what holds it",
            )
    frozen = runs.get("frozen")
    if frozen:
        # a rigid thing has nothing to settle into; a heap, a liquid or a draped cloth can still be relaxing at the
        # reset, so its drift is a note for the report, not a fault
        rigid = {k: t["moved_max_mm"] for k, t in frozen.get("things", {}).items()}
        soft = {k: t["moved_max_mm"] for k, t in frozen.get("deformables", {}).items()}
        worst = max(rigid, key=rigid.get) if rigid else ""
        moved = rigid.get(worst, 0.0)
        drift = {k: v for k, v in soft.items() if v > settle_mm}
        if moved > settle_mm:
            line(
                False,
                "frozen",
                f"{worst} moved {moved:.1f} mm with the robot frozen: it starts inside something, or something other "
                "than the robot pushes it",
            )
        elif drift:
            k = max(drift, key=drift.get)
            line(
                None,
                "frozen",
                f"with the robot frozen no rigid thing moved, but {k} drifted {drift[k]:.1f} mm: still settling",
            )
        else:
            line(True, "frozen", f"with the robot frozen nothing moved more than {max([moved, *soft.values()]):.1f} mm")
    opened = runs.get("open")
    if opened and (written.get("outcome") != "finished" or not names):
        line(
            None,
            "open",
            "not read: the written replay "
            + ("did not finish" if written.get("outcome") != "finished" else "moved nothing"),
        )
    elif opened:
        did = opened["outcome"] == "finished" and _same_end(opened, written, names)
        line(
            None if did else True,
            "open",
            "the routine finished with the fingers held open: say in the report what carries "
            f"{', '.join(names) or 'the thing'} (a support, a hook, two arms squeezing)"
            if did
            else f"with the fingers held open the task did not happen ({opened['outcome'] if opened['outcome'] != 'finished' else 'the things ended elsewhere'})",
        )
    torque = model.get("real_arm_torque")
    robot = f"every friction at most {REAL_MU:g}" + (
        f" and the arm's torque at {'/'.join(f'{v:g}' for v in sorted(set(torque), reverse=True))} Nm"
        if torque
        else " (no torque limits known for this robot)"
    )

    def held(run: dict, what: str) -> tuple[bool | None, str]:
        """Finished, and the things ended where the written run left them: a grip that slipped fails here."""
        if written.get("outcome") != "finished":
            return None, "not read: the written replay did not finish, so there is no end to hold this one to"
        if run["outcome"] != "finished":
            return False, f"with {what} the routine stopped: {run['outcome']}"
        elsewhere = _ended_elsewhere(run, written, names)
        if elsewhere:
            return False, f"with {what} the routine finished, but {elsewhere}"
        return True, f"finished with {what}, the things ending where the written run left them"

    built = runs.get("built")
    if built:
        ok, detail = held(built, robot)
        line(ok, "built", detail)
    heavy = runs.get("heavy")
    if heavy:
        ok, detail = held(heavy, f"the handled things {HEAVY:g} times as heavy")
        line(True if ok else None, "heavy", detail if ok is not False else detail + " (the grasp's margin)")
    if written:
        grip = []
        for k in names:
            for i, (p50, p95) in enumerate(written["things"].get(k, {}).get("hand_force_p50_p95_N", [])):
                if p50:
                    grip.append(f"{k} {p50:g} N (p95 {p95:g}) by hand {i + 1}")
        mu = written.get("finger_mu")
        detail = (
            ("the fingers pressed " + "; ".join(grip))
            if grip
            else "no finger force on a rigid thing (a deformable's is not read)"
        )
        if mu:
            detail += f"; the solver used friction {mu[0]:g} to {mu[1]:g} at the finger contacts"
        line(
            None if (mu and mu[1] > REAL_MU) else True,
            "grip",
            detail + (" (above any real pad: the built line is the one to trust)" if mu and mu[1] > REAL_MU else ""),
        )
    word = {True: "PASS", False: "FAIL", None: "NOTE"}
    return {
        "passed": all(c["ok"] is not False for c in lines),
        "checks": lines,
        "report": "\n".join(f"{word[c['ok']]}  {c['name']:<8} {c['detail']}" for c in lines),
    }
