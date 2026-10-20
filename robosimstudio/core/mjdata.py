# SPDX-License-Identifier: Apache-2.0
"""Reading and writing the MuJoCo data a :class:`newton.solvers.SolverMuJoCo` actually steps.

On a GPU that is MuJoCo-Warp's ``mjw_model`` / ``mjw_data``. On a CPU
(``use_mujoco_cpu=True``) Newton steps MuJoCo's C library instead --
``mujoco.mj_step(mj_model, mj_data)`` -- and the Warp copies are still built,
filled once and **never stepped again**. Everything here that reaches into the
solver's MuJoCo state does it through this module, because the failure it
prevents is the quiet kind: a contact probe reading ``mjw_data`` on a CPU gets
the first frame's contacts forever, a parked body whose ``mjw_model`` row was
set to 1.0 falls anyway, and neither raises.

What Newton mirrors itself and what it does not, read off its source (1.6):
``notify_model_changed`` regenerates the ``mjw_model`` rows and copies world 0
into ``mj_model`` for inertial, body, DOF and joint properties -- so a write that
also lands in the Newton attribute survives a later notify on either device --
and leaves out everything else it regenerates: a shape's friction, solref,
margin and pose, a site, an actuator's gains (which the joint-DOF path writes
too, for a joint target's stiffness), a tendon. :func:`mirror_notify` is that
copy, and :func:`after_notify` runs it after every notify the program sends. The joint-property path
writes ``mjw_data.mocap_pos`` / ``mocap_quat`` only; :func:`sync_mocap` is that
one. Measured before the copy: a cube's friction set to 0.05 left the C model's
at 1.0, and a doubled joint stiffness left its actuator gain at 650.

Newton's notify also moves something the C library's collision reads and no
notify recomputes: each body's inertial frame. Its first one, inside the
solver's constructor, copies MuJoCo-Warp's re-diagonalized ``body_iquat`` --
90 or 180 degrees off the compiled one on eight of the YAM's eighteen bodies,
both left fingers among them -- over the C model, whose bounding-volume tree for
each body MuJoCo's compiler built in the frame it had then. The midphase then
tests a finger against a box that is no longer round its geoms and skips the
pair: measured on ``examples/03_materials/10_cutting``, the right hand closing
on the 20 mm knife handle stalled at 20.4 mm on MuJoCo-Warp and passed through
it to 3.8 mm on the C library, its left finger 8.8 mm inside the handle with no
contact. :func:`refit_bvh` refits the trees to the frames the model holds, after
the solver is built (:func:`robosimstudio.scene.physics.backends.build_solver`)
and after every notify that can move them; with it the hand stalls at 20.3 mm.

``mujoco`` is imported inside the function that needs it, as every optional
backend in this tree is.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "after_notify",
    "contacts",
    "host_side",
    "mirror_notify",
    "read_gravcomp",
    "refit_bvh",
    "set_option",
    "sync_mocap",
    "write_gravcomp",
]

RECONSTANT = ("JOINT_DOF_PROPERTIES", "ACTUATOR_PROPERTIES", "TENDON_PROPERTIES")
"""Flags whose notify ends in ``mj_setConst`` on a CPU, before :func:`mirror_notify` can copy the rows it reads."""

MIRRORED: dict[str, tuple[str, ...]] = {
    "SHAPE_PROPERTIES": (
        "geom_friction", "geom_gap", "geom_margin", "geom_pos", "geom_quat", "geom_size",
        "geom_solimp", "geom_solmix", "geom_solref", "site_pos", "site_quat", "site_size",
    ),
    "JOINT_DOF_PROPERTIES": ("actuator_biasprm", "actuator_gainprm"),
    "ACTUATOR_PROPERTIES": (
        "actuator_actrange", "actuator_biasprm", "actuator_cranklength", "actuator_ctrlrange",
        "actuator_dynprm", "actuator_forcerange", "actuator_gainprm", "actuator_gear",
    ),
    "TENDON_PROPERTIES": (
        "tendon_actfrcrange", "tendon_armature", "tendon_damping", "tendon_frictionloss", "tendon_margin",
        "tendon_range", "tendon_solimp_fri", "tendon_solimp_lim", "tendon_solref_fri", "tendon_solref_lim",
        "tendon_stiffness",
    ),
}  # fmt: skip
"""``newton.ModelFlags`` name -> the ``mjw_model`` fields Newton's notify regenerates for it and does not copy
into the C model on a CPU. Read off ``SolverMuJoCo._update_*_properties``; pair properties are absent because
Newton skips their update on a CPU altogether."""


def host_side(solver: Any) -> bool:
    """Whether ``solver`` steps MuJoCo's C library (a CPU build) rather than MuJoCo-Warp."""
    return bool(getattr(solver, "use_mujoco_cpu", False))


def set_option(solver: Any, param: str, value: Any) -> bool:
    """Set one ``opt`` field on the model the solver steps; ``False`` when it has no such field.

    Both copies on a CPU, so the Warp one -- which nothing steps there -- never
    disagrees with the one that is.
    """
    targets = [getattr(getattr(solver, "mjw_model", None), "opt", None)]
    if host_side(solver):
        targets.insert(0, getattr(getattr(solver, "mj_model", None), "opt", None))
    done = False
    for opt in targets:
        if opt is None or not hasattr(opt, param):
            continue
        setattr(opt, param, type(getattr(opt, param))(value))
        done = True
    return done


def write_gravcomp(solver: Any, tiled: np.ndarray) -> None:
    """Write the per-world ``body_gravcomp`` table, ``[worlds, nbody]``, where it is stepped.

    The Warp row always (it is what a later notify regenerates from on a GPU);
    on a CPU also world 0's row into ``mj_model``, the one ``mj_step`` reads.
    """
    solver.mjw_model.body_gravcomp.assign(tiled)
    if host_side(solver):
        solver.mj_model.body_gravcomp[:] = np.asarray(tiled, dtype=np.float64)[0]


def read_gravcomp(solver: Any, world: int, mjc_body: int) -> float:
    """What the stepped model holds for one MuJoCo body in one world."""
    if host_side(solver):
        if int(world) != 0:
            raise IndexError(f"MuJoCo's C backend steps one world; there is no world {world}")
        return float(solver.mj_model.body_gravcomp[int(mjc_body)])
    return float(solver.mjw_model.body_gravcomp.numpy()[int(world), int(mjc_body)])


def mirror_notify(solver: Any, flags: int) -> int:
    """After ``notify_model_changed(flags)``: copy what it regenerated into the C model a CPU build steps.

    World 0 of each :data:`MIRRORED` field the flags cover. Nothing on a GPU,
    where the regenerated rows are the ones stepped. Returns how many fields
    were copied, for a test.
    """
    if not host_side(solver):
        return 0
    import newton

    copied = 0
    for flag, fields in MIRRORED.items():
        if not int(flags) & int(getattr(newton.ModelFlags, flag, 0)):
            continue
        for name in fields:
            source = getattr(solver.mjw_model, name, None)
            target = getattr(solver.mj_model, name, None)
            if source is None or target is None or not hasattr(source, "numpy"):
                continue
            values = np.asarray(source.numpy())
            if values.ndim == np.ndim(target) + 1:
                values = values[0]
            if values.size != np.size(target):
                continue
            target[:] = values.reshape(np.shape(target))
            copied += 1
    return copied


def after_notify(solver: Any, flags: int) -> int:
    """Finish ``solver.notify_model_changed(flags)`` on a CPU: copy what it left out, then let it recompute.

    Newton's CPU notify ends in ``mj_setConst``, which derives constants from the
    actuator and tendon rows -- and resolves a position actuator's ``dampratio``
    placeholder into its damping from ``kp`` -- while those rows still hold the
    old values. So for :data:`RECONSTANT` flags the notify runs once more after
    the copy: everything it copies itself is unchanged by then, and its
    ``mj_setConst`` now reads the rows the scene is meant to have. A
    ``JOINT_PROPERTIES`` notify also carries the mocap poses across
    (:func:`sync_mocap`). Nothing on a GPU. Returns how many fields were copied.
    """
    import newton

    copied = mirror_notify(solver, flags)
    if copied and any(int(flags) & int(getattr(newton.ModelFlags, name, 0)) for name in RECONSTANT):
        solver.notify_model_changed(int(flags))
    if int(flags) & int(newton.ModelFlags.JOINT_PROPERTIES):
        sync_mocap(solver)
    if int(flags) & int(newton.ModelFlags.BODY_INERTIAL_PROPERTIES | newton.ModelFlags.SHAPE_PROPERTIES):
        refit_bvh(solver)
    return copied


_CORNERS = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])
"""The eight corners of a box, as signs on its half-extents."""


def _rotation(quats: np.ndarray) -> np.ndarray:
    """Rotation matrices ``[n, 3, 3]`` of MuJoCo's ``(w, x, y, z)`` quaternions ``[n, 4]``."""
    q = np.asarray(quats, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=1,
    )


def refit_bvh(solver: Any) -> int:
    """Refit the C model's per-body bounding-volume trees to the inertial frames and geom poses it now holds.

    MuJoCo's compiler builds each body's tree in the body's inertial frame, a
    leaf round each geom's box, and the midphase culls a geom pair whose boxes
    miss. Newton's notify overwrites the inertial frames and the geom poses and
    never the trees (see the module docstring). This recomputes every node of
    every body's tree the way the compiler does -- a leaf is the box round its
    geom's eight corners in the inertial frame, a parent the box round its
    children -- which reproduces the compiled tree to 1e-16 m in the compiled
    frames. A geom whose size changed keeps the box the compiler gave it.
    Nothing on a GPU, where MuJoCo-Warp has no such tree. Returns how many
    nodes moved by more than a micrometre.
    """
    if not host_side(solver):
        return 0
    model = getattr(solver, "mj_model", None)
    if model is None or int(getattr(model, "nbvhstatic", 0)) == 0:
        return 0
    old = np.array(model.bvh_aabb, dtype=np.float64)
    aabb = old.copy()
    nodeid, child = np.asarray(model.bvh_nodeid), np.asarray(model.bvh_child)
    inertial = _rotation(model.body_iquat)
    geom_rot = _rotation(model.geom_quat)
    for body in range(int(model.nbody)):
        start, count = int(model.body_bvhadr[body]), int(model.body_bvhnum[body])
        if start < 0 or count <= 0:
            continue
        rot, origin = inertial[body], np.asarray(model.body_ipos[body], dtype=np.float64)
        # A child's index is past its parent's, so walking the body's nodes backwards meets children first.
        for node in range(start + count - 1, start - 1, -1):
            kids = [start + int(k) for k in child[node] if int(k) >= 0]
            if kids:
                points = np.concatenate([aabb[kids, :3] - aabb[kids, 3:], aabb[kids, :3] + aabb[kids, 3:]])
            else:
                geom = int(nodeid[node])
                corners = model.geom_aabb[geom, :3] + _CORNERS * model.geom_aabb[geom, 3:]
                points = (model.geom_pos[geom] + corners @ geom_rot[geom].T - origin) @ rot
            low, high = points.min(axis=0), points.max(axis=0)
            aabb[node, :3], aabb[node, 3:] = (low + high) / 2, (high - low) / 2
    model.bvh_aabb[:] = aabb
    return int(np.count_nonzero(np.abs(aabb - old).max(axis=1) > 1.0e-6))


def sync_mocap(solver: Any) -> None:
    """Copy world 0's mocap poses from the Warp data into the C data a CPU build steps.

    Newton's joint-property notify writes only ``mjw_data.mocap_*``; on a GPU
    that is the data stepped, and this does nothing.
    """
    if not host_side(solver):
        return
    mjw, data = solver.mjw_data, solver.mj_data
    if int(getattr(data, "mocap_pos", np.zeros((0, 3))).shape[0]) == 0:
        return
    data.mocap_pos[:] = mjw.mocap_pos.numpy()[0]
    data.mocap_quat[:] = mjw.mocap_quat.numpy()[0]


def contacts(solver: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This step's contacts from MuJoCo's C data: ``(geoms [n, 2], force on geom 2 [n, 3], dist [n])``.

    The force is in world coordinates, in the convention MuJoCo-Warp's
    ``contact_force(..., to_world_frame=True)`` uses -- the contact-frame force
    ``mj_contactForce`` returns, ``@`` the contact frame, whose rows are the
    normal and the two tangents -- so the probe above reads the same sign and
    magnitude from either backend.
    """
    import mujoco

    model, data = solver.mj_model, solver.mj_data
    n = int(data.ncon)
    if n == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 3)), np.zeros(0)
    found = data.contact
    frames = np.asarray(found.frame[:n], dtype=np.float64).reshape(n, 3, 3)
    forces = np.zeros((n, 3))
    wrench = np.zeros(6)
    for i in range(n):
        mujoco.mj_contactForce(model, data, i, wrench)
        forces[i] = wrench[:3] @ frames[i]
    geoms = np.asarray(found.geom[:n], dtype=np.int32).reshape(n, 2)
    return geoms, forces, np.asarray(found.dist[:n], dtype=np.float64)
