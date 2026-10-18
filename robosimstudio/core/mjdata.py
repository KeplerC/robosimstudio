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

What Newton mirrors itself and what it does not, read off its source:
``notify_model_changed`` regenerates the ``mjw_model`` rows and copies world 0
into ``mj_model`` for inertial, body and DOF properties -- so a write that also
lands in the Newton attribute survives a later notify on either device -- but
the joint-property path writes ``mjw_data.mocap_pos`` / ``mocap_quat`` only.
:func:`sync_mocap` is that missing copy.

``mujoco`` is imported inside the function that needs it, as every optional
backend in this tree is.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["contacts", "host_side", "read_gravcomp", "set_option", "sync_mocap", "write_gravcomp"]


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
