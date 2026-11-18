"""Where the label is on the parcel: its flat part's centre and turn in the box's frame, against the marked spot.

The label is the tape kind's grid: particle ``j * (nx + 1) + c`` is row ``j`` along the sheet from the tab end and
column ``c`` across it. Its first ``tab`` metres (rows 0..2 at 5 mm) are turned up into the tab; row 3 is the fold;
rows 3..19 are the flat part, 80 x 40 mm.
"""

from __future__ import annotations

import numpy as np

TARGET = (0.0, 0.0)
"""The marked spot's centre on the box top, in the box's frame [m]: the box's middle (art.py prints the marks there)."""
TOP = 0.025
"""The box's half height: its top in its own frame [m]."""
HALF_XY = (0.05, 0.06)


def _rot(q):
    from robosimstudio.core.math import quat_to_mat

    return quat_to_mat(np.asarray(q, dtype=np.float64))


def flat_points(label, state) -> np.ndarray:
    """The flat part's particles [n, 3] in world, rows from the fold to the free end."""
    nx, ny = label.spec.cells
    fold = int(round(float(label.spec.tab) / label.spec.cell[1]))
    q = state.particle_q.numpy()[np.asarray(label.particles)].astype(np.float64)
    return q[fold * (nx + 1) :], (nx, ny, fold)


def label_pose(label, state) -> dict:
    """The flat part in the box's frame: ``off_mm`` from the target, ``yaw_deg`` off the box's -y axis, and
    ``on_top_pct`` of its particles over the box top within 3 mm of it."""
    pts, (nx, ny, fold) = flat_points(label, state)
    box = state.body_q.numpy()[label.body].astype(np.float64)
    local = (pts - box[:3]) @ _rot(box[3:7])
    c = local.mean(axis=0)
    rows = local.reshape(ny + 1 - fold, nx + 1, 3).mean(axis=1)
    d = rows[-1] - rows[0]
    yaw = float(np.degrees(np.arctan2(d[0], -d[1])))
    over = (
        (np.abs(local[:, 0]) <= HALF_XY[0]) & (np.abs(local[:, 1]) <= HALF_XY[1]) & (np.abs(local[:, 2] - TOP) < 0.003)
    )
    return {
        "off_mm": float(1e3 * np.hypot(c[0] - TARGET[0], c[1] - TARGET[1])),
        "yaw_deg": yaw,
        "on_top_pct": float(100.0 * over.mean()),
        "cx_mm": float(1e3 * c[0]),
        "cy_mm": float(1e3 * c[1]),
    }
