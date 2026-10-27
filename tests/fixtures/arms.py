# SPDX-License-Identifier: Apache-2.0
"""One small six-joint arm, three ways: a Newton model on the CPU, a closed-form numpy chain, and a URDF.

The inverse-kinematics solvers are held to properties -- a forward-inverse-forward
round trip, the joint limits, a batch equal to its rows -- and every one of them
needs an arm whose forward kinematics is known exactly. This is that arm, a
spherical-ish wrist on a two-link elbow: joint axes ``z y y z y z``, each joint
``OFFSETS[i]`` metres up its parent's ``+z``, limits of +-2.5 rad, and a tool
centre ``TOOL`` up the last link. The three forms describe the same chain, so a
test can ask the scene-built :class:`~robosimstudio.solvers.ik.arm_ik.ArmIK`,
the batched solvers over :class:`~robosimstudio.solvers.ik.arm.ArmKinematics`,
and PyRoKi over the URDF the same question and expect the same answer.

Importing this module imports nothing of the library or of Newton; each
function imports what it builds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "AXES",
    "BENT",
    "LIMIT",
    "OFFSETS",
    "TOOL",
    "analytic_arm",
    "analytic_fk",
    "cpu_arm",
    "reachable",
    "urdf_arm",
]

AXES = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
"""Each joint's axis in its own frame: base yaw, shoulder, elbow, forearm roll, wrist pitch, wrist roll."""
OFFSETS = (0.1, 0.1, 0.3, 0.3, 0.05, 0.05)
"""Each joint's height up its parent's ``+z`` [m]."""
TOOL = (0.0, 0.0, 0.08)
"""The tool centre in the last link's frame [m]: on the wrist axis."""
LIMIT = 2.5
"""Every joint's limit, symmetric [rad]."""
BENT = 0.4
"""How far from straight :func:`reachable` keeps the elbow and the wrist pitch [rad]."""


def _axis_angle(axis, angles: np.ndarray) -> np.ndarray:
    """Rotations by each of ``angles [W]`` about one unit ``axis``: ``[W, 3, 3]``, Rodrigues' formula."""
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    c, s = np.cos(angles)[:, None, None], np.sin(angles)[:, None, None]
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


def analytic_fk(arm_q, base: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The tool centre of every row of ``arm_q [W, 6]``: ``(position [W, 3], quat_xyzw [W, 4])``.

    ``base`` is the arm's base pose as a 4x4 in the world, the identity when not given.
    """
    from robosimstudio.core.math import mat_to_quat

    rows = np.asarray(arm_q, dtype=np.float64).reshape(-1, len(AXES))
    root = np.eye(4) if base is None else np.asarray(base, dtype=np.float64)
    rot = np.repeat(root[None, :3, :3], len(rows), axis=0)
    pos = np.repeat(root[None, :3, 3], len(rows), axis=0)
    for k, (axis, offset) in enumerate(zip(AXES, OFFSETS, strict=True)):
        pos = pos + rot[:, :, 2] * offset
        rot = rot @ _axis_angle(axis, rows[:, k])
    pos = pos + rot @ np.asarray(TOOL)
    return pos, np.stack([mat_to_quat(m) for m in rot])


def analytic_arm(*, base: np.ndarray | None = None, urdf: str = "", base_xform=None, home_q=None):
    """:class:`ArmKinematics` over :func:`analytic_fk`: no Newton, no device, no scene."""
    from robosimstudio.solvers.ik.arm import ArmKinematics

    n = len(AXES)
    return ArmKinematics(
        arm_coords=list(range(n)),
        arm_dofs=list(range(n)),
        ee_body=n - 1,
        ee_offset=np.asarray(TOOL),
        ee_offset_rot=np.array([0.0, 0.0, 0.0, 1.0]),
        lower=np.full(n, -LIMIT),
        upper=np.full(n, LIMIT),
        home_q=np.full(n, 0.3) if home_q is None else np.asarray(home_q, dtype=np.float64),
        urdf=urdf,
        base_xform=None if base_xform is None else np.asarray(base_xform, dtype=np.float64),
        fk=lambda rows: analytic_fk(rows, base),
    )


def cpu_arm(*, iterations: int = 8):
    """The arm as a finalized Newton model on the CPU, with its :class:`ArmIK`: ``(model, arm_ik)``.

    No URDF, so every solve runs this tree's own damped least squares over
    Newton's forward kinematics, never PyRoKi.
    """
    import newton
    import warp as wp

    from robosimstudio.solvers.ik.arm_ik import ArmIK

    builder = newton.ModelBuilder()
    parent, joints = -1, []
    for i, (axis, offset) in enumerate(zip(AXES, OFFSETS, strict=True)):
        body = builder.add_link(xform=wp.transform_identity(), mass=0.5, label=f"link{i}")
        builder.add_shape_capsule(body, radius=0.02, half_height=0.05)
        joints.append(
            builder.add_joint_revolute(
                parent,
                body,
                axis=axis,
                parent_xform=wp.transform((0.0, 0.0, offset), wp.quat_identity()),
                limit_lower=-LIMIT,
                limit_upper=LIMIT,
            )
        )
        parent = body
    builder.add_articulation(joints)
    model = builder.finalize(device="cpu")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    n = len(AXES)
    arm = ArmIK(
        model,
        arm_coords=list(range(n)),
        arm_dofs=list(range(n)),
        ee_body=parent,
        home_state=state,
        ee_offset=TOOL,
        iterations=iterations,
    )
    return model, arm


def urdf_arm(directory: Path, *, scale: float = 1.0) -> Path:
    """The arm as a URDF written into ``directory``; ``scale`` stretches every link, to make one that is wrong."""
    links, previous = ['<link name="base"/>'], "base"
    for i, (axis, offset) in enumerate(zip(AXES, OFFSETS, strict=True)):
        links.append(
            f'<link name="link{i}"/>'
            f'<joint name="joint{i}" type="revolute"><parent link="{previous}"/><child link="link{i}"/>'
            f'<origin xyz="0 0 {offset * scale}"/><axis xyz="{" ".join(str(v) for v in axis)}"/>'
            f'<limit lower="{-LIMIT}" upper="{LIMIT}" effort="1" velocity="1"/></joint>'
        )
        previous = f"link{i}"
    path = Path(directory) / ("arm.urdf" if scale == 1.0 else f"arm_x{scale:g}.urdf")
    path.write_text('<robot name="arm">' + "".join(links) + "</robot>")
    return path


def reachable(seed: int, count: int, *, spread: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``count`` reachable targets drawn with ``seed``: ``(q [count, 6], position [count, 3], quat_xyzw [count, 4])``.

    Each is the forward kinematics of joint values drawn uniformly within
    ``spread`` of zero -- inside the limits, so a solver has a configuration that
    meets it exactly -- with the elbow (joint 2) and the wrist pitch (joint 4)
    each kept :data:`BENT` from straight. A straight elbow is the edge of the
    workspace, and a straight wrist lines the forearm roll up with the wrist
    roll: both are the singular configurations the solvers' docstrings measure
    apart, not ordinary ones.
    """
    rng = np.random.default_rng(seed)
    q = rng.uniform(-spread, spread, (count, len(AXES)))
    for joint in (2, 4):
        q[:, joint] = rng.choice((-1.0, 1.0), count) * rng.uniform(BENT, max(spread, BENT + 0.1), count)
    position, quat = analytic_fk(q)
    return q, position, quat
