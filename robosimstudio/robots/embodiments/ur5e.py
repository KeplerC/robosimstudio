# SPDX-License-Identifier: Apache-2.0
"""The Universal Robots UR5e, bare, as MuJoCo Menagerie ships it.

Six revolute joints and a tool flange (``attachment_site`` on ``wrist_3_link``:
``pos="0 0.1 0" quat="-1 1 0 0"``), imported from the vendored
``assets/embodiments/ur5e/ur5e.xml`` (BSD-3-Clause, ``assets/PROVENANCE.md``).
No hand, so not a catalog entry: this is the arm a *file-carrying* hand is
welded onto at build -- ``robotiq_2f85_menagerie`` names this very flange as
its ``mount_body`` -- and the arm the composed Sharpa robots
(:mod:`.ur5e_sharpa`) are made from, which are the catalog entries.

The file drives its joints with ``<general>`` position servos (``gainprm 2000``,
``biasprm 0 -2000 -400`` on the three big joints, 500/100 on the wrists, ±150 and
±28 N·m), which is what the arm_actuation restates: the same numbers, stated
here so a reader finds them without opening the MJCF and so the position-mode
check runs on them.
"""

from __future__ import annotations

from robosimstudio.specs import ActuationSpec, ArmSpec, AssetSpec, EmbodimentSpec

__all__ = ["UR5E", "UR5E_ACTUATION", "UR5E_HOME", "UR5E_HOME_LEFT", "UR5E_JOINTS"]

UR5E_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
"""The six joints, base to flange, the file's own labels."""

UR5E_HOME = (2.9248, -1.3242, 1.9259, -0.2118, 1.3698, -1.6527)
"""A posture over the work surface for the Sharpa Wave's pinch: the pinch
point 0.6 m in front of the base, 0.2 m to the arm's right and 0.1 m up,
the fingers pointing forward and 35 degrees down with the palm facing down
and back, as a hand comes to a thing on a table. Pitched rather than flat
because with the fingers level the UR5e's wrist_2 link hangs 20 mm below the
fingertip pads and meets the table before the pinch point is 25 mm over it;
at 35 degrees the wrist stands 0.35 m up and the pads are the lowest thing.
Solved by forward kinematics on the composed file (:mod:`.ur5e_sharpa`); the
shoulder pans by about pi because the arm's zero reaches *behind* its base.
:data:`UR5E_HOME_LEFT` is the same for a hand 0.2 m to the arm's left."""

UR5E_HOME_LEFT = (2.6947, -1.3171, 1.9201, -0.1842, 1.1582, -1.7474)
"""The same hand pose 0.2 m to the arm's *left*: the left arm of a pair, so
that two arms 0.6 m apart hold their pinch points 0.4 m apart over the middle."""

UR5E_ACTUATION = ActuationSpec(
    target_ke=2000.0,
    target_kd=400.0,
    target_ke_per_dof=(2000.0, 2000.0, 2000.0, 500.0, 500.0, 500.0),
    target_kd_per_dof=(400.0, 400.0, 400.0, 100.0, 100.0, 100.0),
    effort_limit_per_dof=(150.0, 150.0, 150.0, 28.0, 28.0, 28.0),
)
"""Menagerie's own servo gains and torque limits, restated."""

UR5E = EmbodimentSpec(
    name="ur5e",
    asset=AssetSpec(path="embodiments/ur5e/ur5e.xml"),
    # On the work surface, 0.6 m behind its middle, facing it.
    base_offset=(-0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    gravcomp=True,
    reset_joint_q=True,
    hull_colliders=True,
    arms=(
        ArmSpec(
            name="arm", joint_names=UR5E_JOINTS, ee_body_name="wrist_3_link", base_body_name="base", home_q=UR5E_HOME
        ),
    ),
    arm_actuation=UR5E_ACTUATION,
)
"""The bare arm, **not registered**: a catalog robot has a hand, and this one
is the wrist a hand goes on. ``replace(UR5E, name=..., arms=(replace(UR5E.arms[0],
gripper="robotiq_2f85_menagerie"),))`` handed to ``rss.robot`` builds it with
Menagerie's 2F-85 welded on at build; :mod:`.ur5e_sharpa` composes the Sharpa
hands into its file instead."""
