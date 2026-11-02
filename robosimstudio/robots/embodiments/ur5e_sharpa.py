# SPDX-License-Identifier: Apache-2.0
"""UR5e arms wearing Sharpa Wave hands: one arm, and a bimanual pair.

Composed files, not runtime welds. ``scripts/compose_ur5e_sharpa.py`` bolts each
hand's flange onto its arm's ``attachment_site`` through MuJoCo's own
``MjSpec.attach`` and writes ``assets/embodiments/ur5e_sharpa/`` (see
``assets/PROVENANCE.md``). The runtime weld (``GripperSpec.asset``) would do the
same for one arm, and cannot for two: Newton attaches a hand only to the
articulation added last. So the pair is a file, and the single arm is the same
file with one arm, so that both build the same way and a measurement on one
holds on the other.

Labels: the arms are ``left_``/``right_`` prefixed (``right_shoulder_pan_joint``,
``right_wrist_3_link``); the hands keep their own files' labels, which already
start with the side (``right_thumb_CMC_FE``, ``right_hand_C_MC``). The single
arm is the right one.

**Where they stand.** The pair's bases are 0.6 m apart in the file (±0.3 m); the
embodiment puts that midpoint 0.6 m behind the work surface's middle, on it.
At home (:data:`~.ur5e.UR5E_HOME` and its mirror for the left arm) each palm
holds its pinch point over the surface's middle, 0.2 m either side of the
centre line and 0.1 m up, the fingers forward and 35 degrees down, the palms
facing down and back and the thumbs inward -- the way two hands come down to
pinch a thing on a table.

The arm gains are Menagerie's own (:data:`~.ur5e.UR5E_ACTUATION`); the hands'
are their file's (:mod:`..grippers.sharpa_wave`, whose left entry carries the
mirrored pinch point). Each hand's joints are driven
from ``joint_target_q`` like any catalog hand's; ``cross_arm_collisions`` keeps
the pair from passing through each other while each arm ignores itself.
"""

from __future__ import annotations

from robosimstudio.specs import ArmSpec, AssetSpec, EmbodimentSpec

from ..registry import register_embodiment
from .ur5e import UR5E_ACTUATION, UR5E_HOME, UR5E_HOME_LEFT, UR5E_JOINTS

__all__ = ["UR5E_SHARPA", "UR5E_SHARPA_BIMANUAL"]


def _arm(side: str) -> ArmSpec:
    return ArmSpec(
        name=side,
        joint_names=tuple(f"{side}_{j}" for j in UR5E_JOINTS),
        ee_body_name=f"{side}_wrist_3_link",
        base_body_name=f"{side}_base",
        gripper="sharpa_wave_left" if side == "left" else "sharpa_wave",
        home_q=UR5E_HOME_LEFT if side == "left" else UR5E_HOME,
    )


UR5E_SHARPA = register_embodiment(
    EmbodimentSpec(
        name="ur5e_sharpa",
        asset=AssetSpec(path="embodiments/ur5e_sharpa/ur5e_sharpa.xml"),
        base_offset=(-0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        gravcomp=True,
        reset_joint_q=True,
        hull_colliders=True,
        arms=(_arm("right"),),
        arm_actuation=UR5E_ACTUATION,
    )
)

UR5E_SHARPA_BIMANUAL = register_embodiment(
    EmbodimentSpec(
        name="ur5e_sharpa_bimanual",
        asset=AssetSpec(path="embodiments/ur5e_sharpa/ur5e_sharpa_bimanual.xml"),
        base_offset=(-0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        gravcomp=True,
        reset_joint_q=True,
        hull_colliders=True,
        cross_arm_collisions=True,
        arms=(_arm("left"), _arm("right")),
        arm_actuation=UR5E_ACTUATION,
    )
)
