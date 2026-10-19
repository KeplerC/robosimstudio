# SPDX-License-Identifier: Apache-2.0
"""The arena YAM with a vacuum cup on its right arm, and the left hand as it is.

A pairing rather than a new robot: :mod:`yam_bimanual`'s spec with the right
arm's :attr:`~robosimstudio.specs.ArmSpec.gripper` replaced. It is registered
under its own name because the gripper axis cannot carry it: a binding's
``gripper`` puts one hand on *every* arm, and ``yam_bimanual_arms.xml`` declares
its two arms as two articulation roots, of which Newton welds a hand only onto
the one added last. So the cup has to be named on one arm, and a task document
stages it as ``(:embodiment yam_bimanual_suction)``.

Three things the pairing says that the plain robot does not:

* **The right arm names its six joints.** The cup has no fingers and clears the
  finger token, so in prefix mode the file's ``right_left_finger`` and
  ``right_right_finger`` would join the arm and make an eight-joint chain.
  Named, the arm is joints 1 to 6 and the file's fingers stay where they are:
  undriven by any command, held shut by the file's own actuator, 44 mm off the
  stem's axis.
* **The cup hangs past the fingers.** The mount is 105 mm along the wrist's +z,
  so the face sits at 165 mm, 26 mm beyond the fingertips' nubs (which end at
  139 mm, measured in :mod:`robosimstudio.robots.grippers.yam_linear_arena`):
  the plate under the face is clear of the fingers, and so is the plate once
  it is carried.
* **The right arm's home points the cup straight down**, its face 106 mm over
  the bench straight ahead of the right shoulder, 100 mm over the top of the
  6 mm plate the examples lay there (:data:`CUP_DOWN`; ``probe_suction_fix``,
  GPU 3), so a hand dragged down on the page lands the face flat. Solved with
  ``sim.arm("right").solve_q_full`` on ``bench_075``; the wrist's last joint is
  left at 0, the middle of its travel, because a round cup has no roll to keep.
"""

from __future__ import annotations

from robosimstudio.specs import replace

from ..registry import register_embodiment
from .yam_bimanual import YAM_BIMANUAL

__all__ = ["CUP_DOWN", "RIGHT_JOINTS", "SUCTION_MOUNT", "YAM_BIMANUAL_SUCTION"]

RIGHT_JOINTS = tuple(f"right_joint{i}" for i in range(1, 7))
"""The right arm's six joints, named so the file's fingers are not swept in."""

SUCTION_MOUNT = (0.0, 0.0, 0.105, 0.0, 0.0, 0.0, 1.0)
"""The cup's flange in ``right_link_6``'s frame: 105 mm along the wrist's +z, unrotated."""

CUP_DOWN = (0.0, 1.8599, 1.6247, -1.3356, 0.0, 0.0)
"""The right arm's home [rad]: the cup's face 106 mm over ``bench_075``, pointing straight down."""

_LEFT, _RIGHT = YAM_BIMANUAL.arms

YAM_BIMANUAL_SUCTION = register_embodiment(
    replace(
        YAM_BIMANUAL,
        name="yam_bimanual_suction",
        arms=(
            _LEFT,
            replace(_RIGHT, joint_names=RIGHT_JOINTS, gripper="suction", gripper_mount=SUCTION_MOUNT, home_q=CUP_DOWN),
        ),
    )
)
