# SPDX-License-Identifier: Apache-2.0
"""A single-cup vacuum hand, attachable to any imported wrist.

The hand a parallel jaw cannot stand in for: a plate lying flat on the bench
has no edge a jaw can reach, and a 6 mm thick one offers the fingers nothing to
close across. A cup seals onto the top face instead, and what it can lift is
decided by the seal rather than by the fingers' stall.

The asset is this package's own (``assets/embodiments/suction/suction_cup.xml``,
60 mm from flange to face): a mount body carrying the flange and a stem, and a
pad body whose origin *is* the face, normal along +z. That convention is what
:mod:`robosimstudio.core.suction` reads when it asks whether the face is on a
part, and :attr:`~robosimstudio.specs.GripperSpec.tcp_body` names the same
body, so the tool-centre point every solver drives is the face's centre:
driving the hand "down onto the plate" puts the rubber on the plate, not the
flange.

The hand has no joints. :attr:`~robosimstudio.specs.GripperSpec.finger_names`
is empty and :attr:`~robosimstudio.specs.GripperSpec.finger_token` is cleared,
because the token is the prefix scan's silent path: left at ``"finger"``, the
YAM's own ``right_left_finger`` would be swept in as this hand's and driven as
a mirrored pair. An arm that carries this hand and still has fingers of its own
in its file names its arm joints explicitly (``yam_bimanual_suction`` does), so
the fingers stay undriven and the arm chain stays six joints. Its span is 0
(:attr:`~robosimstudio.specs.GripperSpec.effective_span`), so no grasp-fit test
offers it a part to pinch.

No mount body: the hand bolts to whatever wrist the arm names, at the arm's
:attr:`~robosimstudio.specs.ArmSpec.gripper_mount`, which is the point of a hand
with its own file. The geometry is authored rather than measured, so the pad
radius and the face offset are the file's; what is measured is the hold, and
that lives with the mechanism in :mod:`robosimstudio.core.suction`.
"""

from __future__ import annotations

from robosimstudio.specs import AssetSpec, GripperSpec

from ..registry import register_gripper

__all__ = ["PAD_RADIUS", "SUCTION", "SUCTION_LENGTH"]

PAD_RADIUS = 0.015
"""The cup's radius [m], as authored: a 30 mm face."""
SUCTION_LENGTH = 0.060
"""Flange to face along the approach axis [m], as authored."""

SUCTION = register_gripper(
    GripperSpec(
        name="suction",
        asset=AssetSpec(path="embodiments/suction/suction_cup.xml"),
        mount_body="",
        finger_names=(),
        finger_token="",
        tcp_body="suction_pad",
        tcp_offset=(0.0, 0.0, 0.0),
        tcp_offset_rot=(0.0, 0.0, 0.0, 1.0),
        approach_axis=(0.0, 0.0, 1.0),
        suction_pad="suction_pad",
        pad_radius=PAD_RADIUS,
        tip_reach=0.0,
        housing_half=0.018,
        actuation=None,
    )
)
