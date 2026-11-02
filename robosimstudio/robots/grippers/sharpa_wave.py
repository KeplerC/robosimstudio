# SPDX-License-Identifier: Apache-2.0
"""The Sharpa Wave, a 22-joint five-finger hand, as Sharpa's own MJCF describes it.

The first hand here with more joints than a jaw. Its file
(``assets/embodiments/sharpa_wave/{left,right}_sharpa_wave.xml``, Apache-2.0,
``assets/PROVENANCE.md``) carries 22 ``<position>`` actuators with the gains,
armature, damping and friction loss Sharpa calibrated against the real hand in
Isaac Lab, convex-hull colliders per phalanx, and a softer ``solref`` on the five
fingertip elastomers. Every number below is read off that file or measured on
it by forward kinematics (``scripts/compose_ur5e_sharpa.py`` builds the composed
arms this hand ships on; the probe is in this module's tests).

**How a 22-joint hand fits a two-value interface.** The scalar command still
runs *open* to *closed*: open is the pinch held 45 mm apart with the other
three fingers curled away (:data:`READY`); closed is the thumb-index pinch
(:data:`PINCH`), the index finger curled a right angle and the thumb solved
onto its tip. The flat hand is a posture (:data:`FLAT`), not
the open end: a hand that arrives flat puts its straight fingers on the table
before its pinch point reaches the thing. So ``robot.set_grip``, the studio's
**Grip** slider and every graph tool that closes a hand close this one into a
pinch. Anything else a five-finger hand does -- a fist, a point, a retargeted
human hand -- goes through :attr:`GripperSpec.postures` and
``Gripper.set_joint_targets``, one target per joint.

**Measured on the right hand, in its palm body's frame** (``right_hand_C_MC``:
``+z`` along the fingers, ``+x`` out of the palm, ``+y`` towards the thumb):

* flat hand: thumb elastomer centre at (0.016, 0.119, 0.136), index at
  (0.003, 0.030, 0.222), 0.124 m apart;
* pinch: the two elastomer centres meet 12.3 mm apart (their own thickness) at
  (0.078, 0.032, 0.132); ready, the index pad stays at (0.078, 0.030, 0.138)
  and the thumb's stands 0.045 m from it along ``y``, which is :attr:`span`,
  and the middle of that opening, (0.078, 0.053, 0.138), is the tool-centre
  point: the index finger does not move to close, so a thing goes where both
  pads clear it, not where they meet;
* the index finger's last phalanx, which the hand approaches along, points
  (0.974, 0, -0.227): out of the palm, tilted 13 degrees towards the wrist. So
  the pinch is made **palm down**, fingers forward, and the grasp frame
  (``+z`` the approach, ``+y`` the closing direction, the thumb's side) is
  the palm frame turned by :attr:`tcp_offset_rot`;
* the hand is 151 mm across the knuckles and 222 mm from flange to the
  longest fingertip; it weighs 1.25 kg.

The left hand is the right one mirrored: the same joint values make the same
pinch, and every ``y`` flips sign. :data:`SHARPA_WAVE_LEFT` is this spec with
the pinch point and the closing axis mirrored, and nothing else -- a spec
templated over ``{arm}`` substitutes names, not numbers, so a hand whose two
sides differ by a sign is two registry entries.

**What is 0 because nobody measured it:** ``pad_*`` and ``housing_rise``. The
grasp sampler reads them, so a planned pinch on this hand is a guess until
someone probes it; a routine that places the tool-centre point itself is not.
"""

from __future__ import annotations

from dataclasses import replace

from robosimstudio.specs import ActuationSpec, GripperSpec

from ..registry import register_gripper

__all__ = [
    "ACTUATION",
    "CURL",
    "FILE_ACTUATION",
    "FINGERS",
    "FIST",
    "FLAT",
    "GAIN",
    "JOINTS",
    "KP",
    "KV",
    "OPEN",
    "PINCH",
    "POINT",
    "READY",
    "SHARPA_WAVE",
    "SHARPA_WAVE_LEFT",
    "posture",
]

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
"""The five fingers, in the file's own order."""

JOINTS = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
)
"""The 22 joints, without their side prefix, in the file's actuator order.

``FE`` flexes and extends, ``AA`` abducts and adducts; the thumb has a
carpometacarpal pair, a metacarpophalangeal pair and an interphalangeal joint;
each finger a metacarpophalangeal pair and two interphalangeals; the little
finger one more carpometacarpal that cups the palm."""

KP = (
    6.95,
    13.2,
    4.76,
    6.62,
    0.9,
    4.76,
    6.62,
    0.9,
    0.9,
    4.76,
    6.62,
    0.9,
    0.9,
    4.76,
    6.62,
    0.9,
    0.9,
    1.38,
    4.76,
    6.62,
    0.9,
    0.9,
)
"""Position gain per joint [N·m/rad], the file's own ``kp``."""

KV = (
    0.2839, 0.4024, 0.2035, 0.2401, 0.0419,
    0.2078, 0.2451, 0.0423, 0.0350,
    0.2078, 0.2451, 0.0423, 0.0350,
    0.2078, 0.2451, 0.0423, 0.0350,
    0.0272, 0.2077, 0.2449, 0.0423, 0.0350,
)  # fmt: skip
"""Damping per joint [N·m·s/rad]: the file's ``dampratio="0.9"`` resolved by
MuJoCo against each joint's armature, read off the compiled model's
``biasprm``. Stated because Newton's importer keeps a damping *ratio* unresolved
and the hand's own file would otherwise drive it with whatever the solver made
of that."""


def posture(**joints: float) -> tuple[float, ...]:
    """A 22-vector from the joints named, every other joint at zero.

    ``posture(thumb_CMC_FE=1.1, index_PIP=0.9)`` reads as the posture it is;
    a bare tuple of 22 numbers does not.
    """
    unknown = set(joints) - set(JOINTS)
    if unknown:
        raise KeyError(f"no such Sharpa Wave joint: {sorted(unknown)}")
    return tuple(float(joints.get(name, 0.0)) for name in JOINTS)


FLAT = posture()
"""The flat hand: every joint at zero, which every range admits."""

CURL = {
    "middle_MCP_FE": 1.2, "middle_PIP": 1.4, "middle_DIP": 0.9,
    "ring_MCP_FE": 1.2, "ring_PIP": 1.4, "ring_DIP": 0.9,
    "pinky_CMC": 0.15, "pinky_MCP_FE": 1.2, "pinky_PIP": 1.4, "pinky_DIP": 0.9,
}  # fmt: skip
"""The three fingers a pinch does not use, curled into the palm and out of the
way: with them straight, a hand coming down on a thing on a table puts them
on the table first. Their hulls overlap each other by a few millimetres there,
as the real fingers press together; nothing else touches."""

PINCH = posture(
    thumb_CMC_FE=1.113, thumb_CMC_AA=-0.109, thumb_MCP_FE=0.198, thumb_MCP_AA=-0.088, thumb_IP=1.579,
    index_MCP_FE=0.8, index_PIP=1.0, index_DIP=0.5, **CURL,
)  # fmt: skip
"""Thumb to index fingertip, the elastomers meeting, the other fingers curled.
The index finger is curled a right angle at its middle joint so its last
phalanx points out of the palm -- what the hand comes down along -- and the
thumb's five joints were solved by forward kinematics to bring its elastomer
onto the index finger's. The two meet 12.3 mm apart, their own thickness."""

READY = posture(
    thumb_CMC_FE=0.984, thumb_CMC_AA=-0.145, thumb_MCP_FE=0.183, thumb_MCP_AA=0.347, thumb_IP=0.318,
    index_MCP_FE=0.8, index_PIP=1.0, index_DIP=0.5, **CURL,
)  # fmt: skip
"""The pinch held open: the index finger as in :data:`PINCH`, the thumb swung
45 mm away from it across the palm (along the hand's ``y``, to the tenth of a
millimetre, with no offset along the approach), solved the same way. What the
scalar command's *open* is, so that a hand coming down to pinch is already
shaped to, the way a jaw arrives open rather than flat -- straight, the index
finger reaches past the pinch point and meets the table before the pinch
point meets the thing."""

OPEN = READY
"""The scalar command's open end. The flat hand is :data:`FLAT`."""

FIST = posture(
    thumb_CMC_FE=1.5, thumb_MCP_FE=1.0, thumb_IP=1.2,
    index_MCP_FE=1.4, index_PIP=1.5, index_DIP=1.0,
    middle_MCP_FE=1.4, middle_PIP=1.5, middle_DIP=1.0,
    ring_MCP_FE=1.4, ring_PIP=1.5, ring_DIP=1.0,
    pinky_CMC=0.2, pinky_MCP_FE=1.4, pinky_PIP=1.5, pinky_DIP=1.0,
)  # fmt: skip
"""A power grasp: every finger curled to near its limits, the thumb over them."""

POINT = posture(
    thumb_CMC_FE=1.2, thumb_MCP_FE=0.8, thumb_IP=0.8,
    middle_MCP_FE=1.4, middle_PIP=1.5, middle_DIP=1.0,
    ring_MCP_FE=1.4, ring_PIP=1.5, ring_DIP=1.0,
    pinky_CMC=0.2, pinky_MCP_FE=1.4, pinky_PIP=1.5, pinky_DIP=1.0,
)  # fmt: skip
"""The index finger out, the rest curled: what presses a button or a key."""

GAIN = 3.0
"""How much stiffer than the file the shipped servo is. Measured on a 25 mm,
50 g cube pinched off the bench through the studio's own gesture (down, Close,
up; ``tests/test_sharpa_wave_grasp.py``): at the file's ``kp`` the thumb
stalls on the cube and the cube slides out as the hand rises, 0 lifts in 3;
at three times the ``kp`` (and twice the damping, for the same ratio) it comes
up and stays, 3 in 3, drifting 5 mm. Sharpa's file states its gains for
Isaac Lab's position mode, which closes its own loop; here the ``<position>``
servo is the whole loop."""

FILE_ACTUATION = ActuationSpec(
    target_ke=4.76, target_kd=0.2, target_ke_per_dof=KP, target_kd_per_dof=KV, require_position_mode=True
)
"""The file's own gains, per joint. ``rss.robot(..., gripper=FILE_ACTUATION)``
drives the hand as Sharpa's MJCF does."""

ACTUATION = ActuationSpec(
    target_ke=4.76 * GAIN,
    target_kd=0.2 * 2.0,
    target_ke_per_dof=tuple(GAIN * k for k in KP),
    target_kd_per_dof=tuple(2.0 * k for k in KV),
    require_position_mode=True,
)
"""The shipped servo: the file's gains times :data:`GAIN`."""

SHARPA_WAVE = register_gripper(
    GripperSpec(
        name="sharpa_wave",
        # Driven joints, templated per arm: the composed arms carry the hand's
        # own labels, so the right arm's hand is ``right_thumb_CMC_FE`` and the
        # left arm's ``left_thumb_CMC_FE``.
        finger_names=tuple(f"{{arm}}_{name}" for name in JOINTS),
        finger_open_q=OPEN,
        finger_closed_q=PINCH,
        postures={"flat": FLAT, "open": READY, "pinch": PINCH, "fist": FIST, "point": POINT},
        # A finger's joint coordinate is an angle; the pinch gap is a curve of
        # several of them. Say so, or a caller divides metres by radians.
        linear_travel=False,
        # The ready hand's thumb-to-index elastomer distance, and the thumb CMC
        # flexion range as the "travel" the studio's readout scales by.
        span=0.045,
        open_width=1.92,
        # The tool-centre point is the middle of the *open* pinch, in the palm
        # body's frame: only the thumb moves to close, so a thing centred here
        # clears both pads on the way down and is pushed onto the index finger
        # by the thumb. The grasp frame: +z the approach (out of the palm,
        # where the curled index finger points), +y the closing direction
        # (towards the thumb).
        tcp_body="{arm}_hand_C_MC",
        tcp_offset=(0.0779, 0.0528, 0.1375),
        tcp_offset_rot=(0.0, 0.7833, 0.0, 0.6216),
        approach_axis=(0.9738, 0.0, -0.2272),
        close_axis=(0.0, 1.0, 0.0),
        # The two distal phalanges of the pinch, one side each: what a grasp
        # test asks "is both of these touching it".
        finger_bodies=(("{arm}_thumb_DP",), ("{arm}_index_DP",)),
        # Past the pinch point along the approach, the thumb's own knuckle; across the knuckles.
        tip_reach=0.02,
        housing_half=0.076,
        actuation=ACTUATION,
    )
)

SHARPA_WAVE_LEFT = register_gripper(
    replace(
        SHARPA_WAVE,
        name="sharpa_wave_left",
        tcp_offset=(0.0779, -0.0528, 0.1375),
        tcp_offset_rot=(0.0, 0.7833, 0.0, -0.6216),
        close_axis=(0.0, -1.0, 0.0),
    )
)
"""The left hand: the pinch point and the closing direction mirrored across
the palm, measured on the left file the same way (the same postures, every
``y`` negated to the tenth of a millimetre)."""
