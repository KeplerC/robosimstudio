#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compose UR5e arms wearing Sharpa Wave hands into the MJCFs ``assets/embodiments/ur5e_sharpa/`` ships.

One file per robot, written from the two vendored sources beside it --
Menagerie's ``embodiments/ur5e/ur5e.xml`` and Sharpa's
``embodiments/sharpa_wave/{left,right}_sharpa_wave.xml`` -- through MuJoCo's own
``MjSpec.attach``, which is the composition the hand's root body is measured
against: its flange sits on the arm's ``attachment_site`` (the tool flange,
``pos="0 0.1 0" quat="-1 1 0 0"`` on ``wrist_3_link``), exactly where Newton's
``test_robot_composer`` bolts a hand to this arm.

Why a composed file rather than the runtime weld (``GripperSpec.asset``): the
weld attaches a hand only to the articulation added last, so on a two-armed
asset one arm would be bare. A file with both arms and both hands in it has no
such rule. The arms are ``left_``/``right_`` prefixed; the hands keep the labels
their own files give (``left_thumb_CMC_FE``, ``right_hand_C_MC``): the hand's own side prefix
is stripped before the arm's is put on, so neither label says it twice. Meshes resolve into the two source folders through
``meshdir=".."``, so nothing is copied twice.

What is *not* baked: gravity compensation, homes and bases. Those are stated
on the embodiment (``robots/embodiments/ur5e_sharpa.py``), where they can be
read and overridden; the file is geometry, inertia, joints, limits and the
sources' own servos, rewritten as ``<position>`` elements so Newton drives
them from ``joint_target_q`` (see ``_servos``).

Usage, from the repository root::

    uv run python scripts/compose_ur5e_sharpa.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import mujoco  # noqa: TID253 -- a development script, run by hand

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "assets" / "embodiments"
UR5E = EMB / "ur5e" / "ur5e.xml"
HAND = {side: EMB / "sharpa_wave" / f"{side}_sharpa_wave.xml" for side in ("left", "right")}
OUT = EMB / "ur5e_sharpa"

SHOULDER_Y = 0.30
"""Half the distance between the two bases of the bimanual file [m]."""


def _arm_with_hand(side: str) -> mujoco.MjSpec:
    arm = mujoco.MjSpec.from_file(str(UR5E))
    hand = mujoco.MjSpec.from_file(str(HAND[side]))
    # The hand's own labels carry the side, and the arm it goes on gets the
    # same side as its prefix: strip the hand's, or every finger reads
    # ``left_left_thumb_CMC_FE``. Meshes keep their names; a geom refers to its
    # mesh by name. The hand's main default class is renamed too, or it lands
    # on the arm's own ``main`` and the written file names one class twice.
    for element in (*hand.bodies, *hand.joints, *hand.geoms, *hand.actuators, *hand.sites):
        if element.name.startswith(f"{side}_"):
            element.name = element.name[len(side) + 1 :]
    for actuator in hand.actuators:
        # An actuator names its joint as text; renaming the joint does not follow.
        if actuator.target.startswith(f"{side}_"):
            actuator.target = actuator.target[len(side) + 1 :]
    hand.default.name = "sharpa"
    arm.attach(hand, prefix="", site=arm.site("attachment_site"))
    return arm


def _relink_meshes(xml: str) -> str:
    """Point every mesh at its vendored source through ``meshdir=".."``.

    Done on the written text rather than on the spec: ``MjSpec.to_xml`` compiles
    the model first, and resolves each mesh against the folder its own source
    file was read from, so the spec has to keep the sources' paths until it is
    written. The arm's OBJs live in ``ur5e/assets/``, the hands' STLs in
    ``sharpa_wave/meshes/``; which is which is told by the extension, since the
    two sources use different formats.
    """
    xml = xml.replace('<compiler angle="radian"/>', '<compiler angle="radian" meshdir=".."/>', 1)

    def relink(match: re.Match[str]) -> str:
        name = Path(match.group(1)).name
        folder = "ur5e/assets" if name.lower().endswith(".obj") else "sharpa_wave/meshes"
        return f'file="{folder}/{name}"'

    return re.sub(r'file="([^"]+)"', relink, xml)


def _servos(path: Path, side: str, strip: str = "") -> list[str]:
    """One ``<position>`` line per actuator of a source file, its gains resolved.

    Written by hand rather than left to ``MjSpec.to_xml``, which writes every
    servo back as a ``<general>`` with ``gainprm``/``biasprm`` -- correct MuJoCo,
    and an element Newton's importer reads as *direct control*: a joint whose
    position target is then ignored. ``<position kp kv>`` is what it reads as a
    position servo. The numbers are the compiled source's own: ``kp`` from
    ``gainprm[0]``, ``kv`` from ``-biasprm[2]`` (which is where MuJoCo puts a
    ``dampratio`` once it has resolved it against the joint's inertia), the
    control and force ranges as declared.
    """
    model = mujoco.MjModel.from_xml_path(str(path))
    lines = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0]))
        if strip and name.startswith(strip):
            name = name[len(strip) :]
        if strip and joint.startswith(strip):
            joint = joint[len(strip) :]
        kp = float(model.actuator_gainprm[i, 0])
        kv = -float(model.actuator_biasprm[i, 2])
        lo, hi = (float(v) for v in model.actuator_ctrlrange[i])
        attrs = f'name="{side}_{name}" joint="{side}_{joint}" kp="{kp:g}" kv="{kv:.4g}"'
        if model.actuator_ctrllimited[i]:
            attrs += f' ctrlrange="{lo:.5g} {hi:.5g}"'
        if model.actuator_forcelimited[i]:
            flo, fhi = (float(v) for v in model.actuator_forcerange[i])
            attrs += f' forcerange="{flo:g} {fhi:g}"'
        lines.append(f"    <position {attrs}/>")
    return lines


def compose(sides: tuple[str, ...], model_name: str) -> str:
    root = mujoco.MjSpec()
    root.modelname = model_name
    servos: list[str] = []
    for side in sides:
        y = {"left": SHOULDER_Y, "right": -SHOULDER_Y}[side] if len(sides) > 1 else 0.0
        frame = root.worldbody.add_frame(pos=[0.0, y, 0.0])
        root.attach(_arm_with_hand(side), prefix=f"{side}_", frame=frame)
        servos += _servos(UR5E, side) + _servos(HAND[side], side, strip=f"{side}_")
    # The sources' keyframes are the arms' own homes; the embodiment states its own.
    for key in list(root.keys):
        key.delete()
    # The sources' servos, restated as <position> elements (see _servos), in
    # place of the <general> block to_xml writes.
    xml = re.sub(r"\s*<actuator>.*?</actuator>", "", root.to_xml(), flags=re.DOTALL)
    xml = xml.replace("</mujoco>", "  <actuator>\n" + "\n".join(servos) + "\n  </actuator>\n</mujoco>")
    return _relink_meshes(xml)  # checked by loading the written file, whose meshdir resolves beside it


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for sides, name in ((("right",), "ur5e_sharpa"), (("left", "right"), "ur5e_sharpa_bimanual")):
        xml = compose(sides, name)
        path = OUT / f"{name}.xml"
        path.write_text(xml)
        model = mujoco.MjModel.from_xml_path(str(path))
        print(
            f"{path.relative_to(ROOT)}: {model.njnt} joints, {model.nu} actuators, {model.nbody} bodies, {model.nmesh} meshes"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
