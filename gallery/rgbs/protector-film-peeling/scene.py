"""Task 4: contact-driven protector-film peeling, with a fixed fitted camera."""

# Array additions below are vector offsets, not list concatenations.
# ruff: noqa: RUF005

from dataclasses import replace
from pathlib import Path

import film_appearance  # noqa: F401 -- registers the display-only tape variant
import numpy as np
from motion import Drive
from scipy.spatial.transform import Rotation

import robosimstudio as rss
from robosimstudio.core.render.finish import MATTE, Finish
from robosimstudio.core.render.room import LAB
from robosimstudio.robots.registry import get_embodiment, get_gripper
from robosimstudio.solvers import grasp_quat
from robosimstudio_materials.tape import PAPER_LABEL, TAPE_60HZ, tape

ROOT = Path(__file__).parent
PHOTO = rss.CameraSpec(
    name="photo",
    width=1920,
    height=1080,
    fx=1990.4,
    fy=1990.4,
    cx=960.0,
    cy=540.0,
    eye=(-0.2737, -0.3821, 1.3623),
    look_at=(0.5505, 0.0108, 0.9544),
    up=(0.3608, 0.191, 0.9129),
)
WIDE = rss.CameraSpec(
    name="wide", width=1280, height=720, eye=(-0.18, -0.05, 1.60), look_at=(0.55, 0.08, 1.00), fov_y=55
)
scene = rss.Scene(
    cell="bench_075",
    name="protector-film-peeling",
    physics=replace(
        TAPE_60HZ.with_backend(
            "*",
            rss.backend(
                "vbd",
                iterations=10,
                rigid_body_contact_buffer_size=256,
                rigid_body_particle_contact_buffer_size=1024,
                rigid_compliant_alm=True,
            ),
        ),
        collision=replace(TAPE_60HZ.collision, rigid_gap=0.0003, contact_ke=1.0e5, contact_kd=100.0),
    ),
    render=replace(LAB, key_intensity=2.2, ambient=(1.7, 1.7, 1.7)),
)
scene.add(PHOTO)
scene.add(WIDE)


def robot_look(builder, shapes):
    count = 0
    for s in range(*shapes):
        b = builder.shape_body[s]
        label = builder.body_label[b].rsplit("/", 1)[-1] if b >= 0 else ""
        if "link_2" in label or "link_3" in label:
            builder.shape_color[s] = (0.88, 0.88, 0.85)
            count += 1
    return count


yam = get_embodiment("yam")
HOME_L = (-1.70632744, 0.0, 0.0, -0.12746412, -1.0720396, -1.68289387)
HOME_R = (0.0, 0.0, 0.0, -0.35, -0.47, -0.38)
yam = replace(
    yam,
    arms=(replace(yam.arms[0], home_q=HOME_L), replace(yam.arms[1], home_q=HOME_R)),
    color=(0.055, 0.06, 0.065),
    finish=Finish(roughness=0.5, metallic=0),
    appearance=robot_look,
)
robot = scene.add(rss.robot(yam, arms_at={"left": (-0.210, 0.4008, 0.010), "right": (-0.193, -0.20056, 0.010)}))
# A physical mat under the renderer's procedurally drawn lab mat.
scene.add(rss.slab(name="mat", half=(0.2975, 0.65, 0.0015), at=(0, 0, 0.0015), color=(0.55, 0.58, 0.59), finish=MATTE))
for i in range(-11, 12):
    scene.add(
        rss.slab(
            name=f"grid_x_{i + 11}",
            half=(0.00085, 0.65, 0.00003),
            at=(i * 0.025, 0, 0.00303),
            color=(0.97, 0.98, 0.98),
            finish=MATTE,
        )
    )
for i in range(-26, 27):
    scene.add(
        rss.slab(
            name=f"grid_y_{i + 26}",
            half=(0.2975, 0.00085, 0.00003),
            at=(0, i * 0.025, 0.00303),
            color=(0.97, 0.98, 0.98),
            finish=MATTE,
        )
    )
# The foam mat is tiled; stepped seams are drawn as physical shallow outlines.
for j, y in enumerate([-0.30, 0.0, 0.30]):
    for k in range(24):
        x = -0.2975 + k * 0.025
        offset = 0.004 if k % 2 else 0.0
        scene.add(
            rss.slab(
                name=f"seam_run_{j}_{k}",
                half=(0.0125, 0.00035, 0.00001),
                at=(x + 0.0125, y + offset, 0.00307),
                color=(0.37, 0.39, 0.39),
                finish=MATTE,
            )
        )
        if k:
            scene.add(
                rss.slab(
                    name=f"seam_step_{j}_{k}",
                    half=(0.00035, 0.002, 0.00001),
                    at=(x, y + 0.002, 0.00307),
                    color=(0.37, 0.39, 0.39),
                    finish=MATTE,
                )
            )
scene.add(rss.slab(name="ledge_collision", half=(0.02, 0.71, 0.405), at=(0.3175, 0, -0.345), visible=False))
scene.add(rss.slab(name="glass_collision", half=(0.004, 1.55, 0.95), at=(0.3875, 0, 0.20), visible=False))
for i, y in enumerate(np.arange(-1.55, 1.55, 0.105)):
    scene.add(
        rss.slab(name=f"slat_collision_{i}", half=(0.01, 0.03, 0.95), at=(0.3675, float(y) + 0.03, 0.20), visible=False)
    )
for i, y in enumerate([-0.66, 0.66]):
    scene.add(rss.slab(name=f"rig_collision_{i}", half=(0.015, 0.015, 0.77), at=(0.3175, y, 0.83), visible=False))
ANGLE = 2.0
R = Rotation.from_euler("y", -ANGLE, degrees=True).as_matrix()
Q = tuple(Rotation.from_matrix(R).as_quat())
PHONE_AT = np.array([0.050, 0.025, 0.044])
PHONE_HALF = (0.0735, 0.036, 0.004)
# Solid white printed block with an open slot at its near end.
scene.add(
    rss.slab(
        name="cradle_body",
        half=(0.047, 0.0285, 0.0175),
        at=(0.0555, 0.025, 0.0205),
        color=(0.92, 0.91, 0.86),
        finish=MATTE,
        friction=0.8,
    )
)
for i, y in enumerate([-0.019, 0.019]):
    scene.add(
        rss.slab(
            name=f"cradle_slot_side_{i}",
            half=(0.0055, 0.0095, 0.0175),
            at=(-0.0025, 0.025 + y, 0.0205),
            color=(0.92, 0.91, 0.86),
            finish=MATTE,
            friction=0.8,
        )
    )
# The near-end slot is shallow: the block continues beneath it.
scene.add(
    rss.slab(
        name="cradle_slot_floor",
        half=(0.00825, 0.0285, 0.0145),
        at=(0.00025, 0.025, 0.0175),
        color=(0.92, 0.91, 0.86),
        finish=MATTE,
        friction=0.8,
    )
)
seat = PHONE_AT + R @ np.array([0, 0, -0.006])
scene.add(
    rss.slab(
        name="cradle_seat",
        half=(0.0525, 0.0285, 0.002),
        at=(*seat, *Q),
        color=(0.92, 0.91, 0.86),
        finish=MATTE,
        friction=0.8,
    )
)
phone = scene.add(
    rss.box(
        name="phone",
        half=PHONE_HALF,
        at=tuple(PHONE_AT),
        quat=Q,
        mass=0.20,
        friction=0.8,
        color=(0.018, 0.022, 0.027),
        finish=Finish(
            roughness=0.12,
            texture=str(ROOT / "assets/phone_surface.png"),
            uv_scale=(1 / 0.147, 1 / 0.072),
            uv_offset=(0.5, 0.5),
        ),
    )
)
# Sheet frame: x across the screen; y from its raised far edge towards the lower end.
RF = R @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
FILM_AT = PHONE_AT + R @ np.array([0.068, -0.030, 0.0045])
FILM_MATERIAL = replace(
    PAPER_LABEL,
    areal_density=0.06,
    thickness=0.00006,
    particle_radius=0.0005,
    tack=2e6,
    break_opening=0.0015,
    friction=0.3,
    edge_ke=0.1,
)
film = scene.add(
    tape(
        name="film",
        entity="protector_film",
        on="phone",
        size=(0.060, 0.145),
        cells=(12, 29),
        at=tuple(FILM_AT),
        quat=tuple(Rotation.from_matrix(RF).as_quat()),
        tab=0.008,
        tab_size=(0.062, 0.008, 0.024),
        tab_mass=0.03,
        tab_hold=3e3,
        tab_color=(0.78, 0.80, 0.83),
        tab_friction=0.8,
        material=FILM_MATERIAL,
        color=(0.78, 0.80, 0.83),
        opacity=0.58,
        finish=Finish(
            roughness=0.35,
            opacity=0.58,
            ior=1.4,
            texture=str(ROOT / "assets/film_roughness.png"),
            uv_scale=(1 / 0.06, 1 / 0.145),
            roughness_texture=str(ROOT / "assets/film_roughness.png"),
        ),
    )
)


def measurements(sim):
    f = sim["film"].entity
    p = np.array(sim["phone"].position)
    pose = np.asarray(sim["phone"].pose)
    local = (sim["film"].particle_q - pose[:3]) @ Rotation.from_quat(pose[3:]).as_matrix()
    clearance = np.linalg.norm(np.maximum(np.abs(local) - PHONE_HALF, 0), axis=1).min() * 1000
    return {
        "time": float(sim.time),
        "peeled": f.peeled(),
        "bonded": f.bonded(),
        "film_clearance_mm": float(clearance),
        "phone_displacement_mm": float(np.linalg.norm(p - (PHONE_AT + np.array([0.6, 0, 0.75]))) * 1000),
        "tab": f.tab_pose(sim.state)[:3].tolist(),
        "pull_N": f.pull(sim.state),
        "tab_height_mm": float((f.tab_pose(sim.state)[2] - 0.75) * 1000),
        "film_min_height_mm": float((sim["film"].particle_q[:, 2].min() - 0.753) * 1000),
        "left_tcp_height_mm": float((sim.arm("left").ee_pose(sim.state)[0][2] - 0.75) * 1000),
        "left_jaw_mm": float(2000 * abs(sim.state.joint_q.numpy()[sim.scene.handles.arms["left"].gripper_coords[0]])),
    }


def grasp_targets(sim):
    hand = get_gripper("yam_linear_arena")
    a = np.deg2rad(110)
    ql = grasp_quat(
        hand.approach_axis,
        hand.close_axis,
        hand.tcp_offset_rot,
        (R[:, 0] * np.cos(a) + np.array([0, -1, 0]) * np.sin(a)) * np.cos(np.deg2rad(3.5))
        - R[:, 2] * np.sin(np.deg2rad(3.5)),
        -R[:, 2],
    )
    pose = np.asarray(sim["phone"].pose)
    pl = pose[:3] + Rotation.from_quat(pose[3:]).apply([-0.065, 0.048, 0.001])
    tab = sim["film"].entity.tab_pose(sim.state)
    rt = Rotation.from_quat(tab[3:]).as_matrix()
    qr = grasp_quat(hand.approach_axis, hand.close_axis, hand.tcp_offset_rot, [0, 0.819152, -0.573576], [-1, 0, 0])
    pr = tab[:3] - rt[:, 0] * 0.020 + np.array([0, 0, 0.014])
    return pl, ql, pr, qr


def perform(sim):
    d = Drive(sim)
    try:
        yield from d.wait(0.0)
        pl, ql, pr, qr = grasp_targets(sim)
        yield from d.close("left", 0.80, 0.5)
        ik = sim.arm("left")
        seed = sim.state.joint_q.numpy().copy()
        seed[ik.arm_coords] = [-0.56952173, 2.39136004, 0.82357496, 1.51327801, 1.34982729, -1.48480880]
        pre = ik.solve_q_full(seed, pl + [0, 0.065, 0], ql, iterations=160)
        end = ik.solve_q_full(seed, pl, ql, iterations=160)
        yield from d.home("left", pre[ik.arm_coords], 2.0)
        yield from d.home("left", end[ik.arm_coords], 0.75)
        d.seed["left"] = end
        yield from d.close("left", 0.92, 0.5)
        print("left brace", measurements(sim))
        yield from d.wait(1.0)
        pl, ql, pr, qr = grasp_targets(sim)
        ik = sim.arm("right")
        seed = sim.state.joint_q.numpy().copy()
        seed[ik.arm_coords] = HOME_R
        pre = ik.solve_q_full(seed, pr + [0, -0.04915, 0.03441], qr, iterations=120)
        yield from d.home("right", pre[ik.arm_coords], 1.5)
        d.seed["right"] = pre
        yield from d.close("right", 0.40, 0.5)
        yield from d.move("right", pr, qr, 1.0)
        bias = np.zeros(3)
        for _ in range(2):
            pl, ql, pr, qr = grasp_targets(sim)
            actual = sim.arm("right").ee_pose(sim.state)[0]
            bias += np.clip(pr - actual, -0.004, 0.004)
            yield from d.move("right", pr + bias, qr, 0.25)
        yield from d.wait(20.0 - sim.time)
        # One final alignment to the measured free corner after the long open hover.
        _pl, _ql, pr, qr = grasp_targets(sim)
        yield from d.move("right", pr + bias, qr, 0.5)
        actual = sim.arm("right").ee_pose(sim.state)[0]
        bias += np.clip(pr - actual, -0.002, 0.002)
        yield from d.move("right", pr + bias, qr, 0.5)
        d.begin_close()
        yield from d.close("right", 1.0, 2.0)
        d.mark_pinch()
        print("right pinch", measurements(sim))
        if d.close_state["peeled"] > 0.03:
            raise RuntimeError("Closing the right fingers broke more than 3% of the bonds")
        p = np.array(sim.arm("right").ee_pose(sim.state)[0])
        hand = get_gripper("yam_linear_arena")

        def peel_quat(degrees):
            a = np.deg2rad(35 + degrees * 0.35)
            yaw = np.deg2rad(degrees * 0.5)
            return grasp_quat(
                hand.approach_axis,
                hand.close_axis,
                hand.tcp_offset_rot,
                [0, np.cos(a), -np.sin(a)],
                [-np.cos(yaw), -np.sin(yaw), 0],
            )

        yield from d.move("right", p + [0, -0.005, 0.018], qr, 1.0)
        for target, tilt, seconds in [
            ([-0.010, -0.008, 0.025], 5, 0.75),
            ([-0.020, -0.012, 0.030], 10, 0.75),
            ([-0.033, -0.019, 0.035], 15, 52 / 60),
            ([-0.045, -0.025, 0.040], 20, 53 / 60),
        ]:
            yield from d.move("right", p + target, peel_quat(tilt), seconds)
            print("peel", measurements(sim))
        # The near hand releases at 27 s, then returns at 33 s.
        yield from d.close("left", 0.8, 0.5)
        yield from d.move("left", pl + [0, 0.065, 0.025], ql, 1.0)
        print("left release", measurements(sim))
        yield from d.move("right", p + [-0.075, -0.045, 0.075], peel_quat(25), 3.75)
        pl, ql, _pr, _qr = grasp_targets(sim)
        # Keep the known left IK branch; the rim grip leaves the near rows free to peel.
        ik = sim.arm("left")
        seed = sim.state.joint_q.numpy().copy()
        seed[ik.arm_coords] = [-0.56952173, 2.39136004, 0.82357496, 1.51327801, 1.34982729, -1.48480880]
        pre = ik.solve_q_full(seed, pl + [0, 0.065, 0], ql, iterations=160)
        end = ik.solve_q_full(seed, pl, ql, iterations=160)
        yield from d.home("left", pre[ik.arm_coords], 0.4)
        yield from d.home("left", end[ik.arm_coords], 0.35)
        d.seed["left"] = end
        yield from d.close("left", 0.92, 0.25)
        yield from d.move("right", p + [-0.115, -0.070, 0.130], peel_quat(30), 0.75)
        yield from d.wait(0.5)
        print("left rebrace", measurements(sim))
        yield from d.move("right", p + [-0.120, -0.075, 0.140], peel_quat(30), 1.5)
        print("peel finish", measurements(sim))
        yield from d.close("left", 0.8, 0.5)
        yield from d.move("left", pl + [0, 0.065, 0.025], ql, 0.75)
        targets = {}
        for arm, target, quat in [
            ("left", pl + [0, 0.085, 0.150], ql),
            ("right", np.array([0.64, -0.04, 1.01]), peel_quat(30)),
        ]:
            ik = sim.arm(arm)
            sol = ik.solve_q_full(sim.state.joint_q.numpy(), target, quat, iterations=160)
            pe, re = ik.pose_errors(sol, target, quat)
            if pe > 0.005 or re > 0.05:
                raise RuntimeError(f"{arm} raised finish IK miss {pe * 1000:.1f} mm / {re:.3f} rad")
            targets[arm] = sol[ik.arm_coords]
        d.action[6] = 0.4
        yield from d.homes(targets, 1.75)
        result = measurements(sim)
        result["peak_pull_N"] = d.peak_pull
        result["max_phone_displacement_mm"] = d.max_phone_mm
        print("RESULT", result, "PINCH_LOAD", d.load_peak, "CROSSING", d.crossing)
        if d.close_state is None or d.close_state["peeled"] > 0.03:
            raise RuntimeError("Closing the right fingers broke more than 3% of the bonds")
        if result["peeled"] < 0.95:
            raise RuntimeError(f"Only {result['peeled']:.1%} of film bonds broke")
        if (
            d.crossing is None
            or d.crossing["grip_drift_mm"] is None
            or d.crossing["tab_tcp_mm"] > 40
            or not d.held_in_pads(d.crossing.get("finger_contacts", {}))
        ):
            raise RuntimeError("95% crossing was not held in the right pinch")
        if result["film_clearance_mm"] < 3:
            raise RuntimeError("Film is not clear of the phone")
        if result["film_min_height_mm"] < 20:
            raise RuntimeError("Film is not hanging clear of the mat")
        for stamp in ("12", "34.8"):
            pads = d.left_contacts.get(stamp, {})
            heights = [pt[2] for pad in pads.values() for pt in pad["points_local_mm"]]
            if len(pads) < 2 or not heights or min(heights) > -3.5 or max(heights) < 3.5:
                raise RuntimeError(f"Left top/bottom grasp absent at {stamp} s")
        if not d.held_in_pads():
            raise RuntimeError("Film is not hanging from both right pads at the finish")
        if d.max_phone_mm >= 10:
            raise RuntimeError("Phone left its seat")
    finally:
        d.save()


def routine(sim):
    yield from perform(sim)


routines = {"routine": routine}
