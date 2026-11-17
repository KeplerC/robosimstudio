from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import newton
import numpy as np
import trimesh
import warp as wp

HERE = Path(__file__).resolve().parent

import robosimstudio as rss  # noqa: E402
from robosimstudio.core.math import mat_to_quat, quat_to_mat  # noqa: E402
from robosimstudio.core.render.finish import Finish  # noqa: E402
from robosimstudio.domains.rigid.entity import EntityRigid  # noqa: E402
from robosimstudio.scene.registry import register_entity  # noqa: E402
from robosimstudio.specs import (  # noqa: E402
    ActuationSpec,
    ArrangementSpec,
    AssetSpec,
    Binding,
    CollisionSpec,
    ObjectSpec,
    StaticSpec,
    backend,
)

ASSETS = HERE / "assets"
SURFACE = np.array([0.6, 0.0, 0.75])
"""bench_075's work-surface origin in world [m]: the centre of its top."""
BENCH_Z = 0.75


# ------------------------------------------------------------------ physics
def physics(impratio: float = 30.0, cone: str = "elliptic") -> ArrangementSpec:
    return ArrangementSpec(
        control_hz=60.0,
        substeps=8,
        backends=(
            ("*", backend("mujoco", use_mujoco_contacts=False, nconmax=4096, njmax=8192, iterations=50, ls_iterations=100, cone=cone, impratio=impratio)),
        ),
        collision=CollisionSpec(rigid_contact_max=16384, rigid_gap=1.0e-3),
    )


PHYSICS = physics()
"""One MuJoCo island, Newton contacts and 480 Hz stepping. Elliptic cones at impedance ratio 30 are inherited from the old contact model; the native-finger measurements are in REPORT.md."""

# -------------------------------------------------------------- the plate
PLATE = json.loads((ASSETS / "plate" / "plate.json").read_text())
PLATE_RADIUS = float(PLATE["R"])
PLATE_DEPTH = float(PLATE["depth"])
PLATE_MASS = float(PLATE["mass"])
PLATE_FRICTION = 0.7
PLATE_KE = 2.0e5
PLATE_KD = 400.0
"""Contact stiffness/damping on the plate's collider pieces [N/m, N s/m]. A bare
OBJ arrives at Newton's 2500 N/m -- 4 mm of give per 10 N."""
PLATE_TORSION = 0.01
PLATE_ROLLING = 0.008
"""Torsional and rolling friction radii [m], inherited assumptions representing a finite contact patch. See REPORT.md for values and validation limits."""
PLATE_SOLIMP = None
"""MuJoCo ``solimp`` on the plate's pieces (None: MuJoCo's default)."""

WOOD = (0.62, 0.43, 0.26)
WOOD_DARK = (0.47, 0.30, 0.17)
RACK_WHITE = (0.90, 0.90, 0.88)
SHELF_FRICTION = 0.8


def _flange_top(r: float) -> float:
    return float(PLATE["well_top"]) + max(0.0, r - float(PLATE["flange_start"])) * math.tan(float(PLATE["flange_angle"]))


RIM = {
    "R": PLATE_RADIUS,
    "flange_angle": float(PLATE["flange_angle"]),
    "thickness": float(PLATE["thickness"]),
}
"""The rim as designed (``make_assets.py``): read, never measured off the mesh."""


def flange_point(r: float) -> np.ndarray:
    """The flange's mid-surface at radius ``r`` on the plate's local +y spoke [m]."""
    half = 0.5 * float(PLATE["thickness"]) / math.cos(float(PLATE["flange_angle"]))
    return np.array([0.0, r, _flange_top(r) - half])


def flange_thickness(r: float, spoke: float = 0.0) -> float:
    del r, spoke
    return float(PLATE["thickness"])


@register_entity("authored_plate")
class EntityPlate(EntityRigid):
    """The plate: its revolved mesh drawn, its own-vertex hulls collided.

    Two additions to the library's rigid entity: the mass properties are the
    watertight mesh's own at the stated density, and the plate's pieces get
    torsional and rolling friction and the contact dimension (6) that lets
    MuJoCo apply them.
    """

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        first = builder.shape_count
        super().build(builder, episode)
        condim = builder.custom_attributes.get("mujoco:condim")
        if condim is None:
            from newton.solvers import SolverMuJoCo

            SolverMuJoCo.register_custom_attributes(builder)
            condim = builder.custom_attributes["mujoco:condim"]
        if condim.values is None:
            condim.values = {}
        solimp = builder.custom_attributes["mujoco:geom_solimp"]
        if solimp.values is None:
            solimp.values = {}
        for s in range(first, builder.shape_count):
            builder.shape_material_mu_torsional[s] = PLATE_TORSION
            builder.shape_material_mu_rolling[s] = PLATE_ROLLING
            condim.values[s] = 6
            if PLATE_SOLIMP is not None:
                solimp.values[s] = PLATE_SOLIMP
        mass = PLATE_MASS
        inertia = np.asarray(PLATE["inertia_unit_density"], dtype=np.float64) * float(PLATE["density"])
        body = self.body
        builder.body_mass[body] = mass
        builder.body_inv_mass[body] = 1.0 / mass
        builder.body_com[body] = wp.vec3(*[float(c) for c in PLATE["com"]])
        builder.body_inertia[body] = wp.mat33(*inertia.reshape(-1).tolist())
        builder.body_inv_inertia[body] = wp.mat33(*np.linalg.inv(inertia).reshape(-1).tolist())


def plate_spec(name: str, pos_world, quat_xyzw) -> ObjectSpec:
    hulls = tuple(str(p) for p in sorted((ASSETS / "plate").glob("hull_*.obj")))
    return ObjectSpec(
        name=name,
        category="plate",
        entity="authored_plate",
        asset=AssetSpec(
            path=str(ASSETS / "plate" / "plate.obj"),
            collision_paths=hulls,
            texture=str(ASSETS / "plate" / "texture.png"),
        ),
        pose=tuple(float(v) for v in np.asarray(pos_world) - SURFACE),
        quat=tuple(float(v) for v in quat_xyzw),
        density=0.0,
        friction=PLATE_FRICTION,
        ke=PLATE_KE,
        kd=PLATE_KD,
        finish=Finish(roughness=0.18),
    )


_HULL_POINTS = None


def lowest_offset(R: np.ndarray) -> float:
    """How far below the plate's origin its lowest collider point is, at orientation ``R`` [m] (negative)."""
    global _HULL_POINTS
    if _HULL_POINTS is None:
        _HULL_POINTS = np.vstack(
            [trimesh.load(p, force="mesh").vertices for p in sorted((ASSETS / "plate").glob("hull_*.obj"))]
        )
    return float((_HULL_POINTS @ R.T)[:, 2].min())


def plate_quat(lean: float = 0.0, facing: float = -1.0) -> np.ndarray:
    """A standing plate's orientation, xyzw.

    The plate's own +z (out of its eating side) points along world ``facing`` x
    (-1: toward the robot); its in-plane "up" is world +z tipped by ``lean``
    about world y (positive puts the top toward +x). Upright is ``lean = 0``.
    """
    up = np.array([math.sin(lean), 0.0, math.cos(lean)])
    normal = np.array([facing * math.cos(lean), 0.0, -facing * math.sin(lean)])
    x_axis = np.cross(up, normal)
    return mat_to_quat(np.column_stack([x_axis, up, normal]))


# --------------------------------------------------------- slots and racks
SLOT_PITCH = 0.043
"""Slot to slot [m]: the plate's collider is 27.5 mm deep front to back, a divider
is 8 mm across, and the plate gets 3.75 mm either side -- the room a plate held
by its rim has to come in along the slot with."""
DIVIDER = (0.004, 0.110)
"""A divider dowel's radius and its height over the board [m]. Tall enough to meet
a leaning plate's foot ring high up, which is what keeps the lean small: the
plate has 3.75 mm to fall back through, and at 85 mm tall the dowels stopped it
at 3.0-4.0 degrees."""
DIVIDER_Y = 0.045
"""Each divider row stands this far either side of the slot's centre line [m].

**This is what makes a plate stand upright.** A plate standing on its rim leans
toward its back (its centre of mass is behind the lip it stands on) until
something stops it. Out at the plate's sides only the thin rim passes a dowel,
set well toward the eating face, so it leans ~11 degrees before its back finds
one. At 45 mm out the dowels pass the foot ring -- the back-most part of the
plate -- from 66 mm up, and the plate stops within ~2 degrees of upright."""
CRADLE = (0.005, 0.030)
"""The two rails a plate's rim sits on: radius, and how far either side of the
slot's centre line [m]. A plate on its rim on a flat board rolls away like a
wheel; on two rails it cannot."""
BOARD = 0.015
"""Board thickness [m]."""
END_PANEL = (0.012, 0.13)
"""The shelf's end panels: thickness and height over the board [m]."""
RACK_PITCH = 0.052
"""Drying-rack divider spacing [m], wider than the shelf for extraction clearance. Each rack holds every other shelf-slot plate."""


def board_top() -> float:
    return BENCH_Z + BOARD


def rest_foot() -> float:
    """Height of a seated plate's lowest collider point over the bench [m]: on the cradle rails."""
    r_c, y_c = CRADLE
    Rc = PLATE_RADIUS + float(PLATE["skin"])
    sag = Rc - math.sqrt(Rc * Rc - y_c * y_c)
    return BOARD + 2.0 * r_c - sag


def rack_specs(name: str, x_slots: list[float], centre_y: float, *, color, end_panels: bool,
               pitch: float = SLOT_PITCH, board_margin: float = 0.03, width: float = 0.17) -> list[StaticSpec]:
    """A slotted rack: a board, a cradle, a divider row either side of each slot, maybe end panels.

    ``pitch`` is the distance between the two divider rows of one slot; slots
    that are exactly one pitch apart share the row between them. Boxes and
    capsules only, so what is drawn is exactly what collides.
    """
    xs = sorted(float(x) for x in x_slots)
    x0 = xs[0] - 0.5 * pitch - board_margin
    x1 = xs[-1] + 0.5 * pitch + board_margin
    look = dict(color=color, friction=SHELF_FRICTION, finish=Finish(roughness=0.55))
    cx = 0.5 * (x0 + x1)
    out = [
        StaticSpec(
            name=f"{name}_board",
            half_extents=(0.5 * (x1 - x0), 0.5 * width, 0.5 * BOARD),
            xform=(cx - SURFACE[0], centre_y - SURFACE[1], 0.5 * BOARD, 0.0, 0.0, 0.0, 1.0),
            **look,
        )
    ]
    r_c, y_c = CRADLE
    along_x = _quat_z_to(np.array([1.0, 0.0, 0.0]))
    for j, dy in enumerate((-y_c, y_c)):
        out.append(
            StaticSpec(
                name=f"{name}_cradle{j}",
                shape=rss.ShapeSpec(kind="capsule", radius=r_c, half_height=0.5 * (x1 - x0) - r_c - 0.004),
                xform=(cx - SURFACE[0], centre_y + dy - SURFACE[1], BOARD + r_c, *along_x),
                **look,
            )
        )
    r_d, h_d = DIVIDER
    walls = sorted({round(x + s * 0.5 * pitch, 6) for x in xs for s in (-1.0, 1.0)})
    k = 0
    for xw in walls:
        for dy in (-DIVIDER_Y, DIVIDER_Y):
            out.append(
                StaticSpec(
                    name=f"{name}_divider{k}",
                    shape=rss.ShapeSpec(kind="capsule", radius=r_d, half_height=0.5 * h_d - r_d),
                    xform=(xw - SURFACE[0], centre_y + dy - SURFACE[1], BOARD + 0.5 * h_d, 0.0, 0.0, 0.0, 1.0),
                    **look,
                )
            )
            k += 1
    if end_panels:
        t, h = END_PANEL
        for j, xe in enumerate((x0 + 0.5 * t, x1 - 0.5 * t)):
            out.append(
                StaticSpec(
                    name=f"{name}_end{j}",
                    half_extents=(0.5 * t, 0.5 * width, 0.5 * h),
                    xform=(xe - SURFACE[0], centre_y - SURFACE[1], BOARD + 0.5 * h, 0.0, 0.0, 0.0, 1.0),
                    **{**look, "color": WOOD_DARK},
                )
            )
    return out


def _quat_z_to(d: np.ndarray) -> tuple[float, float, float, float]:
    """xyzw rotation taking +z onto unit vector ``d``."""
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, d))
    if c > 1.0 - 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    if c < -1.0 + 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    axis = np.cross(z, d)
    axis /= np.linalg.norm(axis)
    half = 0.5 * math.acos(max(-1.0, min(1.0, c)))
    s = math.sin(half)
    return (float(axis[0] * s), float(axis[1] * s), float(axis[2] * s), float(math.cos(half)))


# ------------------------------------------------------------------ layout
@dataclass(frozen=True)
class Layout:
    shelf_x: float = 0.560
    """Shelf centre in world x [m], retaining the original six-slot span. Native YAM placement and grasp waypoints were replanned; recorded planned joint margins range from 4.6 to 16.7 degrees."""
    shelf_slots: int = 6
    rack_y: float = 0.37
    """Drying-rack centre lines at y = ±0.37 m; native arm bases are at y = ±0.21 m."""
    slots: tuple[int, ...] = (5, 4, 3, 2, 1, 0)
    """Which shelf slots are filled, in the order they are filled; slot 0 is the one
    nearest the robot. Far to near, so the plate already on the shelf is always on
    the far side of the hand -- the side the tilted wrist leans away from."""
    arms: tuple[str, ...] = ("right", "left", "right", "left", "right", "left")
    """Which arm fills each of those slots, alternating."""


LAYOUT = Layout()


def slot_x(k: int, L: Layout = LAYOUT) -> float:
    return L.shelf_x + (k - 0.5 * (L.shelf_slots - 1)) * SLOT_PITCH


def slot_world(k: int, L: Layout = LAYOUT, y: float = 0.0) -> dict:
    """Slot ``k`` of the shelf (or, at ``y``, of a rack standing at the same x): its centre plane and floor."""
    return {"x": slot_x(k, L), "y": y, "lean": 0.0, "floor_z": board_top()}


def plate_in_slot(slot: dict, y: float, foot: float) -> np.ndarray:
    """The plate upright in ``slot`` at ``y``, its lowest collider point ``foot`` over the bench [4x4].

    Its mid-depth -- halfway between its foot and its lip -- on the slot's centre
    plane, which is where it has the most room either side as it slides in.
    """
    R = quat_to_mat(plate_quat(slot["lean"]))
    mid_local = np.array([0.0, 0.0, 0.5 * PLATE_DEPTH])
    z_origin = BENCH_Z + foot - lowest_offset(R)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([slot["x"], y, z_origin + (R @ mid_local)[2]]) - R @ mid_local
    return T


def build_scene(L: Layout = LAYOUT, *, render=None, physics_spec=None,
                seed: int | None = None, spawn: list | None = None):
    """Build the local kitchen, native YAM, fixtures and six plates. The live scene uses assets/resting.json; explicit spawn poses override those baked initial poses."""
    rng = np.random.default_rng(seed) if seed is not None else None
    from kitchen import LOOK, add_kitchen
    from dataclasses import replace
    from robosimstudio.scene.registry import register_cell
    register_cell(replace(rss.get_cell("bench_075"), name="plate_kitchen", statics=()))
    scene = rss.Scene(cell="plate_kitchen", physics=physics_spec or PHYSICS, name="plate_shelf", render=render or LOOK)
    scene.add(rss.robot("yam", at=(-0.3475,0,0), arms_at={
        "left": (-0.4475, 0.21, 0.12), "right": (-0.4475, -0.21, 0.12)}))
    add_kitchen(scene)
    scene.add_all(rack_specs("shelf", [slot_x(k, L) for k in range(L.shelf_slots)], 0.0, color=WOOD, end_panels=True))
    per_side: dict[str, list[float]] = {"right": [], "left": []}
    for k, arm in zip(L.slots, L.arms):
        per_side[arm].append(slot_x(k, L))
    for arm, xs in per_side.items():
        if xs:
            scene.add_all(rack_specs(f"rack_{arm}", xs, -L.rack_y if arm == "right" else L.rack_y,
                                     color=RACK_WHITE, end_panels=False, pitch=RACK_PITCH, board_margin=0.02))
    if spawn is None and (ASSETS / "resting.json").exists():
        spawn = json.loads((ASSETS / "resting.json").read_text())
    plates = []
    for i, (k, arm) in enumerate(zip(L.slots, L.arms)):
        y = -L.rack_y if arm == "right" else L.rack_y
        T = plate_in_slot(slot_world(k, L, y), y, rest_foot() + 0.001)
        if rng is not None:
            yaw = math.radians(rng.uniform(-1.0, 1.0))
            c, s_ = math.cos(yaw), math.sin(yaw)
            T[:3, :3] = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]) @ T[:3, :3]
            T[:3, 3] += np.array([rng.uniform(-0.0015, 0.0015), rng.uniform(-0.005, 0.005), 0.0])
        pos, quat = (T[:3, 3], mat_to_quat(T[:3, :3])) if spawn is None else spawn[i]
        plates.append(scene.add(plate_spec(f"plate_{i}", pos, quat)))
    return scene, plates


def resting_poses(sim, plates, seconds: float = 1.5, mode: str = "joint_position") -> list:
    """Where the plates come to rest in their racks, the robot idle: world
    ``(position, quaternion xyzw)`` each. A plate placed upright falls the last
    millimetre onto its rack's cradle and tips back ~5 degrees onto its dividers,
    and the knock pressed a collider 0.33 mm into a divider -- past the 0.3 mm
    between it and the drawn plate. A take starts from where they end up."""
    idle = sim.idle_action(mode)
    for _ in range(int(seconds * 60.0)):
        sim.step(idle, mode=mode)
    sim.sync()
    return [(np.asarray(p.pose[:3], dtype=np.float64).copy(), np.asarray(p.pose[3:7], dtype=np.float64).copy()) for p in plates]


