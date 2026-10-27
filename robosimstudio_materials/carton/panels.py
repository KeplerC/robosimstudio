# SPDX-License-Identifier: Apache-2.0
"""The panels model: a carton as rigid plates joined by crease hinges that yield, on MuJoCo.

The glued panels -- an erected box's bottom and walls, a blank's base -- are one rigid body on a
free joint (or welded to the world, a jig). Every creased panel is a plate on its own link, hinged
to that body along its crease by a revolute joint. The hinge's spring is MuJoCo's passive joint
stiffness (Newton's custom attribute ``mujoco:dof_passive_stiffness``) pulling toward the joint's
spring reference (``mujoco:dof_springref``, MuJoCo's ``qpos_spring``), plus passive damping.

The yield is a kernel. After every solve it reads each hinge's angle from MuJoCo's ``qpos`` and,
where the hinge has bent more than the board's ``yield_angle`` past its reference, carries the
reference along so it stays exactly ``yield_angle`` behind the bend: the perfectly plastic hinge of
:mod:`robosimstudio_materials.paper`'s crease kernel, on a joint instead of a mesh hinge. It writes
``qpos_spring`` on the device in place, so it runs inside the captured substep graph.

A hinge's angle is 0 as built and grows the way its panel ``opens``: a box flap opening upward, a
blank's wall folding down over the edge it lies on.

**What it cannot do.** The plates never bend, dent or buckle, so a squeeze or a stack load reads
nothing here; nor does a crease soften as it is worked. And a solver notification that re-syncs
MuJoCo's joint properties from the model (``JOINT_DOF_PROPERTIES``) puts every spring reference back
to its built value, forgetting the creases: nothing in this tree sends one to a running carton.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.specs.asset import ObjectSpec, ShapeSpec

from .board import C_FLUTE, BoardSpec
from .dieline import STYLES, Dieline, Panel

__all__ = ["CartonSpec", "EntityCarton"]


@dataclass(frozen=True, kw_only=True)
class CartonSpec(ObjectSpec):
    """A carton on the panels model. ``rss.carton`` returns one for a ``"panels"`` board.

    Attributes:
        style: Which die-line: ``"rsc"`` or ``"tray"`` (:mod:`~robosimstudio_materials.carton.dieline`).
        size: The style's size [m]: ``(L, W, H)`` for an RSC, ``(x, y, wall)`` for a tray blank.
        board: The grade, a :class:`~robosimstudio_materials.carton.board.BoardSpec` whose model is ``"panels"``.
        contents: Mass inside the box [kg], a 20 mm layer on its floor. An empty box is easy to drag.
        taped: Whether the die-line's taped flaps start taped shut.
        fixed: Weld the body to the world where it is built: a blank held on a jig.
        cell: The grid the die-line is snapped to, the sheet model's cell, so both build the same carton.
        seam: An RSC's gap between its major flaps [m].
        dieline_options: Any further keyword the style's layout function takes.
    """

    entity: str = "carton"
    name: str = "carton"
    # ObjectSpec wants a shape or an asset; the plates are built from the die-line instead.
    shape: ShapeSpec | None = field(default_factory=lambda: ShapeSpec(kind="box", half_extents=(0.01, 0.01, 0.01)))
    style: str = "rsc"
    size: tuple[float, float, float] = (0.20, 0.15, 0.08)
    board: BoardSpec = field(default_factory=lambda: C_FLUTE)
    contents: float = 0.0
    taped: bool = True
    fixed: bool = False
    cell: float = 0.01
    seam: float = 0.006
    dieline_options: tuple[tuple[str, Any], ...] = ()
    color: tuple[float, float, float] = (0.62, 0.45, 0.27)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.style not in STYLES:
            raise ValueError(f"carton {self.name!r}: style {self.style!r} is not one of {sorted(STYLES)}")
        if self.board.model != "panels":
            raise ValueError(f"carton {self.name!r}: board {self.board.name!r} is a {self.board.model}, not panels")
        if self.contents < 0.0:
            raise ValueError(f"carton {self.name!r}: contents cannot be negative, got {self.contents}")

    def dieline(self) -> Dieline:
        options = dict(self.dieline_options)
        if self.style == "rsc":
            options.setdefault("seam", self.seam)
            # The major flaps' plates over the minor flaps' by half a millimetre, whatever the board's caliper.
            options.setdefault("rise", self.board.thickness + 5.0e-4)
        return STYLES[self.style](self.size, cell=self.cell, **options)


@wp.kernel
def _yield_kernel(
    qpos_adr: wp.array(dtype=wp.int32),
    yield_angle: wp.array(dtype=wp.float32),
    active: wp.array(dtype=wp.int32),
    qpos: wp.array2d(dtype=wp.float32),
    qpos_spring: wp.array2d(dtype=wp.float32),
):
    """Carry each hinge's spring reference to within ``yield_angle`` of its angle. One thread per hinge and world.

    ``qpos_spring`` holds a row per world, or one row all worlds share, which world 0 drives.
    """
    world, i = wp.tid()
    if active[i] == 0 or world >= qpos_spring.shape[0]:
        return
    a = qpos_adr[i]
    ref = qpos_spring[world, a]
    excess = qpos[world, a] - ref
    y = yield_angle[i]
    if excess > y:
        qpos_spring[world, a] = ref + (excess - y)
    elif excess < -y:
        qpos_spring[world, a] = ref + (excess + y)


def _frame(x: np.ndarray, z: np.ndarray) -> wp.quat:
    """The rotation whose columns are ``x``, ``z x x`` and ``z``, for orthonormal ``x`` and ``z``."""
    y = np.cross(z, x)
    m = np.stack([x, y, z], axis=1)
    return wp.quat_from_matrix(wp.mat33(*m.reshape(-1).tolist()))


class EntityCarton(EntityRigid):
    """A carton built from its die-line: the glued panels one body, every creased panel a plate on a plastic hinge.

    After the scene is ready, :attr:`hinges` names each hinge, and :meth:`angles`, :meth:`references`
    and :meth:`cut_tape` read and work them between steps.
    """

    spec: CartonSpec
    domain = "carton"

    def __init__(self, spec: CartonSpec, **kwargs) -> None:
        super().__init__(spec, **kwargs)
        self.hinges: list[dict[str, Any]] = []
        """One per creased panel: its name, link, joint, coordinate, stiffness, damping, built reference and tape."""
        self.taped_now: set[str] = set()
        """The flaps whose tape is still on."""
        self._solver: Any = None
        self._qpos_adr = np.zeros(0, dtype=np.int32)
        self._jnt = np.zeros(0, dtype=np.int32)
        self._active = np.zeros(0, dtype=bool)
        self._host = False
        self._device: dict[str, Any] = {}

    # -------------------------------------------------------------- phase 1
    def build(self, builder, episode) -> None:
        spec, board = self.spec, self.spec.board
        t = board.thickness
        dieline = spec.dieline()
        pose = np.asarray(self._placed(self._episode_pose(episode)), dtype=np.float64)
        root_x = wp.transform(wp.vec3(*pose[:3].tolist()), wp.quat(*pose[3:7].tolist()))
        cfg = builder.default_shape_cfg.copy()
        cfg.density = float(board.areal_density / t)
        cfg.mu = float(board.friction)

        self.body = builder.add_link(xform=root_x, label=spec.name)
        if spec.fixed:
            joints = [builder.add_joint_fixed(-1, self.body, parent_xform=root_x, label=f"{spec.name}_jig")]
        else:
            joints = [builder.add_joint_free(child=self.body, label=f"{spec.name}_free_joint")]
            self.coord_start = int(builder.joint_q_start[joints[0]])
            self.dof_start = int(builder.joint_qd_start[joints[0]])
            builder.joint_q[self.coord_start : self.coord_start + 7] = pose[:7].tolist()

        # The glued panels: plates on the body, each inset by half the board toward the carton's middle,
        # so the bottom's underside is the carton's z = 0 and the walls' outsides its faces.
        middle = np.array([0.0, 0.0, 0.5 * dieline.size[2]])
        for panel in dieline.glued:
            u, v = np.asarray(panel.u, dtype=np.float64), np.asarray(panel.v, dtype=np.float64)
            n = np.cross(u, v)
            centre = np.asarray(panel.origin, dtype=np.float64) + 0.5 * panel.a * u + 0.5 * panel.b * v
            if np.dot(middle - centre, n) < 0.0:
                n = -n
            centre = centre + 0.5 * t * n
            x = wp.transform(wp.vec3(*centre.tolist()), _frame(u, n))
            # The frame's y is n x u, which is v up to sign: a centred plate is the same either way.
            builder.add_shape_box(
                self.body,
                xform=x,
                hx=0.5 * panel.a,
                hy=0.5 * panel.b,
                hz=0.5 * t,
                cfg=cfg,
                label=f"{spec.name}_{panel.name}",
            )
        if spec.contents > 0.0:
            self._contents(builder, cfg, dieline)

        for panel in dieline.creased:
            joints.append(self._hinged(builder, cfg, root_x, dieline, panel))
        builder.add_articulation(joints, label=f"{spec.name}_articulation")

    def _contents(self, builder, cfg, dieline: Dieline) -> None:
        """What is inside: a 20 mm layer on the floor, as a shape that adds mass and nothing else.

        As a shape rather than a mass written to the body, so the body's centre of mass and inertia are a
        box with its load at the bottom, not a heavier hollow shell: a 1 kg load written onto the shell
        sat 27 mm up, at the shell's centroid.
        """
        t = self.spec.board.thickness
        floor = next(p for p in dieline.glued if abs(np.cross(p.u, p.v)[2]) > 0.5)
        hx, hy, hz = 0.5 * floor.a - t, 0.5 * floor.b - t, 0.01
        load = cfg.copy()
        load.density = float(self.spec.contents / (8.0 * hx * hy * hz))
        load.has_shape_collision = False
        load.has_particle_collision = False
        load.is_visible = False
        centre = floor.hinge + 0.5 * floor.b * np.asarray(floor.v, dtype=np.float64) + np.array([0.0, 0.0, t + hz])
        x = wp.transform(wp.vec3(*centre.tolist()), wp.quat_identity())
        builder.add_shape_box(self.body, xform=x, hx=hx, hy=hy, hz=hz, cfg=load, label=f"{self.spec.name}_contents")

    def _hinged(self, builder, cfg, root_x, dieline: Dieline, panel: Panel) -> int:
        """One creased panel: a link at its crease, a plate on it, and the hinge joint. Returns the joint."""
        spec, board = self.spec, self.spec.board
        t = board.thickness
        inward = np.asarray(panel.v, dtype=np.float64)
        if abs(inward[2]) > 1.0e-9:
            raise ValueError(
                f"carton {spec.name!r}: creased panel {panel.name!r} is not built level; the panels model hinges level panels only"
            )
        hinge = panel.hinge + np.array([0.0, 0.0, panel.rise])
        depth = panel.b
        # A flap creased along the top of a wall turns about the wall's mid-plane, not its outer face:
        # the wall's plate is inset by half the board, and so is the crease. The plate is shorter by as
        # much, so its free edge, and the seam between two flaps, stay where the die-line puts them.
        if any(_on_wall_top(wall, panel) for wall in dieline.glued):
            hinge = hinge + 0.5 * t * inward
            depth = panel.b - 0.5 * t
        local = wp.transform(wp.vec3(*hinge.tolist()), _frame(inward, np.array([0.0, 0.0, 1.0])))
        link = builder.add_link(xform=wp.transform_multiply(root_x, local), label=f"{spec.name}_{panel.name}")
        plate = wp.transform(wp.vec3(0.5 * depth, 0.0, 0.5 * t), wp.quat_identity())
        builder.add_shape_box(
            link,
            xform=plate,
            hx=0.5 * depth,
            hy=0.5 * panel.a,
            hz=0.5 * t,
            cfg=cfg,
            label=f"{spec.name}_{panel.name}_plate",
        )

        k = board.crease_stiffness * panel.a
        c = board.crease_damping * panel.a
        # Folded shut from standing before it was built, a box flap's crease already rests ``yield_angle`` open.
        ref = board.yield_angle if panel.folded else 0.0
        taped = spec.taped and panel.name in dieline.taped
        tape = board.tape_stiffness * panel.a
        # The link's x runs from the crease to the free edge and its z is up: an upward opening turns
        # x toward z, which is a turn about -y; a fold down is about +y.
        axis = (0.0, -1.0, 0.0) if panel.opens == "up" else (0.0, 1.0, 0.0)
        lower, upper = (-0.05, 4.6) if panel.opens == "up" else (-0.3, 2.4)
        label = f"{spec.name}_{panel.name}_crease"
        joint = builder.add_joint_revolute(
            parent=self.body,
            child=link,
            parent_xform=local,
            child_xform=wp.transform_identity(),
            axis=axis,
            damping=c,
            armature=board.armature,
            limit_lower=lower,
            limit_upper=upper,
            label=label,
            custom_attributes={
                "mujoco:dof_passive_stiffness": float(k + tape if taped else k),
                "mujoco:dof_springref": float(0.0 if taped else ref),
            },
        )
        self.hinges.append(
            {
                "name": panel.name,
                "link": link,
                "joint": joint,
                "coord": int(builder.joint_q_start[joint]),
                "dof": int(builder.joint_qd_start[joint]),
                "label": label,
                "k": k,
                "c": c,
                "ref": ref,
                "tape": tape,
                "taped": taped,
                "width": panel.a,
                "depth": depth,
            }
        )
        if taped:
            self.taped_now.add(panel.name)
        return joint

    def after_build(self, builder, episode) -> None:
        mine = set(range(*self.range.bodies))
        self.shapes = np.asarray(
            [s for s in range(*self.range.shapes) if int(builder.shape_body[s]) in mine], dtype=np.int32
        )

    def compute_grasp_box(self, builder) -> None:
        return

    # -------------------------------------------------------------- phase 3
    def contribute_hooks(self, hooks) -> None:
        hooks.add("post_solve", self._yield, name=f"crease:{self.spec.name}")

    def _yield(self, state, dt) -> None:
        if self._solver is None:
            return
        if self._host:
            # MuJoCo's C library, on a CPU: the same rule on its arrays, which are numpy views.
            q, spring = self._solver.mj_data.qpos, self._solver.mj_model.qpos_spring
            adr = self._qpos_adr[self._active]
            ref = spring[adr]
            excess = q[adr] - ref
            y = self.spec.board.yield_angle
            spring[adr] = ref + np.where(excess > y, excess - y, np.where(excess < -y, excess + y, 0.0))
            return
        data, model = self._solver.mjw_data, self._solver.mjw_model
        d = self._device
        wp.launch(
            _yield_kernel,
            dim=(data.qpos.shape[0], len(self.hinges)),
            inputs=[d["adr"], d["yield"], d["active"], data.qpos],
            outputs=[model.qpos_spring],
            device=d["adr"].device,
        )

    # -------------------------------------------------------------- phase 4
    def on_scene_ready(self, pipeline) -> None:
        super().on_scene_ready(pipeline)
        self.bind_solver(pipeline)

    def bind_solver(self, pipeline) -> None:
        """Find the MuJoCo solver stepping this carton and each hinge's place in it.

        By the joint's label in MuJoCo's own names, which the importer derives from Newton's, so it
        holds whether the carton's island is the whole model or one of several. On a CPU the solver is
        MuJoCo's C library, and the hinges are worked in its own arrays on the host.
        """
        import mujoco

        program = pipeline.program
        solver = program.solver_for(self.spec.name) if hasattr(program, "solver_for") else program.solver
        if not hasattr(solver, "mj_model"):
            raise RuntimeError(
                f"carton {self.spec.name!r} is stepped by {type(solver).__name__}, not MuJoCo: the panels model's hinges "
                "are MuJoCo joints. Stage rss.CARTON_60HZ, or put the 'carton' domain on the arm's backend."
            )
        m = solver.mj_model
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or "" for i in range(m.njnt)]
        adr, jnt = [], []
        for hinge in self.hinges:
            hits = [i for i, n in enumerate(names) if n == hinge["label"] or n.endswith("/" + hinge["label"])]
            if len(hits) != 1:
                raise RuntimeError(f"carton {self.spec.name!r}: {len(hits)} MuJoCo joints are named {hinge['label']!r}")
            jnt.append(hits[0])
            adr.append(int(m.jnt_qposadr[hits[0]]))
        self._solver = solver
        self._jnt = np.asarray(jnt, dtype=np.int32)
        self._qpos_adr = np.asarray(adr, dtype=np.int32)
        self._active = np.array([not h["taped"] for h in self.hinges], dtype=bool)
        self._host = bool(getattr(solver, "use_mujoco_cpu", False)) or getattr(solver, "mjw_model", None) is None
        if self._host:
            return
        device = solver.mjw_model.qpos_spring.device
        self._device = {
            "adr": wp.array(self._qpos_adr, dtype=wp.int32, device=device),
            "yield": wp.full(len(self.hinges), self.spec.board.yield_angle, dtype=wp.float32, device=device),
            "active": wp.array(
                np.array([0 if h["taped"] else 1 for h in self.hinges], dtype=np.int32), dtype=wp.int32, device=device
            ),
        }
        if getattr(pipeline, "captured", False):
            pipeline.invalidate_graph()

    # -------------------------------------------------------------- phase 5
    def reset(self, pipeline, episode) -> None:
        """Put the box back, its flaps shut or flat, its creases as built and its tape on."""
        super().reset(pipeline, episode)
        for state in (pipeline.state_0, pipeline.state_1):
            q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
            for hinge in self.hinges:
                q[hinge["coord"]] = 0.0
                qd[hinge["dof"]] = 0.0
            state.joint_q.assign(q)
            state.joint_qd.assign(qd)
        if self._solver is None:
            return
        for i, hinge in enumerate(self.hinges):
            self._write(
                i,
                k=hinge["k"] + hinge["tape"] if hinge["taped"] else hinge["k"],
                ref=0.0 if hinge["taped"] else hinge["ref"],
                active=not hinge["taped"],
            )
        self.taped_now = {h["name"] for h in self.hinges if h["taped"]}

    # ---------------------------------------------------------------- runtime
    def index(self, name: str) -> int:
        """Which of :attr:`hinges` is panel ``name``'s."""
        for i, hinge in enumerate(self.hinges):
            if hinge["name"] == name:
                return i
        raise KeyError(
            f"carton {self.spec.name!r} has no hinged panel {name!r}; it has {[h['name'] for h in self.hinges]}"
        )

    def angles(self, world: int = 0) -> np.ndarray:
        """Each hinge's angle now [rad]: 0 as built, growing the way its panel opens."""
        if self._host:
            return np.asarray(self._solver.mj_data.qpos, dtype=np.float64)[self._qpos_adr]
        return self._solver.mjw_data.qpos.numpy()[world, self._qpos_adr].astype(np.float64)

    def references(self, world: int = 0) -> np.ndarray:
        """Each hinge's spring reference now [rad]: the angle it keeps when nothing pushes it."""
        if self._host:
            return np.asarray(self._solver.mj_model.qpos_spring, dtype=np.float64)[self._qpos_adr]
        spring = self._solver.mjw_model.qpos_spring.numpy()
        return spring[min(world, spring.shape[0] - 1), self._qpos_adr].astype(np.float64)

    def angle(self, name: str, world: int = 0) -> float:
        return float(self.angles(world)[self.index(name)])

    def cut_tape(self, names: tuple[str, ...] | None = None) -> None:
        """Cut the tape on ``names`` (every taped flap by default): each hinge drops to its crease and springs open to its yield.

        A flap whose tape is already cut is left alone: its crease keeps what it has been given.
        """
        cut = (
            set(self.taped_now)
            if names is None
            else {self.hinges[self.index(n)]["name"] for n in names} & self.taped_now
        )
        for name in cut:
            i = self.index(name)
            hinge = self.hinges[i]
            self._write(i, k=hinge["k"], ref=hinge["ref"], active=True)
        self.taped_now -= cut

    def _write(self, i: int, *, k: float, ref: float, active: bool) -> None:
        """One hinge's stiffness and reference in MuJoCo's arrays, and its yield switch. Between steps."""
        self._active[i] = active
        if self._host:
            self._solver.mj_model.jnt_stiffness[self._jnt[i]] = k
            self._solver.mj_model.qpos_spring[self._qpos_adr[i]] = ref
            return
        model = self._solver.mjw_model
        stiffness = model.jnt_stiffness.numpy()
        stiffness[:, self._jnt[i]] = k
        model.jnt_stiffness.assign(stiffness)
        spring = model.qpos_spring.numpy()
        spring[:, self._qpos_adr[i]] = ref
        model.qpos_spring.assign(spring)
        on = self._device["active"].numpy()
        on[i] = 1 if active else 0
        self._device["active"].assign(on)


def _on_wall_top(wall: Panel, panel: Panel) -> bool:
    """Whether ``panel``'s crease runs along the top edge of the standing panel ``wall``."""
    u, v = np.asarray(wall.u, dtype=np.float64), np.asarray(wall.v, dtype=np.float64)
    if abs(np.cross(u, v)[2]) > 1.0e-9:  # a level panel: no top edge to stand on
        return False
    top = np.asarray(wall.origin, dtype=np.float64) + wall.b * v
    rel = panel.hinge - top
    return (
        abs(float(np.dot(rel, np.cross(u, v)))) < 1.0e-6
        and abs(rel[2]) < 1.0e-6
        and abs(float(np.dot(rel, u)) - 0.5 * wall.a) <= 0.5 * wall.a + 1.0e-6
    )
