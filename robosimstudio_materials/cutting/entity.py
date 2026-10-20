# SPDX-License-Identifier: Apache-2.0
"""The knife's entity and the block's: a rigid body with a blade, and a cage solver stepped beside Newton.

:class:`EntityKnife` is a free rigid body on the arm's solver, a handle box and a blade box, and
nothing more: the hand pinches its handle by contact like any box. :class:`EntityCuttable` owns no
Newton physics at all. It holds the vendored walk-on-spheres cage solver (``cagecut``) and, once a
control step, after the substeps: reads the knife's pose, applies the edge law (``law.py``), advances
the cut, steps the cage, feeds the push back onto the knife as a wrench for the next control step,
and redraws the cut block. Its eight particles are the block's corners, written each step, so the
renderers treat it as a particle body whose ``surface`` is the cutter's own mesh, kept under the
corners each step wrote so that a replay in this process redraws each step's block
(:meth:`EntityCuttable.surface_mesh`).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

import newton
import numpy as np
import warp as wp

from robosimstudio.core.math import quat_to_mat
from robosimstudio.core.schedule import Stage
from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.scene.build.bodies import set_body_mass
from robosimstudio.scene.entity import Entity, label_index

from .law import BlockFrame, blade_normal, box_cage, cut_line, edge_contact, edge_push
from .spec import CuttableMaterialSpec, CuttableSpec, KnifeSpec

__all__ = ["EntityCuttable", "EntityKnife"]


class EntityKnife(EntityRigid):
    """A knife: one free body, a handle box and a blade box, on whatever solver serves rigid bodies."""

    spec: KnifeSpec

    def __init__(self, spec: KnifeSpec, *, gripper_span: float = 0.08, world: int = 0) -> None:
        super().__init__(spec, gripper_span=gripper_span, world=world)
        self.handle_shape: int = -1
        self.blade_shape: int = -1

    def _add_primitive(self, builder: newton.ModelBuilder, pose) -> None:
        """The handle and the blade on one body; ``static`` and ``mass`` as :class:`EntityRigid` takes them."""
        spec = self.spec
        xform = wp.transform(tuple(float(v) for v in pose[:3]), wp.quat(*[float(v) for v in pose[3:7]]))
        if spec.static:
            self.body = self._add_welded_link(builder, xform)
        else:
            self.body = builder.add_body(xform=xform, label=spec.name)
            joint = builder.joint_count - 1
            self.coord_start = int(builder.joint_q_start[joint])
            self.dof_start = int(builder.joint_qd_start[joint])
        handle = self._contact_cfg(builder)
        handle.density = float(spec.handle_density)
        hx, hy, hz = (0.5 * float(v) for v in spec.handle_size)
        self.handle_shape = int(
            builder.add_shape_box(
                body=self.body,
                xform=wp.transform_identity(),
                hx=hx,
                hy=hy,
                hz=hz,
                cfg=handle,
                label=f"{spec.name}/handle",
            )
        )
        blade = self._contact_cfg(builder)
        blade.density = float(spec.blade_density)
        half = (0.5 * float(spec.blade_length), 0.5 * float(spec.blade_thickness), 0.5 * float(spec.blade_height))
        centre = wp.vec3(hx + half[0], 0.0, hz - half[2])
        self.blade_shape = int(
            builder.add_shape_box(
                body=self.body,
                xform=wp.transform(centre, wp.quat_identity()),
                hx=half[0],
                hy=half[1],
                hz=half[2],
                cfg=blade,
                label=f"{spec.name}/blade",
            )
        )
        if spec.mass is not None and not spec.static:
            # The densities' 96 g rescaled, the balance over the blade kept.
            set_body_mass(builder, self.body, float(spec.mass))

    def apply_display(self, builder: newton.ModelBuilder) -> None:
        """The blade in the spec's colour, the handle in its own."""
        super().apply_display(builder)
        from robosimstudio.core.render.finish import finish_for

        if self.handle_shape >= 0:
            color = tuple(float(v) for v in self.spec.handle_color)
            builder.shape_color[self.handle_shape] = color
            self.finishes[self.handle_shape] = finish_for(color, "objects")

    def edge_world(self, body_q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The edge's two ends and the edgeward direction in world, from a read-back ``body_q``."""
        pose = np.asarray(body_q[self.body], dtype=np.float64)
        return knife_edge(self.spec, pose)


def knife_edge(spec: KnifeSpec, pose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(edge0, edge1, edgeward)`` in world for a knife at ``pose`` ``(x, y, z, qx, qy, qz, qw)``."""
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    rotation = quat_to_mat(pose[3:7])
    origin, direction, length = spec.edge
    e0 = pose[:3] + rotation @ np.asarray(origin)
    e1 = e0 + rotation @ (np.asarray(direction) * length)
    return e0, e1, rotation @ np.asarray(spec.edgeward)


class EntityCuttable(Entity):
    """One block a knife cuts, on the vendored cage solver, stepped once a control step.

    Attributes:
        body: The cagecut ``Body``: the cage, its walk sets and its elastic solve. ``None`` until
            the scene is ready.
        material: The block's matter, the spec's: what the studio and the path tracer draw the block
            in the colour of when the spec states none.
        push: The block's push on the knife's edge at the last step [N].
        timing: Milliseconds the last step spent cutting and solving the cage.
    """

    spec: CuttableSpec
    domain = "cutting"

    def __init__(self, spec: CuttableSpec, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.material: CuttableMaterialSpec = spec.material
        self.particles: np.ndarray = np.zeros(0, dtype=np.int32)
        self.knife_body: int = -1
        self.block: BlockFrame | None = None
        self.body: Any = None
        self.push: float = 0.0
        self.timing: dict[str, float] = {}
        self.history: list[dict[str, float]] = []
        """One row a cut step: ``{"t", "front", "force"}``, cleared by a reset; what tests read."""
        self._plane: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._t_end = 0.0
        self._t_contact = 0.0
        self._fraction = 0.0
        self._depth = 0.0
        self._edge_depth = 0.0
        self._cuts = 0
        self._feed: Any = None
        self._wrench: Any = None
        self._surface: tuple[np.ndarray, np.ndarray] = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))
        self._drawn: OrderedDict[bytes, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        """The last :data:`DRAWN` drawings of the block, each under the corners its step wrote."""
        self._faces: tuple[Any, np.ndarray, np.ndarray] | None = None
        """The block's triangles as last triangulated: the cutter's snapshot they were read off, the cage
        vertices the cut faces gather and every triangle, which hold while the cutter keeps that snapshot."""
        self._pieces_version = -1
        self._pieces = 1
        self._dt = 0.02
        self._steps = 0
        self._armed = True
        self.sharpness = float(spec.knife.sharpness) if spec.knife is not None else 1.0
        """The knife's sharpness the edge law uses: the spec's until :meth:`set_sharpness` changes it."""

    # -------------------------------------------------------------- phase 1
    def build(self, builder: newton.ModelBuilder, episode) -> None:
        """Eight inert particles at the block's corners, and the knife's body found by its name."""
        del episode
        spec = self.spec
        if spec.knife is None:
            raise ValueError(f"cuttable {spec.name!r} names no knife: pass knife=rss.knife(...) added before it")
        self.block = BlockFrame.from_base(self.placed((*spec.at, *spec.quat)), spec.size)
        corners, _ = box_cage(spec.size)
        for corner in self.block.from_cutter(corners * self.block.scale):
            builder.add_particle(
                wp.vec3(*[float(c) for c in corner]), wp.vec3(0.0, 0.0, 0.0), mass=0.0, radius=1.0e-3, flags=0
            )
        try:
            self.knife_body = label_index(builder.body_label, spec.knife.name)
        except KeyError as exc:
            raise KeyError(
                f"cuttable {spec.name!r}: its knife {exc.args[0]}; add the knife to the scene before the block, "
                "under a name nothing else uses"
            ) from None

    def owned_shapes(self) -> np.ndarray:
        return np.zeros(0, dtype=np.int32)

    # -------------------------------------------------------------- phase 2
    def bind(self, model: newton.Model) -> None:
        from robosimstudio.engines.feedback import WrenchAccumulator, WrenchFeed

        super().bind(model)
        self.particles = self.range.particle_indices
        stepping = self.stepping_model or model
        self._feed = WrenchFeed(stepping)
        # The knife rides in the hand, so the cap on what one step's wrench may add to a free body is
        # set by the hand's grip rather than the knife's 70 g: 5 m/s a control step.
        self._wrench = WrenchAccumulator(stepping, alpha=0.35, dv_max=5.0)

    def handle(self, model: newton.Model) -> Any:
        """The corners as a particle body whose surface is the cutter's mesh as drawn at those corners."""
        from robosimstudio.scene.handles import DeformableHandle

        del model
        return DeformableHandle(
            name=self.name, particles=self.particles, point_radius=1.0e-3, surface=self.surface_mesh
        )

    # -------------------------------------------------------------- phase 3
    def contribute_hooks(self, hooks) -> None:
        """The push onto the knife inside every substep, and the cut and the cage after each control step."""
        hooks.add("rigid_force", self._feed.stage, name=f"knife-push:{self.name}")
        hooks.schedule.add(Stage(name=f"cut:{self.name}", fn=self._tick, phase="after", reads=frozenset({"body_q"})))

    # -------------------------------------------------------------- phase 4
    def on_scene_ready(self, pipeline) -> None:
        """Build the cage solver over the uncut block, at the scene's control step."""
        self._dt = float(pipeline.control_dt)
        self._make_body()
        self._write(pipeline)

    def _make_body(self) -> None:
        from .solver import make_body

        self.body = make_body(self.spec, self.block, dt=self._dt)
        self._plane = None
        self._fraction = self._depth = self._edge_depth = self.push = 0.0
        self._cuts = 0
        self._armed = True
        self._pieces_version = -1
        self.history = []
        self._refresh_surface()

    # -------------------------------------------------------------- phase 5
    def reset(self, pipeline, episode, *, world: int = 0) -> None:
        """The block back whole, at rest, and nothing pushing on the knife."""
        del episode, world
        if self.body is None:
            return
        self.body.reset()
        self._plane = None
        self._fraction = self._depth = self._edge_depth = self.push = 0.0
        self._cuts = 0
        self._armed = True
        self._pieces_version = -1
        self.history = []
        if self._wrench is not None:
            self._feed.assign(self._wrench.zero())
        self._refresh_surface()
        self._write(pipeline)

    # ---------------------------------------------------------------- the tick
    def _tick(self, state: newton.State, dt: float) -> None:
        if self.body is None or self.block is None:
            return
        start = time.perf_counter()
        knife = self.spec.knife
        knife_pose = np.asarray(state.body_q.numpy()[self.knife_body], dtype=np.float64)
        e0, e1, edgeward = knife_edge(knife, knife_pose)
        push, point, cage_force = self._cut(e0, e1, edgeward)
        cut_ms = 1e3 * (time.perf_counter() - start)
        self.body.step(cage_force)
        self._wrench.begin()
        if push is not None:
            self._wrench.add(self.knife_body, push, point, knife_pose)
        self._feed.assign(self._wrench.shape(float(dt)))
        self._refresh_surface()
        self._write_state(state)
        self._steps += 1
        self.timing = {"cut_ms": cut_ms, "total_ms": 1e3 * (time.perf_counter() - start)}

    def _cut(self, e0_world, e1_world, edgeward_world) -> tuple[np.ndarray | None, np.ndarray | None, Any]:
        """The edge law for one control step: the push on the knife, where, and the push on the cage."""
        spec, frame, body = self.spec, self.block, self.body
        size = frame.size
        e0, e1 = frame.to_block(e0_world), frame.to_block(e1_world)
        edgeward = frame.vector_to_block(edgeward_world)
        contact = edge_contact(e0, e1, size)
        self.push = self._edge_depth = 0.0
        if contact.length <= 0.0 or not len(contact.points):
            return None, None, None
        self._edge_depth = max(float(size[2]) - float(contact.points[:, 2].min()), 0.0)
        cutter = body.cutter
        if self._plane is not None and cutter.active:
            p1, _, _, normal = self._plane
            off = float(np.max(np.abs((contact.points - p1) @ normal)))
            if off > spec.lateral:
                if cutter.started:
                    return None, None, None  # out of the cut it began; the topology cannot start another
                self._plane = None
        if float(contact.points[:, 2].min()) >= size[2]:
            self._armed = True  # the edge is out over the block: the next entry may start a cut
        if self._plane is None or not cutter.active:
            if not self._armed or float(contact.points[:, 2].min()) >= size[2]:
                return None, None, None
            if not self._begin(e0, e1, edgeward):
                self._armed = False
                return None, None, None
        p1, _, sweep, _ = self._plane
        s = frame.scale
        t_edge = float(np.max((contact.points - p1) @ sweep))
        t_front = float(cutter.t) / s
        threshold = spec.material.toughness / self.sharpness
        force, lag = edge_push(
            depth=t_edge - t_front,
            gap=self._t_end - t_edge,
            length=contact.inside,
            young=spec.material.young_modulus,
            push=spec.push,
            threshold=threshold,
            floor=spec.floor,
        )
        target = self._t_end + 1.0e-3 if lag <= 0.0 and force > 0.0 else t_edge - lag
        if target - t_front >= MIN_ADVANCE or (target > t_front and target >= self._t_end):
            info = body.advance_cut((target - t_front) * s)
            if info.completed:
                self._cuts += 1
                self._fraction, self._plane, self._armed = 1.0, None, False
            self.history.append({"t": float(cutter.t) / s, "front": t_front, "force": force})
        if self._plane is not None:
            span = max(self._t_end - self._t_contact, 1.0e-9)
            self._depth = max(float(cutter.t) / s - self._t_contact, 0.0)
            self._fraction = float(np.clip(self._depth / span, 0.0, 1.0))
        self.push = float(force)
        if force <= 0.0:
            return None, None, None
        sweep_world = frame.vector_to_world(sweep)
        point = frame.from_cutter(contact.inside_centroid * s)
        return -force * sweep_world, point, self._cage_force(force, sweep, contact.inside_centroid)

    def _begin(self, e0, e1, edgeward) -> bool:
        """Start a cut in the blade's plane; False when the topology refuses the plane."""
        from .solver import CutError

        body, s = self.body, self.block.scale
        snap = body.cutter.snapshot
        alive = np.flatnonzero(snap.face_alive)
        normal = blade_normal(e0, e1, edgeward)
        verts = snap.verts[snap.vert_alive] / s
        crossed = []
        for f in alive:
            side = (snap.verts[snap.face(f)] / s - e0) @ normal
            crossed.append(side.min() < 0.0 < side.max())
        line, sweep, normal, _ = cut_line(e0, e1, edgeward, snap.face_normal[alive], np.asarray(crossed, dtype=bool))
        deepest = e0 if (e0 - e1) @ sweep >= 0.0 else e1
        p1 = deepest - sweep * CLEARANCE
        spec = {
            "kind": "blade",
            "p1": p1 * s,
            "p2": (p1 + line) * s,
            "direction": sweep,
            "step_size": MIN_ADVANCE * s,
            "angle": 0.0,
            "bounded": False,
            "plunge": False,
        }
        body.specs = [spec]
        try:
            body.begin_cut(0)
        except CutError:
            self._plane = None
            return False
        self._plane = (p1, line, sweep, normal)
        self._t_end = float(np.max((verts - p1) @ sweep))
        self._t_contact = float(body.cutter.t) / s
        self._fraction = self._depth = 0.0
        return True

    def _cage_force(self, force: float, sweep, centroid) -> np.ndarray:
        """The push's reaction on the cage, in cutter units: on the front's tips, else on the face pressed,
        the one nearest ``centroid``, the middle of the edge inside the block (block frame) [m]."""
        body, s = self.body, self.block.scale
        count = body.sim.num_verts
        out = np.zeros((count, 3))
        tips, _ = body.blade_front()
        ids = [int(v) for v in tips if v < count]
        if not ids:
            snap = body.cutter.snapshot
            best, near = -1, np.inf
            for f in np.flatnonzero(snap.face_alive):
                if float(snap.face_normal[f] @ sweep) > -0.7:
                    continue
                distance = abs(float(snap.face_normal[f] @ (centroid * s)) - float(snap.face_offset[f]))
                if distance < near:
                    best, near = int(f), distance
            if best < 0:
                return out
            ids = [int(v) for v in snap.face(best)]
        # F_cutter = s^2 F_world: the cutter's frame is the block's scaled by s, with density over s^2.
        out[ids] += (force * s * s / len(ids)) * np.asarray(sweep)[None, :]
        return out

    # ---------------------------------------------------------------- drawing
    def _refresh_surface(self) -> None:
        """The skin and the cut faces, deformed, in world, as one mesh.

        The cut faces are triangulated off the cutter's snapshot, which the cutter replaces whenever the
        cut moves or the topology changes; while it holds, a step only gathers the deformed cage.
        """
        body, frame = self.body, self.block
        x = body.sim.x
        skin = body.corrected_mesh_weights() @ x
        snap = body.cutter.snapshot
        if self._faces is None or self._faces[0] is not snap:
            self._faces = (snap, *self._triangulate(snap, len(skin), len(x)))
        _, gather, faces = self._faces
        world = frame.from_cutter(np.concatenate([skin, x[gather]])).astype(np.float32)
        self._surface = (world, faces)
        key = _corner_key(self._corners())
        self._drawn[key] = self._surface
        self._drawn.move_to_end(key)
        while len(self._drawn) > DRAWN:
            self._drawn.popitem(last=False)

    def _triangulate(self, snap: Any, offset: int, count: int) -> tuple[np.ndarray, np.ndarray]:
        """The cage vertices the cut faces of ``snap`` gather, after the ``offset`` skin points, and the
        skin's and the cut faces' triangles; a face reaching past the ``count`` cage vertices is left out."""
        from .solver import triangulate_polygon_with_holes

        body = self.body
        tris = [np.asarray(body.mesh_tris, dtype=np.int64)]
        gather = [np.zeros(0, dtype=np.int64)]
        for f in sorted(body.cut_face_ids):
            if f >= snap.num_faces or not snap.face_alive[f]:
                continue
            loops = snap.face_loops(f)
            flat = np.concatenate(loops).astype(np.int64)
            if flat.max() >= count:
                continue
            u, v = snap.face_u[f], snap.face_v[f]
            loops2d = [np.stack([snap.verts[lp] @ u, snap.verts[lp] @ v], axis=1) for lp in loops]
            faces = triangulate_polygon_with_holes(loops2d)
            if not len(faces):
                continue
            gather.append(flat)
            tris.append(faces + offset)
            offset += len(flat)
        faces = np.concatenate(tris).astype(np.int32)
        last = self._surface[1]
        if last.shape == faces.shape and np.array_equal(last, faces):
            faces = last  # shared by the drawings kept while the topology holds
        return np.concatenate(gather), faces

    def surface_mesh(self, positions: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        """The block as drawn with its corners at ``positions``: ``(vertices, triangles)`` in world.

        The cut's topology and the cage inside the corners live in the cutter, so the block cannot be
        regrown from the eight corners a recording holds. Each step's drawing is kept instead, under
        the corners that step wrote (the last :data:`DRAWN`), and a replay of recorded corners in this
        process -- the page's **Export video** -- redraws each step's block as it was, whole until the
        knife went in. Corners with no drawing kept draw nothing rather than another step's block, so
        a replay in another process, which holds only the block as built, draws none once the first
        step has moved the corners.
        """
        return self._drawn.get(_corner_key(positions), (None, None))

    def _corners(self) -> np.ndarray:
        return self.block.from_cutter(self.body.sim.x[:8]).astype(np.float32)

    def _write_state(self, state: newton.State) -> None:
        if len(self.particles):
            state.particle_q[int(self.particles[0]) : int(self.particles[-1]) + 1].assign(self._corners())

    def _write(self, pipeline) -> None:
        for state in (pipeline.state_0, pipeline.state_1):
            self._write_state(state)
        mark = getattr(pipeline, "mark_dirty", None)
        if mark is not None:
            mark()

    def set_sharpness(self, sharpness: float) -> None:
        """Make the knife sharper or duller from the next step: 1 is the spec's sharp knife. LIVE."""
        if not 0.0 < float(sharpness) <= 1.0e3:
            raise ValueError(f"sharpness must be in (0, 1000], got {sharpness}")
        self.sharpness = float(sharpness)

    # ---------------------------------------------------------------- reading
    def pieces(self) -> int:
        """How many pieces the block is in: the cage's material components."""
        if self.body is None:
            return 1
        version = self.body.topology_version
        if version != self._pieces_version:
            from .solver import piece_labels

            labels = piece_labels(self.body.cutter.snapshot)
            self._pieces = len({int(v) for v in labels} - {-1})
            self._pieces_version = version
        return self._pieces

    def cut_fraction(self) -> float:
        """How far the last cut has run through the block along its sweep, in ``[0, 1]``; 1 once it parted."""
        return float(self._fraction)

    def cut_depth(self) -> float:
        """How far the last cut's front has run from where the edge met the block [m]."""
        return float(self._depth)

    def edge_depth(self) -> float:
        """How far the edge's lowest point over the block is under its top, at the last step [m].

        Read off the knife's pose, so it counts a knife tipped in the hand; 0 with the edge above the
        block or off its footprint.
        """
        return float(self._edge_depth)

    def cuts(self) -> int:
        """How many cuts have gone right through."""
        return int(self._cuts)

    def published_states(self, snapshot) -> dict[str, float]:
        del snapshot
        return {"pieces": float(self.pieces()), "cut_fraction": self.cut_fraction()}


def _corner_key(positions: Any) -> bytes:
    """What a drawing is kept under: the corners as ``particle_q`` holds them, float32."""
    return np.ascontiguousarray(np.asarray(positions, dtype=np.float32).reshape(-1, 3)).tobytes()


DRAWN = 2048
"""How many drawings of the block it keeps for a replay, the newest. The page's video buffer spans
1800 control steps (900 frames, one every second step); in ``tests/test_cut_tofu``'s drive the corners
took 178 values over 728 steps, and the drawings kept under one of them agreed to a float32 ulp. A
drawing is about 14 KB of vertices, its triangles shared with the last while the topology holds."""

MIN_ADVANCE = 5.0e-4
"""The least the front moves in one step [m], so a slow knife costs few cut steps; the last step to the
far side is taken whatever its length."""

CLEARANCE = 5.0e-3
"""How far behind the edge the cut's line starts [m], outside the block, so the cutter finds the first
contact itself."""
