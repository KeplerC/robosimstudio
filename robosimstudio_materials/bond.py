# SPDX-License-Identifier: Apache-2.0
"""Breakable bonds in one file: points that hold two rigid bodies together up to a strength, then let go for good.

This file is what ``rss.bond`` builds, imported as ``robosimstudio_materials.bond``. A bond joins one
point on a rigid body to the point of another body (or of the world) it was built on. It carries
whatever the scene puts on it up to a strength, in tension and in shear, and once that is passed it
is gone until the scene is reset. A seam, a perforation, a glued lid and a shell's score line are rows
of them.

Two blocks hung from the world by a bond each, good for 3 N along its normal: the 100 g block's weight
is 0.98 N and it hangs, the 500 g block's is 4.9 N and it falls (``tests/test_bond_scene`` runs this)::

    import robosimstudio as rss

    scene = rss.Scene(cell="bench_075", name="hung", physics=rss.MUJOCO_500HZ)
    scene.add(rss.box(name="light", half=(0.02, 0.02, 0.02), at=(0.0, -0.1, 0.3), mass=0.1))
    scene.add(rss.box(name="heavy", half=(0.02, 0.02, 0.02), at=(0.0, 0.1, 0.3), mass=0.5))
    down = (0.0, 0.0, -1.0)  # a bond's normal, the way its block comes off: a pull along it is tension
    hooks = scene.add(
        rss.bond(
            name="hooks",
            points=(
                rss.BondPoint(a="light", at=(0.0, -0.1, 0.32), normal=down, strength=3.0),
                rss.BondPoint(a="heavy", at=(0.0, 0.1, 0.32), normal=down, strength=3.0),
            ),
        )
    )
    sim = scene.build()
    sim.run(60)
    print(hooks.entity.let_go())  # [False  True]: the heavy block is on the bench, its bond gone for good
    print(hooks.entity.loads().round(2))  # [0.98 0.  ]: what each carries now, by the rule it breaks on [N]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import newton
import numpy as np
import warp as wp

from robosimstudio.core.pipeline import ScenePipeline
from robosimstudio.scene.entity import Entity
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.base import Vec3
from robosimstudio.specs.physics import SimRequirements

__all__ = [
    "PIECE_IMPEDANCE",
    "STIFF_IMPEDANCE",
    "BondPoint",
    "BondSpec",
    "EntityBond",
    "bond",
    "bond_load",
]


# =============================================================================
# 1. What this file is, and why it is built the way it is
# =============================================================================
# Importing this file registers the bond. The library finds it by scanning ``robosimstudio_materials``,
# like tape, whose bonds these grew from: a tape's hold a cloth to a body inside the cloth solver, and
# these hold two rigid bodies together inside the robot's.
#
# The mechanism. A bond is a MuJoCo CONNECT equality row, written on Newton 1.7's
# ``mujoco:equality_constraint`` table at the build: three constraint rows, along world x, y and z,
# that keep one point of body ``a`` on one point of body ``b``. MuJoCo solves the row with the
# contacts and the robot's joints, implicitly, so a bond between two one-gram bodies is as stable as
# the rest of the scene at the solver's own step. A spring written into ``body_f`` instead is
# explicit: its stiffness has to stay under m/dt^2, 500 N/m for a 0.5 g body at 1 kHz, which is
# 10 mm of stretch at 5 N. After every solver step two kernels (section 3) copy each bond's three row
# forces out of the solver's constraint list (``mjw_data.efc.force``), weigh them by the rule below,
# and clear the row in ``mjw_data.eq_active`` and in the model's ``equality_constraint_enabled``, which
# Newton copies back over the solver's whenever it is told the constraints changed. Nothing in a
# substep writes either back on, so a broken bond stays broken; a scene reset is what makes every
# bond again. Both kernels are device launches the captured graph holds, and they run at the
# ``post_solve`` phase of ``core/schedule.py``.
#
# The rule (:func:`bond_load`). A bond with a ``normal``, the way body ``a`` comes off body ``b``,
# carries tension T along it and shear S across it. Its load is sqrt(T^2 + (S strength /
# shear_strength)^2) and it breaks when that passes ``strength``. Pushed together it carries no
# load at any force: compression is the bodies' own contact to bear, and a bonded seam squeezed shut
# does not come apart. A bond with no normal breaks on the size of the force alone.
#
# What it measures (``tests/test_bond_pull``, ``test_bond_peel``, ``test_bond_hold``, on
# ``rss.MUJOCO_500HZ``, RTX PRO 6000). Two blocks joined by 1, 4 or 8 bonds of 2 N and drawn apart by a
# force ramped over 6 s part at N x 2 N, to within the 0.33% the ramp climbs in a control step. A strip
# of eight plates hinged to each other and held down by rows of bonds, drawn up at 90 degrees at
# 10 mm/s, loses its rows in order, one a second, each within 8% of its row's strength plus the weight
# already hanging. Loaded at 80% of its strength for 10 s a joint breaks nothing and moves under
# 0.0001 mm.
#
# How stiff. A MuJoCo constraint is as stiff as its impedance is near 1 and as the lighter body it joins
# is heavy. A 1 g block hung from the world by four bonds stretches the joint 1.28 mm under 6.4 N at
# MuJoCo's default impedance (0.9 to 0.95) and 0.026 mm at :data:`STIFF_IMPEDANCE` (0.999), on a 500 Hz
# step; a 100 g block 0.029 mm and 0.0002 mm.
#
# The same sum makes a bonded *piece* soft to the world. MuJoCo sizes a contact's softness by the
# lighter body in it, and a shard of a bonded shell is light though the shell is not: a bowl's rim
# sank 2 mm into a 0.4 g shard of one for 0.4 N.
# :attr:`BondSpec.piece_impedance` writes a stiff impedance onto the shapes of every bonded body, with
# a mixing weight that makes it the pair's; friction and contact dimension still come from the larger
# of the two surfaces.
#
# Where it works. Both bodies have to be on the MuJoCo solver, which is where ``rss.MUJOCO_500HZ``
# and the particle tables (``rss.LIQUID_120HZ`` and its siblings) put free rigid bodies, and where
# the robot is. ``rss.GENERIC`` puts free bodies on VBD, whose solver never sees the row; the entity
# raises at the build rather than hold nothing. One world: the rows and the kernels address world 0.
#
# What it is not. A bond does not re-form, and it stores nothing: it is rigid until it breaks, so a
# peel's pull falls to the hanging weight between rows. A piece that every bond has let go of is a free
# body like any other.


# =============================================================================
# 2. The spec: what a set of bonds is, as data
# =============================================================================
# Exported as ``rss.BondPoint`` and ``rss.BondSpec``.

STIFF_IMPEDANCE = (0.999, 0.999, 0.001, 0.5, 2.0)
"""The bond rows' MuJoCo ``solimp``: impedance 0.999, where MuJoCo's default 0.9 to 0.95 stretches (section 1)."""

PIECE_IMPEDANCE = (0.998, 0.998, 0.001, 0.5, 2.0)
"""A contact impedance for the shapes of bonded pieces: what the YAM's own finger colliders ship with."""


@dataclass(frozen=True, kw_only=True)
class BondPoint:
    """One bond: a point of body ``a`` held on body ``b``.

    Attributes:
        a: The name of the body that comes away, a free rigid body added to the scene before the bonds.
        b: The body it is bonded to, or ``""`` for the world.
        at: Where the bond is, in the cell's work-surface frame [m], with both bodies where they are built.
        normal: The way ``a`` comes off ``b``, in the same frame: tension is the pull along it and shear
            the pull across it, and a push along it never breaks the bond. ``None`` breaks on the size of
            the force alone, compression included.
        strength: The tension the bond holds [N].
        shear_strength: The shear it holds [N], or ``None`` for :attr:`strength`. Read only with a normal.
        group: A name shared by the bonds of one seam, for a reader that counts them by seam.
    """

    a: str
    b: str = ""
    at: Vec3 = (0.0, 0.0, 0.0)
    normal: Vec3 | None = None
    strength: float = 1.0
    shear_strength: float | None = None
    group: str = ""

    def __post_init__(self) -> None:
        if not self.a:
            raise ValueError("a bond needs the body that comes away, a=; the world can only be b")
        if self.a == self.b:
            raise ValueError(f"a bond joins two bodies, and both ends name {self.a!r}")
        if not float(self.strength) > 0.0:
            raise ValueError(f"a bond's strength must be > 0 N, got {self.strength}")
        if self.shear_strength is not None and not float(self.shear_strength) > 0.0:
            raise ValueError(f"a bond's shear_strength must be > 0 N, got {self.shear_strength}")
        if self.normal is not None and not float(np.linalg.norm(self.normal)) > 0.0:
            raise ValueError("a bond's normal must have a length; None is the bond with no normal")


@dataclass(frozen=True, kw_only=True)
class BondSpec:
    """A named set of bonds between rigid bodies of the scene.

    Attributes:
        name: Key in handles and readouts; ``sim[name].entity`` is the :class:`EntityBond`.
        points: The bonds. One set may join many pairs of bodies, and costs two kernel launches a
            substep however many it holds.
        collide: Whether two bodies joined by a bond still collide with each other. Off: bonded
            pieces are one thing while they hold, and MuJoCo collides a mesh as its convex hull, so
            two shards of a curved shell overlap where they meet. It stays off after the bonds break.
        solref: The rows' MuJoCo ``solref``, time constant [s] and damping ratio. The default is under
            any step used here and MuJoCo raises it to twice the step, its stiffest.
        solimp: The rows' MuJoCo ``solimp`` (:data:`STIFF_IMPEDANCE`).
        piece_impedance: A MuJoCo contact ``solimp`` for every shape of every bonded body, or
            ``None`` to leave them as built. A light piece of a bonded thing meets the world as softly
            as its own mass says (section 1); :data:`PIECE_IMPEDANCE` is the stiff one.
        piece_mix: The mixing weight (``solmix``) those shapes take, so their impedance is the pair's.
        sim: What these bonds need of physics, or ``None``: the rigid domain, on the bodies' solver.
    """

    sim: SimRequirements | None = None
    stage: ClassVar[str] = "extras"
    """Built after every object, so the bodies a bond names are there."""
    entity: str = "bond"
    name: str = "bond"
    points: tuple[BondPoint, ...] = ()
    collide: bool = False
    solref: tuple[float, float] = (0.0005, 1.0)
    solimp: tuple[float, float, float, float, float] = STIFF_IMPEDANCE
    piece_impedance: tuple[float, float, float, float, float] | None = None
    piece_mix: float = 1000.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple(self.points))
        if not self.name:
            raise ValueError("a set of bonds needs a name")
        for point in self.points:
            if not isinstance(point, BondPoint):
                raise TypeError(f"bonds {self.name!r}: points are BondPoint, got {type(point).__name__}")

    @property
    def bodies(self) -> tuple[str, ...]:
        """Every body a bond names, each once, in the order met."""
        seen: dict[str, None] = {}
        for point in self.points:
            seen[point.a] = None
            if point.b:
                seen[point.b] = None
        return tuple(seen)


# =============================================================================
# 3. The rule, and the two kernels that apply it
# =============================================================================

EQ_CONNECT = 0
"""``mujoco.mjtEq.mjEQ_CONNECT``, which is ``newton.solvers.SolverMuJoCo.EqType.CONNECT``."""
OBJ_BODY = 1
"""``mujoco.mjtObj.mjOBJ_BODY``: the rows name bodies."""
EFC_EQUALITY = 0
"""``mujoco_warp``'s ``ConstraintType.EQUALITY``: the rows of the solver's list that are equalities."""


def bond_load(force, normal=None, strength: float = 1.0, shear_strength: float | None = None) -> float:
    """What one bond's rule makes of the force on it [N]: the bond breaks when this passes ``strength``.

    The kernel's arithmetic on the host, for a reader and for a test.

    Args:
        force: The constraint's force on body ``a``, world [N].
        normal: The way ``a`` comes off ``b``, world, or ``None`` for a bond with no normal.
    """
    f = np.asarray(force, dtype=np.float64)
    if normal is None:
        return float(np.linalg.norm(f))
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    along = float(np.dot(f, n))
    tension = max(-along, 0.0)  # the bond pulls ``a`` back, against the normal
    shear = float(np.linalg.norm(f - along * n))
    ratio = float(strength) / float(strength if shear_strength is None else shear_strength)
    return float(np.hypot(tension, shear * ratio))


@wp.kernel
def _gather_kernel(
    count: wp.array[int],
    row_type: wp.array2d[int],
    row_id: wp.array2d[int],
    row_force: wp.array2d[float],
    bond_of: wp.array[int],
    force: wp.array2d[float],
):
    """Copy each bond's three row forces out of the solver's constraint list (world 0).

    CONNECT rows are the first the solver's constraint builder writes and each takes three in a
    row, along world x, y and z, so a row's component is its index modulo three.
    """
    i = wp.tid()
    if i >= count[0]:
        return
    if row_type[0, i] != 0:
        return
    b = bond_of[row_id[0, i]]
    if b < 0:
        return
    force[b, i % 3] = row_force[0, i]


@wp.kernel
def _break_kernel(
    force: wp.array2d[float],
    body_q: wp.array[wp.transform],
    body: wp.array[int],
    normal: wp.array[wp.vec3],
    strength: wp.array[float],
    shear_strength: wp.array[float],
    equality: wp.array[int],
    row: wp.array[int],
    load: wp.array[float],
    peak: wp.array[float],
    broken: wp.array[int],
    broke_at: wp.array[float],
    active: wp.array2d[wp.bool],
    enabled: wp.array[wp.bool],
    enabled_too: wp.array[wp.bool],
):
    """Weigh each bond's force by :func:`bond_load` and switch off the row of one past its strength."""
    b = wp.tid()
    if broken[b] != 0:
        load[b] = 0.0
        return
    f = wp.vec3(force[b, 0], force[b, 1], force[b, 2])
    s = strength[b]
    value = wp.length(f)
    if wp.length_sq(normal[b]) > 0.0:
        n = wp.quat_rotate(wp.transform_get_rotation(body_q[body[b]]), normal[b])
        along = wp.dot(f, n)
        tension = wp.max(-along, 0.0)
        shear = wp.length(f - along * n)
        ratio = s / shear_strength[b]
        value = wp.sqrt(tension * tension + shear * shear * ratio * ratio)
    load[b] = value
    peak[b] = wp.max(peak[b], value)
    if value > s:
        broken[b] = 1
        broke_at[b] = value
        active[0, equality[b]] = False
        enabled[row[b]] = False
        enabled_too[row[b]] = False


# =============================================================================
# 4. The entity: the rows at the build, the kernels after every substep
# =============================================================================


def _leaf(label: Any) -> str:
    return str(label).split("/")[-1]


def _rotate(quat, vector) -> np.ndarray:
    x, y, z, w = (float(c) for c in quat)
    u, v = np.array([x, y, z]), np.asarray(vector, dtype=np.float64)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


class EntityBond(Entity):
    """One set of bonds: equality rows between bodies other entities built, and the rule that breaks them.

    It owns no body, shape or particle. It composes in the rigid domain, which is where the bodies it
    joins have to be, and reads :attr:`broken`, :meth:`held` and :meth:`loads` for a page or a test.
    """

    spec: BondSpec
    domain = "rigid"

    def __init__(self, spec: BondSpec, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.rows: np.ndarray = np.zeros(0, dtype=np.int32)
        """Each bond's row in the model's ``mujoco:equality_constraint`` table."""
        self.bodies: np.ndarray = np.zeros(0, dtype=np.int32)
        """Each bond's body ``a`` in the model."""
        self.labels: list[str] = []
        """Each bond's row label, ``<name>/<index>``: how the rows are found again in the solver's own model."""
        self.normals: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        """Each bond's normal in body ``a``'s frame; zero for a bond with none."""
        self.solver: Any = None
        """The MuJoCo solver that steps the bonded bodies, found at scene-ready."""
        self.broken: Any = None
        """Per bond, 1 once it has let go, on the device."""
        self._a_labels: list[str] = []
        self._arrays: dict[str, Any] = {}

    # -------------------------------------------------------------- phase 1
    def sim_requirements(self) -> SimRequirements:
        declared = self.spec.sim
        return declared if isinstance(declared, SimRequirements) else SimRequirements(domain="rigid")

    def _named_body(self, builder: newton.ModelBuilder, name: str) -> int:
        hits = [b for b, label in enumerate(builder.body_label) if _leaf(label) == name]
        if len(hits) != 1:
            raise KeyError(
                f"bonds {self.spec.name!r}: {name!r} names {len(hits)} bodies; add the body to the scene before "
                "the bonds, as a free rigid body under a name nothing else uses"
            )
        return int(hits[0])

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec = self.spec
        if "mujoco:equality_constraint_type" not in builder.custom_attributes:
            raise RuntimeError(
                f"bonds {spec.name!r}: this Newton has no mujoco:equality_constraint table, which a bond is a row of"
            )
        defaults = {
            key: attr.default
            for key, attr in builder.custom_attributes.items()
            if getattr(attr, "frequency", None) == "mujoco:equality_constraint"
        }
        frame = np.asarray(self.frame, dtype=np.float64)
        rows, bodies, normals, pairs = [], [], [], set()
        for index, point in enumerate(spec.points):
            a = self._named_body(builder, point.a)
            b = self._named_body(builder, point.b) if point.b else -1
            world = np.asarray(self.placed((*point.at, 0.0, 0.0, 0.0, 1.0)), dtype=np.float64)[:3]
            pose = wp.transform_inverse(wp.transform(*builder.body_q[a]))
            anchor = wp.transform_point(pose, wp.vec3(*[float(v) for v in world]))
            normal = (0.0, 0.0, 0.0)
            if point.normal is not None:
                n = _rotate(frame[3:7], point.normal)
                local = wp.transform_vector(pose, wp.vec3(*[float(v) for v in n / np.linalg.norm(n)]))
                normal = tuple(float(v) for v in local)
            label = f"{spec.name}/{index}"
            values = {
                **defaults,
                "mujoco:equality_constraint_type": EQ_CONNECT,
                "mujoco:equality_constraint_objtype": OBJ_BODY,
                "mujoco:equality_constraint_body1": a,
                "mujoco:equality_constraint_body2": b,
                "mujoco:equality_constraint_anchor": anchor,
                "mujoco:equality_constraint_label": label,
                "mujoco:equality_constraint_enabled": True,
                "mujoco:equality_constraint_world": builder.current_world,
                "mujoco:eq_solref": wp.vec2(*[float(v) for v in spec.solref]),
                "mujoco:eq_solimp": tuple(float(v) for v in spec.solimp),
            }
            rows.append(int(builder.add_custom_values(**values)["mujoco:equality_constraint_type"]))
            bodies.append(a)
            normals.append(normal)
            self.labels.append(label)
            self._a_labels.append(str(builder.body_label[a]))
            if b >= 0:
                pairs.add((min(a, b), max(a, b)))
        self.rows = np.asarray(rows, dtype=np.int32)
        self.bodies = np.asarray(bodies, dtype=np.int32)
        self.normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
        if not spec.collide:
            for a, b in sorted(pairs):
                for sa in builder.body_shapes[a]:
                    for sb in builder.body_shapes[b]:
                        builder.add_shape_collision_filter_pair(int(sa), int(sb))
        if spec.piece_impedance is not None:
            self._stiffen(builder, sorted({self._named_body(builder, name) for name in spec.bodies}))

    def _stiffen(self, builder: newton.ModelBuilder, bodies: list[int]) -> None:
        """Write :attr:`BondSpec.piece_impedance` and :attr:`~BondSpec.piece_mix` onto every shape of ``bodies``."""
        written = {
            "mujoco:geom_solimp": tuple(float(v) for v in self.spec.piece_impedance),
            "mujoco:geom_solmix": float(self.spec.piece_mix),
        }
        for key, value in written.items():
            attribute = builder.custom_attributes.get(key)
            if attribute is None:
                raise RuntimeError(
                    f"bonds {self.spec.name!r}: piece_impedance is a MuJoCo contact parameter ({key}), and this "
                    "scene's arrangement stages no MuJoCo solver to read it"
                )
            if attribute.values is None:
                attribute.values = {}
            for body in bodies:
                for shape in builder.body_shapes[body]:
                    attribute.values[int(shape)] = value

    # -------------------------------------------------------------- phase 2
    def bind(self, model: newton.Model) -> None:
        """Find each bond's body again by its label: a collapse of fixed joints after the build renumbers bodies."""
        super().bind(model)
        index = {str(label): b for b, label in enumerate(model.body_label)}
        self.bodies = np.asarray([index[label] for label in self._a_labels], dtype=np.int32)

    # -------------------------------------------------------------- phase 3
    def contribute_hooks(self, hooks) -> None:
        if len(self.rows):
            hooks.add("post_solve", self._break, name=f"bonds:{self.spec.name}")

    def _break(self, state: newton.State, dt: float) -> None:
        del dt
        arrays = self._arrays
        if not arrays:
            return
        data = self.solver.mjw_data
        arrays["force"].zero_()
        wp.launch(
            _gather_kernel,
            dim=arrays["rows"],
            inputs=[data.ne, data.efc.type, data.efc.id, data.efc.force, arrays["bond_of"]],
            outputs=[arrays["force"]],
            device=arrays["device"],
        )
        wp.launch(
            _break_kernel,
            dim=len(self.rows),
            inputs=[
                arrays["force"],
                state.body_q,
                arrays["body"],
                arrays["normal"],
                arrays["strength"],
                arrays["shear_strength"],
                arrays["equality"],
                arrays["row"],
            ],
            outputs=[
                arrays["load"],
                arrays["peak"],
                self.broken,
                arrays["broke_at"],
                data.eq_active,
                arrays["enabled"],
                arrays["enabled_too"],
            ],
            device=arrays["device"],
        )

    # -------------------------------------------------------------- phase 4
    def on_scene_ready(self, pipeline: ScenePipeline) -> None:
        """Find the MuJoCo solver that holds the rows, and bind the kernels to its arrays.

        Raises:
            RuntimeError: When no MuJoCo solver steps the scene, when it does not hold every row
                (a bonded body on another solver), or when it steps more than one world.
        """
        if not len(self.rows):
            return
        spec = self.spec
        program = getattr(pipeline, "program", None)
        solver = getattr(program, "mujoco_solver", None)
        staging = (
            "A bond is a constraint in the MuJoCo solver, so both of its bodies have to be stepped by it: stage "
            "rss.MUJOCO_500HZ, or a particle table such as rss.LIQUID_120HZ, which put free rigid bodies there"
        )
        if solver is None or getattr(solver, "mjw_data", None) is None:
            raise RuntimeError(f"bonds {spec.name!r}: no MuJoCo-Warp solver steps this scene. {staging}.")
        data = solver.mjw_data
        if int(data.nworld) != 1:
            raise RuntimeError(f"bonds {spec.name!r}: the rows and the kernels address one world, and this is a fleet")
        view = solver.model
        held = {str(label): i for i, label in enumerate(view.mujoco.equality_constraint_label)}
        equality_of = {int(r): e for e, r in enumerate(solver.mjc_eq_to_newton_eq.numpy()[0]) if int(r) >= 0}
        missing = [label for label in self.labels if held.get(label) not in equality_of]
        if missing:
            raise RuntimeError(
                f"bonds {spec.name!r}: the MuJoCo solver holds {len(self.labels) - len(missing)} of its "
                f"{len(self.labels)} rows ({missing[0]} is not one). {staging}."
            )
        rows = np.asarray([held[label] for label in self.labels], dtype=np.int32)
        equality = np.asarray([equality_of[int(r)] for r in rows], dtype=np.int32)
        device = pipeline.model.device
        count = len(rows)
        bond_of = np.full(max(int(solver.mjw_model.neq), 1), -1, dtype=np.int32)
        bond_of[equality] = np.arange(count, dtype=np.int32)
        strength = np.asarray([p.strength for p in spec.points], dtype=np.float32)
        shear = np.asarray(
            [p.strength if p.shear_strength is None else p.shear_strength for p in spec.points], dtype=np.float32
        )
        enabled = view.mujoco.equality_constraint_enabled
        # The solver's own model is a view of the scene's when solvers are coupled, and Newton may give
        # the view its own copy of an array: both are written, so a notify from either keeps a bond broken.
        whole = pipeline.model.mujoco.equality_constraint_enabled
        self.solver = solver
        self.broken = wp.zeros(count, dtype=wp.int32, device=device)
        self._arrays = {
            "device": device,
            "rows": int(data.efc.force.shape[1]),
            "view_rows": rows,
            "equalities": equality,
            "bond_of": wp.array(bond_of, dtype=wp.int32, device=device),
            "equality": wp.array(equality, dtype=wp.int32, device=device),
            "row": wp.array(rows, dtype=wp.int32, device=device),
            "body": wp.array(self.bodies, dtype=wp.int32, device=device),
            "normal": wp.array(self.normals, dtype=wp.vec3, device=device),
            "strength": wp.array(strength, dtype=wp.float32, device=device),
            "shear_strength": wp.array(shear, dtype=wp.float32, device=device),
            "force": wp.zeros((count, 3), dtype=wp.float32, device=device),
            "load": wp.zeros(count, dtype=wp.float32, device=device),
            "peak": wp.zeros(count, dtype=wp.float32, device=device),
            "broke_at": wp.zeros(count, dtype=wp.float32, device=device),
            "enabled": enabled,
            "enabled_too": whole if len(whole) == len(enabled) else enabled,
        }
        # A thing built before this one may have stepped the scene to settle, and a graph captured then
        # holds no break kernel.
        if getattr(pipeline, "captured", False):
            pipeline.invalidate_graph()

    # -------------------------------------------------------------- phase 5
    def reset(self, pipeline: ScenePipeline, episode, *, world: int = 0) -> None:
        """Every bond made again, as built: the rows on in the solver and in the model, nothing remembered."""
        del pipeline, episode, world
        arrays = self._arrays
        if not arrays:
            return
        active = self.solver.mjw_data.eq_active
        host = active.numpy()
        host[0, arrays["equalities"]] = True
        active.assign(host)
        for key in ("enabled", "enabled_too"):
            host = arrays[key].numpy()
            host[arrays["view_rows"]] = True
            arrays[key].assign(host)
        self.broken.zero_()
        for key in ("force", "load", "peak", "broke_at"):
            arrays[key].zero_()

    # ---------------------------------------------------------------- runtime
    def _read(self, key: str) -> np.ndarray:
        if not self._arrays:
            return np.zeros(len(self.rows), dtype=np.float32)
        return self._arrays[key].numpy().copy()

    def let_go(self) -> np.ndarray:
        """Per bond, whether it has broken since the build or the last reset."""
        if self.broken is None:
            return np.zeros(len(self.rows), dtype=bool)
        return self.broken.numpy().astype(bool)

    def held(self) -> int:
        """How many bonds still hold."""
        return int(len(self.rows) - self.let_go().sum())

    def active(self) -> np.ndarray:
        """Per bond, whether the solver itself still holds its row (``eq_active``): never true of a broken one."""
        if not self._arrays:
            return np.ones(len(self.rows), dtype=bool)
        return self.solver.mjw_data.eq_active.numpy()[0][self._arrays["equalities"]].astype(bool)

    def forces(self) -> np.ndarray:
        """Each bond's constraint force on body ``a`` at the last substep, world [N], ``(n, 3)``; zero once broken."""
        return self._read("force").reshape(-1, 3)

    def loads(self) -> np.ndarray:
        """Each bond's load by its rule at the last substep [N]: what is compared with its strength."""
        return self._read("load")

    def peaks(self) -> np.ndarray:
        """Each bond's largest load since the build or the last reset [N]."""
        return self._read("peak")

    def broke_at(self) -> np.ndarray:
        """The load each broken bond let go at [N]; zero for one that holds."""
        return self._read("broke_at")

    def by_group(self) -> dict[str, tuple[int, int]]:
        """Seam name -> ``(bonds let go, bonds)``, for the groups :class:`BondPoint` names."""
        gone = self.let_go()
        out: dict[str, tuple[int, int]] = {}
        for point, g in zip(self.spec.points, gone, strict=True):
            broken, total = out.get(point.group, (0, 0))
            out[point.group] = (broken + int(g), total + 1)
        return out


# =============================================================================
# 5. The constructor
# =============================================================================
# Exported as ``rss.bond``.


def bond(
    *,
    name: str = "bond",
    a: str = "",
    b: str = "",
    at: Any = (),
    normal: Vec3 | None = None,
    strength: float = 1.0,
    shear_strength: float | None = None,
    points: Any = (),
    **kwargs,
) -> BondSpec:
    """Bonds between rigid bodies of the scene: each holds to its strength, then lets go for good.

    ``rss.bond(name="glue", a="lid", b="box", at=[...], normal=(0, 0, 1), strength=2.0)`` bonds one
    pair at several points. A set that joins many pairs, a shell's seams, passes ``points``.

    Args:
        name: Key in handles and readouts.
        a: The body that comes away: a free rigid body added to the scene before the bonds.
        b: The body it is bonded to, or ``""`` for the world.
        at: One point or several, ``(x, y, z)`` each, in the cell's work-surface frame [m], with both
            bodies where they are built. Each is one bond between ``a`` and ``b``.
        normal: The way ``a`` comes off ``b``; with it the bonds break in tension and shear and never in
            compression. ``None`` breaks on the size of the force.
        strength: The tension each bond holds [N].
        shear_strength: The shear each holds [N], or ``None`` for ``strength``.
        points: Further :class:`BondPoint`, between any pairs of bodies.
        **kwargs: Any other :class:`BondSpec` field, ``piece_impedance`` among them.
    """
    made = [
        BondPoint(
            a=a, b=b, at=tuple(float(v) for v in p), normal=normal, strength=strength, shear_strength=shear_strength
        )
        for p in np.asarray(at, dtype=np.float64).reshape(-1, 3)
    ]
    return BondSpec(name=name, points=(*made, *points), **kwargs)


register_entity("bond")(EntityBond)


# Import-only; running it directly would register the bond twice.
