# SPDX-License-Identifier: Apache-2.0
"""Changing a built model while it runs, for anything that is in one.

Newton keeps a cloth's constitutive parameters as Warp arrays the VBD solver
re-reads on every colour group of every iteration of every substep, so assigning
into one takes effect on the next substep and does so even inside a captured CUDA
graph, because a capture bakes the array's *pointer* and not the numbers behind
it. A rigid body's mass and its shapes' friction are Warp arrays on the same
model, reached the same way.

The difference is who is reading them. VBD reads the cloth arrays directly, so a
write is the whole of the change. A solver that keeps its own model -- MuJoCo
converts Newton's into an ``mjm`` at construction -- reads a *copy*, and a write
that nothing tells it about lands in an array it has stopped consulting. That is
the failure this module exists to not have: a slider that moves, a number that
changes, and a simulation that goes on behaving exactly as before.

Newton's answer is :meth:`~newton.solvers.SolverBase.notify_model_changed`, and
every shipped solver implements it. The flags name the arrays exactly:

* ``SHAPE_PROPERTIES`` -- ``shape_material_mu``, ``shape_material_ke``,
  ``shape_material_kd``, transforms and collision radii;
* ``BODY_INERTIAL_PROPERTIES`` -- ``body_mass``, ``body_inv_mass``,
  ``body_inertia``, ``body_inv_inertia``, ``body_com``;
* ``JOINT_DOF_PROPERTIES`` -- ``joint_target_ke``/``kd``, armature, friction and
  both limits.

So a writer here does two things: assign into the model, and say which category
changed. What the solver does about it is the solver's business, which is the
point -- this file does not know that MuJoCo has an ``mjm``.

**Why these are ``RECAPTURE`` and not ``LIVE``.** The assignment is live by the argument above. The *notification* is
host work: MuJoCo's re-sync launches kernels and, for some flags, flips a Python
flag that changes what the next contact conversion does. A captured graph replays
neither. Under capture the honest thing is to drop the graph and take it again --
which is what :meth:`~robosimstudio.core.pipeline.ScenePipeline.invalidate_graph`
is for, and which is free while the studio runs eager, exactly as it already is
for the cloth contact trio.

**What is not here.** Cloth. ``studio.tuning.apply.LiveMaterial`` already writes
it, is calibrated and is tested, and its per-field mapping onto ``tri_materials``,
``edge_bending_properties`` and ``tet_materials`` is genuinely cloth-specific
rather than a special case of anything. It needs no notification, for the reason
its own docstring gives at length.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

__all__ = [
    "ArrayWriter",
    "BodyWriter",
    "JointWriter",
    "ShapeWriter",
    "Tunable",
    "notify",
    "reach_island_copies",
    "set_kinematic",
]


@dataclass(frozen=True)
class Tunable:
    """One group of parameters an entity offers, and how a change to it lands.

    What an entity's ``tunables()`` returns, and the reason it is a protocol
    rather than a table: an asset's
    parameters are the asset's, the same way its published states are, and a
    consumer that carries its own list of what may be tuned on what is carrying a
    copy that can disagree.

    Attributes:
        name: Which group this is -- ``"contact"``, ``"mass"``, ``"drive"``,
            ``"material"``. Shown as the section heading.
        spec: A frozen dataclass whose fields carry ``ui()`` descriptions.
            **Read from the built model**, not from the spec the scene was compiled out of: those
            are different numbers whenever an import directive did not survive
            into the model, which is most of the time for a scene-declared body.
        write: ``(field, value) -> did it land``. ``False`` means the field has
            nothing in this entity to write to -- a bending stiffness on a mesh
            with no edges, a joint gain on a body with no joints -- which is
            information a panel can show rather than an error.
        summary: One line, for the section.
        inert: Fields of ``spec`` with nothing in this entity to write to, which a panel shows
            read-only rather than as a slider that moves nothing.
        write_many: ``({field: value}) -> the fields that landed``, several written as one set, or
            ``None`` to write them one at a time through ``write``. For a group whose fields are
            checked against each other: a tape's contact law refuses a ``peak_opening`` at or past
            its ``reach``, so a whole material written field by field from one law the spec allows
            to another can pass through one it refuses. It raises ``ValueError`` on a set it refuses.
    """

    name: str
    spec: Any
    write: Any
    summary: str = ""
    inert: tuple[str, ...] = ()
    write_many: Any = None

    def write_fields(self, values: dict[str, Any]) -> tuple[str, ...]:
        """Write several fields: as one set through :attr:`write_many`, else one at a time. Returns what landed."""
        if self.write_many is not None:
            return tuple(self.write_many(dict(values)))
        return tuple(name for name, value in values.items() if self.write(name, value))


def notify(pipeline: Any, flags: int) -> None:
    """Tell the scene's solvers that part of the model changed.

    Every one of them: a coupled arrangement runs several solvers over one
    model, and a shape whose friction moved is a shape any of them may be
    resolving contacts for. The program fans the notification out.

    Drops a captured graph, because the notification is host work a replay does
    not perform -- see the module docstring. Free while eager, which is the
    studio's default and the reason this is affordable at slider rate.

    Args:
        pipeline: The scene's :class:`~robosimstudio.core.pipeline.ScenePipeline`,
            or anything carrying a ``program`` or a ``solver``. ``None`` is a
            no-op, so a writer can be exercised against a bare model in a test.
        flags: A bit-mask of :class:`newton.ModelFlags`.
    """
    if pipeline is None or not flags:
        return
    program = getattr(pipeline, "program", None)
    if program is not None:
        program.notify(int(flags))
    else:
        from .mjdata import after_notify

        for solver in (
            getattr(pipeline, "solver", None),
            getattr(pipeline, "solver_rigid", None),
            getattr(pipeline, "solver_cloth", None),
        ):
            if solver is not None:
                solver.notify_model_changed(int(flags))
                after_notify(solver, int(flags))
    if getattr(pipeline, "captured", False):
        pipeline.invalidate_graph()


def reach_island_copies(pipeline: Any, *attrs: str) -> int:
    """Copy model arrays into every island view that holds its own copy of them. Returns how many moved.

    Newton 1.7's coupled solver gives an island a ``ModelView`` that may keep its own, compacted copy
    of a model array, and the island's solver reads that copy. Measured with ``views.py`` (fix round
    3, GPU 1): the VBD island of ``examples/03_materials/08_tape`` and of
    ``examples/01_cloth/02_engines`` holds its own ``tri_materials``, ``edge_bending_properties`` and
    ``particle_radius``, and the one of ``examples/03_materials/01_rope_memory`` its own
    ``vbd.dahl_eps_max`` and ``vbd.dahl_tau``, 18 rows of the model's 34. A write into the model alone
    then moves a slider and nothing else. ``notify`` refreshes the copies of what its flags name;
    nothing names particle, triangle or edge arrays, so a live write calls this instead. The copy is
    refreshed in place, through the solver's own projection, so a captured graph's pointers stay
    valid. An island that reads the model's array is left alone.

    Args:
        pipeline: The scene's pipeline; ``None`` or a one-solver scene does nothing.
        attrs: Model attribute names; a dotted name is a namespace's (``"vbd.dahl_tau"``).
    """
    solver = getattr(pipeline, "solver", None)
    entries = getattr(solver, "_entries", None)
    project = getattr(solver, "_project_compact_attributes", None)
    if not entries or project is None:
        return 0
    names = {a.replace(".", ":") for a in attrs}
    moved = 0
    for entry in entries.values():
        overrides = object.__getattribute__(entry.view, "_overrides")
        own = set()
        for name in names:
            space, _, leaf = name.rpartition(":")
            held = overrides.get(name) if not space else getattr(overrides.get(space), "__dict__", {}).get(leaf)
            if held is not None:
                own.add(name)
        if own:
            project(
                entry.view,
                entry.attribute_projections,
                exclude=set(),
                include=lambda attribute, own=own: attribute.name in own,
                source_model=True,
                update_existing=True,
            )
            moved += len(own)
    return moved


def set_kinematic(
    pipeline: Any, bodies: Iterable[int], on: bool, *, restore: Mapping[int, int] | None = None
) -> dict[int, int]:
    """Flag bodies kinematic, or give them back their old flags, in the model and in every island's copy.

    A kinematic body keeps whatever pose the state hands the solver: that is a
    hand with a capsule in a hard latch, and the way a script rolls or carries
    one body of a chain. Zeroing the body's inverse mass is not enough once the
    scene has two islands. Newton's coupled solver hands each island its own
    state, and VBD's ``coupling_notify_input_state_update`` turns a written pose
    on a body *without* :attr:`newton.BodyFlags.KINEMATIC` into a velocity and
    puts the old pose back (``vbd_coupling_kernels._update_vbd_body_input_state_kernel``),
    so a massless tip held by pose alone stays where it was and every roll written
    into it is dropped: measured on the two-cable twist example with its robot
    (``probe3.py``, GPU 1), 120 tip poses written, none kept, where the
    same drive on a one-island scene keeps them all. The island reads the flag
    off its ``ModelView``, whose ``body_flags`` is a copy-on-write override the
    parent model never refreshes, so the flag is written into each island's
    copy as well as the model's. VBD reads the array live, so no notification
    is needed for the flag; the caller still zeroes the inverse mass and
    inertia and calls :func:`notify` with ``BODY_INERTIAL_PROPERTIES``, which
    is what makes the hold rigid (the flag alone bent a rolled cable to 0.68
    straightness and twisted it unevenly; with the mass gone too it stays at
    1.000 and twists uniformly, ``probe5.py``).

    Only the two state bits are written. A stored body is ``DYNAMIC`` or
    ``KINEMATIC``, and an island's copy may also carry
    :attr:`newton.BodyFlags.PROXY`, Newton's mark on a body another island
    owns; that bit, and any other, is kept as it was.

    Args:
        pipeline: The scene's pipeline (``model`` and, in a coupled scene, ``solver``).
        bodies: Model body indices.
        on: ``True`` makes each body ``KINEMATIC``; ``False`` makes it
            ``DYNAMIC``, which is what every body the builder makes carries,
            or what ``restore`` says it was.
        restore: What a call with ``on=True`` returned. ``on=False`` puts each
            body it lists back into the state it had then, so a body that was
            kinematic before the pin is kinematic after the release.

    Returns:
        The flags each body had in the model before this call, keyed by body.
    """
    import newton

    bodies = [int(b) for b in bodies]
    kinematic = int(newton.BodyFlags.KINEMATIC)
    dynamic = int(newton.BodyFlags.DYNAMIC)
    before: dict[int, int] = {}
    targets: list[tuple[Any, dict[int, int]]] = [(pipeline.model.body_flags, {b: b for b in bodies})]
    solver = getattr(pipeline, "solver", None)
    entries = getattr(solver, "_entries", None)
    if entries and hasattr(solver, "view"):
        # Newton 1.6 exposes the island's local body map on its private entry only.
        for name, entry in entries.items():
            override = solver.view(name).overrides.get("body_flags")
            if override is None:
                continue  # this island reads the model's own array, written above
            local = entry.body_global_to_local.numpy()
            rows = {b: int(local[b]) for b in bodies if 0 <= b < len(local) and local[b] >= 0}
            if rows:
                targets.append((override, rows))
    state = kinematic | dynamic
    for array, rows in targets:
        flags = array.numpy()
        for body, row in rows.items():
            old = int(flags[row])
            if array is pipeline.model.body_flags:
                before[body] = old
            if on:
                want = kinematic
            elif restore is not None and body in restore:
                want = int(restore[body]) & state
            else:
                want = dynamic
            flags[row] = (old & ~state) | want
        array.assign(flags)
    return before


class ArrayWriter:
    """Assigns into a built model's arrays, through a host copy read fresh per write.

    A write downloads the array, edits this entity's rows and uploads it. The copy
    is not kept between writes: these arrays are model-wide -- every entity's
    shapes share one ``shape_material_mu`` -- so a copy kept by one writer goes
    stale the moment another writer, a script or ``ctl eval`` touches the array,
    and uploading it would put the other rows back the way they were.

    Args:
        model: The finalized model.
        pipeline: The scene's pipeline, for the notification. May be ``None``.
    """

    flags: int = 0
    """Which :class:`newton.ModelFlags` this writer's arrays fall under."""

    def __init__(self, model: Any, pipeline: Any = None) -> None:
        self.model = model
        self.pipeline = pipeline

    # ------------------------------------------------------------------ arrays
    def _array(self, attr: str) -> Any:
        """The model array ``attr`` names, or ``None``.

        A dotted name walks attribute by attribute (``"vbd.dahl_eps_max"`` is
        ``model.vbd.dahl_eps_max``): a solver's custom attributes sit in a
        namespace on the model, and a writer over one is otherwise no different
        from a writer over ``joint_target_ke``.
        """
        try:
            return functools.reduce(getattr, attr.split("."), self.model)
        except AttributeError:
            return None

    def _fetched(self, attr: str) -> np.ndarray | None:
        """The array as the model holds it now, as a host copy to edit."""
        array = self._array(attr)
        return None if array is None else np.array(array.numpy(), copy=True)

    def _flush(self, attr: str, values: np.ndarray) -> None:
        self._array(attr).assign(values)
        reach_island_copies(self.pipeline, attr)

    def _write(self, attr: str, rows: Sequence[int], value: Any, *, column: int | None = None) -> bool:
        """Write one value into ``rows`` of one array. ``False`` if there is nowhere to."""
        index = np.asarray(rows, dtype=np.int64)
        if not len(index):
            return False
        values = self._fetched(attr)
        if values is None or not len(values):
            return False
        if column is None:
            values[index] = value
        else:
            values[index, column] = value
        self._flush(attr, values)
        return True

    def _read(self, attr: str, rows: Sequence[int], *, column: int | None = None, default: float = 0.0) -> Any:
        """One value as the model currently holds it, or ``default``.

        The *first* row's, not a mean: these specs are a view of a set of shapes
        or bodies that a person is about to set to one value, and a mean would
        show a number none of them has. When they disagree the first one is at
        least a number something in the entity actually has.
        """
        index = np.asarray(rows, dtype=np.int64)
        array = self._array(attr)
        if array is None or not len(index):
            return default
        values = self._fetched(attr)
        if values is None or not len(values):
            return default
        row = values[index[0]]
        return float(row if column is None else row[column])

    # ------------------------------------------------------------------ notify
    def landed(self, wrote: bool) -> bool:
        """Notify the solvers if anything was written, and report what happened."""
        if wrote:
            notify(self.pipeline, self.flags)
        return wrote


class ShapeWriter(ArrayWriter):
    """One entity's contact material, over the shapes it built.

    Args:
        model: The finalized model.
        shapes: Shape indices belonging to this entity -- ``EntityRange.shape_indices``.
        pipeline: The scene's pipeline.
    """

    flags = 1 << 4
    """``ModelFlags.SHAPE_PROPERTIES``. Spelled as the bit rather than imported so
    this module costs no newton import at definition time; the value is pinned by
    a test against the enum."""

    FIELDS: ClassVar[dict[str, str]] = {
        "mu": "shape_material_mu",
        "mu_rolling": "shape_material_mu_rolling",
        "mu_torsional": "shape_material_mu_torsional",
        "restitution": "shape_material_restitution",
        "ke": "shape_material_ke",
        "kd": "shape_material_kd",
        "kf": "shape_material_kf",
        "ka": "shape_material_ka",
        "thickness": "shape_margin",
    }
    """Spec field to model array. The same set ``_MATERIAL_FIELDS`` (``scene/build/colliders.py``)
    copies between shapes, which is the list that already had to be right. ``thickness`` is Newton's
    ``shape_margin``, the surface offset a shape's ``ShapeConfig.margin`` sets: Newton 1.6 and 1.7 have
    no ``shape_thickness``, so a write there found no array and the slider wrote nothing."""

    def __init__(self, model: Any, shapes: Iterable[int], pipeline: Any = None) -> None:
        super().__init__(model, pipeline)
        self.shapes = np.asarray(list(shapes), dtype=np.int64)

    def read(self) -> dict[str, float]:
        """Every field as the model currently holds it."""
        return {name: self._read(attr, self.shapes) for name, attr in self.FIELDS.items()}

    def __call__(self, field: str, value: Any) -> bool:
        attr = self.FIELDS.get(field)
        if attr is None:
            return False
        return self.landed(self._write(attr, self.shapes, float(value)))


class BodyWriter(ArrayWriter):
    """One entity's mass properties, over the bodies it built.

    **Mass scales inertia with it.** Writing ``body_mass`` alone leaves a body
    that weighs a tenth of what it did and resists rotation exactly as much,
    which is not a lighter object -- it is an object made of a different
    substance in one axis and not the others, and it tumbles wrong. Scaling the
    tensor by the same factor is what "the same shape, less dense" means, and it
    is the assumption stated rather than hidden: a caller wanting a hollow shell
    wants a different tensor, not a smaller number here.

    A massless body -- a static's shell -- stays massless. Its inverse mass is
    what marks it fixed, and quietly giving it one would unweld it from the
    world, which is the failure ``preserve_body_inertia`` (``scene/build/bodies.py``) exists to
    avoid on the other side of the build.
    """

    flags = 1 << 3
    """``ModelFlags.BODY_INERTIAL_PROPERTIES``."""

    def __init__(self, model: Any, bodies: Iterable[int], pipeline: Any = None) -> None:
        super().__init__(model, pipeline)
        self.bodies = np.asarray(list(bodies), dtype=np.int64)

    def read(self) -> dict[str, float]:
        return {"mass": self._read("body_mass", self.bodies)}

    def __call__(self, field: str, value: Any) -> bool:
        if field != "mass":
            return False
        return self.landed(self._scale_mass(float(value)))

    def _scale_mass(self, mass: float) -> bool:
        if not np.isfinite(mass) or mass <= 0.0:
            # Zero would weld a dynamic body to the world -- and then the guard
            # below would refuse it its mass back -- and a negative one would
            # turn its inertia inside out. Neither is a mass.
            return False
        current = self._fetched("body_mass")
        if current is None or not len(self.bodies):
            return False
        wrote = False
        for body in self.bodies:
            was = float(current[body])
            if was <= 0.0:
                # Welded to the world. Leaving it alone is the whole point.
                continue
            factor = mass / was
            current[body] = mass
            self._write("body_inv_mass", [body], 1.0 / mass)
            self._scale_tensor("body_inertia", body, factor)
            self._scale_tensor("body_inv_inertia", body, 1.0 / factor)
            wrote = True
        if wrote:
            self._flush("body_mass", current)
        return wrote

    def _scale_tensor(self, attr: str, body: int, factor: float) -> None:
        values = self._fetched(attr)
        if values is None or body >= len(values):
            return
        values[body] = values[body] * factor
        self._flush(attr, values)


class JointWriter(ArrayWriter):
    """One entity's joint drive gains and limits, over the DOFs it built.

    An articulated asset -- a drawer, a cabinet -- is the case: how hard its
    joint is driven and how far it may travel are the two things somebody opening
    one by hand wants to change, and neither survives into anything a rigid
    body's contact material describes.
    """

    flags = 1 << 1
    """``ModelFlags.JOINT_DOF_PROPERTIES``."""

    FIELDS: ClassVar[dict[str, str]] = {
        "target_ke": "joint_target_ke",
        "target_kd": "joint_target_kd",
        "armature": "joint_armature",
        "friction": "joint_friction",
        "limit_lower": "joint_limit_lower",
        "limit_upper": "joint_limit_upper",
    }

    def __init__(self, model: Any, dofs: Iterable[int], pipeline: Any = None) -> None:
        super().__init__(model, pipeline)
        self.dofs = np.asarray(list(dofs), dtype=np.int64)

    def read(self) -> dict[str, float]:
        return {name: self._read(attr, self.dofs) for name, attr in self.FIELDS.items()}

    def __call__(self, field: str, value: Any) -> bool:
        attr = self.FIELDS.get(field)
        if attr is None:
            return False
        return self.landed(self._write(attr, self.dofs, float(value)))
