# SPDX-License-Identifier: Apache-2.0
"""``SpecTape`` and ``EntityTape``: the generic cloth, plus the tape's physics.

:class:`SpecTape` subclasses :class:`~robosimstudio_materials.cloth.ClothSpec` and
adds the fields adhesion needs; :class:`EntityTape` subclasses
:class:`~robosimstudio_materials.cloth.EntityCloth` and adds the bond field and
the handle pin. Neither the generic cloth entity nor the compiler learns that
adhesive tape exists -- the spec names its entity, the registry resolves it, and
everything tape-specific stays in this directory.

The one ordering that is not obvious: :meth:`EntityTape.on_scene_ready` creates
the bonds on the **flat** strip *before* settling, then settles, then snapshots.
The reference does the same and its own docstring says otherwise; a bond-free
settle sends the strip sliding across the deliberately low-friction box and the
bonds then form against wherever it slid to.

**The readings live here now, not in an evaluator.** The tape suite's Python
evaluator used to compute the broken fraction and the peel front itself, by
reaching into ``ctx.extras[cloth].bonds`` with its own idea of the attribute
names. :meth:`EntityTape.published_states` is the same arithmetic behind a
declared contract -- the reading half of the ``PublishedState`` entries on the
:data:`~rgbs/tasks/rgbs/assets/peel/tape.TAPE` record -- which is what lets a task document say
``(>= (broken-fraction tape) 0.95)`` and have it resolve against something rather
than against whatever attribute happened to be exposed.

Two of those readings carry **episode state**, and it belongs here rather than in
whatever scores the episode:

* ``peel_front`` is reported monotonically. A front that appears to retreat is
  measurement noise on a row that was marginal, not the tape re-adhering, and
  that is a property of the *instrument* -- the domain file has said
  ``(:monotone true)`` about it since before there was anything to read it.
* ``held_ever`` latches. "Did a gripper ever take hold of the tab" is not
  answerable from one instant, and an evaluator that latched it would be the
  second place this fact was tracked.

Both are cleared in :meth:`EntityTape.reset`, or the first steps of one episode
are described by the last steps of the one before it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import newton
import numpy as np
from robosimstudio_materials.cloth import TAPE_MATERIAL, ClothMaterialSpec, ClothSpec, EntityCloth

from robosimstudio.assetlib.vocabulary.measures import cloth_lifted, held_particles
from robosimstudio.core.pipeline import ScenePipeline, SubstepHooks
from robosimstudio.scene.registry import register_entity

from ...batched import instant
from .adhesion import BOND_BREAK, BOND_MARGIN, BOND_STIFFNESS, PIN_MAX_STEP, AdhesionBonds, HandlePin

# The film now lives in robosimstudio.assetlib.materials, with the droop fit
# that justifies its bending attached to it. Re-exported here because this is
# where it has always been imported from.
__all__ = [
    "CLEAR_HEIGHT",
    "GRASP_RADIUS",
    "ROW_BROKEN_FRACTION",
    "TAB_SET",
    "TAPE_MATERIAL",
    "EntityTape",
    "SpecTape",
    "bond_broken_fraction",
    "peel_front_progress",
]

ROW_BROKEN_FRACTION = 0.999
"""Fraction of a width-row's bonds that must be broken for the row to count as
passed by the front. Effectively all of them -- a front that has passed a row
has passed all of it."""

CLEAR_HEIGHT = 0.03
"""How far above where it settled a particle counts as clear of the box [m].

Uncalibrated, and reported rather than thresholded for exactly that reason: it
separates "lying on the box" from "lifted off it" by a comfortable margin at the
bond break distance's six times over, and nobody has measured the two
populations it would have to sit between to be a gate.
"""

GRASP_RADIUS = 0.02
"""How near the tool centre a tab particle must be to count as held [m].

The latch radius, and it is the same number for a reason rather than by
coincidence: closing two rigid fingers on a sub-millimetre film does not grip
it, so a scripted routine and a graph both carry the strip with a kinematic pin
over the particles inside this radius. "Is it held" and "what would a latch
seize" have to be the same question or the metric describes a grasp nobody made.
It used to be the *default argument* of the predicate the evaluator called,
which is the quietest possible way for two constants to agree.
"""

TAB_SET = "tab"
"""The particle set a grasp is measured against: the free end, past the box edge.

Named rather than spelled inline, because it is the same string in the asset's
``tab`` feature, in every layout's ``particle_sets``, and in the reading below.
"""


@dataclass(frozen=True, kw_only=True)
class SpecTape(ClothSpec):
    """A tape strip: a cloth grid, plus adhesion and an optional rigid handle.

    Attributes:
        entity: Registry name of the entity that realises this spec. Read by the
            compiler, which is how the corpus contributes an entity class without
            the compiler enumerating task types.
        surface_center: Centre of the box the strip is stuck to [m], in the
            cell's work-surface frame -- the frame ``pose`` is in. Lifted to
            world where the bonds are built, because bonds compare against
            live particle positions.
        surface_half: Its half-extents [m].
        surface_quat: The box's orientation, xyzw; its local +z is the face
            the strip bonds to. Identity for every axis-aligned layout, which
            is every one but the phone's -- there the face leans back 30
            degrees and the bonds open along its normal rather than world z.
        bond_k: Bond stiffness [N/m]. The reference's 3000 dyn/cm. A bond
            resists ``k * break`` = 0.015 N before letting go, so a row of seven
            across the width costs 0.105 N to lift -- which a gripper pulls
            easily and the tape's own 3.8 g never reaches.
        bond_break: Opening displacement at which a bond lets go [m].
        bond_margin: How close to the box top a particle must rest to bond [m].
        peel_back_dir: Unit vector from the tab back over the box -- the
            direction a peel drags the grasped end, which depends on how the
            strip is laid.
        tab_axis: Axis of the surface frame the strip runs along, for the
            per-row peel front; the world axis whenever ``surface_quat`` is
            the identity.
        handle_set: Name of the particle set held rigid until a gripper takes
            it, or ``""`` for a drooping flap.
        handle_pin_step: Rate limit on the handle pin [m per substep].

    The three bond defaults are :mod:`~rgbs/tasks/rgbs/assets/peel/tape/adhesion.py`'s own
    constants rather than the same numbers typed again. They were typed again
    once -- ``3.0`` here and ``BOND_STIFFNESS = 3.0`` beside the record that
    declares the feature -- and two copies of a physical constant is one copy and
    one place for it to go stale.
    """

    entity: str = "peel_tape"
    surface_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    surface_half: tuple[float, float, float] = (0.1, 0.08, 0.06)
    surface_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    bond_k: float = BOND_STIFFNESS
    bond_break: float = BOND_BREAK
    bond_margin: float = BOND_MARGIN
    peel_back_dir: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    tab_axis: int = 0
    handle_set: str = ""
    handle_pin_step: float = PIN_MAX_STEP
    material: ClothMaterialSpec = field(default_factory=lambda: TAPE_MATERIAL)


@register_entity("peel_tape")
class EntityTape(EntityCloth):
    """A tape strip: the generic cloth, with bonds and a handle pin."""

    spec: SpecTape

    def __init__(self, spec: SpecTape, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.bonds: AdhesionBonds | None = None
        self.pin: HandlePin | None = None
        self.latch: Any = None
        """The :class:`~robosimstudio.core.draggers.ParticleDragger` carrying
        the tab, when a policy holds it that way -- set by the policy that made
        it, read by ``held``.

        A scripted peel and the graph's ``sim.attach`` both carry a
        sub-millimetre sheet on a kinematic pin rather than by friction, and a
        ``held`` that only looked for tab particles within the grasp radius of
        the tool centre read False through every peel that scored 1.000 --
        the instrument the documents say separates a failed grasp from a
        failed peel was dead on the one policy that peels. A policy that
        latches says so here (``RoutineFilmPeel`` does); one that does not
        leaves the pinch reading exactly as it was."""
        self._front: list[float] = [0.0]
        self._front_rows: list[int] = [0]
        self._held_ever: list[bool] = [False]
        self._published_at: tuple[Any, dict[str, Any]] | None = None

    world_aware = True
    """Every phase of this strip carries a world axis.

    Four of them, and the two accumulators are the ones with no failure mode
    that looks like a failure: the bond field is a block per world over that
    world's particles, the handle pin holds and releases each world's tab on its
    own, the peel front's monotone maximum is a row per world, and ``held_ever``
    is a latch per world. A strip that kept one front would report the furthest
    *any* world had peeled as every world's progress -- a reward curve that
    climbs while nothing in that episode has moved.
    """

    @property
    def _num_worlds(self) -> int:
        """Worlds in this build. 1 when nothing replicated it."""
        return int(getattr(self.worlds, "num_worlds", 1) or 1)

    @property
    def _fleet_model(self):
        """The model that steps, which is the template unless one replicated it."""
        return self.stepping_model if self.stepping_model is not None else self.model

    # -------------------------------------------------------------- phase 2
    def bind(self, model: newton.Model) -> None:
        super().bind(model)
        worlds = self._num_worlds
        self._front = [0.0] * worlds
        self._front_rows = [0] * worlds
        self._held_ever = [False] * worlds
        self.bonds = AdhesionBonds(
            # The model that steps: the bond kernel indexes ``particle_q``, which
            # is the fleet's array, and world 1's strip is past the template's
            # end. Only the *device* is taken off it here -- every index the
            # field holds is built in ``create`` from the template set plus a
            # world's stride.
            self._fleet_model,
            self.particles,
            surface_center=self.placed((*self.spec.surface_center, 0.0, 0.0, 0.0, 1.0))[:3],
            surface_half=self.spec.surface_half,
            # The spec's own quaternion, not one round-tripped through
            # ``placed``: a cell's work-surface frame is a pure translation
            # (frames.py), so the orientation is the same in both frames, and
            # handing the identity through a matrix decomposition is how an
            # exactly-identity default stops being exactly identity.
            surface_quat=self.spec.surface_quat,
            k=self.spec.bond_k,
            break_distance=self.spec.bond_break,
            margin=self.spec.bond_margin,
            row_axis=self.spec.tab_axis,
            worlds=self.worlds,
        )

    # -------------------------------------------------------------- phase 3
    def contribute_hooks(self, hooks: SubstepHooks) -> None:
        """Bonds pre-solve (a force), the handle pin post-solve (a position).

        Both are contributed unconditionally at construction, before any capture,
        and both are gated on device -- so the set of launches in the graph is
        fixed even though the pin is released mid-episode.
        """
        if self.bonds is not None:
            hooks.pre_solve.append(self.bonds.apply)
        self._pin_hook_slot = hooks.post_solve
        self._pin_hook_slot.append(self._apply_pin)

    def _apply_pin(self, state: newton.State, dt: float) -> None:
        if self.pin is not None:
            self.pin.apply(state, dt)

    # -------------------------------------------------------------- phase 4
    def on_scene_ready(self, pipeline: ScenePipeline) -> None:
        """Resolve the sets, bond the flat strip, pin the handle, then settle.

        The order is the whole reason this method is overridden rather than
        inherited: bonds must exist before the first settle step runs.
        """
        self.resolve_sets(pipeline.state_0)
        self._name_the_sets_in_every_world()

        if self.bonds is not None:
            self.bonds.create(pipeline.state_0, verbose=False)

        if self.spec.handle_set:
            indices = self.set_indices(self.spec.handle_set)
            # The whole fleet's particle positions, in one host copy: the pin
            # spreads the template set into every world's block itself and
            # gathers each world's own as-laid pose out of this.
            self.pin = HandlePin(
                self._fleet_model,
                indices,
                np.asarray(pipeline.state_0.particle_q.numpy(), dtype=np.float64),
                max_step=self.spec.handle_pin_step,
                worlds=self.worlds,
            )

        self.settle(pipeline)
        self.snapshot_rest(pipeline)

    def _baseline_in_every_world(self, pipeline: ScenePipeline) -> None:
        """Give every world's probe the settled positions it measures against.

        ``DeformableCapability.snapshot_rest`` baselines ``self.probe``, which is
        the **first** world's. Every displacement measure goes through that
        baseline -- ``cloth_lifted`` is the one this suite publishes -- so on
        world 1 it raised ``RuntimeError: no settled baseline`` out of
        ``published_states``, which is every episode past the first failing to
        score at all.

        The same gap and the same right home as
        :meth:`_name_the_sets_in_every_world`: it belongs in the capability that
        opened the probes, and this suite's conversion may not reach there.

        One host copy of the settled state for the whole fleet, handed to each
        probe, rather than a device read per world: they are all reading the same
        instant and the transfer is the whole array whoever asks.
        """
        probes = getattr(self, "probes", None)
        if not probes or len(probes) < 2:
            return
        state = pipeline.state_0
        positions = state.particle_q.numpy()
        velocities = state.particle_qd.numpy()
        for probe in probes:
            probe.update_arrays(positions, velocities)
            probe.snapshot_baseline()

    def _name_the_sets_in_every_world(self) -> None:
        """Register the resolved sets on every world's probe, at its own offset.

        ``EntityCloth.resolve_sets`` names them on ``self.probe`` alone, which is
        the **first** world's -- so a wave's other probes know the strip's whole
        range (``bind`` gave them that) and nothing else. The tab is the one that
        matters: ``held_particles`` asks for ``"<name>/tab"``, and on world 1 that
        raised ``KeyError: unknown particle set 'film/tab'`` out of the middle of
        ``published_states``, which is every episode past the first failing to
        score at all.

        The right home for this is ``resolve_sets`` itself -- every cloth in the
        corpus has the same gap and none of the others has been asked to run
        replicated yet. It is here because this suite's conversion may not reach
        into the shared domain; see the branch's report.

        Idempotent for world 0, whose sets are already registered at offset zero.
        """
        probes = getattr(self, "probes", None)
        if not probes or not self.sets:
            return
        stride = int(getattr(self.worlds, "particles_per_world", 0) or 0)
        for world, probe in enumerate(probes):
            for key, indices in self.sets.items():
                probe.add_set(key, np.asarray(indices, dtype=np.int64) + world * stride)

    def snapshot_rest(self, pipeline: ScenePipeline) -> None:
        super().snapshot_rest(pipeline)
        self._baseline_in_every_world(pipeline)
        if self.bonds is not None:
            # The settled bond field, so a reset restores it bit-exact rather
            # than "all intact" -- a bond that broke during the settle is part
            # of the scene's start state, not damage to be undone.
            self.bonds._rest_active = np.asarray(self.bonds.bond_active.numpy(), dtype=np.int32).copy()

    # -------------------------------------------------------------- phase 5
    def reset(self, pipeline: ScenePipeline, episode, *, world: int = 0) -> None:
        """This world's bonds re-intact, its tab re-pinned, its accumulators cleared.

        The particles themselves are the base class's business and it restores
        the whole fleet in one assign, because the rest snapshot already holds
        every world's copy. What is *this* world's alone is everything below: a
        wave resets its worlds one at a time, so re-intacting the whole bond
        field here would undo the peel of a world that had already been reset
        and started working.
        """
        super().reset(pipeline, episode, world=world)
        if self.bonds is not None:
            self.bonds.reset(world)
        if self.pin is not None:
            self.pin.set_enabled(True, world=world)
        # The policy's latch goes with the episode that installed it. A
        # successful peel ends MID-TAKE -- the evaluator terminates on the gate
        # -- so the dragger is still active and still seized when the episode
        # closes, and an entity that kept pointing at it would report ``held``
        # true from step 0 of the next episode. That is the shape of the
        # manufactured success this suite has been bitten by before.
        self.latch = None
        self._front[world] = 0.0
        self._front_rows[world] = 0
        self._held_ever[world] = False
        self._published_at = None

    # ---------------------------------------------------------------- runtime
    def release_handle(self, world: int = 0) -> None:
        """Let go of one world's rigid pull-tab, e.g. when its gripper takes it.

        Defaulted, so the studio's release button and every single-world caller
        say what they always said. A wave's tools pass the world they are
        driving: one hand closing on its tab is not a reason for another
        episode's strip to come unpinned.
        """
        if self.pin is not None:
            self.pin.set_enabled(False, world=world)

    def observe(self, state: newton.State, step: int | None = None, *, readback=None) -> dict[str, float]:
        """The cloth summary, plus how much of the strip has let go.

        World 0's, which is the world an ``observe`` has always been about: this
        is the single-world observation surface, and a wave scores through
        :meth:`published_states` instead.
        """
        out = super().observe(state, step, readback=readback)
        if self.bonds is not None:
            out["bond_broken_fraction"] = self.bonds.read(0).s
        return out

    # ---------------------------------------------------------------- measure
    def published_states(self, snapshot) -> dict[str, Any]:
        """Everything the ``tape`` record declares it exposes, by name.

        The reading half of the declaration. A goal or a reward naming
        ``broken-fraction`` resolves through here, and so does a description --
        which is the whole point of declaring rather than reaching: before this,
        every consumer indexed ``bonds.read()`` with its own spelling.

        One device read per control step, not one per clause. The bond field is
        copied back from device to be read, and a goal, a reward and three
        observations over one strip would otherwise pay for that five times in
        the same instant while nothing had moved between them. The cache is keyed
        on the context object itself -- the environment builds exactly one per
        step -- so it can neither serve a stale answer nor be fooled by a
        recycled ``id``.
        """
        context = getattr(snapshot, "ctx", snapshot)
        cached = getattr(self, "_published_at", None)
        if cached is not None and cached[0] is context:
            return cached[1]
        out = self._measure(context)
        self._published_at = (context, out)
        return out

    def _measure(self, context) -> dict[str, Any]:
        """The readings themselves, computed once per step by the caller above.

        Takes the unwrapped context rather than the snapshot, because the two
        scene predicates it calls are the toolkit's and were written against
        ``WorldState``.
        """
        if self.bonds is None:
            return {}
        world = int(getattr(context, "world", 0))
        state = self.bonds.read(world, at=instant(context))
        broken = float(state.s)

        rows = np.asarray(state.row)
        if len(rows):
            passed = int(np.argmax(rows < ROW_BROKEN_FRACTION)) if (rows < ROW_BROKEN_FRACTION).any() else len(rows)
            front = passed / len(rows)
            # Monotone: an apparent retreat is noise on a marginal row, so the
            # front is the furthest it has reached rather than where it reads
            # this instant. Per world, because the furthest one episode's peel
            # has reached says nothing about another's.
            if front > self._front[world]:
                self._front[world], self._front_rows[world] = front, passed

        # Whether a hand has the tab, and how far the strip has come off the box.
        # Both are read from the scene rather than from the bond field, and both
        # answer honestly when there is no scene to read: a cell with no arm has
        # nothing holding anything -- which is what ``held_particles`` itself
        # says -- and a strip with no particle probe has no displacement to
        # report. Guarded rather than raising because every consumer of a
        # published state raises when one is *absent*, so an entity that
        # published a quantity only sometimes would be an entity whose task
        # crashes on the scenes that lack an arm.
        handles = getattr(context, "handles", None)
        held, count = (False, 0)
        if handles is not None and getattr(handles, "arms", None):
            # Against the WORK arm's tool centre, not the first arm declared:
            # yam_i2rt lists ``right`` first and ``handle`` works with ``left``,
            # so the pinch reading used to measure the idle hand. And through
            # the latch when a policy declared one -- see ``latch`` above.
            held, count = held_particles(
                context,
                self.name,
                arm=str(getattr(handles, "work_arm", "") or ""),
                particle_set=TAB_SET,
                radius=GRASP_RADIUS,
                dragger=self.latch,
            )
        self._held_ever[world] = self._held_ever[world] or bool(held)
        clear = 0.0
        if self.name in getattr(context, "particles", {}):
            clear = float(cloth_lifted(context, self.name, height=CLEAR_HEIGHT))

        return {
            "bond_fraction": 1.0 - broken,
            "broken_fraction": broken,
            "peel_front": self._front[world],
            "peel_front_rows": float(self._front_rows[world]),
            "anchored": bool(self.pin is not None and self.pin.enabled_for(world)),
            "held": bool(held),
            "held_count": float(count),
            "held_ever": bool(self._held_ever[world]),
            "clear": clear,
        }


def bond_broken_fraction(ctx, cloth: str) -> float:
    """Fraction of a strip's bonds that have let go, in [0, 1].

    Meaningless without an :class:`~rgbs/tasks/rgbs/assets/peel/tape/adhesion.py.AdhesionBonds`,
    which is why this lives beside it rather than in the shared predicate
    library. Zero when the scene has not bonded yet -- fail-quiet, so an
    evaluator built against a half-built scene reports nothing rather than
    raising mid-episode.
    """
    entity = ctx.extras.get(cloth)
    bonds = getattr(entity, "bonds", None)
    return 0.0 if bonds is None else bonds.read(int(getattr(ctx, "world", 0)), at=instant(ctx)).s


def peel_front_progress(ctx, cloth: str) -> tuple[float, int]:
    """How far the peel front has travelled from the tab, in [0, 1] and in rows.

    Contiguous fully-broken width-rows starting at row 0 (the tab end). A strip
    torn out of the middle scores 0 here and a high broken fraction, which is
    exactly the distinction this measure exists to make.

    Instantaneous, unlike :meth:`EntityTape.published_states`' own reading: a
    caller wanting the monotone one reads the published state, and a caller
    doing its own latching wants the raw number.
    """
    entity = ctx.extras.get(cloth)
    bonds = getattr(entity, "bonds", None)
    if bonds is None:
        return 0.0, 0
    row = bonds.read(int(getattr(ctx, "world", 0)), at=instant(ctx)).row
    if not len(row):
        return 0.0, 0
    passed = int(np.argmax(row < ROW_BROKEN_FRACTION)) if (row < ROW_BROKEN_FRACTION).any() else len(row)
    return passed / len(row), passed
