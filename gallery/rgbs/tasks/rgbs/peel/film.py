# SPDX-License-Identifier: Apache-2.0
"""Protective film: the tape's physics, a different material, and a force readout.

Surface-protection film is the same mechanism as packing tape -- a thin sheet
held to a flat substrate by an adhesive that fails in mode I as a front travels
under it -- so this reuses :class:`~rgbs/tasks/rgbs/peel/tape/adhesion.py.AdhesionBonds`
rather than restating it.

:class:`EntityFilm` subclasses :class:`~rgbs/tasks/rgbs/peel/tape/entity.py.EntityTape`
rather than ``EntityCloth`` for one reason worth stating: the tape entity's
``on_scene_ready`` is not a settle, it is an **ordering** -- resolve sets, bond
against the *flat* strip, build the handle pin, settle, snapshot -- and every step
of that ordering is load-bearing (a bond-free settle slides the sheet across a
deliberately low-friction substrate and then bonds it wherever it slid to).
Re-deriving that ordering here would be re-deriving the bug it was written to
fix. Registration is by name, so subclassing does not disturb ``"tape"``.

.. note::

   That import reached into the library's old ``tasks.tape_peel`` while the tape
   suite was still awaiting its own port. It moved with
   ``SpecTape``/``EntityTape``/``AdhesionBonds`` when that port landed, and
   nothing else here changed -- the coupling was one line, deliberately.

**What is genuinely new here is a force readout.** Nothing else in this repo
reports a force on a deformable: ``particle_f`` is cleared every substep and holds
only external forces, a kinematic pin materialises no reaction, ``ContactProbe``
is rigid-shape-to-rigid-shape, and ``ParticleProbe`` carries positions and
velocities only. But the bond field's own restoring force is exact and already in
hand -- ``k`` times the opening displacement of every intact bond -- so
:meth:`EntityFilm.bond_load` is a reduction over arrays that exist.

**Why this asset publishes so much.** ``film_peel``'s Python evaluator computed
nine metrics by reaching into the entity with ``getattr``; a document cannot
threshold a state its entity does not publish, so every quantity that evaluator
read is declared here. ``peel_excess`` is the interesting one: it is what the
``peak`` combinator is pointed at, and it is stated **net of the standing load**
the bond field carries before anything touches it -- see :attr:`EntityFilm.rest_load`.

What is *not* here is the film's layout. Which particles are the tab, where the
panel sits, how the sheet is oriented -- those are facts about the scene
``tasks/peel`` stages, not about protective film, and an asset that
claimed them could not be re-used by a second task that laid the same film out
differently.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from robosimstudio_materials.cloth import TAPE_MATERIAL, ClothMaterialSpec

from robosimstudio.assetlib.record import AssetRecord
from robosimstudio.assetlib.registry import register_asset
from robosimstudio.assetlib.semantics.features import FeatureSpec
from robosimstudio.assetlib.semantics.published import PublishedState
from robosimstudio.assetlib.semantics.states import StateSpec
from robosimstudio.assetlib.sim import SimRequirements, contacts_with
from robosimstudio.assetlib.vocabulary.measures import cloth_lifted, held_particles
from robosimstudio.scene.registry import register_entity

from ..batched import instant

# Protective film IS tape, mechanically: the same bond field, the same break
# distance arithmetic, a different sheet. So it subclasses the tape entity rather
# than restating it, and that import is intra-task -- one asset reusing
# content, crossing no enforced seam. It named the library's old `tasks.tape_peel`
# until the tape port moved the entity here, which is where the two ports met.
from .tape import EntityTape, SpecTape

__all__ = [
    "FILM",
    "FILM_MATERIAL",
    "PEEL_SUCCESS_FRACTION",
    "ROW_BROKEN_FRACTION",
    "EntityFilm",
    "SpecFilm",
]

FILM_THICKNESS = 60e-6
"""Protective film thickness [m]. 60 um is mid-range for PE surface-protection
film (typical stock runs 30-100 um)."""
FILM_DENSITY = 920.0
"""Low-density polyethylene [kg/m^3]."""

_BENDING_RATIO = 0.23
"""Film bending rigidity as a fraction of the tape's.

Plate rigidity goes as ``E * t^3``. Against the tape suite's strip -- BOPP at
roughly 1.5 GPa and 50 um -- a 60 um LDPE film at roughly 0.2 GPa gives
``(0.2/1.5) * (60/50)^3 = 0.23``. Applied to ``TAPE_MATERIAL.edge_ke``, which was
itself *fitted* to the reference's measured droop.

**This is a derivation, not a measurement.** It is stated here rather than
registered as a ``MaterialRecord`` precisely because no drape bench has been run
on it: `examples/scripts/cloth_fold_drape_bench.py` is what would turn it into a calibration, and until
that happens claiming one would be claiming a number nobody took.
"""

PEEL_STRENGTH_RATIO = 0.075
"""Film adhesion as a fraction of the tape's.

Surface-protection film is specified to come off cleanly and runs about
0.3 N/25 mm; packing tape runs about 4 N/25 mm. 12 N/m against 160 N/m is
0.075, and that ratio scales ``bond_k`` -- ``break_distance`` is a geometric
cohesive-zone length and stays put, so per-bond strength (``k * break``) carries
the whole difference. Same caveat as the bending ratio: derived from published
peel-strength ranges, not measured in this simulator.
"""

FILM_MATERIAL: ClothMaterialSpec = replace(
    TAPE_MATERIAL,
    edge_ke=TAPE_MATERIAL.edge_ke * _BENDING_RATIO,
    areal_density=FILM_DENSITY * FILM_THICKNESS,
    thickness=FILM_THICKNESS,
)
"""A protective film, stated from its own physical constants.

``areal_density`` is density times thickness -- 0.0552 kg/m^2, against the tape's
0.2423 -- and everything not named above is the tape's, because it is the same
kind of object and inventing separate numbers for it would be inventing them.
Friction in particular stays **low**: the tape suite's own note is that friction
must not take the shear, or the bonds never load and no peel front forms.
"""

BOND_STIFFNESS = 3.0 * PEEL_STRENGTH_RATIO
"""Bond stiffness [N/m]: the tape's 3.0 scaled by the peel-strength ratio.

0.225 N/m, so a bond lets go at ``0.225 * 0.005`` = 1.125 mN against the tape's
15 mN.
"""
BOND_BREAK = 0.005
"""Opening displacement at which a bond lets go [m]. The tape's, unscaled -- a
cohesive-zone length is geometry."""
BOND_MARGIN = 0.015
"""How close to the surface a particle must rest to bond at all [m]."""

PEEL_SUCCESS_FRACTION = 0.95
"""Broken-bond fraction that counts as peeled.

The tape suite's, transcribed rather than re-derived: the sheet is peeled when
essentially every bond has let go. Not 1.0 -- a handful of bonds under the far
edge can sit inside their break threshold indefinitely while the film is plainly
off the panel, and a benchmark that waits for them measures patience.
"""

ROW_BROKEN_FRACTION = 0.999
"""Fraction of a width-row's bonds that must be broken for the row to count as
passed by the front. Effectively all of them -- a front that has passed a row
has passed all of it."""

CLEAR_HEIGHT = 0.03
"""How far above where it settled a particle counts as lifted clear [m].

Reported, never thresholded: ``film_clear`` is a fraction and nobody has
calibrated a gate for it, which is exactly why it is an ``:observe`` line in
both documents rather than a goal clause.
"""


@dataclass(frozen=True, kw_only=True)
class SpecFilm(SpecTape):
    """A protective film. The tape's spec with the film's defaults."""

    entity: str = "film"
    bond_k: float = BOND_STIFFNESS
    material: ClothMaterialSpec = field(default_factory=lambda: FILM_MATERIAL)


@register_entity("film")
class EntityFilm(EntityTape):
    """A film on a panel, plus the adhesive load the peel is working against."""

    spec: SpecFilm

    def __init__(self, spec: SpecFilm, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.rest_load: list[float] = [0.0]
        """The standing bond load at episode start [N], one reading per world.

        A **build** artefact rather than adhesion: the film is authored a
        millimetre above the panel and the anchors are frozen there, before the
        settle drops it onto the surface. Measured on ``film_peel/panel`` at full
        coverage: 0.0785 N, which is ``k`` times that same millimetre summed over
        its ~350 on-panel bonds. Recorded once the settle has run and subtracted
        from every reading, so ``peel_excess`` is the load the *policy* caused and
        a budget can be stated against a peel rather than against a placement
        convention.

        Per world, and measured per world rather than tiled, because it is the
        number every later reading is stated *net of*: a world subtracting a
        neighbour's floor would report the difference between two builds as
        something the policy did.
        """
        self._published: tuple[Any, dict[str, float]] | None = None

    # -------------------------------------------------------------- phase 4
    def snapshot_rest(self, pipeline) -> None:
        """Record the settled state, and the standing load each world carries.

        Taken here rather than at reset because a reset *restores* this state
        exactly -- particle positions, velocities and the bond field's active
        mask all come back to what this method saved -- so the two readings are
        the same number and this one is taken once instead of per episode.

        One host copy of ``particle_q`` for the whole fleet, sliced per world by
        the bond indices themselves: a read inside the loop would be one full
        transfer and one stream sync per world for the same bytes.
        """
        super().snapshot_rest(pipeline)
        positions = np.asarray(pipeline.state_0.particle_q.numpy(), dtype=np.float64)
        # One key for the whole loop, so the bond field is copied back once for
        # every world rather than once per world: nothing steps between these
        # readings, which is the same claim ``at`` makes during scoring.
        self.rest_load = [self._load_at(positions, world, at=("rest",))[0] for world in range(self._num_worlds)]

    # -------------------------------------------------------------- phase 5
    def reset(self, pipeline, episode, *, world: int = 0) -> None:
        super().reset(pipeline, episode, world=world)
        # A cached reading is keyed on the scoring context, which a new episode
        # builds fresh -- so without this the first step of episode 2 could be
        # scored on the last reading of episode 1.
        self._published = None

    # ---------------------------------------------------------------- measure
    def bond_load(self, probe, world: int = 0, *, at: Any = None) -> tuple[float, float]:
        """One world's total and worst-single adhesive restoring force [N].

        The bonds are linear springs from each stuck particle to a frozen anchor,
        so the force the peel is pulling against is ``k`` times the opening
        displacement, summed over the bonds still intact. Exact, not a proxy.

        Args:
            probe: The step's :class:`ParticleProbe`, already refreshed by the
                environment -- passed in rather than re-read so this costs no
                device-to-host copy of its own. A wave hands each world its own.
            world: Which world's block of the bond field to reduce.

        Returns:
            ``(total_n, worst_bond_n)``, both ``0.0`` before the bonds exist.
        """
        return self._load_at(probe.positions, world, at=at)

    def _load_at(self, positions: np.ndarray, world: int = 0, *, at: Any = None) -> tuple[float, float]:
        """One world's reduction, over an array of whole-model particle positions.

        Both arrays that reach here are in the same space and it is the *fleet's*:
        ``pipeline.state_0.particle_q`` holds every world's particles, and a
        per-world :class:`ParticleProbe` is handed that same array whole (its
        named sets carry the world's offset, not its positions). ``bond_idx`` is
        already in that space too -- ``AdhesionBonds.create`` builds each world's
        block from the template set plus that world's stride -- so the only
        slicing this does is picking out which *bonds* belong to ``world``.
        """
        bonds = getattr(self, "bonds", None)
        if bonds is None or bonds.bond_idx is None or not len(bonds.bond_idx):
            return 0.0, 0.0
        per_world = bonds.per_world or len(bonds.bond_idx)
        block = slice(int(world) * per_world, (int(world) + 1) * per_world)
        # Which bonds are still intact is the only one of the three that moves,
        # so it is the only one read from the device -- and that read is shared
        # across the worlds asking in the same instant. The indices and the
        # anchors were fixed when the field was created and live on the host;
        # copying them back per world per step was three transfers where the
        # arithmetic needed none.
        active = np.asarray(bonds.active_rows(at), dtype=bool)[block]
        if not active.any():
            return 0.0, 0.0
        idx = bonds.idx_host[block][active]
        anchors = bonds.anchor_host[block][active]
        opening = np.asarray(positions, dtype=np.float64)[idx] - anchors
        per_bond = np.linalg.norm(opening, axis=1) * float(self.spec.bond_k)
        return float(per_bond.sum()), float(per_bond.max())

    def peel_front(self, world: int = 0, *, at: Any = None) -> tuple[float, int]:
        """How far the peel front has travelled from the tab, in [0, 1] and rows.

        Contiguous fully-broken width-rows starting at row 0 (the tab end). A
        sheet torn out of the middle scores 0 here and a high broken fraction,
        which is exactly the distinction this measure exists to make.

        Instantaneous, not latched. The Python evaluator carried a monotone
        maximum of it on the grounds that an apparent retreat is noise on a
        marginal row -- true, and a property of *reading* the front rather than
        of the front; a document that wants the largest value an episode reached
        says ``(peak ...)`` and gets it from the combinator that owns that
        question.
        """
        bonds = getattr(self, "bonds", None)
        if bonds is None:
            return 0.0, 0
        row = bonds.read(world, at=at).row
        if not len(row):
            return 0.0, 0
        passed = int(np.argmax(row < ROW_BROKEN_FRACTION)) if (row < ROW_BROKEN_FRACTION).any() else len(row)
        return passed / len(row), passed

    def published_states(self, snapshot) -> dict[str, float]:
        """Everything the ``film_peel`` documents measure, by declared name.

        Cached per instant. Each clause that names a published state asks the
        entity for *all* of them, and a two-clause goal over a two-clause reward
        would otherwise recompute this four times a step -- eight device-to-host
        copies where the Python evaluator paid two.

        **The key is the scoring context, not the instant.** It was
        ``(step, sim_time)``, which is one instant per *scene* -- and a wave
        steps every world on one clock, so at three worlds that key served
        world 0's readings to worlds 1 and 2 and every episode scored on the
        first one's peel. The context object is built once per world per step,
        which is exactly the grain this cache wants, and it is the key
        ``EntityTape`` already uses for the same reason.
        """
        context = getattr(snapshot, "ctx", snapshot)
        if self._published is not None and self._published[0] is context:
            return self._published[1]

        world = int(getattr(context, "world", 0))
        name = self.spec.name
        bonds = getattr(self, "bonds", None)
        at = instant(context)
        broken = float(bonds.read(world, at=at).s) if bonds is not None else 0.0
        front, rows = self.peel_front(world, at=at)

        probe = snapshot.particles.get(name)
        total, worst = self.bond_load(probe, world, at=at) if probe is not None else (0.0, 0.0)

        ctx = context
        # The work arm's pinch, or the latch a policy declared on ``self.latch``
        # (the tape entity says why): ``RoutineFilmPeel`` carries the tab on a
        # ParticleDragger, and a reading that only looked for tab particles
        # near the tool centre reported ``held`` False through a 1.000 peel.
        handles = getattr(ctx, "handles", None)
        held, n_held = held_particles(
            ctx,
            name,
            arm=str(getattr(handles, "work_arm", "") or ""),
            particle_set="tab",
            dragger=self.latch,
        )
        rest_load = self.rest_load[world] if world < len(self.rest_load) else 0.0

        out: dict[str, Any] = {
            "broken_fraction": broken,
            "peel_front": front,
            "peel_front_rows": float(rows),
            "peel_tension": total,
            "rest_tension": rest_load,
            # Clamped at zero: the standing load is the floor, and a reading
            # below it is the film settling further onto the panel rather than a
            # negative pull.
            "peel_excess": max(0.0, total - rest_load),
            "worst_bond": worst,
            "film_clear": cloth_lifted(ctx, name, height=CLEAR_HEIGHT),
            "held": bool(held),
            "held_count": float(n_held),
        }
        self._published = (context, out)
        return out


FILM = register_asset(
    AssetRecord(
        name="film",
        summary="A protective film bonded to a panel, which peels when pulled off it.",
        tags=("film", "protective", "sheet", "adhesive"),
        features=(
            FeatureSpec(
                name="adhesion",
                kind="bond_field",
                payload={"stiffness": BOND_STIFFNESS, "break_distance": BOND_BREAK},
                provenance=(
                    "tape_peel's cohesive-zone model at 0.075 of its stiffness -- surface-protection "
                    "film runs about 0.3 N/25 mm against packing tape's 4 N/25 mm. The break distance "
                    "is unscaled because a cohesive-zone length is geometry, so per-bond strength "
                    "carries the whole difference."
                ),
            ),
            FeatureSpec(
                name="anchor",
                kind="anchor",
                payload={"release_on": "grasp"},
                provenance=(
                    "The tab is held at its as-laid pose until a gripper takes it. A property of how "
                    "this film is staged, not of the peeling task -- a drooping flap and a presented "
                    "one are different starting worlds."
                ),
            ),
        ),
        states=(
            StateSpec(
                name="peeled",
                measure="broken_fraction",
                op=">=",
                value=PEEL_SUCCESS_FRACTION,
                summary="Off the panel: essentially every bond has let go.",
                provenance=(
                    "tape_peel's gate, transcribed. Not 1.0 -- a handful of bonds under the far edge "
                    "can sit inside their break threshold indefinitely while the film is plainly off, "
                    "and waiting for them measures patience rather than skill."
                ),
            ),
            StateSpec(
                name="adhered",
                measure="broken_fraction",
                op="<=",
                value=0.05,
                summary="Still stuck down: essentially every bond intact.",
                provenance="Designed here. A sheet that has lost 5% of its bonds is being peeled, not stuck.",
            ),
        ),
        published=(
            PublishedState(
                name="broken_fraction",
                type="float",
                summary="Share of adhesive bonds that have let go.",
                tier="internal",
                range=(0.0, 1.0),
            ),
            PublishedState(
                name="peel_front",
                type="float",
                summary="How far along the sheet the peel front has travelled, from the tab.",
                tier="measurable",
                range=(0.0, 1.0),
            ),
            PublishedState(
                name="peel_front_rows",
                type="float",
                summary="The same front, as a count of fully-broken width-rows.",
                tier="measurable",
            ),
            PublishedState(
                name="peel_tension",
                type="float",
                summary="Total restoring force the intact bonds are carrying right now.",
                tier="internal",
                unit="N",
            ),
            PublishedState(
                name="rest_tension",
                type="float",
                summary="The standing bond load at episode start, which is a build artefact.",
                tier="internal",
                unit="N",
            ),
            PublishedState(
                name="peel_excess",
                type="float",
                summary="Bond load above the standing one: what the policy is pulling.",
                tier="internal",
                unit="N",
            ),
            PublishedState(
                name="worst_bond",
                type="float",
                summary="The single most-loaded intact bond.",
                tier="internal",
                unit="N",
            ),
            PublishedState(
                name="film_clear",
                type="float",
                summary="Share of the sheet lifted clear of where it settled.",
                tier="perceivable",
                range=(0.0, 1.0),
            ),
            PublishedState(
                name="held",
                type="bool",
                summary="Whether the gripper has hold of the tab.",
                tier="perceivable",
            ),
            PublishedState(
                name="held_count",
                type="float",
                summary="How many tab particles the gripper has hold of.",
                tier="measurable",
            ),
        ),
        sim=SimRequirements(
            domain="cloth",
            contacts=contacts_with("surface", "robot"),
            contributes_hooks=("bond_force", "handle_pin"),
            notes="The tape suite's arrangement, unchanged, and a thinner sheet makes the stability "
            "floor tighter rather than looser: at a coarser dt the film tunnels through the "
            "substrate, and a drag on top of that escalates to NaN. "
            "Bonds break on OPENING displacement, not total -- a servoed arm straddling the "
            "sheet presses particles down and drags them along the panel routinely, and film "
            "pressed onto its surface does not unstick.",
        ),
        provenance=(
            "Adhesion is tape_peel's cohesive-zone bond model, reused. The film's own constants are "
            "derived from published film properties (60 um LDPE; 0.3 N/25 mm peel strength) and are "
            "stated, not yet drape-calibrated -- examples/scripts/cloth_fold_drape_bench.py is what would turn the bending "
            "ratio into a calibration."
        ),
    )
)
