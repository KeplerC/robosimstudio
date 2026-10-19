# SPDX-License-Identifier: Apache-2.0
"""The working material for every deformable in a scene, and how to apply it.

Deliberately knows nothing about viser or about which page is hosting it. A
material rig and a two-armed fold task are the same problem from here: some
deformables were built with some materials, somebody is changing them, and each change
lands in one of three ways depending on what the field is.

Keeping it host-agnostic is what lets the tuning panels sit in the main studio
beside the robot, the policy runner and the episode score -- which is where they
have to be, because "is this cloth right" is not a question a bare strip can
answer.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from robosimstudio.assetlib.materials import get_material, list_materials
from robosimstudio.scene.tuning import write_live
from robosimstudio.specs import restage
from robosimstudio.specs.params import LIVE, RECAPTURE, param_ui
from robosimstudio_materials.cloth import ClothMaterialSpec

from .apply import apply_contact, contact_differs, rebuild_fields

__all__ = ["MaterialTuner"]


class MaterialTuner:
    """Every deformable's material in one scene, live where the physics allows.

    Args:
        env: The compiled environment. Rebuilt with the scene, so a task swap
            makes a new tuner rather than migrating this one -- the old working
            values described deformables that no longer exist.
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        # The deformables bin holds every particle entity, not only cloth: a
        # host-engine soft body rides it too, and its material is bulk matter
        # (Young/Poisson at REBUILD tier), not the live-tunable fabric fields
        # this panel maps onto ``tri_materials`` and friends. Duck-typed on the
        # one field the writer actually dereferences first, so a kind nobody
        # here anticipated degrades to "not on this panel" instead of a crash.
        #
        # ``getattr`` on the spec, and that is the whole of the bug this once
        # had: ``hasattr(e.spec.material, ...)`` guards only its *last* access,
        # so a spec with no ``material`` at all raised out of the expression
        # meant to tolerate it -- the crash the comment above promised to avoid,
        # for exactly the unanticipated kind it was written for. Both shipped
        # deformables declare one, so it waited for the third.
        #
        # Every write goes through the entity's own ``material`` tunable, which the inspector binds
        # too. A kind that adds fields routes them there -- paper's plastic trio into its crease
        # kernel's array, tape's adhesive into the bonds -- and a writer of this panel's own reached
        # none of them: ``yield_angle`` moved on the page and the crease kernel never saw it. A kind
        # whose engine copied the material at build offers no tunable (a bag on SuperDex) and is not
        # on this panel, since none of its sliders could move anything.
        pipeline = getattr(env, "pipeline", None)
        materials = {}
        for n, e in env.scene.by_name.items():
            if not e.carries_particles or not hasattr(getattr(e.spec, "material", None), "tri_ke"):
                continue
            tunable = next((t for t in e.tunables(pipeline) if t.name == "material"), None)
            if tunable is not None:
                materials[n] = tunable
        self.names: tuple[str, ...] = tuple(materials)
        self.entities: dict[str, Any] = {n: env.scene.by_name[n] for n in materials}
        self.working: dict[str, ClothMaterialSpec] = {n: t.spec for n, t in materials.items()}
        self.built: dict[str, ClothMaterialSpec] = dict(self.working)
        self.live: dict[str, _Writer] = {n: _Writer(t.write) for n, t in materials.items()}
        self.inert: dict[str, tuple[str, ...]] = {n: tuple(t.inert) for n, t in materials.items()}
        """Per cloth, the fields with nothing to write to, which the panel shows read-only."""
        self.origin: dict[str, str] = dict.fromkeys(materials, "")
        """Which named preset each cloth was last loaded from, if any."""
        self.selected: str = self.names[0] if self.names else ""

    def __bool__(self) -> bool:
        return bool(self.names)

    # ------------------------------------------------------------------ state
    @property
    def material(self) -> ClothMaterialSpec:
        """The selected cloth's working material."""
        return self.working[self.selected]

    @property
    def entity(self) -> Any:
        return self.entities[self.selected]

    def select(self, name: str) -> None:
        if name in self.working:
            self.selected = name

    # ------------------------------------------------------------------ edits
    def set_field(self, field: str, value: Any) -> str:
        """Change one field of the selected material. Returns the tier it cost.

        Live fields land in the model's arrays before this returns. Recapture
        fields are written and the graph dropped. Rebuild fields only move the
        working spec -- the sheet on screen was built with the old value and no
        assignment can change that.
        """
        name = self.selected
        spec = replace(self.working[name], **{field: value})
        self.working[name] = spec
        described = param_ui(spec, field)
        tier = described.tier if described is not None else ""
        if tier == LIVE:
            self.live[name].write(spec, only=field)
        elif tier == RECAPTURE:
            self.apply_contact()
        return tier

    def presets(self) -> tuple[str, ...]:
        """The named presets the selected cloth can wear: those of its built material's class.

        A tape's spec refuses a material with no adhesive and a paper's one with no plastic trio, so a
        towel preset adopted onto a label moved its sliders and then failed the rebuild; and no sheet
        can be made of rice.
        """
        kind = type(self.built[self.selected])
        return tuple(name for name in list_materials() if isinstance(get_material(name).spec, kind))

    def adopt(self, preset: str) -> None:
        """Load a named preset onto the selected cloth.

        Raises:
            ValueError: For a preset that is not one of :meth:`presets`.
        """
        name = self.selected
        if preset not in self.presets():
            raise ValueError(f"{name!r} is a {type(self.built[name]).__name__}; {preset!r} is not one")
        self.working[name] = get_material(preset).spec
        self.origin[name] = preset
        self.live[name].write(self.working[name])
        self.apply_contact()

    def revert(self) -> None:
        """Back to what this cloth was actually built with."""
        name = self.selected
        self.working[name] = self.built[name]
        self.origin[name] = ""
        self.live[name].write(self.working[name])
        self.apply_contact()

    def apply_contact(self) -> bool:
        """Push the soft-contact scalars, dropping a captured graph if needed.

        Model-wide rather than per cloth: ``soft_contact_ke`` and friends are
        single scalars with no per-cloth dimension to select on. With more than
        one deformable in the scene they are the one thing the sheets cannot
        disagree about, and the last write wins.
        """
        env = self.env
        if not contact_differs(env.model, self.material):
            return False
        apply_contact(env.model, self.material)
        if env.pipeline.captured:
            env.pipeline.invalidate_graph()
        return True

    # ---------------------------------------------------------------- rebuild
    def pending(self) -> tuple[str, ...]:
        """Rebuild-tier fields the built model cannot honour, across every cloth."""
        out: list[str] = []
        for name in self.names:
            changed = rebuild_fields(self.built[name], self.working[name])
            out += [f"{name}.{field}" if len(self.names) > 1 else field for field in changed]
        return tuple(out)

    def task_with_edits(self) -> Any:
        """The current task, with every working material substituted in.

        Handed back to ``Env`` to build a fresh model. The task already carries
        the scene, embodiment and gripper it was compiled with, so rebuilding
        from it reproduces the same pairing with different cloth.
        """
        task = self.env.task
        deformables = tuple(
            replace(spec, material=self.working[spec.name]) if spec.name in self.working else spec
            for spec in task.deformables
        )
        return restage(task, deformables=deformables)

    # ------------------------------------------------------------- provenance
    def preset_name(self) -> str:
        """The preset the selected cloth came from, or an empty string."""
        return self.origin.get(self.selected, "")

    def calibration(self) -> Any:
        """The recorded measurement behind the selected cloth's preset, if any."""
        preset = self.preset_name()
        if not preset:
            return None
        return get_material(preset).calibration


class _Writer:
    """One cloth's live fields, written through its entity's ``material`` tunable."""

    def __init__(self, write_field: Any) -> None:
        self.write_field = write_field

    def write(self, spec: Any, *, only: str = "") -> tuple[str, ...]:
        """Every live field of ``spec``, or just ``only``. Returns the names that landed."""
        return write_live(spec, self.write_field, only=only)
