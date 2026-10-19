# SPDX-License-Identifier: Apache-2.0
"""What an arrangement, and each backend in it, becomes on the device a scene is built on.

On a GPU, nothing: every function here is the identity there. On a CPU, two
things, at two moments, both from what the backends registered
(:attr:`~robosimstudio.specs.physics.BackendInfo.devices`,
:attr:`~robosimstudio.specs.physics.BackendInfo.cpu_params`) rather than from
a list of names here:

* **Before anything is built**, :func:`arrangement_for` adapts the arrangement.
  :data:`~robosimstudio.specs.physics.GENERIC` -- what a scene gets when nobody
  staged one -- becomes :data:`~robosimstudio.specs.physics.CPU`. Any other
  arrangement is kept, except that a backend which cannot run on a CPU is
  replaced by the one ``CPU`` uses for that domain. It has to happen this early
  because the arrangement picks entity classes too: a sponge on SuperDex is a
  different entity from a sponge on VBD.
* **When islands form**, :func:`on_device` gives each backend its CPU kwargs --
  ``use_mujoco_cpu=True`` for MuJoCo -- and refuses one that states otherwise or
  that an entity names itself and cannot run there.

Every change is returned as a sentence, and the build prints them on its
platform line: the physics a scene runs is never different from what was
written without the page saying how.
"""

from __future__ import annotations

import functools
from dataclasses import replace
from typing import Any

from robosimstudio.specs.physics import (
    BACKENDS,
    CPU,
    ENGINES,
    GENERIC,
    ArrangementSpec,
    BackendInfo,
    BackendSpec,
    get_backend,
)

from .errors import CompositionError

__all__ = ["arrangement_for", "device_facts", "on_device", "runs_on_cpu"]


def _is_cpu(device: Any) -> bool:
    return str(getattr(device, "alias", device)) == "cpu"


def device_facts(spec: BackendSpec) -> BackendInfo | None:
    """The registered facts about where ``spec``'s solver runs, or ``None`` for an unknown alias.

    A dotted path is an anonymous backend -- no defaults, no schema -- except
    here: a path naming a registered backend's class carries what that backend
    needs to run *correctly* on a CPU -- its ``cpu_params`` and world limit -- so
    ``backend("newton.solvers.SolverMuJoCo")`` runs MuJoCo's C library there as
    ``backend("mujoco")`` does, rather than MuJoCo-Warp on Warp's CPU device with
    no one-world limit. Never its refusal: a dotted path is an explicit choice,
    which is what makes ``newton.solvers.SolverVBD`` the way to run VBD on a CPU
    anyway. And nothing else: what it is handed stays exactly what it states.
    """
    try:
        info = get_backend(spec.solver)
    except KeyError:
        return None
    if info.devices or "." not in spec.solver:
        return info
    named = _registered_ancestor(spec.solver)
    if named is None:
        return info
    return replace(info, cpu_params=named.cpu_params, cpu_max_worlds=named.cpu_max_worlds, cpu_note=named.cpu_note)


def _class_at(path: str) -> type | None:
    import importlib

    module, _, name = path.rpartition(".")
    try:
        found = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError, ValueError):
        return None
    return found if isinstance(found, type) else None


def _registered_ancestor(path: str) -> BackendInfo | None:
    """The registered backend with CPU facts whose class ``path`` is, or subclasses -- by class, not by spelling.

    Only backends that registered ``cpu_params`` are imported to ask (MuJoCo,
    today), so naming a dotted path imports no engine it is unrelated to.
    """
    ENGINES.load()
    candidates = tuple(known for _, known in BACKENDS.items() if known.cpu_params)
    index = _ancestor_index(path, tuple(known.cls for known in candidates))
    return candidates[index] if index >= 0 else None


@functools.lru_cache(maxsize=256)
def _ancestor_index(path: str, bases: tuple[str, ...]) -> int:
    """Which of ``bases`` the class at ``path`` is or subclasses, or -1. Cached: a scene asks once per entity."""
    cls = _class_at(path)
    if cls is None:
        return bases.index(path) if path in bases else -1
    for index, base_path in enumerate(bases):
        base = _class_at(base_path)
        if base is not None and issubclass(cls, base):
            return index
    return -1


def runs_on_cpu(spec: BackendSpec) -> bool:
    """Whether ``spec``'s backend may run on a CPU: it says so, or says nothing.

    A backend that registered no devices -- a dotted path nobody registered, a
    task's own solver -- is never refused: whoever wrote it knows, and its
    constructor can say.
    """
    info = device_facts(spec)
    if info is None:
        return True  # an unknown alias fails in the composer, by name
    return not info.devices or "cpu" in info.devices


def _domain_words(domains: list[str]) -> str:
    words = ["everything else" if d == "*" else d for d in domains]
    return ", ".join(words)


def arrangement_for(arrangement: ArrangementSpec, *, is_cpu: bool) -> tuple[ArrangementSpec, tuple[str, ...]]:
    """``arrangement`` as it runs on this device, and what changed, one sentence each.

    On a CPU, :data:`GENERIC` becomes :data:`CPU` whole: nobody chose its
    numbers, so the measured CPU ones serve better than GENERIC's own with VBD
    swapped out. Any other arrangement keeps its rates, coupling and collision
    law, and each backend that cannot run on a CPU is replaced by what ``CPU``
    uses for the same domain -- and when that falls to the ``"*"`` entry, the
    domains ``CPU`` routes specially (cloth, rope, soft) get their own entries,
    so a towel that fell to VBD lands on SuperDex rather than on MuJoCo.

    The ``"*"`` entry's replacement reuses the arrangement's own spec for the
    same solver where it has one, so free bodies that were on VBD join the arm's
    MuJoCo island instead of founding a second one beside it. Every other
    domain takes ``CPU``'s own spec for it: a rope that fell to VBD lands on
    ``CPU``'s one-way rope island, not on whatever SuperDex island the
    arrangement had for its sponge.
    """
    if not is_cpu:
        return arrangement, ()
    if arrangement == GENERIC:
        return CPU, ("GENERIC → CPU: MuJoCo for the arm and rigid bodies, SuperDex for cloth, rope and soft bodies",)

    table = list(arrangement.backends)
    stated = dict(table)

    def replacement(domain: str) -> BackendSpec:
        wanted = CPU.backend_for(domain)
        if domain != "*":
            return wanted
        own = next((s for _, s in table if s.solver == wanted.solver and runs_on_cpu(s)), None)
        return own or wanted

    moved: dict[str, list[tuple[str, str]]] = {}
    out: list[tuple[str, BackendSpec]] = []
    for domain, spec in table:
        if runs_on_cpu(spec):
            out.append((domain, spec))
            continue
        sub = replacement(domain)
        out.append((domain, sub))
        moved.setdefault(spec.solver, []).append((domain, sub.solver))
    fallback = stated.get("*")
    if fallback is not None and not runs_on_cpu(fallback):
        for domain, _ in CPU.backends:
            if domain != "*" and domain not in stated:
                sub = replacement(domain)
                out.append((domain, sub))
                moved.setdefault(fallback.solver, []).append((domain, sub.solver))
    if not moved:
        return arrangement, ()
    notes = []
    for solver, pairs in moved.items():
        by_target: dict[str, list[str]] = {}
        for domain, target in pairs:
            by_target.setdefault(target, []).append(domain)
        where = "; ".join(f"{_domain_words(domains)} → {target}" for target, domains in by_target.items())
        notes.append(f"{solver} has no CPU path: {where}")
    return replace(arrangement, backends=tuple(out)), tuple(notes)


def on_device(
    spec: BackendSpec, device: Any, *, owner: str, domain: str, declared: bool = False, num_worlds: int = 1
) -> BackendSpec:
    """``spec`` as it is constructed on ``device``: with its backend's CPU kwargs on a CPU.

    Otherwise the identity, and for a backend that registered no CPU kwargs.

    On a GPU it refuses a CPU-only solver that is not a host engine: a host
    engine (SuperDex) runs on the CPU beside any device, and is how a material
    with no GPU path sits in a GPU scene.

    Args:
        declared: Whether an entity named ``spec`` itself (``sim=SimRequirements(backend=...)``)
            rather than the arrangement serving its domain. Only that is refused
            on a CPU for having no CPU path: the entity class was chosen for
            that engine, so nothing can stand in for it, while an arrangement's
            backend reached here either was replaced by :func:`arrangement_for`
            before the build -- every build through ``compile_scene`` -- or was
            handed to ``compose`` directly by a caller who chose it, and runs,
            slowly.
        num_worlds: Worlds in the model. Past the backend's ``cpu_max_worlds``
            its CPU kwargs are not applied: ``compile_scene`` refuses such a
            build, so only a direct ``compose`` reaches here with one, and the
            only engine that steps it is the one stated -- MuJoCo-Warp on Warp's
            CPU device rather than MuJoCo's one-world C library.

    Raises:
        CompositionError: On a CPU, when an entity-declared backend cannot run
            there, or when ``spec`` states a kwarg its CPU form sets otherwise
            (``use_mujoco_contacts=False``, which MuJoCo's C backend would
            ignore rather than refuse).
    """
    if not _is_cpu(device):
        try:
            info = get_backend(spec.solver)
        except KeyError:
            return spec
        # A host engine runs on the CPU beside any device; a solver that steps
        # Newton's arrays where they live has to be able to run there.
        if info.devices and "cuda" not in info.devices and not info.host_engine:
            raise CompositionError(
                f"{owner!r} ({domain}) is on {spec.solver!r}, which runs on {' and '.join(info.devices)} only, and "
                f"this scene is on {getattr(device, 'alias', device)}. Build it on a CPU (device='cpu'), or register "
                "it as a host engine (register_backend(..., host_engine=True)) to step it on the CPU beside the GPU."
            )
        return spec
    info = device_facts(spec)
    if info is None:
        return spec
    if declared and info.devices and "cpu" not in info.devices:
        instead = CPU.backend_for(domain)
        raise CompositionError(
            f"{owner!r} ({domain}) is on {spec.solver!r}, which runs on {' and '.join(info.devices)} only, and this "
            f"scene is on a CPU. Stage rss.CPU, or put {domain!r} on rss.backend({instead.solver!r}"
            f"{', substeps=2' if instead.substeps == 2 else ''}) -- or build on a GPU."
        )
    if not info.cpu_params or (info.cpu_max_worlds and int(num_worlds) > info.cpu_max_worlds):
        return spec
    stated = spec.kwargs()
    for key, value in info.cpu_params:
        if key in stated and stated[key] != value:
            how = f" ({info.cpu_note})" if info.cpu_note else ""
            raise CompositionError(
                f"{owner!r} ({domain}) states {spec.solver}({key}={stated[key]!r}), but on a CPU {spec.solver!r} "
                f"runs with {key}={value!r}{how}. Drop it, or build on a GPU."
            )
    return spec.with_(**dict(info.cpu_params))
