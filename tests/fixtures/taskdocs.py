# SPDX-License-Identifier: Apache-2.0
"""Task documents, the catalog they are checked against, and an env to score them in -- all without a scene.

The checker, the gauntlet's static stages and the simulation check are pure
functions of a document and a vocabulary, and the docstrings that say so also
say their tests "hand in a five-line fake". This is that fake, written once:

* :func:`doc` -- a document string, parsed.
* :data:`DOCUMENT` -- one cube, one square, one containment goal: the smallest
  document every stage of the gauntlet passes, staged on a real cell with real
  physics so lowering it is lowering a real task.
* :func:`cube_asset` -- the cube :data:`DOCUMENT` places, registered for the
  block and taken back afterwards.
* :class:`FakeCatalog` -- both catalogs a document meets: the typechecker's
  four-method protocol and :mod:`~robosimstudio_bench.goals.simcheck`'s record
  lookup, from one table of assets.
* :class:`FakeEnv` -- what :func:`~robosimstudio_bench.run.runner.rollout` and
  the do-nothing floor drive, scripted step by step.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from robosimstudio.assetlib.semantics.atoms import ArgSpec, AtomSpec
from robosimstudio_bench.taskdoc.ast import TaskDoc
from robosimstudio_bench.taskdoc.parse import parse_problem
from robosimstudio_bench.taskdoc.reader import parse_sexp
from tests.fixtures.registries import restored

__all__ = ["CUBE", "DOCUMENT", "TWO", "Asset", "FakeCatalog", "FakeEnv", "cube_asset", "doc", "write"]

CUBE = "text_probe_cube"
"""The asset :data:`DOCUMENT` places; :func:`cube_asset` registers it."""

DOCUMENT = f"""
(define (problem text_probe)
  (:language put the cube in the square)
  (:staging (:scene bench_075) (:embodiment panda) (:physics mujoco_500hz@v1))
  (:objects cube - {CUBE})
  (:regions (square (:ranges ((0.05 -0.20 0.15 -0.10)))))
  (:goal (>= (containment cube square) 1.0)))
"""

TWO = """
(define (problem two)
  (:objects a b - block)
  (:regions (square (:ranges ((0.05 -0.20 0.15 -0.10)))))
  (:goal (and (>= (containment a square) 1.0) (>= (containment b square) 1.0)))
  (:reward (progress (containment REWARDED square) (:gate 1.0) (:gated-by (<= (speed GATED) 0.02)))))
"""
"""Two blocks, a goal about both and a reward with a mask: ``REWARDED`` and ``GATED`` name which block each reads."""


def doc(text: str) -> TaskDoc:
    """One ``(define (problem ...))`` string, parsed."""
    return parse_problem(parse_sexp(text))


def write(folder: str | Path, text: str, name: str = "t.task") -> Path:
    """``text`` as a document file in ``folder``, for the stages that read a path."""
    path = Path(folder) / name
    path.write_text(text)
    return path


@contextlib.contextmanager
def cube_asset(name: str = CUBE) -> Iterator[str]:
    """A 50 mm rigid cube registered as ``name`` for the block; the asset registry is put back after."""
    import robosimstudio as rss
    from robosimstudio.assetlib import register_asset
    from robosimstudio.assetlib.registry import ASSETS
    from robosimstudio.domains.rigid import Body

    with restored(ASSETS):
        register_asset(Body.from_spec(rss.box(name="cube", half=(0.025,) * 3, at=(0.0, 0.1, 0.03)), name=name))
        yield name


CONTAINMENT = AtomSpec(
    name="containment",
    summary="the share of an object inside a region",
    args=(ArgSpec(name="object"), ArgSpec(name="region")),
    returns=("fraction",),
)
SPEED = AtomSpec(name="speed", summary="how fast", args=(ArgSpec(name="object"),), returns=("speed",))
THREADED = AtomSpec(
    name="threaded",
    summary="a rod through a loop",
    args=(ArgSpec(name="rod"), ArgSpec(name="loop", kinds=("loop",))),
    returns=("turns", "clearance"),
)
WATCH = AtomSpec(
    name="watch", summary="observe only", args=(ArgSpec(name="object"),), directions=("observe",), returns=("v",)
)


@dataclass
class Asset:
    """One catalogued asset: its features (name -> kind), its states, and its ``sim`` declaration."""

    features: dict[str, str] = field(default_factory=dict)
    states: tuple[str, ...] = ()
    sim: Any = None


class FakeCatalog:
    """The typechecker's :class:`~robosimstudio_bench.taskdoc.typecheck.Catalog`, and simcheck's lookup.

    ``assets`` maps an asset name to an :class:`Asset`; ``atoms`` are real
    :class:`AtomSpec` objects, never registered, defaulting to four that cover
    the checker's branches: two whole-object measures, one whose second operand
    must be a ``loop``, and one that only observes.
    """

    def __init__(self, assets: dict[str, Asset] | None = None, atoms: Sequence[AtomSpec] | None = None) -> None:
        self.assets = dict(assets or {})
        self.atoms = {a.name: a for a in (atoms if atoms is not None else (CONTAINMENT, SPEED, THREADED, WATCH))}

    def has_asset(self, name: str) -> bool:
        return name in self.assets

    def features_of(self, asset: str) -> dict[str, str]:
        return dict(self.assets[asset].features) if asset in self.assets else {}

    def atom(self, name: str) -> AtomSpec | None:
        return self.atoms.get(name) or self.atoms.get(name.replace("-", "_"))

    def states_of(self, asset: str) -> tuple[str, ...]:
        return tuple(self.assets[asset].states) if asset in self.assets else ()

    def __call__(self, name: str) -> Any:
        """simcheck's catalog: the asset's record (``.sim``), or ``None`` for one nothing declares."""
        asset = self.assets.get(name)
        return None if asset is None else SimpleNamespace(name=name, sim=asset.sim)


class FakeEnv:
    """An env that plays back ``steps`` -- ``(reward, terminated, truncated, info)`` -- one per call.

    It records what it was asked: the actions stepped with, how often it was
    reset, and which step observers are installed. ``raise_at`` makes that
    step raise instead, as a solver that blew up would.
    """

    def __init__(self, steps: Sequence[tuple[float, bool, bool, dict]], *, max_steps: int = 0, raise_at: int = -1):
        self.steps = list(steps)
        self.max_steps = max_steps
        self.raise_at = raise_at
        self.actions: list[Any] = []
        self.resets = 0
        self.observers: list[Any] = []
        self.step_count = 0
        self.task = SimpleNamespace(full_name="suite/fake")
        self.episode = SimpleNamespace(episode_id=0, seed=0)
        self.closed = False

    def reset(self, episode_id: int = 0):
        self.resets += 1
        self.step_count = 0
        self.episode = SimpleNamespace(episode_id=episode_id, seed=100 + episode_id)
        return {"obs": 0}, {}

    def idle_action(self) -> str:
        return "idle"

    def step(self, action: Any, mode: str = ""):
        if self.step_count == self.raise_at:
            raise RuntimeError("the solver blew up")
        self.actions.append(action)
        reward, terminated, truncated, info = self.steps[min(self.step_count, len(self.steps) - 1)]
        self.step_count += 1
        return {"obs": self.step_count}, reward, terminated, truncated, dict(info)

    def add_step_observer(self, fn: Any) -> None:
        self.observers.append(fn)

    def remove_step_observer(self, fn: Any) -> None:
        self.observers.remove(fn)

    def close(self) -> None:
        self.closed = True
