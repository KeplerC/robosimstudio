# SPDX-License-Identifier: Apache-2.0
"""Which kinds this checkout carries, for a test that has to say "if it is here".

The deformable kinds -- ``cloth``, ``soft`` and the rope -- are the modules of
``robosimstudio_materials``, a package the library never imports and a release
may leave out, whole or in part. Most tests about a kind belong *to* the kind and
are withheld with it. A few are about a seam the library keeps on both sides of
that line -- which layouts exist, which actor kinds are registered, which feature
kinds have a contract -- and those have to state both answers rather than assume
the kind. :func:`has_pack` is the one question they ask.

The other half is a *tree without* a kind, for the tests that prove the library
is whole without it. That is a fresh process whose ``robosimstudio_materials`` is
a copy of the package with the kind left out (:func:`materials_without`): a
scratch directory holding a link to every module but that one, put first on
``PYTHONPATH`` so it shadows the checkout's. The subprocess has to run with
``-P`` (or a working directory outside the checkout), or the checkout's own copy,
which ``python -c`` puts first on ``sys.path`` as the working directory, wins.
The real layout rather than an import hook, because a registry *scan* lists a
directory, and a hook that refused the import of a module still on disk would
test a state no release is ever in.

Asked of the import system rather than of the filesystem, so the kind counts
wherever it is installed and does not count when it is only lying somewhere.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import os
import unittest
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

__all__ = [
    "HAS_ROD_PACK",
    "MATERIALS",
    "NEEDS",
    "PYTHON_FLAGS",
    "env_without",
    "has_pack",
    "imports",
    "in_parallel",
    "materials_without",
    "needs_rod_pack",
    "once",
]

MATERIALS = Path(__file__).resolve().parents[2] / "robosimstudio_materials"
"""The sibling package, as it lies in this checkout."""

NEEDS: dict[str, tuple[str, ...]] = {"paper": ("cloth",), "bag": ("cloth",), "brush": ("rod",), "tape": ("cloth",)}
"""Kind -> the kinds it is built on.

``paper``, ``bag`` and ``tape`` extend cloth's spec, entity and measures, and ``brush`` builds
its bristles with the rod kind's capsule chain and reads as a rod, so a tree that
leaves cloth or the rope out has left paper, the bag and the tape or the brush out too:
:func:`materials_without` drops a kind's dependents with it, and the stand-alone
tests read this table for which kinds are expected to go. Nothing else leans on
another kind.
"""

PYTHON_FLAGS = ("-P",)
"""Interpreter flags for a subprocess whose ``robosimstudio_materials`` is a scratch copy.

``-P`` keeps the working directory off ``sys.path``, so the copy first on
``PYTHONPATH`` is the one found; the editable install's ``.pth`` still puts the
checkout on the path for ``robosimstudio`` itself, *after* ``PYTHONPATH``.
"""


def has_pack(name: str) -> bool:
    """Whether ``robosimstudio_materials.<name>`` is installed here."""
    try:
        return importlib.util.find_spec(f"robosimstudio_materials.{name}") is not None
    except (ImportError, ValueError):
        return False


def materials_without(stack: ExitStack, *names: str) -> Path:
    """A directory holding ``robosimstudio_materials`` with the kinds ``names`` left out.

    Every other file and package of the real one is linked in, so what remains
    is exactly the release that never shipped those kinds. A kind built on one
    of ``names`` (:data:`NEEDS`) goes with it, since it could not import. Returns
    the directory to put on ``PYTHONPATH``; the package inside it is
    ``<dir>/robosimstudio_materials``.
    """
    dropped = set(names)
    for kind, needs in NEEDS.items():
        if dropped & set(needs):
            dropped.add(kind)
    site = Path(stack.enter_context(TemporaryDirectory())).resolve()
    copy = site / MATERIALS.name
    copy.mkdir()
    for child in sorted(MATERIALS.iterdir()):
        if child.name == "__pycache__" or child.stem in dropped:
            continue
        (copy / child.name).symlink_to(child, target_is_directory=child.is_dir())
    return site


def env_without(stack: ExitStack, *names: str) -> dict[str, str]:
    """The environment for a subprocess (run with :data:`PYTHON_FLAGS`) whose materials lack ``names``.

    ``os.environ`` plus ``PYTHONPATH`` with the scratch directory first, and no
    GPU: these processes ask the registries, not the simulator.
    """
    site = materials_without(stack, *names)
    inherited = os.environ.get("PYTHONPATH", "")
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(p for p in (str(site), inherited) if p),
        "CUDA_VISIBLE_DEVICES": "",
    }


def _package_of(path: Path) -> list[str]:
    """The dotted package ``path`` is a module of, as parts: every directory up that holds an ``__init__.py``."""
    parts: list[str] = []
    here = path.parent
    while (here / "__init__.py").is_file():
        parts.insert(0, here.name)
        here = here.parent
    return parts


def imports(path: Path) -> list[tuple[str, int]]:
    """Every module an ``import`` statement in ``path`` names, at any scope, with its line.

    Absolute names, always: a relative import is resolved against the package
    ``path`` lies in (read off the ``__init__.py`` files above it), so ``from
    .cloth import resolve_selector`` inside a function of ``soft.py`` reads as
    ``robosimstudio_materials.cloth`` -- the import the stand-alone law exists to
    catch, and the one a check that matched only absolute names let through.
    ``from a import b`` names ``a`` and, since ``b`` may be a module, ``a.b``
    too: ``from . import cloth`` is an import of the cloth kind.
    """
    found: list[tuple[str, int]] = []
    package = _package_of(path)
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if node.level > len(package):
                    raise ValueError(f"{path}:{node.lineno}: a relative import above the top package")
                base = ".".join([*package[: len(package) - node.level + 1], *filter(None, [node.module])])
            else:
                base = node.module or ""
            found.append((base, node.lineno))
            found.extend((f"{base}.{alias.name}", node.lineno) for alias in node.names if alias.name != "*")
    return found


def in_parallel[T, R](fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
    """``[fn(item) for item in items]``, a few at a time on threads of their own, in order.

    For the claims that are each about a fresh *process*: every one still gets
    a process of its own, but several run at once rather than one after
    another. Bounded -- a quarter of the machine's CPUs, never fewer than two
    nor more than eight -- because a tier already runs a module per worker, and
    a module that started all of its eighteen interpreters together would be
    eighteen more.
    """
    items = list(items)
    if not items:
        return []
    workers = min(len(items), max(2, (os.cpu_count() or 1) // 4), 8)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def once[R](fn: Callable[[], R]) -> Callable[[], R]:
    """``fn``, run on the first call only, its answer -- or the exception it raised -- kept for every later one.

    ``functools.cache`` keeps an answer but not a failure, so a module whose
    fresh processes timed out once would start every one of them again for
    each test that asks.
    """
    kept: list[tuple[bool, Any]] = []

    @functools.wraps(fn)
    def wrapper() -> R:
        if not kept:
            try:
                kept.append((True, fn()))
            except Exception as exc:
                kept.append((False, exc))
        ok, value = kept[0]
        if not ok:
            # A new exception each time, chained to the kept one: raising the kept
            # object again would grow its traceback with every test that asks.
            raise RuntimeError(f"{fn.__name__} failed on its first call: {value!r}") from value
        return value

    return wrapper


HAS_ROD_PACK = has_pack("rod")
"""Whether the rope -- the ``rod`` kind -- is in this checkout."""

needs_rod_pack = unittest.skipUnless(HAS_ROD_PACK, "the rod kind (robosimstudio_materials.rod) is not in this checkout")
"""Decorator for a test or a class that is about something only the rod kind provides."""
