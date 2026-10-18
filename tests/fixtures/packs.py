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

import importlib.util
import os
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

__all__ = [
    "HAS_ROD_PACK",
    "MATERIALS",
    "NEEDS",
    "PYTHON_FLAGS",
    "env_without",
    "has_pack",
    "materials_without",
    "needs_rod_pack",
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


HAS_ROD_PACK = has_pack("rod")
"""Whether the rope -- the ``rod`` kind -- is in this checkout."""

needs_rod_pack = unittest.skipUnless(HAS_ROD_PACK, "the rod kind (robosimstudio_materials.rod) is not in this checkout")
"""Decorator for a test or a class that is about something only the rod kind provides."""
