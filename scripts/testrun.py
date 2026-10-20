# SPDX-License-Identifier: Apache-2.0
"""Run one tier of the test suite.

The suite splits by what a test *needs*, not by what it is about:

``fast``
    No GPU. Runs in seconds and gates every commit. Schema and typecheck claims,
    handler contracts, atom arithmetic on synthetic geometry, goal-tree semantics
    on scripted snapshots, structural rules about the tree itself. Run with every
    GPU hidden, as CI's runners have none: a test that composes a model puts it
    on whatever device Warp finds, and on a GPU machine that would quietly be
    the GPU.
``medium``
    A GPU, and a scene or two. Build, reset, settle, a handful of steps. Run by
    hand on a GPU machine; no CI workflow runs it.
``slow``
    A GPU and minutes. Full rollouts, routine baselines, render backends, capture
    cycles. Run by hand on a GPU machine, like ``medium``.
``cpu``
    No GPU, and real scenes: built and stepped on Warp's CPU device with MuJoCo's
    C library and SuperDex, the way ``robosimstudio[cpu]`` runs them. Needs the
    ``[cpu]`` extra; about two minutes warm and two and a half the first time,
    while Warp compiles its kernels for the CPU, and two fifths of it the cutting
    kind: its walks and the two cutting examples' builds. Run with
    every GPU hidden, like ``fast``, so a GPU machine proves what a machine
    without one does; CI runs it.

The tier is declared by each module as a ``TIER`` constant and read **by AST**,
never by importing -- so listing the slow tier costs nothing and does not
initialise CUDA. A ``test_*.py`` module that declares no ``TIER`` would run in no
tier at all, so this script refuses to list or run anything while one exists,
and exits non-zero naming it. That matters more than it sounds: a selector that imports what
it is selecting is a selector nobody can run on a laptop.

    uv run python scripts/testrun.py --list
    uv run python scripts/testrun.py --tier fast
    uv run python scripts/testrun.py --tier slow -v
    uv run --extra cpu --no-group rtx python scripts/testrun.py --tier cpu

Modules are handed to ``python -m unittest``, which is the repo's runner --
Newton's convention, and no plugin, no discovery magic, and no third dependency
to explain.
"""

from __future__ import annotations

import argparse
import ast
import os
import pathlib
import subprocess
import sys

TIERS = ("fast", "medium", "slow", "cpu")

NO_GPU_TIERS = ("fast", "cpu")
"""The tiers run with every GPU hidden, so a GPU machine runs exactly what CI does."""

NO_GPU = {"CUDA_VISIBLE_DEVICES": "", "ROBOSIMSTUDIO_DEVICE": "", "ROBOSIMSTUDIO_RENDERER": ""}
"""What hides them: the driver's variable, and the device rule's own two cleared,
so the device is picked as a machine without a GPU picks it -- not forced, and not
left to whatever a shell exported."""
ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"


def declared_tier(path: pathlib.Path) -> str:
    """The module's ``TIER``, read without importing it.

    Returns ``""`` for a module that declares none, which :func:`main` turns
    into a failure -- an undeclared tier means a test silently runs nowhere,
    which is worse than one that runs in the wrong place.
    """
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return ""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "TIER" and isinstance(node.value, ast.Constant):
                return str(node.value.value)
    return ""


def test_files() -> list[pathlib.Path]:
    """Every ``test_*.py`` the runner selects from, sorted."""
    found = list(TESTS.rglob("test_*.py"))
    # A collection (a top-level directory with a ``tasks/`` of its own -- see
    # robosimstudio.roots) carries its tests beside its suites; they are selected
    # by the same TIER rule and run under the same runner.
    for child in sorted(ROOT.iterdir()):
        if child.name[:1] in ("_", ".") or not child.is_dir() or child.name in ("tests", "outputs"):
            continue
        if (child / "tasks").is_dir() and (child / "tests").is_dir():
            found.extend((child / "tests").rglob("test_*.py"))
    return sorted(found)


def modules(tier: str = "") -> list[str]:
    """Dotted module names for one tier, or for every declared module.

    Sorted, so a run is reproducible and a failure is findable in the output of
    the run before it.
    """
    out: list[str] = []
    for path in test_files():
        got = declared_tier(path)
        if not got or (tier and got != tier):
            continue
        out.append(".".join(path.relative_to(ROOT).with_suffix("").parts))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", choices=TIERS, help="which tier to run")
    parser.add_argument("--list", action="store_true", help="print the modules per tier and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="pass -v through to unittest")
    parser.add_argument("rest", nargs="*", help="extra arguments for unittest")
    args = parser.parse_args(argv)

    undeclared = [str(p.relative_to(ROOT)) for p in test_files() if declared_tier(p) not in TIERS]
    if undeclared:
        print(
            f"error: no TIER = {' | '.join(map(repr, TIERS))} in {undeclared}: these would run in no tier",
            file=sys.stderr,
        )
        return 1

    if args.list or not args.tier:
        for tier in TIERS:
            names = modules(tier)
            print(f"{tier:<7} {len(names):>3} module(s)")
            for name in names:
                print(f"          {name}")
        return 0

    names = modules(args.tier)
    if not names:
        print(f"no modules declare TIER = {args.tier!r}")
        return 0

    command = [sys.executable, "-m", "unittest", *names, *(["-v"] if args.verbose else []), *args.rest]
    print(f"$ {' '.join(command[:4])} ... ({len(names)} modules)")
    env = {**os.environ, **NO_GPU} if args.tier in NO_GPU_TIERS else None
    return subprocess.run(command, cwd=ROOT, check=False, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
