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
    uv run python scripts/testrun.py --tier fast -j 16                  # a process per module, 16 at once
    uv run --extra gap python scripts/testrun.py --tier medium --gpus 0,1,2,3
    uv run --extra dev python scripts/testrun.py --tier fast -j 16 --coverage

Modules are handed to ``python -m unittest``, which is the repo's runner --
Newton's convention, and no plugin, no discovery magic, and no third dependency
to explain. With ``-j 1`` (the default) the whole tier is one ``unittest``
process, as CI runs it. With more, each module is its own process and as many
run at once; a module's output is printed only when it fails, and the run ends
with the slowest modules.

A GPU tier runs in parallel only with ``--gpus``: every worker gets one card of
its own (``CUDA_VISIBLE_DEVICES``), because two contact-heavy rollouts sharing a
card change each other's outcome -- a held handle reads as dropped -- and that
failure looks exactly like a regression.
"""

from __future__ import annotations

import argparse
import ast
import os
import pathlib
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time

TIERS = ("fast", "medium", "slow", "cpu")

NO_GPU_TIERS = ("fast", "cpu")
"""The tiers run with every GPU hidden, so a GPU machine runs exactly what CI does."""

NO_GPU = {"CUDA_VISIBLE_DEVICES": "", "ROBOSIMSTUDIO_DEVICE": "", "ROBOSIMSTUDIO_RENDERER": ""}
"""What hides them: the driver's variable, and the device rule's own two cleared,
so the device is picked as a machine without a GPU picks it -- not forced, and not
left to whatever a shell exported."""
ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

SOURCES = (
    "robosimstudio",
    "robosimstudio_materials",
    "robosimstudio_studio",
    "robosimstudio_bench",
    "robosimstudio_policy_agent",
    "robosimstudio_real2sim_agent",
)
"""The packages ``--coverage`` measures: the library, not the tests or the examples."""


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


def classes(module: str) -> list[str]:
    """The ``TestCase`` classes a module defines, as dotted names, read by AST.

    A class counts when it derives, by name, from something ending ``TestCase``;
    a helper base with no tests of its own is still listed and runs zero tests.
    """
    path = ROOT.joinpath(*module.split(".")).with_suffix(".py")
    tree = ast.parse(path.read_text())
    out = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and any(ast.unparse(base).endswith("TestCase") for base in node.bases):
            out.append(f"{module}.{node.name}")
    return out or [module]


class Outcome:
    """What one ``unittest`` process said about the tests it ran.

    A plain class, not a dataclass: tests load this script by file path, with no
    entry in ``sys.modules``, and a dataclass looks its module up there.
    """

    def __init__(self, name: str, code: int, seconds: float, output: str) -> None:
        self.name = name
        self.code = code
        self.seconds = seconds
        self.output = output
        self.ran = 0
        self.failures = 0
        self.errors = 0
        self.skipped = 0

    @property
    def ok(self) -> bool:
        return self.code == 0


_RAN = re.compile(r"^Ran (\d+) tests? in ", re.M)
_COUNT = re.compile(r"(failures|errors|skipped|expected failures|unexpected successes)=(\d+)")


def read_outcome(name: str, code: int, seconds: float, output: str) -> Outcome:
    """Parse the ``Ran N tests`` line and the closing ``OK``/``FAILED (...)`` line of one run."""
    out = Outcome(name=name, code=code, seconds=seconds, output=output)
    ran = list(_RAN.finditer(output))
    if not ran:
        return out
    out.ran = int(ran[-1].group(1))
    for key, value in _COUNT.findall(output[ran[-1].start() :]):
        if key in ("failures", "errors", "skipped"):
            setattr(out, key, int(value))
    return out


def run_parallel(
    names: list[str],
    jobs: int,
    env: dict[str, str] | None,
    extra: list[str],
    gpus: list[str] | None = None,
    coverage: pathlib.Path | None = None,
    stream=sys.stdout,
) -> list[Outcome]:
    """Run each name in its own ``python -m unittest`` process, ``jobs`` at a time.

    ``gpus`` pins worker *i* to card ``gpus[i]`` for every name it takes, so the
    number of workers is the number of cards. A name is taken from a shared
    queue as soon as a worker is free, so one slow module does not hold a fixed
    shard of the others behind it.
    """
    work: queue.Queue[str] = queue.Queue()
    for name in names:
        work.put(name)
    outcomes: list[Outcome] = []
    lock = threading.Lock()
    workers = len(gpus) if gpus else max(1, jobs)

    def worker(slot: int) -> None:
        own = dict(env if env is not None else os.environ)
        if gpus:
            own["CUDA_VISIBLE_DEVICES"] = gpus[slot]
        if coverage is not None:
            own["COVERAGE_FILE"] = str(coverage)
        while True:
            try:
                name = work.get_nowait()
            except queue.Empty:
                return
            prefix = [sys.executable, "-m"]
            if coverage is not None:
                prefix += ["coverage", "run", "-p", f"--source={','.join(SOURCES)}", "-m"]
            start = time.monotonic()
            done = subprocess.run(
                [*prefix, "unittest", name, *extra],
                cwd=ROOT,
                env=own,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            outcome = read_outcome(name, done.returncode, time.monotonic() - start, done.stdout)
            with lock:
                outcomes.append(outcome)
                where = f" [gpu {gpus[slot]}]" if gpus else ""
                verdict = "ok  " if outcome.ok else "FAIL"
                print(
                    f"{verdict} {name}  ({outcome.ran} tests, {outcome.seconds:.1f} s){where}",
                    file=stream,
                    flush=True,
                )

    threads = [threading.Thread(target=worker, args=(slot,), daemon=True) for slot in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sorted(outcomes, key=lambda o: o.name)


def report(outcomes: list[Outcome], wall: float, durations: int, stream=sys.stdout) -> int:
    """Print every failing run's output, the slowest runs and the totals; return the exit status."""
    failed = [o for o in outcomes if not o.ok]
    for outcome in failed:
        print(f"\n{'=' * 70}\n{outcome.name} exited {outcome.code}\n{'=' * 70}", file=stream)
        print(outcome.output.rstrip(), file=stream)
    if durations:
        print(f"\nslowest {min(durations, len(outcomes))}:", file=stream)
        for outcome in sorted(outcomes, key=lambda o: -o.seconds)[:durations]:
            print(f"  {outcome.seconds:7.1f} s  {outcome.name}", file=stream)
    totals = {key: sum(getattr(o, key) for o in outcomes) for key in ("ran", "failures", "errors", "skipped")}
    print(
        f"\nRan {totals['ran']} tests in {len(outcomes)} processes, {wall:.1f} s wall: "
        f"{totals['failures']} failures, {totals['errors']} errors, {totals['skipped']} skipped, "
        f"{len(failed)} process(es) failed",
        file=stream,
    )
    if failed:
        print("FAILED: " + " ".join(o.name for o in failed), file=stream)
    return 1 if failed else 0


def coverage_report(data: pathlib.Path, stream=sys.stdout) -> int:
    """Combine the per-process data files beside ``data`` and print the report."""
    env = {**os.environ, "COVERAGE_FILE": str(data)}
    combine = [sys.executable, "-m", "coverage", "combine", "-q", str(data.parent)]
    if subprocess.run(combine, cwd=ROOT, env=env, check=False).returncode:
        return 1
    print("", file=stream, flush=True)
    show = [sys.executable, "-m", "coverage", "report", "--skip-covered", "--sort=miss"]
    code = subprocess.run(show, cwd=ROOT, env=env, check=False).returncode
    print(f"\ncoverage data: {data}  (open it with: COVERAGE_FILE={data} python -m coverage html)", file=stream)
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", choices=TIERS, help="which tier to run")
    parser.add_argument("--list", action="store_true", help="print the modules per tier (or of --tier) and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="pass -v through to unittest")
    parser.add_argument(
        "-j", "--jobs", type=int, default=1, help="processes at once, one module each; 0 is one per CPU (default 1)"
    )
    parser.add_argument("--gpus", default="", help="comma-separated cards for a GPU tier: one worker per card")
    parser.add_argument("--by-class", action="store_true", help="with -j or --gpus: a process per TestCase class")
    parser.add_argument("--durations", type=int, default=10, help="with -j or --gpus: list this many slowest runs")
    parser.add_argument("--coverage", action="store_true", help="measure line coverage (needs the dev extra)")
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
        for tier in [args.tier] if args.tier else TIERS:
            names = modules(tier)
            print(f"{tier:<7} {len(names):>3} module(s)")
            for name in names:
                print(f"          {name}")
        return 0

    names = modules(args.tier)
    if not names:
        print(f"no modules declare TIER = {args.tier!r}")
        return 0

    gpus = [card.strip() for card in args.gpus.split(",") if card.strip()]
    if gpus and args.tier in NO_GPU_TIERS:
        print(f"error: --gpus is for a GPU tier; {args.tier!r} runs with every GPU hidden", file=sys.stderr)
        return 2
    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    if jobs > 1 and args.tier not in NO_GPU_TIERS and not gpus:
        print(
            f"error: {args.tier!r} in parallel needs --gpus, one card per worker: runs sharing a card "
            "change each other's contact outcomes",
            file=sys.stderr,
        )
        return 2
    if args.coverage:
        try:
            import coverage  # noqa: F401
        except ImportError:
            print("error: --coverage needs coverage: uv run --extra dev ...", file=sys.stderr)
            return 2

    env = {**os.environ, **NO_GPU} if args.tier in NO_GPU_TIERS else None
    extra = [*(["-v"] if args.verbose else []), *args.rest]
    data = pathlib.Path(tempfile.mkdtemp(prefix="rss-coverage-")) / ".coverage" if args.coverage else None

    if jobs == 1 and not gpus:
        prefix = [sys.executable, "-m"]
        if data is not None:
            prefix += ["coverage", "run", f"--source={','.join(SOURCES)}", "-m"]
            env = {**(env if env is not None else os.environ), "COVERAGE_FILE": str(data) + ".serial"}
        command = [*prefix, "unittest", *names, *extra]
        print(f"$ {' '.join(command[:4])} ... ({len(names)} modules)")
        code = subprocess.run(command, cwd=ROOT, check=False, env=env).returncode
        return (coverage_report(data) or code) if data is not None else code

    if args.by_class:
        names = [name for module in names for name in classes(module)]
    workers = len(gpus) if gpus else jobs
    print(f"{len(names)} runs of tier {args.tier!r} on {workers} worker(s)" + (f", cards {gpus}" if gpus else ""))
    start = time.monotonic()
    outcomes = run_parallel(names, jobs, env, extra, gpus=gpus or None, coverage=data)
    code = report(outcomes, time.monotonic() - start, args.durations)
    return (coverage_report(data) or code) if data is not None else code


if __name__ == "__main__":
    raise SystemExit(main())
