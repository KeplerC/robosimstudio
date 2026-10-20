# SPDX-License-Identifier: Apache-2.0
"""Film the clips the example READMEs show: each example's own page, driven by a visible cursor.

Each take (``scripts/recording/takes.py``) runs its example's script unchanged
and puts a person on the page it opens: the cursor glides to the arm's gizmo,
drags an arrow, presses **Close**, drags it up. The page is the studio's own --
3D view and sidebar -- drawn on the GPU by a headless Chromium, stepped one film
frame at a time so the clip is the scene's own time whatever the physics costs,
and written as a looping 960x540 WebP.

A take whose expectations hold is written into its folder's ``media/``; one
that fails, or is marked pending, is written only to ``outputs/record/<take>/``
with its contact sheet and timeline, for a look. A slipped grasp never ships.

Needs the ``media`` group and its browser, once::

    uv sync --group media && uv run --group media playwright install chromium

Usage, from the repository root::

    uv run --group media python scripts/record_examples.py --list
    CUDA_VISIBLE_DEVICES=1 uv run --group media python scripts/record_examples.py 00_hello_world
    CUDA_VISIBLE_DEVICES=1 uv run --group media python scripts/record_examples.py drag_cube --probe
    uv run --group media python scripts/record_examples.py --gpus 1,2 -j 4      # every take
    uv run --group media python scripts/record_examples.py --gpus 5 -j 4 --port 9104 tape   # beside it

A name is matched as a substring of ``<folder>/<clip>``. ``--probe`` opens the
page, frames it and writes ``probe.png`` without playing a beat: for placing a
camera. One process per take; ``CUDA_VISIBLE_DEVICES`` (or ``--gpus``) picks the
card for the simulation and the browser both. A recorder serves its pages on
``--port`` to ``--port + j - 1``; recorders side by side take blocks that do not
meet, and one whose block meets another's is refused, naming the port.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import runpy
import shutil
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from recording.takes import KEPT, TAKES, Take, find  # noqa: E402

CHILD_TIMEOUT_S = 1800
"""A take that has not finished in half an hour is stuck, not slow."""


PORTS = range(9100, 9200)
"""Where the pages serve. A recorder serves on a block of these, ``--port`` to ``--port + j - 1``,
a port for each take running at once, and hands each child the lowest one no running child holds:
a child that looked for a free one would look long before viser binds it, and two started together
would both find the same. The block is locked for as long as the recorder runs, so a second
recorder whose block meets it is refused, naming the port."""

LOCKS = Path(os.environ.get("ROBOSIMSTUDIO_RECORD_LOCKS") or "/tmp/robosimstudio-record-ports")
"""Where every recorder on the machine locks its block, a file a port: one path whatever ``TMPDIR`` says,
because the ports are the machine's. Under ``tempfile.gettempdir()`` two recorders started with
different ``TMPDIR`` values locked in two directories, each blind to the other's locks. Opened to every
user, as ``/tmp`` is, so another user's recorder locks beside this one's. ``ROBOSIMSTUDIO_RECORD_LOCKS``
names another, for a test: recorders meet only where they name the same one."""


def page_ports(port: int, jobs: int) -> range:
    """The block a recorder started at ``--port`` with ``jobs`` takes at once serves its pages on."""
    last = port + jobs - 1
    if port not in PORTS:
        raise ValueError(f"--port {port} is outside {PORTS.start}-{PORTS.stop - 1}, where the pages serve")
    if last not in PORTS:
        raise ValueError(
            f"--port {port} with -j {jobs} serves on {port}-{last}, past {PORTS.stop - 1}: "
            f"start it at {PORTS.stop - jobs} or lower"
        )
    return range(port, last + 1)


def port_free(port: int) -> bool:
    """Nothing listens on ``port``. Bound as viser binds, over the connections a closed page left waiting."""
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


def _open_shared(path: Path) -> int:
    """``path`` opened for reading and appending, made 0o666 if it is not there: a lock file, whoever made it.

    Opened without ``O_CREAT`` when it is there. :data:`LOCKS` is sticky and writable by all, and there
    ``fs.protected_regular`` refuses an ``O_CREAT`` open of another user's file, 0o666 or not, unless the
    directory is the opener's: one ``O_CREAT`` open locked a user out of every port another user's
    recorder had once locked, since a lock file is never removed. Made ``O_EXCL``, so a recorder that
    makes it an instant after another opens the other's. This user's own, made under an older recorder
    or another umask, is opened to every user here."""
    for _ in range(3):
        try:
            fd = os.open(path, os.O_RDWR | os.O_APPEND)
        except FileNotFoundError:
            try:
                fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL, 0o666)
            except FileExistsError:  # another recorder made it between the two
                continue
            os.fchmod(fd, 0o666)  # whatever the umask took off
            return fd
        st = os.fstat(fd)
        if st.st_uid == os.geteuid() and stat.S_IMODE(st.st_mode) & 0o666 != 0o666:
            os.fchmod(fd, 0o666)
        return fd
    raise FileNotFoundError(f"{path} was made and removed three times while this recorder opened it")


def _open_lock(port: int):
    """``port``'s lock file, open; a lock this user may not open is refused by name, its owner said."""
    path = LOCKS / f"{port}.lock"
    try:
        LOCKS.mkdir(parents=True, exist_ok=True)
        st = LOCKS.stat()
        if st.st_uid == os.geteuid() and stat.S_IMODE(st.st_mode) & 0o1777 != 0o1777:
            LOCKS.chmod(0o1777)  # this user's, whichever recorder made it: opened to every user, as /tmp is
        fd = _open_shared(path)
        return os.fdopen(fd, "a+")  # held until the recorder exits
    except PermissionError as exc:
        import pwd

        there = next((p for p in (path, LOCKS) if p.exists()), LOCKS)
        try:
            owner = f"user {pwd.getpwuid(there.stat().st_uid).pw_name}"
        except (KeyError, OSError):
            owner = "another user"
        raise ValueError(
            f"port {port}'s lock {path} cannot be opened ({exc.strerror}); {there} is owned by {owner}: "
            f"run as them, or remove {LOCKS} once no recorder of theirs is running"
        ) from None


def claim(block: range) -> list:
    """Lock ``block``'s ports for this recorder until it exits: the open lock files, to keep.

    A port another recorder has locked, that something else listens on, or whose lock this user may
    not open, is refused by name, and nothing of the block stays locked."""
    held = []
    try:
        for port in block:
            lock = _open_lock(port)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                lock.seek(0)
                holder = lock.read().strip() or "another recorder"
                lock.close()
                raise ValueError(
                    f"port {port} is a page port of {holder}: recorders side by side take blocks "
                    "(--port to --port + j - 1) that do not meet"
                ) from None
            held.append(lock)
            lock.truncate(0)
            lock.write(f"the recorder of pid {os.getpid()}, ports {block.start}-{block[-1]}")
            lock.flush()
            if not port_free(port):
                raise ValueError(f"port {port} is in use by something that is not a recorder: pick another --port")
    except BaseException:
        for lock in held:
            lock.close()
        raise
    return held


# =============================================================================
# One take, in this process
# =============================================================================


def film_one(
    take: Take, out_root: Path, *, port: int, software: bool, probe: bool, install: bool, kmax: int, frames: bool
) -> int:
    """Run the take's example under the hooks and write what it filmed. 0 when it may ship."""
    from recording import encode, lockstep

    out = out_root / take.key
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    def log(message: str) -> None:
        print(f"[{take.key} {time.perf_counter() - t0:6.1f}s] {message}", flush=True)

    if not port_free(port):
        log(f"port {port} is in use: a take serves only on the port its recorder hands it")
        return 1
    film = lockstep.Film(take, port=port, scratch=out_root, software=software, probe=probe, log=log)
    lockstep.install(film, port=port, out=str(out / "studio"), overrides=dict(take.config))
    script = REPO / take.example
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *take.argv]
    log(f"{take.example} {' '.join(take.argv)}".rstrip())
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            log(f"the example exited {exc.code}")
    if film.session is None:
        log("the example opened no page")
        return 1
    if probe:
        if film.screenshot:
            (out / "probe.png").write_bytes(film.screenshot)
        record = {"take": take.key, "verdict": "probe", "renderer": film.renderer, "camera": film.camera}
        encode.timeline(out / "probe.json", record | {"error": film.error, "probe": str(out / "probe.png")})
        log(f"camera {film.camera}; wrote {out / 'probe.png'}")
        return 0 if not film.error else 1

    verdicts = film.judge()
    for v in verdicts:
        log(f"{'PASS' if v['passed'] else 'FAIL'} {v['expect']}: {v['reading']}")
    record = {
        "take": take.key,
        "example": take.example,
        "argv": list(take.argv),
        "pending": take.pending,
        "renderer": film.renderer,
        "camera": film.camera,
        "fps": take.fps,
        "frames": len(film.frames),
        "sim_s": round(film.times[-1], 3) if film.times else 0.0,
        "wait_frame_ms": _percentiles(film.latency),
        "unsynced_drag_frames": film.unsynced,
        "beats": film.spans,
        "marks": film.marks,
        "expect": verdicts,
        "routine_ended": film.ended,
        "error": film.error,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    if frames:
        (out / "frames").mkdir(exist_ok=True)
        for i, image in enumerate(film.frames):
            image.save(out / "frames" / f"{i:04d}.png")
    if film.frames:
        record["webp"] = encode.write_webp(film.frames, out / "clip.webp", take.fps, budget=take.budget, kmax=kmax)
        encode.sheet(film.frames, film.times, film.captions, out / "sheet.png")
    ok = bool(film.frames) and not film.error and all(v["passed"] for v in verdicts) and record["webp"]["fits"]
    shipped = ok and install and not take.pending
    if shipped:
        take.media.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / "clip.webp", take.media)
    record["verdict"] = "shipped" if shipped else ("passed" if ok else "failed")
    encode.timeline(out / "timeline.json", record)
    webp = record.get("webp", {})
    log(
        f"{record['verdict']}: {len(film.frames)} frames, {webp.get('bytes', 0) / 1000:.0f} KB at q{webp.get('quality')}, "
        f"waitFrame p95 {record['wait_frame_ms'].get('p95')} ms -> {out / 'sheet.png'}"
        + (f" and {take.media.relative_to(REPO)}" if shipped else "")
    )
    return 0 if ok else 1


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def at(q: float) -> float:
        return round(ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))], 1)

    return {"p50": at(0.5), "p95": at(0.95), "max": round(ordered[-1], 1)}


# =============================================================================
# Many takes, a process each
# =============================================================================


def gpus(spec: str) -> list[str]:
    if spec != "auto":
        return [g.strip() for g in spec.split(",") if g.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return [g.strip() for g in visible.split(",") if g.strip()] or ["0"]


def run_all(chosen: list[Take], args) -> int:
    """A child process per take, at most ``-j`` at once, dealt across the GPUs and the block's ports."""
    cards = gpus(args.gpus)
    jobs = max(1, args.jobs or len(cards))
    try:
        block = page_ports(args.port, jobs)
        locks = claim(block)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    try:
        return _deal(chosen, args, cards, jobs, block)
    finally:
        for lock in locks:
            lock.close()


def _deal(chosen: list[Take], args, cards: list[str], jobs: int, block: range) -> int:
    queue = list(chosen)
    running: list[tuple[subprocess.Popen, Take, float, str, int]] = []
    failed: list[str] = []
    dealt = 0
    while queue or running:
        while queue and len(running) < jobs:
            take = queue.pop(0)
            card = cards[dealt % len(cards)]
            dealt += 1
            out = args.out / take.key
            out.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--child", take.key, "--out", str(args.out)]
            port = min(set(block) - {item[4] for item in running})
            cmd += ["--port", str(port)]
            cmd += ["--kmax", str(args.kmax)]
            flags = {
                "--software": args.software,
                "--probe": args.probe,
                "--no-install": not args.install,
                "--frames": args.frames,
            }
            cmd += [flag for flag, on in flags.items() if on]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": card}
            with open(out / "log.txt", "w") as sink:
                proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=sink, stderr=subprocess.STDOUT)
            running.append((proc, take, time.perf_counter(), card, port))
            print(f"{take.key}: started on GPU {card}, port {port} (log {out / 'log.txt'})", flush=True)
        time.sleep(1.0)
        for item in list(running):
            proc, take, started, card, _port = item
            if proc.poll() is None and time.perf_counter() - started < CHILD_TIMEOUT_S:
                continue
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            running.remove(item)
            print(_summary(take, args.out, proc.returncode, time.perf_counter() - started, args.probe), flush=True)
            if proc.returncode != 0:
                failed.append(take.key)
    if failed:
        print(f"{len(failed)} of {len(chosen)} failed: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


def _summary(take: Take, out_root: Path, code: int, wall: float, probe: bool) -> str:
    path = out_root / take.key / ("probe.json" if probe else "timeline.json")
    if not path.is_file():
        tail = (out_root / take.key / "log.txt").read_text(errors="replace").splitlines()[-15:]
        return f"{take.key}: exit {code} after {wall:.0f} s, no timeline\n    " + "\n    ".join(tail)
    record = json.loads(path.read_text())
    if record.get("verdict") == "probe":
        return f"{take.key}: probe ({wall:.0f} s) -> {record['probe']}" + (
            f"; {record['error']}" if record["error"] else ""
        )
    webp = record.get("webp", {})
    fails = [v["reading"] for v in record.get("expect", []) if not v["passed"]]
    return (
        f"{take.key}: {record.get('verdict')} ({wall:.0f} s): {record.get('frames')} frames, "
        f"{webp.get('bytes', 0) / 1000:.0f} KB q{webp.get('quality')}"
        + (f"; {record['error']}" if record.get("error") else "")
        + (f"; FAIL {'; '.join(fails)}" if fails else "")
    )


def list_takes() -> None:
    for take in TAKES:
        state = f"pending: {take.pending}" if take.pending else "final"
        extras = f" [needs {', '.join(take.needs)}]" if take.needs else ""
        argv = f" {' '.join(take.argv)}" if take.argv else ""
        print(f"{take.key:34s} {take.example}{argv}{extras}  ({state})")
    for key, reason in KEPT.items():
        print(f"{key:34s} kept: {reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", help="substrings of <folder>/<clip>; every take when none")
    parser.add_argument("--list", action="store_true", help="the takes and the kept clips, and stop")
    parser.add_argument("--probe", action="store_true", help="frame each page and write probe.png; play no beat")
    parser.add_argument(
        "--out", type=Path, default=REPO / "outputs" / "record", help="where each take's clip, sheet and timeline go"
    )
    parser.add_argument(
        "--gpus", default="auto", help="cards to deal the takes across: auto (CUDA_VISIBLE_DEVICES) or 1,2"
    )
    parser.add_argument("-j", "--jobs", type=int, default=0, help="takes at once (default: one per card)")
    parser.add_argument("--final", action="store_true", help="only the takes that are not pending")
    parser.add_argument("--software", action="store_true", help="record even when the page is not drawn on the GPU")
    parser.add_argument("--no-install", dest="install", action="store_false", help="never write into examples/")
    parser.add_argument("--kmax", type=int, default=9, help="most frames between WebP key frames")
    parser.add_argument("--frames", action="store_true", help="also keep every frame as a PNG under the take's frames/")
    parser.add_argument("--child", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--port",
        type=int,
        default=PORTS.start,
        help=f"the first of the -j ports the pages serve on ({PORTS.start}-{PORTS.stop - 1}); recorders side by "
        "side take blocks that do not meet, and one that meets another's is refused",
    )
    args = parser.parse_args(argv)
    args.out = args.out.resolve()

    if args.child:
        return film_one(
            find(args.child),
            args.out,
            port=args.port,
            software=args.software,
            probe=args.probe,
            install=args.install,
            kmax=args.kmax,
            frames=args.frames,
        )
    if args.list:
        list_takes()
        return 0
    chosen = [t for t in TAKES if not args.names or any(n in t.key for n in args.names)]
    if args.final:
        chosen = [t for t in chosen if not t.pending]
    if not chosen:
        print(f"no take matches {args.names}: see --list", file=sys.stderr)
        return 2
    return run_all(chosen, args)


if __name__ == "__main__":
    sys.exit(main())
