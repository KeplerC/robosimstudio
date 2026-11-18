#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write the gallery's pages from its manifest, and put its media on the Hugging Face Hub.

``gallery/index.toml`` lists every gallery entry: its project, title, status,
robot, clips, source, licence and note. The clips live on the Hub, in the
dataset :data:`robosimstudio.hub.DATASET`, under ``gallery/<project>/<entry>/media/``;
the pages link to them at the pinned :data:`robosimstudio.hub.REVISION`.

Usage, from the repository root::

    uv run python scripts/gallery.py                    # rewrite gallery/README.md and each project's page
    uv run python scripts/gallery.py --check            # exit 1 if a page is stale or the manifest is wrong
    uv run python -m robosimstudio.hub fetch gallery/rgbs/strip-polishing   # one entry's files, here

Every clip is kept twice on the Hub: the mp4, and a short looping WebP of it that
GitHub draws inline, so a page shows every clip playing and each links to its mp4.

Maintainers publish new clips in two steps. ``stage`` encodes them from a local
sources file (outside the repository: it names paths on one machine) into a
folder laid out like the dataset, with a WebP per clip, a poster per entry and
the dataset card; ``upload`` sends that folder, moves ``REVISION`` to the new
commit and rewrites the pages (``--prune`` also deletes the clips, previews and
posters the manifest no longer names; a ported twin's assets are never pruned)::

    uv run python scripts/gallery.py stage SOURCES.toml OUT/
    uv run --with huggingface_hub python scripts/gallery.py upload OUT/ [--prune]

A sources file maps ``"<project>/<entry>"`` to ``{ "<clip>.mp4" = "/path/to/source" }``,
optionally with ``poster = "/path/to/image"`` (otherwise a frame of the first clip).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from robosimstudio import hub  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from recording.encode import write_webp  # noqa: E402

STATUSES = {
    "verified": "runs on current main and passes the harness checks; the commit it was checked on is given",
    "partial": "runs on current main, but the task is not completed yet or a check fails; the note says which",
    "video": "a video from older code, not yet ported to current main",
    "scene": "the scene is rebuilt, but no robot task in it is filmed yet",
}
ID = re.compile(r"[a-z0-9][a-z0-9-]*")
MEDIA = re.compile(r"[a-z0-9][a-z0-9_-]*\.(mp4|jpg|png)")
SHA = re.compile(r"[0-9a-f]{7,40}")
FORBIDDEN_STAGE = re.compile(r"(^|/)(auth\.json|codexhome|logs|review|\.env|gallery\.json)(/|$)|\.jsonl$")
MAX_STAGED = 200_000_000
PREVIEW_WIDTH = 480
"""A preview WebP's width in pixels: three to a row on a GitHub page."""
PREVIEW_FPS = 10
PREVIEW_MAX_S = 12.0
"""A longer clip is sped up to play in this long, at most 8 times: a preview shows the whole chore."""
PREVIEW_BUDGET = 2_000_000
"""Bytes per preview: GitHub's image proxy serves it, and a page of forty must still load."""
HEADER = "<!-- written by scripts/gallery.py from gallery/index.toml; edit the manifest, not this page -->\n"


def load(root: Path = REPO) -> tuple[list[dict], list[dict]]:
    """The manifest's projects and entries, in file order."""
    data = tomllib.loads((root / "gallery" / "index.toml").read_text(encoding="utf-8"))
    return data.get("project", []), data.get("entry", [])


def problems(root: Path = REPO) -> list[str]:
    """Everything wrong with the manifest and the gallery folder, one line each."""
    projects, entries = load(root)
    out = []
    known = set()
    for project in projects:
        pid = project.get("id", "")
        if not ID.fullmatch(pid) or pid in known:
            out.append(f"project {pid!r}: needs a unique lowercase id")
        for key in ("title", "about"):
            if not project.get(key):
                out.append(f"project {pid}: no {key}")
        known.add(pid)
        if not any(e.get("project") == pid for e in entries):
            out.append(f"project {pid}: has no entries")
    seen = set()
    for entry in entries:
        name = f"{entry.get('project')}/{entry.get('id')}"
        if entry.get("project") not in known:
            out.append(f"{name}: project is not declared")
        if not ID.fullmatch(str(entry.get("id", ""))) or name in seen:
            out.append(f"{name}: needs a unique lowercase id within its project")
        seen.add(name)
        for key in ("title", "robot", "source", "licence", "note"):
            if not entry.get(key):
                out.append(f"{name}: no {key}")
        status = entry.get("status")
        if status not in STATUSES:
            out.append(f"{name}: status {status!r} is not one of {', '.join(STATUSES)}")
        media = entry.get("media", [])
        if (
            not media
            or any(not MEDIA.fullmatch(str(m)) for m in media)
            or "poster.jpg" in media
            or len(set(media)) != len(media)
        ):
            out.append(f"{name}: media must be distinct lowercase .mp4/.jpg/.png names, never poster.jpg")
        code = entry.get("code")
        if code and not Path(code).is_relative_to(Path("gallery") / str(entry.get("project", ""))):
            out.append(f"{name}: its code must live under gallery/{entry.get('project')}/")
        elif status in ("verified", "partial") and (not code or not (root / code).exists()):
            out.append(f"{name}: a {status} entry needs its code in the checkout (code = ...)")
        if status == "verified" and not SHA.fullmatch(str(entry.get("verified_on", ""))):
            out.append(f"{name}: a verified entry needs verified_on = the commit it was checked on")
    if (root / "gallery" / "tasks").exists():
        out.append("gallery/tasks/ must not exist: a top-level <dir>/tasks/ is loaded as a task collection")
    return out


def _asset(entry: dict, name: str, revision: str | None = None) -> str:
    """A clip's link on the Hub itself, never a mirror: these links are written into committed pages."""
    return hub.url(f"gallery/{entry['project']}/{entry['id']}/media/{name}", revision, endpoint=hub.PUBLIC)


def preview(name: str) -> str:
    """What a page shows for one clip: its WebP; a still is shown as itself."""
    return f"{Path(name).stem}.webp" if name.endswith(".mp4") else name


def _status(entry: dict) -> str:
    if entry["status"] == "verified":
        return f"verified on main ({entry['verified_on'][:7]})"
    return {"partial": "partial", "video": "video, older code", "scene": "scene only"}[entry["status"]]


def _links(entry: dict) -> str:
    links = entry.get("links", [])
    return " ".join(f"[{i + 1}]({link})" for i, link in enumerate(links))


def project_page(project: dict, entries: list[dict]) -> str:
    """One project's page: every clip playing, three to a row, each linking to its mp4; then the details."""
    lines = [f"# {project['title']}\n", HEADER, f"{project['about'].strip()}\n", "[All projects](../README.md)\n"]
    cells = []
    for entry in entries:
        for name in entry["media"]:
            label = f" ({Path(name).stem})" if len(entry["media"]) > 1 else ""
            cells.append(
                f'<td width="33%" valign="top"><a href="{_asset(entry, name)}">'
                f'<img src="{_asset(entry, preview(name))}" width="100%" alt="{entry["title"]}{label}"></a><br>'
                f"<b>{entry['title']}</b>{label}<br><sub>{_status(entry)} · {entry['robot']}</sub></td>"
            )
    lines.append("<table>")
    for row in range(0, len(cells), 3):
        lines += ["<tr>", *cells[row : row + 3], "</tr>"]
    lines.append("</table>\n")
    lines.append("| Entry | Status | Clips | Source | Licence | Note |")
    lines.append("|---|---|---|---|---|---|")
    for entry in entries:
        clips = " · ".join(f"[{Path(m).stem}]({_asset(entry, m)})" for m in entry["media"])
        code = (
            f" ([code]({Path(entry['code']).relative_to('gallery/' + entry['project'])}))" if entry.get("code") else ""
        )
        source = f"{entry['source']} {_links(entry)}".strip()
        lines.append(
            f"| {entry['title']}{code} | {_status(entry)} | {clips} | {source} | {entry['licence']} | {entry['note']} |"
        )
    lines.append("")
    return "\n".join(lines)


def index_page(projects: list[dict], entries: list[dict]) -> str:
    """``gallery/README.md``: what the gallery is, what a status means, and one row per project."""
    lines = [
        "# Gallery\n",
        HEADER,
        "Tasks built with RoboSimStudio, by project, in simulation. Every clip is hosted on Hugging Face, in the",
        f"dataset [{hub.DATASET}](https://huggingface.co/datasets/{hub.DATASET}); each project's page plays them all,",
        "and each links to its full-length video.\n",
        "| Status | Meaning |",
        "|---|---|",
        *(f"| {k} | {v} |" for k, v in STATUSES.items()),
        "",
        "| Project | Entries | Verified | |",
        "|---|---|---|---|",
    ]
    for project in projects:
        own = [e for e in entries if e["project"] == project["id"]]
        verified = sum(e["status"] == "verified" for e in own)
        cover = f'<img src="{_asset(own[0], preview(own[0]["media"][0]))}" width="240" alt="">' if own else ""
        lines.append(f"| [{project['title']}]({project['id']}/README.md) | {len(own)} | {verified} | {cover} |")
    lines += [
        "",
        "An entry's files (its clips, and for a ported scene its meshes and references) come down with:\n",
        "```bash",
        "uv run python -m robosimstudio.hub fetch gallery/<project>/<entry>",
        "```",
        "",
    ]
    return "\n".join(lines)


def entry_page(entry: dict) -> str:
    """A ported scene's own README: its clips, where it stands, and how to fetch and run it."""
    where = f"gallery/{entry['project']}/{entry['id']}"
    lines = [f"# {entry['title']}\n", HEADER, f"{entry['note']}\n"]
    lines += [
        f'<a href="{_asset(entry, name)}"><img src="{_asset(entry, preview(name))}" width="49%" alt="{Path(name).stem}"></a>'
        for name in entry["media"]
    ]
    lines += [
        "",
        f"**Status:** {_status(entry)} · **robot:** {entry['robot']}  ",
        f"**Source:** {entry['source']} {_links(entry)}  ".rstrip() + "  ",
        f"**Licence:** {entry['licence']}\n",
        "Its meshes, textures and reference photos are on the Hub; fetch them, then open the live page and play",
        "the routine (`REPORT.md` says how the scene was built and what it measures):\n",
        "```bash",
        f"uv run python -m robosimstudio.hub fetch {where}",
        f"cd {where}",
        "uv run ./ctl up               # the live page; it prints its address",
        "uv run ./ctl play routine --wait",
        "uv run ./ctl check && uv run ./ctl physics",
        "```",
        "",
    ]
    return "\n".join(lines)


def pages(root: Path = REPO) -> dict[Path, str]:
    """Every generated page, by path, as it should read: the index, each project's, each ported scene's."""
    projects, entries = load(root)
    out = {root / "gallery" / "README.md": index_page(projects, entries)}
    for project in projects:
        own = [e for e in entries if e["project"] == project["id"]]
        out[root / "gallery" / project["id"] / "README.md"] = project_page(project, own)
    for entry in entries:
        if entry.get("code") == f"gallery/{entry['project']}/{entry['id']}":
            out[root / entry["code"] / "README.md"] = entry_page(entry)
    return out


def card(projects: list[dict], entries: list[dict]) -> str:
    """The dataset card: every entry's clips play inline on the Hub, with its source and licence."""

    def hub_url(entry: dict, name: str) -> str:
        return _asset(entry, name, "main")

    lines = [
        "---",
        "license: other",
        "license_name: per-item",
        "license_link: https://github.com/KeplerC/robosimstudio/blob/main/gallery/README.md",
        "pretty_name: RoboSimStudio gallery",
        "tags: [robotics, simulation, real2sim, manipulation]",
        "---",
        "",
        "# RoboSimStudio gallery",
        "",
        "Clips and scene files for the [RoboSimStudio](https://github.com/KeplerC/robosimstudio) gallery, by project.",
        "Each item states its source and licence; the simulation code is Apache-2.0. Fetch an entry's files with",
        "`python -m robosimstudio.hub fetch gallery/<project>/<entry>` from a RoboSimStudio checkout.",
        "",
    ]
    for project in projects:
        lines += [f"## {project['title']}", "", project["about"].strip(), ""]
        for entry in (e for e in entries if e["project"] == project["id"]):
            lines += [f"### {entry['title']}", ""]
            for name in entry["media"]:
                if name.endswith(".mp4"):
                    lines.append(
                        f'<video src="{hub_url(entry, name)}" poster="{hub_url(entry, "poster.jpg")}" '
                        'controls muted loop playsinline preload="none" width="480"></video>'
                    )
                else:
                    lines.append(f'<img src="{hub_url(entry, name)}" width="480">')
            lines += [
                "",
                f"*{_status(entry)} · {entry['robot']}.* {entry['note']}  ",
                f"Source: {entry['source']} {_links(entry)}  ",
                f"Licence: {entry['licence']}",
                "",
            ]
    return "\n".join(lines)


def _duration(path: Path) -> float:
    probe = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    return float(subprocess.run(probe, check=True, capture_output=True, text=True).stdout)


def _encode(src: Path, out: Path) -> None:
    """A web clip: H.264, at most 1280 px wide, no audio, moov atom first so it streams."""
    vf = "scale='min(1280,iw)':-2:flags=lanczos,format=yuv420p"
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-c:v", "libx264", "-preset", "slow"]
    subprocess.run([*cmd, "-crf", "24", "-movflags", "+faststart", "-an", str(out)], check=True)


def _still(src: Path, out: Path, at: float | None = None) -> None:
    """A JPEG at most 1280 px wide: an image as is, or a video's frame at ``at`` seconds."""
    seek = ["-ss", f"{at:.2f}"] if at is not None else []
    cmd = ["ffmpeg", "-v", "error", "-y", *seek, "-i", str(src), "-frames:v", "1"]
    subprocess.run([*cmd, "-vf", "scale='min(1280,iw)':-2", "-q:v", "3", str(out)], check=True)


def _webp(src: Path, out: Path) -> None:
    """A looping preview of a clip: :data:`PREVIEW_WIDTH` wide, at most :data:`PREVIEW_MAX_S` long, in budget."""
    from PIL import Image

    speed = min(8.0, max(1.0, _duration(src) / PREVIEW_MAX_S))
    with tempfile.TemporaryDirectory() as tmp:
        vf = f"setpts=PTS/{speed:.4f},fps={PREVIEW_FPS},scale={PREVIEW_WIDTH}:-2:flags=lanczos"
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-an", f"{tmp}/f%05d.png"]
        subprocess.run(cmd, check=True)
        frames = [Image.open(p).convert("RGB") for p in sorted(Path(tmp).glob("f*.png"))]
        if not frames:
            raise SystemExit(f"{src}: no frames to preview")
        written = write_webp(frames, out, PREVIEW_FPS, budget=PREVIEW_BUDGET)
    if not written["fits"]:
        raise SystemExit(f"{out}: {written['bytes']} bytes at the lowest quality, over {PREVIEW_BUDGET}")


def _source(key: str, name: str, given: str) -> Path:
    """A staging source, refused when it is a credential, a log, a symlink or not a file."""
    src = Path(given).expanduser()
    if FORBIDDEN_STAGE.search(src.name) or src.is_symlink() or not src.is_file():
        raise SystemExit(f"{key}/{name}: refusing {src} (a credential, a log, a symlink or not a file)")
    return src


def stage(sources_file: Path, out: Path) -> list[Path]:
    """Encode every source the sources file names into ``out``, laid out like the dataset.

    Each file is written beside its target and moved into place when it is complete, and
    ``out/.stage.json`` records which source each staged file came from, so a re-run redoes
    exactly the files whose source changed or was re-pointed and trusts nothing half-written.
    """
    projects, entries = load()
    sources = tomllib.loads(sources_file.read_text(encoding="utf-8"))
    wanted = {f"{e['project']}/{e['id']}": e for e in entries}
    unknown = sorted(set(sources) - set(wanted))
    if unknown:
        raise SystemExit(f"sources name entries the manifest does not have: {unknown}")
    record_path = out / ".stage.json"
    record = json.loads(record_path.read_text()) if record_path.is_file() else {}
    jobs = []
    for key, given in sources.items():
        entry = wanted[key]
        stray = sorted(set(given) - set(entry["media"]) - {"poster"})
        if stray:
            raise SystemExit(f"{key}: sources for names the manifest does not list: {stray}")
        folder = out / "gallery" / key / "media"
        folder.mkdir(parents=True, exist_ok=True)
        for name in entry["media"]:
            if name not in given:
                raise SystemExit(f"{key}: no source for {name}")
            jobs.append((_source(key, name, given[name]), folder / name, None))
        poster, first = folder / "poster.jpg", folder / entry["media"][0]
        if "poster" in given:
            jobs.append((_source(key, "poster", given["poster"]), poster, None))
        else:
            jobs.append((first, poster, "frame"))

    def stamp(src: Path) -> str:
        info = src.stat()
        return f"{src.resolve()}:{info.st_size}:{info.st_mtime_ns}"

    def make(job: tuple[Path, Path, str | None]) -> Path:
        src, target, how = job
        key = target.relative_to(out).as_posix()
        if target.is_file() and record.get(key) == stamp(src):
            return target
        partial = target.with_name(f".{target.stem}.part{target.suffix}")
        if how == "frame":
            _still(src, partial, at=0.55 * _duration(src) if src.suffix == ".mp4" else None)
        elif how == "webp":
            _webp(src, partial)
        else:
            (_encode if target.suffix == ".mp4" else _still)(src, partial)
        partial.replace(target)
        record[key] = stamp(src)
        return target

    clips = [job for job in jobs if job[2] is None]
    with ThreadPoolExecutor(max_workers=6) as pool:
        written = list(pool.map(make, clips))
        previews = [(target, target.with_suffix(".webp"), "webp") for _, target, _ in clips if target.suffix == ".mp4"]
        written += list(pool.map(make, previews))
    written += [make(job) for job in jobs if job[2] == "frame"]
    record_path.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    (out / "README.md").write_text(card(projects, entries), encoding="utf-8")
    written.append(out / "README.md")
    for path in written:
        if path.stat().st_size > MAX_STAGED:
            raise SystemExit(f"{path} is over {MAX_STAGED / 1e6:.0f} MB")
    for path in sorted(written):
        print(f"{path.stat().st_size / 1e6:8.2f} MB  {path.relative_to(out)}")
    print(f"{sum(p.stat().st_size for p in written) / 1e6:.1f} MB in {len(written)} files")
    return written


def needed(entries: list[dict]) -> set[str]:
    """Every dataset path the manifest names under ``gallery/``: each clip, its WebP, and each entry's poster."""
    return {
        f"gallery/{e['project']}/{e['id']}/media/{n}"
        for e in entries
        for n in {*e["media"], *(preview(m) for m in e["media"]), "poster.jpg"}
    }


def upload(folder: Path, message: str, *, prune: bool = False) -> str:
    """Send a staged folder to the dataset; pin ``REVISION`` to the new commit only if it holds every clip.

    The manifest's clips, previews and posters are checked against the new commit before anything
    is pinned: a partial upload leaves ``REVISION`` and the pages where they were. ``prune`` first
    deletes, in the same commit, every file in an entry's ``media/`` folder that the manifest no longer
    names. Only media: the manifest does not list a twin's assets, so it cannot say which are stale.
    """
    from huggingface_hub import HfApi

    for path in folder.rglob("*"):
        rel = path.relative_to(folder).as_posix()
        if path.is_symlink() or FORBIDDEN_STAGE.search(rel) or ".part." in path.name:
            raise SystemExit(f"refusing to upload {rel}")
    _, entries = load()
    want = needed(entries)
    staged = {p.relative_to(folder).as_posix() for p in (folder / "gallery").rglob("*") if p.is_file()}
    media = (f.path for f in hub.files("gallery") if f.path.split("/")[3:4] == ["media"])
    stale = sorted(p for p in media if p not in want and p not in staged) if prune else []
    info = HfApi().upload_folder(
        folder_path=str(folder),
        repo_id=hub.DATASET,
        repo_type="dataset",
        commit_message=message,
        ignore_patterns=[".stage.json"],
        delete_patterns=stale or None,
    )
    revision = info.oid
    present = {f.path for f in hub.files("gallery", revision)}
    missing = sorted(want - present)
    if missing:
        raise SystemExit(
            f"uploaded as {revision}, but it lacks files the manifest names; REVISION not moved:\n  "
            + "\n  ".join(missing)
        )
    source = REPO / "robosimstudio" / "hub.py"
    text, count = re.subn(
        r'^REVISION = "[^"]*"', f'REVISION = "{revision}"', source.read_text(encoding="utf-8"), count=1, flags=re.M
    )
    if count != 1:
        raise SystemExit(f"uploaded as {revision}, but found no REVISION line in {source} to pin")
    source.write_text(text, encoding="utf-8")
    hub.REVISION = revision
    for path, page in pages().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(page, encoding="utf-8")
    print(f"uploaded {folder} as {revision}; {len(stale)} stale files deleted; REVISION pinned; pages rewritten")
    return revision


def orphan_pages(root: Path = REPO) -> list[Path]:
    """Generated project pages left behind by a project the manifest no longer has."""
    projects, _ = load(root)
    keep = {root / "gallery" / p["id"] / "README.md" for p in projects}
    return [
        page
        for page in sorted((root / "gallery").glob("*/README.md"))
        if page not in keep and HEADER.strip() in page.read_text(encoding="utf-8")
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if a page is stale or the manifest is wrong")
    sub = parser.add_subparsers(dest="command")
    staging = sub.add_parser(
        "stage", help="encode the clips a sources file names into a folder laid out like the dataset"
    )
    staging.add_argument("sources", type=Path)
    staging.add_argument("out", type=Path)
    sending = sub.add_parser("upload", help="upload a staged folder and pin REVISION to it")
    sending.add_argument("folder", type=Path)
    sending.add_argument("-m", "--message", default="gallery: clips and posters")
    sending.add_argument(
        "--prune", action="store_true", help="also delete clips, previews and posters the manifest no longer names"
    )
    args = parser.parse_args(argv)
    if args.command == "stage":
        stage(args.sources, args.out)
        return 0
    if args.command == "upload":
        upload(args.folder, args.message, prune=args.prune)
        return 0
    wrong = problems()
    if not re.fullmatch(r"[0-9a-f]{40}", hub.REVISION):
        wrong.append(
            f"robosimstudio/hub.py REVISION is {hub.REVISION!r}: pin it to a dataset commit (gallery.py upload)"
        )
    if wrong:
        for line in wrong:
            print(line, file=sys.stderr)
        return 1
    written = pages()
    if args.check:
        stale = [p for p, text in written.items() if not p.is_file() or p.read_text(encoding="utf-8") != text]
        stale += orphan_pages()
        for path in stale:
            print(f"{path.relative_to(REPO)} is out of date: run scripts/gallery.py", file=sys.stderr)
        return 1 if stale else 0
    for path, text in written.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    for path in orphan_pages():
        path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
