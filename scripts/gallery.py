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

Maintainers publish new clips in two steps. ``stage`` encodes them from a local
sources file (outside the repository: it names paths on one machine) into a
folder laid out like the dataset, with a poster per entry and the dataset card;
``upload`` sends that folder, moves ``REVISION`` to the new commit and rewrites
the pages::

    uv run python scripts/gallery.py stage SOURCES.toml OUT/
    uv run --with huggingface_hub python scripts/gallery.py upload OUT/

A sources file maps ``"<project>/<entry>"`` to ``{ "<clip>.mp4" = "/path/to/source" }``,
optionally with ``poster = "/path/to/image"`` (otherwise a frame of the first clip).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from robosimstudio import hub  # noqa: E402

STATUSES = {
    "verified": "runs on current main and passes the harness checks; the commit it was checked on is given",
    "partial": "runs on current main, but the task is not completed yet or a check fails; the note says which",
    "video": "a video from older code, not yet ported to current main",
    "scene": "the scene is rebuilt, but no robot task in it is filmed yet",
}
ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
MEDIA = re.compile(r"^[a-z0-9][a-z0-9_-]*\.(mp4|jpg|png)$")
SHA = re.compile(r"^[0-9a-f]{7,40}$")
FORBIDDEN_STAGE = re.compile(r"(^|/)(auth\.json|codexhome|logs|review|\.env|gallery\.json)(/|$)|\.jsonl$")
MAX_STAGED = 200_000_000
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
        if not ID.match(pid) or pid in known:
            out.append(f"project {pid!r}: needs a unique lowercase id")
        for key in ("title", "about"):
            if not project.get(key):
                out.append(f"project {pid}: no {key}")
        known.add(pid)
    seen = set()
    for entry in entries:
        name = f"{entry.get('project')}/{entry.get('id')}"
        if entry.get("project") not in known:
            out.append(f"{name}: project is not declared")
        if not ID.match(entry.get("id", "")) or name in seen:
            out.append(f"{name}: needs a unique lowercase id within its project")
        seen.add(name)
        for key in ("title", "robot", "source", "licence", "note"):
            if not entry.get(key):
                out.append(f"{name}: no {key}")
        status = entry.get("status")
        if status not in STATUSES:
            out.append(f"{name}: status {status!r} is not one of {', '.join(STATUSES)}")
        media = entry.get("media", [])
        if not media or any(not MEDIA.match(m) for m in media) or media[0] == "poster.jpg":
            out.append(f"{name}: media must be a non-empty list of lowercase .mp4/.jpg/.png names")
        code = entry.get("code")
        if status in ("verified", "partial"):
            if not code or not (root / code).exists():
                out.append(f"{name}: a {status} entry needs its code in the checkout (code = ...)")
            elif not Path(code).is_relative_to(Path("gallery") / entry.get("project", "")):
                out.append(f"{name}: its code must live under gallery/{entry.get('project')}/")
        if status == "verified" and not SHA.match(entry.get("verified_on", "")):
            out.append(f"{name}: a verified entry needs verified_on = the commit it was checked on")
    if (root / "gallery" / "tasks").exists():
        out.append("gallery/tasks/ must not exist: a top-level <dir>/tasks/ is loaded as a task collection")
    return out


def _asset(entry: dict, name: str) -> str:
    return hub.url(f"gallery/{entry['project']}/{entry['id']}/media/{name}")


def _status(entry: dict) -> str:
    if entry["status"] == "verified":
        return f"verified on main ({entry['verified_on'][:7]})"
    return {"partial": "partial", "video": "video, older code", "scene": "scene only"}[entry["status"]]


def _links(entry: dict) -> str:
    links = entry.get("links", [])
    return " ".join(f"[{i + 1}]({link})" for i, link in enumerate(links))


def project_page(project: dict, entries: list[dict]) -> str:
    """One project's page: a grid of posters, each linking to its clip, then a table of the details."""
    lines = [f"# {project['title']}\n", HEADER, f"{project['about'].strip()}\n", "[All projects](../README.md)\n"]
    lines.append("<table>")
    for row in range(0, len(entries), 3):
        lines.append("<tr>")
        for entry in entries[row : row + 3]:
            lines.append(
                f'<td width="33%" valign="top"><a href="{_asset(entry, entry["media"][0])}">'
                f'<img src="{_asset(entry, "poster.jpg")}" width="100%" alt="{entry["title"]}"></a><br>'
                f"<b>{entry['title']}</b><br><sub>{_status(entry)} · {entry['robot']}</sub></td>"
            )
        lines.append("</tr>")
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
        "Tasks built with RoboSimStudio, by project. Every clip is hosted on Hugging Face, in the dataset",
        f"[{hub.DATASET}](https://huggingface.co/datasets/{hub.DATASET}); a card links to its clip.\n",
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
        cover = _asset(own[0], "poster.jpg") if own else ""
        lines.append(
            f"| [{project['title']}]({project['id']}/README.md) | {len(own)} | {verified} | "
            f'<img src="{cover}" width="160" alt="">'
            " |"
        )
    lines += [
        "",
        "An entry's files (its clips, and for a ported scene its meshes and references) come down with:\n",
        "```bash",
        "uv run python -m robosimstudio.hub fetch gallery/<project>/<entry>",
        "```",
        "",
    ]
    return "\n".join(lines)


def pages(root: Path = REPO) -> dict[Path, str]:
    """Every generated page, by path, as it should read."""
    projects, entries = load(root)
    out = {root / "gallery" / "README.md": index_page(projects, entries)}
    for project in projects:
        own = [e for e in entries if e["project"] == project["id"]]
        out[root / "gallery" / project["id"] / "README.md"] = project_page(project, own)
    return out


def card(projects: list[dict], entries: list[dict]) -> str:
    """The dataset card: every entry's clips play inline on the Hub, with its source and licence."""

    def hub_url(entry: dict, name: str) -> str:
        return f"https://huggingface.co/datasets/{hub.DATASET}/resolve/main/gallery/{entry['project']}/{entry['id']}/media/{name}"

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


def stage(sources_file: Path, out: Path) -> list[Path]:
    """Encode every source the sources file names into ``out``, laid out like the dataset."""
    projects, entries = load()
    sources = tomllib.loads(sources_file.read_text(encoding="utf-8"))
    wanted = {f"{e['project']}/{e['id']}": e for e in entries}
    unknown = sorted(set(sources) - set(wanted))
    if unknown:
        raise SystemExit(f"sources name entries the manifest does not have: {unknown}")
    jobs = []
    for key, given in sources.items():
        entry = wanted[key]
        folder = out / "gallery" / key / "media"
        folder.mkdir(parents=True, exist_ok=True)
        for name in entry["media"]:
            if name not in given:
                raise SystemExit(f"{key}: no source for {name}")
            src = Path(given[name]).expanduser()
            if FORBIDDEN_STAGE.search(src.name) or src.is_symlink() or not src.is_file():
                raise SystemExit(f"{key}/{name}: refusing {src} (a credential, a log, a symlink or not a file)")
            jobs.append((src, folder / name))

    def make(job: tuple[Path, Path]) -> Path:
        src, target = job
        if not target.exists() or target.stat().st_mtime < src.stat().st_mtime:
            (_encode if target.suffix == ".mp4" else _still)(src, target)
        return target

    with ThreadPoolExecutor(max_workers=6) as pool:
        written = list(pool.map(make, jobs))
    for key, given in sources.items():
        folder = out / "gallery" / key / "media"
        poster, first = folder / "poster.jpg", folder / wanted[key]["media"][0]
        if "poster" in given:
            _still(Path(given["poster"]).expanduser(), poster)
        elif first.suffix == ".mp4":
            _still(first, poster, at=0.55 * _duration(first))
        else:
            _still(first, poster)
        written.append(poster)
    (out / "README.md").write_text(card(projects, entries), encoding="utf-8")
    written.append(out / "README.md")
    for path in written:
        if path.stat().st_size > MAX_STAGED:
            raise SystemExit(f"{path} is over {MAX_STAGED / 1e6:.0f} MB")
    for path in sorted(written):
        print(f"{path.stat().st_size / 1e6:8.2f} MB  {path.relative_to(out)}")
    print(f"{sum(p.stat().st_size for p in written) / 1e6:.1f} MB in {len(written)} files")
    return written


def upload(folder: Path, message: str) -> str:
    """Send a staged folder to the dataset, pin ``REVISION`` to the new commit and rewrite the pages."""
    from huggingface_hub import HfApi

    for path in folder.rglob("*"):
        rel = path.relative_to(folder).as_posix()
        if path.is_symlink() or FORBIDDEN_STAGE.search(rel):
            raise SystemExit(f"refusing to upload {rel}")
    info = HfApi().upload_folder(
        folder_path=str(folder), repo_id=hub.DATASET, repo_type="dataset", commit_message=message
    )
    revision = info.oid
    source = REPO / "robosimstudio" / "hub.py"
    text = source.read_text(encoding="utf-8")
    source.write_text(
        re.sub(r'^REVISION = "[^"]*"', f'REVISION = "{revision}"', text, count=1, flags=re.M), encoding="utf-8"
    )
    hub.REVISION = revision
    present = {f.path for f in hub.files("gallery", revision)}
    _, entries = load()
    missing = [
        f"gallery/{e['project']}/{e['id']}/media/{n}"
        for e in entries
        for n in [*e["media"], "poster.jpg"]
        if f"gallery/{e['project']}/{e['id']}/media/{n}" not in present
    ]
    for path, text in pages().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(f"uploaded {folder} as {revision}; REVISION pinned; pages rewritten")
    if missing:
        print("the manifest names files the dataset does not have:", *missing, sep="\n  ", file=sys.stderr)
    return revision


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
    args = parser.parse_args(argv)
    if args.command == "stage":
        stage(args.sources, args.out)
        return 0
    if args.command == "upload":
        upload(args.folder, args.message)
        return 0
    wrong = problems()
    stale = [path for path, text in pages().items() if not path.is_file() or path.read_text(encoding="utf-8") != text]
    if not args.check:
        for path, text in pages().items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        stale = []
    if not re.fullmatch(r"[0-9a-f]{40}", hub.REVISION):
        wrong.append(
            f"robosimstudio/hub.py REVISION is {hub.REVISION!r}: pin it to a dataset commit (gallery.py upload)"
        )
    for line in wrong:
        print(line, file=sys.stderr)
    for path in stale:
        print(f"{path.relative_to(REPO)} is out of date: run scripts/gallery.py", file=sys.stderr)
    return 1 if wrong or stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
