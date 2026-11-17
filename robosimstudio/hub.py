# SPDX-License-Identifier: Apache-2.0
"""Files RoboSimStudio keeps on the Hugging Face Hub, fetched only when asked for.

The repository carries the code, the robots the examples use and the examples'
clips. What is too heavy for it lives in one public dataset, at the path it
would have in a checkout: the gallery's videos and scenes under
``gallery/<project>/<entry>/``. Nothing here downloads on its own. A fetch is
asked for by name, from the checkout's root::

    python -m robosimstudio.hub fetch gallery/robosimstudio-real2sim/kitchen
    python -m robosimstudio.hub list gallery/rgbs

:data:`REVISION` pins the dataset's commit, so a checkout fetches exactly the
files it was written against; uploading new files moves it (``scripts/gallery.py
upload``). ``HF_ENDPOINT`` points the whole module at a mirror, as it does for
``huggingface_hub``. Only the standard library is used, so a base install can
fetch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DATASET", "REVISION", "RemoteFile", "fetch", "files", "url"]

DATASET = "keplerccc/robosimstudio-agentic-real2sim"
"""The dataset everything here lives in."""

REVISION = "4ee035a510543a5e281ab7567bc14a932bfec32c"
"""The dataset commit this checkout was written against."""

PUBLIC = "https://huggingface.co"
"""The Hub itself: what links written into pages point at, whatever mirror this machine uses."""

_CHUNK = 1 << 20


def _endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", PUBLIC).rstrip("/")


def url(path: str, revision: str | None = None, *, endpoint: str | None = None) -> str:
    """The download URL of one file in the dataset, from ``endpoint`` (default: ``HF_ENDPOINT`` or the Hub)."""
    rev = urllib.parse.quote(revision or REVISION, safe="")
    return f"{(endpoint or _endpoint()).rstrip('/')}/datasets/{DATASET}/resolve/{rev}/{urllib.parse.quote(path)}"


@dataclass(frozen=True)
class RemoteFile:
    """One file in the dataset: where it is, how big, and the hash its bytes must have."""

    path: str
    size: int
    sha256: str | None
    """Set for a file stored by LFS/Xet; None for a small file kept in git."""
    git_oid: str
    """The git blob id, which is what a small file is checked against."""

    def matches(self, data_path: Path) -> bool:
        """Whether ``data_path`` holds exactly this file's bytes."""
        if not data_path.is_file() or data_path.stat().st_size != self.size:
            return False
        if self.sha256:
            return _sha256(data_path) == self.sha256
        return _git_blob_id(data_path) == self.git_oid


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_id(path: Path) -> str:
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _get(address: str):
    request = urllib.request.Request(address, headers={"User-Agent": "robosimstudio-hub"})
    return urllib.request.urlopen(request, timeout=60)


_NEXT = re.compile(r'<([^>]+)>;\s*rel="next"')


def _listing(prefix: str, revision: str | None, *, recursive: bool) -> list[RemoteFile] | None:
    """The files the Hub lists under ``prefix``, or None when it has no such folder (its 404)."""
    rev = urllib.parse.quote(revision or REVISION, safe="")
    query = "?recursive=true" if recursive else ""
    address: str | None = f"{_endpoint()}/api/datasets/{DATASET}/tree/{rev}/{urllib.parse.quote(prefix)}{query}"
    out: list[RemoteFile] = []
    while address:
        try:
            with _get(address) as response:
                entries = json.loads(response.read().decode())
                found = _NEXT.search(response.headers.get("Link", "") or "")
        except urllib.error.HTTPError as error:
            if error.code == 404 and not out:
                return None
            raise
        for entry in entries:
            if entry.get("type") == "file":
                lfs = entry.get("lfs") or {}
                out.append(RemoteFile(entry["path"], int(entry["size"]), lfs.get("oid"), entry.get("oid", "")))
        address = urllib.parse.urljoin(address, found.group(1)) if found else None
    return out


def files(prefix: str, revision: str | None = None) -> list[RemoteFile]:
    """Every file under ``prefix`` (a folder or a single file), recursively, at ``revision``; empty if none.

    The Hub answers a listing of a single file, or of a path it does not have, with a 404, so a prefix it
    does not list as a folder is looked up as one file in its parent folder.
    """
    prefix = prefix.strip("/")
    listed = _listing(prefix, revision, recursive=True)
    if listed is not None:
        return listed
    parent = prefix.rpartition("/")[0]
    siblings = _listing(parent, revision, recursive=False) or [] if parent else []
    return [f for f in siblings if f.path == prefix]


def _target(dest: Path, path: str) -> Path:
    target = (dest / path).resolve()
    if not target.is_relative_to(dest.resolve()):
        raise ValueError(f"refusing {path!r}: it would land outside {dest}")
    return target


def fetch(
    prefix: str, dest: str | Path, revision: str | None = None, *, overwrite: bool = False, quiet: bool = False
) -> list[Path]:
    """Download every file under ``prefix`` into ``dest``, at its dataset path; return the local paths.

    A file already there with the right bytes is kept (it is re-read to check its
    hash, so nothing is downloaded twice). A file there with *other* bytes is
    refused unless ``overwrite``: it may be somebody's edit. Each download is
    written beside its target and moved into place only once its hash matches, so
    an interrupted or corrupted fetch leaves nothing behind that a later run
    would trust.
    """
    dest = Path(dest)
    remote = files(prefix, revision)
    if not remote:
        raise FileNotFoundError(f"nothing under {prefix!r} in {DATASET} at {revision or REVISION}")
    out = []
    for item in remote:
        target = _target(dest, item.path)
        if item.matches(target):
            out.append(target)
            continue
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target} differs from the dataset's {item.path}; pass overwrite to replace it")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not quiet:
            print(f"fetch {item.path} ({item.size / 1e6:.1f} MB)", file=sys.stderr)
        handle, partial = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".part")
        try:
            with os.fdopen(handle, "wb") as sink, _get(url(item.path, revision)) as response:
                while chunk := response.read(_CHUNK):
                    sink.write(chunk)
            if not item.matches(Path(partial)):
                raise OSError(f"{item.path}: downloaded bytes do not match the dataset's hash")
            os.replace(partial, target)
        finally:
            if os.path.exists(partial):
                os.unlink(partial)
        out.append(target)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m robosimstudio.hub", description=__doc__.split("\n\n")[0])
    parser.add_argument("--revision", default=None, help=f"dataset commit (default: the pinned {REVISION})")
    sub = parser.add_subparsers(dest="command", required=True)
    get = sub.add_parser("fetch", help="download a folder or file into DEST at its dataset path")
    get.add_argument("prefix")
    get.add_argument("--dest", default=".", help="where the dataset's paths start (default: here)")
    get.add_argument("--overwrite", action="store_true", help="replace local files whose bytes differ")
    show = sub.add_parser("list", help="list the files under a folder")
    show.add_argument("prefix")
    args = parser.parse_args(argv)
    if args.command == "list":
        for item in files(args.prefix, args.revision):
            print(f"{item.size / 1e6:9.2f} MB  {item.path}")
        return 0
    for path in fetch(args.prefix, args.dest, args.revision, overwrite=args.overwrite):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
