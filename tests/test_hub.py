# SPDX-License-Identifier: Apache-2.0
"""``robosimstudio.hub`` fetches the dataset's files, and only bytes whose hash matches.

Against a local stand-in for the Hub's two endpoints (the tree listing and the
resolve redirect), never the network: a fetch lands files at their dataset
paths, keeps a file it already has, refuses bytes that do not match, and
refuses a path that would leave the destination.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from unittest import mock

from robosimstudio import hub

TIER = "fast"
"""No GPU: a local HTTP server and a temporary directory."""

BIG = b"\x00clip bytes\xff" * 1000
SMALL = b"a small text file\n"


def _git_oid(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


class _Hub(BaseHTTPRequestHandler):
    files: ClassVar[dict[str, bytes]] = {}
    listing: ClassVar[list[dict]] = []
    downloads: ClassVar[list[str]] = []
    corrupt: ClassVar[set[str]] = set()

    def log_message(self, *args) -> None:  # quiet
        pass

    def _send(self, code: int, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.send_response(code)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        tree = f"/api/datasets/{hub.DATASET}/tree/rev1/"
        resolve = f"/datasets/{hub.DATASET}/resolve/rev1/"
        if parsed.path.startswith(tree):
            prefix = urllib.parse.unquote(parsed.path[len(tree) :])
            recursive = "recursive=true" in parsed.query
            under = [
                e
                for e in self.listing
                if e["path"].startswith(prefix + "/") and (recursive or "/" not in e["path"][len(prefix) + 1 :])
            ]
            if not under:  # the real Hub: a missing path, or a single file, lists as 404
                self._send(404, b"")
                return
            page = int(urllib.parse.parse_qs(parsed.query).get("cursor", ["0"])[0])
            chunk = under[page : page + 1]
            headers = {}
            if page + 1 < len(under):
                headers["Link"] = f'<{parsed.path}?recursive=true&cursor={page + 1}>; rel="next"'
            self._send(200, json.dumps(chunk).encode(), headers)
        elif parsed.path.startswith(resolve):
            name = urllib.parse.unquote(parsed.path[len(resolve) :])
            self._send(302, b"", {"Location": f"/cdn/{urllib.parse.quote(name)}"})
        elif parsed.path.startswith("/cdn/"):
            name = urllib.parse.unquote(parsed.path[len("/cdn/") :])
            self.downloads.append(name)
            data = self.files[name]
            self._send(200, data[:-1] + b"!" if name in self.corrupt else data)
        else:
            self._send(404, b"")


class TestFetch(unittest.TestCase):
    def setUp(self) -> None:
        _Hub.files = {"gallery/p/e/media/clip.mp4": BIG, "gallery/p/e/notes.txt": SMALL}
        _Hub.listing = [
            {"type": "directory", "path": "gallery/p/e/media", "oid": "x", "size": 0},
            {
                "type": "file",
                "path": "gallery/p/e/media/clip.mp4",
                "size": len(BIG),
                "oid": "x",
                "lfs": {"oid": hashlib.sha256(BIG).hexdigest(), "size": len(BIG)},
            },
            {"type": "file", "path": "gallery/p/e/notes.txt", "size": len(SMALL), "oid": _git_oid(SMALL)},
        ]
        _Hub.downloads, _Hub.corrupt = [], set()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Hub)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        endpoint = f"http://127.0.0.1:{self.server.server_address[1]}"
        patches = [mock.patch.dict(os.environ, {"HF_ENDPOINT": endpoint}), mock.patch.object(hub, "REVISION", "rev1")]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.dest = Path(scratch.name)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def test_files_land_at_their_dataset_paths(self) -> None:
        got = hub.fetch("gallery/p/e", self.dest, quiet=True)
        self.assertEqual(BIG, (self.dest / "gallery/p/e/media/clip.mp4").read_bytes())
        self.assertEqual(SMALL, (self.dest / "gallery/p/e/notes.txt").read_bytes())
        self.assertEqual(2, len(got))

    def test_a_listing_over_several_pages_is_read_whole(self) -> None:
        self.assertEqual(2, len(hub.files("gallery/p")))

    def test_a_file_already_there_is_kept(self) -> None:
        hub.fetch("gallery/p/e", self.dest, quiet=True)
        hub.fetch("gallery/p/e", self.dest, quiet=True)
        self.assertEqual(2, len(_Hub.downloads))

    def test_bytes_that_do_not_match_are_refused_and_left_nowhere(self) -> None:
        _Hub.corrupt = {"gallery/p/e/media/clip.mp4"}
        with self.assertRaises(OSError):
            hub.fetch("gallery/p/e", self.dest, quiet=True)
        media = self.dest / "gallery/p/e/media"
        self.assertEqual([], [p.name for p in media.iterdir()] if media.exists() else [])

    def test_a_path_outside_the_destination_is_refused(self) -> None:
        _Hub.listing.append({"type": "file", "path": "gallery/p/e/../../../../escape.txt", "size": 1, "oid": "x"})
        with self.assertRaises(ValueError):
            hub.fetch("gallery/p/e", self.dest, quiet=True)

    def test_a_single_file_is_fetched_by_its_path(self) -> None:
        got = hub.fetch("gallery/p/e/notes.txt", self.dest, quiet=True)
        self.assertEqual([self.dest.resolve() / "gallery/p/e/notes.txt"], got)
        self.assertEqual(SMALL, got[0].read_bytes())

    def test_a_local_file_with_other_bytes_is_not_overwritten_unless_asked(self) -> None:
        mine = self.dest / "gallery/p/e/notes.txt"
        mine.parent.mkdir(parents=True)
        mine.write_bytes(b"my edit\n")
        with self.assertRaises(FileExistsError):
            hub.fetch("gallery/p/e", self.dest, quiet=True)
        self.assertEqual(b"my edit\n", mine.read_bytes())
        hub.fetch("gallery/p/e", self.dest, overwrite=True, quiet=True)
        self.assertEqual(SMALL, mine.read_bytes())

    def test_an_empty_prefix_says_so(self) -> None:
        with self.assertRaises(FileNotFoundError):
            hub.fetch("gallery/nothing", self.dest, quiet=True)


if __name__ == "__main__":
    unittest.main()
