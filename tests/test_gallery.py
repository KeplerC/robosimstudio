# SPDX-License-Identifier: Apache-2.0
"""The gallery's pages are what its manifest says, and the manifest says enough.

``scripts/gallery.py`` writes ``gallery/README.md`` and one page per project
from ``gallery/index.toml``. An entry with no source or licence, an unknown
status, a "verified" entry with no code or no commit, or a ``gallery/tasks/``
(which would load as a task collection) is a gallery that has stopped being
true.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

TIER = "fast"
"""No GPU: reads files."""

REPO = Path(__file__).resolve().parents[1]

GOOD = """
[[project]]
id = "p"
title = "P"
about = "A project."

[[entry]]
project = "p"
id = "e"
title = "An entry"
status = "video"
robot = "YAM"
media = ["clip.mp4"]
source = "authored"
licence = "Apache-2.0"
note = "A note."
"""


def _tool():
    spec = importlib.util.spec_from_file_location("gallery_tool", REPO / "scripts" / "gallery.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _root(manifest: str) -> Path:
    root = Path(tempfile.mkdtemp())
    (root / "gallery").mkdir()
    (root / "gallery" / "index.toml").write_text(manifest, encoding="utf-8")
    return root


class TestTheGallery(unittest.TestCase):
    def test_the_pages_are_what_the_manifest_says(self) -> None:
        self.assertEqual(0, _tool().main(["--check"]), "run scripts/gallery.py")

    def test_a_good_manifest_has_no_problems(self) -> None:
        self.assertEqual([], _tool().problems(_root(GOOD)))

    def test_a_missing_licence_is_refused(self) -> None:
        found = _tool().problems(_root(GOOD.replace('licence = "Apache-2.0"\n', "")))
        self.assertTrue(any("no licence" in line for line in found), found)

    def test_an_unknown_status_is_refused(self) -> None:
        found = _tool().problems(_root(GOOD.replace('"video"', '"done"')))
        self.assertTrue(any("status" in line for line in found), found)

    def test_a_verified_entry_needs_its_code_and_commit(self) -> None:
        found = _tool().problems(_root(GOOD.replace('"video"', '"verified"')))
        self.assertTrue(any("needs its code" in line for line in found), found)
        self.assertTrue(any("verified_on" in line for line in found), found)

    def test_a_tasks_folder_under_gallery_is_refused(self) -> None:
        root = _root(GOOD)
        (root / "gallery" / "tasks").mkdir()
        self.assertTrue(any("gallery/tasks/" in line for line in _tool().problems(root)))

    def test_every_project_page_links_each_entry_to_its_clip_and_poster(self) -> None:
        tool = _tool()
        root = _root(GOOD)
        page = tool.pages(root)[root / "gallery" / "p" / "README.md"]
        self.assertIn("gallery/p/e/media/clip.mp4", page)
        self.assertIn("gallery/p/e/media/poster.jpg", page)


if __name__ == "__main__":
    unittest.main()
