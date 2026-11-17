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


def _root(case: unittest.TestCase, manifest: str) -> Path:
    scratch = tempfile.TemporaryDirectory()
    case.addCleanup(scratch.cleanup)
    root = Path(scratch.name)
    (root / "gallery").mkdir()
    (root / "gallery" / "index.toml").write_text(manifest, encoding="utf-8")
    return root


class TestTheGallery(unittest.TestCase):
    def test_the_pages_are_what_the_manifest_says(self) -> None:
        self.assertEqual(0, _tool().main(["--check"]), "run scripts/gallery.py")

    def test_a_good_manifest_has_no_problems(self) -> None:
        self.assertEqual([], _tool().problems(_root(self, GOOD)))

    def test_a_missing_licence_is_refused(self) -> None:
        found = _tool().problems(_root(self, GOOD.replace('licence = "Apache-2.0"\n', "")))
        self.assertTrue(any("no licence" in line for line in found), found)

    def test_an_unknown_status_is_refused(self) -> None:
        found = _tool().problems(_root(self, GOOD.replace('"video"', '"done"')))
        self.assertTrue(any("status" in line for line in found), found)

    def test_a_verified_entry_needs_its_code_and_commit(self) -> None:
        found = _tool().problems(_root(self, GOOD.replace('"video"', '"verified"')))
        self.assertTrue(any("needs its code" in line for line in found), found)
        self.assertTrue(any("verified_on" in line for line in found), found)

    def test_a_poster_named_as_a_clip_is_refused(self) -> None:
        found = _tool().problems(_root(self, GOOD.replace('["clip.mp4"]', '["clip.mp4", "poster.jpg"]')))
        self.assertTrue(any("never poster.jpg" in line for line in found), found)

    def test_code_outside_its_project_folder_is_refused_whatever_the_status(self) -> None:
        found = _tool().problems(_root(self, GOOD + 'code = "examples/live"\n'))
        self.assertTrue(any("must live under gallery/p/" in line for line in found), found)

    def test_a_trailing_newline_in_an_id_is_refused(self) -> None:
        found = _tool().problems(_root(self, GOOD.replace('id = "e"', 'id = "e\\n"')))
        self.assertTrue(any("unique lowercase id" in line for line in found), found)

    def test_a_tasks_folder_under_gallery_is_refused(self) -> None:
        root = _root(self, GOOD)
        (root / "gallery" / "tasks").mkdir()
        self.assertTrue(any("gallery/tasks/" in line for line in _tool().problems(root)))

    def test_every_clip_plays_on_its_page_and_links_to_its_video(self) -> None:
        tool = _tool()
        root = _root(self, GOOD.replace('["clip.mp4"]', '["clip.mp4", "close.mp4", "still.jpg"]'))
        page = tool.pages(root)[root / "gallery" / "p" / "README.md"]
        for name in ("clip", "close"):
            self.assertIn('href="https://huggingface.co/datasets/', page)
            self.assertIn(f"gallery/p/e/media/{name}.mp4", page)
            self.assertIn(f"gallery/p/e/media/{name}.webp", page)
        self.assertIn("gallery/p/e/media/still.jpg", page)

    def test_a_ported_scene_gets_a_page_that_says_how_to_fetch_and_run_it(self) -> None:
        tool = _tool()
        root = _root(self, GOOD.replace('status = "video"', 'status = "partial"\ncode = "gallery/p/e"'))
        (root / "gallery" / "p" / "e").mkdir(parents=True)
        page = tool.pages(root)[root / "gallery" / "p" / "e" / "README.md"]
        self.assertIn("python -m robosimstudio.hub fetch gallery/p/e", page)
        self.assertIn("./ctl play routine", page)

    def test_an_upload_must_hold_each_clip_its_preview_and_the_poster(self) -> None:
        tool = _tool()
        entry = {"project": "p", "id": "e", "media": ["clip.mp4", "still.jpg"]}
        self.assertEqual(
            {f"gallery/p/e/media/{n}" for n in ("clip.mp4", "clip.webp", "still.jpg", "poster.jpg")},
            tool.needed([entry]),
        )


if __name__ == "__main__":
    unittest.main()
