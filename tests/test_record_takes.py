# SPDX-License-Identifier: Apache-2.0
"""Every clip a README shows is filmed by a take, or kept for a reason that is written down.

``scripts/recording/takes.py`` is the table ``scripts/record_examples.py``
films from: each example's own page, driven by a cursor. A script added without
a take is a README whose clip nobody can remake; a take that names a script
which moved writes nothing; a clip in ``media/`` that no take films and no
reason keeps is one nobody can say the provenance of. All three are table
facts, read here without the library, a GPU or a browser.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import unittest
from pathlib import Path

TIER = "fast"
"""Reads the take table and lists files; imports nothing of the library."""

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from recording.takes import KEPT, TAKES  # noqa: E402

NUMBERED = re.compile(r"^\d\d_\w+$")


def scripts() -> list[Path]:
    """The scripts a clip is owed to: ``examples/NN_x/NN_y.py``, and ``examples/08_demos/<name>/<name>.py``."""
    found = [p for p in (REPO / "examples").glob("[0-9][0-9]_*/[0-9][0-9]_*.py") if NUMBERED.match(p.parent.name)]
    found += [p for p in (REPO / "examples" / "08_demos").glob("*/*.py") if p.stem == p.parent.name]
    return sorted(found)


def key(path: Path) -> str:
    """How the table names a script's clip or a file in a ``media/`` folder: ``01_cloth/00_towel``."""
    folder = path.parent.parent if path.parent.name == "media" else path.parent
    return f"{folder.relative_to(REPO).as_posix().removeprefix('examples/')}/{path.stem}"


class TestTheTakeTable(unittest.TestCase):
    def test_every_take_names_a_script_that_is_there(self) -> None:
        """A pending take may name a script another branch has not landed yet, if its folder is new too."""
        for take in TAKES:
            with self.subTest(take=take.key):
                script = REPO / take.example
                if take.pending and not script.parent.is_dir():
                    continue
                self.assertTrue(script.is_file(), f"{take.example} is not in this tree")

    def test_every_clip_lands_in_a_media_folder(self) -> None:
        for take in TAKES:
            with self.subTest(take=take.key):
                parts = take.media.relative_to(REPO).parts
                self.assertEqual("media", parts[-2])
                self.assertEqual(".webp", take.media.suffix)
                inside_examples = parts[0] == "examples" and len(parts) == 4 and NUMBERED.match(parts[1])
                inside_demo = parts[:2] == ("examples", "08_demos") and len(parts) == 5
                self.assertTrue(
                    inside_examples or inside_demo,
                    f"{take.media} is not in examples/NN_*/media or examples/08_demos/*/media",
                )

    def test_no_two_takes_write_one_clip(self) -> None:
        keys = [take.key for take in TAKES]
        self.assertEqual(sorted(set(keys)), sorted(keys))
        self.assertFalse(set(keys) & set(KEPT), "a clip is both filmed and kept")

    def test_every_kept_clip_says_why(self) -> None:
        for name, reason in KEPT.items():
            with self.subTest(kept=name):
                self.assertGreater(len(reason.split()), 4, "a reason is a sentence")

    def test_every_kept_clip_is_on_disk(self) -> None:
        """A kept clip is one no take can film, whose file stands: kept is no reason for a script to show nothing.

        The eight scored scripts in ``06_tasks`` were kept as "its clip is that one's", their material's,
        and their index rows showed no clip at all."""
        for name in KEPT:
            folder, stem = name.rsplit("/", 1)
            with self.subTest(kept=name):
                self.assertTrue((REPO / "examples" / folder / "media" / f"{stem}.webp").is_file(), "no such clip")

    def test_every_script_has_a_take_or_a_reason(self) -> None:
        owed = {take.key for take in TAKES} | set(KEPT)
        for script in scripts():
            with self.subTest(script=str(script.relative_to(REPO))):
                self.assertTrue(
                    key(script) in owed, f"{key(script)}: add a take to scripts/recording/takes.py, or a KEPT reason"
                )

    def test_every_clip_on_disk_is_filmed_or_kept(self) -> None:
        owed = {take.key for take in TAKES} | set(KEPT)
        clips = sorted([*REPO.glob("examples/*/media/*.webp"), *REPO.glob("examples/08_demos/*/media/*.webp")])
        self.assertTrue(clips)
        for clip in clips:
            with self.subTest(clip=str(clip.relative_to(REPO))):
                self.assertTrue(key(clip) in owed, f"{key(clip)}: no take films this clip and no reason keeps it")

    def test_every_scored_script_is_filmed_with_its_material_drive(self) -> None:
        """A scored script is its material's scene as a task: its take plays that drive and ends on the goal.

        Beat for beat after what stands in for the material take's opening pause, the same checks and
        one more, the goal met; the Goal folder open, and the view the material take films from once
        the hand is over its work."""
        from recording import takes

        scored = {
            "03_granular_scored": "03_materials/11_granular",
            "04_brush_scored": "03_materials/07_brush",
            "05_suction_scored": "04_extending/02_suction",
            "06_bag_scored": "03_materials/06_bag",
            "07_band_scored": "03_materials/04_band",
            "08_tape_scored": "03_materials/08_tape",
            "09_cutting_scored": "03_materials/10_cutting",
            "10_water_scored": "03_materials/14_water",
        }
        on_disk = {p.stem for p in (REPO / "examples" / "06_tasks").glob("[0-9][0-9]_*_scored.py")}
        self.assertEqual(set(scored), on_disk, "a scored script with no material drive here")
        for stem, of in scored.items():
            with self.subTest(scored=stem):
                take, material = takes.find(f"06_tasks/{stem}"), takes.find(of)
                drive = material.beats[1:]
                at = next(i for i in range(len(take.beats)) if take.beats[i : i + len(drive)] == drive)
                self.assertEqual(len(take.beats) - 2, at + len(drive), "the drive, and then the goal")
                self.assertIs(take.beats[-2].until, takes._goal_reads_true)
                self.assertEqual("goal", take.beats[-1].mark)
                self.assertEqual((*material.expect, takes.Check(takes._goal_met)), take.expect)
                self.assertEqual(("Goal",), take.layout.expand)
                if at == 1:
                    self.assertEqual(material.beats[0], take.beats[0], "no stand-in: the same opening")
                    self.assertEqual(material.camera, take.camera)
                else:
                    self.assertEqual(
                        material.camera, take.beats[at - 1].fn.args[0], "the view closes in to the material's"
                    )

    def test_every_take_of_a_superdex_script_says_it_needs_the_extra(self) -> None:
        """A script whose docstring says it needs ``[superdex]`` is listed with it: ``--list`` prints ``needs``."""
        for take in TAKES:
            script = REPO / take.example
            if not script.is_file():
                continue
            doc = " ".join((ast.get_docstring(ast.parse(script.read_text())) or "").split())
            if re.search(r"Needs (the )?``\[superdex\]``", doc):
                with self.subTest(take=take.key):
                    self.assertIn(
                        "superdex", take.needs, f"{take.example} needs [superdex]; its take says {take.needs}"
                    )

    def test_reading_the_table_imports_nothing_heavy(self) -> None:
        """The table is read here and by ``--list`` on machines with no GPU and no browser."""
        heavy = ("robosimstudio", "robosimstudio_studio", "playwright", "PIL", "numpy", "viser", "warp", "newton")
        probe = (
            f"import sys; sys.path.insert(0, {str(REPO / 'scripts')!r}); import recording.takes; "
            f"print(sorted({{m.split('.')[0] for m in sys.modules}} & set({heavy!r})))"
        )
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=True)
        self.assertEqual("[]", done.stdout.strip())


class TestAMaterialTakeIsItsScriptsOwnClip(unittest.TestCase):
    """``_material`` finds the take a scored script plays: its material script's own clip, never an extra one."""

    GROUPS = (
        "DRAFTS",
        "ROBOT_PARTICLES",
        "ROBOT_BAGS",
        "ROBOT_BRUSH",
        "ROBOT_SUCTION",
        "ROBOT_TAPE",
        "ROBOT_WATER",
        "ROBOT_CUTTING",
    )

    def test_a_named_take_of_the_script_is_passed_over_in_every_group(self) -> None:
        """A second, named take of a script in any group leaves its own clip the one found, not two."""
        from dataclasses import replace
        from unittest import mock

        from recording import takes

        for group in self.GROUPS:
            with self.subTest(group=group):
                own = next(t for t in getattr(takes, group) if not t.name)
                extra = replace(own, name=f"{own.stem}_again")
                with mock.patch.object(takes, group, (*getattr(takes, group), extra)):
                    self.assertIs(own, takes._material(own.example))


class TestAScoredTakeDropsOnlyAnOpeningPause(unittest.TestCase):
    """``scored(over=...)`` puts its drag over the work where the material take's opening pause was."""

    TAPE = "examples/03_materials/08_tape.py"

    def test_a_material_take_that_opens_on_a_drag_is_refused(self) -> None:
        """Its first beat is a step of the drive, not a pause: dropping it is refused, naming the script.

        The four shipped takes that drag over the work first are checked as the table is read."""
        from dataclasses import replace
        from unittest import mock

        from recording import takes

        own = takes._material(self.TAPE)
        drag = next(b for b in own.beats if not isinstance(b, takes.Wait))
        moving = replace(own, beats=(drag, *own.beats))
        group = tuple(moving if t is own else t for t in takes.ROBOT_TAPE)
        with mock.patch.object(takes, "ROBOT_TAPE", group):
            refusal = rf"08_tape\.py's take opens on {type(drag).__name__}, not a Wait"
            with self.assertRaisesRegex(ValueError, refusal):
                takes.scored("examples/06_tasks/08_tape_scored.py", self.TAPE, over=(0.0, 0.0, 0.1), what="the tab")
            kept = takes.scored("examples/06_tasks/08_tape_scored.py", self.TAPE)
        self.assertEqual(moving.beats, kept.beats[: len(moving.beats)], "without over, every beat is kept")


if __name__ == "__main__":
    unittest.main()
