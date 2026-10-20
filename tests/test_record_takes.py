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


class TestRecordersSideBySide(unittest.TestCase):
    """``record_examples.py`` serves each recorder's pages on its own ports: ``--port`` to ``--port + j - 1``.

    It used to hand one climbing port a take dealt, from ``--port``, so recorders at 9100 and 9101
    were handed the same ports, and a ``--port`` outside 9100-9199 was wrapped back into it unsaid."""

    def setUp(self) -> None:
        import tempfile
        from unittest import mock

        import record_examples

        self.rec = record_examples
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        for patch in (
            mock.patch.object(record_examples, "LOCKS", self.tmp / "locks"),
            mock.patch.object(record_examples, "port_free", lambda port: True),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def _claim(self, port: int, jobs: int) -> list:
        locks = self.rec.claim(self.rec.page_ports(port, jobs))
        self.addCleanup(lambda: [lock.close() for lock in locks])
        return locks

    def test_a_port_outside_the_range_is_refused_by_name(self) -> None:
        for port in (80, 9099, 9200, 9300):
            with self.subTest(port=port):
                with self.assertRaisesRegex(ValueError, rf"^--port {port} is outside 9100-9199"):
                    self.rec.page_ports(port, 1)
        with self.assertRaisesRegex(
            ValueError, r"^--port 9198 with -j 4 serves on 9198-9201, past 9199: start it at 9196"
        ):
            self.rec.page_ports(9198, 4)
        self.assertEqual(range(9196, 9200), self.rec.page_ports(9196, 4))

    def test_a_recorder_whose_ports_meet_anothers_is_refused_by_name(self) -> None:
        self._claim(9104, 4)
        with self.assertRaisesRegex(
            ValueError, r"^port 9104 is a page port of the recorder of pid \d+, ports 9104-9107"
        ):
            self._claim(9101, 4)
        self._claim(9100, 4)  # the refused claim kept none of 9101-9103
        self._claim(9108, 4)

    def test_a_port_something_else_listens_on_is_refused_by_name(self) -> None:
        from unittest import mock

        with mock.patch.object(self.rec, "port_free", lambda port: port != 9102):
            with self.assertRaisesRegex(ValueError, r"^port 9102 is in use by something that is not a recorder"):
                self._claim(9100, 4)
        self._claim(9100, 4)

    def test_a_lock_this_user_may_not_open_is_refused_by_name(self) -> None:
        """A lock directory another user made is not writable here: ``claim`` raised ``PermissionError`` past ``run_all``.

        Only root can make one another user's, so it is this user's 0o555 directory, read as another's: a
        recorder opens its own to every user.
        """
        import contextlib
        import io
        import os
        from unittest import mock

        if os.geteuid() == 0:
            self.skipTest("root opens any file")
        locks = self.tmp / "locks"
        locks.mkdir()
        locks.chmod(0o555)
        self.addCleanup(locks.chmod, 0o755)
        someone = mock.patch.object(self.rec.os, "geteuid", lambda: os.getuid() + 1)
        someone.start()
        self.addCleanup(someone.stop)
        with self.assertRaisesRegex(ValueError, rf"^port 9100's lock {locks}/9100.lock cannot be opened"):
            self._claim(9100, 4)
        args = _parse(self.rec, ["--gpus", "1", "-j", "4", "--port", "9100", "--out", str(self.tmp / "out")])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(2, self.rec.run_all(list(TAKES[:2]), args))
        self.assertRegex(err.getvalue(), r"^port 9100's lock .* cannot be opened")

    def test_recorders_under_different_tmpdirs_lock_in_one_place(self) -> None:
        """The locks lived under ``tempfile.gettempdir()``, so recorders started with different ``TMPDIR``s never met.

        A recorder in a process of its own holds a port under one ``TMPDIR``; a second, under another, asks
        for the same port. Both lock in a directory of this test's, which ``ROBOSIMSTUDIO_RECORD_LOCKS``
        names, so the machine's own is never touched; and one with only a ``TMPDIR`` locks in the machine's.
        """
        import os

        head = f"import sys; sys.path.insert(0, {str(REPO / 'scripts')!r}); import record_examples as rec; "
        head += "rec.port_free = lambda port: True\n"
        hold = "locks = rec.claim(range(9150, 9151)); print(9150, flush=True); sys.stdin.read()\n"
        ask = "import sys\ntry:\n    rec.claim(range(int(sys.argv[1]), int(sys.argv[1]) + 1)); print('claimed')\n"
        ask += "except ValueError as exc:\n    print(exc)\n"
        environ = {k: v for k, v in os.environ.items() if k != "ROBOSIMSTUDIO_RECORD_LOCKS"}

        def env(name: str, locks: bool = True) -> dict:
            (self.tmp / name).mkdir()
            moved = {"ROBOSIMSTUDIO_RECORD_LOCKS": str(self.tmp / "locks")} if locks else {}
            return {**environ, "TMPDIR": str(self.tmp / name), **moved}

        holder = subprocess.Popen(
            [sys.executable, "-c", head + hold], env=env("a"), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        self.addCleanup(holder.wait)
        self.addCleanup(holder.stdin.close)
        self.addCleanup(holder.stdout.close)
        port = int(holder.stdout.readline())
        asked = subprocess.run(
            [sys.executable, "-c", head + ask, str(port)], env=env("b"), capture_output=True, text=True, check=True
        )
        self.assertRegex(asked.stdout, rf"^port {port} is a page port of the recorder of pid {holder.pid}")
        named = subprocess.run(
            [sys.executable, "-c", head + "print(rec.LOCKS)"],
            env=env("c", locks=False),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual("/tmp/robosimstudio-record-ports", named.stdout.strip())

    def test_two_recorders_side_by_side_hand_out_disjoint_ports(self) -> None:
        """Four takes at once each, at 9100 and at 9104: each child serves on a port of its own recorder's four."""
        import contextlib
        import io
        from unittest import mock

        handed: dict[int, list[int]] = {}

        class Done:
            returncode = 0

            def __init__(self, cmd, **_) -> None:
                handed.setdefault(start, []).append(int(cmd[cmd.index("--port") + 1]))

            def poll(self) -> int:
                return 0

        chosen = list(TAKES[:10])
        with mock.patch.object(self.rec.subprocess, "Popen", Done), mock.patch.object(self.rec.time, "sleep"):
            for start in (9100, 9104):
                argv = ["--gpus", "1", "-j", "4", "--port", str(start), "--out", str(self.tmp / str(start))]
                args = _parse(self.rec, argv)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(0, self.rec.run_all(chosen, args))
        self.assertEqual(10, len(handed[9100]))
        self.assertEqual({9100, 9101, 9102, 9103}, set(handed[9100]))
        self.assertEqual({9104, 9105, 9106, 9107}, set(handed[9104]))


class TestALockOpensForEveryUser(unittest.TestCase):
    """A recorder opens a lock another user's recorder made, and opens its own directory and locks to every user.

    ``/tmp`` is sticky and writable by all, and under ``fs.protected_regular`` (2 on the box the
    recorders run on) the kernel refuses an ``O_CREAT`` open there of a file that belongs to another
    user who does not own the directory either, 0o666 or not. Every recorder opened its lock
    ``O_CREAT``, and a lock file is never removed, so a port another user's recorder had locked once
    could never be claimed. And the directory was made writable by all only by the call that made it,
    so one an earlier recorder had made 0o755, with 0o644 locks, stayed so.
    """

    def setUp(self) -> None:
        import tempfile
        from unittest import mock

        import record_examples

        self.rec = record_examples
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        self.locks = self.tmp / "locks"
        for patch in (
            mock.patch.object(record_examples, "LOCKS", self.locks),
            mock.patch.object(record_examples, "port_free", lambda port: True),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def _claim(self, opener=None) -> list[int]:
        """Claim 9100 with ``os.open`` replaced by ``opener(path, flags, real_open)``: the flags it was handed."""
        import os
        from unittest import mock

        real, flags = os.open, []

        def recording(path, how, *args):
            if str(path).endswith("9100.lock"):
                flags.append(how)
            return opener(path, how, lambda: real(path, how, *args)) if opener else real(path, how, *args)

        with mock.patch.object(self.rec.os, "open", recording):
            locks = self.rec.claim(range(9100, 9101))
        for lock in locks:
            lock.close()
        return flags

    def test_a_lock_that_is_there_is_opened_without_creating_it(self) -> None:
        import os

        self.locks.mkdir()
        (self.locks / "9100.lock").touch()
        flags = self._claim()
        self.assertTrue(flags)
        self.assertEqual([0] * len(flags), [how & os.O_CREAT for how in flags], "opened O_CREAT")

    def test_a_lock_another_users_recorder_made_is_claimed(self) -> None:
        """The kernel's rule, modelled: ``O_CREAT`` without ``O_EXCL`` on a lock that is there is refused."""
        import errno
        import os

        def protected(path, how, call):
            if how & os.O_CREAT and not how & os.O_EXCL and os.path.exists(path):
                raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(path))
            return call()

        self.locks.mkdir()
        (self.locks / "9100.lock").touch()
        self._claim(protected)

    def test_a_lock_made_between_the_look_and_the_make_is_opened(self) -> None:
        """Two recorders that find no lock both make it: the one whose make finds it there opens the other's."""
        import os

        def raced(path, how, call):
            if how & os.O_EXCL:
                open(path, "a").close()  # the other recorder's, an instant before; not through os.open
            return call()

        flags = self._claim(raced)
        plain, make = os.O_RDWR | os.O_APPEND, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL
        self.assertEqual([plain, make, plain], flags)

    def test_this_users_directory_and_locks_are_opened_to_every_user(self) -> None:
        import os
        import stat

        self.locks.mkdir()
        self.locks.chmod(0o755)
        (self.locks / "9100.lock").touch()
        (self.locks / "9100.lock").chmod(0o644)
        umask = os.umask(0o022)
        try:
            locks = self.rec.claim(range(9100, 9102))
        finally:
            os.umask(umask)
        for lock in locks:
            lock.close()
        modes = {p.name: stat.S_IMODE(p.stat().st_mode) for p in (self.locks, *sorted(self.locks.iterdir()))}
        self.assertEqual({"locks": 0o1777, "9100.lock": 0o666, "9101.lock": 0o666}, modes)


class TestAPageServesOnItsClaimedPort(unittest.TestCase):
    """viser moves to the next port by itself when its bind fails (``viser/infra/_infra.py``), so a page
    whose claimed port was taken between the recorder's check and the bind served on another one,
    perhaps a port another take of the block was handed, and the browser was pointed at the claimed one."""

    def _film(self, served: int):
        from types import SimpleNamespace
        from unittest import mock

        from recording import lockstep

        film = lockstep.Film(None, port=9104, scratch=None, software=False, probe=False, log=lambda _m: None)
        session = SimpleNamespace(env=None, server=SimpleNamespace(get_port=lambda: served))
        with mock.patch.object(lockstep.Film, "_shoot") as shoot:
            film(session)
        return film, shoot

    def test_a_page_on_another_port_is_refused_by_name(self) -> None:
        film, shoot = self._film(9105)
        self.assertRegex(film.error, r"^the page serves on port 9105, not 9104, the port its recorder claimed")
        shoot.assert_not_called()

    def test_a_page_on_its_port_is_filmed(self) -> None:
        film, shoot = self._film(9104)
        self.assertEqual("", film.error)
        shoot.assert_called_once()


def _parse(rec, argv: list[str]):
    """``record_examples``'s own command line, parsed as ``main`` parses it, without running anything."""
    from unittest import mock

    seen = []
    with mock.patch.object(rec, "run_all", lambda chosen, args: seen.append(args) or 0):
        rec.main(argv)
    return seen[0]


if __name__ == "__main__":
    unittest.main()
