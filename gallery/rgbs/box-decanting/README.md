# Task 2: Box Decanting

<!-- written by scripts/gallery.py from gallery/index.toml; edit the manifest, not this page -->

A YAM picks a safety cutter up while its other hand braces the carton's corner, draws the blade along the yellow-taped seam, sets the cutter down, then the two hands lift the two top flaps open, each phase within about 2 s of the clip. On current main the tape is 100% cut, both flaps stand at 95 degrees, the carton slides at most 8.9 mm and the cutter ends released on the mat. The tape is this twin's own material law, since the library has no cuttable seal: it parts only where the held blade presses while drawn, and its hold on the flaps fades with the uncut tape. The inner flaps hang down where the clip's lie flat, the seam gap is 12 mm and the flap catches come in pitched 40 degrees. Check and the physics replays pass, except that with the cutter and carton twice as heavy the routine's own goal check missed (a note). The main clip is a crop of the photo camera's take.

<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/rgbs/box-decanting/media/box-decanting.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/rgbs/box-decanting/media/box-decanting.webp" width="49%" alt="box-decanting"></a>
<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/rgbs/box-decanting/media/box-decanting-photo.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/rgbs/box-decanting/media/box-decanting-photo.webp" width="49%" alt="box-decanting-photo"></a>

**Status:** verified on main (2cc9c0c) · **robot:** YAM  
**Source:** Twin of a real RGBS bench video (not published), built by Codex on the live page and accepted by a reviewer agent  
**Licence:** Apache-2.0

Its meshes, textures and reference photos are on the Hub; fetch them, then open the live page and play
the routine (`REPORT.md` says how the scene was built and what it measures):

```bash
uv run python -m robosimstudio.hub fetch gallery/rgbs/box-decanting
cd gallery/rgbs/box-decanting
uv run ./ctl up               # the live page; it prints its address
uv run ./ctl play routine --wait
uv run ./ctl check && uv run ./ctl physics
```
