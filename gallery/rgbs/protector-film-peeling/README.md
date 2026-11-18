# Task 4: Protector Film Peeling

<!-- written by scripts/gallery.py from gallery/index.toml; edit the manifest, not this page -->

The left YAM hand holds a phone at its corner in a small cradle while the right hand pinches the protector film's handle and peels the film back across the glass until it comes away; the film is the library's tape, stuck by its own adhesion, and comes off only through the pinch's pull. On current main the film is 100% peeled, none of it before the pull starts, and the phone moves at most 0.6 mm; check and the physics replays pass (heavy and drop do not apply to a film). The film's handle is a 62 x 8 x 24 mm bar the YAM's pads can hold, so the peel runs straight across rather than as the clip's corner arc. The clips show the measured peel live.

<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/rgbs/protector-film-peeling/media/protector-film-peeling.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/rgbs/protector-film-peeling/media/protector-film-peeling.webp" width="49%" alt="protector-film-peeling"></a>
<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/rgbs/protector-film-peeling/media/protector-film-peeling-wide.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/rgbs/protector-film-peeling/media/protector-film-peeling-wide.webp" width="49%" alt="protector-film-peeling-wide"></a>

**Status:** verified on main (2cc9c0c) · **robot:** YAM  
**Source:** Twin of a real RGBS bench video (not published), built by Codex on the live page and accepted by a reviewer agent  
**Licence:** Apache-2.0

Its meshes, textures and reference photos are on the Hub; fetch them, then open the live page and play
the routine (`REPORT.md` says how the scene was built and what it measures):

```bash
uv run python -m robosimstudio.hub fetch gallery/rgbs/protector-film-peeling
cd gallery/rgbs/protector-film-peeling
uv run ./ctl up               # the live page; it prints its address
uv run ./ctl play routine --wait
uv run ./ctl check && uv run ./ctl physics
```
