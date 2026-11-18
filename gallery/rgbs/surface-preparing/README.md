# Task 5: Surface Preparing

<!-- written by scripts/gallery.py from gallery/index.toml; edit the manifest, not this page -->

A YAM picks up a white ridged foam sponge, presses it over a steel grab bar so the foam wraps it, wipes the bar from right to left in five strokes that lift where the clip's do, and puts the sponge back on the mat. On current main the blue marker stripe along the bar's top is 100% wiped, read only off pressed, sliding contact, the foam wraps up to 134 degrees of the bar and the bar does not move; check and the physics replays pass (heavy and drop do not apply to a soft sponge). The foam drapes less deeply over the bar than the real one. The sponge is a SuperDex soft body: it needs the superdex extra (uv sync --extra superdex).

<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/rgbs/surface-preparing/media/surface-preparing.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/rgbs/surface-preparing/media/surface-preparing.webp" width="49%" alt="surface-preparing"></a>
<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/rgbs/surface-preparing/media/surface-preparing-wide.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/rgbs/surface-preparing/media/surface-preparing-wide.webp" width="49%" alt="surface-preparing-wide"></a>

**Status:** verified on main (2cc9c0c) · **robot:** YAM  
**Source:** Twin of a real RGBS bench video (not published), built by Codex on the live page and accepted by a reviewer agent  
**Licence:** Apache-2.0

Its meshes, textures and reference photos are on the Hub; fetch them, then open the live page and play
the routine (`REPORT.md` says how the scene was built and what it measures):

```bash
uv run python -m robosimstudio.hub fetch gallery/rgbs/surface-preparing
cd gallery/rgbs/surface-preparing
uv run ./ctl up               # the live page; it prints its address
uv run ./ctl play routine --wait
uv run ./ctl check && uv run ./ctl physics
```
