# Task 3: Filing

<!-- written by scripts/gallery.py from gallery/index.toml; edit the manifest, not this page -->

Two YAM hands pick a wooden strip off the mat by its ends, press its marked face up against a rotary sanding drum that spins on its own motor and work it back and forth until the blue mark is filed away, then set it back down, over the clip's 99 s and within about 1 s of each of its phases. On current main 98.8% of the mark is cleared, only where the strip pressed on the spinning drum; check and every physics replay pass. The main clip is a 2x crop of the wide camera, because the YAM's jaws hide the 80 mm strip from the clip's own viewpoint; the stock YAM's elbows ride higher than the real arms'.

<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/rgbs/filing/media/filing.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/rgbs/filing/media/filing.webp" width="49%" alt="filing"></a>
<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/rgbs/filing/media/filing-photo.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/rgbs/filing/media/filing-photo.webp" width="49%" alt="filing-photo"></a>

**Status:** verified on main (2cc9c0c) · **robot:** YAM  
**Source:** Twin of a real RGBS bench video (not published), built by Codex on the live page and accepted by a reviewer agent  
**Licence:** Apache-2.0

Its meshes, textures and reference photos are on the Hub; fetch them, then open the live page and play
the routine (`REPORT.md` says how the scene was built and what it measures):

```bash
uv run python -m robosimstudio.hub fetch gallery/rgbs/filing
cd gallery/rgbs/filing
uv run ./ctl up               # the live page; it prints its address
uv run ./ctl play routine --wait
uv run ./ctl check && uv run ./ctl physics
```
