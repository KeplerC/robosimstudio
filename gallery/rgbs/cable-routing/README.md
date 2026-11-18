# Task 1: Cable Routing

<!-- written by scripts/gallery.py from gallery/index.toml; edit the manifest, not this page -->

The left YAM hand presses down the lid strip of a cable box while the right hand picks up a power strip's cord and feeds it into the channel under the lid in short strokes; then the left hand lets go, catches the cord's end where it comes out at the far end and pulls it through, at the clip's pace. When it completes, the cord fills 100% of the channel, 387 mm of tail stays on the mat and the box moves under 1 mm; check passes. Partial because it is not yet reliable on current main: it completed 4 of 5 runs on freshly started pages, and one of two physics passes failed its written replay; every miss is the left hand's catch of the emerging end, which leaves the connector on the mat. The mouth of the channel has a small lead-in the clip's box does not, and the feed uses more, shorter strokes than the real arm. The cord is a SuperDex rod: it needs the superdex extra (uv sync --extra superdex).

<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/rgbs/cable-routing/media/cable-routing.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/rgbs/cable-routing/media/cable-routing.webp" width="49%" alt="cable-routing"></a>
<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/rgbs/cable-routing/media/cable-routing-photo.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/rgbs/cable-routing/media/cable-routing-photo.webp" width="49%" alt="cable-routing-photo"></a>

**Status:** partial · **robot:** YAM  
**Source:** Twin of a real RGBS bench video (not published), built by Codex on the live page and accepted by a reviewer agent  
**Licence:** Apache-2.0

Its meshes, textures and reference photos are on the Hub; fetch them, then open the live page and play
the routine (`REPORT.md` says how the scene was built and what it measures):

```bash
uv run python -m robosimstudio.hub fetch gallery/rgbs/cable-routing
cd gallery/rgbs/cable-routing
uv run ./ctl up               # the live page; it prints its address
uv run ./ctl play routine --wait
uv run ./ctl check && uv run ./ctl physics
```
