# Brush: write RSS in dust

<!-- written by scripts/gallery.py from gallery/index.toml; edit the manifest, not this page -->

A YAM picks a bristle brush up off its rest by contact, writes RSS in a 1 mm layer of dust (IoU 0.843 against the letters, every letter above 0.82) and sets the brush back on its rest within 0.4 mm, in a textured workshop. Check and all seven physics replays pass on current main.

<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/515584e0cc4963f09af93a6e6b56053b29d895e4/gallery/robosimstudio-real2sim/brush-rss/media/brush.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/515584e0cc4963f09af93a6e6b56053b29d895e4/gallery/robosimstudio-real2sim/brush-rss/media/brush.webp" width="49%" alt="brush"></a>
<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/515584e0cc4963f09af93a6e6b56053b29d895e4/gallery/robosimstudio-real2sim/brush-rss/media/close.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/515584e0cc4963f09af93a6e6b56053b29d895e4/gallery/robosimstudio-real2sim/brush-rss/media/close.webp" width="49%" alt="close"></a>

**Status:** verified on main (3c20d49) · **robot:** YAM  
**Source:** Designed scene, no reference photo; ported to current main by Codex and accepted by a reviewer agent  
**Licence:** Apache-2.0

Its meshes, textures and reference photos are on the Hub; fetch them, then open the live page and play
the routine (`REPORT.md` says how the scene was built and what it measures):

```bash
uv run python -m robosimstudio.hub fetch gallery/robosimstudio-real2sim/brush-rss
cd gallery/robosimstudio-real2sim/brush-rss
uv run ./ctl up               # the live page; it prints its address
uv run ./ctl play routine --wait
uv run ./ctl check && uv run ./ctl physics
```
