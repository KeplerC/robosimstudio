# Plate rack: put six plates away on a shelf

<!-- written by scripts/gallery.py from gallery/index.toml; edit the manifest, not this page -->

Two YAM arms take turns lifting six stoneware plates out of drying racks by their rims and standing them in the slots of a plate shelf, every plate within 2 degrees of vertical, in a vintage kitchen modelled from the photo. Check and the physics replays pass on current main; with the plates twice as heavy the first one tilts 5.6 degrees, so the grip has little margin.

<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/3ca1d941efb520209f4bdd9106d4407ce1b8b0de/gallery/robosimstudio-real2sim/plate-rack/media/plates.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/3ca1d941efb520209f4bdd9106d4407ce1b8b0de/gallery/robosimstudio-real2sim/plate-rack/media/plates.webp" width="49%" alt="plates"></a>
<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/3ca1d941efb520209f4bdd9106d4407ce1b8b0de/gallery/robosimstudio-real2sim/plate-rack/media/rack.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/3ca1d941efb520209f4bdd9106d4407ce1b8b0de/gallery/robosimstudio-real2sim/plate-rack/media/rack.webp" width="49%" alt="rack"></a>

**Status:** verified on main (c3c0f3c) · **robot:** YAM  
**Source:** Designed scene, staged in a kitchen built from a Pexels photo; ported to current main by Codex and accepted by a reviewer agent [1](https://www.pexels.com/photo/33375679/)  
**Licence:** Photo: Pexels licence. Clips and simulation: Apache-2.0.

Its meshes, textures and reference photos are on the Hub; fetch them, then open the live page and play
the routine (`REPORT.md` says how the scene was built and what it measures):

```bash
uv run python -m robosimstudio.hub fetch gallery/robosimstudio-real2sim/plate-rack
cd gallery/robosimstudio-real2sim/plate-rack
uv run ./ctl up               # the live page; it prints its address
uv run ./ctl play routine --wait
uv run ./ctl check && uv run ./ctl physics
```
