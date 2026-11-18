# Spilled rice: sweep it into a dustpan

<!-- written by scripts/gallery.py from gallery/index.toml; edit the manifest, not this page -->

6,420 grains of spilled rice as live MPM particles. One YAM hand pinches a dustpan, the other sweeps the rice in with a wooden pusher, then the pan is lifted and tipped toward the camera: 79.7% of the grains end in the pan. Check and every physics replay pass on current main.

<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/3ca1d941efb520209f4bdd9106d4407ce1b8b0de/gallery/robosimstudio-real2sim/rice/media/sweep.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/3ca1d941efb520209f4bdd9106d4407ce1b8b0de/gallery/robosimstudio-real2sim/rice/media/sweep.webp" width="49%" alt="sweep"></a>
<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/3ca1d941efb520209f4bdd9106d4407ce1b8b0de/gallery/robosimstudio-real2sim/rice/media/wide.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/3ca1d941efb520209f4bdd9106d4407ce1b8b0de/gallery/robosimstudio-real2sim/rice/media/wide.webp" width="49%" alt="wide"></a>

**Status:** verified on main (c3c0f3c) · **robot:** YAM  
**Source:** Twin built by a coding agent from a Pexels photo; ported to current main by Codex and accepted by a reviewer agent [1](https://www.pexels.com/photo/8108179/)  
**Licence:** Photo: Pexels licence. Clips and simulation: Apache-2.0.

Its meshes, textures and reference photos are on the Hub; fetch them, then open the live page and play
the routine (`REPORT.md` says how the scene was built and what it measures):

```bash
uv run python -m robosimstudio.hub fetch gallery/robosimstudio-real2sim/rice
cd gallery/robosimstudio-real2sim/rice
uv run ./ctl up               # the live page; it prints its address
uv run ./ctl play routine --wait
uv run ./ctl check && uv run ./ctl physics
```
