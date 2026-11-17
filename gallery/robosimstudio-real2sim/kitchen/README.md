# Kitchen counter: sweep up coffee grounds and beans

<!-- written by scripts/gallery.py from gallery/index.toml; edit the manifest, not this page -->

Spilled coffee grounds (8,638 MPM particles) and 30 beans on a kitchen counter rebuilt from the photos. One YAM hand pinches a dustpan, the other sweeps with a wooden pusher, then the pan is lifted 105 mm: 89.8% of the grounds end in the pan, and a trail of grounds stays along the sweep path. Check and every physics replay pass on current main.

<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/515584e0cc4963f09af93a6e6b56053b29d895e4/gallery/robosimstudio-real2sim/kitchen/media/sweep.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/515584e0cc4963f09af93a6e6b56053b29d895e4/gallery/robosimstudio-real2sim/kitchen/media/sweep.webp" width="49%" alt="sweep"></a>
<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/515584e0cc4963f09af93a6e6b56053b29d895e4/gallery/robosimstudio-real2sim/kitchen/media/wide.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/515584e0cc4963f09af93a6e6b56053b29d895e4/gallery/robosimstudio-real2sim/kitchen/media/wide.webp" width="49%" alt="wide"></a>

**Status:** verified on main (a2a4f53) · **robot:** YAM  
**Source:** Twin built by a coding agent from two Pexels photos; ported to current main by Codex and accepted by a reviewer agent [1](https://www.pexels.com/photo/10568265/) [2](https://www.pexels.com/photo/10568268/)  
**Licence:** Photo: Pexels licence. Clips and simulation: Apache-2.0.

Its meshes, textures and reference photos are on the Hub; fetch them, then open the live page and play
the routine (`REPORT.md` says how the scene was built and what it measures):

```bash
uv run python -m robosimstudio.hub fetch gallery/robosimstudio-real2sim/kitchen
cd gallery/robosimstudio-real2sim/kitchen
uv run ./ctl up               # the live page; it prints its address
uv run ./ctl play routine --wait
uv run ./ctl check && uv run ./ctl physics
```
