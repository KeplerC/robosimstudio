# Workshop: scrub a paver patch with the brush

<!-- written by scripts/gallery.py from gallery/index.toml; edit the manifest, not this page -->

A YAM picks up the scrubbing brush by its handle, scrubs a patch of the paving (84% of it scrubbed, bar 80%) and stands the brush back within 1.8 mm, in the workshop rebuilt from the photos. Check and the physics replays pass on current main.

<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/4ee035a510543a5e281ab7567bc14a932bfec32c/gallery/robosimstudio-real2sim/workshop/media/scrub.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/4ee035a510543a5e281ab7567bc14a932bfec32c/gallery/robosimstudio-real2sim/workshop/media/scrub.webp" width="49%" alt="scrub"></a>
<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/4ee035a510543a5e281ab7567bc14a932bfec32c/gallery/robosimstudio-real2sim/workshop/media/wide.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/4ee035a510543a5e281ab7567bc14a932bfec32c/gallery/robosimstudio-real2sim/workshop/media/wide.webp" width="49%" alt="wide"></a>

**Status:** verified on main (0bd56ef) · **robot:** YAM  
**Source:** Twin built by a coding agent from two Unsplash photos; ported to current main by Codex and accepted by a reviewer agent [1](https://unsplash.com/photos/mHdaRVIBq0g) [2](https://unsplash.com/photos/4v0mqRgdzVI)  
**Licence:** Photo: Unsplash licence. Clips and simulation: Apache-2.0.

Its meshes, textures and reference photos are on the Hub; fetch them, then open the live page and play
the routine (`REPORT.md` says how the scene was built and what it measures):

```bash
uv run python -m robosimstudio.hub fetch gallery/robosimstudio-real2sim/workshop
cd gallery/robosimstudio-real2sim/workshop
uv run ./ctl up               # the live page; it prints its address
uv run ./ctl play routine --wait
uv run ./ctl check && uv run ./ctl physics
```
