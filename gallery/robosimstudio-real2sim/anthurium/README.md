# Anthurium: water the plant

<!-- written by scripts/gallery.py from gallery/index.toml; edit the manifest, not this page -->

A YAM picks up a cup of water by its rim in the room rebuilt from the photo, pours a stream onto the anthurium's large outer leaf, which swings under it and springs back, then pours the rest into the pot and turns the cup upright again. On current main 85% of the water ends in the pot and 15% on the table, none below it; the stream off the leaf runs onto the table, not into the pot. Check passes and the physics replays show no FAIL, but with the cup twice as heavy its pour pose is out of reach.

<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/3ca1d941efb520209f4bdd9106d4407ce1b8b0de/gallery/robosimstudio-real2sim/anthurium/media/water.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/3ca1d941efb520209f4bdd9106d4407ce1b8b0de/gallery/robosimstudio-real2sim/anthurium/media/water.webp" width="49%" alt="water"></a>

**Status:** verified on main (3f3970f) · **robot:** YAM  
**Source:** Twin built by a coding agent from an Unsplash photo; ported to current main by Codex and accepted by a reviewer agent [1](https://unsplash.com/photos/eyR5MNVAMQw)  
**Licence:** Photo: Unsplash licence. Clips and simulation: Apache-2.0.

Its meshes, textures and reference photos are on the Hub; fetch them, then open the live page and play
the routine (`REPORT.md` says how the scene was built and what it measures):

```bash
uv run python -m robosimstudio.hub fetch gallery/robosimstudio-real2sim/anthurium
cd gallery/robosimstudio-real2sim/anthurium
uv run ./ctl up               # the live page; it prints its address
uv run ./ctl play routine --wait
uv run ./ctl check && uv run ./ctl physics
```
