# Alocasia: water the plant from a cup

<!-- written by scripts/gallery.py from gallery/index.toml; edit the manifest, not this page -->

A YAM picks up a cup of water by its rim, pours it into the alocasia's pot and sets the cup back upright, in the dining room rebuilt from the photo; the leaves bend under the stream and spring back. On current main 88% of the water lands in the pot (about 83% below the rim in the filmed takes; bar 80%) and none goes through the table or soil. Check and the physics replays pass; with the robot frozen the water still drifts 2 mm as it settles.

<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/robosimstudio-real2sim/plant/media/water.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/robosimstudio-real2sim/plant/media/water.webp" width="49%" alt="water"></a>
<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/robosimstudio-real2sim/plant/media/wide.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/6cc45a03286f5f2cf6521c8a8d5ccb114a884ea7/gallery/robosimstudio-real2sim/plant/media/wide.webp" width="49%" alt="wide"></a>

**Status:** verified on main (3f3970f) · **robot:** YAM  
**Source:** Twin built by a coding agent from an Unsplash photo; ported to current main by Codex and accepted by a reviewer agent [1](https://unsplash.com/photos/Mfjb8E7LjWQ)  
**Licence:** Photo: Unsplash licence. Clips and simulation: Apache-2.0.

Its meshes, textures and reference photos are on the Hub; fetch them, then open the live page and play
the routine (`REPORT.md` says how the scene was built and what it measures):

```bash
uv run python -m robosimstudio.hub fetch gallery/robosimstudio-real2sim/plant
cd gallery/robosimstudio-real2sim/plant
uv run ./ctl up               # the live page; it prints its address
uv run ./ctl play routine --wait
uv run ./ctl check && uv run ./ctl physics
```
