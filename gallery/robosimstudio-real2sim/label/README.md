# Box: stick an RSS shipping label on it

<!-- written by scripts/gallery.py from gallery/index.toml; edit the manifest, not this page -->

A YAM picks an RSS shipping label up by its pull tab, lays it inside the dashed outline on a parcel in a workshop and presses it down in nine strokes; the label is tape and its own adhesive makes the bond. On current main it ends 100% bonded, 1.3 mm off centre and 2.3 degrees off square, with the parcel moved 1.3 mm. Clips play at 4x. Check and the physics replays pass, except that the unwatched replay loses the tab at the pickup (a contact-buffer overflow there), so the pickup does not always repeat.

<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/3ca1d941efb520209f4bdd9106d4407ce1b8b0de/gallery/robosimstudio-real2sim/label/media/label.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/3ca1d941efb520209f4bdd9106d4407ce1b8b0de/gallery/robosimstudio-real2sim/label/media/label.webp" width="49%" alt="label"></a>
<a href="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/3ca1d941efb520209f4bdd9106d4407ce1b8b0de/gallery/robosimstudio-real2sim/label/media/wide.mp4"><img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/3ca1d941efb520209f4bdd9106d4407ce1b8b0de/gallery/robosimstudio-real2sim/label/media/wide.webp" width="49%" alt="wide"></a>

**Status:** verified on main (3f3970f) · **robot:** YAM  
**Source:** Designed scene, no reference photo; ported to current main by Codex and accepted by a reviewer agent  
**Licence:** Apache-2.0

Its meshes, textures and reference photos are on the Hub; fetch them, then open the live page and play
the routine (`REPORT.md` says how the scene was built and what it measures):

```bash
uv run python -m robosimstudio.hub fetch gallery/robosimstudio-real2sim/label
cd gallery/robosimstudio-real2sim/label
uv run ./ctl up               # the live page; it prints its address
uv run ./ctl play routine --wait
uv run ./ctl check && uv run ./ctl physics
```
