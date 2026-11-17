# Gallery

<!-- written by scripts/gallery.py from gallery/index.toml; edit the manifest, not this page -->

Tasks built with RoboSimStudio, by project. Every clip is hosted on Hugging Face, in the dataset
[keplerccc/robosimstudio-agentic-real2sim](https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim); a card links to its clip.

| Status | Meaning |
|---|---|
| verified | runs on current main and passes the harness checks; the commit it was checked on is given |
| partial | runs on current main, but the task is not completed yet or a check fails; the note says which |
| video | a video from older code, not yet ported to current main |
| scene | the scene is rebuilt, but no robot task in it is filmed yet |

| Project | Entries | Verified | |
|---|---|---|---|
| [Real to sim](robosimstudio-real2sim/README.md) | 14 | 0 | <img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/67e7b1729973344fc1978ed495bc978ca2019ece/gallery/robosimstudio-real2sim/kitchen/media/poster.jpg" width="160" alt=""> |
| [ABC-130k twins](abc130/README.md) | 8 | 0 | <img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/67e7b1729973344fc1978ed495bc978ca2019ece/gallery/abc130/load-the-cups-into-the-dish-rack/media/poster.jpg" width="160" alt=""> |
| [VABAR](vabar/README.md) | 42 | 0 | <img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/67e7b1729973344fc1978ed495bc978ca2019ece/gallery/vabar/tool-sorting-hammer/media/poster.jpg" width="160" alt=""> |
| [Bench tasks](rgbs/README.md) | 6 | 0 | <img src="https://huggingface.co/datasets/keplerccc/robosimstudio-agentic-real2sim/resolve/67e7b1729973344fc1978ed495bc978ca2019ece/gallery/rgbs/bar-wipe-with-sponge/media/poster.jpg" width="160" alt=""> |

An entry's files (its clips, and for a ported scene its meshes and references) come down with:

```bash
uv run python -m robosimstudio.hub fetch gallery/<project>/<entry>
```
