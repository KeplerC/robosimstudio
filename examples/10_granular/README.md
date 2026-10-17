# Granular

Rice, sand, beans and flour: a heap of particles on Newton's implicit MPM,
pushed by a rigid tool in the robot's hand.
[`robosimstudio_materials/granular.py`](../../robosimstudio_materials/granular.py)
is the kind behind `rss.heap`; the shared implementation is
[`robosimstudio_materials/mpm/`](../../robosimstudio_materials/mpm/).

<!-- examples:start: written by scripts/example_readmes.py from each script's docstring -->

## Scoop rice into the tray

A heap of rice under the right hand, a paddle beside it and a tray of four slabs: drag the hand down onto the paddle, close on it, and push the rice into the tray.

The rice is 27,648 particles on Newton's implicit MPM, stepped once per 60 Hz
control step beside the YAM (`rss.GRANULAR_60HZ`: 16.6 ms per control step on
an RTX PRO 6000, graph-captured, measured with this script). The paddle is
12 mm thick because a tool thinner than two 4 mm voxels keeps rice inside it as
it pushes. The panel reads the share of the rice inside the tray's box.

```bash
uv run examples/10_granular/00_scoop_rice.py
```

Source: [`00_scoop_rice.py`](00_scoop_rice.py)

## Scoop rice, scored

The same rice, paddle and tray as a task document, with a goal on the share of rice in the tray: scoop, and the goal turns true at half.

The heap is a `Heap.from_spec` record the document places under its own name,
the tray walls are `Body.from_spec` fixtures, and the physics is the granular
arrangement under a task-language name. `(:z-range 0 0.03)` is the tray's
depth; a region's default window is 150 mm, which would count rice carried
over the tray in the paddle. 16.7 ms per control step on an RTX PRO 6000,
graph-captured.

```bash
uv run examples/10_granular/01_scoop_rice_scored.py
```

Source: [`01_scoop_rice_scored.py`](01_scoop_rice_scored.py)

<!-- examples:end -->
