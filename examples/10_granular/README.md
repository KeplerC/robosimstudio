# Granular

Rice, sand, beans and flour: a heap of particles on Newton's implicit MPM,
pushed by a rigid tool in the robot's hand.
[`robosimstudio_materials/granular.py`](../../robosimstudio_materials/granular.py)
is the kind behind `rss.heap`; the shared implementation is
[`robosimstudio_materials/mpm/`](../../robosimstudio_materials/mpm/).

<!-- examples:start: written by scripts/example_readmes.py from each script's docstring -->

## Sweep rice into the tray

A heap of rice under the right hand, a pusher standing on its edge beyond it and a three-walled tray on the near side: turn the hand to point down, drag it onto the pusher, close on it, and sweep the heap into the tray at a walking pace.

The rice is 16,000 particles on Newton's implicit MPM, stepped once per 60 Hz
control step beside the YAM (`rss.GRANULAR_60HZ`: 16.5 ms per control step on
an RTX PRO 6000, GPU 3, graph-captured, measured by `tests/test_sweep_rice`
on this scene). The pusher is 20 mm thick and 40 mm tall so the jaws close on
it from above, and the heap names it in `containers` so the grid sees it one
voxel fatter than the hand does; without that the bottom voxel of rice passes
under it. The tray's walls are five voxels thick and its open side faces the
heap, so the sweep ends inside it. Swept at 0.15 m/s as the test sweeps it,
93% of the rice is in the tray and the rest is on the bench, but for a grain
the squeeze shot off the bench's edge in two sweeps of thirty (the others lost
none); a tool faster than ~0.36 m/s outruns the grid. The panel reads the
share of the rice inside the tray.

```bash
uv run examples/10_granular/00_sweep_rice.py
```

Source: [`00_sweep_rice.py`](00_sweep_rice.py)

## Sweep rice, scored

The same rice, pusher and tray as a task document, with a goal on the share of rice in the tray: sweep, and the goal turns true at half.

The heap is a `Heap.from_spec` record the document places under its own name,
the tray walls are `Body.from_spec` fixtures, and the physics is the granular
arrangement under a task-language name. The region is the tray's inside;
`(:z-range -0.005 0.03)` is the walls' height, starting 5 mm under the bench
top because the bottom layer of a pushed heap rests at the top to within a
rounding, and a region's default window is 150 mm, which would count rice
carried over the tray. `tests/test_sweep_rice` drives the hand through the
sweep and the goal turns true, at 93%; 16.5 ms per control step on an RTX PRO
6000, GPU 3, graph-captured.

```bash
uv run examples/10_granular/01_sweep_rice_scored.py
```

Source: [`01_sweep_rice_scored.py`](01_sweep_rice_scored.py)

<!-- examples:end -->
