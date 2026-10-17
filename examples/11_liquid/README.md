# Liquid

Honey, ketchup and water: a pool of particles on Newton's implicit MPM, poured
from a cup or pushed by a wiper.
[`robosimstudio_materials/liquid.py`](../../robosimstudio_materials/liquid.py)
is the kind behind `rss.liquid`; its docstring states the two Newton 1.6
caveats these examples are built around (the water preset loses volume; thin
walls leak without a margin), and the shared implementation is
[`robosimstudio_materials/mpm/`](../../robosimstudio_materials/mpm/).

<!-- examples:start: written by scripts/example_readmes.py from each script's docstring -->

## Pour honey into a bowl

A cup of honey standing beside a bowl: click the cup, drag it over the bowl and tip it with the gizmo's rotation ring.

Honey rather than water, because Newton 1.6's water preset loses volume at
tabletop scale (see `robosimstudio_materials.liquid`). The cup is one rigid
body with 8 mm walls -- two voxels -- and is named in the honey's `containers`,
which gives its shapes the half-voxel margin a liquid needs to stay inside under
graph capture; `rss.LIQUID_120HZ` keeps the cup on the robot's island, turns
the granular projection off and steps the MPM twice per control step, because
honey falling from the lip to the bowl is faster after 7 mm than a 60 Hz step
can catch at a floor (poured from 100 mm, 60 Hz lost 85 of these 8,000
particles through the bowl and the bench and put 4% in the bowl; 120 Hz lost
none and put 42%). The panel counts the honey in the bowl. 30.0 ms per
control step for 8,000 particles on an RTX PRO 6000, graph-captured.

```bash
uv run examples/11_liquid/00_pour_honey.py
```

Source: [`00_pour_honey.py`](00_pour_honey.py)

## Wipe a spill

A ketchup puddle under the right hand and a rubber-faced wiper beside it: drag the hand down onto the wiper, close on it, and push the puddle off its patch.

Ketchup is a Bingham fluid here (a 300 Pa yield stress, viscosity 100), so the
puddle holds its edge until the wiper moves it. The panel reads how much of it
has been pushed off the patch it started on and lies on the bench
(`spill_fraction`), and how far it has spread. A deformable sponge in place
of the rigid wiper is a later phase. The puddle is only pushed, never dropped,
so it would hold at 60 Hz; it runs on the one liquid table, `rss.LIQUID_120HZ`,
at 26.0 ms per control step on an RTX PRO 6000, graph-captured.

```bash
uv run examples/11_liquid/01_wipe_spill.py
```

Source: [`01_wipe_spill.py`](01_wipe_spill.py)

<!-- examples:end -->
