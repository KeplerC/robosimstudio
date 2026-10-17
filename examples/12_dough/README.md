# Dough

Dough, plasticine, clay and butter: a plastic solid on Newton's implicit MPM
that keeps what a rolling pin or a gripper does to it.
[`robosimstudio_materials/dough.py`](../../robosimstudio_materials/dough.py)
is the kind behind `rss.dough`; the shared implementation is
[`robosimstudio_materials/mpm/`](../../robosimstudio_materials/mpm/).

<!-- examples:start: written by scripts/example_readmes.py from each script's docstring -->

## Roll the dough flat

A block of dough under the right hand and a rolling pin beside it: drag the hand down onto the pin, close on it, and roll it across the dough.

Dough here is a plastic solid with a 1 kPa yield stress: it keeps what the pin
does to it. The pin is a capsule on the robot's island (`rss.DOUGH_60HZ`),
so the hand can hold it. The panel reads the dough's footprint against the one
it settled into and its height at the 95th percentile of its particles.
24.1 ms per control step on an RTX PRO 6000, graph-captured: the dough table
runs 150 solver iterations, three times the heap's, so the lump holds its
height while nothing presses it.

```bash
uv run examples/12_dough/00_roll_dough.py
```

Source: [`00_roll_dough.py`](00_roll_dough.py)

## Press a dough ball

A puck of dough under the right hand: close the gripper, drag the hand down into the dough and lift it away again.

What a plastic solid keeps is the measure: the press probe behind the `dough`
preset kept 83% of a 15 mm compression after the plate lifted (plasticine 75%,
clay 66%). The panel reads the puck's height as a share of the height it
settled at, and its footprint against the settled one. Change `material` to
`"plasticine"` or `"clay"` for a stiffer lump. 23.4 ms per control step on
an RTX PRO 6000, graph-captured.

```bash
uv run examples/12_dough/01_press_dough.py
```

Source: [`01_press_dough.py`](01_press_dough.py)

<!-- examples:end -->
