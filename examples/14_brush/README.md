# Brush

A handle over a field of short stiff bristles, scrubbing the bench.
[`robosimstudio_materials/brush.py`](../../robosimstudio_materials/brush.py) is the
kind behind `rss.brush`: each bristle is a chain of capsules on the rope's rod
joints, welded under a sole of strips, and a lattice of 5 mm cells records where
the tips have been while down, which `scrubbed` reads for a patch. Two materials
ship, `nylon` (a dish brush) and `soft` (a paint brush). The hand holds the brush
by a named handle on its own solver (`rss.brush(handle=...)` under
`rss.BRUSH_60HZ`); the module's docstring says why and what that costs.

<!-- examples:start: written by scripts/example_readmes.py from each script's docstring -->

## Scrub a patch clean

A nylon dish brush stands on the bench under the right hand, which points down at its handle, beside a marked square: drag the hand down onto the handle, close, press the bristles to the bench and scrub the square.

The brush is 400 bristles of four capsules each on VBD, rooted in a sole of
twenty strips, under a block on the arm's solver that the hand closes on
(`rss.BRUSH_60HZ`); the field cannot hold that block up across the two
solvers, so it is gravity-compensated, and the arm does not feel the bristles
push back. The panel reads the share of the square's 5 mm cells a bristle tip
has touched while down on the bench. `tests/test_scrub_bench` closes the hand,
presses 4 mm and makes three strokes across the square: all of its cells
scrubbed, every tip down throughout, the handle slipping 0.5 mm in the
fingers, at 22.4 ms per control step on an RTX PRO 6000, GPU 3,
graph-captured.

```bash
uv run examples/14_brush/00_scrub_the_bench.py
```

Source: [`00_scrub_the_bench.py`](00_scrub_the_bench.py)

## Scrub a patch, scored

The same brush, handle and square as a task document, with a goal on the share of the square the bristle tips have scrubbed: turn the hand to point down, close on the handle, scrub, and the goal turns true at half.

The brush is a `Brush.from_spec` record and its handle a `Body.from_spec`
one, placed by the document under the names the brush's `handle` and the
goal use, and the physics is the brush arrangement under a task-language name.
The region is the square, and `scrubbed` reads its 5 mm cells off the
lattice the brush keeps on the device; a reset clears it. `tests/test_scrub_bench`
drives the hand through the same three strokes from the robot's own home and
the goal turns true, at 1.0 (22.3 ms per control step, GPU 3).

```bash
uv run examples/14_brush/01_scrub_scored.py
```

Source: [`01_scrub_scored.py`](01_scrub_scored.py)

<!-- examples:end -->
