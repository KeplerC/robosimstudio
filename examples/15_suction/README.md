# Suction

A vacuum cup on the robot's wrist, for the flat things a parallel jaw cannot
pinch. The hand is
[`robosimstudio/robots/grippers/suction.py`](../../robosimstudio/robots/grippers/suction.py),
a 30 mm cup with its own MJCF that bolts to any imported wrist;
`yam_bimanual_suction` carries it on the YAM's right arm and keeps the left
hand's jaws. The seal and the hold are
[`robosimstudio/core/suction.py`](../../robosimstudio/core/suction.py): the cup
seals when its face is within 2 mm of a flat face of the part and square to it
within 15 degrees, and holds by a MuJoCo weld where the part shares the arm's
solver, else by a spring that lets go past the vacuum's pull. The arm's panel
has a **Vacuum** box and a **Seal** readout; a graph switches the same cup with
`sim.vacuum`.

<!-- examples:start: written by scripts/example_readmes.py from each script's docstring -->

## Pick up a plate

A plate lying flat under the right hand's suction cup and a square 200 mm toward the middle of the bench: drag the hand down onto the plate, tick **Vacuum** in the right arm's panel, lift, carry it over the square and untick.

The plate is 150 by 150 by 6 mm and 0.3 kg, flat on the bench, so a parallel
jaw has nothing to close across; `yam_bimanual_suction` carries a vacuum cup
on its right arm instead, its face 100 mm over the bench and pointing down at
home, and keeps the left hand's jaws. The cup seals when its face is within
2 mm of the plate and square to it within 15 degrees, and the arm's **Seal**
readout says sealed or not, the gap and the mass held; the panel stops a few
millimetres short of where the gizmo is put, so put it a little into the plate.
The arm and the plate share one MuJoCo solver (`rss.MUJOCO_500HZ`), so the
seal is a weld switched on inside the captured graph and the arm carries the
plate's weight. Driven through the panel, the plate rides 198 mm with the cup
less than 1 mm off where it sealed and rests in the square with its top 6 mm
over the bench, at 3.5 ms per control step (`tests/test_suction_pick`, RTX
PRO 6000, GPU 5).

```bash
uv run examples/15_suction/00_pick_a_plate.py
```

Source: [`00_pick_a_plate.py`](00_pick_a_plate.py)

## Pick up a plate, scored

The same plate, cup and square as a task document, with a goal on the plate resting in the square: carry it over, untick **Vacuum**, lift the hand away, and the goal turns true.

The plate is a `Body.from_spec` record the document places under its own
name, the robot is `yam_bimanual_suction` and the physics the one MuJoCo
island under its task-language name, `mujoco_500hz@v1`. The square's height
window, `(:z-range 0.0 0.01)`, counts the plate's centre only within 10 mm of
the bench, so a plate carried over the square does not count, and
`hands-clear` asks every tool centre to be 40 mm from the plate's, so a plate
still held down on the square does not either. `tests/test_suction_pick`
drives the hand through the pick and the goal turns true.

```bash
uv run examples/15_suction/01_pick_scored.py
```

Source: [`01_pick_scored.py`](01_pick_scored.py)

<!-- examples:end -->
