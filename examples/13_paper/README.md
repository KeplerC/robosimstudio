# Paper

A sheet whose folds stay.
[`robosimstudio_materials/paper.py`](../../robosimstudio_materials/paper.py) is the
implementation behind `rss.paper`: cloth with a bending hinge that yields, so a
fold pressed past the yield angle is kept and a gentle curl springs back. Two
materials ship, `rss.papers.PAPER` (80 g/m² copy paper) and `rss.papers.CARDBOARD`
(folding carton board); `crease_angle` reads how far a sheet stays turned
through a line.

<!-- examples:start: written by scripts/example_readmes.py from each script's docstring -->

## Fold a sheet of paper

An A4 sheet of copy paper on the bench between two YAM arms: carry a corner over, press the fold flat, let go, and it stays folded.

Click the sheet to pin a patch to a handle and carry it over the far half; click the standing
loop for a second handle and press it flat, or push it down with a hand. Press **Release all**:
the hinges bent past the yield keep their angle, so the flap stays where it was pressed instead
of springing open. The **Crease** folder reads `crease_angle` through the sheet's centreline
(how far the sheet stays turned there, in radians) and `fold_extent` (its span across that
line over its flat span). Set `yield_angle` to 0 under **Material** and the same fold springs open.

```bash
uv run examples/13_paper/00_fold_paper.py
```

Source: [`00_fold_paper.py`](00_fold_paper.py)

## Close a cardboard flap

A carton-board flap hinged on the rim of an open box: push it over with a hand and it creases at the hinge and stays shut.

The box is four slabs; the flap is a strip of CARDBOARD whose bound edge is `pinned` along
the far rim, standing out level over the bench. Drag a hand onto its free end and push it down
over the opening: the hinge row bends past the yield and keeps it, so the flap stays down when
the hand leaves. Board this stiff needs a better-converged solve than the default: VBD runs 20
iterations here instead of 5 (on the drape bench a 100 mm flap sags 19 % of its length under
its own weight at 5 iterations and 5 % at 20). The **Hinge** readout is `crease_angle` there.

```bash
uv run examples/13_paper/01_box_flap.py
```

Source: [`01_box_flap.py`](01_box_flap.py)

<!-- examples:end -->
