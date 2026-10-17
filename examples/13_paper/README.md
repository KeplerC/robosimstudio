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

![Close a cardboard flap](media/01_box_flap.webp)

A carton-board flap standing open on the rim of a box and the left hand pointing down beside it: press **Close**, drag the hand toward the box to push the flap over, down to press it into the box, and up again, and it stays shut.

The box is four slabs; the flap is a strip of CARDBOARD `pinned` on the far rim, standing up
and leaning out 15 degrees as an opened carton's does, because a hand cannot lift one lying out
level: the YAM's closed fingers stand 100 mm over their tips, the height of the box. Pushed over
and pressed down near the hinge until its free edge is 74 mm under the rim, the hinge rows bend
past the yield and keep 1.9 rad, so the flap comes back level when the hand lifts (5 mm under);
pressed near its free end the board bends along its length instead, the hinge keeps 1.3 rad
and the flap springs back up 26 degrees (`flapdrive.py`, GPU 7, RTX PRO 6000). Board this stiff
needs VBD at 20 iterations instead of 5 (on the drape bench a 100 mm flap sags 19 % of its
length at 5 and 5 % at 20), 40 ms per control step. The **Hinge** readout is `crease_angle`.

```bash
uv run examples/13_paper/01_box_flap.py
```

Source: [`01_box_flap.py`](01_box_flap.py)

<!-- examples:end -->
