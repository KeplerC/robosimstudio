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

![Fold a sheet of paper](media/00_fold_paper.webp)

An A4 sheet of copy paper between two YAM arms: close both hands, press the left one down on the far end, put the right one on the near edge and tick **Edge on the right hand**, carry the edge up, across and down onto the middle, stamp the fold flat with the left hand, untick, and the sheet stays folded in thirds.

The carry is an attachment grasp and the page says so: ticked, the row along the near edge is
pinned at the right hand's frame, because the YAM's pads cannot pinch a sheet lying flat, nor
one hanging over the bench's edge, where the inner finger cannot get under the drape. The rest
is contact. Unheld, the sheet slides 0.29 m along the bench ahead of its own fold; the left hand
pressing the far end keeps it within 2 mm. The crease is the left hand's: the flat underside of
its closed fingers stamps the loop at three places along the fold while the right hand still holds
the edge down, and the hinges bent past the yield keep 2.3 to 3.1 rad over three drives
(`foldletter.py`, GPU 7); pressed after the edge is let go, the loop rolls open instead (1.4). The
**Crease** folder reads `crease_angle` through the fold line and `fold_extent` (the sheet's span
over its flat span). Set `yield_angle` to 0 under **Material** and the same fold springs open.

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
