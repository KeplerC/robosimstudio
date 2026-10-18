# Bags

Cloth that touches itself.
[`robosimstudio_materials/bag.py`](../../robosimstudio_materials/bag.py) is the
implementation behind `rss.bag`: two layers of cloth sewn on three edges, one
surface that SuperDex simulates as a shell colliding with itself, so the layers
lie a contact radius apart and do not pass through each other. One layer runs
on past the mouth as a turned-up lip, the tab a hand takes hold of. One fabric
ships, `rss.bags.CANVAS`, a canvas pouch stiff enough that its lip stands;
`mouth_opening` reads how far the mouth stands open and `layer_gap` whether the
layers stayed apart. Every example needs the `[superdex]` extra (Python 3.12).

<!-- examples:start: written by scripts/example_readmes.py from each script's docstring -->

## Open a bag

![Open a bag](media/00_open_a_bag.webp)

A canvas bag lies with its mouth toward the right hand and its top layer's lip turned up: drag the left hand down onto the bag's closed end, drag the right hand down until its open jaws straddle the lip, press **Close**, and drag it up and back over the bag, and the mouth opens.

Nothing is pinned: the lip comes up because the pads hold it, and the left hand's closed fingertips
keep the bag from sliding after it (its closed end slid 15-22 mm under them, 36 mm without). Over
eight drives the mouth opened 54-91 mm, the pads closed 20-30 mm over the bench and up to 4 mm
either side of the lip (`p9_drive.py`, GPUs 3 and 5), and 33-74 mm in four filmed takes; closed
15 mm over it the pads take both layers and drag the bag. The bag is one SuperDex shell that collides with itself: its layers rest 3.9 mm
apart, a contact radius, and stayed at least 2.6 mm apart while the hand opened it. Set
`self_contact` to False and the same gesture shows them through each other: -0.24 mm at rest,
-5.2 mm held open. The grip is stiffer than the YAM's own servo, 8 N a finger: at its own gains the
pads pressed the lip with 0.1 N and it slid out. The **Bag** folder reads `mouth_opening`, the
widest span across the rim and the area it encloses, and `layer_gap`. About 9 ms per control
step at rest and 40-60 ms while the pads press the cloth (RTX PRO 6000). Needs the `[superdex]`
extra (Python 3.12).

```bash
uv run --extra superdex examples/16_bags/00_open_a_bag.py
```

Source: [`00_open_a_bag.py`](00_open_a_bag.py)

## Open a bag, scored

The same bag and hands as a task document, with a goal on how far the mouth stands open: turn both hands to point down, open the bag as in `00_open_a_bag`, and the goal turns true at 30 mm.

The bag is a `Bag.from_spec` record the document places under its own name, the physics is
`rss.BAG_60HZ` under a task-language name, and the staging states the stiffer finger servo the
pinch needs. The hands start at the YAM's home, pointing ahead, so turn each gizmo down first.
`tests/test_open_a_bag_scored` drives both hands through the page and `evaluate` says success,
the mouth 60 mm open; 8.5 ms per control step at rest on an RTX PRO 6000, GPU 3. Needs the
`[superdex]` extra (Python 3.12).

```bash
uv run --extra superdex examples/16_bags/01_open_a_bag_scored.py
```

Source: [`01_open_a_bag_scored.py`](01_open_a_bag_scored.py)

<!-- examples:end -->
