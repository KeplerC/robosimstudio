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

A canvas bag lies with its mouth toward the right hand and its top layer's lip turned up: drag the right hand down until its open jaws straddle the lip, press **Close**, drag it up 80 mm and back over the bag, and the mouth opens.

Nothing is pinned: the lip comes up because the pads squeeze it, about 0.3 N a side, and
nothing holds the bag down; its closed end moved 5 mm. Closed 25 mm over the bench and drawn
60 mm back, the mouth read 75-77 mm in seven drives; closed 12-30 mm over it, 66-90 mm; closed
over the lip's top, 45 mm up, the hand lifted nothing and the bag read shut (`f_bag.py`, GPUs
1 and 3). The bag is one SuperDex shell that collides with itself: its layers rest 3.9 mm apart,
a contact radius, and stayed 3.5 mm apart while the hand opened it. Set `self_contact` to
False and the top layer lies in the bottom one instead: -0.24 mm at rest, -0.29 mm held open.
The grip is stiffer than the YAM's own servo, 8 N a finger: at its own gains the pads squeezed
the lip with 0.02-0.05 N, it slipped, and the mouth opened 64-67 mm. The **Bag** folder reads
`mouth_opening`, the widest span across the rim and the area it encloses, and `layer_gap`.
Through the page, 11 ms per control step at rest and 41 ms while the pads hold the cloth (RTX
PRO 6000, GPU 1). Needs the `[superdex]` extra (Python 3.12).

```bash
uv run --extra superdex examples/16_bags/00_open_a_bag.py
```

Source: [`00_open_a_bag.py`](00_open_a_bag.py)

## Open a bag, scored

The same bag and hand as a task document, with a goal on how far the mouth stands open: turn the right hand to point down, open the bag as in `00_open_a_bag`, and the goal turns true at 30 mm.

The bag is a `Bag.from_spec` record the document places under its own name, the physics is
`rss.BAG_60HZ` under a task-language name, and the staging states the stiffer finger servo the
pinch needs. The hands start at the YAM's home, pointing ahead, so turn the right gizmo down
first. `tests/test_open_a_bag_scored` drives the right hand through the page and `evaluate`
says success, the mouth 75 mm open; 12 ms per control step at rest through the page on an
RTX PRO 6000, GPU 3. Needs the `[superdex]` extra (Python 3.12).

```bash
uv run --extra superdex examples/16_bags/01_open_a_bag_scored.py
```

Source: [`01_open_a_bag_scored.py`](01_open_a_bag_scored.py)

<!-- examples:end -->
