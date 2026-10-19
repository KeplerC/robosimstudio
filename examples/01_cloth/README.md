# Cloth

How a sheet is built, section by section:
[`robosimstudio_materials/cloth.py`](../../robosimstudio_materials/cloth.py), the
implementation behind `rss.sheet`.

A gripper holds a sheet by friction alone, which needs one-way coupling and
full-surface contact (`00_towel.py`, `04_your_own_cloth.py`). On a flat sheet,
lower the fingertips to the bench, or the pads close over the edge. The arm
feels no force from the cloth, so a fully shut hand passes through it.

The coupler can still read that force: `coupler("one_way", contact_forces=True)`
has Newton 1.7's VBD report what each contact pushes on its shape, and
`contact_force(sim, "right")` sums it over a hand (`03_scrub_a_stain.py`). It
reads a press, from the first touch on; it does not resolve a resting sheet's
own weight.

![A click on the cloth pins a patch to a handle. Lift it, carry it, let go.](media/pinch_towel.webp)

![Two holds on two corners with Move together on: one drag carries the sheet flat.](media/move_together.webp)

<!-- examples:start: written by scripts/example_readmes.py from each script's docstring -->

## Pinch a towel

![Pinch a towel](media/00_towel.webp)

A terry towel on a bench, its edge under an open hand: drag the hand down onto the edge, close it and lift.

Drag the hand down until the fingertips reach the bench, press **Close** and lift: the pads
hold the edge by friction. That needs one-way coupling and full-surface contact; under the
default physics a hand can push a sheet but not pinch it. Or click the cloth to pin a patch to
a handle; two handles with `Move together` carry it by two corners.

```bash
uv run examples/01_cloth/00_towel.py
```

Source: [`00_towel.py`](00_towel.py)

## Fabrics

![Fabrics](media/01_fabrics.webp)

The same square of cloth in four shipped fabrics, each draped over its own post.

Each sheet takes a `rss.ClothMaterialSpec` (stretch and bending stiffness,
weight per area, thickness) from `rss.materials`. Along +y, limp to stiff: jersey tee, packing
tape, sweatshirt fleece, terry towelling. In the studio, the **Material** folder's `Cloth` picks
one and its sliders change it while it hangs.

```bash
uv run examples/01_cloth/01_fabrics.py
```

Source: [`01_fabrics.py`](01_fabrics.py)

## Cloth on another engine

![Cloth on another engine](media/02_engines.webp)

The same sheet on Newton, SuperDex or Genesis: change one string at the top of `main()` and run it again.

The spec is unchanged and the arm stays on MuJoCo; the printed island table names the solver
each thing runs on. `newton` runs on a base install; `[superdex]` (Python 3.12) and
`[genesis]` are optional extras.

```bash
uv run examples/01_cloth/02_engines.py
```

Source: [`02_engines.py`](02_engines.py)

## Scrub a stain

![Scrub a stain](media/03_scrub_a_stain.webp)

A towel with a stain along its middle under the right hand: press **Close**, drag the hand down onto the stain until **push** reads between 20 and 60 N, and drag it along the stain: only what it rubbed inside that band is wiped.

The push is what the towel puts on the hand's colliders: Newton 1.7's VBD contact force, read through
`coupler("one_way", contact_forces=True)`, while the arm still goes where it is dragged. It reads zero
until the fingers touch, then, stroking, 32 N with the fingertips 6 mm over the bench and 116 N at 4 mm,
which the bench's own reading takes up to within a newton (`tests/test_scrub_a_stain.py`, GPU 5). A
stroke that only brushes the stain wipes nothing, one pressed past 60 N marks what it rubs **too hard**
until the hand comes off, and pressing or lifting without sliding counts for nothing. Ease the hand up as
it leaves the stain: stopped dead, the towel it pushed ahead presses back. The stain is drawn brown, red
where pressed too hard, gone where wiped. Contact forces resolve a press, not a resting weight.

```bash
uv run examples/01_cloth/03_scrub_a_stain.py
```

Source: [`03_scrub_a_stain.py`](03_scrub_a_stain.py)

## Your own cloth

![Your own cloth](media/04_your_own_cloth.webp)

A second implementation of cloth, written in this file and draped over a ball beside a Panda.

`MyCloth` adds a grid of particles and triangles through Newton's builder, and
`ParticleMixin` handles pinning, reset and reads. A stock `rss.ClothSpec`
with `entity="my_cloth"` selects it. Click the sheet to pin and drag a patch, or drag the hand
onto the hem, set **Grip amount** to 0.95 and lift; **Close** squeezes the sheet out. Limits:
flat grid only; the fabric's thickness and contact values are ignored.

```bash
uv run examples/01_cloth/04_your_own_cloth.py
```

Source: [`04_your_own_cloth.py`](04_your_own_cloth.py)

<!-- examples:end -->
