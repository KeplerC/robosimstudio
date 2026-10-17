# Soft bodies

Volumetric (tetrahedral) soft bodies.
[`robosimstudio_materials/soft.py`](../../robosimstudio_materials/soft.py) is the
implementation behind `rss.sponge`.

![The hand driven down onto a sponge: it squashes under the jaw and recovers.](media/press_sponge.webp)

<!-- examples:start: written by scripts/example_readmes.py from each script's docstring -->

## Lift a sponge

![Lift a sponge](media/00_sponge.webp)

A foam sponge under an open Panda hand: drag the hand down onto it, press **Close** and lift.

Drag the hand down until the fingertips are a centimetre over the bench, press
**Close**, and lift: friction alone holds the foam, nothing pinned. The sponge is
a volumetric FEM body stepped by SuperDex. Needs the `[superdex]` extra
(Python 3.12).

```bash
uv run --extra superdex examples/02_soft/00_sponge.py
```

Source: [`00_sponge.py`](00_sponge.py)

## Soft body from a file

![Soft body from a file](media/01_tet_mesh.webp)

A rubber O-ring read out of a tetrahedral mesh file and dropped between two arms: click it and drag.

The script writes the mesh itself so you can see the format: an `.npz` holding
`vertices` and `tets`. Anything `meshio` reads that has tetrahedra in it
works too, so point `AssetSpec(path=...)` at your own file and delete the
writer. Wind every tetrahedron so its rest volume is positive: Newton drops the
others without a word. The ring runs on Newton's VBD here, on a base install.

```bash
uv run examples/02_soft/01_tet_mesh.py
```

Source: [`01_tet_mesh.py`](01_tet_mesh.py)

## Your own soft body

![Your own soft body](media/02_your_own_soft.webp)

A soft-body implementation written in this file: a block of tetrahedra dropped onto the bench beside a Panda.

`MySoft` cuts a lattice into six tetrahedra per cell and hands them to
Newton's builder; `ParticleMixin` handles binding, reset and reading, so a
click in the studio pins a patch you can drag. The stock `rss.SoftSpec`
describes the block, and its `entity="my_soft"` field asks for this class. It
runs on Newton's VBD, so no extra is needed. On VBD the hand passes into the
block rather than pressing it (30 mm of travel dents the top 3 mm);
`00_sponge.py` is the press, on SuperDex.

```bash
uv run examples/02_soft/02_your_own_soft.py
```

Source: [`02_your_own_soft.py`](02_your_own_soft.py)

<!-- examples:end -->
