# Water

Water that keeps its volume, poured by the robot's hand.
[`robosimstudio_materials/water.py`](../../robosimstudio_materials/water.py) is
the kind behind `rss.water`: particles on Genesis's SPH (weakly compressible,
with viscosity and surface tension), which runs as a host engine beside the
Newton islands through its legacy coupler
([`robosimstudio/engines/genesis/sph.py`](../../robosimstudio/engines/genesis/sph.py)).
Newton's MPM water loses volume at tabletop scale, which is why
[`11_liquid`](../11_liquid/README.md) pours honey; this water stands in a glass at
the height its particle count fills. `fill_level` reads the share of the water in
a container and the height it stands there; the shared particle measures
(`fill_fraction`, `spill_fraction`) read it too. The cup is a rigid body on the
arm's island, and the water's weight comes back onto it.

Genesis lives in a virtual environment of its own, beside the checkout's, because
installing the `[genesis]` extra moves twenty packages. From the repository root:

```bash
uv export --frozen --extra genesis --extra dev --no-group rtx --no-emit-project --no-hashes > /tmp/genesis-req.txt
uv venv --python 3.12 /tmp/gvenv && uv pip install --python /tmp/gvenv/bin/python -r /tmp/genesis-req.txt
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 QD_VISIBLE_DEVICE=0 /tmp/gvenv/bin/python examples/21_water/00_pour_water.py
```

Without Genesis the scripts stop at the build and name the extra.

<!-- examples:start: written by scripts/example_readmes.py from each script's docstring -->

## Pour water into a glass

![Pour water into a glass](media/00_pour_water.webp)

A cup of water under the right hand and a glass beside it: drag the hand down until its open jaws straddle the cup, close on it, drag it up to 170 mm over the bench and 125 mm across, then turn the gizmo 120 degrees about the axis pointing away from the robot, and the water runs into the glass past its red line.

The water is Genesis's SPH (`rss.water` under `rss.WATER_60HZ`), which keeps its volume where Newton's MPM
water loses it; it runs where the `[genesis]` extra is installed, in a virtual environment of its own (this
folder's README). The jaws close on the cup's 66 mm walls, and the water's reaction comes back onto the cup:
carried still, it reads 1.09 N where the water weighs 1.085 N. The wrist stops at 115 degrees: in three drives
99.2 to 99.9% of the 1,728 particles ended in the glass, standing 29.4 to 29.7 mm where their count fills 30.5 to
30.7 mm, 0.1 to 0.8% on the bench (`tests/test_pour_water`'s gesture, GPU 1, RTX PRO 6000). Turned 60 degrees
it pours nothing; carried 100 mm beside the glass, 99.2% lands on the bench. 24 ms per control step.

```bash
uv run examples/21_water/00_pour_water.py
```

Source: [`00_pour_water.py`](00_pour_water.py)

## Pour water, scored

The same cup, water and glass as a task document, with a goal on the water's level in the glass: pour as in `00_pour_water`, and the goal turns true at the 25 mm line.

The water is a `Water.from_spec` record the document places under its own name, the cup a
`Body.from_spec` object, the glass's walls and its painted line `Body.from_spec` fixtures, and
the physics `rss.WATER_60HZ` under a task-language name. The region is the glass's inside, from
the bench top to its 55 mm rim, and `fill_level` reads the height the water stands there. The
hands start at the YAM's home, pointing ahead. `tests/test_pour_water` drives the same pour
through the right arm's panel and `evaluate` says success, the water 29.5 mm deep in both of two
drives, 99.3 and 99.7% of it in the glass; 25 ms per control step on an RTX PRO 6000, GPUs 1 and 3.
The `[genesis]` extra lives in a virtual environment of its own (this folder's README).

```bash
uv run examples/21_water/01_pour_water_scored.py
```

Source: [`01_pour_water_scored.py`](01_pour_water_scored.py)

<!-- examples:end -->
