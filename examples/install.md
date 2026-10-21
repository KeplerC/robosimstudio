# Installation

This guide covers the recommended ways to install RoboSimStudio: with pip into an
environment of your own, or with [uv](https://docs.astral.sh/uv/) from a checkout
of the repository, which is also how the examples run. For working on
RoboSimStudio itself, see [Contributing](../CONTRIBUTING.md).

## System Requirements

### Minimum Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.12 | 3.12 is what CI tests, and the only version SuperDex ships wheels for |
| OS | Linux x86-64 | The platform RoboSimStudio is developed and tested on |
| GPU | Optional | An NVIDIA GPU with a driver that Warp 1.18 supports (see Newton's [system requirements](https://newton-physics.github.io/newton/latest/guide/installation.html#system-requirements)). Without one, scenes run on the CPU |

### Platform-Specific Requirements

**Linux**

The OVRTX path tracer (the `rtx` extra) ships wheels for glibc 2.35 or newer,
such as Ubuntu 22.04 and later.

**Linux aarch64 (ARM64)**

SuperDex ships no wheel for ARM64, so the `superdex` and `cpu` extras install
nothing there. Sponges and bags cannot be built, and on a CPU neither can cloth or
rope.

**GPUs without RT cores**

The path tracer needs RT cores: a T4, A10, A40, L4, L40S or an RTX card. On an
A100, H100 or B200, pictures are drawn by the `tiled` renderer, which draws the
colliders alone.

> [!NOTE]
> macOS and Windows are not tested.

## Installing RoboSimStudio

RoboSimStudio installs from its Git repository. It is not on PyPI, because one
of its dependencies (PyRoKi) is only available from Git:

```bash
pip install "robosimstudio @ git+https://github.com/robosimstudio/robosimstudio"
```

On a machine without an NVIDIA GPU, install the `cpu` extra:

```bash
pip install "robosimstudio[cpu] @ git+https://github.com/robosimstudio/robosimstudio"
```

We recommend installing RoboSimStudio inside a virtual environment to avoid
conflicts with other packages:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install "robosimstudio @ git+https://github.com/robosimstudio/robosimstudio"
```

### From a Checkout with uv

The examples live in the repository. To run them, clone it and let uv build the
environment:

```bash
git clone https://github.com/robosimstudio/robosimstudio.git
cd robosimstudio
uv run examples/00_getting_started/00_hello_world.py   # builds the environment on first run
```

`uv run` and `uv sync` also install the `rtx` dependency group by default:
NVIDIA's OVRTX path tracer, 1.8 GB from NVIDIA's package index. A GPU without RT
cores never uses it; leave it out with `--no-group rtx`, or for every uv command
in a shell with `UV_NO_GROUP=rtx`.

### Without a GPU

```bash
export UV_NO_GROUP=rtx
uv run --extra cpu examples/00_getting_started/00_hello_world.py
```

> [!NOTE]
> `UV_NO_GROUP=rtx` is `--no-group rtx` for every uv command. Set it in the
> shell rather than passing the flag once: `uv run` syncs the environment each
> time, and a plain `uv run` would install the path tracer again. pip never
> installs dependency groups, so a pip install needs neither.

Nothing else changes between the two: a scene built on a machine with no CUDA
device runs on the CPU by itself.

- MuJoCo's C library steps the arm and the rigid bodies.
- SuperDex steps cloth, bags, rope and soft bodies.
- Newton's tiled tracer renders.
- A scene left on the default physics (`rss.GENERIC`) runs as `rss.CPU`, and a
  solver with no CPU path is replaced by `rss.CPU`'s for that kind of thing, with
  the change printed.
- One world per process, and no signed-distance-field contact
  (`full_surface_contact` against a mesh, `hydroelastic`).

### Which Materials Run Without a GPU

| Kind | Made with | Without a GPU |
|---|---|---|
| Cloth | `rss.sheet` | Runs, on SuperDex. Its contact forces are VBD's, so reading them (`01_cloth/03_scrub_a_stain`) is refused |
| Soft bodies | `rss.sponge` | Runs, on SuperDex |
| Rope | `rss.rod` | Runs, on SuperDex, one-way: the cable does not push back on what touches it (`rss.CPU` says why). The capsule chain's Dahl memory, twist joint and pinned capsule are VBD's, so `03_materials/01_rope_memory` and `03_materials/02_twist` build with `device="cuda"` and are refused without one |
| Bags | `rss.bag` | Runs, on SuperDex, the engine a bag is on with a GPU too |
| Cutting | `rss.cuttable`, `rss.knife` | Runs: the cage cutter on Warp's CPU device, the knife on MuJoCo's C library |
| Paper | `rss.paper` | Refused: its yielding hinges are VBD's |
| Tape | `rss.tape` | Refused: its bonds are VBD's |
| Brush | `rss.brush` | Refused: its bristles are capsule chains on VBD |
| Granular, liquid, dough | `rss.heap`, `rss.liquid`, `rss.dough` | Refused: Newton's implicit MPM |
| Water | `rss.water` | Refused: Genesis's SPH |

Each material's script in [`03_materials`](03_materials/README.md) and each
scored variant in [`06_tasks`](06_tasks/README.md) runs or is refused as its
kind's row says. The knot and the band are rope on SuperDex. The suction cup
(`04_extending/02_suction`, `06_tasks/05_suction_scored`) runs on MuJoCo's C
library, where it seals by weld and carries the plate into the square.

A scene holding a refused kind is refused as it starts to build, before anything
in it is built, with one sentence per thing naming it, its kind, the solver it is
on and why that solver has no CPU path, rather than by a failure inside a solver. VBD and the
implicit MPM run their Warp kernels on one CPU thread there: a 1922-particle towel
stepped at 0.003x realtime, and the granular example's 28,812-grain rice bed at
0.009x. Genesis is initialised on its GPU backend. A solver named by its dotted
path, such as `rss.backend("newton.solvers.SolverVBD")`, states no devices and is
not refused.

Cutting runs, slowly while it cuts: the cutter's walks run on one CPU thread, so
a control step in which the edge advances took 1 to 2 s on the development
machine's CPU, against 7.6 ms on a GPU. The cutting example's whole cut, driven
through its page, averaged 273 ms a control step there, against 13 ms, and the
tofu parted the same way.

### Verifying the Installation

```bash
python -m robosimstudio.machine
```

It prints what a scene built in this environment would run on, and why:

```console
cuda:0 (NVIDIA L40S (sm_89)) · CUDA graphs on · renderer rtx (NVIDIA L40S (sm_89))
cpu (no CUDA device) · CUDA graphs off · renderer tiled (cpu is not a CUDA device)
```

Every build prints the same line once, and `sim.platform` keeps it.

## Choosing the Device and Renderer

The device and the renderer are picked automatically. A name given in code wins
over the environment, and the environment wins over the machine:

| | In code | For a shell | Values |
|---|---|---|---|
| Device | `scene.build(device="cpu")` | `ROBOSIMSTUDIO_DEVICE` | `auto`, `cpu`, `cuda`, `cuda:1` |
| Renderer | `renderer="tiled"` | `ROBOSIMSTUDIO_RENDERER` | `auto`, `rtx`, `tiled` |

The physics follows the device and has no setting of its own. The renderer runs
on the same device as the physics.

### A CPU-Only Material in a GPU Scene

The device belongs to the scene and the engine belongs to the material. SuperDex
runs on the CPU in any scene, so a GPU scene can put one kind of thing, or one
object, on it while everything else stays on the GPU:

```python
physics = rss.GENERIC.with_backend("cloth", rss.backend("superdex", substeps=2))  # every cloth

rss.sheet(name="towel", size=(0.2, 0.2),
          sim=rss.SimRequirements(domain="cloth", backend=rss.backend("superdex", substeps=2)))  # one sheet
```

## Extra Dependencies

| Extra | Adds |
|---|---|
| `cpu` | What a machine without a GPU uses: SuperDex, for cloth, bags, rope and soft bodies |
| `superdex` | SuperDex, the CPU FEM engine behind `rss.sponge` (Python 3.12, no ARM64 wheel) |
| `rtx` | The OVRTX path tracer; uv installs it by default as the `rtx` group |
| `usd` | USD scene import |
| `genesis` | Genesis as a second foreign engine: its IPC coupler, and the SPH behind `rss.water` (NVIDIA GPU only) |
| `gap` | The graph runtime `robosimstudio_policy_agent` runs policies with; large, as it brings torch and cuRobo |
| `curobo` | cuRobo, the collision-aware planner behind `motion="plan"` |
| `lerobot-feetech` | The Feetech motor bus behind leader-arm teleoperation |
| `dev` | ruff and the test tooling |

`cpu` includes `superdex`, so install the most specific one. From a checkout,
name an extra with `--extra`:

```bash
uv sync --extra superdex
```

With pip, the path tracer comes from NVIDIA's index:

```bash
pip install "robosimstudio[rtx] @ git+https://github.com/robosimstudio/robosimstudio" \
    --extra-index-url https://pypi.nvidia.com
```

> [!NOTE]
> Install `genesis` in a virtual environment of its own: it moves about twenty
> packages, and pins trimesh below 5.

## Next Steps

- Build a first scene with [Getting started](00_getting_started/README.md).
- Browse every example in the [examples index](README.md).
- Rebuild a real scene with a coding agent on the [live page](live/README.md).
- Run the tests before changing code: see [Contributing](../CONTRIBUTING.md).
